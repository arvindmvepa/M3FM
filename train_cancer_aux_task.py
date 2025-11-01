import os
import sys
import json
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import collections
import re

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from dataclasses import dataclass, field
from transformers import HfArgumentParser

import monai.transforms as mtf
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score

# Import from M3FM
from models.m3fm import M3FM
from util import load_config


# Add filter selection rules
_RULES = [
    (0, re.compile(r"\bb7\d|b70f", re.I)),  # Siemens very-sharp
    (1, re.compile(r"\bb50f?", re.I)),  # Siemens sharp
    (1, re.compile(r"\bbone|lspluslung|qxd|lung", re.I)),  # GE bone / lung
    (1, re.compile(r"fc5\d", re.I)),  # Toshiba FC51/FC53
    (1, re.compile(r"\bphil.*d\b", re.I)),  # Philips D kernels
    (2, re.compile(r"\bb40f", re.I)),  # Siemens medium-sharp
    (3, re.compile(r"\bb3\d+f?", re.I)),  # Siemens B30 family
    (3, re.compile(r"\bfc10|fc0[12]", re.I)),  # Toshiba FC10/FC02/FC01
    (4, re.compile(r"\bstandard|std", re.I)),  # GE Standard
    (4, re.compile(r"\bphil.*[bc]\b", re.I)),  # Philips C / B
    (9, re.compile(r".*")),  # fallback: worst
]


def setup_logger(log_file="training.log", log_to_console=True):
    logger = logging.getLogger("training_logger")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    fh = logging.FileHandler(log_file, mode="w")
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    if log_to_console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger


def bce_loss(logits, labels, pos_weight=None):
    if pos_weight is not None:
        return F.binary_cross_entropy_with_logits(logits, labels.float(), pos_weight=pos_weight)
    else:
        return F.binary_cross_entropy_with_logits(logits, labels.float())


def compute_aux_loss(logits, targets, pos_weight=None):
    cancer_loss = bce_loss(logits.squeeze(-1), targets.squeeze(-1), pos_weight=pos_weight)
    return cancer_loss


class AuxVisionDataset(Dataset):
    def __init__(self, json_path, mode="train", transform=None, img_size=96):
        super().__init__()
        self.mode = mode
        self.transform = transform
        self.img_size = img_size

        with open(json_path, "r") as f:
            self.data_list = json.load(f)

        self.samples = []
        for datum_dict in self.data_list:
            self.samples.append({
                "img_files": datum_dict["img_files"],
                "filters": datum_dict.get("filters", []),
                "target": datum_dict["numeric_answer"]
            })

        # M3FM-style transforms matching data.py
        if self.transform is None:
            if mode == "train":
                self.transform = mtf.Compose([
                    mtf.AddChannel(),
                    mtf.Orientation(axcodes="RAS"),
                    mtf.Spacing(pixdim=(2.0, 2.0, 2.0), mode=("bilinear")),
                    mtf.ScaleIntensityRange(a_min=-1024, a_max=1024, b_min=0.0, b_max=1.0, clip=True),
                    mtf.CropForeground(),
                    mtf.RandSpatialCrop(roi_size=(self.img_size, self.img_size, self.img_size), random_size=False),
                    mtf.RandFlip(prob=0.5, spatial_axis=0),
                    mtf.RandFlip(prob=0.5, spatial_axis=1),
                    mtf.RandFlip(prob=0.5, spatial_axis=2),
                    mtf.RandRotate90(prob=0.5, spatial_axes=(0, 1)),
                    mtf.RandRotate90(prob=0.5, spatial_axes=(1, 2)),
                    mtf.RandRotate90(prob=0.5, spatial_axes=(0, 2)),
                    mtf.ToTensor(dtype=torch.float32),
                ])
            else:
                self.transform = mtf.Compose([
                    mtf.AddChannel(),
                    mtf.Orientation(axcodes="RAS"),
                    mtf.Spacing(pixdim=(2.0, 2.0, 2.0), mode=("bilinear")),
                    mtf.ScaleIntensityRange(a_min=-1024, a_max=1024, b_min=0.0, b_max=1.0, clip=True),
                    mtf.CropForeground(),
                    mtf.CenterSpatialCrop(roi_size=(self.img_size, self.img_size, self.img_size)),
                    mtf.ToTensor(dtype=torch.float32),
                ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = self.samples[idx]
        
        # Select best filter
        best_idx = self.best_filter_index(data["filters"])
        img_file = data["img_files"][best_idx]
        target = data["target"]

        # Load npy file
        img_npy = np.load(img_file)

        if self.transform is not None:
            img_tensor = self.transform(img_npy)

        return {
            "image": img_tensor,
            "target": target,
        }

    def best_filter_index(self, filters) -> int:
        """Select best filter based on kernel priority."""
        if not filters:
            return 0
        priorities = [self._priority(f.get("kernel", "")) for f in filters]
        return int(np.argmin(priorities))
    
    def _priority(self, kernel) -> int:
        """Return priority of kernel (lower = better)."""
        for priority, pattern in _RULES:
            if pattern.search(kernel):
                return priority
        return 9


class CTViTCancerClassifier(nn.Module):
    def __init__(
        self,
        ctvit_model: nn.Module,
        hidden_dim=768
    ):
        super().__init__()
        self.ctvit = ctvit_model
        self.hidden_dim = hidden_dim
        
        # Single linear layer - we'll use global average pooling
        self.cancer_head = nn.Linear(self.hidden_dim, 1)

    def forward(self, image):
        B = image.size(0)

        # Extract features from CTViT
        with torch.no_grad():
            feats = self.ctvit.forward_encoder(image, mask_ratio=0.0)
            # feats shape: [B, num_patches, hidden_dim]

        # Global average pooling across all patch tokens
        mdl_feats = feats.mean(dim=1)  # [B, hidden_dim]

        logits = self.cancer_head(mdl_feats)  # [B, 1]
        return logits


@dataclass
class TrainingArguments:
    model_path: str = field(
        default="./demo_data/model_cancer_risk.pth",
        metadata={"help": "Path to the pretrained M3FM checkpoint."}
    )
    config_path: str = field(
        default="./config_files/config_m3mf_cancer_risk.py",
        metadata={"help": "Path to the M3FM config file."}
    )
    freeze_ctvit: bool = field(
        default=True,
        metadata={"help": "Whether to freeze CTViT weights."}
    )
    
    train_json: str = field(
        default="/home/avepa/MedTrinity-25M/nlst_aux_cancer_train_v6.json",
        metadata={"help": "Path to training JSON file."}
    )
    val_json: str = field(
        default="/home/avepa/MedTrinity-25M/nlst_aux_cancer_val_v6.json",
        metadata={"help": "Path to validation JSON file."}
    )
    test_json: str = field(
        default="/home/avepa/MedTrinity-25M/nlst_aux_cancer_test_v6.json",
        metadata={"help": "Path to test JSON file."}
    )
    
    batch_size: int = field(default=4, metadata={"help": "Batch size for training."})
    num_epochs: int = field(default=5, metadata={"help": "Number of training epochs."})
    learning_rate: float = field(default=1e-4, metadata={"help": "Learning rate."})
    output_dir: str = field(default="./cancer_aux_output", metadata={"help": "Output directory."})
    device: str = field(default="cuda", metadata={"help": "Device to use."})
    tag: str = field(default="", metadata={"help": "Additional tag for output directory."})
    use_weighted_loss: bool = field(default=False, metadata={"help": "Use weighted BCE loss."})
    pos_weight: float = field(default=None, metadata={"help": "Positive class weight."})
    img_size: int = field(default=96, metadata={"help": "Image size for cropping."})


def evaluate(loader, model, device):
    model.eval()

    all_predictions = []
    all_targets = []
    all_logits = []

    for batch in loader:
        img = batch["image"].to(device)
        target = batch["target"].to(device)
        target = target.squeeze(-1).long()
        
        with torch.no_grad():
            out = model(img)

        logits = out.squeeze(-1)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).long()

        all_logits.extend(probs.detach().cpu().numpy())
        all_predictions.extend(preds.detach().cpu().numpy())
        all_targets.extend(target.detach().cpu().numpy())

    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    all_logits = np.array(all_logits)

    accuracy = accuracy_score(all_targets, all_predictions)
    f1 = f1_score(all_targets, all_predictions)
    precision = precision_score(all_targets, all_predictions, zero_division=0)
    recall = recall_score(all_targets, all_predictions, zero_division=0)

    if len(np.unique(all_targets)) > 1:
        auc = roc_auc_score(all_targets, all_logits)
    else:
        auc = 0.0

    return {
        "accuracy": accuracy,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "auc": auc
    }


def main():
    parser = HfArgumentParser(TrainingArguments)
    (args,) = parser.parse_args_into_dataclasses()

    output_dir = args.output_dir + f"_freeze_{args.freeze_ctvit}_epochs_{args.num_epochs}_weighted_{args.use_weighted_loss}" + args.tag
    os.makedirs(output_dir, exist_ok=True)
    
    logger = setup_logger(
        log_file=os.path.join(output_dir, "training.log"),
        log_to_console=True
    )
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load config and M3FM model
    logger.info(f"Loading config from {args.config_path}")
    config = load_config(args.config_path)
    
    logger.info(f"Loading M3FM model from {args.model_path}")
    model_full = M3FM(**config.model)
    checkpoint = torch.load(args.model_path, map_location='cpu')
    
    # Load checkpoint
    msg = model_full.load_state_dict(checkpoint['model'], strict=False)
    logger.info(f"Loaded checkpoint with message: {msg}")
    
    # Extract just the CT encoder from the full M3FM model
    model_ctvit = model_full.ct_encoder
    model_ctvit = model_ctvit.to(device)

    # Freeze CTViT if requested
    if args.freeze_ctvit:
        for param in model_ctvit.parameters():
            param.requires_grad = False
        logger.info("CTViT encoder is frozen.")

    # Build cancer classifier
    model = CTViTCancerClassifier(
        ctvit_model=model_ctvit
    ).to(device)

    # Build datasets with M3FM transforms
    train_dataset = AuxVisionDataset(args.train_json, mode="train", img_size=args.img_size)
    val_dataset = AuxVisionDataset(args.val_json, mode="val", img_size=args.img_size)
    test_dataset = AuxVisionDataset(args.test_json, mode="test", img_size=args.img_size)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

    logger.info(f"Dataset sizes => train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")

    # Setup optimizer
    pos_weight = None
    if args.use_weighted_loss:
        assert args.pos_weight is not None, "use_weighted_loss=True but no pos_weight specified."
        pos_weight = torch.tensor(args.pos_weight).to(device)
        logger.info(f"Using pos_weight: {pos_weight.item():.3f}")

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    best_val_loss = float('inf')
    best_model_path = os.path.join(output_dir, "best_model.pt")

    # Training loop
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} [Train]"):
            image = batch["image"].to(device)
            targets = batch['target'].to(device)

            optimizer.zero_grad()
            logits = model(image)

            loss = compute_aux_loss(logits, targets, pos_weight=pos_weight)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        logger.info(f"[Epoch {epoch + 1}] Train loss = {avg_train_loss:.5f}")

        # Validation
        model.eval()
        val_total_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1} [Val]"):
                image = batch["image"].to(device)
                targets = batch["target"].to(device)

                logits = model(image)
                v_loss = compute_aux_loss(logits, targets, pos_weight=pos_weight)

                val_total_loss += v_loss.item()

        avg_val_loss = val_total_loss / len(val_loader)
        val_metrics = evaluate(val_loader, model, device)

        logger.info(f"[Epoch {epoch + 1}] Val loss = {avg_val_loss:.5f} " +
                    " ".join([f"{k}={v:.4f}" for k, v in val_metrics.items()]))

        # Save best checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"  ➜ New best model saved ({best_val_loss:.5f})")

    # Test evaluation
    logger.info("========== TEST ==========")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    test_total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="[Test]"):
            image = batch["image"].to(device)
            targets = batch["target"].to(device)

            logits = model(image)
            t_loss = compute_aux_loss(logits, targets, pos_weight=pos_weight)

            test_total_loss += t_loss.item()

    avg_test_loss = test_total_loss / len(test_loader)
    test_metrics = evaluate(test_loader, model, device)

    logger.info(f"Best-val model Test loss = {avg_test_loss:.5f} " +
                " ".join([f"{k}={v:.4f}" for k, v in test_metrics.items()]))


if __name__ == "__main__":
    main()