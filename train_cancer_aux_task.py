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

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from dataclasses import dataclass, field
from transformers import HfArgumentParser

import monai.transforms as mtf
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score

# Import from M3FM
from model.M3FM import M3FM
from model.pos_embed import interpolate_pos_embed


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
    def __init__(self, json_path, mode="train", transform=None):
        super().__init__()
        self.mode = mode
        self.transform = transform

        with open(json_path, "r") as f:
            self.data_list = json.load(f)

        self.samples = []
        for datum_dict in self.data_list:
            self.samples.append({
                "img_files": datum_dict["img_files"],
                "target": datum_dict["numeric_answer"]
            })

        if self.transform is None:
            if mode == "train":
                self.transform = mtf.Compose([
                    mtf.RandRotate90(prob=0.5, spatial_axes=(1, 2)),
                    mtf.RandFlip(prob=0.10, spatial_axis=0),
                    mtf.RandFlip(prob=0.10, spatial_axis=1),
                    mtf.RandFlip(prob=0.10, spatial_axis=2),
                    mtf.RandScaleIntensity(factors=0.1, prob=0.5),
                    mtf.RandShiftIntensity(offsets=0.1, prob=0.5),
                    mtf.ToTensor(dtype=torch.float),
                ])
            else:
                self.transform = mtf.Compose([
                    mtf.ToTensor(dtype=torch.float),
                ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = self.samples[idx]
        img_file = data["img_files"][0]  # Take first image file
        target = data["target"]

        # Load npy file
        img_npy = np.load(img_file)

        if self.transform is not None:
            img_tensor = self.transform(img_npy)

        return {
            "image": img_tensor,
            "target": target,
        }


class CTViTCancerClassifier(nn.Module):
    def __init__(
        self,
        ctvit_model: nn.Module,
        use_cls=True,
        hidden_dim=768
    ):
        super().__init__()
        self.ctvit = ctvit_model
        self.use_cls = use_cls
        self.hidden_dim = hidden_dim
        
        if self.use_cls:
            self.cancer_head = nn.Linear(self.hidden_dim, 1)
        else:
            # Assuming patch tokens, need to adjust based on actual output
            self.cancer_head = nn.Linear(self.hidden_dim * 2048, 1)

    def forward(self, image):
        B = image.size(0)

        # Extract features from CTViT
        with torch.no_grad():
            feats = self.ctvit.forward_encoder(image, mask_ratio=0.0)

        if self.use_cls:
            # Use CLS token (first token)
            cls_feats = feats[:, 0]
            cls_feats = cls_feats.view(B, self.hidden_dim)
            mdl_feats = cls_feats
        else:
            # Use all patch tokens
            non_cls_feats = feats.view(B, -1)
            mdl_feats = non_cls_feats

        logits = self.cancer_head(mdl_feats).view(B, 1)
        return logits


@dataclass
class TrainingArguments:
    model_path: str = field(
        default="./ckpt/M3FM.pth",
        metadata={"help": "Path to the pretrained M3FM checkpoint."}
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
    use_cls: bool = field(default=True, metadata={"help": "Use CLS token for classification."})
    use_weighted_loss: bool = field(default=False, metadata={"help": "Use weighted BCE loss."})
    pos_weight: float = field(default=None, metadata={"help": "Positive class weight."})


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

    output_dir = args.output_dir + f"_freeze_{args.freeze_ctvit}_epochs_{args.num_epochs}_use_cls_{args.use_cls}_weighted_{args.use_weighted_loss}" + args.tag
    os.makedirs(output_dir, exist_ok=True)
    
    logger = setup_logger(
        log_file=os.path.join(output_dir, "training.log"),
        log_to_console=True
    )
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load M3FM model
    logger.info(f"Loading M3FM model from {args.model_path}")
    model_ctvit = M3FM()
    checkpoint = torch.load(args.model_path, map_location='cpu')
    
    # Interpolate position embeddings if needed
    interpolate_pos_embed(model_ctvit, checkpoint['model'])
    
    # Load checkpoint
    msg = model_ctvit.load_state_dict(checkpoint['model'], strict=False)
    logger.info(f"Loaded checkpoint with message: {msg}")
    
    model_ctvit = model_ctvit.to(device)

    # Freeze CTViT if requested
    if args.freeze_ctvit:
        for param in model_ctvit.parameters():
            param.requires_grad = False
        logger.info("CTViT is frozen.")

    # Build cancer classifier
    model = CTViTCancerClassifier(
        ctvit_model=model_ctvit,
        use_cls=args.use_cls
    ).to(device)

    # Build datasets
    train_dataset = AuxVisionDataset(args.train_json, mode="train")
    val_dataset = AuxVisionDataset(args.val_json, mode="val")
    test_dataset = AuxVisionDataset(args.test_json, mode="test")

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