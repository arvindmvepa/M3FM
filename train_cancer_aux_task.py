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
import models.m3fm as m3fm
from util import ConfigFile, txt2embed, get_sincos_size_embed
# Import the get_data function from M3FM
from data import get_data


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


def center_crop_resize(x, crop_size):
    """Center crop the volume to crop_size, resizing if necessary."""
    s, h, w = x.shape
    ts, th, tw = crop_size
    
    # Calculate center coordinates
    center_s, center_h, center_w = s // 2, h // 2, w // 2
    
    # Calculate crop boundaries (half crop_size around center)
    half_ts, half_th, half_tw = ts // 2, th // 2, tw // 2
    
    # Ensure we don't go out of bounds
    ss = max(0, center_s - half_ts)
    se = min(s, center_s + half_ts)
    hs = max(0, center_h - half_th) 
    he = min(h, center_h + half_th)
    ws = max(0, center_w - half_tw)
    we = min(w, center_w + half_tw)
    
    # If the volume is smaller than crop_size, pad with zeros
    if (se - ss) < ts or (he - hs) < th or (we - ws) < tw:
        # Create zero-padded volume of crop_size
        cropped = np.zeros((ts, th, tw), dtype=x.dtype)
        
        # Calculate where to place the actual data in the padded volume
        actual_s, actual_h, actual_w = se - ss, he - hs, we - ws
        pad_s_start = (ts - actual_s) // 2
        pad_h_start = (th - actual_h) // 2  
        pad_w_start = (tw - actual_w) // 2
        
        cropped[pad_s_start:pad_s_start + actual_s,
                pad_h_start:pad_h_start + actual_h,
                pad_w_start:pad_w_start + actual_w] = x[ss:se, hs:he, ws:we]
        
        x = cropped
    else:
        # Direct crop
        x = x[ss:se, hs:he, ws:we]
    
    # Resize to exact crop_size if needed
    if x.shape != (ts, th, tw):
        x = torch.nn.functional.interpolate(
            torch.from_numpy(x).unsqueeze(0).unsqueeze(0).to(torch.float32),
            size=(ts, th, tw),
            mode="trilinear",
            align_corners=False,
        ).numpy().squeeze()
    
    return x

def get_data_center_crop(input_dict, args):
    """Modified get_data function that uses center cropping instead of coordinate-based cropping."""
    
    # Extract required parameters (same as original)
    ct_path = input_dict['ct_path']
    data_name = args.data_name
    crop_size = args.crop_size
    patch_size = args.cube_size
    hu_range = args.hu_range
    embed_dim = args.embed_dim
    question = input_dict['question']
    clinical_txt = input_dict['clinical_txt']

    # Load and preprocess data
    data_ori = np.load(ct_path)
    
    # Use center crop instead of coordinate-based crop
    data = center_crop_resize(data_ori, crop_size)
    
    # Use the same normalization and tensor conversion as original
    from data import normalize, to_tensor
    data = normalize([data], hu_range[0], hu_range[1])[0]
    data = to_tensor([data])[0]

    # Calculate size embeddings - use uniform scaling for center crop
    # Since we're doing center crop, use the patch size directly for embedding
    sizes = np.array([[patch_size[0], patch_size[1], patch_size[2]]])
    size_embed = get_sincos_size_embed(embed_dim, sizes)
    size_embed = torch.from_numpy(size_embed).to(torch.float32)

    # Text processing (same as original)
    question_ids, question_masks = txt2embed(question)
    question_ids = torch.LongTensor(question_ids).unsqueeze(0)
    question_masks = torch.LongTensor(question_masks).unsqueeze(0)

    txt_ids, txt_masks = txt2embed(clinical_txt, max_length=160)
    txt_ids = torch.LongTensor(txt_ids).unsqueeze(0)
    txt_masks = torch.LongTensor(txt_masks).unsqueeze(0)
    
    # Return same data_dict format as original
    data_dict = {
        'data': data.unsqueeze(0), 
        'questions': question,
        'questions_ids': question_ids.unsqueeze(0), 
        'questions_mask': question_masks.unsqueeze(0),
        'data_size': torch.LongTensor(crop_size).unsqueeze(0),
        'txt_ids': txt_ids.unsqueeze(0), 
        'txt_mask': txt_masks.unsqueeze(0),
        'size_embed': size_embed.unsqueeze(0), 
        "clinical_txt": clinical_txt,
        'patch_size': patch_size,
        'data_name': [data_name]
    }

    return data_dict


def get_npy_path(volume_path, img_root="/hsuraid/avepa/nlst_npy_v1"):
    volume_name = os.path.basename(volume_path)
    time_point_dir = os.path.basename(os.path.dirname(volume_path))
    pid_dir = os.path.basename(os.path.dirname(os.path.dirname(volume_path)))
    volume_path_npy = os.path.join(img_root, pid_dir, time_point_dir, volume_name + ".npy")
    return volume_path_npy


class AuxVisionDataset(Dataset):
    def __init__(self, json_path, mode="train", config_args=None, img_root="/hsuraid/avepa/nlst_npy_v1"):
        super().__init__()
        self.mode = mode
        self.config_args = config_args
        self.img_root = img_root  # Store img_root as instance variable

        with open(json_path, "r") as f:
            self.data_list = json.load(f)

        self.samples = []
        for datum_dict in self.data_list:
            self.samples.append({
                "img_files": datum_dict["img_files"],
                "filters": datum_dict.get("filters", []),
                "target": datum_dict["numeric_answer"]
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = self.samples[idx]
        
        # Select best filter
        best_idx = self.best_filter_index(data["filters"])
        img_file = data["img_files"][best_idx]
        target = data["target"]

        # Create input dict for center crop preprocessing (no pixel_size or coords needed)
        input_dict = {
            'ct_path': get_npy_path(img_file, self.img_root),  # Use instance variable
            'question': 'Predict cancer risk',
            'clinical_txt': '',  # Empty since we're not using clinical text for this task
        }
        
        # Use our custom center crop function instead of original get_data
        processed_data = get_data_center_crop(input_dict, self.config_args)
        
        return {
            "image": processed_data['data'][0],  # Extract the preprocessed image
            "target": target,
        }

    def best_filter_index(self, filters) -> int:
        """Select best filter based on kernel priority."""
        if not filters:
            return 0
        
        # Handle both dict and string cases
        priorities = []
        for f in filters:
            if isinstance(f, dict):
                kernel = f.get("kernel", "")
            elif isinstance(f, str):
                kernel = f
            else:
                kernel = ""
            priorities.append(self._priority(kernel))
        
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
        m3fm_model: nn.Module,
        hidden_dim=768
    ):
        super().__init__()
        self.m3fm_model = m3fm_model
        self.hidden_dim = hidden_dim
        
        # Single linear layer - we'll use global average pooling
        self.cancer_head = nn.Linear(self.hidden_dim, 1)

    def forward(self, image):
        B = image.size(0)

        # Copy the exact approach from M3FM.pred_embeds (lines 240-261)
        with torch.no_grad():
            # Line 240: self.ims = (imgs.shape[2], imgs.shape[3], imgs.shape[4])
            self.m3fm_model.ims = (image.shape[2], image.shape[3], image.shape[4])
            
            # Line 241: img_embeds = self.m3fm_model.img_tokenizer(imgs)
            img_embeds = self.m3fm_model.img_tokenizer(image)
            
            # Lines 252-254: Get input_size exactly as in the original code
            # if 'data' not in data_dict.keys():
            #     input_size = self.img_tokenizer.__getattr__('tokenizer_{}'.format(self.ims)).input_size
            # else:
            #     input_size = self.img_tokenizer.__getattr__('tokenizer_{}'.format(self.ims)).input_size
            input_size = self.m3fm_model.img_tokenizer.__getattr__('tokenizer_{}'.format(self.m3fm_model.ims)).input_size
            
            # Lines 255-261: Pass through encoder_img
            feats = self.m3fm_model.encoder_img(
                img_embeds,
                window_size=self.m3fm_model.window_size,
                window_block_indexes=self.m3fm_model.window_block_indexes,
                spatial_size=input_size,
                cls_embed=self.m3fm_model.cls_embed_img,
                attention_mask=None,
                drop_path=0.0,
                drop=0.0
            )

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
    
    # Add img_root as an argument
    img_root: str = field(
        default="/hsuraid/avepa/nlst_npy_v1",
        metadata={"help": "Root directory for .npy files."}
    )
    
    train_json: str = field(
        default="/home/avepa/MedTrinity-25M/nlst_aux_cancer_train_v7.json",
        metadata={"help": "Path to training JSON file."}
    )
    val_json: str = field(
        default="/home/avepa/MedTrinity-25M/nlst_aux_cancer_val_v7.json",
        metadata={"help": "Path to validation JSON file."}
    )
    test_json: str = field(
        default="/home/avepa/MedTrinity-25M/nlst_aux_cancer_test_v7.json",
        metadata={"help": "Path to test JSON file."}
    )
    
    batch_size: int = field(default=4, metadata={"help": "Batch size for training."})
    num_epochs: int = field(default=5, metadata={"help": "Number of training epochs."})
    learning_rate: float = field(default=1e-4, metadata={"help": "Learning rate."})
    output_dir: str = field(default="./cancer_aux_output", metadata={"help": "Output directory."})
    device: str = field(default="cuda", metadata={"help": "Device to use."})
    gpu: int = field(default=0, metadata={"help": "GPU ID to use."})
    tag: str = field(default="", metadata={"help": "Additional tag for output directory."})
    use_weighted_loss: bool = field(default=False, metadata={"help": "Use weighted BCE loss."})
    pos_weight: float = field(default=None, metadata={"help": "Positive class weight."})
    # Remove img_size and let crop_size be loaded from config


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

    # Load config using ConfigFile (matching inference.py)
    logger.info(f"Loading config from {args.config_path}")
    config_args = ConfigFile(args.config_path)
    
    # Get crop_size from config
    crop_size = config_args.crop_size
    logger.info(f"Using crop_size from config: {crop_size}")
    
    # Create model using the config (matching inference.py)
    torch.backends.cudnn.benchmark = True
    config_args.modalities = list(config_args.modalities.split(","))
    logger.info(f"Loading M3FM model from {args.model_path}")
    
    model_full = m3fm.__dict__[config_args.model](**vars(config_args))
    model_full.cuda(args.gpu)
    
    # Load checkpoint (matching inference.py)
    state_dict = torch.load(args.model_path, map_location='cpu')
    msg = model_full.load_state_dict(state_dict, strict=False)
    logger.info(f"Loaded checkpoint with message: {msg}")
    
    # Get embed_dim_img from the loaded model (line 76 in m3fm.py)
    embed_dim_img = model_full.embed_dim_img
    logger.info(f"Using embed_dim_img={embed_dim_img} from M3FM model")

    # Freeze M3FM components if requested (img_tokenizer and encoder_img)
    if args.freeze_ctvit:
        for param in model_full.img_tokenizer.parameters():
            param.requires_grad = False
        for param in model_full.encoder_img.parameters():
            param.requires_grad = False
        logger.info("M3FM image tokenizer and encoder are frozen.")

    # Build cancer classifier - pass the full model so we can access tokenizer and encoder
    model = CTViTCancerClassifier(
        m3fm_model=model_full,
        hidden_dim=embed_dim_img  # Use the model's embed_dim_img
    ).cuda(args.gpu)

    # Build datasets using M3FM's get_data function - pass img_root
    train_dataset = AuxVisionDataset(args.train_json, mode="train", config_args=config_args, img_root=args.img_root)
    val_dataset = AuxVisionDataset(args.val_json, mode="val", config_args=config_args, img_root=args.img_root)
    test_dataset = AuxVisionDataset(args.test_json, mode="test", config_args=config_args, img_root=args.img_root)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=4)

    logger.info(f"Dataset sizes => train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
    logger.info(f"Using crop_size: {crop_size}")

    # Setup optimizer
    pos_weight = None
    if args.use_weighted_loss:
        assert args.pos_weight is not None, "use_weighted_loss=True but no pos_weight specified."
        pos_weight = torch.tensor(args.pos_weight).cuda(args.gpu)
        logger.info(f"Using pos_weight: {pos_weight.item():.3f}")

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    best_val_loss = float('inf')
    best_model_path = os.path.join(output_dir, "best_model.pt")

    # Training loop
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} [Train]"):
            image = batch["image"].cuda(args.gpu)
            targets = batch['target'].cuda(args.gpu)

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
                image = batch["image"].cuda(args.gpu)
                targets = batch["target"].cuda(args.gpu)

                logits = model(image)
                v_loss = compute_aux_loss(logits, targets, pos_weight=pos_weight)

                val_total_loss += v_loss.item()

        avg_val_loss = val_total_loss / len(val_loader)
        val_metrics = evaluate(val_loader, model, torch.device(f'cuda:{args.gpu}'))

        logger.info(f"[Epoch {epoch + 1}] Val loss = {avg_val_loss:.5f} " +
                    " ".join([f"{k}={v:.4f}" for k, v in val_metrics.items()]))

        # Save best checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"  ➜ New best model saved ({best_val_loss:.5f})")

    # Test evaluation
    logger.info("========== TEST ==========")
    model.load_state_dict(torch.load(best_model_path, map_location=f'cuda:{args.gpu}'))
    model.eval()

    test_total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="[Test]"):
            image = batch["image"].cuda(args.gpu)
            targets = batch["target"].cuda(args.gpu)

            logits = model(image)
            t_loss = compute_aux_loss(logits, targets, pos_weight=pos_weight)

            test_total_loss += t_loss.item()

    avg_test_loss = test_total_loss / len(test_loader)
    test_metrics = evaluate(test_loader, model, torch.device(f'cuda:{args.gpu}'))

    logger.info(f"Best-val model Test loss = {avg_test_loss:.5f} " +
                " ".join([f"{k}={v:.4f}" for k, v in test_metrics.items()]))


if __name__ == "__main__":
    main()