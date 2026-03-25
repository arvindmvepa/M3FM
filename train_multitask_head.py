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
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, f1_score, precision_score, recall_score, r2_score, classification_report


# Import from M3FM
import models.m3fm as m3fm
from util import ConfigFile, txt2embed, get_sincos_size_embed
# Import the get_data function from M3FM
from data import get_data


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

    # Calculate size embeddings with realistic physical dimensions
    # Use default pixel spacing that matches M3FM training data
    pix_size = [2.0, 0.53, 0.53]  # Default spacing: slice=2mm, in-plane=0.53mm
    
    # Calculate actual physical patch size (what the model expects)
    sizes = np.array([[
        pix_size[0] * patch_size[0],  # Physical depth: 2.0 * 16 = 32.0 mm
        pix_size[1] * patch_size[1],  # Physical height: 0.53 * 16 = 8.48 mm
        pix_size[2] * patch_size[2]   # Physical width: 0.53 * 16 = 8.48 mm
    ]])
    
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
            self.samples = json.load(f)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = self.samples[idx]
        
        # Extract labels (only first entry of each list)
        content_info = data['content_info']
        
        # Multi-class tasks (first 5)
        location = content_info['location'][0]
        interval_change = content_info['interval_change'][0]
        interval_growth = content_info['interval_growth'][0]
        margins = content_info['margins'][0]
        predominant_attenuation = content_info['predominant_attenuation'][0]
        
        # Regression tasks (last 2)
        longest_diameter = content_info['longest_diameter'][0]
        longest_perpendicular_diameter = content_info['longest_perpendicular_diameter'][0]
        
        # Create labels tensor
        classification_labels = torch.tensor([
            location, margins, predominant_attenuation, interval_change, interval_growth
        ], dtype=torch.long)
        
        regression_labels = torch.tensor([
            longest_diameter, longest_perpendicular_diameter
        ], dtype=torch.float32)
        
        # Create masks for valid labels (not -1)
        classification_mask = (classification_labels != -1)
        regression_mask = (regression_labels != -1.0)

        # Create input dict for center crop preprocessing
        input_dict = {
            'ct_path': data["embedding_path"],
            'question': 'Predict multitask answers',
            'clinical_txt': '',
        }
        
        # Use our custom center crop function
        processed_data = get_data_center_crop(input_dict, self.config_args)
        
        return {
            "image": processed_data['data'][0],  # Extract the preprocessed image
            "size_embed": processed_data['size_embed'][0],  # P0ass the precomputed size_embed
            'classification_labels': classification_labels,
            'regression_labels': regression_labels,
            'classification_mask': classification_mask,
            'regression_mask': regression_mask,
        }


class CTViTMultitaskHead(nn.Module):
    def __init__(
        self,
        m3fm_model: nn.Module,
        num_classes_per_task, 
        num_regression_tasks,
        hidden_dim=1024,
        freeze_encoder=True
    ):
        super().__init__()
        self.m3fm_model = m3fm_model
        self.hidden_dim = hidden_dim
        self.freeze_encoder = freeze_encoder

        # Classification heads
        self.classification_heads = nn.ModuleList()
        for num_classes in num_classes_per_task:
            self.classification_heads.append(
                nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(self.hidden_dim // 2, num_classes)
                )
            )
        # Regression head
        self.regression_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, num_regression_tasks)
        )

    def forward(self, image, size_embed):  # Accept size_embed as parameter
        B = image.size(0)

        # Feature extraction with proper gradient handling
        if self.freeze_encoder:
            with torch.no_grad():
                # Set image size for tokenizer selection
                self.m3fm_model.ims = (image.shape[2], image.shape[3], image.shape[4])
                
                # Tokenize image
                img_embeds = self.m3fm_model.img_tokenizer(image)
                
                # Use the precomputed size_embed (no need to recalculate)
                img_embeds = img_embeds + size_embed.to(img_embeds.device)
                
                # Get input size for spatial processing
                input_size = self.m3fm_model.img_tokenizer.__getattr__('tokenizer_{}'.format(self.m3fm_model.ims)).input_size
                
                # Pass through encoder_img
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
        else:
            # Allow gradients for fine-tuning
            self.m3fm_model.ims = (image.shape[2], image.shape[3], image.shape[4])
            img_embeds = self.m3fm_model.img_tokenizer(image)
            
            # Use the precomputed size_embed (no need to recalculate)
            img_embeds = img_embeds + size_embed.to(img_embeds.device)
            
            input_size = self.m3fm_model.img_tokenizer.__getattr__('tokenizer_{}'.format(self.m3fm_model.ims)).input_size
            
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
        mdl_feats = feats.mean(dim=1)  # [B, embed_dim_img]

        # Classification outputs
        classification_outputs = []
        for head in self.classification_heads:
            classification_outputs.append(head(mdl_feats))
        
        # Regression output
        regression_output = self.regression_head(mdl_feats)
        
        return classification_outputs, regression_output


class MultiTaskLoss(nn.Module):
    def __init__(self, num_classification_tasks, num_regression_tasks):
        super(MultiTaskLoss, self).__init__()
        self.num_classification_tasks = num_classification_tasks
        self.num_regression_tasks = num_regression_tasks
        
        self.classification_criterion = nn.CrossEntropyLoss(reduction='none')
        self.regression_criterion = nn.MSELoss(reduction='none')
    
    def forward(self, classification_outputs, regression_output, 
                classification_labels, regression_labels,
                classification_mask, regression_mask):
        
        total_loss = 0.0
        losses = {}
        
        # Classification losses
        for i, output in enumerate(classification_outputs):
            task_mask = classification_mask[:, i]
            if task_mask.sum() > 0:  # Only compute loss if there are valid labels
                valid_labels = classification_labels[:, i][task_mask]
                valid_outputs = output[task_mask]
                task_loss = self.classification_criterion(valid_outputs, valid_labels).mean()
                total_loss += task_loss
                losses[f'classification_task_{i}'] = task_loss.item()
        
        # Regression losses
        for i in range(self.num_regression_tasks):
            task_mask = regression_mask[:, i]
            if task_mask.sum() > 0:  # Only compute loss if there are valid labels
                valid_labels = regression_labels[:, i][task_mask]
                valid_outputs = regression_output[:, i][task_mask]
                task_loss = self.regression_criterion(valid_outputs, valid_labels).mean()
                total_loss += task_loss
                losses[f'regression_task_{i}'] = task_loss.item()
        
        losses['total'] = total_loss.item()
        return total_loss, losses


def evaluate_model(model, dataloader, criterion, gpu, 
classification_task_names=('location', 'margins', 'predominant_attenuation','interval_change', 'interval_growth'), 
regression_task_names = ('longest_diameter', 'longest_perpendicular_diameter')):
    model.eval()
    total_loss = 0.0
    all_classification_preds = [[] for _ in range(len(classification_task_names))]
    all_classification_probs = [[] for _ in range(len(classification_task_names))]
    all_classification_labels = [[] for _ in range(len(classification_task_names))]
    all_regression_preds = [[] for _ in range(len(regression_task_names))]
    all_regression_labels = [[] for _ in range(len(regression_task_names))]
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            embeddings = batch['embedding'].cuda(gpu)
            classification_labels = batch['classification_labels'].cuda(gpu)
            regression_labels = batch['regression_labels'].cuda(gpu)
            classification_mask = batch['classification_mask'].cuda(gpu)
            regression_mask = batch['regression_mask'].cuda(gpu)
            
            classification_outputs, regression_output = model(embeddings)
            
            loss, _ = criterion(classification_outputs, regression_output,
                              classification_labels, regression_labels,
                              classification_mask, regression_mask)
            total_loss += loss.item()
            
            # Collect predictions and probabilities for metrics
            for i, output in enumerate(classification_outputs):
                task_mask = classification_mask[:, i]
                if task_mask.sum() > 0:
                    valid_labels = classification_labels[:, i][task_mask]
                    valid_outputs = output[task_mask]
                    valid_probs = torch.softmax(valid_outputs, dim=1)
                    valid_preds = torch.argmax(valid_outputs, dim=1)
                    
                    all_classification_preds[i].extend(valid_preds.cpu().numpy())
                    all_classification_probs[i].extend(valid_probs.cpu().numpy())
                    all_classification_labels[i].extend(valid_labels.cpu().numpy())
            
            for i in range(2):
                task_mask = regression_mask[:, i]
                if task_mask.sum() > 0:
                    valid_labels = regression_labels[:, i][task_mask]
                    valid_preds = regression_output[:, i][task_mask]
                    all_regression_preds[i].extend(valid_preds.cpu().numpy())
                    all_regression_labels[i].extend(valid_labels.cpu().numpy())
    
    # Calculate metrics
    metrics = {}
    avg_loss = total_loss / len(dataloader)
    metrics['loss'] = avg_loss
    
    # Classification metrics
    for i, task_name in enumerate(classification_task_names):
        if len(all_classification_preds[i]) > 0:
            accuracy = accuracy_score(all_classification_labels[i], all_classification_preds[i])
            metrics[f'{task_name}_accuracy'] = accuracy
            
            # Calculate AUROC using one-vs-rest
            try:
                labels = np.array(all_classification_labels[i])
                probs = np.array(all_classification_probs[i])
                
                # Check if we have more than one class present
                unique_labels = np.unique(labels)
                if len(unique_labels) > 1:
                    auroc = roc_auc_score(labels, probs, multi_class='ovr', average='macro')
                    metrics[f'{task_name}_auroc'] = auroc
                else:
                    metrics[f'{task_name}_auroc'] = np.nan  # Cannot compute AUROC with only one class
                    
            except Exception as e:
                print(f"Warning: Could not compute AUROC for {task_name}: {e}")
                metrics[f'{task_name}_auroc'] = np.nan
    
    # Regression metrics
    for i, task_name in enumerate(regression_task_names):
        if len(all_regression_preds[i]) > 0:
            mse = mean_squared_error(all_regression_labels[i], all_regression_preds[i])
            r2 = r2_score(all_regression_labels[i], all_regression_preds[i])
            metrics[f'{task_name}_mse'] = mse
            metrics[f'{task_name}_r2'] = r2
    
    return metrics


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
        default="/hsuraid/avepa/nlst_npy_m3fm",
        metadata={"help": "Root directory for .npy files."}
    )
    
    train_json: str = field(
        default="/home/avepa/MedTrinity-25M/nlst_train_aux_vqa_delta2True_v1_m3fm.json",
        metadata={"help": "Path to training JSON file."}
    )
    val_json: str = field(
        default="/home/avepa/MedTrinity-25M/nlst_val_aux_vqa_delta2True_v1_m3fm.json",
        metadata={"help": "Path to validation JSON file."}
    )
    test_json: str = field(
        default="/home/avepa/MedTrinity-25M/nlst_test_aux_vqa_delta2True_v1_m3fm.json",
        metadata={"help": "Path to test JSON file."}
    )
    
    batch_size: int = field(default=4, metadata={"help": "Batch size for training."})
    num_epochs: int = field(default=5, metadata={"help": "Number of training epochs."})
    learning_rate: float = field(default=1e-4, metadata={"help": "Learning rate."})
    output_dir: str = field(default="./multitask_aux_output", metadata={"help": "Output directory."})
    device: str = field(default="cuda", metadata={"help": "Device to use."})
    gpu: int = field(default=0, metadata={"help": "GPU ID to use."})
    tag: str = field(default="", metadata={"help": "Additional tag for output directory."})



def main():
    parser = HfArgumentParser(TrainingArguments)
    (args,) = parser.parse_args_into_dataclasses()

    output_dir = args.output_dir + f"_freeze_{args.freeze_ctvit}_epochs_{args.num_epochs}" + args.tag
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

    # Build cancer classifier
    model = CTViTMultitaskHead(
        m3fm_model=model_full,
        num_classes_per_task=(8, 4, 7, 3, 3),
        num_regression_tasks=2,
        hidden_dim=embed_dim_img,
        freeze_encoder=args.freeze_ctvit
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

    criterion = MultiTaskLoss(num_classification_tasks=5, num_regression_tasks=2).cuda(args.gpu)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    best_val_loss = float('inf')
    best_model_path = os.path.join(output_dir, "best_model.pt")

    # Training loop
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} [Train]"):
            img = batch["image"].cuda(args.gpu)
            size_embed = batch["size_embed"].cuda(args.gpu)  # Get precomputed size_embed
            classification_labels = batch['classification_labels'].cuda(args.gpu)
            regression_labels = batch['regression_labels'].cuda(args.gpu)
            classification_mask = batch['classification_mask'].cuda(args.gpu)
            regression_mask = batch['regression_mask'].cuda(args.gpu)

            optimizer.zero_grad()
            classification_outputs, regression_output = model(img, size_embed)
            
            loss, loss_dict = criterion(classification_outputs, regression_output,
                                      classification_labels, regression_labels,
                                      classification_mask, regression_mask)
            
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation
        val_metrics = evaluate_model(model, val_loader, criterion, args.gpu)
        val_loss = val_metrics['loss']
        
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        
        # Print validation metrics
        for metric_name, value in val_metrics.items():
            if metric_name != 'loss':
                print(f"Val {metric_name}: {value:.4f}")
        print("-" * 50)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_metrics': val_metrics
            }
            torch.save(checkpoint, os.path.join(args.save_dir, 'best_model.pt'))
            print(f"New best model saved with val loss: {val_loss:.4f}")
    
    print(f"Training completed. Best epoch: {best_epoch+1}, Best val loss: {best_val_loss:.4f}")
    
    # Load best model for testing
    print("Loading best model for testing...")
    checkpoint = torch.load(os.path.join(args.save_dir, 'best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test evaluation
    print("Evaluating on test set...")
    test_metrics = evaluate_model(model, test_loader, criterion, args.gpu)
    
    print("\n=== TEST RESULTS ===")
    for metric_name, value in test_metrics.items():
        print(f"Test {metric_name}: {value:.4f}")
    
    # Save test results
    with open(os.path.join(args.save_dir, 'test_results.json'), 'w') as f:
        json.dump(test_metrics, f, indent=2)
    
    print(f"Test results saved to {os.path.join(args.save_dir, 'test_results.json')}")


if __name__ == "__main__":
    main()