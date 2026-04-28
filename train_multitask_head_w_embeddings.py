import os
import sys
import json
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from safetensors.torch import load_file

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from dataclasses import dataclass, field
from transformers import HfArgumentParser

import monai.transforms as mtf
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, f1_score, precision_score, recall_score, r2_score, classification_report


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
    def __init__(self, json_path, mode="train"):
        super().__init__()
        self.mode = mode

        with open(json_path, "r") as f:
            self.samples = json.load(f)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = self.samples[idx]

        pid = data['pid']
        embedding_path_ts0 = data["embedding_path_ts0"]
        embedding = load_file(embedding_path_ts0)
        
        content_info = data['numeric_dict']
        
        att_ts0 = content_info['att_ts0'] - 1
        att_ts1 = content_info['att_ts1'] - 1
        att_ts2 = content_info['att_ts2'] - 1
        margins_ts0 = content_info['margins_ts0'] - 1
        margins_ts1 = content_info['margins_ts1'] - 1
        margins_ts2 = content_info['margins_ts2'] - 1
        cancer = content_info['cancer'] - 1
        
        classification_labels = torch.tensor([
            att_ts0, att_ts1, att_ts2, margins_ts0, margins_ts1, margins_ts2, cancer
        ], dtype=torch.long)
        
        classification_mask = (classification_labels != -1)
        
        return {
            "pid": pid,
            "embedding": embedding, 
            'classification_labels': classification_labels,
            'classification_mask': classification_mask,
        }


class CTViTMultitaskHead(nn.Module):
    def __init__(
        self,
        num_classes_per_task,
        embedding_dim=1024,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Classification heads
        self.classification_heads = nn.ModuleList()
        for num_classes in num_classes_per_task:
            self.classification_heads.append(
                nn.Sequential(
                    nn.Linear(self.embedding_dim, self.embedding_dim // 2),
                    nn.ReLU(),
                    nn.Linear(self.embedding_dim // 2, num_classes)
                )
            )

    def forward(self, feats):  # Accept size_embed as parameter

        # Classification outputs
        classification_outputs = []
        for head in self.classification_heads:
            classification_outputs.append(head(feats))
        
        return classification_outputs


class MultiTaskLoss(nn.Module):
    def __init__(self, num_classification_tasks):
        super(MultiTaskLoss, self).__init__()
        self.num_classification_tasks = num_classification_tasks
        self.classification_criterion = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, classification_outputs, classification_labels, classification_mask):
        
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
        
        losses['total'] = total_loss.item()
        return total_loss, losses


def evaluate_model(model, dataloader, criterion, gpu, classification_task_names=('att_ts0', 
                                                                                 'att_ts1', 
                                                                                 'att_ts2',
                                                                                 'margins_ts0', 
                                                                                 'margins_ts1', 
                                                                                 'margins_ts2', 
                                                                                 'cancer')):
    model.eval()
    total_loss = 0.0
    all_classification_preds = [[] for _ in range(len(classification_task_names))]
    all_classification_probs = [[] for _ in range(len(classification_task_names))]
    all_classification_labels = [[] for _ in range(len(classification_task_names))]
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            embedding = batch["embedding"].cuda(gpu)
            classification_labels = batch['classification_labels'].cuda(gpu)
            classification_mask = batch['classification_mask'].cuda(gpu)
            
            classification_outputs = model(embedding)
            
            loss, _ = criterion(classification_outputs, classification_labels, classification_mask)
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
    
    return metrics


@dataclass
class TrainingArguments:
    
    train_json: str = field(
        default="/home/avepa/MedTrinity-25M/nlst_train_aux_vqa_traj_v3_seed0.json",
        metadata={"help": "Path to training JSON file."}
    )
    val_json: str = field(
        default="/home/avepa/MedTrinity-25M/nlst_val_aux_vqa_traj_v3_seed0.json",
        metadata={"help": "Path to validation JSON file."}
    )
    test_json: str = field(
        default="/home/avepa/MedTrinity-25M/nlst_test_aux_vqa_traj_v3_seed0.json",
        metadata={"help": "Path to test JSON file."}
    )
    
    batch_size: int = field(default=4, metadata={"help": "Batch size for training."})
    num_epochs: int = field(default=5, metadata={"help": "Number of training epochs."})
    learning_rate: float = field(default=1e-4, metadata={"help": "Learning rate."})
    output_dir: str = field(default="./multitask_embeddings_aux_output", metadata={"help": "Output directory."})
    device: str = field(default="cuda", metadata={"help": "Device to use."})
    gpu: int = field(default=0, metadata={"help": "GPU ID to use."})
    tag: str = field(default="", metadata={"help": "Additional tag for output directory."})



def main():
    parser = HfArgumentParser(TrainingArguments)
    (args,) = parser.parse_args_into_dataclasses()

    output_dir = args.output_dir + f"_epochs_{args.num_epochs}" + args.tag
    os.makedirs(output_dir, exist_ok=True)
    
    logger = setup_logger(
        log_file=os.path.join(output_dir, "training.log"),
        log_to_console=True
    )

    # Build cancer classifier
    model = CTViTMultitaskHead(
        num_classes_per_task=(8, 8, 8, 5, 5, 5, 2),
    ).cuda(args.gpu)

    # Build datasets using M3FM's get_data function - pass img_root
    train_dataset = AuxVisionDataset(args.train_json, mode="train")
    val_dataset = AuxVisionDataset(args.val_json, mode="val")
    test_dataset = AuxVisionDataset(args.test_json, mode="test")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=4)

    logger.info(f"Dataset sizes => train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")

    criterion = MultiTaskLoss(num_classification_tasks=5).cuda(args.gpu)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    best_val_loss = float('inf')
    best_model_path = os.path.join(output_dir, "best_model.pt")

    # Training loop
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} [Train]"):
            print(f'batch["embedding"].cuda(gpu) {batch["embedding"].cuda(gpu)}')
            embedding = batch["embedding"].cuda(args.gpu)
            classification_labels = batch['classification_labels'].cuda(args.gpu)

            optimizer.zero_grad()
            classification_outputs = model(embedding)
            
            loss, loss_dict = criterion(classification_outputs, classification_labels)
            
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        
        # Validation
        val_metrics = evaluate_model(model, val_loader, criterion, args.gpu)
        val_loss = val_metrics['loss']
        
        print(f"Epoch {epoch+1}/{args.num_epochs}")
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
            torch.save(checkpoint, os.path.join(output_dir, 'best_model.pt'))
            print(f"New best model saved with val loss: {val_loss:.4f}")
    
    print(f"Training completed. Best epoch: {best_epoch+1}, Best val loss: {best_val_loss:.4f}")
    
    # Load best model for testing
    print("Loading best model for testing...")
    checkpoint = torch.load(os.path.join(output_dir, 'best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.cuda(args.gpu)
    # Test evaluation
    print("Evaluating on test set...")
    test_metrics = evaluate_model(model, test_loader, criterion, args.gpu)
    
    print("\n=== TEST RESULTS ===")
    for metric_name, value in test_metrics.items():
        print(f"Test {metric_name}: {value:.4f}")
    
    # Save test results
    with open(os.path.join(output_dir, 'test_results.json'), 'w') as f:
        json.dump(test_metrics, f, indent=2)
    
    print(f"Test results saved to {os.path.join(output_dir, 'test_results.json')}")
if __name__ == "__main__":
    main()