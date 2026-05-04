#!/usr/bin/env python3
"""
Train latent trajectory models that predict future image embeddings from time step 0.

This script is designed for JSON files like your multitask script:

  sample["embedding_path_ts0"]
  sample["embedding_path_ts1"]
  sample["embedding_path_ts2"]
  sample["numeric_dict"]["att_ts0"], etc.

It supports:
  - Models: residual_mlp, autoreg_mlp, gru, lowrank_linear, transformer
  - Losses: MSE or Huber, cosine, delta, contrastive InfoNCE
  - Optional auxiliary multitask loss:
      * compact: task-type heads shared across future steps: att, margins, cancer
      * attached: compatible with your existing 7-head multitask checkpoint

Example 1: strong first baseline
python train_embedding_trajectory.py \
  --train_json /path/train.json \
  --val_json /path/val.json \
  --test_json /path/test.json \
  --num_steps 2 \
  --model_type residual_mlp \
  --base_loss mse \
  --cosine_weight 0.5 \
  --delta_weight 0.5 \
  --standardize_loss

Example 2: GRU with frozen attached-style auxiliary classifier
python train_embedding_trajectory.py \
  --train_json /path/train.json \
  --val_json /path/val.json \
  --test_json /path/test.json \
  --num_steps 2 \
  --model_type gru \
  --aux_mode frozen \
  --aux_style attached \
  --aux_checkpoint /path/to/best_model.pt \
  --aux_loss_weight 0.1

Example 3: joint compact auxiliary head for arbitrary horizons
python train_embedding_trajectory.py \
  --train_json /path/train.json \
  --val_json /path/val.json \
  --test_json /path/test.json \
  --num_steps 5 \
  --model_type autoreg_mlp \
  --aux_mode joint \
  --aux_style compact \
  --aux_loss_weight 0.05
"""

import argparse
import json
import logging
import math
import os
import random
from dataclasses import asdict, dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def setup_logger(output_dir: str, log_to_console: bool = True) -> logging.Logger:
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger("trajectory_training")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    fh = logging.FileHandler(os.path.join(output_dir, "training.log"), mode="w")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    if log_to_console:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device_arg: str, gpu: int) -> torch.device:
    if device_arg == "cuda" and torch.cuda.is_available():
        return torch.device(f"cuda:{gpu}")
    return torch.device("cpu")


def load_embedding(path: str) -> torch.Tensor:
    """Load an embedding vector from a safetensors file."""
    tensors = load_file(path)
    if "embeddings" not in tensors:
        raise KeyError(f"{path} does not contain key 'embeddings'. Found keys: {list(tensors.keys())}")
    emb = tensors["embeddings"].float()
    # Expected shape is [D]. If it is [1, D], squeeze the singleton. If token-level,
    # mean-pool to keep this script vector-based.
    if emb.ndim == 2 and emb.size(0) == 1:
        emb = emb.squeeze(0)
    elif emb.ndim > 1:
        emb = emb.mean(dim=0)
    return emb


def safe_int_label(value, offset: int = 1) -> int:
    """Convert 1-indexed labels to 0-indexed labels; return -1 for missing/invalid."""
    if value is None:
        return -1
    try:
        if isinstance(value, float) and math.isnan(value):
            return -1
        out = int(value) - offset
        return out if out >= 0 else -1
    except Exception:
        return -1


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------


class EmbeddingTrajectoryDataset(Dataset):
    """
    Loads e0 and future target embeddings e1...eN from a JSON file.

    The dataset is flexible:
      - Uses explicit keys embedding_path_ts{k} if present.
      - If a future key is missing, tries to infer it by replacing _ts0 with _ts{k}.
      - If a target is missing and require_all_targets=False, a target mask marks it invalid.
    """

    COMPACT_TASKS = ("att", "margins", "cancer")
    ATTACHED_TASK_NAMES = (
        "att_ts0", "att_ts1", "att_ts2",
        "margins_ts0", "margins_ts1", "margins_ts2",
        "cancer",
    )

    def __init__(
        self,
        json_path: str,
        num_steps: int,
        require_all_targets: bool = True
    ):
        super().__init__()
        self.json_path = json_path
        self.num_steps = num_steps
        self.require_all_targets = require_all_targets

        with open(json_path, "r") as f:
            raw_samples = json.load(f)

        self.samples = []
        skipped_no_e0 = 0
        skipped_missing_targets = 0

        for sample in raw_samples:
            e0_path = self.resolve_embedding_path(sample, 0)
            if not e0_path or not os.path.exists(e0_path):
                skipped_no_e0 += 1
                continue

            target_paths = [self.resolve_embedding_path(sample, t) for t in range(1, num_steps + 1)]
            target_exists = [bool(p) and os.path.exists(p) for p in target_paths]

            if require_all_targets and not all(target_exists):
                skipped_missing_targets += 1
                continue

            self.samples.append(sample)

        self.skipped_no_e0 = skipped_no_e0
        self.skipped_missing_targets = skipped_missing_targets

    def __len__(self) -> int:
        return len(self.samples)

    def resolve_embedding_path(self, sample: Dict, t: int) -> Optional[str]:
        explicit_key = f"embedding_path_ts{t}"
        path = sample[explicit_key]
        if "mfm_embeddings1" not in path:
            path = path.replace("mfm_embeddings", "mfm_embeddings1")
        return path

    def make_compact_aux_labels(self, numeric: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        labels/masks for compact shared heads.
        Shape: [num_steps, 3] corresponding to att, margins, cancer.
        """
        labels = torch.full((self.num_steps, len(self.COMPACT_TASKS)), -1, dtype=torch.long)
        mask = torch.zeros((self.num_steps, len(self.COMPACT_TASKS)), dtype=torch.bool)

        for step_idx, t in enumerate(range(1, self.num_steps + 1)):
            values = {
                "att": safe_int_label(numeric.get(f"att_ts{t}")),
                "margins": safe_int_label(numeric.get(f"margins_ts{t}")),
                "cancer": safe_int_label(numeric.get("cancer")),
            }
            for task_idx, task_name in enumerate(self.COMPACT_TASKS):
                val = values[task_name]
                labels[step_idx, task_idx] = val
                mask[step_idx, task_idx] = val >= 0

        return labels, mask

    def make_attached_aux_labels(self, numeric: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        labels/masks for the original 7-head multitask head from your attached script.
        Shape: [num_steps, 7].

        For future step t:
          - att head index t, valid for t <= 2 in the original 7-head setup.
          - margins head index 3 + t, valid for t <= 2.
          - cancer head index 6, valid at all future steps.
        """
        labels = torch.full((self.num_steps, len(self.ATTACHED_TASK_NAMES)), -1, dtype=torch.long)
        mask = torch.zeros((self.num_steps, len(self.ATTACHED_TASK_NAMES)), dtype=torch.bool)

        for step_idx, t in enumerate(range(1, self.num_steps + 1)):
            # Original attached script has heads for ts0, ts1, ts2 only.
            if t <= 2:
                att_val = safe_int_label(numeric.get(f"att_ts{t}"))
                margins_val = safe_int_label(numeric.get(f"margins_ts{t}"))
                if att_val >= 0:
                    labels[step_idx, t] = att_val
                    mask[step_idx, t] = True
                if margins_val >= 0:
                    labels[step_idx, 3 + t] = margins_val
                    mask[step_idx, 3 + t] = True

            cancer_val = safe_int_label(numeric.get("cancer"))
            if cancer_val >= 0:
                labels[step_idx, 6] = cancer_val
                mask[step_idx, 6] = True

        return labels, mask

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        pid = str(sample.get("pid", idx))

        e0_path = self.resolve_embedding_path(sample, 0)
        e0 = load_embedding(e0_path)

        targets = []
        target_mask = []
        target_paths = []
        for t in range(1, self.num_steps + 1):
            path = self.resolve_embedding_path(sample, t)
            target_paths.append(path if path else "")
            if path and os.path.exists(path):
                targets.append(load_embedding(path))
                target_mask.append(True)
            else:
                targets.append(torch.zeros_like(e0))
                target_mask.append(False)

        targets = torch.stack(targets, dim=0)  # [N, D]
        target_mask = torch.tensor(target_mask, dtype=torch.bool)  # [N]

        numeric = sample.get("numeric_dict", {}) or {}
        compact_labels, compact_mask = self.make_compact_aux_labels(numeric)
        attached_labels, attached_mask = self.make_attached_aux_labels(numeric)

        return {
            "pid": pid,
            "e0": e0,
            "targets": targets,
            "target_mask": target_mask,
            "compact_aux_labels": compact_labels,
            "compact_aux_mask": compact_mask,
            "attached_aux_labels": attached_labels,
            "attached_aux_mask": attached_mask,
            "e0_path": e0_path,
            "target_paths": target_paths,
        }


# -----------------------------------------------------------------------------
# Forecasters
# -----------------------------------------------------------------------------


class ResidualMLPForecaster(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, num_steps: int, dropout: float):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_steps = num_steps
        self.net = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_steps * embedding_dim),
        )

    def forward(self, e0: torch.Tensor) -> torch.Tensor:
        deltas = self.net(e0).view(e0.size(0), self.num_steps, self.embedding_dim)
        return e0.unsqueeze(1) + deltas


class AutoregressiveMLPForecaster(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, num_steps: int, dropout: float):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_steps = num_steps
        self.time_embed = nn.Embedding(num_steps, embedding_dim)
        self.step_net = nn.Sequential(
            nn.LayerNorm(embedding_dim * 2),
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, e0: torch.Tensor) -> torch.Tensor:
        x = e0
        preds = []
        for t in range(self.num_steps):
            t_emb = self.time_embed.weight[t].unsqueeze(0).expand(e0.size(0), -1)
            delta = self.step_net(torch.cat([x, t_emb], dim=-1))
            x = x + delta
            preds.append(x)
        return torch.stack(preds, dim=1)


class GRUForecaster(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, num_steps: int, dropout: float):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_steps = num_steps
        self.init = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, hidden_dim),
            nn.Tanh(),
        )
        self.gru = nn.GRUCell(embedding_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, e0: torch.Tensor) -> torch.Tensor:
        h = self.init(e0)
        x = e0
        preds = []
        for _ in range(self.num_steps):
            h = self.gru(x, h)
            delta = self.out(self.dropout(h))
            x = x + delta
            preds.append(x)
        return torch.stack(preds, dim=1)


class LowRankLinearForecaster(nn.Module):
    def __init__(self, embedding_dim: int, rank: int, num_steps: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_steps = num_steps
        self.down = nn.ModuleList([nn.Linear(embedding_dim, rank, bias=False) for _ in range(num_steps)])
        self.up = nn.ModuleList([nn.Linear(rank, embedding_dim, bias=True) for _ in range(num_steps)])
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, e0: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm(e0)
        preds = []
        for t in range(self.num_steps):
            delta = self.up[t](self.down[t](x_norm))
            preds.append(e0 + delta)
        return torch.stack(preds, dim=1)


class TransformerForecaster(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        num_steps: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_steps = num_steps
        self.memory_proj = nn.Sequential(nn.LayerNorm(embedding_dim), nn.Linear(embedding_dim, embedding_dim))
        self.time_queries = nn.Parameter(torch.randn(num_steps, embedding_dim) * 0.02)
        layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.out = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, e0: torch.Tensor) -> torch.Tensor:
        batch_size = e0.size(0)
        memory = self.memory_proj(e0).unsqueeze(1)  # [B, 1, D]
        queries = self.time_queries.unsqueeze(0).expand(batch_size, -1, -1)  # [B, N, D]
        decoded = self.decoder(tgt=queries, memory=memory)
        deltas = self.out(decoded)
        return e0.unsqueeze(1) + deltas


def build_forecaster(args, embedding_dim: int) -> nn.Module:
    if args.model_type == "residual_mlp":
        return ResidualMLPForecaster(embedding_dim, args.hidden_dim, args.num_steps, args.dropout)
    if args.model_type == "autoreg_mlp":
        return AutoregressiveMLPForecaster(embedding_dim, args.hidden_dim, args.num_steps, args.dropout)
    if args.model_type == "gru":
        return GRUForecaster(embedding_dim, args.hidden_dim, args.num_steps, args.dropout)
    if args.model_type == "lowrank_linear":
        return LowRankLinearForecaster(embedding_dim, args.lowrank_dim, args.num_steps)
    if args.model_type == "transformer":
        return TransformerForecaster(
            embedding_dim,
            args.hidden_dim,
            args.num_steps,
            args.transformer_layers,
            args.transformer_heads,
            args.dropout,
        )
    raise ValueError(f"Unknown model_type: {args.model_type}")


# -----------------------------------------------------------------------------
# Auxiliary heads and losses
# -----------------------------------------------------------------------------


class CompactAuxHead(nn.Module):
    """Shared task-type heads applied to each predicted future embedding."""

    TASKS = ("att", "margins", "cancer")
    NUM_CLASSES = (8, 5, 2)

    def __init__(self, embedding_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or embedding_dim // 2
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_classes),
            )
            for num_classes in self.NUM_CLASSES
        ])

    def forward(self, embeddings: torch.Tensor) -> List[torch.Tensor]:
        # embeddings: [B, N, D]
        bsz, steps, dim = embeddings.shape
        flat = embeddings.reshape(bsz * steps, dim)
        outputs = []
        for head in self.heads:
            logits = head(flat).view(bsz, steps, -1)
            outputs.append(logits)
        return outputs


class AttachedStyleAuxHead(nn.Module):
    """
    Same architecture as your attached multitask script:
      heads: att_ts0/1/2, margins_ts0/1/2, cancer
      num_classes: 8,8,8,5,5,5,2
    """

    TASK_NAMES = (
        "att_ts0", "att_ts1", "att_ts2",
        "margins_ts0", "margins_ts1", "margins_ts2",
        "cancer",
    )
    NUM_CLASSES = (8, 8, 8, 5, 5, 5, 2)

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.classification_heads = nn.ModuleList()
        for num_classes in self.NUM_CLASSES:
            self.classification_heads.append(
                nn.Sequential(
                    nn.Linear(embedding_dim, embedding_dim // 2),
                    nn.ReLU(),
                    nn.Linear(embedding_dim // 2, num_classes),
                )
            )

    def forward(self, embeddings: torch.Tensor) -> List[torch.Tensor]:
        # embeddings: [B, N, D]
        bsz, steps, dim = embeddings.shape
        flat = embeddings.reshape(bsz * steps, dim)
        outputs = []
        for head in self.classification_heads:
            logits = head(flat).view(bsz, steps, -1)
            outputs.append(logits)
        return outputs


def load_aux_checkpoint(aux_head: nn.Module, checkpoint_path: str, logger: logging.Logger) -> None:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = aux_head.load_state_dict(state_dict, strict=False)
    logger.info(f"Loaded auxiliary checkpoint from {checkpoint_path}")
    if missing:
        logger.info(f"Aux checkpoint missing keys: {missing}")
    if unexpected:
        logger.info(f"Aux checkpoint unexpected keys: {unexpected}")


def build_aux_head(args, embedding_dim: int, logger: logging.Logger) -> Optional[nn.Module]:
    if args.aux_mode == "none" or args.aux_loss_weight <= 0:
        return None

    aux_style = args.aux_style
    if aux_style == "auto":
        aux_style = "attached" if args.aux_checkpoint else "compact"

    if aux_style == "compact":
        aux_head = CompactAuxHead(embedding_dim, hidden_dim=embedding_dim // 2)
    elif aux_style == "attached":
        aux_head = AttachedStyleAuxHead(embedding_dim)
    else:
        raise ValueError(f"Unknown aux_style: {args.aux_style}")

    if args.aux_checkpoint:
        load_aux_checkpoint(aux_head, args.aux_checkpoint, logger)

    if args.aux_mode == "frozen":
        for p in aux_head.parameters():
            p.requires_grad = False
        aux_head.eval()
        logger.info("Auxiliary head is frozen.")
    elif args.aux_mode == "joint":
        logger.info("Auxiliary head is trained jointly.")
    else:
        raise ValueError(f"Unknown aux_mode: {args.aux_mode}")

    aux_head.aux_style_resolved = aux_style  # type: ignore[attr-defined]
    return aux_head


def masked_classification_loss(
    outputs: List[torch.Tensor],
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    outputs: list of logits; each is [B, N, C_i]
    labels: [B, N, T]
    mask: [B, N, T]
    """
    total = None
    metrics = {}

    for task_idx, logits in enumerate(outputs):
        task_mask = mask[:, :, task_idx]
        if task_mask.sum() == 0:
            continue
        task_labels = labels[:, :, task_idx][task_mask]
        task_logits = logits[task_mask]
        loss_i = F.cross_entropy(task_logits, task_labels, reduction="mean")
        total = loss_i if total is None else total + loss_i
        with torch.no_grad():
            preds = task_logits.argmax(dim=-1)
            acc = (preds == task_labels).float().mean().item()
        metrics[f"aux_task_{task_idx}_loss"] = float(loss_i.detach().cpu())
        metrics[f"aux_task_{task_idx}_acc"] = acc

    if total is None:
        # Return a differentiable zero if no labels are available.
        device = outputs[0].device if outputs else labels.device
        total = torch.zeros((), device=device, requires_grad=True)
    metrics["aux_total"] = float(total.detach().cpu())
    return total, metrics


# -----------------------------------------------------------------------------
# Embedding losses and metrics
# -----------------------------------------------------------------------------


@dataclass
class EmbeddingStats:
    mean: torch.Tensor
    std: torch.Tensor


def compute_embedding_stats(dataset: EmbeddingTrajectoryDataset, logger: logging.Logger) -> EmbeddingStats:
    """Compute mean/std over e0 and available future embeddings in the training set."""
    logger.info("Computing embedding mean/std for standardized loss...")
    total = None
    total_sq = None
    count = 0

    for sample in tqdm(dataset.samples, desc="Embedding stats"):
        paths = []
        p0 = dataset.resolve_embedding_path(sample, 0)
        if p0 and os.path.exists(p0):
            paths.append(p0)
        for t in range(1, dataset.num_steps + 1):
            pt = dataset.resolve_embedding_path(sample, t)
            if pt and os.path.exists(pt):
                paths.append(pt)

        for path in paths:
            emb = load_embedding(path)
            if total is None:
                total = torch.zeros_like(emb)
                total_sq = torch.zeros_like(emb)
            total += emb
            total_sq += emb * emb
            count += 1

    if count == 0 or total is None or total_sq is None:
        raise RuntimeError("Could not compute embedding stats: no embeddings found.")

    mean = total / count
    var = torch.clamp(total_sq / count - mean * mean, min=1e-8)
    std = torch.sqrt(var)
    logger.info(f"Computed stats from {count} embedding vectors.")
    return EmbeddingStats(mean=mean, std=std)


def apply_standardization(x: torch.Tensor, stats: Optional[EmbeddingStats]) -> torch.Tensor:
    if stats is None:
        return x
    mean = stats.mean.to(x.device)
    std = stats.std.to(x.device)
    while mean.ndim < x.ndim:
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)
    return (x - mean) / std


def masked_vector_mse(preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # preds/targets: [B, N, D], mask: [B, N]
    diff = (preds - targets) ** 2
    mask_f = mask.unsqueeze(-1).float()
    denom = torch.clamp(mask_f.sum() * preds.size(-1), min=1.0)
    return (diff * mask_f).sum() / denom


def masked_vector_huber(preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, beta: float) -> torch.Tensor:
    loss = F.smooth_l1_loss(preds, targets, reduction="none", beta=beta)
    mask_f = mask.unsqueeze(-1).float()
    denom = torch.clamp(mask_f.sum() * preds.size(-1), min=1.0)
    return (loss * mask_f).sum() / denom


def masked_cosine_loss(preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    cos = F.cosine_similarity(preds, targets, dim=-1)  # [B, N]
    loss = 1.0 - cos
    denom = torch.clamp(mask.float().sum(), min=1.0)
    return (loss * mask.float()).sum() / denom


def contrastive_future_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """InfoNCE per future step using in-batch negatives."""
    losses = []
    for t in range(preds.size(1)):
        valid = mask[:, t]
        if valid.sum() < 2:
            continue
        p = F.normalize(preds[:, t][valid], dim=-1)
        y = F.normalize(targets[:, t][valid], dim=-1)
        logits = p @ y.T / temperature
        labels = torch.arange(p.size(0), device=p.device)
        losses.append(F.cross_entropy(logits, labels))
    if not losses:
        return torch.zeros((), device=preds.device)
    return torch.stack(losses).mean()


def compute_total_loss(
    args,
    preds: torch.Tensor,
    targets: torch.Tensor,
    e0: torch.Tensor,
    target_mask: torch.Tensor,
    aux_head: Optional[nn.Module],
    batch: Dict,
    stats: Optional[EmbeddingStats],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    metrics = {}

    preds_loss_space = apply_standardization(preds, stats)
    targets_loss_space = apply_standardization(targets, stats)
    e0_loss_space = apply_standardization(e0, stats)

    if args.base_loss == "mse":
        base = masked_vector_mse(preds_loss_space, targets_loss_space, target_mask)
    elif args.base_loss == "huber":
        base = masked_vector_huber(preds_loss_space, targets_loss_space, target_mask, beta=args.huber_beta)
    else:
        raise ValueError(f"Unknown base_loss: {args.base_loss}")

    total = args.base_loss_weight * base
    metrics[f"loss_{args.base_loss}"] = float(base.detach().cpu())

    if args.cosine_weight > 0:
        cos = masked_cosine_loss(preds, targets, target_mask)
        total = total + args.cosine_weight * cos
        metrics["loss_cosine"] = float(cos.detach().cpu())

    if args.delta_weight > 0:
        pred_delta = preds_loss_space - e0_loss_space.unsqueeze(1)
        true_delta = targets_loss_space - e0_loss_space.unsqueeze(1)
        delta = masked_vector_mse(pred_delta, true_delta, target_mask)
        total = total + args.delta_weight * delta
        metrics["loss_delta"] = float(delta.detach().cpu())

    if args.contrastive_weight > 0:
        contrast = contrastive_future_loss(preds, targets, target_mask, temperature=args.contrastive_temperature)
        total = total + args.contrastive_weight * contrast
        metrics["loss_contrastive"] = float(contrast.detach().cpu())

    if aux_head is not None and args.aux_loss_weight > 0:
        aux_outputs = aux_head(preds)
        aux_style = getattr(aux_head, "aux_style_resolved", args.aux_style)
        if aux_style == "compact":
            labels = batch["compact_aux_labels"].to(preds.device)
            mask = batch["compact_aux_mask"].to(preds.device)
        elif aux_style == "attached":
            labels = batch["attached_aux_labels"].to(preds.device)
            mask = batch["attached_aux_mask"].to(preds.device)
        else:
            raise ValueError(f"Unknown aux style: {aux_style}")

        # Do not compute aux loss for missing target embeddings.
        mask = mask & target_mask.unsqueeze(-1)
        aux_loss, aux_metrics = masked_classification_loss(aux_outputs, labels, mask)
        total = total + args.aux_loss_weight * aux_loss
        metrics.update(aux_metrics)

    metrics["loss_total"] = float(total.detach().cpu())
    return total, metrics


@torch.no_grad()
def compute_eval_metrics(preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> Dict[str, float]:
    metrics = {}
    valid = mask.bool()
    if valid.sum() == 0:
        return {"mse": float("nan"), "cosine": float("nan"), "r2": float("nan")}

    p = preds[valid]
    y = targets[valid]
    metrics["mse"] = float(F.mse_loss(p, y).detach().cpu())
    metrics["cosine"] = float(F.cosine_similarity(p, y, dim=-1).mean().detach().cpu())

    # R^2 over all valid elements.
    sse = torch.sum((p - y) ** 2)
    centered = y - y.mean(dim=0, keepdim=True)
    sst = torch.sum(centered ** 2).clamp_min(1e-8)
    metrics["r2"] = float((1.0 - sse / sst).detach().cpu())

    for t in range(preds.size(1)):
        valid_t = mask[:, t]
        if valid_t.sum() == 0:
            continue
        pt = preds[:, t][valid_t]
        yt = targets[:, t][valid_t]
        metrics[f"ts{t+1}_mse"] = float(F.mse_loss(pt, yt).detach().cpu())
        metrics[f"ts{t+1}_cosine"] = float(F.cosine_similarity(pt, yt, dim=-1).mean().detach().cpu())

    return metrics


# -----------------------------------------------------------------------------
# Train/eval loops
# -----------------------------------------------------------------------------


def move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def train_one_epoch(
    args,
    forecaster: nn.Module,
    aux_head: Optional[nn.Module],
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    stats: Optional[EmbeddingStats],
    epoch: int,
) -> Dict[str, float]:
    forecaster.train()
    if aux_head is not None:
        if args.aux_mode == "frozen":
            aux_head.eval()
        else:
            aux_head.train()

    running = {}
    n_batches = 0

    for batch in tqdm(loader, desc=f"Epoch {epoch} [train]"):
        batch = move_batch_to_device(batch, device)
        e0 = batch["e0"]
        targets = batch["targets"]
        target_mask = batch["target_mask"]

        optimizer.zero_grad(set_to_none=True)
        preds = forecaster(e0)
        loss, loss_metrics = compute_total_loss(
            args, preds, targets, e0, target_mask, aux_head, batch, stats
        )
        loss.backward()
        optimizer.step()

        for k, v in loss_metrics.items():
            running[k] = running.get(k, 0.0) + float(v)
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in running.items()}


@torch.no_grad()
def evaluate(
    args,
    forecaster: nn.Module,
    aux_head: Optional[nn.Module],
    loader: DataLoader,
    device: torch.device,
    stats: Optional[EmbeddingStats],
    split_name: str,
) -> Dict[str, float]:
    forecaster.eval()
    if aux_head is not None:
        aux_head.eval()

    running_loss = {}
    all_preds = []
    all_targets = []
    all_masks = []
    n_batches = 0

    for batch in tqdm(loader, desc=f"Evaluate [{split_name}]"):
        batch = move_batch_to_device(batch, device)
        e0 = batch["e0"]
        targets = batch["targets"]
        target_mask = batch["target_mask"]
        preds = forecaster(e0)

        _, loss_metrics = compute_total_loss(
            args, preds, targets, e0, target_mask, aux_head, batch, stats
        )
        for k, v in loss_metrics.items():
            running_loss[k] = running_loss.get(k, 0.0) + float(v)

        all_preds.append(preds.cpu())
        all_targets.append(targets.cpu())
        all_masks.append(target_mask.cpu())
        n_batches += 1

    metrics = {k: v / max(n_batches, 1) for k, v in running_loss.items()}
    if all_preds:
        preds_cat = torch.cat(all_preds, dim=0)
        targets_cat = torch.cat(all_targets, dim=0)
        masks_cat = torch.cat(all_masks, dim=0)
        metrics.update(compute_eval_metrics(preds_cat, targets_cat, masks_cat))
    return metrics


def save_json(obj: Dict, path: str) -> None:
    def convert(x):
        if isinstance(x, (np.floating, np.integer)):
            return x.item()
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().tolist()
        return x

    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=convert)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Train embedding trajectory prediction models.")

    # Data
    parser.add_argument("--train_json", type=str, 
                        default="/home/avepa/MedTrinity-25M/nlst_train_aux_vqa_traj_v3_seed0.json", 
                        help="JSON file with training samples.")
    parser.add_argument("--val_json", type=str, 
                        default="/home/avepa/MedTrinity-25M/nlst_val_aux_vqa_traj_v3_seed0.json", 
                        help="JSON file with validation samples.")
    parser.add_argument("--test_json", type=str, 
                        default="/home/avepa/MedTrinity-25M/nlst_test_aux_vqa_traj_v3_seed0.json", 
                        help="JSON file with test samples.")
    parser.add_argument("--num_steps", type=int, default=2, help="Predict embeddings ts1...tsN from ts0.")
    parser.add_argument("--require_all_targets", default=False, action="store_true", 
                        help="Skip samples missing any future embedding.")

    # Model
    parser.add_argument(
        "--model_type",
        type=str,
        default="residual_mlp",
        choices=["residual_mlp", "autoreg_mlp", "gru", "lowrank_linear", "transformer"],
    )
    parser.add_argument("--embedding_dim", type=int, default=1024)
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lowrank_dim", type=int, default=128)
    parser.add_argument("--transformer_layers", type=int, default=2)
    parser.add_argument("--transformer_heads", type=int, default=8)

    # Losses
    parser.add_argument("--base_loss", type=str, default="mse", choices=["mse", "huber"])
    parser.add_argument("--base_loss_weight", type=float, default=1.0)
    parser.add_argument("--huber_beta", type=float, default=1.0)
    parser.add_argument("--cosine_weight", type=float, default=0.5)
    parser.add_argument("--delta_weight", type=float, default=0.5)
    parser.add_argument("--contrastive_weight", type=float, default=0.0)
    parser.add_argument("--contrastive_temperature", type=float, default=0.1)
    parser.add_argument("--standardize_loss", action="store_true", help="Compute train mean/std and standardize MSE/Huber/delta losses.")

    # Auxiliary multitask loss
    parser.add_argument("--aux_mode", type=str, default="none", choices=["none", "joint", "frozen"])
    parser.add_argument("--aux_style", type=str, default="auto", choices=["auto", "compact", "attached"])
    parser.add_argument("--aux_checkpoint", type=str, default="", help="Optional checkpoint for frozen/joint aux head.")
    parser.add_argument("--aux_loss_weight", type=float, default=0.0)

    # Optimization
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--gpu", type=int, default=0)

    # Output
    parser.add_argument("--output_dir", type=str, default="./embedding_trajectory_output")
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--save_every_epoch", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    run_name = (
        f"{args.model_type}_N{args.num_steps}_{args.base_loss}"
        f"_cos{args.cosine_weight}_delta{args.delta_weight}"
        f"_aux{args.aux_mode}{args.aux_loss_weight}{args.tag}"
    )
    output_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logger(output_dir)
    logger.info(f"Arguments: {vars(args)}")

    device = get_device(args.device, args.gpu)
    logger.info(f"Using device: {device}")

    train_dataset = EmbeddingTrajectoryDataset(
        args.train_json,
        num_steps=args.num_steps,
        require_all_targets=args.require_all_targets
    )
    val_dataset = EmbeddingTrajectoryDataset(
        args.val_json,
        num_steps=args.num_steps,
        require_all_targets=args.require_all_targets
    )
    test_dataset = EmbeddingTrajectoryDataset(
        args.test_json,
        num_steps=args.num_steps,
        require_all_targets=args.require_all_targets
    )

    logger.info(
        f"Dataset sizes: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}"
    )
    logger.info(
        f"Skipped train samples: no_e0={train_dataset.skipped_no_e0}, "
        f"require_all_targets={train_dataset.require_all_targets}"
    )
    if len(train_dataset) == 0:
        raise RuntimeError("No training samples found. Check JSON paths and path prefix rewrite args.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    stats = compute_embedding_stats(train_dataset, logger) if args.standardize_loss else None
    if stats is not None:
        torch.save({"mean": stats.mean, "std": stats.std}, os.path.join(output_dir, "embedding_stats.pt"))

    forecaster = build_forecaster(args, args.embedding_dim).to(device)
    aux_head = build_aux_head(args, args.embedding_dim, logger)
    if aux_head is not None:
        aux_head = aux_head.to(device)

    # Optimizer includes joint aux head params but excludes frozen params automatically.
    params = list(forecaster.parameters())
    if aux_head is not None and args.aux_mode == "joint":
        params += list(aux_head.parameters())
    params = [p for p in params if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=args.weight_decay)

    save_json(vars(args), os.path.join(output_dir, "args.json"))

    best_val_loss = float("inf")
    best_epoch = -1
    history = []
    best_path = os.path.join(output_dir, "best_model.pt")

    for epoch in range(1, args.num_epochs + 1):
        train_metrics = train_one_epoch(
            args, forecaster, aux_head, train_loader, optimizer, device, stats, epoch
        )
        val_metrics = evaluate(args, forecaster, aux_head, val_loader, device, stats, "val")

        row = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(row)
        save_json({"history": history}, os.path.join(output_dir, "history.json"))

        logger.info(f"Epoch {epoch}/{args.num_epochs}")
        logger.info(f"Train: {train_metrics}")
        logger.info(f"Val:   {val_metrics}")

        val_loss = val_metrics.get("loss_total", float("inf"))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            checkpoint = {
                "epoch": epoch,
                "args": vars(args),
                "forecaster_state_dict": forecaster.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "val_metrics": val_metrics,
            }
            if aux_head is not None:
                checkpoint["aux_head_state_dict"] = aux_head.state_dict()
                checkpoint["aux_style"] = getattr(aux_head, "aux_style_resolved", args.aux_style)
            if stats is not None:
                checkpoint["embedding_stats"] = {"mean": stats.mean, "std": stats.std}
            torch.save(checkpoint, best_path)
            logger.info(f"Saved new best checkpoint to {best_path}")

        if args.save_every_epoch:
            torch.save(
                {
                    "epoch": epoch,
                    "forecaster_state_dict": forecaster.state_dict(),
                    "aux_head_state_dict": aux_head.state_dict() if aux_head is not None else None,
                    "val_metrics": val_metrics,
                },
                os.path.join(output_dir, f"epoch_{epoch}.pt"),
            )

    logger.info(f"Training complete. Best epoch={best_epoch}, best val loss={best_val_loss:.6f}")

    # Load best model for final test evaluation.
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    forecaster.load_state_dict(checkpoint["forecaster_state_dict"])
    if aux_head is not None and "aux_head_state_dict" in checkpoint:
        aux_head.load_state_dict(checkpoint["aux_head_state_dict"])

    test_metrics = evaluate(args, forecaster, aux_head, test_loader, device, stats, "test")
    logger.info(f"Test: {test_metrics}")
    save_json(
        {
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "test_metrics": test_metrics,
        },
        os.path.join(output_dir, "test_results.json"),
    )
    logger.info(f"Saved test results to {os.path.join(output_dir, 'test_results.json')}")


if __name__ == "__main__":
    main()
