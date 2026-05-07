#!/usr/bin/env python3
"""
Generate predicted future embeddings from ts0 using a trained trajectory model.

Loads ts0 embeddings, predicts ts1..tsN with a saved checkpoint, and writes
each prediction to a safetensors file (default extension: .st) in the output
directory, with the same "embeddings" key the training-side loader expects.
Default filename pattern is pid{pid}_ts{t}.st (configurable via
--filename_template). Also writes a manifest JSON cataloging the predicted
paths per pid, and (optionally) a copy of each input JSON whose
embedding_path_ts{t} keys point at the predicted files so downstream training
scripts can consume them transparently.

Example: predict for train/val/test splits and write companion JSONs.

python predict_embedding_trajectory.py \
  --input_jsons \
      /home/avepa/MedTrinity-25M/nlst_train_aux_vqa_traj_v3_seed0.json \
      /home/avepa/MedTrinity-25M/nlst_val_aux_vqa_traj_v3_seed0.json \
      /home/avepa/MedTrinity-25M/nlst_test_aux_vqa_traj_v3_seed0.json \
  --checkpoint ./embedding_trajectory_output/<run_name>/best_model.pt \
  --output_dir  ./predicted_embeddings/<run_name> \
  --write_replaced_json \
  --batch_size 64
"""

import argparse
import json
import logging
import os
from typing import Dict, List, Optional

import torch
from safetensors.torch import save_file
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Reuse model construction + helpers from the training script. The training
# script must be on PYTHONPATH (or sit next to this file) for the import to
# resolve.
from train_embedding_trajectory import (
    build_forecaster,
    EmbeddingStats,
    get_device,
    load_embedding,
    set_seed,
)


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------


def setup_inference_logger(output_dir: str) -> logging.Logger:
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger("trajectory_inference")
    logger.setLevel(logging.INFO)
    logger.handlers = []  # avoid duplicate handlers if main() called twice
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    fh = logging.FileHandler(os.path.join(output_dir, "predict.log"), mode="w")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


# -----------------------------------------------------------------------------
# Dataset (ts0 only, mirrors training-side path resolution)
# -----------------------------------------------------------------------------


class TS0InferenceDataset(Dataset):
    """Resolves and loads only the ts0 embedding for each sample."""

    def __init__(self, json_path: str, split_name: str = "predict",
                 logger: Optional[logging.Logger] = None):
        super().__init__()
        self.json_path = json_path
        self.split_name = split_name

        with open(json_path, "r") as f:
            raw_samples = json.load(f)

        self.samples: List[Dict] = []
        self.skipped_no_e0 = 0
        for sample in raw_samples:
            e0_path = self.resolve_embedding_path(sample, 0)
            if not e0_path or not os.path.exists(e0_path):
                self.skipped_no_e0 += 1
                continue
            self.samples.append({**sample, "_resolved_e0_path": e0_path})

        msg = (
            f"[{split_name}] {os.path.basename(json_path)}: kept {len(self.samples)} samples, "
            f"skipped {self.skipped_no_e0} (no ts0)."
        )
        if logger is not None:
            logger.info(msg)
        else:
            print(msg)

    @staticmethod
    def resolve_embedding_path(sample: Dict, t: int) -> Optional[str]:
        """Same resolution rule as EmbeddingTrajectoryDataset in the training script."""
        key = f"embedding_path_ts{t}"
        path = sample.get(key, "")
        if not path:
            return None
        return path

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        e0 = load_embedding(sample["_resolved_e0_path"])
        return {
            "pid": str(sample.get("pid", idx)),
            "e0": e0,
            "e0_path": sample["_resolved_e0_path"],
        }


def inference_collate(batch: List[Dict]) -> Dict:
    """Custom collate so list-of-strings fields aren't transposed by default_collate."""
    return {
        "e0": torch.stack([b["e0"] for b in batch], dim=0),
        "pid": [b["pid"] for b in batch],
        "e0_path": [b["e0_path"] for b in batch],
    }


# -----------------------------------------------------------------------------
# Checkpoint loading
# -----------------------------------------------------------------------------


class _ArgsNS:
    """Lightweight namespace that exposes a dict's keys as attributes."""

    def __init__(self, d: Dict):
        for k, v in d.items():
            setattr(self, k, v)


def load_forecaster_from_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    num_steps_override: int = -1,
    embedding_dim_override: int = -1,
    logger: Optional[logging.Logger] = None,
):
    """Reconstruct the forecaster using args saved at training time and load weights."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved_args = ckpt.get("args")
    if not saved_args:
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} does not contain an 'args' dict; "
            f"cannot reconstruct the forecaster automatically."
        )

    args_ns = _ArgsNS(saved_args)

    if num_steps_override > 0 and num_steps_override != args_ns.num_steps:
        if logger is not None:
            logger.warning(
                f"Overriding num_steps from {args_ns.num_steps} -> {num_steps_override}. "
                f"Note: residual_mlp, autoreg_mlp, lowrank_linear, and transformer all have "
                f"per-step parameters and will load with strict=False; only gru truly extrapolates "
                f"with a fully shared cell."
            )
        args_ns.num_steps = num_steps_override

    embedding_dim = (
        embedding_dim_override if embedding_dim_override > 0
        else getattr(args_ns, "embedding_dim", 1024)
    )

    forecaster = build_forecaster(args_ns, embedding_dim)
    state_dict = ckpt["forecaster_state_dict"]
    missing, unexpected = forecaster.load_state_dict(state_dict, strict=False)
    if logger is not None:
        if missing:
            logger.warning(f"Missing keys when loading forecaster: {missing}")
        if unexpected:
            logger.warning(f"Unexpected keys when loading forecaster: {unexpected}")

    forecaster = forecaster.to(device)
    forecaster.eval()

    stats = None
    if "embedding_stats" in ckpt:
        s = ckpt["embedding_stats"]
        stats = EmbeddingStats(mean=s["mean"].to(device), std=s["std"].to(device))

    if logger is not None:
        logger.info(
            f"Loaded forecaster from {checkpoint_path} "
            f"(model_type={args_ns.model_type}, num_steps={args_ns.num_steps}, "
            f"embedding_dim={embedding_dim}, best_epoch={ckpt.get('epoch')})."
        )
        if stats is not None:
            logger.info(
                "Embedding stats present in checkpoint; predictions are in the original "
                "embedding space (stats are training-loss-only), so no de-standardization is applied."
            )

    return forecaster, args_ns, stats, embedding_dim


# -----------------------------------------------------------------------------
# Inference
# -----------------------------------------------------------------------------


@torch.no_grad()
def predict_dataset(
    forecaster: torch.nn.Module,
    dataset: TS0InferenceDataset,
    output_dir: str,
    filename_template: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    overwrite: bool,
    include_ts0: bool,
    num_steps: int,
    logger: logging.Logger,
) -> List[Dict]:
    """Run forecaster on the dataset and write each predicted embedding to disk.

    Returns a list of manifest entries (one per pid).
    """
    os.makedirs(output_dir, exist_ok=True)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=inference_collate,
    )

    entries: List[Dict] = []
    n_saved = 0
    n_skipped_existing = 0
    seen_pids = set()
    duplicate_pids: List[str] = []

    for batch in tqdm(loader, desc=f"Predicting [{dataset.split_name}]"):
        e0 = batch["e0"].to(device, non_blocking=True)
        preds = forecaster(e0)  # [B, num_steps, D]
        preds_cpu = preds.detach().cpu()
        e0_cpu = e0.detach().cpu()

        for b in range(preds_cpu.size(0)):
            pid = batch["pid"][b]
            if pid in seen_pids:
                duplicate_pids.append(pid)
            seen_pids.add(pid)

            entry: Dict = {
                "pid": pid,
                "input_split": dataset.split_name,
                "embedding_path_ts0": batch["e0_path"][b],
            }

            if include_ts0:
                fn0 = filename_template.format(pid=pid, t=0)
                outp0 = os.path.join(output_dir, fn0)
                if overwrite or not os.path.exists(outp0):
                    save_file({"embeddings": e0_cpu[b].contiguous()}, outp0)
                    n_saved += 1
                else:
                    n_skipped_existing += 1
                entry["predicted_embedding_path_ts0"] = outp0

            for t in range(num_steps):
                step = t + 1
                fn = filename_template.format(pid=pid, t=step)
                outp = os.path.join(output_dir, fn)
                if overwrite or not os.path.exists(outp):
                    save_file({"embeddings": preds_cpu[b, t].contiguous()}, outp)
                    n_saved += 1
                else:
                    n_skipped_existing += 1
                entry[f"predicted_embedding_path_ts{step}"] = outp

            entries.append(entry)

    logger.info(
        f"[{dataset.split_name}] wrote {n_saved} files, "
        f"skipped {n_skipped_existing} pre-existing (overwrite={overwrite})."
    )
    if duplicate_pids:
        logger.warning(
            f"[{dataset.split_name}] saw {len(duplicate_pids)} duplicate pids; "
            f"later writes overwrote earlier ones (use --filename_template to disambiguate)."
        )
    return entries


# -----------------------------------------------------------------------------
# Optional: write a copy of each input JSON with predicted paths injected
# -----------------------------------------------------------------------------


def write_replaced_input_json(
    input_json_path: str,
    output_path: str,
    pid_to_predictions: Dict[str, Dict[int, str]],
    num_steps: int,
    replace_originals: bool,
    logger: logging.Logger,
) -> None:
    """Write a JSON next to the manifest that mirrors the input but adds predicted paths.

    By default, predicted paths are added under predicted_embedding_path_ts{t}, leaving
    embedding_path_ts{t} untouched. With replace_originals=True, embedding_path_ts{t}
    is overwritten with the predicted path and the original is preserved under
    original_embedding_path_ts{t}.
    """
    with open(input_json_path, "r") as f:
        samples = json.load(f)

    n_updated = 0
    out_samples = []
    for sample in samples:
        pid = str(sample.get("pid", ""))
        new_sample = dict(sample)
        if pid in pid_to_predictions:
            preds = pid_to_predictions[pid]
            for t in range(1, num_steps + 1):
                if t not in preds:
                    continue
                key = f"embedding_path_ts{t}"
                if replace_originals:
                    if key in new_sample:
                        new_sample[f"original_{key}"] = new_sample[key]
                    new_sample[key] = preds[t]
                else:
                    new_sample[f"predicted_embedding_path_ts{t}"] = preds[t]
            n_updated += 1
        out_samples.append(new_sample)

    with open(output_path, "w") as f:
        json.dump(out_samples, f, indent=2)
    logger.info(
        f"Wrote {output_path}: {n_updated}/{len(samples)} samples updated "
        f"(replace_originals={replace_originals})."
    )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate predicted future embeddings using a trained trajectory model."
    )

    # Inputs
    p.add_argument("--input_jsons", type=str, nargs="+", required=True,
                   help="One or more JSON files (same format as training) to predict for.")
    p.add_argument("--input_split_names", type=str, nargs="+", default=None,
                   help="Optional names for each input JSON; recorded in the manifest. "
                        "If omitted, defaults to train/val/test for 3 inputs, predict for 1, "
                        "predict_{i} otherwise.")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to a trained checkpoint (e.g. .../best_model.pt).")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Directory in which to save predicted embedding files.")

    # Architecture overrides (default: use values stored inside the checkpoint)
    p.add_argument("--num_steps", type=int, default=-1,
                   help="Override number of future steps to predict. -1 (default) = use checkpoint setting.")
    p.add_argument("--embedding_dim", type=int, default=-1,
                   help="Override embedding dim. -1 (default) = use checkpoint setting.")

    # Output options
    p.add_argument("--filename_template", type=str, default="pid{pid}_ts{t}.st",
                   help="Template for output filenames. Available variables: {pid}, {t}.")
    p.add_argument("--manifest_json", type=str, default="",
                   help="Path to write the global manifest JSON. Default: <output_dir>/manifest.json.")
    p.add_argument("--include_ts0", action="store_true",
                   help="Also copy each ts0 embedding to the output directory.")
    p.add_argument("--write_replaced_json", action="store_true",
                   help="For each input JSON, also write <basename>_with_predictions.json into "
                        "output_dir, populated with predicted_embedding_path_ts{t} keys.")
    p.add_argument("--replace_originals", action="store_true",
                   help="When --write_replaced_json is set, overwrite embedding_path_ts{t} with "
                        "predicted paths (originals preserved under original_embedding_path_ts{t}).")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing output files; otherwise skip them.")

    # Compute
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)

    return p.parse_args()


def default_split_name(idx: int, total: int) -> str:
    if total == 1:
        return "predict"
    if total == 3:
        return ["train", "val", "test"][idx]
    return f"predict_{idx}"


def main():
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_inference_logger(args.output_dir)
    logger.info(f"Arguments: {vars(args)}")

    device = get_device(args.device, args.gpu)
    logger.info(f"Using device: {device}")

    # Load forecaster from checkpoint
    forecaster, saved_args, _stats, embedding_dim = load_forecaster_from_checkpoint(
        args.checkpoint,
        device=device,
        num_steps_override=args.num_steps,
        embedding_dim_override=args.embedding_dim,
        logger=logger,
    )
    num_steps = saved_args.num_steps

    # Resolve split names
    if args.input_split_names is not None:
        if len(args.input_split_names) != len(args.input_jsons):
            raise ValueError(
                f"--input_split_names ({len(args.input_split_names)}) must match "
                f"--input_jsons ({len(args.input_jsons)})."
            )
        split_names = args.input_split_names
    else:
        split_names = [default_split_name(i, len(args.input_jsons))
                       for i in range(len(args.input_jsons))]

    all_entries: List[Dict] = []

    for input_path, split_name in zip(args.input_jsons, split_names):
        dataset = TS0InferenceDataset(input_path, split_name=split_name, logger=logger)
        if len(dataset) == 0:
            logger.warning(f"No usable samples in {input_path}; skipping.")
            continue

        entries = predict_dataset(
            forecaster=forecaster,
            dataset=dataset,
            output_dir=args.output_dir,
            filename_template=args.filename_template,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            overwrite=args.overwrite,
            include_ts0=args.include_ts0,
            num_steps=num_steps,
            logger=logger,
        )
        all_entries.extend(entries)

        if args.write_replaced_json:
            # Build pid -> {t: predicted_path} map for this input only
            pid_map: Dict[str, Dict[int, str]] = {}
            for e in entries:
                sub = pid_map.setdefault(e["pid"], {})
                for t in range(1, num_steps + 1):
                    key = f"predicted_embedding_path_ts{t}"
                    if key in e:
                        sub[t] = e[key]

            base = os.path.splitext(os.path.basename(input_path))[0]
            out_json = os.path.join(args.output_dir, f"{base}_with_predictions.json")
            write_replaced_input_json(
                input_json_path=input_path,
                output_path=out_json,
                pid_to_predictions=pid_map,
                num_steps=num_steps,
                replace_originals=args.replace_originals,
                logger=logger,
            )

    # Write the global manifest
    manifest_path = args.manifest_json or os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(
            {
                "num_steps": num_steps,
                "embedding_dim": embedding_dim,
                "model_type": getattr(saved_args, "model_type", "unknown"),
                "checkpoint": os.path.abspath(args.checkpoint),
                "filename_template": args.filename_template,
                "include_ts0": args.include_ts0,
                "entries": all_entries,
            },
            f,
            indent=2,
        )
    logger.info(f"Wrote global manifest with {len(all_entries)} entries to {manifest_path}.")
    logger.info(f"All predicted embeddings live in {os.path.abspath(args.output_dir)}.")


if __name__ == "__main__":
    main()