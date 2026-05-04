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
from safetensors.torch import save_file, load_file


# Import from M3FM
import models.m3fm as m3fm
from util import ConfigFile, txt2embed, get_sincos_size_embed
# Import the get_data function from M3FM
from data import get_data


def setup_logger(log_file="training.log", log_to_console=True):
    logger = logging.getLogger("embedding_logger")
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
            embedding_paths = sorted(set([s["embedding_path"] for s in self.samples]))
            embedding_paths_ = set()
            for path in embedding_paths:
                if "_ts0" in path:
                    path1 = path.replace("_ts0", "_ts1")
                    path2 = path.replace("_ts0", "_ts2")
                    embedding_paths_.add(path)
                    if os.path.exists(path1):
                        embedding_paths_.add(path1)
                    if os.path.exists(path2):
                        embedding_paths_.add(path2)
            self.samples = sorted(embedding_paths_)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        embedding_path = self.samples[idx]

        # Create input dict for center crop preprocessing
        input_dict = {
            'ct_path': embedding_path,
            'question': 'Predict multitask answers',
            'clinical_txt': '',
        }
        
        # Use our custom center crop function
        processed_data = get_data_center_crop(input_dict, self.config_args)
        
        return {
            "image": processed_data['data'][0],  # Extract the preprocessed image
            "size_embed": processed_data['size_embed'][0],  # Pass the precomputed size_embed
            'ct_path': embedding_path,
        }


class CTViTEmbedHead(nn.Module):
    def __init__(
        self,
        m3fm_model: nn.Module,
        hidden_dim=1024,
        freeze_encoder=True
    ):
        super().__init__()
        self.m3fm_model = m3fm_model
        self.hidden_dim = hidden_dim

    def forward(self, image, size_embed):  # Accept size_embed as parameter
        B = image.size(0)

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
            mdl_feats = feats.mean(dim=1)
        
        return mdl_feats


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
    output_dir: str = field(default="./m3fm_embeddings1", metadata={"help": "Output directory."})
    device: str = field(default="cuda", metadata={"help": "Device to use."})
    gpu: int = field(default=0, metadata={"help": "GPU ID to use."})
    tag: str = field(default="", metadata={"help": "Additional tag for output directory."})


def generate_embeddings(model, loader, args, output_dir, tag="train"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"[Generate {tag} Embeddings]"):
            path = batch['ct_path'][0]
            img = batch["image"].cuda(args.gpu)
            size_embed = batch["size_embed"].cuda(args.gpu)  # Get precomputed size_embed
            embedding_feats = model(img, size_embed)[0]
            new_save_path = os.path.join(output_dir, os.path.basename(path)[:-3] + "st")
            save_file({"embeddings": embedding_feats}, new_save_path) 

def main():
    parser = HfArgumentParser(TrainingArguments)
    (args,) = parser.parse_args_into_dataclasses()

    output_dir = args.output_dir + args.tag
    os.makedirs(output_dir, exist_ok=True)
    
    logger = setup_logger(
        log_file=os.path.join(output_dir, "embeddings.log"),
        log_to_console=True
    )

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
    if "best_model.pt" in args.model_path:
        state_dict = torch.load(args.model_path, map_location='cpu', weights_only=False)['model_state_dict']
        state_dict = {key.replace("m3fm_model.", ""): val for key, val in state_dict.items()}
    else:
        state_dict = torch.load(args.model_path, map_location='cpu', weights_only=False)
    msg = model_full.load_state_dict(state_dict, strict=False)
    logger.info(f"Loaded checkpoint with message: {msg}")
    
    # Get embed_dim_img from the loaded model (line 76 in m3fm.py)
    embed_dim_img = model_full.embed_dim_img
    logger.info(f"Using embed_dim_img={embed_dim_img} from M3FM model")

    # Build cancer classifier
    model = CTViTEmbedHead(
        m3fm_model=model_full,
        hidden_dim=embed_dim_img,
    ).cuda(args.gpu)
    model.eval()

    # Build datasets using M3FM's get_data function - pass img_root
    train_dataset = AuxVisionDataset(args.train_json, mode="train", config_args=config_args, img_root=args.img_root)
    val_dataset = AuxVisionDataset(args.val_json, mode="val", config_args=config_args, img_root=args.img_root)
    test_dataset = AuxVisionDataset(args.test_json, mode="test", config_args=config_args, img_root=args.img_root)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, drop_last=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, drop_last=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=True, num_workers=4)

    logger.info(f"Dataset sizes => train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
    logger.info(f"Using crop_size: {crop_size}")

    generate_embeddings(model, train_loader, args, os.path.join(output_dir, "train"), tag="train")
    generate_embeddings(model, val_loader, args, os.path.join(output_dir, "val"), tag="val")
    generate_embeddings(model, test_loader, args, os.path.join(output_dir, "test"), tag="test")

if __name__ == "__main__":
    main()