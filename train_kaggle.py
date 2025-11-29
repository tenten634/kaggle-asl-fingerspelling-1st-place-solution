"""
Kaggle-compatible training script for ASL Fingerspelling Recognition
This is a modified version of train.py that works in Kaggle Notebooks
"""

import os
import glob
import gc
from copy import copy
import numpy as np
import pandas as pd
import importlib
import sys
from tqdm import tqdm
import argparse
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import transformers
import random
from utils import calc_grad_norm, set_seed

# Check if running in Kaggle
IN_KAGGLE = os.path.exists('/kaggle/input') and os.path.exists('/kaggle/working')
if IN_KAGGLE:
    print("✅ Running in Kaggle Notebook")
    print(f"   Working directory: /kaggle/working")
    print(f"   Input directory: /kaggle/input")
else:
    print("⚠️  Not running in Kaggle - using standard settings")

# Try to import neptune, make it optional
try:
    import neptune
    from neptune.utils import stringify_unsupported
    NEPTUNE_AVAILABLE = True
except ImportError:
    NEPTUNE_AVAILABLE = False
    print("⚠️  Neptune not available - logging will be disabled")

# TPU support (optional)
USE_TPU = False
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    USE_TPU = True
    print("✅ TPU support detected")
except ImportError:
    pass

# Set base directory - in Kaggle, we work from /kaggle/working
if IN_KAGGLE:
    # Check if we're in the repository directory or need to find it
    repo_dir = '/kaggle/working/kaggle-asl-fingerspelling-1st-place-solution'
    if os.path.exists(repo_dir) and os.path.exists(os.path.join(repo_dir, 'configs')):
        BASEDIR = repo_dir
        os.chdir(BASEDIR)
        print(f"✅ Changed to repository directory: {BASEDIR}")
    elif os.path.exists('configs'):
        # Already in the repo directory
        BASEDIR = os.getcwd()
        print(f"✅ Already in repository directory: {BASEDIR}")
    else:
        BASEDIR = '/kaggle/working'
        os.chdir(BASEDIR)
        print(f"⚠️  Using base directory: {BASEDIR}")
        print(f"   Make sure configs exist at: {BASEDIR}/configs/")
else:
    BASEDIR = './'

for DIRNAME in 'configs data models postprocess metrics'.split():
    dir_path = f'{BASEDIR}/{DIRNAME}/'
    if os.path.exists(dir_path):
        sys.path.append(dir_path)
        print(f"✅ Added to path: {dir_path}")
    else:
        print(f"⚠️  Directory not found: {dir_path}")

parser = argparse.ArgumentParser(description="Kaggle Training Script")

parser.add_argument("-C", "--config", help="config filename", default="cfg_1")
parser.add_argument("-G", "--gpu_id", default="", help="GPU ID")
parser.add_argument("--fold", type=int, default=None, help="Fold number (overrides config)")
parser.add_argument("--use_tpu", action="store_true", help="Use TPU instead of GPU")
parser.add_argument("--disable_neptune", action="store_true", help="Disable Neptune logging")
parser_args, other_args = parser.parse_known_args(sys.argv)

# Load config
try:
    cfg = copy(importlib.import_module(parser_args.config).cfg)
except Exception as e:
    print(f"❌ Error loading config: {e}")
    print(f"   Make sure {parser_args.config}.py exists in configs/ directory")
    sys.exit(1)

if parser_args.gpu_id != "":
    os.environ['CUDA_VISIBLE_DEVICES'] = str(parser_args.gpu_id)

# Overwrite fold if specified
if parser_args.fold is not None:
    print(f'overwriting cfg.fold: {cfg.fold} -> {parser_args.fold}')
    cfg.fold = parser_args.fold

# Overwrite params in config with additional args
if len(other_args) > 1:
    other_args = {k.replace('-',''):v for k, v in zip(other_args[1::2], other_args[2::2])}
    for key in other_args:
        if key in cfg.__dict__:
            print(f'overwriting cfg.{key}: {cfg.__dict__[key]} -> {other_args[key]}')
            cfg_type = type(cfg.__dict__[key])
            if cfg_type == bool:
                cfg.__dict__[key] = other_args[key] == 'True'
            elif cfg_type == type(None):
                cfg.__dict__[key] = other_args[key]
            else:
                cfg.__dict__[key] = cfg_type(other_args[key])

# Kaggle-specific adjustments
if IN_KAGGLE:
    # Reduce num_workers for Kaggle (Kaggle doesn't handle high num_workers well)
    if cfg.num_workers > 2:
        print(f"⚠️  Reducing num_workers from {cfg.num_workers} to 2 for Kaggle compatibility")
        cfg.num_workers = 2
    # Disable pin_memory in Kaggle if it causes issues
    if cfg.pin_memory:
        print("⚠️  Setting pin_memory to False for Kaggle compatibility")
        cfg.pin_memory = False
    
    # Check if data is in input directory and create symlink if needed
    input_dir = '/kaggle/input'
    if os.path.exists(input_dir):
        # Look for the dataset in input directory
        for dataset_name in os.listdir(input_dir):
            dataset_path = os.path.join(input_dir, dataset_name)
            if os.path.isdir(dataset_path):
                # Check if this looks like our dataset
                landmarks_path = os.path.join(dataset_path, 'train_landmarks_npy')
                if os.path.exists(landmarks_path):
                    # Create symlink in working directory
                    target_path = os.path.join(BASEDIR, 'datamount', 'train_landmarks_npy')
                    if not os.path.exists(target_path):
                        os.makedirs(os.path.dirname(target_path), exist_ok=True)
                        os.symlink(landmarks_path, target_path)
                        print(f"✅ Created symlink: {target_path} -> {landmarks_path}")
                    
                    # Also check for CSV files
                    for csv_file in ['train_folded.csv', 'character_to_prediction_index.json', 'symmetry.csv']:
                        src = os.path.join(dataset_path, csv_file)
                        dst = os.path.join(BASEDIR, 'datamount', csv_file)
                        if os.path.exists(src) and not os.path.exists(dst):
                            os.makedirs(os.path.dirname(dst), exist_ok=True)
                            os.symlink(src, dst)
                            print(f"✅ Created symlink: {dst} -> {src}")

if cfg.seed < 0:
    cfg.seed = np.random.randint(1_000_000)
print("seed", cfg.seed)
set_seed(cfg.seed)

# Import experiment modules
try:
    post_process_pipeline = importlib.import_module(cfg.post_process_pipeline).post_process_pipeline
    calc_metric = importlib.import_module(cfg.metric).calc_metric
    Net = importlib.import_module(cfg.model).Net
    CustomDataset = importlib.import_module(cfg.dataset).CustomDataset
    tr_collate_fn = importlib.import_module(cfg.dataset).tr_collate_fn
    val_collate_fn = importlib.import_module(cfg.dataset).val_collate_fn
    batch_to_device = importlib.import_module(cfg.dataset).batch_to_device
except ZeroDivisionError as e:
    print(f"❌ Division by zero error when importing {cfg.dataset}: {e}")
    print("   This might be due to empty or invalid data. Check:")
    print(f"   1. inference_args.json exists and is valid")
    print(f"   2. train_folded_oof_supp.csv exists and has data")
    print(f"   3. Data files are accessible")
    import traceback
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    print(f"❌ Error importing modules: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Initialize Neptune (optional)
neptune_run = None
if NEPTUNE_AVAILABLE and not parser_args.disable_neptune:
    try:
        fns = [parser_args.config] + [getattr(cfg, s) for s in 'dataset model metric post_process_pipeline'.split()]
        fns = sum([glob.glob(f"{BASEDIR}/*/{fn}.py") for fn in fns], [])
        
        if cfg.neptune_project == "common/quickstarts":
            neptune_api_token = neptune.ANONYMOUS_API_TOKEN
        else:
            neptune_api_token = os.environ.get('NEPTUNE_API_TOKEN', neptune.ANONYMOUS_API_TOKEN)
        
        neptune_run = neptune.init_run(
            project=cfg.neptune_project,
            tags="kaggle",
            mode="async",
            api_token=neptune_api_token,
            capture_stdout=False,
            capture_stderr=False,
            source_files=fns
        )
        print(f"✅ Neptune initialized")
        print(f"   Neptune system id : {neptune_run._sys_id}")
        print(f"   Neptune URL       : {neptune_run.get_url()}")
        neptune_run["cfg"] = stringify_unsupported(cfg.__dict__)
    except Exception as e:
        print(f"⚠️  Could not initialize Neptune: {e}")
        print("   Continuing without Neptune logging")
        neptune_run = None
else:
    print("⚠️  Neptune logging disabled")

# Read training data
try:
    df = pd.read_csv(cfg.train_df)
    train_df = df[df["fold"] != cfg.fold].copy()
    if cfg.fold == -1:
        val_df = df[df["fold"] == 0].copy()
    else:
        val_df = df[df["fold"] == cfg.fold].copy()
    print(f"✅ Loaded data: {len(train_df)} train, {len(val_df)} val samples")
except Exception as e:
    print(f"❌ Error loading data: {e}")
    print(f"   Make sure {cfg.train_df} exists")
    sys.exit(1)

# Set device (GPU, TPU, or CPU)
if parser_args.use_tpu and USE_TPU:
    cfg.device = xm.xla_device()
    print(f"✅ Using TPU: {cfg.device}")
elif torch.cuda.is_available():
    cfg.device = 'cuda'
    print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
else:
    cfg.device = 'cpu'
    print("⚠️  Using CPU (GPU/TPU not available)")

# Set up dataset and dataloader
train_dataset = CustomDataset(train_df, cfg, aug=cfg.train_aug, mode="train")
val_dataset = CustomDataset(val_df, cfg, aug=cfg.train_aug, mode="val")

train_dataloader = DataLoader(
    train_dataset,
    shuffle=True,
    batch_size=cfg.batch_size,
    num_workers=cfg.num_workers,
    pin_memory=cfg.pin_memory,
    collate_fn=tr_collate_fn,
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=cfg.batch_size,
    num_workers=cfg.num_workers,
    pin_memory=cfg.pin_memory,
    collate_fn=val_collate_fn,
)

# Set up model
model = Net(cfg).to(cfg.device)
print(f"✅ Model initialized on {cfg.device}")

total_steps = len(train_dataset)
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
scheduler = transformers.get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=cfg.warmup * (total_steps // cfg.batch_size),
    num_training_steps=cfg.epochs * (total_steps // cfg.batch_size),
    num_cycles=0.5
)
scaler = GradScaler()

# Create output directory
if not os.path.exists(f"{cfg.output_dir}/fold{cfg.fold}/"):
    os.makedirs(f"{cfg.output_dir}/fold{cfg.fold}/")
    print(f"✅ Created output directory: {cfg.output_dir}/fold{cfg.fold}/")

# Training loop
cfg.curr_step = 0
optimizer.zero_grad()
total_grad_norm = None
total_grad_norm_after_clip = None
i = 0

print("\n" + "="*60)
print("STARTING TRAINING")
print("="*60)
print(f"Epochs: {cfg.epochs}")
print(f"Batch size: {cfg.batch_size}")
print(f"Learning rate: {cfg.lr}")
print(f"Device: {cfg.device}")
print("="*60 + "\n")

for epoch in range(cfg.epochs):
    cfg.curr_epoch = epoch
    progress_bar = tqdm(range(len(train_dataloader)), desc=f'Train epoch {epoch}')
    tr_it = iter(train_dataloader)
    losses = []
    gc.collect()

    model.train()
    for itr in progress_bar:
        i += 1
        cfg.curr_step += cfg.batch_size
        data = next(tr_it)
        torch.set_grad_enabled(True)
        batch = batch_to_device(data, cfg.device)
        
        if cfg.mixed_precision:
            with autocast():
                output_dict = model(batch)
        else:
            output_dict = model(batch)
        
        loss = output_dict["loss"]
        losses.append(loss.item())

        if cfg.grad_accumulation > 1:
            loss /= cfg.grad_accumulation

        if cfg.mixed_precision:
            scaler.scale(loss).backward()
            if i % cfg.grad_accumulation == 0:
                if (cfg.track_grad_norm) or (cfg.clip_grad > 0):
                    scaler.unscale_(optimizer)
                if cfg.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            loss.backward()
            if i % cfg.grad_accumulation == 0:
                if cfg.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
                optimizer.step()
                optimizer.zero_grad()

        if scheduler is not None:
            scheduler.step()

        # Log to Neptune (if available)
        if neptune_run:
            try:
                loss_names = [key for key in output_dict if 'loss' in key]
                for l in loss_names:
                    neptune_run[f"train/{l}"].log(value=output_dict[l].item(), step=cfg.curr_step)
                neptune_run["lr"].log(
                    value=optimizer.param_groups[0]["lr"], step=cfg.curr_step
                )
                if total_grad_norm is not None:
                    neptune_run["total_grad_norm"].log(value=total_grad_norm.item(), step=cfg.curr_step)
                    neptune_run["total_grad_norm_after_clip"].log(value=total_grad_norm_after_clip.item(), step=cfg.curr_step)
            except Exception as e:
                pass  # Silently fail if Neptune logging fails

    # Validation
    if (epoch + 1) % cfg.eval_epochs == 0 or (epoch + 1) == cfg.epochs:
        model.eval()
        torch.set_grad_enabled(False)
        val_data = defaultdict(list)
        val_score = 0
        
        for ind_, data in enumerate(tqdm(val_dataloader, desc=f'Val epoch {epoch}')):
            batch = batch_to_device(data, cfg.device)
            if cfg.mixed_precision:
                with autocast():
                    output = model(batch)
            else:
                output = model(batch)
            for key, val in output.items():
                val_data[key] += [output[key]]
        
        for key, val in output.items():
            value = val_data[key]
            if isinstance(value[0], list):
                val_data[key] = [item for sublist in value for item in sublist]
            else:
                if len(value[0].shape) == 0:
                    val_data[key] = torch.stack(value)
                else:
                    val_data[key] = torch.cat(value, dim=0)

        if cfg.save_val_data:
            torch.save(val_data, f"{cfg.output_dir}/fold{cfg.fold}/val_data_seed{cfg.seed}.pth")

        loss_names = [key for key in output if 'loss' in key]
        loss_names += [key for key in output if 'score' in key]

        val_df = val_dataloader.dataset.df
        pp_out = post_process_pipeline(cfg, val_data, val_df)
        val_score = calc_metric(cfg, pp_out, val_df, "val")
        
        if type(val_score) != dict:
            val_score = {f'score': val_score}

        for k, v in val_score.items():
            print(f"val_{k}: {v:.3f}")
            if neptune_run:
                try:
                    neptune_run[f"val/{k}"].log(v, step=cfg.curr_step)
                except:
                    pass

    # Save checkpoint
    if not cfg.save_only_last_ckpt:
        torch.save({"model": model.state_dict()}, 
                  f"{cfg.output_dir}/fold{cfg.fold}/checkpoint_last_seed{cfg.seed}.pth")

# Final save
torch.save({"model": model.state_dict()}, 
          f"{cfg.output_dir}/fold{cfg.fold}/checkpoint_last_seed{cfg.seed}.pth")
print(f"\n✅ Checkpoint saved: {cfg.output_dir}/fold{cfg.fold}/checkpoint_last_seed{cfg.seed}.pth")

# Close Neptune run
if neptune_run:
    try:
        neptune_run.stop()
        print("✅ Neptune run stopped")
    except:
        pass

print("\n" + "="*60)
print("TRAINING COMPLETE")
print("="*60)

