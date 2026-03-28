"""
Stage2 Alignment Training Script

Trains paradigm-specific projection modules to align with the anchor space
from Stage1 using contrastive learning.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Add parent directories to path
SCRIPT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(SCRIPT_DIR / "src"))

from data.dataset import Stage2AlignmentDataset, create_dataloaders, collate_fn
from models.alignment import AlignmentModel, AlignmentLoss


def parse_args():
    parser = argparse.ArgumentParser(description="Stage2 Alignment Training")
    
    # Data
    parser.add_argument("--memory_outputs_dir", type=str, required=True,
                        help="Directory containing memory outputs from baselines")
    parser.add_argument("--evidence_embeddings_path", type=str, default=None,
                        help="Path to evidence embeddings from Stage1")
    parser.add_argument("--paradigms", type=str, nargs="+", 
                        default=["amem", "streaming"],
                        help="Paradigms to include in training")
    parser.add_argument("--model_size", type=str, default="1.5B",
                        help="Model size for paradigms (1.5B, 3B, 7B)")
    
    # Model
    parser.add_argument("--stage1_checkpoint", type=str, default=None,
                        help="Path to Stage1 checkpoint (directory or file)")
    parser.add_argument("--anchor_dim", type=int, default=1536,
                        help="Anchor space dimension")
    parser.add_argument("--freeze_anchor", action="store_true", default=True,
                        help="Freeze anchor align module")
    parser.add_argument("--unfreeze_anchor", action="store_true",
                        help="Unfreeze anchor align module (fine-tune)")
    
    # Training
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Training batch size")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Warmup ratio")
    
    # Loss
    parser.add_argument("--temperature", type=float, default=0.07,
                        help="Contrastive loss temperature")
    parser.add_argument("--use_mse", action="store_true",
                        help="Add MSE loss component")
    parser.add_argument("--mse_weight", type=float, default=0.1,
                        help="MSE loss weight")
    parser.add_argument("--use_consistency", action="store_true",
                        help="Add consistency loss between paradigms")
    parser.add_argument("--consistency_weight", type=float, default=0.1,
                        help="Consistency loss weight")
    
    # Output
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for checkpoints")
    parser.add_argument("--save_every", type=int, default=1,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--log_every", type=int, default=10,
                        help="Log every N steps")
    
    # Hardware
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    # Handle freeze_anchor logic
    if args.unfreeze_anchor:
        args.freeze_anchor = False
    
    return args


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_paradigm_dims(paradigms: list, model_size: str) -> dict:
    """Get paradigm dimensions based on model size."""
    dims = {}
    
    size_to_hidden = {
        "1.5B": 1536,
        "3B": 2048,
        "7B": 3584,
    }
    
    hidden_dim = size_to_hidden.get(model_size, 1536)
    
    for paradigm in paradigms:
        if paradigm == "amem":
            dims["amem"] = 384  # MiniLM, fixed
        elif paradigm == "streaming":
            key = f"streaming_{model_size.replace('.', '_')}"
            dims[key] = hidden_dim
        elif paradigm == "memoryllm":
            key = f"memoryllm_{model_size.replace('.', '_')}"
            dims[key] = hidden_dim
        else:
            # Default to hidden_dim
            dims[paradigm] = hidden_dim
    
    return dims


def train_epoch(
    model: AlignmentModel,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: AlignmentLoss,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int,
    log_every: int = 10,
) -> dict:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    num_batches = 0
    loss_accumulator = {}
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for step, batch in enumerate(pbar):
        # Move to device
        evidence = batch['evidence'].to(device)
        
        # Collect paradigm memories
        paradigm_memories = {}
        for key in batch.keys():
            if key.endswith('_implicit') and key != 'evidence':
                paradigm = key.replace('_implicit', '')
                paradigm_memories[paradigm] = batch[key].to(device)
        
        if not paradigm_memories:
            continue
        
        # Forward
        anchor, projected = model(evidence, paradigm_memories)
        
        # Loss
        loss, loss_dict = loss_fn(anchor, projected)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.get_trainable_params(), max_norm=1.0)
        optimizer.step()
        
        # Accumulate losses
        total_loss += loss.item()
        num_batches += 1
        
        for k, v in loss_dict.items():
            if k not in loss_accumulator:
                loss_accumulator[k] = 0.0
            loss_accumulator[k] += v.item()
        
        # Log
        if (step + 1) % log_every == 0:
            avg_loss = total_loss / num_batches
            pbar.set_postfix(loss=f"{avg_loss:.4f}")
    
    # Average losses
    metrics = {k: v / num_batches for k, v in loss_accumulator.items()}
    metrics['loss'] = total_loss / num_batches
    
    return metrics


@torch.no_grad()
def validate(
    model: AlignmentModel,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: AlignmentLoss,
    device: str,
) -> dict:
    """Validate model."""
    model.eval()
    
    total_loss = 0.0
    num_batches = 0
    loss_accumulator = {}
    
    for batch in tqdm(dataloader, desc="Validation"):
        evidence = batch['evidence'].to(device)
        
        paradigm_memories = {}
        for key in batch.keys():
            if key.endswith('_implicit') and key != 'evidence':
                paradigm = key.replace('_implicit', '')
                paradigm_memories[paradigm] = batch[key].to(device)
        
        if not paradigm_memories:
            continue
        
        anchor, projected = model(evidence, paradigm_memories)
        loss, loss_dict = loss_fn(anchor, projected)
        
        total_loss += loss.item()
        num_batches += 1
        
        for k, v in loss_dict.items():
            if k not in loss_accumulator:
                loss_accumulator[k] = 0.0
            loss_accumulator[k] += v.item()
    
    metrics = {k: v / max(num_batches, 1) for k, v in loss_accumulator.items()}
    metrics['loss'] = total_loss / max(num_batches, 1)
    
    return metrics


def main():
    args = parse_args()
    
    # Setup
    set_seed(args.seed)
    device = args.device if torch.cuda.is_available() else "cpu"
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"stage2_alignment_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save args
    with open(output_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    
    print("=" * 60)
    print("Stage2 Alignment Training")
    print("=" * 60)
    print(f"Memory outputs: {args.memory_outputs_dir}")
    print(f"Paradigms: {args.paradigms}")
    print(f"Model size: {args.model_size}")
    print(f"Stage1 checkpoint: {args.stage1_checkpoint}")
    print(f"Output: {output_dir}")
    print("=" * 60)
    
    # Get paradigm dimensions
    paradigm_dims = get_paradigm_dims(args.paradigms, args.model_size)
    print(f"Paradigm dimensions: {paradigm_dims}")
    
    # Create model
    model = AlignmentModel(
        anchor_dim=args.anchor_dim,
        paradigm_dims=paradigm_dims,
        freeze_anchor=args.freeze_anchor,
        stage1_checkpoint=args.stage1_checkpoint,
    )
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.get_trainable_params())
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        memory_outputs_dir=args.memory_outputs_dir,
        evidence_embeddings_path=args.evidence_embeddings_path,
        paradigms=args.paradigms,
        model_size=args.model_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    
    # Create loss function
    loss_fn = AlignmentLoss(
        temperature=args.temperature,
        use_mse=args.use_mse,
        mse_weight=args.mse_weight,
        use_consistency=args.use_consistency,
        consistency_weight=args.consistency_weight,
    )
    
    # Create optimizer
    optimizer = AdamW(
        model.get_trainable_params(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    # Create scheduler
    total_steps = len(train_loader) * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
    
    # Training loop
    best_val_loss = float('inf')
    history = []
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*60}")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, loss_fn, optimizer,
            device, epoch, args.log_every
        )
        scheduler.step()
        
        # Validate
        val_metrics = validate(model, val_loader, loss_fn, device)
        
        # Log
        print(f"\nTrain Loss: {train_metrics['loss']:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}")
        
        history.append({
            'epoch': epoch,
            'train': train_metrics,
            'val': val_metrics,
            'lr': optimizer.param_groups[0]['lr'],
        })
        
        # Save checkpoint
        if epoch % args.save_every == 0:
            ckpt_path = output_dir / f"checkpoint_epoch_{epoch}.pt"
            model.save(str(ckpt_path))
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_path = output_dir / "best_model.pt"
            model.save(str(best_path))
            print(f"Saved best model (val_loss={best_val_loss:.4f})")
    
    # Save final model
    final_path = output_dir / "final_model.pt"
    model.save(str(final_path))
    
    # Save history
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
