"""
train.py — Training script for PRISM Decision Transformer.

Usage:
    python train.py [--config overrides...]
"""

import os
import sys
import time
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from config import PRISMConfig, default_config
from prism_dt import PRISMDecisionTransformer
from data_loader import create_dataloaders


def parse_args():
    parser = argparse.ArgumentParser(description='Train PRISM Decision Transformer')
    parser.add_argument('--data-path', type=str, default=default_config.data_path)
    parser.add_argument('--save-dir', type=str, default=default_config.save_dir)
    parser.add_argument('--log-dir', type=str, default=default_config.log_dir)
    parser.add_argument('--context-len', type=int, default=default_config.context_len)
    parser.add_argument('--hidden-size', type=int, default=default_config.hidden_size)
    parser.add_argument('--n-layer', type=int, default=default_config.n_layer)
    parser.add_argument('--n-head', type=int, default=default_config.n_head)
    parser.add_argument('--dropout', type=float, default=default_config.dropout)
    parser.add_argument('--batch-size', type=int, default=default_config.batch_size)
    parser.add_argument('--lr', type=float, default=default_config.learning_rate)
    parser.add_argument('--weight-decay', type=float, default=default_config.weight_decay)
    parser.add_argument('--warmup-steps', type=int, default=default_config.warmup_steps)
    parser.add_argument('--max-epochs', type=int, default=default_config.max_epochs)
    parser.add_argument('--grad-clip', type=float, default=default_config.grad_clip)
    parser.add_argument('--device', type=str, default=default_config.device)
    return parser.parse_args()


def train(config: PRISMConfig):
    """Main training loop."""
    
    # ──── Setup ────
    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    
    device = torch.device(config.device)
    writer = SummaryWriter(log_dir=config.log_dir)
    
    print("=" * 60)
    print("PRISM Decision Transformer — Training")
    print("=" * 60)
    print(f"  Device: {device}")
    print(f"  Data: {config.data_path}")
    print(f"  Context length: {config.context_len}")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Layers: {config.n_layer}, Heads: {config.n_head}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print("=" * 60)
    
    # ──── Data ────
    train_loader, val_loader, test_loader = create_dataloaders(config)
    
    # ──── Model ────
    model = PRISMDecisionTransformer(
        state_dim=config.state_dim,
        act_dim=config.act_dim,
        vaso_dim=config.vaso_dim,
        iv_dim=config.iv_dim,
        hidden_size=config.hidden_size,
        max_length=config.context_len,
        max_ep_len=config.max_ep_len,
        n_layer=config.n_layer,
        n_head=config.n_head,
        activation_function=config.activation,
        dropout=config.dropout,
        lambda_return=config.lambda_return,
        lambda_state=config.lambda_state,
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # ──── Optimizer & Scheduler ────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps + 1) / config.warmup_steps, 1.0)
    )
    
    # ──── Training Loop ────
    global_step = 0
    best_val_loss = float('inf')  # NOTE: Model selection via val loss (heuristic).
    # True model selection uses WIS + ESS >= 200 in Phase 3 (evaluate.py).
    # Val loss here is a proxy for early stopping + checkpoint saving only.
    
    for epoch in range(config.max_epochs):
        model.train()
        epoch_losses = []
        
        t0 = time.time()
        for batch_idx, batch in enumerate(train_loader):
            # Move to device
            batch = [b.to(device) for b in batch]
            
            # Forward + loss
            loss, loss_dict = model.compute_loss(batch)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            scheduler.step()
            
            epoch_losses.append(loss_dict['action'])
            global_step += 1
            
            # Logging
            if global_step % config.log_every == 0:
                avg_loss = np.mean(epoch_losses[-config.log_every:])
                writer.add_scalar('train/action_loss', loss_dict['action'], global_step)
                writer.add_scalar('train/vaso_loss', loss_dict['vaso'], global_step)
                writer.add_scalar('train/iv_loss', loss_dict['iv'], global_step)
                writer.add_scalar('train/return_loss', loss_dict['return'], global_step)
                writer.add_scalar('train/state_loss', loss_dict['state'], global_step)
                writer.add_scalar('train/total_loss', loss_dict['total'], global_step)
                writer.add_scalar('train/lr', scheduler.get_last_lr()[0], global_step)
                
                print(f"  [Epoch {epoch+1}/{config.max_epochs}] "
                      f"Step {global_step} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"LR: {scheduler.get_last_lr()[0]:.2e}")
            
            # Validation
            if global_step % config.eval_every == 0:
                val_loss = validate(model, val_loader, device, config)
                writer.add_scalar('val/action_loss', val_loss, global_step)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(model, optimizer, epoch, global_step, 
                                   config, is_best=True, best_val_loss=best_val_loss)
                    print(f"  → New best model! Val loss: {val_loss:.4f}")
            
            # Checkpoint
            if global_step % config.save_every == 0:
                save_checkpoint(model, optimizer, epoch, global_step, config,
                               best_val_loss=best_val_loss)
        
        # End of epoch
        epoch_time = time.time() - t0
        avg_epoch_loss = np.mean(epoch_losses)
        print(f"\n  Epoch {epoch+1} complete ({epoch_time:.1f}s) | "
              f"Avg loss: {avg_epoch_loss:.4f}\n")
    
    # ──── Final evaluation ────
    print("=" * 60)
    print("Training complete. Running final evaluation...")
    print("=" * 60)
    
    test_loss = validate(model, test_loader, device, config)
    print(f"  Test action loss: {test_loss:.4f}")
    
    save_checkpoint(model, optimizer, config.max_epochs, global_step, config, 
                   is_final=True, best_val_loss=best_val_loss)
    
    writer.close()
    return model


def validate(model, dataloader, device, config):
    """Run validation and return average action loss."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            batch = [b.to(device) for b in batch]
            _, loss_dict = model.compute_loss(batch)
            total_loss += loss_dict['action']
            n_batches += 1
    
    model.train()
    return total_loss / max(n_batches, 1)


def save_checkpoint(model, optimizer, epoch, global_step, config,
                   is_best=False, is_final=False, best_val_loss=None):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'best_val_loss': best_val_loss,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
    }
    
    if is_best:
        path = os.path.join(config.save_dir, 'best_model.pt')
    elif is_final:
        path = os.path.join(config.save_dir, 'final_model.pt')
    else:
        path = os.path.join(config.save_dir, f'checkpoint_step{global_step}.pt')
    
    torch.save(checkpoint, path)
    print(f"  → Saved checkpoint: {path}")


if __name__ == '__main__':
    args = parse_args()
    config = PRISMConfig(**vars(args))
    train(config)
