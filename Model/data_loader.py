"""
data_loader.py — Load shifted trajectory data for PRISM training.

Loads trajectories_shifted.pt and provides PyTorch DataLoader
that samples subsequences of length context_len for DT training.
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random


class PRISMDataset(Dataset):
    """
    Dataset for PRISM Decision Transformer training.
    
    Each __getitem__ returns a subsequence of length context_len
    randomly sampled from a trajectory.
    """
    
    def __init__(self, data_path=None, context_len=20, split='train',
                 train_ratio=0.7, val_ratio=0.15, device='cpu',
                 data_dict=None):
        """
        Args:
            data_path: path to trajectories_shifted.pt (mutually exclusive with data_dict)
            data_dict: pre-loaded data dict — used to share data across datasets
            context_len: K — number of past steps to include
            split: 'train', 'val', or 'test'
        """
        if data_dict is not None:
            data = data_dict
        elif data_path is not None:
            data = torch.load(data_path, map_location='cpu')
        else:
            raise ValueError("Either data_path or data_dict must be provided")
        
        self.statevecs      = data['statevecs']
        self.actions         = data['actions']
        self.subactionvecs   = data['subactionvecs']    # [N, T, 10] one-hot
        self.returns_to_go   = data['returns_to_go']
        self.rewards         = data['rewards']           # [N, T] raw per-step reward
        self.timesteps       = data['timesteps']
        self.lengths         = data['lengths']
        self.pibs            = data.get('pibs', None)    # [N, T, 25] π_b from k-NN
        self.subpibs         = data.get('subpibs', None) # [N, T, 10] factored π_b
        
        self.context_len = context_len
        self.device = device
        self.state_dim = self.statevecs.shape[-1]
        
        # ──── Train/val/test split ────
        N = len(self.statevecs)
        indices = np.random.RandomState(42).permutation(N)
        train_end = int(N * train_ratio)
        val_end = train_end + int(N * val_ratio)
        
        if split == 'train':
            self.indices = indices[:train_end]
        elif split == 'val':
            self.indices = indices[train_end:val_end]
        elif split == 'test':
            self.indices = indices[val_end:]
        else:
            raise ValueError(f"Unknown split: {split}")
        
        # Precompute trajectory lengths
        self.traj_lengths = [int(self.lengths[i].item()) for i in self.indices]
        
        print(f"PRISMDataset [{split}]: {len(self.indices)} trajectories, "
              f"{sum(self.traj_lengths)} timesteps")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        """
        Sample a random subsequence from trajectory idx.
        
        Returns:
            states           : [context_len, state_dim]
            actions_factored : [context_len, 10]      ← factored one-hot
            rtg              : [context_len, 1]
            timesteps        : [context_len]
            mask             : [context_len]           ← 1 for real, 0 for pad
            vaso_targets     : [context_len]           ← target vaso indices
            iv_targets       : [context_len]           ← target iv indices
            return_targets   : [context_len, 1]        ← next-step return
            state_targets    : [context_len, state_dim] ← next-step state
        """
        traj_idx = self.indices[idx]
        L = int(self.lengths[traj_idx].item())
        
        if L <= 2:
            L = 2
        
        # Sample start position: any valid position with at least 1 step remaining for target.
        # For L > context_len: samples a random window of length context_len within the trajectory.
        # For L <= context_len: samples a random suffix (padded to context_len).
        # This follows the DT paper convention (Chen et al., NeurIPS 2021) of random crop sampling.
        max_start = max(1, L - 1)  # Need at least 1 step for the target (vaso/iv at that step)
        si = random.randint(0, max_start - 1) if max_start > 1 else 0
        ei = min(si + self.context_len, L)
        
        # Extract sequence
        states    = self.statevecs[traj_idx, si:ei]           # [len, state_dim]
        actions   = self.subactionvecs[traj_idx, si:ei]       # [len, 10]
        rtg       = self.returns_to_go[traj_idx, si:ei].unsqueeze(-1)  # [len, 1]
        timesteps_seq = self.timesteps[traj_idx, si:ei]       # [len]
        
        # Targets: predict NEXT step's vaso, iv, return, state
        seq_len = ei - si
        vaso_targets  = self.actions[traj_idx, si:ei] % 5     # [len]
        iv_targets    = self.actions[traj_idx, si:ei] // 5    # [len]
        
        # Return target is the return-to-go at next step (or 0 if terminal)
        return_targets = torch.zeros(seq_len, 1)
        return_targets[:-1, 0] = self.returns_to_go[traj_idx, si+1:ei]
        # Last step return target = 0 (terminal)
        
        # State target is the state at next step
        state_targets = torch.zeros(seq_len, self.state_dim)
        state_targets[:-1] = self.statevecs[traj_idx, si+1:ei]
        # Last step state target = zeros (terminal, not used with mask)
        
        # Mask: 1 for real steps, 0 for padded
        mask = torch.ones(seq_len)
        
        # Pad to context_len if needed
        if seq_len < self.context_len:
            pad_len = self.context_len - seq_len
            
            states = torch.cat([
                torch.zeros(pad_len, self.state_dim), states
            ], dim=0)
            actions = torch.cat([
                torch.zeros(pad_len, 10), actions
            ], dim=0)
            rtg = torch.cat([
                torch.zeros(pad_len, 1), rtg
            ], dim=0)
            timesteps_seq = torch.cat([
                torch.zeros(pad_len, dtype=torch.long), timesteps_seq
            ], dim=0)
            mask = torch.cat([
                torch.zeros(pad_len), mask
            ], dim=0)
            vaso_targets = torch.cat([
                torch.zeros(pad_len, dtype=torch.long), vaso_targets
            ], dim=0)
            iv_targets = torch.cat([
                torch.zeros(pad_len, dtype=torch.long), iv_targets
            ], dim=0)
            return_targets = torch.cat([
                torch.zeros(pad_len, 1), return_targets
            ], dim=0)
            state_targets = torch.cat([
                torch.zeros(pad_len, self.state_dim), state_targets
            ], dim=0)
        
        return (
            states, actions, rtg, timesteps_seq, mask,
            vaso_targets, iv_targets, return_targets, state_targets
        )


def collate_fn(batch):
    """Stack batch items into tensors."""
    states, actions, rtg, timesteps_seq, masks, vaso_t, iv_t, ret_t, state_t = zip(*batch)
    return (
        torch.stack(states).float(),
        torch.stack(actions).float(),
        torch.stack(rtg).float(),
        torch.stack(timesteps_seq).long(),
        torch.stack(masks).float(),
        torch.stack(vaso_t).long(),
        torch.stack(iv_t).long(),
        torch.stack(ret_t).float(),
        torch.stack(state_t).float(),
    )


def create_dataloaders(config):
    """Create train, val, test dataloaders.
    
    Data is loaded ONCE and shared across datasets to avoid 3× memory.
    """
    data = torch.load(config.data_path, map_location='cpu')
    
    train_dataset = PRISMDataset(
        context_len=config.context_len,
        split='train',
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        data_dict=data,
    )
    val_dataset = PRISMDataset(
        context_len=config.context_len,
        split='val',
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        data_dict=data,
    )
    test_dataset = PRISMDataset(
        context_len=config.context_len,
        split='test',
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        data_dict=data,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )
    
    return train_loader, val_loader, test_loader
