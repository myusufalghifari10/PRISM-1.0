"""
config.py — All hyperparameters and paths for PRISM.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PRISMConfig:
    # ──── Paths ────
    data_path: str = "../Data/trajectories_shifted.pt"
    save_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    
    # ──── Model Architecture ────
    state_dim: int = 47           # Clinical features (colbin 4 + colnorm 32 + collog 11)
    act_dim: int = 25             # Total joint actions (5×5)
    vaso_dim: int = 5             # Vasopressor sub-actions
    iv_dim: int = 5               # IV fluid sub-actions
    hidden_size: int = 128        # GPT-2 embedding dimension
    n_layer: int = 3              # Transformer layers
    n_head: int = 4               # Attention heads
    dropout: float = 0.1          # Dropout probability
    activation: str = "gelu"      # Activation function
    
    # ──── Sequence ────
    context_len: int = 20         # K: max context length (past steps)
    max_ep_len: int = 50          # Maximum trajectory length (padding)
    
    # ──── Training ────
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 1000
    max_epochs: int = 50
    grad_clip: float = 1.0
    
    # ──── Loss weights ────
    lambda_return: float = 0.1    # Weight for auxiliary return prediction
    lambda_state: float = 0.01    # Weight for auxiliary state prediction
    
    # ──── Data split ────
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # ──── Evaluation ────
    eval_discount: float = 0.99
    eval_eps: float = 0.01        # Softening for OPE
    
    # ──── Device ────
    device: str = "cuda" if __import__('torch').cuda.is_available() else "cpu"
    
    # ──── Logging ────
    use_wandb: bool = False
    log_every: int = 100          # Log every N training steps
    eval_every: int = 500         # Evaluate every N training steps
    save_every: int = 1000        # Save checkpoint every N steps
    
    # ──── Reward remapping (for WIS evaluation) ────
    R_survival: float = 100.0     # Reward for survival / discharge
    R_death: float = 0.0          # Reward for death
    R_immediate: float = 0.0      # Reward for intermediate steps


# Default config
default_config = PRISMConfig()
