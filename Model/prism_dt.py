"""
prism_dt.py — PRISM Decision Transformer with Factored Action Output Heads.

This is the core model file. It combines:
1. Decision Transformer architecture (Chen et al., NeurIPS 2021)
2. Factored output heads: separate vaso/IV classifiers (inspired by Tang et al., NeurIPS 2022)
3. Designed for shifted temporal alignment (Tang et al., npj Digital Med 2026)

Architecture:
    Input:  (R₁, s₁, a₁, R₂, s₂, a₂, ..., R_T, s_T, a_T)
    Embed:  Each modality → Linear(hidden)
    Backbone: GPT-2 causal transformer
    Output: P_vaso(a_vaso | context), P_iv(a_iv | context)
            + auxiliary: R_pred, s_pred
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

from trajectory_gpt2 import GPT2Model


class TrajectoryModel(nn.Module):
    """Abstract base class for trajectory-level models."""
    def __init__(self, state_dim, act_dim, max_length=None):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.max_length = max_length
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError
    
    def get_action(self, *args, **kwargs):
        raise NotImplementedError


class PRISMDecisionTransformer(TrajectoryModel):
    """
    PRISM: DT with Factored Output Heads.
    
    Key difference from standard DT:
    - Standard: predict_action = Linear(hidden, 25)  → flat 25-way
    - PRISM:    predict_vaso  = Linear(hidden, 5)    }
                predict_iv    = Linear(hidden, 5)    } factored!
    """
    
    def __init__(
        self,
        state_dim=47,
        act_dim=25,
        vaso_dim=5,
        iv_dim=5,
        hidden_size=128,
        max_length=20,
        max_ep_len=50,
        n_layer=3,
        n_head=4,
        activation_function='gelu',
        dropout=0.1,
        lambda_return=0.1,
        lambda_state=0.01,
    ):
        super().__init__(state_dim, act_dim, max_length=max_length)
        
        self.hidden_size = hidden_size
        self.vaso_dim = vaso_dim
        self.iv_dim = iv_dim
        self.lambda_return = lambda_return
        self.lambda_state = lambda_state
        
        # GPT-2 backbone
        config = transformers.GPT2Config(
            vocab_size=1,         # Not used (we provide input_embeds)
            n_embd=hidden_size,
            n_layer=n_layer,
            n_head=n_head,
            activation_function=activation_function,
            resid_pdrop=dropout,
            embd_pdrop=dropout,
            attn_pdrop=dropout,
        )
        self.transformer = GPT2Model(config)
        
        # ──── Embedding layers ────
        # Action embedding uses 10-dim factored one-hot (vaso 5 + iv 5)
        # NOT 25-dim flat one-hot. This is key to the factored approach.
        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return   = nn.Linear(1, hidden_size)
        self.embed_state    = nn.Linear(state_dim, hidden_size)
        self.embed_action   = nn.Linear(vaso_dim + iv_dim, hidden_size)  # 10-dim
        
        self.embed_ln = nn.LayerNorm(hidden_size)
        
        # ──── Prediction heads ────
        # Factored action heads (instead of one 25-way head)
        self.predict_vaso   = nn.Linear(hidden_size, vaso_dim)   # 5-way
        self.predict_iv     = nn.Linear(hidden_size, iv_dim)     # 5-way
        
        # Auxiliary prediction heads
        self.predict_return = nn.Linear(hidden_size, 1)
        self.predict_state  = nn.Linear(hidden_size, state_dim)
        
        # Initialize
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
    
    def forward(
        self,
        states,           # [B, T, state_dim]
        actions,          # [B, T, vaso_dim+iv_dim]   ← factored one-hot (10-dim!)
        rewards,          # [B, T, 1] (unused, kept for API compatibility)
        returns_to_go,    # [B, T, 1]
        timesteps,        # [B, T]
        attention_mask=None,
    ):
        batch_size, seq_length = states.shape[0], states.shape[1]
        
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), 
                                        dtype=torch.long, device=states.device)
        
        # Embed each modality
        state_embeddings   = self.embed_state(states)
        action_embeddings  = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings    = self.embed_timestep(timesteps)
        
        # Add time embeddings (acts as positional encoding)
        state_embeddings   = state_embeddings + time_embeddings
        action_embeddings  = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings
        
        # Stack: (R₁, s₁, a₁, R₂, s₂, a₂, ...)
        # shape: [B, 3*T, hidden]
        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3*seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)
        
        # Stack attention mask to match 3× token count
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 3*seq_length)
        
        # Forward through GPT-2
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']
        
        # Reshape: [B, T, 3, hidden] → group by modality
        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)
        # x[:, 0, :, :] = return tokens, x[:, 1, :, :] = state tokens, x[:, 2, :, :] = action tokens
        
        # ──── Factored predictions ────
        # Predict action given state token (position 1)
        vaso_logits = self.predict_vaso(x[:, 1])    # [B, T, vaso_dim]
        iv_logits   = self.predict_iv(x[:, 1])      # [B, T, iv_dim]
        
        # Auxiliary predictions (from action token, position 2)
        return_preds = self.predict_return(x[:, 2])  # [B, T, 1]
        state_preds  = self.predict_state(x[:, 2])   # [B, T, state_dim]
        
        return vaso_logits, iv_logits, return_preds, state_preds
    
    def get_action(self, states, actions, rewards, returns_to_go, timesteps, **kwargs):
        """
        Generate next action given context. Used at inference time.
        
        Returns:
            vaso_idx: int (0–4)
            iv_idx: int (0–4)
        """
        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.vaso_dim + self.iv_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)
        
        if self.max_length is not None:
            # Truncate to max context length
            states = states[:, -self.max_length:]
            actions = actions[:, -self.max_length:]
            returns_to_go = returns_to_go[:, -self.max_length:]
            timesteps = timesteps[:, -self.max_length:]
            
            # Pad
            cur_len = states.shape[1]
            pad_len = self.max_length - cur_len
            attention_mask = torch.cat([
                torch.zeros(pad_len), 
                torch.ones(cur_len)
            ]).to(dtype=torch.long, device=states.device).reshape(1, -1)
            
            states = torch.cat([
                torch.zeros(1, pad_len, self.state_dim, device=states.device),
                states
            ], dim=1).to(dtype=torch.float32)
            actions = torch.cat([
                torch.zeros(1, pad_len, self.vaso_dim + self.iv_dim, device=actions.device),
                actions
            ], dim=1).to(dtype=torch.float32)
            returns_to_go = torch.cat([
                torch.zeros(1, pad_len, 1, device=returns_to_go.device),
                returns_to_go
            ], dim=1).to(dtype=torch.float32)
            timesteps = torch.cat([
                torch.zeros(1, pad_len, device=timesteps.device),
                timesteps
            ], dim=1).to(dtype=torch.long)
        else:
            attention_mask = None
        
        vaso_logits, iv_logits, _, _ = self.forward(
            states, actions, None, returns_to_go, timesteps,
            attention_mask=attention_mask, **kwargs
        )
        
        # Last timestep prediction
        vaso_probs = F.softmax(vaso_logits[0, -1], dim=-1)
        iv_probs   = F.softmax(iv_logits[0, -1], dim=-1)
        
        vaso_idx = torch.argmax(vaso_probs).item()
        iv_idx   = torch.argmax(iv_probs).item()
        
        return vaso_idx, iv_idx
    
    @staticmethod
    def action_to_factored_onehot(vaso_idx, iv_idx):
        """Convert (vaso_idx, iv_idx) to 10-dim factored one-hot vector.
        
        This MUST be called before feeding get_action() output back into
        forward(). The forward() expects actions as [B, T, 10] one-hot.
        
        Args:
            vaso_idx: int or tensor of vasopressor level (0-4)
            iv_idx: int or tensor of IV fluid level (0-4)
        Returns:
            10-dim tensor: first 5 = vaso onehot, last 5 = iv onehot
        """
        vec = torch.zeros(10)
        vec[vaso_idx] = 1.0
        vec[5 + iv_idx] = 1.0
        return vec
    
    def compute_loss(self, batch):
        """
        Compute training loss from a batch of trajectory data.
        
        batch: tuple from PRISMDataLoader
            (states, actions_factored, rtg, timesteps, masks,
             vaso_targets, iv_targets, return_targets, state_targets)
        
        Returns:
            total_loss, loss_dict
        """
        states, actions_factored, rtg, timesteps, masks, \
            vaso_targets, iv_targets, return_targets, state_targets = batch
        
        # Forward pass
        vaso_logits, iv_logits, return_preds, state_preds = self.forward(
            states, actions_factored, None, rtg, timesteps,
            attention_mask=masks
        )
        
        # ──── Factored action loss ────
        # vaso_logits: [B, T, 5], vaso_targets: [B, T]
        vaso_loss = F.cross_entropy(
            vaso_logits.reshape(-1, self.vaso_dim),
            vaso_targets.reshape(-1),
            reduction='none'
        )
        vaso_loss = (vaso_loss * masks.reshape(-1)).sum() / masks.sum()
        
        # iv_logits: [B, T, 5], iv_targets: [B, T]
        iv_loss = F.cross_entropy(
            iv_logits.reshape(-1, self.iv_dim),
            iv_targets.reshape(-1),
            reduction='none'
        )
        iv_loss = (iv_loss * masks.reshape(-1)).sum() / masks.sum()
        
        action_loss = vaso_loss + iv_loss
        
        # ──── Auxiliary losses ────
        # Return prediction
        return_loss = F.mse_loss(
            return_preds[masks.bool()],
            return_targets[masks.bool()]
        )
        
        # State prediction
        state_loss = F.mse_loss(
            state_preds[masks.bool()],
            state_targets[masks.bool()]
        )
        
        # ──── Total loss ────
        total_loss = (
            action_loss 
            + self.lambda_return * return_loss 
            + self.lambda_state * state_loss
        )
        
        loss_dict = {
            'total': total_loss.item(),
            'action': action_loss.item(),
            'vaso': vaso_loss.item(),
            'iv': iv_loss.item(),
            'return': return_loss.item(),
            'state': state_loss.item(),
        }
        
        return total_loss, loss_dict
