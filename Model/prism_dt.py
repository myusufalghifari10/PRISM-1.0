"""
prism_dt.py — PRISM Decision Transformer with Autoregressive Sub-Action Heads.

This is the core model file. It combines:
1. Decision Transformer architecture (Chen et al., NeurIPS 2021)
2. Autoregressive sub-action heads via chain rule: P(a_v,a_i|τ) = P(a_v|τ)·P(a_i|τ,a_v)
   IV head conditioned on REALIZED vaso action via Embed(a_v) lookup table.
   NOT independence — chain rule is a probability identity, not a constraint.
3. Designed for shifted temporal alignment (Tang et al., npj Digital Med 2026)

Architecture:
    Input:  (R₁, s₁, a₁, R₂, s₂, a₂, ..., R_T, s_T, a_T)
    Embed:  Each modality → Linear(hidden)
    Backbone: GPT-2 causal transformer
    Output: P_vaso(a_vaso | context), P_iv(a_iv | context, a_v)
            + auxiliary: R_pred, s_pred

Modes:
    Training (teacher forcing): vaso_realized = a_v_observed (from clinician data)
    Inference (get_action):     vaso_realized = None → get vaso → argmax → call again
    WIS Evaluation:             vaso_realized = a_v_observed (same as training)
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
    PRISM: DT with Autoregressive Sub-Action Heads (Chain Rule).
    
    P(a_v, a_i | τ) = P(a_v | τ) · P(a_i | τ, a_v)
    
    The IV head receives Embed(a_v_realized) — a learnable lookup table
    that is NOT a function of the hidden state h. This makes the decomposition
    irreducible to a product of marginals (see VALIDITY.md §2.2 for proof).
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
        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return   = nn.Linear(1, hidden_size)
        self.embed_state    = nn.Linear(state_dim, hidden_size)
        self.embed_action   = nn.Linear(vaso_dim + iv_dim, hidden_size)  # 10-dim factored one-hot
        
        self.embed_ln = nn.LayerNorm(hidden_size)
        
        # ──── Chain Rule: Embed(a_v) for IV head conditioning ────
        # This is a FREE lookup table — NOT a function of h.
        # It makes P(a_i | τ, a_v) irreducible to P(a_i | τ) alone.
        self.embed_vaso_for_iv = nn.Embedding(vaso_dim, hidden_size)
        
        # ──── Prediction heads ────
        # Vaso head: P(a_v | τ) — standard
        self.predict_vaso = nn.Linear(hidden_size, vaso_dim)   # 5-way
        
        # IV head: P(a_i | τ, a_v) — conditioned on realized vaso
        # Input: h ⊕ Embed(a_v_realized) → 2 * hidden_size
        self.predict_iv = nn.Linear(hidden_size + hidden_size, iv_dim)  # 5-way
        
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
    
    def _transformer_forward(
        self,
        states,
        actions,
        returns_to_go,
        timesteps,
        attention_mask,
    ):
        """Run the GPT-2 backbone and return hidden states.
        
        Extracted to avoid code duplication between vaso and iv prediction steps
        in get_action() which share the same transformer output.
        
        Returns:
            h_state: [B, T, hidden_size] — hidden states at state token positions
            h_action: [B, T, hidden_size] — hidden states at action token positions
        """
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
        # x[:, 0] = return tokens, x[:, 1] = state tokens, x[:, 2] = action tokens
        
        return x[:, 1], x[:, 2]  # h_state, h_action
    
    def forward(
        self,
        states,           # [B, T, state_dim]
        actions,          # [B, T, vaso_dim+iv_dim]   ← factored one-hot (10-dim!)
        rewards,          # [B, T, 1] (unused, kept for API compatibility)
        returns_to_go,    # [B, T, 1]
        timesteps,        # [B, T]
        attention_mask=None,
        vaso_realized=None,  # [B, T] — realized vaso indices for chain rule conditioning
    ):
        """
        Forward pass with chain rule decomposition.
        
        Args:
            vaso_realized: If None, iv_logits predicted from h alone (for vaso inference).
                          If Tensor [B,T] of vaso indices, iv_logits conditioned on realized a_v
                          via Embed lookup (for training and WIS evaluation).
        
        Returns:
            vaso_logits:    [B, T, vaso_dim]
            iv_logits:      [B, T, iv_dim]
            return_preds:   [B, T, 1]
            state_preds:    [B, T, state_dim]
        """
        h_state, h_action = self._transformer_forward(
            states, actions, returns_to_go, timesteps, attention_mask
        )
        
        # Vaso head: P(a_v | τ) — always from state hidden state only
        vaso_logits = self.predict_vaso(h_state)       # [B, T, vaso_dim]
        
        # IV head: P(a_i | τ, a_v) — conditioned on realized vaso if provided
        if vaso_realized is not None:
            # Chain rule: IV head sees h_state ⊕ Embed(a_v_realized)
            vaso_emb = self.embed_vaso_for_iv(vaso_realized)  # [B, T, hidden_size]
            iv_input = torch.cat([h_state, vaso_emb], dim=-1)  # [B, T, 2*hidden_size]
        else:
            # Vaso inference mode: IV head gets zero embedding for vaso
            # (placeholder — caller should re-run with realized vaso after argmax)
            vaso_emb = torch.zeros_like(h_state)
            iv_input = torch.cat([h_state, vaso_emb], dim=-1)
        
        iv_logits = self.predict_iv(iv_input)          # [B, T, iv_dim]
        
        # Auxiliary predictions (from action token, position 2)
        return_preds = self.predict_return(h_action)   # [B, T, 1]
        state_preds  = self.predict_state(h_action)    # [B, T, state_dim]
        
        return vaso_logits, iv_logits, return_preds, state_preds
    
    def get_action(self, states, actions, rewards, returns_to_go, timesteps, **kwargs):
        """
        Generate next action via sequential chain rule.
        
        1. Predict vaso from context
        2. Condition IV head on predicted vaso argmax
        3. Predict iv
        
        Returns:
            vaso_idx: int (0–4)
            iv_idx: int (0–4)
        """
        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.vaso_dim + self.iv_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)
        
        if self.max_length is not None:
            cur_len = states.shape[1]
            if cur_len > self.max_length:
                states = states[:, -self.max_length:]
                actions = actions[:, -self.max_length:]
                returns_to_go = returns_to_go[:, -self.max_length:]
                timesteps = timesteps[:, -self.max_length:]
                cur_len = self.max_length
            
            pad_len = self.max_length - cur_len
            if pad_len > 0:
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
        else:
            attention_mask = None
        
        # ──── Step 1: Predict vaso ────
        vaso_logits, _, _, _ = self.forward(
            states, actions, None, returns_to_go, timesteps,
            attention_mask=attention_mask, vaso_realized=None, **kwargs
        )
        vaso_probs = F.softmax(vaso_logits[0, -1], dim=-1)
        vaso_idx = torch.argmax(vaso_probs).item()
        
        # ──── Step 2: Predict iv conditioned on predicted vaso ────
        vaso_tensor = torch.tensor([[vaso_idx]], device=states.device, dtype=torch.long)
        _, iv_logits, _, _ = self.forward(
            states, actions, None, returns_to_go, timesteps,
            attention_mask=attention_mask, vaso_realized=vaso_tensor, **kwargs
        )
        iv_probs = F.softmax(iv_logits[0, -1], dim=-1)
        iv_idx = torch.argmax(iv_probs).item()
        
        return vaso_idx, iv_idx
    
    @staticmethod
    def action_to_factored_onehot(vaso_idx, iv_idx):
        """Convert (vaso_idx, iv_idx) to 10-dim factored one-hot vector."""
        vec = torch.zeros(10)
        vec[vaso_idx] = 1.0
        vec[5 + iv_idx] = 1.0
        return vec
    
    def compute_loss(self, batch):
        """
        Compute training loss via teacher forcing (chain rule).
        
        IV head is conditioned on OBSERVED clinician vaso action,
        NOT on model prediction. This matches chain rule exactly:
        P(a_v_obs, a_i_obs | τ) = P(a_v_obs | τ) · P(a_i_obs | τ, a_v_obs)
        """
        states, actions_factored, rtg, timesteps, masks, \
            vaso_targets, iv_targets, return_targets, state_targets = batch
        
        # Forward pass with teacher forcing: IV head sees OBSERVED vaso
        vaso_logits, iv_logits, return_preds, state_preds = self.forward(
            states, actions_factored, None, rtg, timesteps,
            attention_mask=masks, vaso_realized=vaso_targets
        )
        
        # ──── Factored action loss ────
        vaso_loss = F.cross_entropy(
            vaso_logits.reshape(-1, self.vaso_dim),
            vaso_targets.reshape(-1),
            reduction='none'
        )
        vaso_loss = (vaso_loss * masks.reshape(-1)).sum() / masks.sum()
        
        iv_loss = F.cross_entropy(
            iv_logits.reshape(-1, self.iv_dim),
            iv_targets.reshape(-1),
            reduction='none'
        )
        iv_loss = (iv_loss * masks.reshape(-1)).sum() / masks.sum()
        
        action_loss = vaso_loss + iv_loss
        
        # ──── Auxiliary losses ────
        return_loss = F.mse_loss(
            return_preds[masks.bool()],
            return_targets[masks.bool()]
        )
        state_loss = F.mse_loss(
            state_preds[masks.bool()],
            state_targets[masks.bool()]
        )
        
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
