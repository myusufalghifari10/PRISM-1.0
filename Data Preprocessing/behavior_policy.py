"""
behavior_policy.py — Estimate π_b from trajectories using k-NN.

This is CRITICAL for valid WIS off-policy evaluation.
Without this, importance ratios π_e/π_b cannot be computed.

Usage:
    from behavior_policy import estimate_behavior_policy
    pibs, subpibs, estm_subpibs = estimate_behavior_policy(
        statevecs, actions, subactions, lengths, k=50
    )
"""

import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors


def estimate_behavior_policy(statevecs, actions, subactions, lengths, k=50,
                              smooth_eps=0.01):
    """
    Estimate π_b(a|s) using k-nearest neighbors.
    
    Args:
        statevecs  : [N, T, state_dim]  — all states
        actions    : [N, T]             — joint action indices 0-24
        subactions : [N, T, 2]          — [vaso_idx, iv_idx]
        lengths    : [N]                — actual trajectory lengths
        k          : number of neighbors
        smooth_eps : ε for Laplace smoothing (prevents zero probabilities)
    
    Returns:
        pibs         : [N, T, 25]  — π_b(a|s) as joint distribution
        subpibs      : [N, T, 10]  — π_b(a_vaso|s), π_b(a_iv|s) separately
        estm_subpibs : [N, T, 10]  — smoothed/factored estimate
    """
    N, T, D = statevecs.shape
    
    # Flatten all valid states
    valid_states = []
    valid_actions = []
    valid_vaso = []
    valid_iv = []
    
    for i in range(N):
        L = int(lengths[i].item())
        valid_states.append(statevecs[i, :L].numpy())
        valid_actions.append(actions[i, :L].numpy())
        valid_vaso.append(subactions[i, :L, 0].numpy())
        valid_iv.append(subactions[i, :L, 1].numpy())
    
    X = np.concatenate(valid_states, axis=0)  # [total_steps, D]
    y_action = np.concatenate(valid_actions)   # [total_steps]
    y_vaso = np.concatenate(valid_vaso)         # [total_steps]
    y_iv = np.concatenate(valid_iv)             # [total_steps]
    
    total_steps = len(X)
    print(f"  k-NN π_b: {total_steps} states, k={k}")
    
    # Fit k-NN
    k_actual = min(k, total_steps)
    nn = NearestNeighbors(n_neighbors=k_actual, algorithm='auto', n_jobs=-1)
    nn.fit(X)
    
    # Estimate π_b for each state
    pibs = torch.zeros(N, T, 25)
    subpibs = torch.zeros(N, T, 10)
    estm_subpibs = torch.zeros(N, T, 10)
    
    for i in range(N):
        L = int(lengths[i].item())
        states_i = statevecs[i, :L].numpy()
        
        if L == 0:
            continue
        
        distances, indices = nn.kneighbors(states_i)
        
        for t in range(L):
            neighbor_actions = y_action[indices[t]]
            neighbor_vaso = y_vaso[indices[t]]
            neighbor_iv = y_iv[indices[t]]
            
            # Joint action distribution π_b(a|s) — 25-dim
            for a in range(25):
                pibs[i, t, a] = (np.sum(neighbor_actions == a) + smooth_eps) / (k_actual + 25 * smooth_eps)
            
            # Factored sub-action distributions — 10-dim (5 vaso + 5 iv)
            for v in range(5):
                subpibs[i, t, v] = (np.sum(neighbor_vaso == v) + smooth_eps) / (k_actual + 5 * smooth_eps)
            for v in range(5):
                subpibs[i, t, 5+v] = (np.sum(neighbor_iv == v) + smooth_eps) / (k_actual + 5 * smooth_eps)
            
            # Estimated factored π_b (smoothed) — used for OPE denominator
            estm_subpibs[i, t] = subpibs[i, t].clone()
    
    print(f"  π_b estimation complete.")
    return pibs, subpibs, estm_subpibs
