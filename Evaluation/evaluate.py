"""
evaluate.py — Full off-policy evaluation for PRISM.

Implements 6 metrics:
  PRIMARY (Quantitative):
    M1. WIS Value + 95% Bootstrap CI (1000 samples)
    M2. ESS + Pareto Frontier Model Selection
    M3. Cross-Alignment WIS
    M4. Clinician vs PRISM Value Gap

  SECONDARY (Qualitative):
    M5. Action Frequency Heatmap 5x5 + Total Variation Distance (TVD)
    M6. Disagreement Quadrant Analysis + Mann-Whitney U + Bonferroni

References:
  - Tang et al., NeurIPS 2022 (Factored Action — WIS, ESS, Pareto frontier, k-NN π_b)
  - Tang et al., npj Digital Med 2026 (Shifted — cross-alignment, quadrant analysis, TVD, DR→WIS)
  - Chen et al., NeurIPS 2021 (Decision Transformer — online return replaced by WIS)

π_b: k-NN (k=50) in 47-dim continuous state space, Laplace smoothing ε=0.01
π_e: Softmax from DT factored output heads (no softening by default — softmax ensures π_e > 0)

Usage:
    python evaluate.py --checkpoint Model/checkpoints/best_model.pt
"""

import torch
import torch.nn.functional as F
import numpy as np
import argparse
import os
from pathlib import Path
import sys
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent / 'Model'))
from config import PRISMConfig
from prism_dt import PRISMDecisionTransformer
from data_loader import PRISMDataset, collate_fn
from torch.utils.data import DataLoader
from scipy import stats


# ═══════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def remap_rewards(rewards, config):
    """Remap sparse rewards for OPE evaluation.
    
    Original: 0 (intermediate), +1 (survival), -1 (death)
    OPE:      0, +R_survival, +R_death
    """
    result = np.select(
        [rewards == 0, rewards == -1, rewards == 1],
        [config.R_immediate, config.R_death, config.R_survival]
    )
    return result


def compute_discounted_return(rewards_raw, config):
    """Compute discounted return from raw per-step rewards."""
    rewards = remap_rewards(rewards_raw, config)
    return np.sum(rewards * (config.eval_discount ** np.arange(len(rewards))))


def total_variation_distance(P, Q):
    """Total Variation Distance between two probability distributions.
    
    TVD(P, Q) = 0.5 * Σ |P_i - Q_i|
    Range: [0, 1] where 0 = identical, 1 = completely different.
    """
    return 0.5 * np.sum(np.abs(P - Q))


# ═══════════════════════════════════════════════════════════════
# M1: WIS Value + Bootstrap CI
# ═══════════════════════════════════════════════════════════════

def evaluate_wis_single(model, test_dataset, config, device='cpu', eps_soften=None):
    """
    Compute WIS estimate for a SINGLE evaluation (no bootstrap).
    
    Args:
        eps_soften: If not None, use ε-softening: π_e = (1-ε)*π_DT + ε*π_b
                    If None (default), use pure softmax π_e.
    
    Returns:
        wis_value: float
        ess: float
        per_traj_returns: [N] array
        per_traj_weights: [N] array
    """
    model.eval()
    
    n_traj = len(test_dataset.indices)
    all_weights = []
    all_returns = []
    
    for idx in range(n_traj):
        traj_idx = test_dataset.indices[idx]
        L = int(test_dataset.lengths[traj_idx].item())
        
        if L < 2:
            continue
        
        # Get trajectory data
        states         = test_dataset.statevecs[traj_idx, :L]
        true_actions   = test_dataset.actions[traj_idx, :L]
        true_subactionvecs = test_dataset.subactionvecs[traj_idx, :L]
        rewards_raw    = test_dataset.rewards[traj_idx, :L].numpy()
        rtg            = test_dataset.returns_to_go[traj_idx, :L].unsqueeze(-1)
        subpibs_traj   = test_dataset.subpibs[traj_idx, :L]  # [L, 10]
        
        # Compute discounted return
        discounted_return = compute_discounted_return(rewards_raw, config)
        
        # ──── Autoregressive importance ratio ────
        ir = 1.0
        context_states = []
        context_actions = []
        context_rtg = []
        
        with torch.no_grad():
            for t in range(L):
                ctx_len = len(context_states)
                ctx_start = max(0, ctx_len - config.context_len)
                
                if ctx_len == 0:
                    s_t = states[t:t+1].unsqueeze(0).to(device)
                    a_prev = torch.zeros(1, 1, 10).to(device)
                    r_t = rtg[t:t+1].unsqueeze(0).to(device)
                    ts_t = torch.tensor([[0]]).to(device)
                else:
                    s_t = torch.stack(context_states[ctx_start:]).unsqueeze(0).to(device)
                    a_prev = torch.stack(context_actions[ctx_start:]).unsqueeze(0).to(device)
                    r_t = torch.stack(context_rtg[ctx_start:]).unsqueeze(0).to(device)
                    ts_t = torch.tensor([list(range(len(context_states[ctx_start:])))], device=device)
                
                # ──── Chain rule teacher forcing (same as compute_loss) ────
                # Step 1: Get P_vaso(a_v_obs | τ) — vaso_realized=None for vaso inference
                vaso_logits, _, _, _ = model(
                    s_t, a_prev, None, r_t, ts_t, vaso_realized=None
                )
                vaso_probs = F.softmax(vaso_logits[0, -1], dim=-1).cpu().numpy()
                
                # Observed action
                a_true = int(true_actions[t].item())
                a_vaso_true = a_true % 5
                a_iv_true   = a_true // 5
                
                # π_e_vaso = P(a_v_obs | τ)
                pi_e_vaso = vaso_probs[a_vaso_true]
                
                # Step 2: Get P_iv(a_i_obs | τ, a_v_obs) — conditioned on OBSERVED vaso
                # vaso_obs_tensor must match time dimension of context [1, T→s_t.shape[1]]
                vaso_obs_tensor = torch.full((1, s_t.shape[1]), a_vaso_true, device=device, dtype=torch.long)
                _, iv_logits, _, _ = model(
                    s_t, a_prev, None, r_t, ts_t, vaso_realized=vaso_obs_tensor
                )
                iv_probs = F.softmax(iv_logits[0, -1], dim=-1).cpu().numpy()
                
                # π_e_iv = P(a_i_obs | τ, a_v_obs)
                pi_e_iv = iv_probs[a_iv_true]
                
                # π_e(a_obs | τ) = P(a_v_obs | τ) · P(a_i_obs | τ, a_v_obs)  ← chain rule
                pi_e = pi_e_vaso * pi_e_iv
                
                # π_b(a_true | s_t) from k-NN
                pi_b = (subpibs_traj[t, a_vaso_true].item() *
                        subpibs_traj[t, 5 + a_iv_true].item())
                pi_b = max(pi_b, 1e-6)
                
                # Optional ε-softening
                if eps_soften is not None:
                    pi_e = (1.0 - eps_soften) * pi_e + eps_soften * pi_b
                
                ir *= (pi_e / pi_b)
                
                # Update context with GROUND TRUTH for IS
                context_states.append(states[t])
                context_actions.append(true_subactionvecs[t])
                if t < L - 1:
                    context_rtg.append(rtg[t+1].reshape(1))
                else:
                    context_rtg.append(torch.tensor([[0.0]]))
        
        ir = np.clip(ir, 0, 1e3)
        all_weights.append(ir)
        all_returns.append(discounted_return)
    
    weights = np.array(all_weights)
    returns = np.array(all_returns)
    
    sum_w = np.sum(weights)
    wis = np.sum(weights * returns) / (sum_w + 1e-8)
    ess = (sum_w ** 2) / (np.sum(weights ** 2) + 1e-8)
    
    return wis, ess, returns, weights


def evaluate_wis_bootstrap(model, test_dataset, config, device='cpu',
                           n_bootstrap=1000, eps_soften=None):
    """
    M1: WIS Value with 95% Bootstrap Confidence Interval.
    
    Performs M1 metric from Factored Action + Shifted papers.
    """
    _, _, returns, weights = evaluate_wis_single(
        model, test_dataset, config, device, eps_soften
    )
    
    n = len(returns)
    rng = np.random.RandomState(42)
    bootstrap_vals = np.zeros(n_bootstrap)
    
    for b in range(n_bootstrap):
        indices = rng.choice(n, size=n, replace=True)
        w_b = weights[indices]
        r_b = returns[indices]
        bootstrap_vals[b] = np.sum(w_b * r_b) / (np.sum(w_b) + 1e-8)
    
    ci_lower = np.percentile(bootstrap_vals, 2.5)
    ci_upper = np.percentile(bootstrap_vals, 97.5)
    mean_wis = np.mean(bootstrap_vals)
    
    return mean_wis, ci_lower, ci_upper, bootstrap_vals


# ═══════════════════════════════════════════════════════════════
# M2: ESS + Pareto Frontier Model Selection
# ═══════════════════════════════════════════════════════════════

def evaluate_ess_wis(model, test_dataset, config, device='cpu', eps_soften=None):
    """
    M2: Compute ESS and WIS for a single model checkpoint.
    
    Used in Pareto frontier: select model with max WIS where ESS >= threshold.
    """
    wis, ess, _, _ = evaluate_wis_single(
        model, test_dataset, config, device, eps_soften
    )
    return wis, ess


# ═══════════════════════════════════════════════════════════════
# M3: Cross-Alignment WIS
# ═══════════════════════════════════════════════════════════════

def evaluate_cross_alignment(model_shifted, model_original,
                              test_dataset_shifted, test_dataset_original,
                              config, device='cpu'):
    """
    M3: Cross-Alignment WIS Evaluation.
    
    Evaluate shifted-trained policy on BOTH shifted and original test data.
    Evaluate original-trained policy on BOTH shifted and original test data.
    
    Expectation:
        WIS(shifted_policy, shifted_test) > WIS(original_policy, shifted_test)
    
    Returns:
        dict with all 4 combinations
    """
    results = {}
    
    # Shifted policy on shifted test (should be best)
    wis_ss, ess_ss, _, _ = evaluate_wis_single(
        model_shifted, test_dataset_shifted, config, device
    )
    results['shifted_policy_shifted_test'] = {'WIS': wis_ss, 'ESS': ess_ss}
    
    # Shifted policy on original test (should still be good)
    # Need to load original test data
    wis_so, ess_so, _, _ = evaluate_wis_single(
        model_shifted, test_dataset_original, config, device
    )
    results['shifted_policy_original_test'] = {'WIS': wis_so, 'ESS': ess_so}
    
    # Original policy on shifted test (should be WORSE — proves misalignment)
    if model_original is not None:
        wis_os, ess_os, _, _ = evaluate_wis_single(
            model_original, test_dataset_shifted, config, device
        )
        results['original_policy_shifted_test'] = {'WIS': wis_os, 'ESS': ess_os}
        
        # Original policy on original test 
        wis_oo, ess_oo, _, _ = evaluate_wis_single(
            model_original, test_dataset_original, config, device
        )
        results['original_policy_original_test'] = {'WIS': wis_oo, 'ESS': ess_oo}
    
    return results


# ═══════════════════════════════════════════════════════════════
# M4: Clinician vs PRISM Value Gap
# ═══════════════════════════════════════════════════════════════

def evaluate_clinician_value(test_dataset, config):
    """
    M4: Compute clinician policy value (mean discounted return on test set).
    
    This is the ground-truth outcome of what actually happened.
    PRISM WIS should exceed this value.
    """
    returns = []
    n_traj = len(test_dataset.indices)
    
    for idx in range(n_traj):
        traj_idx = test_dataset.indices[idx]
        L = int(test_dataset.lengths[traj_idx].item())
        if L < 2:
            continue
        
        rewards_raw = test_dataset.rewards[traj_idx, :L].numpy()
        discounted = compute_discounted_return(rewards_raw, config)
        returns.append(discounted)
    
    return np.mean(returns), np.std(returns)


# ═══════════════════════════════════════════════════════════════
# M5: Action Frequency Heatmap 5x5 + Total Variation Distance
# ═══════════════════════════════════════════════════════════════

def compute_action_frequencies(model, test_dataset, config, device='cpu'):
    """
    Compute action frequency heatmap for both clinician and PRISM.
    
    Returns:
        clinician_heatmap: [5, 5] — frequency of each (vaso, iv) by clinician
        prism_heatmap: [5, 5] — frequency of each (vaso, iv) by PRISM
        tvd: float — Total Variation Distance between the two heatmaps
    """
    model.eval()
    
    clinician_counts = np.zeros((5, 5))
    prism_counts = np.zeros((5, 5))
    
    n_traj = len(test_dataset.indices)
    
    for idx in range(n_traj):
        traj_idx = test_dataset.indices[idx]
        L = int(test_dataset.lengths[traj_idx].item())
        if L < 2:
            continue
        
        true_actions = test_dataset.actions[traj_idx, :L]
        states = test_dataset.statevecs[traj_idx, :L]
        rtg = test_dataset.returns_to_go[traj_idx, :L].unsqueeze(-1)
        
        # Count clinician actions
        for t in range(L):
            a = int(true_actions[t].item())
            vaso = a % 5
            iv = a // 5
            clinician_counts[vaso, iv] += 1
        
        # PRISM predictions (autoregressive with ground-truth context)
        context_states = []
        context_actions = []
        context_rtg = []
        
        with torch.no_grad():
            for t in range(L):
                ctx_len = len(context_states)
                ctx_start = max(0, ctx_len - config.context_len)
                
                if ctx_len == 0:
                    s_t = states[t:t+1].unsqueeze(0).to(device)
                    a_prev = torch.zeros(1, 1, 10).to(device)
                    r_t = rtg[t:t+1].unsqueeze(0).to(device)
                    ts_t = torch.tensor([[0]]).to(device)
                else:
                    s_t = torch.stack(context_states[ctx_start:]).unsqueeze(0).to(device)
                    a_prev = torch.stack(context_actions[ctx_start:]).unsqueeze(0).to(device)
                    r_t = torch.stack(context_rtg[ctx_start:]).unsqueeze(0).to(device)
                    ts_t = torch.tensor([list(range(len(context_states[ctx_start:])))], device=device)
                
                # Sequential generation for visualization: vaso → argmax → iv conditioned on argmax
                vaso_logits, _, _, _ = model(
                    s_t, a_prev, None, r_t, ts_t, vaso_realized=None
                )
                vaso_pred = torch.argmax(vaso_logits[0, -1]).item()
                
                vaso_cond = torch.full((1, s_t.shape[1]), vaso_pred, device=device, dtype=torch.long)
                _, iv_logits, _, _ = model(
                    s_t, a_prev, None, r_t, ts_t, vaso_realized=vaso_cond
                )
                iv_pred = torch.argmax(iv_logits[0, -1]).item()
                
                prism_counts[vaso_pred, iv_pred] += 1
                
                # Update context with ground truth
                true_sub = test_dataset.subactionvecs[traj_idx, t]
                context_states.append(states[t])
                context_actions.append(true_sub)
                if t < L - 1:
                    context_rtg.append(rtg[t+1].reshape(1))
                else:
                    context_rtg.append(torch.tensor([[0.0]]))
    
    # Normalize to probability distributions
    clinician_heatmap = clinician_counts / clinician_counts.sum()
    prism_heatmap = prism_counts / prism_counts.sum()
    
    # Total Variation Distance
    tvd = total_variation_distance(clinician_heatmap, prism_heatmap)
    
    return clinician_heatmap, prism_heatmap, tvd


def plot_action_heatmaps(clinician_heatmap, prism_heatmap, tvd, save_path):
    """Plot side-by-side action frequency heatmaps."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    vaso_labels = ['None', 'Low', 'Med-Low', 'Med-High', 'High']
    iv_labels = ['None', 'Low', 'Med-Low', 'Med-High', 'High']
    
    vmax = max(clinician_heatmap.max(), prism_heatmap.max())
    
    # Clinician heatmap
    sns.heatmap(clinician_heatmap, annot=True, fmt='.3f', cmap='Blues',
                xticklabels=iv_labels, yticklabels=vaso_labels,
                vmin=0, vmax=vmax, ax=axes[0], cbar=False)
    axes[0].set_title('Clinician Policy')
    axes[0].set_xlabel('IV Fluid Dose')
    axes[0].set_ylabel('Vasopressor Dose')
    
    # PRISM heatmap
    sns.heatmap(prism_heatmap, annot=True, fmt='.3f', cmap='Oranges',
                xticklabels=iv_labels, yticklabels=vaso_labels,
                vmin=0, vmax=vmax, ax=axes[1], cbar=False)
    axes[1].set_title('PRISM Policy')
    axes[1].set_xlabel('IV Fluid Dose')
    axes[1].set_ylabel('Vasopressor Dose')
    
    # Difference heatmap
    diff = prism_heatmap - clinician_heatmap
    abs_max = np.abs(diff).max()
    sns.heatmap(diff, annot=True, fmt='+.3f', cmap='RdBu_r',
                xticklabels=iv_labels, yticklabels=vaso_labels,
                vmin=-abs_max, vmax=abs_max, ax=axes[2], center=0)
    axes[2].set_title(f'Difference (PRISM − Clinician)\nTVD = {tvd:.4f}')
    axes[2].set_xlabel('IV Fluid Dose')
    axes[2].set_ylabel('Vasopressor Dose')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Saved action heatmap: {save_path}")


# ═══════════════════════════════════════════════════════════════
# M6: Disagreement Quadrant Analysis
# ═══════════════════════════════════════════════════════════════

QUADRANT_LABELS = {
    'agree_treat': 'Both recommend treatment, same action',
    'agree_notreat': 'Both recommend no treatment',
    'disagree_prism_treats': 'PRISM treats, clinician does not',
    'disagree_clinician_treats': 'Clinician treats, PRISM does not',
    'disagree_both_diff': 'Both treat, different actions',
}

def classify_quadrant(a_clinician, a_prism):
    """
    Classify a state into one of 5 quadrants.
    
    Returns: quadrant name string
    """
    # "No treatment" = action 0 (no vaso, no IV)
    clinician_notreat = (a_clinician == 0)
    prism_notreat = (a_prism == 0)
    
    if clinician_notreat and prism_notreat:
        return 'agree_notreat'
    elif clinician_notreat and not prism_notreat:
        return 'disagree_prism_treats'
    elif not clinician_notreat and prism_notreat:
        return 'disagree_clinician_treats'
    elif a_clinician == a_prism:
        return 'agree_treat'
    else:
        return 'disagree_both_diff'



def quadrant_analysis(model, test_dataset, config, device='cpu'):
    """
    M6: Disagreement Quadrant Analysis with statistical tests.
    
    Partitions all test states into 5 quadrants based on clinician vs PRISM
    action agreement. For each quadrant, computes GCS, SOFA, MeanBP statistics
    and performs Mann-Whitney U tests with Bonferroni correction.
    
    Based on: Tang et al., npj Digital Medicine 2026, Fig 2a-d.
    """
    model.eval()
    
    # Collect per-quadrant clinical features
    quadrant_features = defaultdict(list)
    quadrant_counts = defaultdict(int)
    
    n_traj = len(test_dataset.indices)
    
    for idx in range(n_traj):
        traj_idx = test_dataset.indices[idx]
        L = int(test_dataset.lengths[traj_idx].item())
        if L < 2:
            continue
        
        true_actions = test_dataset.actions[traj_idx, :L]
        states = test_dataset.statevecs[traj_idx, :L]
        rtg = test_dataset.returns_to_go[traj_idx, :L].unsqueeze(-1)
        
        context_states = []
        context_actions = []
        context_rtg = []
        
        with torch.no_grad():
            for t in range(L):
                ctx_len = len(context_states)
                ctx_start = max(0, ctx_len - config.context_len)
                
                if ctx_len == 0:
                    s_t = states[t:t+1].unsqueeze(0).to(device)
                    a_prev = torch.zeros(1, 1, 10).to(device)
                    r_t = rtg[t:t+1].unsqueeze(0).to(device)
                    ts_t = torch.tensor([[0]]).to(device)
                else:
                    s_t = torch.stack(context_states[ctx_start:]).unsqueeze(0).to(device)
                    a_prev = torch.stack(context_actions[ctx_start:]).unsqueeze(0).to(device)
                    r_t = torch.stack(context_rtg[ctx_start:]).unsqueeze(0).to(device)
                    ts_t = torch.tensor([list(range(len(context_states[ctx_start:])))], device=device)
                
                # Sequential generation for quadrant classification
                vaso_logits, _, _, _ = model(
                    s_t, a_prev, None, r_t, ts_t, vaso_realized=None
                )
                vaso_pred = torch.argmax(vaso_logits[0, -1]).item()
                
                vaso_cond = torch.full((1, s_t.shape[1]), vaso_pred, device=device, dtype=torch.long)
                _, iv_logits, _, _ = model(
                    s_t, a_prev, None, r_t, ts_t, vaso_realized=vaso_cond
                )
                iv_pred = torch.argmax(iv_logits[0, -1]).item()
                a_prism = vaso_pred + iv_pred * 5
                
                a_clinician = int(true_actions[t].item())
                
                quadrant = classify_quadrant(a_clinician, a_prism)
                quadrant_counts[quadrant] += 1
                
                # Extract clinical features from state
                s_np = states[t].numpy()
                gcs_val = s_np[6] if len(s_np) > 6 else np.nan
                sofa_val = s_np[31] if len(s_np) > 31 else np.nan
                meanbp_val = s_np[9] if len(s_np) > 9 else np.nan
                
                quadrant_features[quadrant].append({
                    'gcs': gcs_val,
                    'sofa': sofa_val,
                    'meanbp': meanbp_val,
                })
                
                # Update context
                true_sub = test_dataset.subactionvecs[traj_idx, t]
                context_states.append(states[t])
                context_actions.append(true_sub)
                if t < L - 1:
                    context_rtg.append(rtg[t+1].reshape(1))
                else:
                    context_rtg.append(torch.tensor([[0.0]]))
    
    # ──── Statistical Tests ────
    print("\n" + "=" * 70)
    print("M6: DISAGREEMENT QUADRANT ANALYSIS")
    print("=" * 70)
    
    total_states = sum(quadrant_counts.values())
    print(f"\n  Total states analyzed: {total_states}")
    print(f"\n  Quadrant Distribution:")
    for quadrant in ['agree_treat', 'agree_notreat', 'disagree_prism_treats',
                     'disagree_clinician_treats', 'disagree_both_diff']:
        count = quadrant_counts.get(quadrant, 0)
        pct = 100 * count / total_states if total_states > 0 else 0
        label = QUADRANT_LABELS.get(quadrant, quadrant)
        print(f"    {quadrant:30s}: {count:6d} ({pct:5.1f}%)  — {label}")
    
    # Overall statistics
    all_gcs = []
    all_sofa = []
    all_meanbp = []
    for q_features in quadrant_features.values():
        for f in q_features:
            if not np.isnan(f['gcs']): all_gcs.append(f['gcs'])
            if not np.isnan(f['sofa']): all_sofa.append(f['sofa'])
            if not np.isnan(f['meanbp']): all_meanbp.append(f['meanbp'])
    
    mean_gcs = np.mean(all_gcs) if all_gcs else 0
    mean_sofa = np.mean(all_sofa) if all_sofa else 0
    mean_meanbp = np.mean(all_meanbp) if all_meanbp else 0
    
    print(f"\n  Overall means — GCS: {mean_gcs:.2f}, SOFA: {mean_sofa:.2f}, MeanBP: {mean_meanbp:.2f}")
    print(f"  (NOTE: Values are z-scored. For clinical interpretation, raw values needed for publication.)")
    print(f"\n  Per-Quadrant Clinical Features:")
    print(f"  {'Quadrant':30s} | {'GCS mean':>8s} | {'SOFA mean':>9s} | {'MeanBP mean':>11s} | vs Overall")
    print(f"  {'-'*30} | {'-'*8} | {'-'*9} | {'-'*11} | {'-'*9}")
    
    # Mann-Whitney U tests per quadrant vs all others
    # Collect ALL quadrant data first (Loop 1), then test (Loop 2).
    # Previous code tested quadrants against incomplete sets — BUG FIXED.
    quadrant_gcs = {}
    quadrant_sofa = {}
    quadrant_meanbp = {}
    
    for quadrant in ['agree_treat', 'agree_notreat', 'disagree_prism_treats',
                     'disagree_clinician_treats', 'disagree_both_diff']:
        features = quadrant_features.get(quadrant, [])
        gcs_vals = [f['gcs'] for f in features if not np.isnan(f['gcs'])]
        sofa_vals = [f['sofa'] for f in features if not np.isnan(f['sofa'])]
        meanbp_vals = [f['meanbp'] for f in features if not np.isnan(f['meanbp'])]
        
        if len(gcs_vals) < 5:
            continue
        
        quadrant_gcs[quadrant] = np.array(gcs_vals)
        quadrant_sofa[quadrant] = np.array(sofa_vals)
        quadrant_meanbp[quadrant] = np.array(meanbp_vals)
        
        gcs_mean = np.mean(gcs_vals)
        sofa_mean = np.mean(sofa_vals)
        meanbp_mean = np.mean(meanbp_vals)
        
        # Note: MWU tests done in Loop 2 below, after ALL quadrant data is collected
        direction = "same"
        if gcs_mean > mean_gcs + 0.01:
            direction = "healthier ↑"
        elif gcs_mean < mean_gcs - 0.01:
            direction = "sicker ↓"
        
        print(f"  {quadrant:30s} | {gcs_mean:8.2f} | {sofa_mean:9.2f} | {meanbp_mean:11.2f} | {direction}")
    
    # ──── Loop 2: MWU tests with COMPLETE quadrant data ────
    for quadrant in ['agree_treat', 'agree_notreat', 'disagree_prism_treats',
                     'disagree_clinician_treats', 'disagree_both_diff']:
        if quadrant not in quadrant_gcs:
            continue
        gcs_vals = quadrant_gcs[quadrant]
        
        other_gcs = np.concatenate([v for q, v in quadrant_gcs.items() if q != quadrant])
        
        if len(other_gcs) > 0 and len(gcs_vals) > 0:
            u_gcs, p_gcs = stats.mannwhitneyu(gcs_vals, other_gcs, alternative='two-sided')
        else:
            u_gcs, p_gcs = 0, 1.0
        
        # Bonferroni correction: 5 quadrants × 3 tests = 15 comparisons
        p_gcs_corrected = min(p_gcs * 15, 1.0)
        
        sig_gcs = "***" if p_gcs_corrected < 0.001 else ("**" if p_gcs_corrected < 0.01 else ("*" if p_gcs_corrected < 0.05 else ""))
        print(f"  {'':30s}   MWU vs others: U={u_gcs:.0f}, p_corr={p_gcs_corrected:.2e} {sig_gcs}")
    
    # ──── Clinical Interpretation ────
    print(f"\n  Clinical Interpretation:")
    
    prism_treats_more = quadrant_counts.get('disagree_prism_treats', 0)
    clinician_treats_more = quadrant_counts.get('disagree_clinician_treats', 0)
    
    if prism_treats_more > clinician_treats_more:
        print(f"  → PRISM is MORE aggressive (treats in {prism_treats_more} vs {clinician_treats_more} states)")
    elif clinician_treats_more > prism_treats_more:
        print(f"  → PRISM is LESS aggressive (treats in {prism_treats_more} vs {clinician_treats_more} states)")
    
    # Check if PRISM overtreats healthier patients or undertreats sicker ones
    for quadrant in ['disagree_prism_treats', 'disagree_clinician_treats']:
        if quadrant in quadrant_gcs and len(quadrant_gcs[quadrant]) > 5:
            q_mean = np.mean(quadrant_gcs[quadrant])
            if quadrant == 'disagree_prism_treats' and q_mean > mean_gcs:
                print(f"  → PRISM may OVERTREAT healthier patients (GCS {q_mean:.1f} > avg {mean_gcs:.1f})")
            if quadrant == 'disagree_clinician_treats' and q_mean < mean_gcs:
                print(f"  → Clinician may UNDERTREAT sicker patients (GCS {q_mean:.1f} < avg {mean_gcs:.1f})")
    
    return quadrant_counts, quadrant_features


# ═══════════════════════════════════════════════════════════════
# MAIN — RUN ALL METRICS
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='PRISM — Full Offline Evaluation')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint (.pt)')
    parser.add_argument('--data-path', type=str, default='../Data/trajectories_shifted.pt',
                       help='Path to shifted trajectory data')
    parser.add_argument('--original-data', type=str, default=None,
                       help='Path to ORIGINAL-alignment data for cross-alignment eval (optional)')
    parser.add_argument('--output-dir', type=str, default='../Evaluation/results',
                       help='Output directory for plots and results')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device: cuda or cpu')
    parser.add_argument('--eps-soften', type=float, default=None,
                       help='ε for softening π_e (default: None = no softening)')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    config = PRISMConfig(data_path=args.data_path)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # ──── Load Model ────
    print("=" * 70)
    print("PRISM — FULL OFFLINE EVALUATION (6 Metrics)")
    print("=" * 70)
    
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
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
        dropout=config.dropout,
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"\n  Checkpoint: {args.checkpoint}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}, Step: {checkpoint.get('global_step', 'N/A')}")
    print(f"  Device: {device}")
    print(f"  ε-softening: {args.eps_soften if args.eps_soften else 'None (pure softmax)'}")
    
    # ──── Load Data ────
    test_dataset = PRISMDataset(
        config.data_path,
        context_len=config.context_len,
        split='test',
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
    )
    
    # Load original data if provided
    test_dataset_orig = None
    if args.original_data:
        config_orig = PRISMConfig(data_path=args.original_data)
        test_dataset_orig = PRISMDataset(
            args.original_data,
            context_len=config.context_len,
            split='test',
            train_ratio=config.train_ratio,
            val_ratio=config.val_ratio,
        )
    
    # ════════════════════════════════════════════════════════
    # M1: WIS Value + 95% Bootstrap CI
    # ════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("M1: WIS VALUE + 95% BOOTSTRAP CI")
    print("=" * 70)
    
    wis, wis_lo, wis_hi, _ = evaluate_wis_bootstrap(
        model, test_dataset, config, device,
        n_bootstrap=1000, eps_soften=args.eps_soften
    )
    print(f"\n  WIS = {wis:.2f}  [{wis_lo:.2f}, {wis_hi:.2f}] (95% CI, 1000 bootstrap)")
    
    # ════════════════════════════════════════════════════════
    # M2: ESS
    # ════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("M2: ESS + MODEL SELECTION")
    print("=" * 70)
    
    _, ess, _, _ = evaluate_wis_single(
        model, test_dataset, config, device, eps_soften=args.eps_soften
    )
    ess_ok = "✅ PASS (≥ 200)" if ess >= 200 else f"⚠️ FAIL (< 200)"
    print(f"\n  ESS = {ess:.1f}  {ess_ok}")
    
    # ════════════════════════════════════════════════════════
    # M3: Cross-Alignment WIS (if original data available)
    # ════════════════════════════════════════════════════════
    if test_dataset_orig is not None and args.original_data:
        print("\n" + "=" * 70)
        print("M3: CROSS-ALIGNMENT WIS")
        print("=" * 70)
        
        wis_shifted_orig, _, _, _ = evaluate_wis_single(
            model, test_dataset_orig, config, device, eps_soften=args.eps_soften
        )
        print(f"\n  Shifted-trained policy on ORIGINAL test:  WIS = {wis_shifted_orig:.2f}")
        print(f"  Shifted-trained policy on SHIFTED test:   WIS = {wis:.2f}")
        
        if wis > wis_shifted_orig:
            print(f"  → ✅ Shifted alignment outperforms original (gap = {wis - wis_shifted_orig:.2f})")
        else:
            print(f"  → ⚠️ Unexpected: original alignment scores higher")
    
    # ════════════════════════════════════════════════════════
    # M4: Clinician vs PRISM Value Gap
    # ════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("M4: CLINICIAN vs PRISM VALUE GAP")
    print("=" * 70)
    
    clin_mean, clin_std = evaluate_clinician_value(test_dataset, config)
    gap = wis - clin_mean
    
    print(f"\n  Clinician value (mean discounted return): {clin_mean:.2f} ± {clin_std:.2f}")
    print(f"  PRISM WIS:                               {wis:.2f}  [{wis_lo:.2f}, {wis_hi:.2f}]")
    print(f"  Gap (PRISM − Clinician):                 {gap:+.2f}")
    
    if gap > 0:
        print(f"  → ✅ PRISM outperforms clinician by {gap:.2f}")
    else:
        print(f"  → ❌ PRISM does NOT outperform clinician (gap = {gap:.2f})")
    
    # ════════════════════════════════════════════════════════
    # M5: Action Frequency Heatmap 5x5 + TVD
    # ════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("M5: ACTION FREQUENCY HEATMAP + TVD")
    print("=" * 70)
    
    clin_heatmap, prism_heatmap, tvd = compute_action_frequencies(
        model, test_dataset, config, device
    )
    print(f"\n  TVD(Clinician, PRISM) = {tvd:.4f}")
    
    heatmap_path = os.path.join(args.output_dir, 'action_heatmap.png')
    plot_action_heatmaps(clin_heatmap, prism_heatmap, tvd, heatmap_path)
    
    # ════════════════════════════════════════════════════════
    # M6: Disagreement Quadrant Analysis
    # ════════════════════════════════════════════════════════
    # (Already prints inside the function)
    quadrant_analysis(model, test_dataset, config, device)
    
    # ──── Final Summary ────
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"""
    M1: WIS = {wis:.2f}  [{wis_lo:.2f}, {wis_hi:.2f}] (95% CI)
    M2: ESS = {ess:.1f}
    M3: Cross-Alignment = {'✅ Evaluated' if test_dataset_orig else '⚠️ Skipped (no original data)'}
    M4: Gap = {gap:+.2f}  (Clinician = {clin_mean:.2f})
    M5: TVD = {tvd:.4f}
    M6: Quadrant Analysis = ✅ Complete
    """)
    
    print(f"  Plots saved to: {args.output_dir}/")


if __name__ == '__main__':
    main()
