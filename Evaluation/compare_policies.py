"""
compare_policies.py — Compare PRISM against baseline policies.

Uses M4 (Clinician vs PRISM gap) and M5 (Heatmap + TVD) from evaluate.py.
For full evaluation with all 6 metrics, use evaluate.py directly.

Usage:
    python compare_policies.py --checkpoint Model/checkpoints/best_model.pt
"""

import torch
import numpy as np
import argparse
from pathlib import Path
import sys
import os

sys.path.insert(0, str(Path(__file__).parent.parent / 'Model'))
from config import PRISMConfig
from prism_dt import PRISMDecisionTransformer
from data_loader import PRISMDataset
from evaluate import (
    evaluate_wis_bootstrap,
    evaluate_clinician_value,
    compute_action_frequencies,
    plot_action_heatmaps,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data-path', type=str, default='../Data/trajectories_shifted.pt')
    parser.add_argument('--output-dir', type=str, default='../Evaluation/results')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--eps-soften', type=float, default=None)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    config = PRISMConfig(data_path=args.data_path)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Load model
    print("=" * 70)
    print("PRISM — POLICY COMPARISON")
    print("=" * 70)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = PRISMDecisionTransformer(
        state_dim=config.state_dim, act_dim=config.act_dim,
        vaso_dim=config.vaso_dim, iv_dim=config.iv_dim,
        hidden_size=config.hidden_size, max_length=config.context_len,
        max_ep_len=config.max_ep_len, n_layer=config.n_layer,
        n_head=config.n_head, dropout=config.dropout,
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_dataset = PRISMDataset(config.data_path, config.context_len, split='test')
    
    # ──── Clinician vs PRISM Value Gap (M4) ────
    print("\n" + "=" * 70)
    print("M4: CLINICIAN vs PRISM VALUE GAP")
    print("=" * 70)
    
    clin_mean, clin_std = evaluate_clinician_value(test_dataset, config)
    
    wis, wis_lo, wis_hi, _ = evaluate_wis_bootstrap(
        model, test_dataset, config, device,
        n_bootstrap=1000, eps_soften=args.eps_soften
    )
    
    gap = wis - clin_mean
    print(f"\n  Clinician value: {clin_mean:.2f} ± {clin_std:.2f}")
    print(f"  PRISM WIS:       {wis:.2f}  [{wis_lo:.2f}, {wis_hi:.2f}] (95% CI)")
    print(f"  Gap:             {gap:+.2f}")
    
    if gap > 0:
        print(f"  → ✅ PRISM outperforms clinician")
    else:
        print(f"  → ❌ PRISM does not outperform clinician")
    
    # ──── Action Frequency Heatmap (M5) ────
    print("\n" + "=" * 70)
    print("M5: ACTION FREQUENCY HEATMAP + TVD")
    print("=" * 70)
    
    clin_heatmap, prism_heatmap, tvd = compute_action_frequencies(
        model, test_dataset, config, device
    )
    print(f"  TVD(Clinician, PRISM) = {tvd:.4f}")
    
    heatmap_path = os.path.join(args.output_dir, 'action_heatmap.png')
    plot_action_heatmaps(clin_heatmap, prism_heatmap, tvd, heatmap_path)
    
    print("\n✅ Comparison complete.")


if __name__ == '__main__':
    main()
