"""
build_trajectories.py — Master orchestrator for PRISM data preprocessing.

Usage:
    python build_trajectories.py --db-host localhost --db-name mimic --db-user postgres --db-pass xxx

Output:
    Data/trajectories_shifted.pt  — Dict with keys:
        'statevecs'       : [N, horizon, 47]     float32 tensor
        'actions'         : [N, horizon]          int64 tensor  (0–24)
        'subactions'      : [N, horizon, 2]       int64 tensor  (vaso, iv)
        'subactionvecs'   : [N, horizon, 10]      float32 tensor (one-hot encoding)
        'rewards'         : [N, horizon]          float32 tensor (RAW per-step, NOT reconstructed)
        'returns_to_go'   : [N, horizon]          float32 tensor (undiscounted cumsum)
        'notdones'        : [N, horizon]          float32 tensor
        'timesteps'       : [N, horizon]          int64 tensor
        'lengths'         : [N]                   int64 tensor
        'pibs'            : [N, horizon, 25]      float32 tensor (π_b from k-NN)
        'subpibs'         : [N, horizon, 10]      float32 tensor (factored π_b)
        'estm_subpibs'    : [N, horizon, 10]      float32 tensor (smoothed factored π_b)
"""

import argparse
import sys
import os
import numpy as np
import pandas as pd
import torch

# Add parent to path
sys.path.insert(0, os.path.dirname(__file__))

# timestep_utils.py is designed for discretized 750-state pipeline.
# Since PRISM uses 47 continuous features, we apply shifted alignment
# inline to preserve o:* (observation) columns.
from behavior_policy import estimate_behavior_policy


def main():
    parser = argparse.ArgumentParser(description='PRISM Data Preprocessing Pipeline')
    parser.add_argument('--db-host', default='localhost')
    parser.add_argument('--db-name', default='mimic')
    parser.add_argument('--db-user', default='postgres')
    parser.add_argument('--db-pass', required=True)
    parser.add_argument('--data-dir', default='../Data')
    parser.add_argument('--skip-extract', action='store_true',
                        help='Skip preprocess.py (use existing processed_files/)')
    parser.add_argument('--skip-cohort', action='store_true',
                        help='Skip sepsis_cohort.py (use existing sepsis_final_data_withTimes.csv)')
    args = parser.parse_args()
    
    data_dir = os.path.abspath(args.data_dir)
    os.makedirs(data_dir, exist_ok=True)
    
    # ──── Step 1: Extract from MIMIC-III ────
    if not args.skip_extract:
        print("=" * 60)
        print("STEP 1/5: Extracting from MIMIC-III PostgreSQL...")
        print("=" * 60)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        exit_code = os.system(
            f"cd '{script_dir}' && python preprocess.py "
            f"--db-host {args.db_host} --db-name {args.db_name} "
            f"--db-user {args.db_user} --db-pass {args.db_pass}"
        )
        if exit_code != 0:
            print("ERROR: preprocess.py failed")
            sys.exit(1)
        # Move output to data dir
        os.system(f"mv '{script_dir}/processed_files' '{data_dir}/'")
    
    # ──── Step 2: Build sepsis cohort ────
    if not args.skip_cohort:
        print("=" * 60)
        print("STEP 2/5: Building sepsis cohort (this WILL take 2-3 hours)...")
        print("=" * 60)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        exit_code = os.system(f"cd '{script_dir}' && python sepsis_cohort.py")
        if exit_code != 0:
            print("ERROR: sepsis_cohort.py failed")
            sys.exit(1)
    
    # ──── Step 3: Load & apply shifted alignment ────
    print("=" * 60)
    print("STEP 3/5: Applying SHIFTED temporal alignment...")
    print("=" * 60)
    
    csv_path = f'{data_dir}/sepsis_final_data_withTimes.csv'
    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found. Run sepsis_cohort.py first.")
        sys.exit(1)
    
    df = pd.read_csv(csv_path)
    
    shifted_trajs = []
    for traj_id in df['traj'].unique():
        traj_df = df[df['traj'] == traj_id].copy()
        
        # ─── Apply SHIFTED alignment inline (preserving o:* columns) ───
        # make_traj_shifted() expects discretized s:state column (750 clusters)
        # which doesn't exist in our continuous-feature pipeline.
        # We replicate its logic directly:
        #   1. Shift action backward: state[t] now paired with action[t-1]
        #   2. Shift reward earlier (to keep terminal reward after dropping last step)
        #   3. Drop last step (NaN action)
        
        traj_df['a:action'] = traj_df['a:action'].shift(-1)
        traj_df['r:reward'] = traj_df['r:reward'].shift(-1)
        traj_df = traj_df.iloc[:-1]  # drop last step
        traj_df['a:action'] = traj_df['a:action'].astype('Int64')
        
        if len(traj_df) > 1:  # Need at least 2 steps
            shifted_trajs.append(traj_df)
    
    print(f"  → {len(shifted_trajs)} trajectories after shifted alignment")
    
    # ──── Step 4: Convert to DT format ────
    print("=" * 60)
    print("STEP 4/5: Converting to DT-compatible tensor format...")
    print("=" * 60)
    
    # Determine max trajectory length
    max_len = max(len(t) for t in shifted_trajs)
    print(f"  → Max trajectory length: {max_len}")
    
    # State dimension: all observation columns (o:*) = 47 features
    # = colbin(4) + colnorm(32) + collog(11)
    # colbin: gender, mechvent, max_dose_vaso, re_admission
    # colnorm: age, Weight_kg, GCS, HR, SysBP, MeanBP, DiaBP, RR, Temp_C,
    #          FiO2_1, Potassium, Sodium, Chloride, Glucose, Magnesium,
    #          Calcium, Hb, WBC_count, Platelets_count, PTT, PT, Arterial_pH,
    #          paO2, paCO2, Arterial_BE, HCO3, Arterial_lactate, SOFA, SIRS,
    #          Shock_Index, PaO2_FiO2, cumulated_balance
    # collog: SpO2, BUN, Creatinine, SGOT, SGPT, Total_bili, INR,
    #         input_total, input_4hourly, output_total, output_4hourly
    obs_cols = [c for c in shifted_trajs[0].columns if c.startswith('o:')]
    state_dim = len(obs_cols)
    print(f"  → State dimension: {state_dim}")
    assert state_dim == 47, f"Expected 47 state features, got {state_dim}"
    
    N = len(shifted_trajs)
    
    # Initialize tensors
    statevecs = torch.zeros(N, max_len, state_dim)
    actions = torch.zeros(N, max_len, dtype=torch.long)
    subactions = torch.zeros(N, max_len, 2, dtype=torch.long)
    subactionvecs = torch.zeros(N, max_len, 10)
    rewards = torch.zeros(N, max_len)
    returns_to_go = torch.zeros(N, max_len)
    notdones = torch.zeros(N, max_len)
    timesteps = torch.zeros(N, max_len, dtype=torch.long)
    lengths = torch.zeros(N, dtype=torch.long)
    
    for i, traj in enumerate(shifted_trajs):
        L = len(traj)
        lengths[i] = L
        
        # States (observation columns only)
        state_vals = traj[obs_cols].values.astype(np.float32)
        statevecs[i, :L] = torch.from_numpy(state_vals)
        
        # Actions
        act_vals = traj['a:action'].values.astype(np.int64)
        actions[i, :L] = torch.from_numpy(act_vals)
        
        # Sub-actions (vaso = action % 5, iv = action // 5)
        vaso_idx = act_vals % 5        # 0-4
        iv_idx = act_vals // 5         # 0-4
        subactions[i, :L, 0] = torch.from_numpy(vaso_idx)
        subactions[i, :L, 1] = torch.from_numpy(iv_idx)
        
        # Sub-action one-hot vectors (10-dim: first 5 vaso, last 5 iv)
        for t in range(L):
            subactionvecs[i, t, vaso_idx[t]] = 1.0
            subactionvecs[i, t, 5 + iv_idx[t]] = 1.0
        
        # Rewards — STORED EXPLICITLY as raw per-step values
        # make_traj_shifted already shifted the reward column via df['r:reward'].shift(-1)
        rewards[i, :L] = torch.from_numpy(traj['r:reward'].values.astype(np.float32))
        
        # Returns-to-go (undiscounted cumsum, γ=1 for sparse terminal reward)
        rtg = np.cumsum(traj['r:reward'].values[::-1])[::-1].copy()
        returns_to_go[i, :L] = torch.from_numpy(rtg.astype(np.float32))
        
        # Not-done flags (0 at terminal, 1 otherwise)
        nd = np.ones(L, dtype=np.float32)
        nd[-1] = 0.0  # Last step is terminal
        notdones[i, :L] = torch.from_numpy(nd)
        
        # Timesteps
        timesteps[i, :L] = torch.arange(L, dtype=torch.long)
    
    # ──── Step 4b: Estimate behavior policy π_b via k-NN ────
    # CRITICAL: Without proper π_b estimation, WIS evaluation is meaningless.
    print("  Estimating π_b via k-NN (this may take a few minutes)...")
    pibs, subpibs, estm_subpibs = estimate_behavior_policy(
        statevecs, actions, subactions, lengths, k=50
    )
    
    # ──── Step 5: Save ────
    print("=" * 60)
    print("STEP 5/5: Saving trajectory tensors...")
    print("=" * 60)
    
    output = {
        'statevecs': statevecs,
        'actions': actions,
        'subactions': subactions,
        'subactionvecs': subactionvecs,
        'rewards': rewards,
        'returns_to_go': returns_to_go,
        'notdones': notdones,
        'timesteps': timesteps,
        'lengths': lengths,
        'pibs': pibs,
        'subpibs': subpibs,
        'estm_subpibs': estm_subpibs,
    }
    
    output_path = f'{data_dir}/trajectories_shifted.pt'
    torch.save(output, output_path)
    
    print(f"\n✅ Saved to {output_path}")
    print(f"   {N} trajectories")
    print(f"   {int(lengths.sum().item())} total timesteps")
    print(f"   State dim: {state_dim}")
    print(f"   Max length: {max_len}")
    print(f"   Actions: 25 (5 vaso × 5 IV)")

if __name__ == '__main__':
    main()
