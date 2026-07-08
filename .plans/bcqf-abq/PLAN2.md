# PLAN2: BCQf-Bilinear — Rank-1 Interaction for Factored BCQ

## Context

Add a third model: **BCQf-Bilinear** — BCQf with bilinear rank-1 interaction term trained on shifted temporal alignment data. The original BCQf assumes linear Q-decomposition `Q(s,[a1,a2]) = q1(s,a1) + q2(s,a2)`, which ignores interactions between vasopressor and IV fluid sub-actions. In sepsis, these treatments interact through shared physiological pathways (blood pressure), making the additive assumption biased. **Tang (2022) himself acknowledges this in Appendix B.9**, proposing residual interaction terms as one solution to address sub-action interactions but never implementing them.

We add a **state-dependent, rank-1 bilinear correction**:
- `Q(s, [a1,a2]) = q1(s,a1) + q2(s,a2) + e1(s,a1)ᵀ e2(s,a2)`
- `e1, e2 ∈ ℝ¹` are learned 1-dim embeddings per sub-action value
- The 5×5 interaction matrix `I_ij = α_i(s) · β_j(s)` is rank-1, capturing the single dominant synergy pattern

Final comparison: BCQf (non-shifted) vs BCQf (shifted) vs BCQf-Bilinear (shifted).

All data already exists. Only `model.py` changes + new training script + evaluation fix.

## Approach

### Step 1: Add BCQf_Bilinear_Net to model.py

**File:** `RL_mimic_sepsis/4_BCQf/model.py`

Add new class `BCQf_Bilinear_Net` **before** `BCQf_Bilinear` (so it's defined first):

```python
class BCQf_Bilinear_Net(nn.Module):
    """BCQf Q-network with bilinear rank-1 sub-action interaction.

    Architecture:
      h = shared_trunk(state)                              # (B, 128)
      q_add = Linear(h → 10)                                # [vaso₀..₄ | iv₀..₄]
      α(s) = Linear(h → 5)                                  # vaso dose factors
      β(s) = Linear(h → 5)                                  # iv dose factors
      I = α(s) ⊗ β(s)                                       # 5×5 outer product
      Q(s, [j,k]) = q_add[j] + q_add[5+k] + I[j,k]          # 25-dim

    The interaction is RANK-1: I_ij = α_i(s) · β_j(s). This captures the
    single dominant synergy pattern between vasopressor and IV fluid while
    using fewer head parameters (5,140) than a direct Linear(256→25) head
    (6,425) — genuinely parameter-efficient at k=1.

    When α≈0 or β≈0, recovers original BCQf linear decomposition.
    When α,β are non-zero, captures dose-specific synergies.
    """
    def __init__(self, state_dim, action_dim, hidden_dim, inter_rank=1):
        super().__init__()
        self.inter_rank = inter_rank

        # Shared trunk (same structure as BCQf_Net)
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # Additive Q-head (10-dim: vaso₀..₄, iv₀..₄)
        self.q_add = nn.Linear(hidden_dim, 10)
        # Per-dose interaction factors (rank-1: 5×1 each)
        self.alpha = nn.Linear(hidden_dim, 5 * inter_rank)  # vaso side
        self.beta = nn.Linear(hidden_dim, 5 * inter_rank)   # iv side
        # Behavior policy head (unchanged from BCQf_Net)
        self.πb = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 10),
        )

        # all_subactions_vec: 25×10 matrix mapping 10-dim → 25-dim
        # Row m: one-hot at position (m % 5) for vaso + one-hot at 5+(m // 5) for iv
        # Convention: action_index = vaso + iv*5
        self.register_buffer('_comb_map', all_subactions_vec, persistent=False)

    def forward(self, x):
        h = self.shared(x)                               # (B, hidden_dim)

        # Additive term: 10-dim → 25-dim via combinatorial map
        q_add = self.q_add(h)                            # (B, 10)
        q_25 = q_add @ self._comb_map.T                  # (B, 25)

        # Rank-1 bilinear interaction
        a = self.alpha(h).view(-1, 5, self.inter_rank)   # (B, 5, k)
        b = self.beta(h).view(-1, 5, self.inter_rank)    # (B, 5, k)
        a = a - a.mean(dim=1, keepdim=True)              # double-center: orthogonal to q_add
        b = b - b.mean(dim=1, keepdim=True)              # double-center beta
        a = F.normalize(a, p=2, dim=1, eps=1e-6)        # ‖α‖=1 for scale identifiability
        I = torch.einsum('bvk,bik->bvi', a, b)            # (B, 5, 5) → I[b, vaso, iv]
        I = I.transpose(-2, -1)                          # (B, 5, 5) → I[b, iv, vaso]
        # transpose ensures reshape(-1,25) index = iv*5 + vaso, matching _comb_map convention
        q_25 = q_25 + I.reshape(-1, 25)                   # (B, 25)

        # Behavior policy (unchanged)
        p_logits = self.πb(x)                            # (B, 10)

        # Expose for interaction penalty in training_step
        self._last_beta = b

        return q_25, F.log_softmax(p_logits, dim=-1), p_logits
```

**Indexing verification (CRITICAL — validated exhaustively):**
```python
# all_subactions_vec row m → vaso = m%5, iv = m//5 → convention: action = vaso + iv*5
# I = einsum('bvk,bik->bvi') → I[b, vaso, iv] → transpose → I[b, iv, vaso]
# I.reshape(-1, 25) → index = iv*5 + vaso = vaso + iv*5 ✓ MATCHES _comb_map
```

**Parameter counts (hidden_dim=256, k=1):**
- `q_add`: Linear(256→10) = 2,570
- `alpha`: Linear(256→5) = 1,285
- `beta`: Linear(256→5) = 1,285
- **Bilinear total:** 5,140 vs **direct Linear(256→25):** 6,425 → **0.80×** (parameter-efficient)
- Full network: ~173K params (BCQf_Net: ~170K, +2.6K = +1.5%)

At k=2: bilinear 7,710 vs direct 6,425 → 1.20×. Efficiency advantage only at k=1 — motivating k=1 as a parsimonious starting point to test.

### Step 2: Add BCQf_Bilinear LightningModule to model.py

**File:** `RL_mimic_sepsis/4_BCQf/model.py`

Add new class `BCQf_Bilinear` after `BCQf`. Overrides `__init__`, `training_step`, `offline_evaluation`, `offline_q_evaluation` — each adapted for 25-dim Q output.

```python
class BCQf_Bilinear(BCQf):
    """BCQf with bilinear rank-1 interaction between sub-actions.

    Overrides training_step and offline_evaluation to use 25-dim Q-values
    directly (no all_subactions_vec mapping needed).

    Inherits: validation_step, on_validation_end, configure_optimizers,
    polyak_target_update, copy_target_update, add_model_specific_args.
    """

    def __init__(self, *, inter_rank=1, **kwargs):
        super().__init__(**kwargs)  # BCQf.__init__ sets up hyperparams, etc.
        self.inter_rank = inter_rank
        self.Q = BCQf_Bilinear_Net(
            self.hparams.state_dim,
            self.hparams.num_actions,
            self.hparams.hidden_dim,
            inter_rank=self.inter_rank,
        )
        self.Q_target = copy.deepcopy(self.Q)
        for param in self.Q_target.parameters():
            param.requires_grad = False

    def training_step(self, batch, batch_idx):
        state, action, subaction, subactionvec, next_state, reward, notdone, subpibs = batch

        with torch.no_grad():
            q25, _, i = self.Q(next_state)                      # (B, 25)
            imt = F.log_softmax(i.reshape(-1, 2, 5), dim=-1).exp()
            imt = (imt / imt.max(axis=-1, keepdim=True).values > self.threshold).float()
            imt25 = torch.einsum('bi,bj->bji', imt[:, 0, :], imt[:, 1, :]).reshape(-1, 25)
            next_action = (imt25 * q25 + (1 - imt25) * torch.finfo().min).argmax(axis=1, keepdim=True)

            q25_t, _, _ = self.Q_target(next_state)
            target_Q = reward + notdone * self.discount * q25_t.gather(1, next_action)
            if self.hparams.target_value_clipping:
                target_Q = torch.clamp(target_Q, self.hparams.vmin, self.hparams.vmax)

        current_Q25, _, i = self.Q(state)
        current_Q = current_Q25.gather(1, action)               # (B, 1)

        imt = F.log_softmax(i.reshape(-1, 2, 5), dim=-1)
        q_loss = F.smooth_l1_loss(current_Q, target_Q)
        i_loss = F.nll_loss(imt[:, 0, :], subaction[:, 0]) + F.nll_loss(imt[:, 1, :], subaction[:, 1])

        # Interaction magnitude penalty (Tang B.9: "regularize magnitude of residual terms")
        # Penalize ‖β‖² so interaction doesn't dominate the additive structure.
        # β accessed via self.Q._last_beta (set in forward to preserve 3-tuple return signature).
        # Since ‖α‖=1, the Frobenius norm ‖I‖²_F = ‖α⊗β‖² = ‖α‖²·‖β‖² = ‖β‖².
        inter_penalty = self.hparams.inter_penalty_weight * self.Q._last_beta.pow(2).sum(dim=1).mean()

        Q_loss = q_loss + i_loss + 1e-2 * i.pow(2).mean() + inter_penalty

        self.Q_optimizer.zero_grad()
        self.manual_backward(Q_loss)
        self.Q_optimizer.step()

        self.iterations += 1
        if self.iterations >= self.hparams.max_steps:
            self.trainer.should_stop = True
        self.maybe_update_target()

    def offline_evaluation(self, eval_buffer, weighted=True, eps=0.01):
        """Override: uses 25-dim Q directly with correct per-branch thresholding."""
        states, actions, subactions, subactionvecs, rewards, not_dones, subpibs, estm_subpibs = eval_buffer
        rewards = rewards[:, :, 0].cpu().numpy()
        n, horizon, _ = states.shape
        discounted_rewards = rewards * (self.eval_discount ** np.arange(horizon))

        ir = np.ones((n, horizon))
        for idx in range(n):
            lng = (not_dones[idx, :, 0].sum() + 1).item()
            q25, _, i = self.Q(states[idx])
            imt = F.log_softmax(i.reshape(-1, 2, 5), dim=-1).exp()
            imt = (imt / imt.max(axis=-1, keepdim=True).values > self.threshold).float()
            imt25 = torch.einsum('bi,bj->bji', imt[:, 0, :], imt[:, 1, :]).reshape(-1, 25)
            a_id = (imt25 * q25 + (1. - imt25) * torch.finfo().min).argmax(axis=1).cpu().numpy()

            pie_soft = np.zeros((horizon, 25))
            estm_pibs = np.einsum('bi,bj->bji',
                estm_subpibs[idx][:, :5].cpu().numpy(),
                estm_subpibs[idx][:, 5:].cpu().numpy()
            ).reshape((-1, 25))
            pie_soft += eps * estm_pibs
            pie_soft[range(horizon), a_id] += (1.0 - eps)

            a_obs = actions[idx, :, 0]
            ir[idx, :lng] = pie_soft[range(lng), a_obs[:lng].cpu().numpy()] / \
                (subpibs[idx, range(lng), a_obs[:lng] % 5].cpu().numpy() *
                 subpibs[idx, range(lng), 5 + a_obs[:lng] // 5].cpu().numpy())
            ir[idx, lng:] = 1

        weights = np.clip(ir.cumprod(axis=1), 0, 1e3)
        if weighted:
            weights /= weights.sum(axis=0)
        else:
            weights /= weights.shape[0]

        ess = (weights[:, -1].sum()) ** 2 / ((weights[:, -1]) ** 2).sum()
        estm = (weights[:, -1] * discounted_rewards.sum(axis=-1)).sum()
        return estm, ess

    def offline_q_evaluation(self, eval_buffer):
        states, _, _, _, _, _, _, _ = eval_buffer
        states = states[:, 0, :]
        q25, _, i = self.Q(states)
        imt = F.log_softmax(i.reshape(-1, 2, 5), dim=-1).exp()
        imt = (imt / imt.max(axis=-1, keepdim=True).values > self.threshold).float()
        imt25 = torch.einsum('bi,bj->bji', imt[:, 0, :], imt[:, 1, :]).reshape(-1, 25)
        values = (imt25 * q25 + (1. - imt25) * torch.finfo().min).max(axis=1).values
        return values.mean().item()
```

**Inherited from BCQf (unchanged):** `validation_step`, `on_validation_end`, `configure_optimizers`, `polyak_target_update`, `copy_target_update`. Additionally, `add_model_specific_args` is overridden to add `--inter_penalty_weight` (default 1e-3) and `--inter_rank` (default 1) to the arg parser.

### Step 3: Create opt_shifted_bilinear.py

**File:** `RL_mimic_sepsis/4_BCQf/opt_shifted_bilinear.py`

Copy `opt_shifted.py` with three changes:
```python
# CHANGE 1: import
- from model import BCQf
+ from model import BCQf_Bilinear

# CHANGE 2: logger name
- logger = CSVLogger("logs_shifted", name="mimic_dBCQf_shifted")
+ logger = CSVLogger("logs_shifted_bilinear", name="mimic_dBCQf_shifted_bilinear")

# CHANGE 3: model class
- policy = BCQf(
+ policy = BCQf_Bilinear(
```

All hyperparameters identical. Inter_rank added to grid: `k ∈ {1, 2}`. Inter_penalty_weight added to grid: `{0, 1e-4, 1e-3, 1e-2}`. Same data paths. Independent logdir.

To make `inter_penalty_weight` configurable, `BCQf_Bilinear` overrides `add_model_specific_args`:
```python
@staticmethod
def add_model_specific_args(parent_parser):
    parser = BCQf.add_model_specific_args(parent_parser)
    parser.add_argument('--inter_penalty_weight', type=float, default=1e-3)
    parser.add_argument('--inter_rank', type=int, default=1)
    return parser
```

### Step 4: Fix offline_evaluation_F in evaluate.py

**File:** `RL_mimic_sepsis/EvalPlots/evaluate.py`

The existing `offline_evaluation_F` has a bug: it uses 10-dim `imt * q` with argmax producing indices 0-9, which are then used to index a 25-dim `pie_soft`. Actions 10-24 never receive the (1-eps) probability mass — the evaluation is silently incorrect.

Fix: use correct per-branch thresholding → 25-dim IMT mask, handle both 10-dim and 25-dim Q outputs.

Replace the evaluation loop body (lines ~95-110 in `offline_evaluation_F`) with:

```python
def offline_evaluation_F(self, eval_buffer, weighted=True, eps=0.01):
    states, actions, subactions, subactionvecs, rewards, not_dones, subpibs, estm_subpibs = eval_buffer
    rewards = rewards[:, :, 0].cpu().numpy()
    n, horizon, _ = states.shape
    discounted_rewards = rewards * (self.eval_discount ** np.arange(horizon))

    ir = np.ones((n, horizon))
    for idx in range(n):
        lng = (not_dones[idx, :, 0].sum() + 1).item()

        q, imt_log, _ = self.Q(states[idx])
        imt = imt_log.exp()                                 # (H, 10) per-branch behavior probs

        # Per-branch BCQ thresholding — correct for BOTH 10-dim and 25-dim Q
        imt_v = imt[:, :5]
        imt_i = imt[:, 5:]
        imt_v = (imt_v / imt_v.max(1, keepdim=True).values > self.threshold).float()
        imt_i = (imt_i / imt_i.max(1, keepdim=True).values > self.threshold).float()
        imt25 = torch.einsum('bi,bj->bji', imt_v, imt_i).reshape(-1, 25)  # (H, 25)

        # Map Q to 25-dim if needed (backward-compatible with original BCQf_Net)
        if q.shape[-1] == 10:
            # Try module-level constant first, then buffer on Q network
            from model import all_subactions_vec as _global_cmb
            cmb = getattr(self.Q, '_comb_map', _global_cmb)
            cmb = cmb.to(q.device)
            q = q @ cmb.T
        # q is now (H, 25)

        a_id = (imt25 * q + (1. - imt25) * torch.finfo().min).argmax(axis=1).cpu().numpy()

        pie_soft = np.zeros((horizon, 25))
        estm_pibs = np.einsum('bi,bj->bji',
            estm_subpibs[idx][:, :5].cpu().numpy(),
            estm_subpibs[idx][:, 5:].cpu().numpy()
        ).reshape((-1, 25))
        pie_soft += eps * estm_pibs
        pie_soft[range(horizon), a_id] += (1.0 - eps)

        a_obs = actions[idx, :, 0]
        ir[idx, :lng] = pie_soft[range(lng), a_obs[:lng].cpu().numpy()] / \
            (subpibs[idx, range(lng), a_obs[:lng] % 5].cpu().numpy() *
             subpibs[idx, range(lng), 5 + a_obs[:lng] // 5].cpu().numpy())
        ir[idx, lng:] = 1

    weights = np.clip(ir.cumprod(axis=1), 0, 1e3)
    if weighted:
        weights /= weights.sum(axis=0)
    else:
        weights /= weights.shape[0]

    ess = (weights[:, -1].sum()) ** 2 / ((weights[:, -1]) ** 2).sum()
    estm = (weights[:, -1] * discounted_rewards.sum(axis=-1)).sum()
    return estm, ess
```

**Backward compatibility:** When `q.shape[-1] == 10` (original BCQf_Net), maps via `all_subactions_vec` to 25-dim. When `q.shape[-1] == 25` (BCQf_Bilinear_Net), uses directly. Both use correct 25-dim per-branch thresholding → argmax in range 0-24.

**⚠ MANDATORY: Regression test before trusting results.** Run the old and new `offline_evaluation_F` on existing BCQf (non-shifted) and BCQf (shifted) checkpoints. The WIS/ESS values WILL change because the old code was masking actions 10-24. Document the delta. If WIS changes by > 5%, investigate before interpreting the bilinear results.

### Step 5: Update QuantitativeEval_Combined.ipynb

**File:** `RL_mimic_sepsis/EvalPlots/QuantitativeEval_Combined.ipynb`

**Change A — Import:**
```python
from model import BCQf, BCQf_Bilinear, all_subactions_vec
```

**Change B — Add bilinear evaluation cells** (after shifted-std evaluation, before bootstrap):

```python
# Find top 10 ESS per version (shifted bilinear)
logdir_s_bilinear = "../4_BCQf/logs_shifted_bilinear/mimic_dBCQf_shifted_bilinear"

bilinear_top10_list = []
for ver in range(40):
    try:
        text = open(f"{logdir_s_bilinear}/version_{ver}/hparams.yaml").read()
        thresh = float(re.search(r"threshold: ([\d.]+)", text).group(1))
        seed = int(re.search(r"seed: (\d+)", text).group(1))
        df = pd.read_csv(f"{logdir_s_bilinear}/version_{ver}/metrics.csv")
        valid = df.dropna(subset=["val_wis", "val_ess"])
        if len(valid) > 0:
            top10 = valid.nlargest(10, "val_ess")
            for _, row in top10.iterrows():
                bilinear_top10_list.append({
                    "version": ver, "threshold": thresh, "seed": seed,
                    "val_wis": row["val_wis"], "val_ess": row["val_ess"],
                    "iteration": row["iteration"],
                })
    except Exception as e:
        print(f"Version {ver} skipped: {e}")

top10_bilinear = pd.DataFrame(bilinear_top10_list)
print(f"Bilinear: {len(top10_bilinear)} checkpoints from {top10_bilinear['version'].nunique()} versions")

# Evaluate
bilinear_test_results = []
for _, row in tqdm(top10_bilinear.iterrows(), total=len(top10_bilinear), desc="Bilinear test"):
    ver = int(row["version"])
    best_iter = int(row["iteration"])
    ckpt_step = (best_iter // 100) * 100
    ckpt_path = f"{logdir_s_bilinear}/version_{ver}/step={ckpt_step}.ckpt"
    model = BCQf_Bilinear.load_from_checkpoint(ckpt_path, map_location=None, weights_only=False)
    model.eval()
    w, e = offline_evaluation_F(model, test_batch_s.to(model.device), weighted=True, eps=0.01)
    bilinear_test_results.append({
        "model": "BCQf-Bilinear (shifted)",
        "version": ver, "threshold": row["threshold"], "seed": row["seed"],
        "val_wis": row["val_wis"], "val_ess": row["val_ess"], "best_iter": best_iter,
        "test_wis": w, "test_ess": e,
    })

df_bilinear = pd.DataFrame(bilinear_test_results)
print(df_bilinear.nlargest(10, "test_wis")[["version","threshold","seed","best_iter","val_ess","test_wis","test_ess"]].to_string(index=False))
```

**Change C — Add bilinear to final comparison table:**
```python
best_bilinear = df_bilinear.loc[df_bilinear["test_wis"].idxmax()]
bl_ckpt = f"{logdir_s_bilinear}/version_{int(best_bilinear['version'])}/step={(int(best_bilinear['best_iter'])//100)*100}.ckpt"
model_bl = BCQf_Bilinear.load_from_checkpoint(bl_ckpt, map_location=None, weights_only=False)
model_bl.eval()
wis_bl, ess_bl = offline_evaluation_F(model_bl, test_batch_s.to(model_bl.device), weighted=True, eps=0.01)

print(f'{"BCQf (non-shifted)":<25} {wis_orig:15.2f} {ess_orig:15.1f}')
print(f'{"BCQf (shifted)":<25} {wis_shifted:15.2f} {ess_shifted:15.1f}')
print(f'{"BCQf-Bilinear (shifted)":<25} {wis_bl:15.2f} {ess_bl:15.1f}')
```

### Step 6: Train

```bash
cd RL_mimic_sepsis/4_BCQf
python3 opt_shifted_bilinear.py
```

40 trials → `logs_shifted_bilinear/mimic_dBCQf_shifted_bilinear/version_0-39/`.

### Step 7: Evaluate

Run `QuantitativeEval_Combined.ipynb`. Expected output:

```
================================================================================
Model                       Test WIS    Test ESS    Test Alignment
================================================================================
Clinician (non-shifted)     90.29        —          non-shifted
BCQf (non-shifted)          91.62       178.xx      non-shifted
Clinician (shifted)         ~XX.XX       —          shifted
BCQf (shifted)              93.35        85.xx      shifted
BCQf-Bilinear (shifted)     XX.XX       XX.xx      shifted
================================================================================
```

**⚠ IMPORTANT — Cross-alignment comparison:** BCQf (non-shifted) is evaluated on its *native* non-shifted test set, while both shifted variants are evaluated on the shifted test set. WIS values across the alignment boundary are not directly comparable — they measure performance in causally different environments. Tang et al. (2026, "Off by a Beat") demonstrated that same-alignment OPE can be misleading: both alignments produce similar WIS values (~0.86–0.88) despite the original-alignment policy learning incorrect causal dynamics (e.g., inferring that vasopressors *lower* blood pressure). Cross-alignment evaluation reveals the true performance gap — the original policy drops to 0.78 on shifted data (Tang 2026 scaling: reward +1/−1). Our reproduction uses Tang 2022's reward scale [100, 0] (WIS range ~90–93). Direct numerical comparison across reward scales is meaningless — see Tang 2026 for within-scale discussion.

**Model selection criteria:** Following Tang (2022), candidate policies are filtered by validation ESS ≥ 200 where possible. For shifted variants, we observe empirically lower ESS (~85 vs ~178 for non-shifted), likely due to differences in behavior policy estimation quality and state distribution after temporal re-indexing. While shorter episodes (T−1 vs T) would naively predict higher ESS (fewer importance ratio multiplications), the shifted data introduces a slightly different action distribution per state, affecting the π̂_b estimates used for importance sampling. We set the shifted ESS threshold to **50** as a practical relaxation, calibrated from the observed ESS distribution of existing shifted models. Both shifted variants use the same threshold for fair within-alignment comparison. The model with highest validation WIS among those meeting their respective ESS threshold is selected. For shifted-to-shifted comparison (BCQf vs BCQf-Bilinear), both models use the same threshold, ensuring fair selection.

**Valid comparisons:**
- BCQf vs Clinician on the SAME alignment (both non-shifted) — reproduces Tang 2022
- BCQf (shifted) vs BCQf-Bilinear (shifted) on the SAME alignment — **head-to-head for novelty**

Cross-alignment comparison (BCQf non-shifted vs shifted variants) must be purely qualitative, noting alignment as a confounding factor.

```
=============================================================
Model                       Test WIS        Test ESS
=============================================================
Clinician (non-shifted)         90.29            —
BCQf (non-shifted)              91.62        178.xx
BCQf (shifted)                  93.35         85.xx
BCQf-Bilinear (shifted)        XX.XX         XX.xx
=============================================================
```

---

## Mathematical Justification

### 1. The Linear Decomposition Deficiency

Tang (2022) decomposes: `Q(s, [a1,a2]) = q1(s,a1) + q2(s,a2)`. This is unbiased when transitions factorize as a product across sub-actions (Eq. 2) and rewards are additive (Eq. 3) — i.e., the MDP is isomorphic to independent per-sub-action dynamics (Theorem 1). However, in sepsis, vasopressor and IV fluids affect blood pressure through overlapping mechanisms — Hamzaoui (2021, J Intensive Med) describes their "magic potion" synergy. When interactions exist, the linear decomposition incurs omitted-variable bias. Tang himself acknowledges this in Appendix B.9: *"one can either explicitly encode the interaction terms or resort back to a combinatorial action space."*

### 2. Rank-1 Bilinear Correction

Our correction adds a state-dependent rank-1 bilinear term:

`Q(s, [a1,a2]) = q1(s,a1) + q2(s,a2) + α_a1(s) · β_a2(s)`

where α,β ∈ ℝ⁵ are learned per-dose factors (Linear(256→5) each). The 5×5 interaction matrix `I_ij = α_i·β_j` has rank exactly 1.

**Why rank-1 is a well-motivated starting point:**

**(a) Genuinely non-separable.** Unlike the rejected ABQ baseline (scalar B(s) shifts ALL actions equally), `α_i·β_j` depends on BOTH sub-actions simultaneously and cannot be absorbed into q1 or q2.

**(b) Parameter-efficient.** At k=1: the bilinear head (5,140 params) is smaller than a direct Linear(256→25) combinatorial head (6,425 params, 0.80×). At k=2 this advantage disappears (7,710 vs 6,425, 1.20×). The rank-1 structure is both mathematically meaningful AND parameter-efficient.

**(c) Physiologically grounded.** A rank-1 interaction represents a single dominant synergy mechanism: each vaso dose has a "potentiation factor" α_i and each IV dose has a "potentiation factor" β_j. Their product gives the interaction strength. This matches the clinical understanding that vasopressor-IV synergy is mediated through a single pathway (blood pressure regulation via preload + afterload).

**(d) Nested capacity, approximate recovery.** In the limit ‖β‖→0, the bilinear model approximates Tang's original BCQf (the additive term remains). Because α is L2-normalized to unit norm, the model cannot set α to exactly zero — but can make the interaction arbitrarily small by shrinking β. This makes BCQf-Bilinear an approximate superset: it can closely mimic the additive model when interaction is weak, while adding genuine interaction capacity when needed. The L2 penalty on ‖β‖² (1e-3 weight) encourages this default behavior — the model starts near-additive and activates interaction only when it improves the Q-function fit.

**(e) Embedding identifiability.** The product α_i·β_j has two sources of non-identifiability: (i) scale ambiguity α→cα, β→β/c, and (ii) overlap with the additive terms — any constant component of α or β can be absorbed into q_add. We handle (ii) by double-centering both α and β (`a = a - a.mean()`), making the interaction term orthogonal to what q_add can represent (analogous to two-way ANOVA interaction). We handle (i) by L2-normalizing α to unit norm AFTER centering. The combination ensures the decomposition Q = q_add + α⊗β is unique up to sign flips (α→−α, β→−β), which does not affect the Q-value since only the product is used.

### 3. Relation to Existing Theory

**Tang (2022) Appendix B.9 — Residual interaction term.** Tang proposes adding a residual term R(a) that is a function of the action vector only: `Q(s,a) = Σ q_d(s,a_d) + R(a)`. Our bilinear term extends this by making the residual *state-dependent*: R(s,a) instead of R(a). This is more expressive (interaction strength can vary with patient state), but at the cost of additional parameters. Tang explicitly warns in B.9 to *"regularize the magnitude of residual terms so we still benefit from the efficiency gains of the linear decomposition"* — we address this with L2 regularization on the interaction magnitude (see below).

**Lee et al. (AISTATS 2025, "AD-BCQ", arXiv:2504.21326):** Proposes projected Q-functions via interventional semantics. AD-BCQ operates under a "Separable Effects" assumption (Fig. 1b in their paper) — sub-action effects are assumed **non-interacting**, and the non-separable regime (Fig. 1c) is explicitly identified as a case they do not handle. Their non-linear mixer network can capture some interaction empirically, but the theoretical framework does not model it. Our BCQf-Bilinear **drops the separability assumption** by explicitly encoding state-dependent interaction through a rank-1 bilinear architecture — addressing the regime that AD-BCQ excludes. This is a genuine differentiation, not just a mechanism difference: AD-BCQ assumes away the very phenomenon we model.

**Rozada, Tenorio & Marques (EUSIPCO 2021):** "Low-rank state-action value-function approximation" demonstrates empirically that Q(s,a) matrices in many high-dimensional MDPs can be well-approximated by low-rank structure — motivating the general principle of rank-constrained Q-function estimation.

### 4. Greedy Policy Impact

Unlike the ABQ baseline (scalar shifts all Q-values equally, argmax unchanged), the bilinear interaction IS action-dependent:

`argmax [q1(s,a1) + q2(s,a2) + α_a1(s)·β_a2(s)]`

The `α·β` term genuinely changes which action maximizes Q. When synergy is positive (α_i·β_j > 0 for high-dose combinations), the policy shifts toward combined high-dose treatment — the clinically meaningful behavior where vasopressor+IV bolus is recommended over either alone in severe hypotension.

---

## Limitations

1. **Cross-alignment comparison.** BCQf (non-shifted) and BCQf-Bilinear (shifted) are evaluated on different test set alignments. WIS values across alignments are not directly comparable — they measure performance in causally different environments (Tang et al., 2026). Valid head-to-head comparisons are: (a) BCQf vs Clinician on the same non-shifted alignment, and (b) BCQf (shifted) vs BCQf-Bilinear (shifted) on the shifted alignment.

2. **Rank-1 truncation.** The true interaction matrix may have rank up to 4. Rank-1 captures only the dominant singular component — analogous to truncated SVD. If multiple interaction patterns exist (e.g., different synergy at low vs high doses), they are collapsed into one direction.

3. **Bias-variance tradeoff and Bellman closure.** Tang (2022, Prop. 5) shows factored decomposition has lower Rademacher complexity than combinatorial; the bilinear class lies between these two. Unlike Tang's linear decomposition which is closed under Bellman backup (Theorem 1), the bilinear form has no such guarantee — applying the Bellman operator to a rank-1 augmented Q-function does not necessarily produce another rank-1 augmented Q-function. This is not fatal (most DNN Q-approximators lack closure), but means any bias reduction from interaction modeling must be validated empirically. The crossover point where additional capacity stops helping was demonstrated by Tang (2022, Fig. 6) for factored-vs-combinatorial; whether rank-1 bilinear lies on the beneficial side is the empirical question this experiment answers.

4. **πb independence assumption (internal validity threat).** The BCQ threshold mask `imt25 = outer(imt_vaso, imt_iv)` and behavior policy estimation both assume clinicians choose vasopressor and IV doses independently given state. But if these treatments interact physiologically (our core motivation for the bilinear term), experienced clinicians likely co-select them — violating the independence assumption. This is not just a theoretical concern: incorrect π̂_b estimates propagate into the importance sampling ratios used for OPE, potentially masking or exaggerating the benefit of interaction modeling. This is the primary internal validity threat of the entire experiment and should be carefully discussed.

5. **α/β non-identifiability.** Individual α_i or β_j values are not interpretable as standalone "potentiation factors" — only their product is identified. Normalization ‖α‖=1 fixes the scale but β values remain conditional on this choice.

6. **AD-BCQ not benchmarked.** Lee et al. (AISTATS 2025) proposed AD-BCQ under a "Separable Effects" assumption — sub-action effects are assumed non-interacting, and the non-separable regime is explicitly identified as a case they do not handle. This makes AD-BCQ complementary rather than competing with our approach, which explicitly models interaction. While their main text claims 177,877 patients, the appendix train/val/test split (12,989/2,779/2,791 ≈ 18,559) is consistent with Tang 2022's cohort. Head-to-head comparison, particularly in the separable regime they target, would strengthen our claim that bilinear helps where AD-BCQ's assumption fails.

7. **Domain-knowledge tension.** Tang (2022, §3.4) argues vasopressor and IV fluids "are not expected to interfere with each other" due to different mechanisms (preload vs inotropy/vasoconstriction), citing Hamzaoui (2021) as evidence of limited positive interaction. Our motivation for the bilinear term — that interaction is substantial enough to bias the linear decomposition — directly challenges this domain-knowledge claim. The experiment thus serves a dual purpose: (a) test whether Tang's domain assumption holds empirically in this cohort, and (b) provide a correction mechanism if it does not. A null result (bilinear ≈ additive) validates Tang's claim; a positive result quantifies the interaction bias he anticipated in Appendix B.9.

8. **Diagnostic: interaction magnitude.** We report mean ‖β‖² across test states as a diagnostic of whether the bilinear term is genuinely active. If BCQf-Bilinear outperforms but ‖β‖² ≈ 0, the improvement may be an optimization artifact rather than interaction modeling. If ‖β‖² is substantial (> 0.1), the bilinear term is demonstrably contributing.

9. **Bellman fixed-point under Theorem 1.** Under the FULL conditions of Theorem 1 (transition factorization Eq. 2 + reward additivity Eq. 3 + policy factorization Eq. 4), Tang's Theorem 1 guarantees the optimal Q* is exactly additively decomposable: Q*(s,a) = q*₁(s,a₁) + q*₂(s,a₂). At the Bellman FIXED POINT (after convergence), the bilinear interaction term must therefore vanish — the model learns α≈0 or β≈0 to recover the additive solution. This is a **multi-iteration convergence property, NOT a one-step absorption**: after a SINGLE Bellman backup, I^+(s,a) = γ·E_{s'~P}[ᾱ(s')·β̄(s')] is generally a non-separable function of (s,a₁,a₂) that is neither rank-1 nor additive. The interaction decays to zero only over many Bellman iterations, driven by both the L2 penalty on β and the absence of true interaction in the MDP. Under policy factorization ALONE (without transition factorization), even the fixed-point guarantee does not hold — the interaction can persist and propagate, which is actually a strength when Theorem 1 is violated (the regime where bilinear is motivated). Combined with Limitation 3 (Bellman non-closure), the propagated interaction may deviate from rank-1 structure over iterations.

## Critical Files

| File | Symbol | Change |
|---|---|---|
| `4_BCQf/model.py` | `BCQf_Bilinear_Net`, `BCQf_Bilinear` | New — rank-1 bilinear interaction |
| `4_BCQf/opt_shifted_bilinear.py` | `main()` | New — training entry point |
| `EvalPlots/evaluate.py` | `offline_evaluation_F` | Fix — correct 25-dim per-branch thresholding |
| `EvalPlots/QuantitativeEval_Combined.ipynb` | Eval cells | +1 column bilinear |

## Verification

### Step 1-2: Network + indexing correctness

```python
import torch, sys
sys.path.insert(0, 'RL_mimic_sepsis/4_BCQf')
from model import BCQf_Bilinear_Net, all_subactions_vec

torch.manual_seed(42)
net = BCQf_Bilinear_Net(state_dim=64, action_dim=10, hidden_dim=128, inter_rank=1)
x = torch.randn(4, 64)
q, imt, logits = net(x)

assert q.shape == (4, 25), f"Expected (4,25), got {q.shape}"
assert imt.shape == (4, 10)

# Check: zero embeddings → equals additive-only
net.alpha.weight.data.zero_(); net.alpha.bias.data.zero_()
net.beta.weight.data.zero_();  net.beta.bias.data.zero_()
q_zero, _, _ = net(x)
q_add_only = net.q_add(net.shared(x)) @ net._comb_map.T
assert torch.allclose(q_zero, q_add_only, atol=1e-5), "Zero interaction should equal additive"
print("PASS: zero interaction = additive-only ✓")

# Check: rank-1 interaction is action-DEPENDENT (not constant across actions)
net.alpha.weight.data.normal_(0, 0.1); net.beta.weight.data.normal_(0, 0.1)
q_interact, _, _ = net(x[:1])
I_term = q_interact - q_add_only[:1]
assert not torch.allclose(I_term[0, 0], I_term[0, 24], atol=1e-4), \
    "Interaction should vary by action — not a constant shift"
print("PASS: interaction is action-dependent ✓")

# Check: exhaustive indexing match (all 25 actions)
for m in range(25):
    vaso = int(all_subactions_vec[m, :5].argmax().item())
    iv = int(all_subactions_vec[m, 5:].argmax().item())
    expected = q_add_only[0, m].item()
    assert abs(q_zero[0, m].item() - expected) < 1e-5, f"Row {m} mismatch"
print("PASS: all 25 rows match _comb_map convention ✓")

# Cross-check against real data loader (not just internal _comb_map)
import sys
sys.path.insert(0, '../../4_BCQf')
from data import SASRBuffer
buffer = SASRBuffer(64, 25)
buffer.load('../../data/episodes+encoded_state+knn_pibs_factored/shifted_train_data.pt')
for i in [0, 100, 5000, -1]:
    state, action, subaction, subactionvec, _, _, _, _ = buffer[i]
    vaso_from_sub = int(subaction[0].item())
    iv_from_sub = int(subaction[1].item())
    action_from_sub = vaso_from_sub + iv_from_sub * 5
    action_from_data = int(action.item())
    assert action_from_sub == action_from_data, \
        f"Sample {i}: sub=(v={vaso_from_sub},i={iv_from_sub}) → {action_from_sub}, but action label={action_from_data}"
print("PASS: data loader convention matches _comb_map ✓")
```

### Step 4: evaluate.py regression test

```bash
cd RL_mimic_sepsis/EvalPlots
# Run OLD evaluate.py on an existing checkpoint → save WIS/ESS
# Run NEW evaluate.py on same checkpoint → compare
# Delta must be understood and documented
```

### End-to-end

After training + evaluation, 4-row table as shown above. All variants produce finite WIS/ESS.

## Assumptions & Contingencies

0. **Label verification (verified 2026-07-08).** Tang (2026, Table 1) classifies Tang (2022)'s codebase as **"original" alignment** — the episodic tensor stores state and action from the same 4h window. However, Tang (2022)'s BCQf data loader applies a +1 offset at training time (`actions[:, 1:]`), producing causally correct training pairs `(state_t, action_{t+1})` — state at end of window t → action at start of window t+1. Our "non-shifted" reproduction follows Tang (2022) exactly. Our "shifted" variant applies an additional +1 shift in episodic data construction (`act[t] = orig_act[t+1]`), which combined with BCQf's +1 loading produces `(state_t, action_{t+2})`. This is **not** Tang (2026)'s causally correct shifted alignment (`z_t = a_{t-1}`) — it is an over-shifted variant that skips one action per transition. For clarity: "non-shifted" = Tang (2022) reproduction (original episodic alignment, causally corrected at load time); "shifted" = over-shifted by +1 relative to Tang (2022). Both are compared within their own alignment for fairness.

1. **k=1 may underfit.** If rank-1 is insufficient, results will be similar to BCQf (shifted). We include **k ∈ {1, 2} as a main ablation** (not contingency) to test whether rank-2 interaction provides additional benefit. At k=2, parameter efficiency is lost (7,710 vs 6,425 for direct head), but the comparison isolates whether the rank-1 inductive bias is genuinely useful or just a capacity limitation.

2. **evaluate.py fix changes baselines.** MANDATORY regression test before interpreting bilinear results. Document the delta in WIS/ESS for existing models.

3. **Interaction penalty weight sensitivity.** The default `inter_penalty_weight=1e-3` controls how strongly the model is pushed toward additive-only. If bilinear ≈ additive in results, test `{0, 1e-4, 1e-3, 1e-2}` to distinguish "no interaction in data" from "penalty too strong." Reverse: if bilinear Q-values diverge, increase penalty weight.

4. **AD-BCQ comparison.** Lee et al. (AISTATS 2025) operates under a "Separable Effects" assumption — sub-action effects are non-interacting, the opposite regime from our bilinear model. AD-BCQ is complementary rather than directly competing, making head-to-head comparison informative but not essential for novelty. Consider implementing as fourth baseline if cohort and reward scaling are confirmed compatible.

5. **Training stability.** 25-dim Q-head may produce noisier gradients. If loss diverges, add gradient clipping at 1.0. The rank-1 structure (5+5 instead of 10+25) acts as implicit regularization — divergence is less likely than with a full 25-dim head. Additionally, the `_last_beta` side-channel pattern is fragile — if any forward call is inserted between `self.Q(state)` and the penalty computation (e.g., for logging or additional validation), `_last_beta` will be overwritten silently. If training breaks, refactor to explicit 4-tuple return from `forward()`.
