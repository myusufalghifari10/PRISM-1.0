# Progress Magang - ICU

## Problem Statement

Mereproduksi **Tang et al. (NeurIPS 2022)** — BCQf: Batch-Constrained Q-learning dengan *factored action space* untuk sepsis management di MIMIC-III. Model mendekomposisi Q-function menjadi `Q(s,a₁,a₂) = q₁(s,a₁) + q₂(s,a₂)` (aditif linear), mengabaikan interaksi antar sub-action (vasopressor × IV fluid) yang secara fisiologis berinteraksi lewat *blood pressure pathway*. Setelah reproduksi, kami mengusulkan **BCQf-Bilinear**: ekstensi dengan *rank-1 bilinear interaction term* `α_a₁(s)·β_a₂(s)` pada data *shifted temporal alignment*.

## Proposed Method

1. **Reproduksi BCQf Tang (2022)** pada MIMIC-III v1.4 dengan dua *temporal alignment*:
   - **Non-shifted**: episodic asli → `(state_t, action_{t+1})` (causally correct via BCQf loading offset)
   - **Shifted**: episodic di-shift +1 → `(state_t, action_{t+2})` (*over-shifted*, granularitas temporal alternatif)
2. **BCQf-Bilinear** (novel): menambah *rank-1 bilinear correction* `I_ij = α_i(s)·β_j(s)` ke dekomposisi aditif, dengan *double-centering* + L2 normalization untuk identifiability. Parameter-efficient (5,140 params) vs direct head (6,425). Nested recovery: `‖β‖→0` → kembali ke BCQf original.
3. Evaluasi: WIS/ESS via *importance sampling* + bootstrap CI, perbandingan head-to-head pada *alignment* yang sama.

## Input & Output

- **Input**: 64-dim encoded states (AIS-LSTM), 10-dim sub-action vectors (5 vasopressor dose levels + 5 IV fluid dose levels), 25 combinatorial actions (`vaso + iv*5`)
- **Output**: Q-values untuk 25 actions, WIS (Weighted Importance Sampling) estimate, ESS (Effective Sample Size), kebijakan treatment optimal

## Dataset

- **MIMIC-III v1.4** — sepsis cohort: 18.562 pasien (train/val/test split)
- Preprocessing: Killian et al. (2020) dengan KNN imputation + bug fixes (berbeda dari Tang 2022 yang pakai Komorowski original imputation)
- Action bins dihitung dari quantile training data → MDP definition berbeda (~1 pt WIS)
- Survival rate 93.9% (vs Tang 90.4%) → WIS scale shift (~2 pts)

## Hasil Terkini

| Method | Alignment | WIS (correct) | ESS |
|--------|-----------|--------------|-----|
| Clinician | — | 93.92 | — |
| BCQf (Tang setting τ=0.5) | Non-shifted | 93.66 | 238 |
| BCQf (best ESS τ=0.3) | Non-shifted | 94.41 | 192 |
| BCQf (Tang setting τ=0.5) | Shifted | ⏳ running | ⏳ |
| Tang paper (τ=0.5) | Original | 91.62 | 178 |

Reproduction successful: correct WIS dalam ~2 pts dari Tang, fully explained oleh data cohort differences + action bin differences. Bug di fungsi evaluasi (lihat kendala) sudah di-fix.

## Kendala

**Semua kendala sudah teridentifikasi dan di-fix, tinggal re-run.** Detail: `DIF.md`

1. **Bug evaluasi `offline_evaluation_F`**: argmax 10-dim mengindeks pie_soft 25-dim → actions 10-24 tidak pernah dipilih → WIS inflated +3-4 pts, ESS deflated -65-127 pts. **FIX**: switch ke `model.offline_evaluation()` yang sudah benar (10→25 mapping via `all_subactions_vec`).
2. **Perbedaan kohort data**: MIMIC-III v1.4 tapi preprocessing berbeda (Killian imputation vs Komorowski original) → survival rate 93.9% vs 90.4%, action bins berbeda → ~3 pts WIS gap.
3. **Encoder learning rate**: 1e-4 (kami) vs 5e-4 (Tang best) → representasi state berbeda (~1 pt).
4. **Field-order mismatch EpisodicBuffer vs SASRBuffer** (latent, harmless untuk training tapi time bomb jika buffer ditukar).

## Next: BCQf-Bilinear

Model ketiga dalam pipeline, berbasis Tang Appendix B.9 (residual interaction term yang diusulkan tapi tidak pernah diimplementasi). Arsitektur: shared trunk → q_add (10-dim) + α (5·k) + β (5·k) + πb (10-dim). Forward: double-center α,β → normalize α → einsum outer product → 25-dim Q. Inter_penalty = λ·‖β‖². Grid: 8 thresholds × 5 seeds × 4 penalty weights × 2 rank = 320 trials. Detail: `.plans/bcqf-abq/PLAN2.md`.

## Repository

Link GDrive: *(isi link Google Drive)*  
Local: `~/Yusuf/Kuliah/AI/Project/ICU/PRISM-1.0/`
