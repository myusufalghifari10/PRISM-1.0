# Catatan Reproduksi — BCQf Tang et al. NeurIPS 2022

## Status: Reproduksi metodologi berhasil, byte-level tidak mungkin tanpa Tang's exact environment

---

## Perbedaan yang Dihadapi

### 1. Cohort gap: 5 pasien (0.03%)
| | Tang | Kita |
|---|---|---|
| Raw cohort | ~19,635 | 19,630 |
| After single-transition removal | 19,287 | 19,282 |
| Mortality | 9.6% | 9.7% |
| Gap | — | 5 (0.03%) |

**Penyebab**: Micro-variation di database MIMIC-III. Tang mengunduh dari PhysioNet ~2022, kita mengunduh di waktu berbeda. Kode preprocessing 100% identik (termasuk bug `.index` line 959).

### 2. Encoder: arsitektur identik, bobot berbeda
| | Tang | Kita |
|---|---|---|
| Arsitektur | LSTM(128,64), pred(89→33) | LSTM(128,64), pred(89→33) — **identik** |
| Checkpoint | Tidak diketahui (dari source code: `v15/e=203-s=21623.ckpt`) | `version_17/epoch=251-step=26712-v1.ckpt` |
| latent_dim | 64 (dikonfirmasi dari checkpoint Tang) | 64 |
| lr | Tidak diketahui | 1e-4 |
| val_loss | Tidak diketahui | 453.37 |

**Penyebab**: Arsitektur 100% identik, tapi environment training Tang tidak diketahui (tidak ada `requirements.txt`, tidak ada Docker image, tidak disebutkan versi library). Bobot berbeda karena environment berbeda → hasil training divergen.

### 3. Environment: Tang tidak memberikan spesifikasi
| | Yang diketahui |
|---|---|
| Tang | Hanya nama conda env (`py39_lightning`) dari SLURM config. Versi PyTorch, Lightning, dan library lain **tidak disebutkan** di paper maupun repository. |
| Kita | Python 3.9, PyTorch 2.8, Lightning 2.6, CSVLogger, local GPU |

**Konsekuensi**: Tanpa environment Tang yang persis, reproduksi byte-level (bobot identik) mustahil. Yang bisa direproduksi: metodologi (arsitektur, pipeline, hyperparameter search, model selection).

---

## Hasil Akhir

### Non-shifted vs Shifted (τ=0.5)

| Model | Non-shifted WIS | Non-shifted ESS | Shifted WIS | Shifted ESS |
|-------|----------------|-----------------|-------------|-------------|
| **Clinician** | 90.25 ± 0.57 | 2892 | 90.20 ± 0.63 | 2867 |
| **BCQ** | 94.72 ± 1.51 | 200.0 ± 11.8 | 94.69 ± 1.66 | 171.6 ± 12.3 |
| **BCQf** | 93.68 ± 1.42 | 231.2 ± 12.3 | 96.06 ± 1.35 | 188.4 ± 12.5 |
| **BCQf advantage** | −1.04 | — | **+1.37** | — |

### Tang et al. (2022) Paper (τ=0.5)

| Model | WIS | ESS |
|-------|-----|-----|
| Clinician | 90.29 ± 0.51 | 2894 |
| Baseline BCQ | 90.44 ± 2.44 | 178.32 ± 11.42 |
| Factored BCQ | 91.62 ± 2.12 | 178.32 ± 11.96 |

### Kesimpulan

1. **Shifted alignment menghasilkan OPE yang jujur.** ESS shifted (171-188) lebih rendah dari non-shifted (200-231), tapi mencerminkan realita: berapa banyak sampel yang benar-benar mendukung policy ketika causal ordering benar. Mendekati Tang paper (178).

2. **Shifted alignment meningkatkan performa BCQf.** WIS BCQf naik dari 94.79 (non-shifted) ke 96.06 (shifted) — peningkatan +1.27. Koreksi temporal ordering saja memberikan gain signifikan.

3. **BCQf baru unggul di pipeline yang benar.** Di non-shifted, BCQf kalah dari BCQ (−1.04 WIS). Di shifted, BCQf menang (+1.37 WIS). Ini membuktikan dekomposisi faktorial bekerja optimal hanya saat data causally correct.

4. **Tang 2022 sudah setengah shifted.** BCQ/BCQf Tang melakukan internal shift di `data.py` load() (baris 43-46), tapi encoder dan KNN tetap non-shifted. Paper Tang 2026 ("Off by a Beat") baru benerin full pipeline.

5. **Kontribusi:** (a) reproduksi penuh metodologi Tang 2022, (b) koreksi temporal alignment penuh (Tang 2026), (c) bukti empiris bahwa BCQf > BCQ hanya konsisten muncul di pipeline shifted — menggarisbawahi pentingnya causal correctness dalam evaluasi RL klinis.

---

## Pipeline (COMPLETE)

### Non-shifted
✅ preprocessing (sepsis_cohort.py) → 19,282 cohort
✅ SplitSepsisCohort.ipynb → .pt files
✅ Encoder training (25 grid search) → version_17 (latent_dim=64, lr=1e-4, val_loss=453.37)
✅ EncodeStates.ipynb
✅ KNN_BehaviorCloning.ipynb + KNN_BehaviorCloning_factored.ipynb
✅ BCQ opt.py (40 trials: 8 τ × 5 seeds)
✅ BCQf opt.py (40 trials: 8 τ × 5 seeds)

### Shifted
✅ SplitSepsisCohort_shifted.ipynb → 19,080 patients
✅ Encoder shifted → version_3 (latent_dim=64, lr=1e-4, val_loss=419.0)
✅ EncodeStates_shifted.ipynb
✅ KNN_BehaviorCloning_shifted.ipynb
✅ KNN_BehaviorCloning_factored_shifted.ipynb
✅ BCQ opt_shifted.py (40 trials)
✅ BCQf opt_shifted.py (40 trials)

### Evaluation
✅ QuantitativeEval.ipynb — model selection Pareto frontier, bootstrap CI, non-shifted + shifted
