══╡ ALUR EKSEKUSI END-TO-END TANG 2022 — REPRODUKSI BCQf ╞══════════════════════════

Dua pipeline: NON-SHIFTED (completed) dan SHIFTED (in progress).
Perbedaan: shifted memperbaiki temporal alignment (state_t + action_{t-1}),
sisanya logika identik. Detail di file SplitSepsisCohort_shifted.ipynb.


┌─────────────────────────────────────────────────────────────────────────────┐
│ BAGIAN 0 — PREPROCESSING (sekali di awal, shared oleh kedua pipeline)        │
└─────────────────────────────────────────────────────────────────────────────┘

  ENV: .venv (Python 3.6, Tang exact dependencies)

  [0a] preprocess.py
       └─ Extract raw data dari MIMIC-III PostgreSQL → CSV files
       └─ Hanya credential database yang diedit

  [0b] sepsis_cohort.py
       └─ Apply sepsis-3 criteria, inclusion/exclusion → cohort 19,287
       └─ Output: sepsis_final_data_withTimes.csv
       └─ Termasuk BUG .index line 959
       └─ Gap: kita 19,282 (5 pasien beda = micro-variation database)


┌─────────────────────────────────────────────────────────────────────────────┐
│ BAGIAN 1 — ACTION REMAPPING & SPLIT (folder 1_cohort/)                       │
└─────────────────────────────────────────────────────────────────────────────┘

  ENV: venv_offlinerl (Python 3.9, PyTorch 2.8, Lightning 2.6)

  [1a] RemapActions.ipynb
       └─ Remap dosages → 5×5 discrete bins (kuantil)
       └─ Output: *_newActions.csv

  [1b] compute_acuity_scores.py
       └─ Hitung OASIS, SAPS II, SOFA scores
       └─ Output: acuity_scores.csv

  ─── NON-SHIFTED ───

  [1c] SplitSepsisCohort.ipynb
       └─ Split 70/15/15 (stratified by mortality), buat episodic .pt files
       └─ Alignment ORIGINAL: state_t + action_t di timestep sama
       └─ Output: episodes/train_set.pt     (13,484 patients)
                   episodes/val_set.pt       ( 2,906 patients)
                   episodes/test_set.pt      ( 2,892 patients)
                   Total: 19,282

  ─── SHIFTED ───

  [1c'] SplitSepsisCohort_shifted.ipynb
       └─ Sama seperti [1c] tapi SHIFTED: action & reward digeser mundur 1
       └─ observations[:new_length], actions[1:length], rewards[1:length]
       └─ Output: episodes/shifted_train_set.pt  (13,338 patients)
                   episodes/shifted_val_set.pt    ( 2,875 patients)
                   episodes/shifted_test_set.pt   ( 2,867 patients)
                   Total: 19,080


┌─────────────────────────────────────────────────────────────────────────────┐
│ BAGIAN 2 — TRAIN ENCODER (folder 2_train_encoder/)                           │
└─────────────────────────────────────────────────────────────────────────────┘

  ENV: venv_offlinerl

  Encoder: AIS_LSTM(gen: 63→128→128→LSTM(128,64), pred: 89→128→128→33)
  Training: prediksi next_obs dari (state_summary, next_action)

  ─── NON-SHIFTED ───

  [2a] opt.py
       ├── model.py          ← AIS_LSTM architecture
       └── data.py           ← MIMIC3SepsisDataModule (load train_set.pt, dll)
       └─ Grid search: 5 latent_dim × 5 lr = 25 kombinasi
          latent_dim ∈ [8,16,32,64,128], lr ∈ [1e-5,5e-4,1e-4,5e-3,1e-3]
       └─ Best: version_17, latent_dim=64, lr=1e-4, val_loss=453.37
       └─ Output: logs/AIS_LSTM_model/version_17/checkpoints/
                  epoch=251-step=26712-v1.ckpt

  [2b] EncodeStates.ipynb
       └─ Load best checkpoint → encode semua states → statevecs
       └─ Output: episodes+encoded_state/train_data.pt
                   episodes+encoded_state/val_data.pt
                   episodes+encoded_state/test_data.pt

  ─── SHIFTED ───

  [2a'] opt_shifted.py
       ├── model.py          ← AIS_LSTM architecture (SAMA)
       └── data_shifted.py   ← ShiftedMIMIC3SepsisDataModule (load shifted_*.pt)
       └─ 1 trial: latent_dim=64, lr=1e-4 (hyperparameter optimal gak geser)
       └─ Output: logs_shifted/AIS_LSTM_model_shifted/version_3/checkpoints/
                  epoch=323-step=34020-v1.ckpt, val_loss=419.0

  [2b'] EncodeStates_shifted.ipynb
       └─ Load best shifted checkpoint → encode shifted states
       └─ Output: episodes+encoded_state/shifted_train_data.pt
                   episodes+encoded_state/shifted_val_data.pt
                   episodes+encoded_state/shifted_test_data.pt


┌─────────────────────────────────────────────────────────────────────────────┐
│ BAGIAN 3 — KNN BEHAVIOR POLICY (folder 3_kNN/)                               │
└─────────────────────────────────────────────────────────────────────────────┘

  ENV: venv_offlinerl

  K=100 nearest neighbors. Dua varian:
    Standard: π_b(a|s)       → pibs (25-dim)
    Factored: π_b(i,j|s) = π_b(i|s)×π_b(j|s) → subpibs (10-dim = 5+5)

  ─── NON-SHIFTED ───

  [3a] KNN_BehaviorCloning.ipynb (STANDARD)
       └─ Train on encoded states → pibs
       └─ Output: episodes+encoded_state+knn_pibs/{train,val,test}_data.pt
                   knn_output.npz

  [3b] KNN_BehaviorCloning_factored.ipynb (FACTORED)
       └─ Train on encoded states → subpibs
       └─ Output: episodes+encoded_state+knn_pibs_factored/{train,val,test}_data.pt
                   factored_knn_output.npz

  ─── SHIFTED ───

  [3a'] KNN_BehaviorCloning_shifted.ipynb (STANDARD)
       └─ Sama seperti [3a], load shifted encoded states
       └─ Output: episodes+encoded_state+knn_pibs/shifted_{train,val,test}_data.pt
                   knn_output_shifted.npz

  [3b'] KNN_BehaviorCloning_factored_shifted.ipynb (FACTORED)
       └─ Sama seperti [3b], load shifted encoded states
       └─ Output: episodes+encoded_state+knn_pibs_factored/shifted_{train,val,test}_data.pt
                   factored_knn_output_shifted.npz


┌─────────────────────────────────────────────────────────────────────────────┐
│ BAGIAN 4 — BCQ + BCQf TRAINING (folder 4_BCQ/ dan 4_BCQf/)                    │
└─────────────────────────────────────────────────────────────────────────────┘

  ENV: venv_offlinerl

  BCQ (Baseline): Q(s,a) 25 actions, π_b(a|s) 25-dim
  BCQf (Factored): Q(s,a) = Σ_d q_d(s,a_d), π_b(i,j|s) = π_b(i|s)·π_b(j|s)

  Grid search: 8 thresholds (τ) × 5 seeds = 40 trials per model
  τ ∈ [0, 0.01, 0.05, 0.1, 0.3, 0.5, 0.75, 0.9999]
  seeds ∈ [0,1,2,3,4]

  INTERNAL SHIFT (EpisodicBuffer/SASRBuffer .load()):
    state[:,:-1] + action[:,1:] + reward[:,1:]
    → Model belajar state[t] → action[t+1] (NEXT action)
    → Tetap dibutuhkan di SHIFTED data (jika tidak → circular)

  ─── NON-SHIFTED ───

  [4a] 4_BCQ/opt.py
       ├── model.py   ← BCQ architecture (Q_net + πb_net, 25-dim)
       └── data.py    ← EpisodicBuffer, SASRBuffer (load knn_pibs standard)
       └─ 40 trials → checkpoint setiap 100 step, max 10,000
       └─ StopAndSave callback (bypass Lightning 2.6 bugs)
       └─ Output: logs/mimic_dBCQ/version_X/step=100.ckpt ... step=10000.ckpt

  [4b] 4_BCQf/opt.py
       ├── model.py   ← BCQf architecture (factored, 10-dim sub-action)
       └── data.py    ← EpisodicBuffer, SASRBuffer (load knn_pibs_factored)
       └─ 40 trials, sama seperti [4a]
       └─ Output: logs/mimic_dBCQf/version_X/step=100.ckpt ... step=10000.ckpt

  ─── SHIFTED ───

  [4a'] 4_BCQ/opt_shifted.py
       ├── model.py          ← BCQ architecture (SAMA)
       └── data_shifted.py   ← EpisodicBuffer, SASRBuffer (load knn_pibs shifted)
       └─ 40 trials, logika SAMA dengan [4a]
       └─ Output: logs_shifted/mimic_dBCQ_shifted/version_X/...

  [4b'] 4_BCQf/opt_shifted.py
       ├── model.py          ← BCQf architecture (SAMA)
       └── data_shifted.py   ← load knn_pibs_factored shifted
       └─ 40 trials, logika SAMA dengan [4b]
       └─ Output: logs_shifted/mimic_dBCQf_shifted/version_X/...


┌─────────────────────────────────────────────────────────────────────────────┐
│ BAGIAN 5 — MODEL SELECTION & EVALUASI (folder EvalPlots/)                     │
└─────────────────────────────────────────────────────────────────────────────┘

  ENV: venv_offlinerl

  [5] QuantitativeEval.ipynb
      ├── evaluate.py        ← EpisodicBufferO, EpisodicBufferF, offline_eval
      ├── 4_BCQ/model.py     ← BCQ (load checkpoint)
      ├── 4_BCQf/model.py    ← BCQf (load checkpoint)
      ├── 4_BCQ/data.py      ← remap_rewards, EpisodicBuffer
      └── 4_BCQf/data.py     ← remap_rewards, EpisodicBuffer
      └─ Mencakup NON-SHIFTED + SHIFTED sekaligus

      Procedure:
        a. Scan semua checkpoint dari setiap version → (val_wis, val_ess)
        b. Hitung Pareto frontier (maximize both WIS & ESS)
        c. Evaluasi semua model Pareto di test set
        d. Model selection (Tang criterion): val_ess ≥ 200 → max val_wis
        e. Evaluasi khusus Tang τ=0.5
        f. Bootstrap 100× untuk standard error
        g. Report hasil

      Structure cells:
        [0-3]   Setup: imports, load non-shifted, load shifted, pareto()
        [4-8]   BCQ: scan→test (non-shifted) → scan→test (shifted)
        [9-13]  BCQf: scan→test (non-shifted) → scan→test (shifted)
        [14-18] Bootstrap: select→boot (non-shifted) → select→boot (shifted)
        [19-22] Results: non-shifted → shifted


┌─────────────────────────────────────────────────────────────────────────────┐
│ PERBANDINGAN FILE NON-SHIFTED vs SHIFTED                                     │
└─────────────────────────────────────────────────────────────────────────────┘

  Setiap file shifted dibuat dengan COPY file non-shifted, lalu:

  File                    │ Perubahan
  ────────────────────────┼────────────────────────────────────────────────────
  1c' SplitSepsisCohort   │ + shifted alignment (actions/rewards offset +1)
                          │ + output file prefix "shifted_"
                          │ + length dikurangi 1
  ────────────────────────┼────────────────────────────────────────────────────
  2a' opt_shifted.py      │ import data_shifted (bukan data)
                          │ log ke logs_shifted/
  ────────────────────────┼────────────────────────────────────────────────────
  2b' data_shifted.py     │ train/val/test file paths → shifted_*.pt
                          │ class: ShiftedMIMIC3SepsisDataModule
  ────────────────────────┼────────────────────────────────────────────────────
  2b' EncodeStates_shftd  │ load checkpoint shifted, import data_shifted
                          │ output prefix "shifted_"
  ────────────────────────┼────────────────────────────────────────────────────
  3a'/3b' KNN shifted     │ load shifted encoded states
                          │ output prefix "shifted_"
  ────────────────────────┼────────────────────────────────────────────────────
  4a'/4b' opt_shifted.py  │ import data_shifted, log ke logs_shifted/
                          │ data path → shifted knn_pibs
  ────────────────────────┼────────────────────────────────────────────────────
  4a'/4b' data_shifted.py │ IDENTIK (no changes needed — internal shift
                          │ tetap dibutuhkan, paths via opt)
  ────────────────────────┼────────────────────────────────────────────────────
  4a'/4b' model.py        │ IDENTIK (no changes)
  ────────────────────────┼────────────────────────────────────────────────────
  5  QuantitativeEval     │ NON-SHIFTED cells untouched, SHIFTED cells
                          │ ditambahkan dengan logika persis sama


┌─────────────────────────────────────────────────────────────────────────────┐
│ RINGKASAN EKSEKUSI                                                            │
└─────────────────────────────────────────────────────────────────────────────┘

  [0a] python preprocess.py                     # sekai di awal
  [0b] python sepsis_cohort.py --process_raw    # sekali di awal

  [1a] jupyter RemapActions.ipynb
  [1b] python compute_acuity_scores.py
  [1c] jupyter SplitSepsisCohort.ipynb          # non-shifted
  [1c']jupyter SplitSepsisCohort_shifted.ipynb  # shifted

  [2a] for d in 8 16 32 64 128; do             # non-shifted encoder
         for lr in 1e-5 5e-4 1e-4 5e-3 1e-3; do
           python opt.py --latent_dim $d --lr $lr
         done
       done
  [2b] jupyter EncodeStates.ipynb

  [2a']python opt_shifted.py --latent_dim 64 --lr 1e-4  # shifted encoder
  [2b']jupyter EncodeStates_shifted.ipynb

  [3a] jupyter KNN_BehaviorCloning.ipynb             # standard non-shifted
  [3b] jupyter KNN_BehaviorCloning_factored.ipynb    # factored non-shifted

  [3a']jupyter KNN_BehaviorCloning_shifted.ipynb          # standard shifted
  [3b']jupyter KNN_BehaviorCloning_factored_shifted.ipynb # factored shifted

  [4a] python 4_BCQ/opt.py          # BCQ non-shifted (40 trials)
  [4b] python 4_BCQf/opt.py         # BCQf non-shifted (40 trials)

  [4a']python 4_BCQ/opt_shifted.py   # BCQ shifted (40 trials)
  [4b']python 4_BCQf/opt_shifted.py  # BCQf shifted (40 trials)

  [5]  jupyter QuantitativeEval.ipynb  # evaluasi keduanya


┌─────────────────────────────────────────────────────────────────────────────┐
│ HASIL NON-SHIFTED                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

  ============================================================
  Clinician    WIS: 90.25 ± 0.57  ESS: 2892
  Baseline BCQ WIS: 95.61 ± 1.26  ESS: 212.9 ± 11.5
  Factored BCQ WIS: 94.79 ± 1.42  ESS: 224.7 ± 12.7
  Tang τ=0.5 BCQ : WIS: 94.72 ± 1.51  ESS: 200.0 ± 11.8
  Tang τ=0.5 BCQf: WIS: 93.68 ± 1.42  ESS: 231.2 ± 12.3
  ============================================================

  Tang (2022) paper (τ=0.5):
    Clinician:    WIS=90.29  ESS=2894
    Baseline BCQ: WIS=90.44  ESS=178.32
    Factored BCQ: WIS=91.62  ESS=178.32
