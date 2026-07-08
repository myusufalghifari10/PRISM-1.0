# Cohort Difference Root Cause: 18,562 vs 19,287 (Δ = 725 patients)

## Executive Summary

**The 725-patient gap is caused by a SINGLE bug on line 958 of `sepsis_cohortOri.py`** (Tang's original code). The "exclude patients who died in ICU during data collection" filter is a **complete no-op** in Tang's code due to taking `.index` on the wrong object. The standalone Microsoft version and PRISM's version (prior to the recent revert) correctly exclude these patients. The fix is to change line 958 from:

```python
ii = reformat4t['icustayid'][ii].isin(icustayidlist).index  # BUG: returns DataFrame row indices, not icustay IDs
```

to:

```python
ii = reformat4t['icustayid'][ii][reformat4t['icustayid'][ii].isin(icustayidlist)]  # CORRECT: returns actual icustay IDs
```

---

## Files Examined

| # | File | Role | Lines | MD5 |
|---|------|------|-------|-----|
| A | `Repo Source/OfflineRL_FactoredActions/RL_mimic_sepsis/mimic_sepsis.microsoft@GitHub/sepsis_cohortOri.py` | Tang's original (produces 19,287) | 1766 | `3e43243...` |
| B | `Repo Source/mimic_sepsis/sepsis_cohort.py` | Standalone Microsoft (produces 18,562) | 1766 | `8f6e38e...` |
| C | `RL_mimic_sepsis/mimic_sepsis.microsoft@GitHub/sepsis_cohort.py` | Our version (+4 lines comment) | 1770 | `2dd430c...` |

No third version of `sepsis_cohort.py` exists in Tang's repository. Only `sepsis_cohortOri.py` (original) was bundled.

---

## Complete Diff Analysis

### Difference 1: Line 958 — "Exclude patients who died in ICU" filter (**THE ROOT CAUSE**)

**Tang's original (A) — BROKEN:**
```python
# line 958
ii = reformat4t['icustayid'][ii].isin(icustayidlist).index
```
**Standalone Microsoft (B) — CORRECT:**
```python
# line 958
ii = reformat4t['icustayid'][ii][reformat4t['icustayid'][ii].isin(icustayidlist)]
```

**Trace of the bug:**

| Step | Code | Tang's original result | Correct result |
|------|------|----------------------|----------------|
| 1 | `ii = (reformat4t['bloc'] == 1) & (died...) & (delay < 24)` | Boolean mask, e.g., True at rows [1500, 2500, 3000] | Same |
| 2 | `reformat4t['icustayid'][ii]` | Series of icustay IDs, e.g., [200123, 200456, 200789] | Same |
| 3 | `.isin(icustayidlist)` | Boolean Series: [True, True, True] | Same |
| 4 | `.index` | **Returns DataFrame integer row indices: [1500, 2500, 3000]** | — |
| 4' | `[boolean_mask]` | — | **Returns actual icustay IDs: [200123, 200456, 200789]** |
| 5 | `reformat4t['icustayid'].isin(ii)` | Check if icustay IDs (200123, ...) are in [1500, 2500, 3000] → **ALL FALSE** → **NO ROWS EXCLUDED** | Check if icustay IDs are in [200123, ...] → correctly matches |

**Impact:** The broken filter fails to exclude ~725 patients who died in ICU during the data collection period. Their trajectories end early, but they still contribute to the sepsis cohort (SOFA ≥ 2). The correct filter removes these patients, reducing the cohort from 19,287 to 18,562.

---

### Difference 2: `fillna(inplace=True)` → `column = column.fillna()` patterns (Cosmetic/Compatibility)

**Tang's original (A):**
```python
demog['morta_90'].fillna(0, inplace=True)
demog['morta_hosp'].fillna(0, inplace=True)
demog['elixhauser'].fillna(0, inplace=True)
reformat4t['mechvent'].fillna(0, inplace=True)
reformat4t['elixhauser'].loc[np.isnan(...)] = ...
reformat4t['median_dose_vaso'].fillna(0, inplace=True)
reformat4t['max_dose_vaso'].fillna(0, inplace=True)
reformat4t['Shock_Index'].fillna(d, inplace=True)
```

**Our version (C):**
```python
demog['morta_90'] = demog['morta_90'].fillna(0)
demog['morta_hosp'] = demog['morta_hosp'].fillna(0)
demog['elixhauser'] = demog['elixhauser'].fillna(0)
reformat4t['mechvent'] = reformat4t['mechvent'].fillna(0)
reformat4t.loc[np.isnan(...), 'elixhauser'] = ...
reformat4t['median_dose_vaso'] = reformat4t['median_dose_vaso'].fillna(0)
reformat4t['max_dose_vaso'] = reformat4t['max_dose_vaso'].fillna(0)
reformat4t['Shock_Index'] = reformat4t['Shock_Index'].fillna(d)
```

**Verdict:** NO functional difference. Both patterns produce identical data. The `= ...fillna()` pattern avoids SettingWithCopyWarning and Pandas Copy-on-Write issues. Does NOT affect cohort count.

---

### Difference 3: `np.NaN` → `np.nan` (No-op)

**Tang's original (A)** and **standalone (B):** `np.NaN`
**Our version (C):** `np.nan`

In NumPy, `np.NaN` is an alias for `np.nan` (they are the same object: `np.NaN is np.nan` → `True`). NO functional difference.

---

### Difference 4: `.astype(np.int)` → `.astype(int)` (No-op)

**Tang's original/standalone:** `.astype(np.int)`
**Our version:** `.astype(int)`

Both produce identical results. `.astype(int)` is the preferred modern API. NO functional difference.

---

### Difference 5: `.values` → `.values.copy()` (No-op in context)

**Tang's original/standalone (B):**
```python
vc = MIMICtable['max_dose_vaso'].values
```
**Our version (C):**
```python
vc = MIMICtable['max_dose_vaso'].values.copy()
```

`vc` is never mutated in the subsequent code — only `vcr` is created and mutated. So `.copy()` is unnecessary and has no effect. NO functional difference.

---

### Difference 6: SOFA computation missing backslash (Standalone Microsoft BUG — does NOT affect cohort count)

**Standalone Microsoft (B) — BROKEN at line 1581:**
```python
t = max(p[s1[:,i]], default=0) + max(p[s2[:,i]], default=0) + max(p[s3[:,i]], default=0) + max(p[s4[:,i]], default=0) 
    + max(p[s5[:,i]], default=0) + max(p[s6[:,i]], default=0)
```

Without the line continuation `\`, the second line `+ max(p[s5[:,i]], default=0) + max(p[s6[:,i]], default=0)` is a standalone no-op expression. The SOFA score `t` only sums the first 4 criteria (s1–s4), NOT s5 (GCS) and s6 (Creatinine/UO).

**Tang's original (A) and Our version (C) — CORRECT:**
```python
t = max(p[s1[:,i]], default=0) + max(p[s2[:,i]], default=0) + max(p[s3[:,i]], default=0) + max(p[s4[:,i]], default=0) \
    + max(p[s5[:,i]], default=0) + max(p[s6[:,i]], default=0)
```

**Why this does NOT affect cohort count:** This SOFA computation is in the **second phase** (Sepsis Cohort re-formatting), which happens AFTER the cohort has already been selected. The cohort selection uses the **first phase** SOFA computation (line ~881-890), which has the correct `\` in all versions. So this bug only affects the output trajectory features, not which patients are included in the cohort.

---

### Difference 7: `\ ` backslash-space → `\` no-space (No-op)

**Standalone Microsoft (B):**
```python
colnorm = ['age', 'Weight_kg', 'GCS', 'HR', ..., 'FiO2_1',\ 
```
(note: space after `\`)

**Tang's/Our:**
```python
colnorm = ['age', 'Weight_kg', 'GCS', 'HR', ..., 'FiO2_1',\
```

The space after `\` in the standalone version is technically a "line continuation with trailing whitespace" which some linters flag but Python handles identically. NO functional difference.

---

### Difference 8: Output CSV paths (No cohort effect)

**Tang's original/standalone:**
```python
df.to_csv('sepsis_final_data_withTimes.csv', index=False)
df.to_csv('sepsis_final_data_RAW_withTimes.csv', index=False)
```

**Our version:**
```python
df.to_csv('../data/sepsis_final_data_withTimes.csv', index=False)
df.to_csv('../data/sepsis_final_data_RAW_withTimes.csv', index=False)
```

Only changes the output directory. NO effect on cohort computation.

---

## Preprocess.py Analysis

All three `preprocess.py` files are **functionally identical**:

| File | DB Host | Schema | Other |
|------|---------|--------|-------|
| Tang's bundled (`Repo Source/OfflineRL_FactoredActions/...`) | `127.0.0.1` | `mimimciii` (misspelled) | Uses MATERIALIZED VIEW |
| Standalone Microsoft (`Repo Source/mimic_sepsis/...`) | `mimic` | `mimimciii` (misspelled) | Uses MATERIALIZED VIEW |
| Our version (`RL_mimic_sepsis/...`) | `127.0.0.1` | `mimimciii` (misspelled) | Uses MATERIALIZED VIEW |

- The host difference (`mimic` vs `127.0.0.1`) is just network naming for the same database.
- The schema misspelling `mimimciii` (should be `mimiciii`) is consistent across ALL versions — another bug, but not a differential bug.
- All versions use `CREATE MATERIALIZED VIEW`, not `CREATE TABLE`. No deleted_caregivers filter exists anywhere.
- **Conclusion:** All preprocess.py files produce identical data files when connected to the same MIMIC-III database.

---

## Complete Accounting of the 725-Patient Gap

| Cause | Effect | Magnitude |
|-------|--------|-----------|
| Line 958 bug: `.index` returns row indices instead of icustay IDs | "Exclude patients who died in ICU during data collection" filter is a complete no-op in Tang's code. ~725 patients who died in ICU are kept in the cohort. | **725 patients** |
| All other differences (fillna patterns, np.nan, .astype, .values.copy(), csv paths) | No functional difference | 0 patients |
| preprocess.py differences (host string, schema name) | No functional difference | 0 patients |
| SOFA computation missing backslash (standalone Microsoft only) | Second-phase bug, doesn't affect cohort selection | 0 patients |

**Total explained gap: 725 patients** (19,287 − 18,562 = 725)

---

## The Bug in Detail: Death-in-ICU Exclusion

The broken filter at lines 957–961 of `sepsis_cohortOri.py`:

```python
# Exclude patients who died in ICU during data collection period
print('Full ICU -- excluding patients who died in ICU during data collection period')
ii = (reformat4t['bloc'] == 1) & (reformat4t['died_within_48h_of_out_time'] == 1) & (reformat4t['delay_end_of_record_and_discharge_or_death'] < 24)
ii = reformat4t['icustayid'][ii].isin(icustayidlist).index  # ← BUG
ii = reformat4t['icustayid'].isin(ii)
reformat4t = reformat4t.loc[~ii]
```

### Why `.index` breaks the filter

1. `reformat4t['icustayid'][ii]` returns a Pandas Series of icustay_id values for rows matching the death condition.
2. `.isin(icustayidlist)` returns a boolean Series (True/False for each value).
3. `.index` on a boolean Series returns the **DataFrame integer row positions** (e.g., `Int64Index([1500, 2500, 3000])`), NOT the icustay_id values (e.g., `[200123, 200456, 200789]`).
4. The subsequent `reformat4t['icustayid'].isin(ii)` checks whether icustay_id values (all ≥ 200000) are in a list of small integers (row indices like 1500-3000). This is **always False**, so NO rows are ever excluded.

### The correct code (from standalone Microsoft)

```python
ii = reformat4t['icustayid'][ii][reformat4t['icustayid'][ii].isin(icustayidlist)]
```

Here, the boolean Series from `.isin(icustayidlist)` is used as a **boolean indexer** (`[mask]`) to filter the icustay_id Series, returning the actual icustay_id values that need to be removed. These then correctly match via the outer `.isin(ii)`.

---

## Current State of Our Version

Our `RL_mimic_sepsis/mimic_sepsis.microsoft@GitHub/sepsis_cohort.py` currently contains the **buggy** version of the filter, with an explicit comment:

```python
# NOTE: Reverted to Tang's original (buggy) code for reproduction fidelity.
# Tang's sepsis_cohortOri.py uses .index which returns DataFrame index positions
# instead of actual icustay IDs, causing this exclusion to be a no-op.
# This keeps ~725 extra patients in the cohort (19,287 vs 18,562).
ii = reformat4t['icustayid'][ii].isin(icustayidlist).index  # Tang's original: buggy but matches paper cohort
```

This means our code now intentionally reproduces Tang's 19,287 cohort.

---

## Recommendation

### For exact Tang reproduction (19,287 patients):
Use the buggy `.index` version (line 958) as currently present in our `sepsis_cohort.py`.

### For correct exclusion logic (18,562 patients):
Change line 958 to:
```python
ii = reformat4t['icustayid'][ii][reformat4t['icustayid'][ii].isin(icustayidlist)]
```

### Additional recommended fix (SOFA computation):
Fix the standalone Microsoft SOFA bug (not in our code, but worth noting):
```python
# Line 1581 in standalone Microsoft — missing backslash
t = max(p[s1[:,i]], default=0) + max(p[s2[:,i]], default=0) + max(p[s3[:,i]], default=0) + max(p[s4[:,i]], default=0) \
    + max(p[s5[:,i]], default=0) + max(p[s6[:,i]], default=0)
```

---

## Key Files Summary

| File | Path | Key Lines |
|------|------|-----------|
| Tang's original | `Repo Source/OfflineRL_FactoredActions/RL_mimic_sepsis/mimic_sepsis.microsoft@GitHub/sepsis_cohortOri.py` | 958 (buggy .index), 1581 (correct SOFA `\`) |
| Standalone Microsoft | `Repo Source/mimic_sepsis/sepsis_cohort.py` | 958 (correct filter), 1581 (broken SOFA missing `\`) |
| Our version | `RL_mimic_sepsis/mimic_sepsis.microsoft@GitHub/sepsis_cohort.py` | 958-962 (buggy + comment), 1581 (correct SOFA `\`) |
| Preprocess (all identical) | All three `preprocess.py` files | Line 160 (MATERIALIZED VIEW comment), Line 193-194 (SQL) |

---

## Start Here

Another agent should open `/home/yusuf/Yusuf/Kuliah/AI/Project/ICU/PRISM-1.0/RL_mimic_sepsis/mimic_sepsis.microsoft@GitHub/sepsis_cohort.py` and inspect lines 956–965. The bug is at line 962 (originally 958 in the Ori version, but our file has 4 lines of comment added). To fix: replace `.index` with `[reformat4t['icustayid'][ii].isin(icustayidlist)]`.
