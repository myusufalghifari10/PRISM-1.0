# KNN Imputation Gap Investigation

**Verdict: KNN imputation CANNOT cause the 14-patient gap.**

All three versions use identical KNN logic. There are zero differences in the imputation pipeline.

---

## Files Retrieved

1. `Repo Source/OfflineRL_FactoredActions/RL_mimic_sepsis/mimic_sepsis.microsoft@GitHub/sepsis_cohortOri.py` — Tang's original (uses `fancyimpute.KNN`)
2. `Repo Source/mimic_sepsis/sepsis_cohort.py` — Microsoft fixed version (uses `fancyimpute.KNN`)
3. `RL_mimic_sepsis/mimic_sepsis.microsoft@GitHub/sepsis_cohort.py` — our version (uses `fancyimpute.KNN`)

---

## 1. KNN Import and Parameters — All Identical

All three files share exactly the same import and KNN invocation:

```python
# All 3 files, line 56:
from fancyimpute import KNN

# All 3 files, Phase 1 KNN (lines ~837-839):
ref[i:i+9999,:] = KNN(k=1).fit_transform(ref[i:i+9999,:])

# All 3 files, Phase 2 KNN (lines ~1545-1549):
ref[i:i+9999,:] = KNN(k=1).fit_transform(ref[i:i+9999,:])
```

| Parameter | Value | Same across all? |
|-----------|-------|------------------|
| Library | `fancyimpute.KNN` | YES |
| k | 1 | YES |
| Weighting | default (uniform) | YES |
| Metric | default (Euclidean) | YES |
| Chunk size | 9999 rows | YES |
| Column range (Phase 1) | `[:,11:mechventcol]` | YES |
| Column range (Phase 2) | `[:,12:mechventcol]` | YES |

There is **no sklearn KNNImputer** in any of these files. All three use `fancyimpute.KNN`.

---

## 2. Which Filters Depend on KNN-Imputed Values?

The three post-imputation outlier filters:

| Filter | Column index | Within KNN range? | KNN-affected? |
|--------|-------------|-------------------|---------------|
| `output_4hourly > 12000` | 83 | NO (after `mechventcol`) | ❌ NOT imputed |
| `Total_bili > 10000` | 51 | YES | ✅ IS imputed |
| `input_4hourly > 10000` | 81 | NO (after `mechventcol`) | ❌ NOT imputed |

Column index breakdown (Phase 1, columns 0-84):
- `[:,0:11]` — demographics (never imputed)
- `[:,11:74]` — clinical variables incl. `Total_bili` (index 51) — **KNN-imputed**
- `[:,74:]` — mechvent, extubated, Shock_Index, PaO2_FiO2, vasopressors, input/output totals — **NOT imputed**

**Only `Total_bili > 10000` could theoretically be affected by KNN differences.** The other two filters (`output_4hourly`, `input_4hourly`) operate on columns that are never KNN-imputed.

---

## 3. Can KNN Differences Cause 14 Patients to Pass/Fail?

**No.** Even though `Total_bili` IS KNN-imputed, all three files use the **same** `fancyimpute.KNN(k=1)` on the **same** input data. The imputed values for every row are deterministic and identical across all three versions.

The only way KNN could produce different outputs is if input data differs — but the data processing pipeline from raw CSV → `reformat3` is identical across all three files (the fillna style differences are cosmetic — see §4).

Chunk boundaries also cannot explain a difference: the row count and ordering of `reformat3` are identical across all versions, so the 10K-row chunks are identically formed.

---

## 4. Real Differences Between Versions (All Outside KNN)

### 4a. Cosmetic fillna style differences — **NO EFFECT**

| Location | Our version | Microsoft/Tang | Effect |
|----------|------------|----------------|--------|
| Line ~97-99 | `demog['col'] = demog['col'].fillna(0)` | `demog['col'].fillna(0, inplace=True)` | Same result |
| Line ~858 | `reformat4t['mechvent'] = reformat4t['mechvent'].fillna(0)` | `reformat4t['mechvent'].fillna(0, inplace=True)` | Same result |
| Lines ~865-866 | `reformat4t['col'] = reformat4t['col'].fillna(0)` | `reformat4t['col'].fillna(0, inplace=True)` | Same result |
| Line ~862 | `reformat4t.loc[np.isnan(reformat4t['elixhauser']), 'elixhauser'] = ...` | `reformat4t['elixhauser'].loc[np.isnan(...)] = ...` | Same result |

### 4b. `np.NaN` vs `np.nan` — **CRASHES with NumPy 2.0, but same NaN value otherwise**

```python
# Our version (lines 874, 1565):
reformat4t.loc[np.isinf(reformat4t['Shock_Index']), 'Shock_Index'] = np.nan

# Microsoft/Tang (lines 874, 1565):
reformat4t.loc[np.isinf(reformat4t['Shock_Index']), 'Shock_Index'] = np.NaN
```

Verified: `np.NaN` was **removed in NumPy 2.0**. On NumPy < 2.0, `np.NaN` is an alias for `np.nan` and produces identical behavior. On NumPy ≥ 2.0, the Microsoft/Tang version **crashes with AttributeError** before reaching this line. Our version is safe.

This cannot explain the 14-patient gap — it's either a crash or identical.

### 4c. `reformat4t['Shock_Index'].fillna(d)` vs assignment style — **NO EFFECT**

Both produce the same result (fill NaN with scalar `d`).

### 4d. `.astype(int)` vs `.astype(np.int)` — **NO EFFECT**

`np.int` was deprecated but produces the same cast as `int`.

### 4e. `.values.copy()` vs `.values` — **NO EFFECT**

Our version uses defensive copy; doesn't change any values.

### 4f. **Early death exclusion** — **MAJOR difference (~725 patients, NOT 14)**

```python
# Our version + Tang (Buggy — returns DataFrame index positions, not icustay IDs):
ii = reformat4t['icustayid'][ii].isin(icustayidlist).index
# ^ .index returns 0,1,2,... (DataFrame row indices)
# These small integers never match real icustay IDs like 200001
# → Exclusion is a NO-OP or near-no-op

# Microsoft fixed (Correct):
ii = reformat4t['icustayid'][ii][reformat4t['icustayid'][ii].isin(icustayidlist)]
# Properly filters to values that are actually in icustayidlist
```

This explains the ~725-patient difference (19,287 vs 18,562) — NOT the 14-patient gap.

### 4g. Output path — **NO EFFECT**

Our version writes to `../data/`, Microsoft to `./`.

---

## 5. KNN Parameters Deep Dive

### Phase 1 (Full ICU) — Line 834-839
```python
reformat3t_cols = reformat3t.columns.tolist()
mechventcol = reformat3t_cols.index('mechvent')  # = 74
ref = np.copy(reformat3[:,11:mechventcol])        # columns 11-73 (clinical vars)
for i in range(0,reformat3.shape[0],9999):
    ref[i:i+9999,:] = KNN(k=1).fit_transform(ref[i:i+9999,:])
```

### Phase 2 (Sepsis Cohort) — Lines 1542-1549
```python
reformat3t_cols = reformat3t.columns.tolist()
mechventcol = reformat3t_cols.index('mechvent')  # varies with kept columns
ref = np.copy(reformat3[:,12:mechventcol])        # columns 12 to mechvent
for i in range(0,reformat3.shape[0],9999):
    ref[i:i+9999,:] = KNN(k=1).fit_transform(ref[i:i+9999,:])
```

Phase 2 starts at column 12 (not 11) because the Sepsis Cohort `reformat3t` has a `presumed_onset` column at index 7 that Phase 1 doesn't have, shifting all clinical columns right by 1.

### KNN Chunk Behavior
- Chunks of up to 9999 rows are processed independently
- Within each chunk, missing values are filled by the single nearest neighbor (k=1)
- Since row ordering and data are identical across all three versions, chunk contents are identical
- **Deterministic**: same input → same output

---

## 6. Conclusion

**KNN imputation is NOT the cause of the 14-patient gap.** The KNN implementation, parameters, input data, and column ranges are identical across all three versions. Even the one filter that operates on a KNN-imputed column (`Total_bili > 10000`) would produce identical results.

### Where to look instead for the 14-patient gap:
1. **Input data differences** — Do the three versions use the same processed CSV files (different MIMIC-III versions, different preprocess.py runs)?
2. **Other exclusion logic** — The `output_4hourly` and `input_4hourly` filters (which are NOT KNN-imputed) could differ if earlier fluid/UO calculations differ
3. **The `np.NaN` → `np.nan` fix** — If one version was run on NumPy ≥ 2.0, `np.NaN` would crash. If both ran on NumPy < 2.0, this is identical
4. **`fixgaps` linear interpolation** — occurs before KNN, could differ if data differs
5. **SAH (Sample and Hold)** — occurs before KNN, same logic across all versions but sensitive to data differences

### No fix recommended for KNN
No KNN-related fix is needed because there's no KNN difference between the files.
