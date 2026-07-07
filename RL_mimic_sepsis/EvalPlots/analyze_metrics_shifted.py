"""Analyze BCQf shifted training results across all versions."""
import pandas as pd, numpy as np, re, os

logdir = os.path.join(os.path.dirname(__file__), '../4_BCQf/logs_shifted/mimic_dBCQf_shifted')

all_data = []
for ver in range(40):
    try:
        # Read hparams
        text = open(f'{logdir}/version_{ver}/hparams.yaml').read()
        thresh = float(re.search(r'threshold: ([\d.]+)', text).group(1))
        seed = int(re.search(r'seed: (\d+)', text).group(1))
        
        # Read metrics
        df = pd.read_csv(f'{logdir}/version_{ver}/metrics.csv')
        valid = df.dropna(subset=['val_wis', 'val_ess'])
        if len(valid) == 0:
            continue
        
        # Top 10 WIS
        top_wis = valid.nlargest(10, 'val_wis')[['val_wis', 'val_ess', 'iteration']].copy()
        top_wis['metric'] = 'top_wis'
        
        # Top 10 ESS
        top_ess = valid.nlargest(10, 'val_ess')[['val_wis', 'val_ess', 'iteration']].copy()
        top_ess['metric'] = 'top_ess'
        
        combined = pd.concat([top_wis, top_ess])
        combined['version'] = ver
        combined['threshold'] = thresh
        combined['seed'] = seed
        all_data.append(combined)
    except:
        pass

results = pd.concat(all_data, ignore_index=True)
print(f"Total: {len(results)} rows from {results['version'].nunique()} versions")
print(f"\n{'='*70}")
print("ALL rows with val_ess > 200:")
print(f"{'='*70}")
filtered = results[results['val_ess'] > 200].sort_values('val_ess', ascending=False)
if len(filtered) > 0:
    print(filtered[['version','threshold','seed','metric','val_wis','val_ess','iteration']].to_string(index=False))
else:
    print("(none)")

print(f"\n{'='*70}")
print(f"TOP 20 by val_ess:")
print(f"{'='*70}")
print(results.nlargest(20, 'val_ess')[['version','threshold','seed','metric','val_wis','val_ess']].to_string(index=False))
