import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

DEFAULT_CLASS_NAMES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']


def load_run(run_dir: Path):
    probs = np.load(run_dir / 'test_probs.npy')
    targets = np.load(run_dir / 'test_targets.npy')
    thresholds = np.load(run_dir / 'thresholds.npy')
    meta_path = run_dir / 'meta.json'
    meta = json.loads(meta_path.read_text(encoding='utf-8')) if meta_path.exists() else {}
    return probs, targets, thresholds, meta


def compute_pair_metrics(probs, targets, thresholds, class_names, a_name, b_name):
    idx_a = class_names.index(a_name)
    idx_b = class_names.index(b_name)

    mask_a = (targets[:, idx_a] >= 0.5) & (targets[:, idx_b] < 0.5)
    mask_b = (targets[:, idx_b] >= 0.5) & (targets[:, idx_a] < 0.5)
    mask = mask_a | mask_b
    n = int(mask.sum())
    if n == 0:
        return None

    sub_probs = probs[mask]
    sub_targets = targets[mask]
    y_true = (sub_targets[:, idx_a] >= 0.5).astype(int)
    # pairwise score: positive => more MI-like, negative => more STTC-like
    score = sub_probs[:, idx_a] - sub_probs[:, idx_b]
    auc = roc_auc_score(y_true, score)

    # thresholded pairwise F1 using one-vs-one decision
    y_pred = (sub_probs[:, idx_a] > sub_probs[:, idx_b]).astype(int)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    thr_a = float(thresholds[idx_a])
    thr_b = float(thresholds[idx_b])
    # calibrated decision with per-class thresholds
    margin_pred = ((sub_probs[:, idx_a] - thr_a) > (sub_probs[:, idx_b] - thr_b)).astype(int)
    f1_thr = f1_score(y_true, margin_pred, zero_division=0)

    return {
        'Pair': f'{a_name}_vs_{b_name}',
        'n_samples': n,
        'pair_AUC': float(auc),
        'pair_F1_raw': float(f1),
        'pair_F1_thresholded': float(f1_thr),
    }


def summarize(rows):
    df = pd.DataFrame(rows)
    key_cols = ['pair_AUC', 'pair_F1_raw', 'pair_F1_thresholded']
    group_cols = ['Protocol', 'Pair', 'Method']
    out = []
    for keys, sub in df.groupby(group_cols, sort=False):
        rec = dict(zip(group_cols, keys))
        rec['n_runs'] = int(len(sub))
        rec['n_samples_mean'] = float(sub['n_samples'].mean())
        for c in key_cols:
            rec[c] = f"{sub[c].mean():.4f} ± {sub[c].std(ddof=1):.4f}"
        out.append(rec)
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser(description='Build confusable-pair subset AUC/F1 tables from saved test arrays.')
    ap.add_argument('--root', type=str, required=True, help='Root dir that contains protocol/seed run dirs with test_probs.npy/test_targets.npy/thresholds.npy')
    ap.add_argument('--pairs', nargs='+', default=['MI:STTC', 'MI:CD', 'STTC:HYP'])
    ap.add_argument('--class-names', nargs='+', default=DEFAULT_CLASS_NAMES)
    ap.add_argument('--method-name', type=str, default='MCKI(stage8)')
    ap.add_argument('--output-dir', type=str, default='supp_confusable_pair_table')
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for protocol_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        protocol = protocol_dir.name
        for run_dir in sorted([p for p in protocol_dir.iterdir() if p.is_dir()]):
            probs_path = run_dir / 'test_probs.npy'
            targets_path = run_dir / 'test_targets.npy'
            thresholds_path = run_dir / 'thresholds.npy'
            if not (probs_path.exists() and targets_path.exists() and thresholds_path.exists()):
                continue
            probs, targets, thresholds, meta = load_run(run_dir)
            seed = meta.get('seed', run_dir.name)
            for pair in args.pairs:
                a_name, b_name = pair.split(':', 1)
                rec = compute_pair_metrics(probs, targets, thresholds, args.class_names, a_name, b_name)
                if rec is None:
                    continue
                rec['Protocol'] = protocol
                rec['Seed'] = seed
                rec['Method'] = args.method_name
                rows.append(rec)

    if not rows:
        raise RuntimeError('No valid runs found. Make sure Stage8 saved test_probs.npy/test_targets.npy/thresholds.npy.')

    raw_df = pd.DataFrame(rows)
    raw_df.to_csv(out_dir / 'confusable_pair_metrics_raw.csv', index=False, encoding='utf-8-sig')
    summary_df = summarize(rows)
    summary_df.to_csv(out_dir / 'confusable_pair_metrics_summary.csv', index=False, encoding='utf-8-sig')

    md_lines = ['# Confusable-pair subset metrics\n']
    for protocol, sub in summary_df.groupby('Protocol', sort=False):
        md_lines.append(f'## {protocol}\n')
        md_lines.append('| Pair | Method | pair_AUC | pair_F1_raw | pair_F1_thresholded | n_runs | n_samples_mean |')
        md_lines.append('|---|---|---:|---:|---:|---:|---:|')
        for _, r in sub.iterrows():
            md_lines.append(
                f"| {r['Pair']} | {r['Method']} | {r['pair_AUC']} | {r['pair_F1_raw']} | {r['pair_F1_thresholded']} | {r['n_runs']} | {r['n_samples_mean']:.1f} |"
            )
        md_lines.append('')
    (out_dir / 'confusable_pair_metrics_summary.md').write_text('\n'.join(md_lines), encoding='utf-8')
    print(f'[OK] Saved -> {out_dir}')


if __name__ == '__main__':
    main()
