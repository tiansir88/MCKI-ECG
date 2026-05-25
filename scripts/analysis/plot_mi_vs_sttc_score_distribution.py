import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

DEFAULT_CLASS_NAMES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']


def load_run(run_dir: Path):
    probs = np.load(run_dir / 'test_probs.npy')
    targets = np.load(run_dir / 'test_targets.npy')
    meta_path = run_dir / 'meta.json'
    meta = json.loads(meta_path.read_text(encoding='utf-8')) if meta_path.exists() else {}
    return probs, targets, meta


def collect_pair_margin(probs, targets, class_names, pos_name='MI', neg_name='STTC'):
    idx_pos = class_names.index(pos_name)
    idx_neg = class_names.index(neg_name)
    mask_pos = (targets[:, idx_pos] >= 0.5) & (targets[:, idx_neg] < 0.5)
    mask_neg = (targets[:, idx_neg] >= 0.5) & (targets[:, idx_pos] < 0.5)
    pos_margin = probs[mask_pos, idx_pos] - probs[mask_pos, idx_neg]
    neg_margin = probs[mask_neg, idx_pos] - probs[mask_neg, idx_neg]
    return pos_margin, neg_margin


def main():
    ap = argparse.ArgumentParser(description='Plot MI-vs-STTC score distributions from saved test arrays.')
    ap.add_argument('--root', type=str, required=True, help='Root dir that contains protocol/seed run dirs')
    ap.add_argument('--protocol', type=str, required=True, help='Protocol subdir to plot, e.g. Linear_Probing or Few_Shot_10pct')
    ap.add_argument('--pos-name', type=str, default='MI')
    ap.add_argument('--neg-name', type=str, default='STTC')
    ap.add_argument('--class-names', nargs='+', default=DEFAULT_CLASS_NAMES)
    ap.add_argument('--output-dir', type=str, default='supp_mi_vs_sttc_plot')
    args = ap.parse_args()

    protocol_dir = Path(args.root) / args.protocol
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pos_all, neg_all = [], []
    for run_dir in sorted([p for p in protocol_dir.iterdir() if p.is_dir()]):
        if not ((run_dir / 'test_probs.npy').exists() and (run_dir / 'test_targets.npy').exists()):
            continue
        probs, targets, meta = load_run(run_dir)
        pos_margin, neg_margin = collect_pair_margin(probs, targets, args.class_names, args.pos_name, args.neg_name)
        if len(pos_margin):
            pos_all.append(pos_margin)
        if len(neg_margin):
            neg_all.append(neg_margin)

    if not pos_all or not neg_all:
        raise RuntimeError('No valid MI-vs-STTC pair subset found for this protocol.')

    pos_all = np.concatenate(pos_all)
    neg_all = np.concatenate(neg_all)

    plt.figure(figsize=(7, 5))
    bins = np.linspace(-1.0, 1.0, 50)
    plt.hist(pos_all, bins=bins, alpha=0.60, density=True, label=f'{args.pos_name}+ / {args.neg_name}-')
    plt.hist(neg_all, bins=bins, alpha=0.60, density=True, label=f'{args.neg_name}+ / {args.pos_name}-')
    plt.axvline(0.0, linestyle='--', linewidth=1.0)
    plt.xlabel(f'Margin: p({args.pos_name}) - p({args.neg_name})')
    plt.ylabel('Density')
    plt.title(f'{args.pos_name} vs {args.neg_name} score distribution ({args.protocol})')
    plt.legend()
    plt.tight_layout()

    png_path = out_dir / f'{args.pos_name.lower()}_vs_{args.neg_name.lower()}_{args.protocol}_score_distribution.png'
    plt.savefig(png_path, dpi=220)
    plt.close()

    summary = {
        'protocol': args.protocol,
        'positive_pair': f'{args.pos_name}+ / {args.neg_name}-',
        'negative_pair': f'{args.neg_name}+ / {args.pos_name}-',
        'n_positive_subset': int(len(pos_all)),
        'n_negative_subset': int(len(neg_all)),
        'mean_margin_positive_subset': float(pos_all.mean()),
        'mean_margin_negative_subset': float(neg_all.mean()),
    }
    (out_dir / f'{args.pos_name.lower()}_vs_{args.neg_name.lower()}_{args.protocol}_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    print(f'[OK] Saved plot -> {png_path}')


if __name__ == '__main__':
    main()
