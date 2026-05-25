import os
import sys
import json
import copy
import argparse
import importlib.util
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch

# =========================================================================
# 真实数据驱动的共现矩阵 (来自 y_train_mh.npy)
# =========================================================================
S_RAW = np.array([
    [1.0000, 0.0001, 0.0020, 0.0297, 0.0002],
    [0.0001, 1.0000, 0.1405, 0.2093, 0.1111],
    [0.0020, 0.1405, 1.0000, 0.1184, 0.2404],
    [0.0297, 0.2093, 0.1184, 1.0000, 0.1157],
    [0.0002, 0.1111, 0.2404, 0.1157, 1.0000],
], dtype=np.float32)

# =========================================================================
# 非线性锐化（Prior Sharpening）: 防止特征同化
# =========================================================================
S_REFINED = np.copy(S_RAW)
np.fill_diagonal(S_REFINED, 0.0)
S_REFINED = np.power(S_REFINED, 2.0)
np.fill_diagonal(S_REFINED, 1.0)

S_IDENTITY = np.eye(5, dtype=np.float32)
ABLATION_THETA = 0.10
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ABLATION_SAVE_DIR = str(REPO_ROOT / "outputs" / "MCKI_core_ablation_outputs_stage8_loco")
DEFAULT_HARD_PAIRS = str(REPO_ROOT / "resources" / "confusable_pairs_v1.csv")

# 测试基准：LP 和 1%、10% 少样本
DEFAULT_PROTOCOLS = ['Full_Finetune', 'Linear_Probing', 'Few_Shot_10%']
DEFAULT_SEEDS = [42, 123, 1024]


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_module_from_path(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load module from: {file_path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def resolve_confusable_csv_path(csv_path: Optional[str], base_script_path: str) -> str:
    candidates = []
    if csv_path:
        candidates.append(csv_path)
        if not os.path.isabs(csv_path):
            candidates.append(str((Path(base_script_path).resolve().parent / csv_path).resolve()))
            candidates.append(str((Path(base_script_path).resolve().parent.parent / csv_path).resolve()))
    candidates.extend([
        str((Path(base_script_path).resolve().parent / 'confusable_pairs_v1.csv').resolve()),
        str((Path(base_script_path).resolve().parent.parent / 'confusable_pairs_v1.csv').resolve()),
        str((Path(base_script_path).resolve().parent / 'src3' / 'confusable_pairs_v1.csv').resolve()),
        str((Path(base_script_path).resolve().parent.parent / 'src3' / 'confusable_pairs_v1.csv').resolve()),
    ])
    for p in candidates:
        if p and os.path.exists(p):
            return p
    raise FileNotFoundError(f'Cannot find confusable_pairs_v1.csv. Checked: {candidates}')


def build_variant_cfgs(base_cfg: Dict) -> List[Tuple[str, Dict]]:
    variants: List[Tuple[str, Dict]] = []

    # 1) w/o_Relation_Matri
    # x
    # 去掉显式类别关系建模，但保留 Stage8 的结构增强模块
    cfg = copy.deepcopy(base_cfg)
    cfg['relation_matrix_values'] = S_IDENTITY.tolist()
    cfg['lambda_prior'] = 1.0
    cfg['lambda_conf'] = 0.0
    cfg['use_continuous_weights'] = False
    cfg['hard_negative_threshold'] = 1.1
    variants.append(('w/o_Relation_Matrix', cfg))

    # 2) w/o_Dynamic_Correction
    # 保留静态临床关系先验，但关闭数据驱动动态修正
    cfg = copy.deepcopy(base_cfg)
    cfg['relation_matrix_values'] = S_RAW.tolist()
    cfg['lambda_prior'] = 1.0
    cfg['lambda_conf'] = 0.0
    cfg['use_continuous_weights'] = True
    cfg['hard_negative_threshold'] = ABLATION_THETA
    variants.append(('w/o_Dynamic_Correction', cfg))

    # 3) w/o_Lead_Aware
    # 去掉 lead-aware 输入建模
    cfg = copy.deepcopy(base_cfg)
    cfg['use_lead_aware_input'] = False
    variants.append(('w/o_Lead_Aware', cfg))

    # 4) w/o_Multiscale_Local_Branch
    # 去掉 multiscale local branch
    cfg = copy.deepcopy(base_cfg)
    cfg['use_multiscale_local_branch'] = False
    variants.append(('w/o_Multiscale_Local_Branch', cfg))


    return variants


def summarize_rows(rows: List[Dict]) -> pd.DataFrame:
    metric_cols = [
        'test_Macro_AUC', 'test_AUPRC', 'test_Macro_F1',
        'test_MI_F1', 'test_HNDR_Pair', 'test_HNDR_Inst'
    ]
    out = []
    df = pd.DataFrame(rows)
    for method, sub in df.groupby('mode_name', sort=False):
        mean = sub[metric_cols].mean()
        std = sub[metric_cols].std().fillna(0.0)
        out.append({
            'Method': method,
            'Macro_AUC': f"{mean['test_Macro_AUC']:.4f} ± {std['test_Macro_AUC']:.4f}",
            'AUPRC': f"{mean['test_AUPRC']:.4f} ± {std['test_AUPRC']:.4f}",
            'F1': f"{mean['test_Macro_F1']:.4f} ± {std['test_Macro_F1']:.4f}",
            'MI_F1': f"{mean['test_MI_F1']:.4f} ± {std['test_MI_F1']:.4f}",
            'HNDR_Pair': f"{mean['test_HNDR_Pair']:.4f} ± {std['test_HNDR_Pair']:.4f}",
            'HNDR_Inst': f"{mean['test_HNDR_Inst']:.4f} ± {std['test_HNDR_Inst']:.4f}",
        })
    return pd.DataFrame(out)


def patch_base_module_for_absolute_hndr(base_module, abs_csv_path: str):
    import numpy as _np
    import pandas as _pd
    from sklearn.metrics import accuracy_score as _accuracy_score

    def _patched_calculate_hndr(probs, targets, class_names, csv_path='confusable_pairs_v1.csv'):
        if not os.path.exists(abs_csv_path):
            return None, None
        pairs_df = _pd.read_csv(abs_csv_path)
        pair_metrics = {}
        total_hard_samples = 0
        correct_hard_samples = 0
        for _, row in pairs_df.iterrows():
            disease_a = str(row['disease_a']).strip()
            disease_b = str(row['disease_b']).strip()
            if disease_a not in class_names or disease_b not in class_names:
                continue
            idx_a = class_names.index(disease_a)
            idx_b = class_names.index(disease_b)
            condition_a = (targets[:, idx_a] >= 0.5) & (targets[:, idx_b] < 0.5)
            condition_b = (targets[:, idx_a] < 0.5) & (targets[:, idx_b] >= 0.5)
            valid_mask = condition_a | condition_b
            subset_targets = targets[valid_mask]
            subset_probs = probs[valid_mask]
            if len(subset_targets) == 0:
                continue
            binary_targets = (subset_targets[:, idx_a] >= 0.5).astype(int)
            binary_preds = (subset_probs[:, idx_a] > subset_probs[:, idx_b]).astype(int)
            acc = _accuracy_score(binary_targets, binary_preds)
            pair_metrics[f'{disease_a}_vs_{disease_b}'] = acc
            total_hard_samples += len(binary_targets)
            correct_hard_samples += _np.sum(binary_preds == binary_targets)
        if not pair_metrics:
            return None, None
        hndr_pair = _np.mean(list(pair_metrics.values()))
        hndr_inst = correct_hard_samples / total_hard_samples if total_hard_samples > 0 else 0.0
        return hndr_pair, hndr_inst

    def _patched_evaluate_from_probs(probs: np.ndarray, targets: np.ndarray, thresholds: Optional[np.ndarray] = None):
        if thresholds is None:
            thresholds = np.full(probs.shape[1], 0.5, dtype=np.float32)
        preds = (probs >= thresholds.reshape(1, -1)).astype(int)
        macro_auc = base_module.roc_auc_score(targets, probs, average='macro')
        auprc = base_module.average_precision_score(targets, probs, average='macro')
        macro_f1 = base_module.f1_score(targets, preds, average='macro', zero_division=0)
        idx_mi = base_module.CLASS_NAMES.index('MI')
        mi_f1 = base_module.f1_score(targets[:, idx_mi], preds[:, idx_mi], zero_division=0)
        hndr_pair, hndr_inst = _patched_calculate_hndr(probs, targets, base_module.CLASS_NAMES, abs_csv_path)
        return {
            'Macro_AUC': float(macro_auc),
            'AUPRC': float(auprc),
            'Macro_F1': float(macro_f1),
            'MI_F1': float(mi_f1),
            'HNDR_Pair': float(0.0 if hndr_pair is None else hndr_pair),
            'HNDR_Inst': float(0.0 if hndr_inst is None else hndr_inst),
        }

    base_module.calculate_hndr = _patched_calculate_hndr
    base_module.evaluate_from_probs = _patched_evaluate_from_probs


def run_ablation(base_module, protocol: str, seeds: List[int], save_dir: str):
    base_cfg = copy.deepcopy(base_module.CFG)
    variants = build_variant_cfgs(base_cfg)
    raw_rows: List[Dict] = []

    original_protocols = copy.deepcopy(base_module.DEFAULT_PROTOCOLS)
    try:
        base_module.DEFAULT_PROTOCOLS = [protocol]

        print('\n' + '=' * 100)
        print(f'🚀 Stage8 (Multiscale+LeadAware) Ablation | protocol={protocol}')
        print(f'Base script : {base_module.__file__}')
        print(f'Seeds       : {seeds}')
        print('=' * 100)

        for variant_name, variant_cfg in variants:
            variant_root = os.path.join(save_dir, protocol.replace('%', 'pct'), variant_name)
            ensure_dir(variant_root)

            print(f'\n----> Running variant: {variant_name}')
            variant_rows = []

            for seed in seeds:
                rows = base_module.run_single_seed(
                    seed=int(seed),
                    cfg=copy.deepcopy(variant_cfg),
                    save_dir=variant_root,
                )
                for row in rows:
                    row = dict(row)
                    row['mode_name'] = variant_name
                    row['ablation_variant'] = variant_name
                    row['base_script'] = base_module.__file__
                    raw_rows.append(row)
                    variant_rows.append(row)

            with open(os.path.join(variant_root, 'used_config.json'), 'w', encoding='utf-8') as f:
                json.dump(variant_cfg, f, indent=2, ensure_ascii=False)

            print('\n' + '-' * 80)
            print(f'🟢 [实时出分] {variant_name} 在 {protocol} 上的结果：')
            df_variant_summary = summarize_rows(variant_rows)
            print(df_variant_summary.to_string(index=False))
            print('-' * 80 + '\n')

    finally:
        base_module.DEFAULT_PROTOCOLS = original_protocols

    df_raw = pd.DataFrame(raw_rows)
    df_summary = summarize_rows(raw_rows)

    ensure_dir(save_dir)
    safe_protocol = protocol.replace('%', 'pct')
    raw_path = os.path.join(save_dir, f'MCKI_core_ablation_raw_{safe_protocol}.csv')
    summary_path = os.path.join(save_dir, f'MCKI_core_ablation_summary_{safe_protocol}.csv')
    cfg_path = os.path.join(save_dir, f'MCKI_core_ablation_used_configs_{safe_protocol}.json')

    df_raw.to_csv(raw_path, index=False, encoding='utf-8-sig')
    df_summary.to_csv(summary_path, index=False, encoding='utf-8-sig')
    with open(cfg_path, 'w', encoding='utf-8') as f:
        json.dump({
            'protocol': protocol,
            'seeds': [int(s) for s in seeds],
            'base_script': base_module.__file__,
            'S_REFINED': S_REFINED.tolist(),
            'variants': {name: cfg for name, cfg in variants},
        }, f, indent=2, ensure_ascii=False)

    print('\n' + '★' * 100)
    print(f'🏆 Stage 8 Ablation Summary [{protocol}]')
    print('★' * 100)
    print(df_summary.to_string(index=False))
    print(f'\n[Saved] {summary_path}')


def build_argparser():
    parser = argparse.ArgumentParser(description='Stage8 MCKI ablation runner (Fair & LOCO Edition)')
    # 默认指向刚刚做好的 Stage 8 5协议主脚本
    parser.add_argument('--base-script', type=str,
                        default='run_stage8_leadaware_multiscale_MCKI_5protocols.py')
    parser.add_argument('--protocols', nargs='+', default=DEFAULT_PROTOCOLS,
                        choices=['Full_Finetune', 'Linear_Probing', 'Few_Shot_1%', 'Few_Shot_10%', 'Few_Shot_25%'])
    parser.add_argument('--seeds', nargs='+', type=int, default=DEFAULT_SEEDS)
    parser.add_argument('--save-dir', type=str, default=DEFAULT_ABLATION_SAVE_DIR)
    parser.add_argument('--csv-path', type=str, default=DEFAULT_HARD_PAIRS)
    return parser


def main():
    args = build_argparser().parse_args()

    base_script_path = args.base_script
    if not os.path.isabs(base_script_path):
        base_script_path = str((Path(__file__).resolve().parent / base_script_path).resolve())
    if not os.path.exists(base_script_path):
        raise FileNotFoundError(f'Base script not found: {base_script_path}')

    abs_csv_path = resolve_confusable_csv_path(args.csv_path, base_script_path)

    # 动态加载 Stage 8 主文件
    base_module = load_module_from_path('stage8_ablation_base_module', base_script_path)
    patch_base_module_for_absolute_hndr(base_module, abs_csv_path)

    for protocol in args.protocols:
        run_ablation(
            base_module=base_module,
            protocol=protocol,
            seeds=[int(s) for s in args.seeds],
            save_dir=args.save_dir,
        )


if __name__ == '__main__':
    main()