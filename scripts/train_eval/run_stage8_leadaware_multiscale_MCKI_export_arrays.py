import argparse
import copy
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch


def load_module_from_path(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load module from: {file_path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def safe_protocol_name(protocol: str) -> str:
    return protocol.replace('%', 'pct')


def patch_hndr_csv(base_module, csv_path: str):
    orig_calculate_hndr = base_module.calculate_hndr

    def _calculate_hndr_abs(probs, targets, class_names, csv_path_inner=csv_path):
        return orig_calculate_hndr(probs, targets, class_names, csv_path_inner)

    base_module.calculate_hndr = _calculate_hndr_abs


def run_single_seed_with_exports(base_module, seed: int, cfg: Dict, save_dir: str, array_root: str) -> List[Dict]:
    base_module.seed_everything(seed)
    ensure_dir(save_dir)
    ensure_dir(array_root)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    base_train_loader, base_val_loader, base_test_loader = base_module.get_dataloader_v3(
        base_module.DEFAULT_DATA_DIR,
        batch_size=cfg['batch_size'],
        num_workers=cfg['num_workers'],
    )
    train_dataset = base_train_loader.dataset
    val_dataset = base_val_loader.dataset
    test_dataset = base_test_loader.dataset

    model = base_module.build_model(cfg, device)

    prior = base_module.load_prior_matrix(cfg.get('relation_matrix_values'))
    warmup_epochs = min(cfg['warmup_pretrain_epochs'], cfg['pretrain_epochs'])
    hybrid_epochs = max(0, cfg['pretrain_epochs'] - warmup_epochs)

    print(
        f'\n🚀 [Seed {seed}] Stage8-LeadAware-MultiScale-MCKI-EXPORT | warmup={warmup_epochs} | hybrid={hybrid_epochs}'
    )
    base_module._print_matrix_stats('S_prior', prior)

    model, rhythm_projector, local_projector = base_module.pretrain_with_leadaware_multiscale_relation(
        model=model,
        train_loader=base_train_loader,
        cfg=cfg,
        device=device,
        relation_matrix=prior,
        epochs=warmup_epochs,
        rhythm_projector=None,
        local_projector=None,
        stage_desc='MCKI Stage8 Warmup(prior+leadaware+multiscale)',
    )

    conf = base_module.bootstrap_confusion_matrix(model, base_train_loader, base_val_loader, cfg, device)
    base_module._print_matrix_stats('S_conf', conf)

    hybrid = base_module.blend_relation_matrices(prior, conf, cfg['lambda_prior'], cfg['lambda_conf'])
    base_module._print_matrix_stats('S_hybrid_lambda05', hybrid)

    seed_dir = os.path.join(save_dir, f'seed_{seed}')
    ensure_dir(seed_dir)
    base_module.save_relation_artifacts(seed_dir, prior, conf, hybrid)
    with open(os.path.join(seed_dir, 'leadaware_multiscale_config.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'lambda_prior': float(cfg['lambda_prior']),
            'lambda_conf': float(cfg['lambda_conf']),
            'use_multiscale_local_branch': bool(cfg['use_multiscale_local_branch']),
            'local_window_size': int(cfg['local_window_size']),
            'local_num_windows': int(cfg['local_num_windows']),
            'local_jitter': int(cfg['local_jitter']),
            'local_loss_weight': float(cfg['local_loss_weight']),
            'align_loss_weight': float(cfg['align_loss_weight']),
            'use_lead_aware_input': bool(cfg['use_lead_aware_input']),
            'lead_emb_dim': int(cfg['lead_emb_dim']),
            'use_dynamic_lead_mask': bool(cfg['use_dynamic_lead_mask']),
            'lead_mask_prob': float(cfg['lead_mask_prob']),
            'lead_mask_min_drop': int(cfg['lead_mask_min_drop']),
            'lead_mask_max_drop': int(cfg['lead_mask_max_drop']),
            'warmup_epochs': int(warmup_epochs),
            'hybrid_epochs': int(hybrid_epochs),
        }, f, ensure_ascii=False, indent=2)

    if hybrid_epochs > 0:
        model, rhythm_projector, local_projector = base_module.pretrain_with_leadaware_multiscale_relation(
            model=model,
            train_loader=base_train_loader,
            cfg=cfg,
            device=device,
            relation_matrix=hybrid,
            epochs=hybrid_epochs,
            rhythm_projector=rhythm_projector,
            local_projector=local_projector,
            stage_desc='MCKI Stage8 Hybrid(lambda05+leadaware+multiscale)',
        )

    rows = []
    pretrained_state = copy.deepcopy(model.state_dict())

    for protocol in base_module.DEFAULT_PROTOCOLS:
        proto_train_loader, proto_val_loader, proto_test_loader, few_shot_indices = base_module.prepare_protocol_loaders_from_base(
            train_dataset, val_dataset, test_dataset, cfg, protocol, seed
        )

        proto_model = base_module.build_model(cfg, device)
        proto_model.load_state_dict(pretrained_state)
        proto_model, thresholds = base_module.train_protocol(
            proto_model, proto_train_loader, proto_val_loader, cfg, device, protocol
        )
        val_probs, val_targets = base_module.collect_probs(proto_model, proto_val_loader, device)
        test_probs, test_targets = base_module.collect_probs(proto_model, proto_test_loader, device)
        val_metrics = base_module.evaluate_from_probs(val_probs, val_targets, thresholds)
        test_metrics = base_module.evaluate_from_probs(test_probs, test_targets, thresholds)

        protocol_dir = os.path.join(array_root, safe_protocol_name(protocol), f'seed_{seed}')
        ensure_dir(protocol_dir)
        np.save(os.path.join(protocol_dir, 'test_probs.npy'), test_probs)
        np.save(os.path.join(protocol_dir, 'test_targets.npy'), test_targets)
        np.save(os.path.join(protocol_dir, 'thresholds.npy'), thresholds)
        np.save(os.path.join(protocol_dir, 'val_probs.npy'), val_probs)
        np.save(os.path.join(protocol_dir, 'val_targets.npy'), val_targets)

        meta = {
            'seed': int(seed),
            'protocol': protocol,
            'protocol_safe': safe_protocol_name(protocol),
            'n_train_samples': int(len(proto_train_loader.dataset)),
            'few_shot_indices': [int(x) for x in (few_shot_indices or [])],
            'class_names': list(base_module.CLASS_NAMES),
        }
        with open(os.path.join(protocol_dir, 'meta.json'), 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        row = {
            'seed': seed,
            'protocol': protocol,
            'n_train_samples': len(proto_train_loader.dataset),
            'few_shot_indices': json.dumps(few_shot_indices, ensure_ascii=False) if few_shot_indices is not None else '',
            'export_run_dir': protocol_dir,
            **{f'val_{k}': v for k, v in val_metrics.items()},
            **{f'test_{k}': v for k, v in test_metrics.items()},
        }
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(
        description='Stage8 rerun file that additionally saves test_probs/test_targets/thresholds for pair analysis and score-distribution plots.'
    )
    ap.add_argument('--base-script', type=str, default='run_stage8_leadaware_multiscale_MCKI_5protocols.py')
    ap.add_argument('--data-dir', type=str, default='./data/processed_v3')
    ap.add_argument('--save-dir', type=str, default='./formal_eval_outputs_stage8_leadaware_multiscale_MCKI_export_arrays')
    ap.add_argument('--array-root', type=str, default='./stage8_saved_arrays')
    ap.add_argument('--protocols', nargs='+', default=['Linear_Probing', 'Few_Shot_10%'])
    ap.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 1024])
    ap.add_argument('--csv-path', type=str, default='confusable_pairs_v1.csv')
    args = ap.parse_args()

    base_script_path = args.base_script
    if not os.path.isabs(base_script_path):
        base_script_path = str((Path.cwd() / base_script_path).resolve())
    if not os.path.exists(base_script_path):
        raise FileNotFoundError(f'Base script not found: {base_script_path}')

    base_module = load_module_from_path('stage8_export_base_module', base_script_path)
    base_module.DEFAULT_PROTOCOLS = list(args.protocols)
    base_module.DEFAULT_SEEDS = [int(s) for s in args.seeds]
    base_module.DEFAULT_DATA_DIR = args.data_dir
    base_module.DEFAULT_SAVE_DIR = args.save_dir
    patch_hndr_csv(base_module, args.csv_path)

    ensure_dir(args.save_dir)
    ensure_dir(args.array_root)

    print('\n' + '▼' * 80)
    print('🔬 启动 Stage8 导出版：自动保存 test_probs / test_targets / thresholds')
    print(f'   Base script = {base_script_path}')
    print(f'   Data dir    = {base_module.DEFAULT_DATA_DIR}')
    print(f'   Save dir    = {base_module.DEFAULT_SAVE_DIR}')
    print(f'   Array root  = {args.array_root}')
    print(f'   Protocols   = {base_module.DEFAULT_PROTOCOLS}')
    print(f'   Seeds       = {base_module.DEFAULT_SEEDS}')
    print(f'   HNDR csv    = {args.csv_path}')
    print(json.dumps(base_module.CFG, indent=2, ensure_ascii=False))
    print('▲' * 80)

    all_rows = []
    for seed in base_module.DEFAULT_SEEDS:
        rows = run_single_seed_with_exports(base_module, seed, base_module.CFG, args.save_dir, args.array_root)
        all_rows.extend(rows)
        pd.DataFrame(all_rows).to_csv(os.path.join(args.save_dir, 'per_seed_results.csv'), index=False)

    base_module.summarize_and_save(all_rows, args.save_dir)
    print(f'✅ Results saved to {args.save_dir}')
    print(f'✅ Export arrays saved to {args.array_root}')


if __name__ == '__main__':
    main()
