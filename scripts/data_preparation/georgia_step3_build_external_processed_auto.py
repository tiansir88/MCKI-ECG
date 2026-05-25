#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
from scipy.io import loadmat

PROJECT_ROOT = Path("/root/gpufree-data/MCKI_Project")
RAW_ROOT = PROJECT_ROOT / "data" / "external_raw"
IN_DIR = PROJECT_ROOT / "data" / "processed" / "georgia"
OUT_DIR = PROJECT_ROOT / "data" / "external_processed" / "georgia_5class"

MANIFEST_FILE = IN_DIR / "manifest_ptbxl5.csv"
AUDIT_FILE = IN_DIR / "raw_audit.csv"

TARGET_FS = 100
TARGET_LEN = 1000
NUM_LEADS = 12
CLASS_NAMES = ["NORM", "MI", "STTC", "CD", "HYP"]

OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_waveform_from_mat(mat_path: Path) -> np.ndarray:
    obj = loadmat(str(mat_path))

    arr = None
    if "val" in obj:
        arr = obj["val"]
    else:
        for k, v in obj.items():
            if k.startswith("__"):
                continue
            if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.number) and v.ndim == 2:
                arr = v
                break

    if arr is None:
        raise ValueError(f"No usable 2D numeric array found in {mat_path}")

    arr = np.asarray(arr, dtype=np.float32)

    if arr.ndim != 2:
        raise ValueError(f"Unexpected ndim={arr.ndim} in {mat_path}")

    if arr.shape[0] == NUM_LEADS and arr.shape[1] != NUM_LEADS:
        arr = arr.T
    elif arr.shape[1] == NUM_LEADS:
        pass
    else:
        raise ValueError(f"Unexpected waveform shape {arr.shape} in {mat_path}")

    if arr.shape[1] != NUM_LEADS:
        raise ValueError(f"Lead dim mismatch after transpose: {arr.shape}")

    return arr.astype(np.float32)


def parse_gain_baseline_from_hea(hea_path: Path):
    """
    从 .hea 每个导联行解析 gain 和 baseline。
    典型行:
    E00002.mat 16x1+24 1000.0(0)/mV 16 0 -29 22732 0 I

    第 3 个 token 里:
    1000.0(0)/mV
      gain=1000.0
      baseline=0
    """
    text = hea_path.read_text(encoding="utf-8", errors="ignore")
    lines = [x.strip() for x in text.splitlines() if x.strip()]

    if len(lines) < 1 + NUM_LEADS:
        raise ValueError(f"Header too short: {hea_path}")

    signal_lines = lines[1:1 + NUM_LEADS]

    gains = []
    baselines = []
    lead_names = []

    for ln in signal_lines:
        parts = ln.split()
        if len(parts) < 9:
            raise ValueError(f"Malformed signal line in {hea_path}: {ln}")

        gain_token = parts[2]
        lead_name = parts[-1]

        # 支持类似:
        # 1000.0(0)/mV
        # 200.0/mV
        m = re.match(r"^([+-]?\d+(?:\.\d+)?)(?:\(([+-]?\d+)\))?(?:/.*)?$", gain_token)
        if not m:
            raise ValueError(f"Cannot parse gain token '{gain_token}' in {hea_path}")

        gain = float(m.group(1))
        baseline = float(m.group(2)) if m.group(2) is not None else 0.0

        if abs(gain) < 1e-12:
            raise ValueError(f"Bad gain={gain} in {hea_path}")

        gains.append(gain)
        baselines.append(baseline)
        lead_names.append(lead_name)

    gains = np.asarray(gains, dtype=np.float32)
    baselines = np.asarray(baselines, dtype=np.float32)

    if gains.shape[0] != NUM_LEADS or baselines.shape[0] != NUM_LEADS:
        raise ValueError(f"gain/baseline count mismatch in {hea_path}")

    return gains, baselines, lead_names


def digital_to_physical_mV(x_digital: np.ndarray, gains: np.ndarray, baselines: np.ndarray) -> np.ndarray:
    """
    x_digital: (T, 12)
    gains:     (12,)
    baselines: (12,)
    输出 physical units，通常是 mV
    """
    return ((x_digital - baselines[None, :]) / gains[None, :]).astype(np.float32)


def resample_timefirst_numpy(x: np.ndarray, orig_fs: float, target_fs: float) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError(f"x must be 2D, got {x.shape}")

    if orig_fs <= 0 or target_fs <= 0:
        raise ValueError(f"Bad sampling rate: orig_fs={orig_fs}, target_fs={target_fs}")

    n_old, n_ch = x.shape
    if n_old == 0:
        raise ValueError("Empty signal")

    if abs(orig_fs - target_fs) < 1e-8:
        return x.copy()

    n_new = int(round(n_old * target_fs / orig_fs))
    n_new = max(n_new, 1)

    old_idx = np.arange(n_old, dtype=np.float32)
    new_idx = np.linspace(0, n_old - 1, n_new, dtype=np.float32)

    y = np.empty((n_new, n_ch), dtype=np.float32)
    for c in range(n_ch):
        y[:, c] = np.interp(new_idx, old_idx, x[:, c]).astype(np.float32)

    return y


def fix_length_pad_right(x: np.ndarray, target_len: int) -> np.ndarray:
    t, c = x.shape

    if t == target_len:
        return x.astype(np.float32)

    if t > target_len:
        return x[:target_len].astype(np.float32)

    out = np.zeros((target_len, c), dtype=np.float32)
    out[:t] = x
    return out


def make_label_vector(row: pd.Series) -> np.ndarray:
    return np.array([int(row[k]) for k in CLASS_NAMES], dtype=np.float32)


def main():
    if not MANIFEST_FILE.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_FILE}")
    if not AUDIT_FILE.exists():
        raise FileNotFoundError(f"Audit not found: {AUDIT_FILE}")

    manifest = pd.read_csv(MANIFEST_FILE)
    audit = pd.read_csv(AUDIT_FILE)

    audit = audit.drop_duplicates(subset=["record_path"]).copy()
    merged = manifest.merge(
        audit[["record_path", "fs", "sig_len", "age", "sex"]],
        on="record_path",
        how="left"
    )

    xs = []
    ys = []
    meta_rows = []
    log_rows = []

    total = len(merged)
    ok = 0
    bad = 0

    print(f"[INFO] total manifest records = {total}")

    for i, row in merged.iterrows():
        record_path = str(row["record_path"]).strip()
        mat_path = RAW_ROOT / f"{record_path}.mat"
        hea_path = RAW_ROOT / f"{record_path}.hea"

        try:
            if not mat_path.exists():
                raise FileNotFoundError(f"mat not found: {mat_path}")
            if not hea_path.exists():
                raise FileNotFoundError(f"hea not found: {hea_path}")

            orig_fs = float(row["fs"])
            if np.isnan(orig_fs):
                raise ValueError(f"Missing fs for {record_path}")

            x_digital = load_waveform_from_mat(mat_path)   # (T, 12)
            orig_shape = tuple(x_digital.shape)

            gains, baselines, lead_names = parse_gain_baseline_from_hea(hea_path)
            x = digital_to_physical_mV(x_digital, gains, baselines)   # (T, 12), mV
            physical_shape = tuple(x.shape)

            x = resample_timefirst_numpy(x, orig_fs=orig_fs, target_fs=TARGET_FS)
            resampled_shape = tuple(x.shape)

            x = fix_length_pad_right(x, TARGET_LEN)
            final_shape = tuple(x.shape)

            if x.shape != (TARGET_LEN, NUM_LEADS):
                raise ValueError(f"Final shape mismatch: {x.shape}")

            y = make_label_vector(row)

            xs.append(x.astype(np.float32))
            ys.append(y.astype(np.float32))

            meta_rows.append({
                "record_path": record_path,
                "mat_path": str(mat_path),
                "hea_path": str(hea_path),
                "labels": row.get("labels", ""),
                "NORM": int(row["NORM"]),
                "MI": int(row["MI"]),
                "STTC": int(row["STTC"]),
                "CD": int(row["CD"]),
                "HYP": int(row["HYP"]),
                "orig_fs": orig_fs,
                "orig_sig_len_header": None if pd.isna(row["sig_len"]) else int(row["sig_len"]),
                "orig_shape": json.dumps(orig_shape),
                "physical_shape": json.dumps(physical_shape),
                "resampled_shape": json.dumps(resampled_shape),
                "final_shape": json.dumps(final_shape),
                "gains": json.dumps(gains.tolist()),
                "baselines": json.dumps(baselines.tolist()),
                "lead_names": json.dumps(lead_names),
                "age": row.get("age", ""),
                "sex": row.get("sex", ""),
                "status": "ok",
            })

            log_rows.append({
                "record_path": record_path,
                "status": "ok",
                "error": "",
            })
            ok += 1

        except Exception as e:
            bad += 1
            log_rows.append({
                "record_path": record_path,
                "status": "bad",
                "error": repr(e),
            })
            if bad <= 10:
                print(f"[BAD] {record_path} -> {repr(e)}")

        if (i + 1) % 500 == 0 or (i + 1) == total:
            print(f"[INFO] processed {i + 1}/{total} | ok={ok} bad={bad}")

    if len(xs) == 0:
        raise RuntimeError("No valid Georgia samples were built.")

    X = np.stack(xs, axis=0).astype(np.float32)
    Y = np.stack(ys, axis=0).astype(np.float32)

    meta_df = pd.DataFrame(meta_rows)
    log_df = pd.DataFrame(log_rows)

    np.save(OUT_DIR / "X_test.npy", X)
    np.save(OUT_DIR / "y_test_mh.npy", Y)
    meta_df.to_csv(OUT_DIR / "meta.csv", index=False)
    log_df.to_csv(OUT_DIR / "build_log.csv", index=False)

    print("\n=== Georgia external_processed build finished ===")
    print(f"X_test.npy shape   = {X.shape}, dtype={X.dtype}")
    print(f"y_test_mh.npy shape= {Y.shape}, dtype={Y.dtype}")
    print(f"meta.csv rows      = {len(meta_df)}")
    print(f"build_log.csv rows = {len(log_df)}")
    print(f"ok={ok}, bad={bad}")

    print("\nLabel counts:")
    print(meta_df[CLASS_NAMES].sum())

    print(f"\n[OK] saved -> {OUT_DIR}")


if __name__ == "__main__":
    main()