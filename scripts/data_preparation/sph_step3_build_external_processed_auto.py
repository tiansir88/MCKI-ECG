from pathlib import Path
import numpy as np
import pandas as pd
import h5py
from scipy.signal import resample_poly
from numpy.lib.format import open_memmap

MANIFEST = Path("/home/tianshuang-25/MCKI_Project/data/processed/sph/manifest_ptbxl5.csv")
REF_X = Path("/home/tianshuang-25/MCKI_Project/data/processed_v3/X_test.npy")
OUT_DIR = Path("/home/tianshuang-25/MCKI_Project/data/external_processed/sph_5class")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PTBXL5 = ["NORM", "MI", "STTC", "CD", "HYP"]
SOURCE_FS = 500

def infer_layout_and_len(ref_shape):
    if len(ref_shape) != 3:
        raise RuntimeError(f"unexpected ref shape: {ref_shape}")

    # ref: (N, T, 12)
    if ref_shape[2] == 12:
        return True, ref_shape[1]
    # ref: (N, 12, T)
    if ref_shape[1] == 12:
        return False, ref_shape[2]

    raise RuntimeError(f"cannot infer lead dimension from ref shape: {ref_shape}")

def infer_target_fs(target_len):
    if target_len == 1000:
        return 100
    if target_len == 5000:
        return 500
    raise RuntimeError(
        f"target_len={target_len} 无法自动推断采样率；"
        "目前脚本只自动支持 1000(=100Hz*10s) 或 5000(=500Hz*10s)"
    )

def load_ecg_h5(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as f:
        if "ecg" in f:
            x = f["ecg"][:]
        else:
            keys = list(f.keys())
            if not keys:
                raise RuntimeError(f"empty h5: {path}")
            x = f[keys[0]][:]
    x = np.asarray(x, dtype=np.float32)

    if x.ndim != 2:
        raise RuntimeError(f"unexpected ndim={x.ndim} for {path}")

    # 统一成 (12, L)
    if x.shape[0] == 12:
        return x
    if x.shape[1] == 12:
        return x.T

    raise RuntimeError(f"cannot infer lead dimension for {path}, shape={x.shape}")

def resample_ecg(x: np.ndarray, src_fs: int, tgt_fs: int) -> np.ndarray:
    if src_fs == tgt_fs:
        return x
    return resample_poly(x, up=tgt_fs, down=src_fs, axis=1).astype(np.float32)

def fix_length_center(x: np.ndarray, target_len: int) -> np.ndarray:
    cur_len = x.shape[1]
    if cur_len == target_len:
        return x
    if cur_len > target_len:
        start = (cur_len - target_len) // 2
        return x[:, start:start + target_len]
    out = np.zeros((x.shape[0], target_len), dtype=np.float32)
    start = (target_len - cur_len) // 2
    out[:, start:start + cur_len] = x
    return out

def main():
    ref_x = np.load(REF_X, mmap_mode="r")
    ref_shape = ref_x.shape
    output_time_first, target_len = infer_layout_and_len(ref_shape)
    target_fs = infer_target_fs(target_len)

    print("Reference X:", REF_X)
    print("Reference shape:", ref_shape)
    print("output_time_first:", output_time_first)
    print("target_len:", target_len)
    print("target_fs:", target_fs)

    df = pd.read_csv(MANIFEST).copy()
    n = len(df)

    x_shape = (n, target_len, 12) if output_time_first else (n, 12, target_len)

    X_path = OUT_DIR / "X_test.npy"
    Y_path = OUT_DIR / "y_test_mh.npy"
    M_path = OUT_DIR / "meta.csv"

    X_mm = open_memmap(X_path, mode="w+", dtype="float32", shape=x_shape)
    Y = df[PTBXL5].to_numpy(dtype=np.float32)

    meta_rows = []

    for i, row in df.iterrows():
        path = Path(row["path"])
        x = load_ecg_h5(path)                    # (12, L)
        x = resample_ecg(x, SOURCE_FS, target_fs)
        x = fix_length_center(x, target_len)

        if output_time_first:
            X_mm[i] = x.T
        else:
            X_mm[i] = x

        meta_rows.append({
            "ecg_id": row["ecg_id"],
            "patient_id": row["patient_id"],
            "src_path": str(path),
            "labels_str": row.get("labels_str", ""),
            "orig_num_labels": int(row["num_labels"]),
        })

        if (i + 1) % 1000 == 0:
            print(f"[{i+1}/{n}] done")

    np.save(Y_path, Y)
    pd.DataFrame(meta_rows).to_csv(M_path, index=False)

    print("\\nSaved:")
    print(X_path)
    print(Y_path)
    print(M_path)

    print("\\nSummary:")
    print("X shape =", x_shape)
    print("Y shape =", Y.shape)
    print("class counts:")
    print(pd.DataFrame(Y, columns=PTBXL5).sum())

if __name__ == "__main__":
    main()
