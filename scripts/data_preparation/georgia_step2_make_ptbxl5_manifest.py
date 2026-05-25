from pathlib import Path
import pandas as pd

IN_DIR = Path("/root/gpufree-data/MCKI_Project/data/processed/georgia")
raw_dx = pd.read_csv(IN_DIR / "exploded_labels_raw.csv", dtype=str)
mapping = pd.read_csv(IN_DIR / "manual_mapping.csv", dtype=str)

raw_dx["dx_code_or_text"] = raw_dx["dx_code_or_text"].astype(str).str.strip()
mapping["dx_code_or_text"] = mapping["dx_code_or_text"].astype(str).str.strip()
mapping["ptbxl5"] = mapping["ptbxl5"].astype(str).str.strip()

df = raw_dx.merge(mapping, on="dx_code_or_text", how="left")
df = df[df["ptbxl5"].notna() & (df["ptbxl5"] != "")].copy()

grouped = df.groupby("record_path")["ptbxl5"].apply(lambda x: sorted(set(x.tolist()))).reset_index()

rows = []
for _, row in grouped.iterrows():
    labs = row["ptbxl5"]
    abnormal = [x for x in labs if x != "NORM"]
    if len(abnormal) > 0:
        labs = abnormal
    else:
        labs = ["NORM"] if "NORM" in labs else []

    rows.append({
        "record_path": row["record_path"],
        "labels": ";".join(labs),
        "NORM": int("NORM" in labs),
        "MI": int("MI" in labs),
        "STTC": int("STTC" in labs),
        "CD": int("CD" in labs),
        "HYP": int("HYP" in labs),
    })

manifest = pd.DataFrame(rows)
manifest = manifest[(manifest[["NORM","MI","STTC","CD","HYP"]].sum(axis=1) > 0)].copy()

manifest.to_csv(IN_DIR / "manifest_ptbxl5.csv", index=False)
print("[OK] saved:", IN_DIR / "manifest_ptbxl5.csv")
print(manifest.head().to_string(index=False))
print("\nCounts:")
print(manifest[["NORM","MI","STTC","CD","HYP"]].sum())
print("num_records =", len(manifest))
