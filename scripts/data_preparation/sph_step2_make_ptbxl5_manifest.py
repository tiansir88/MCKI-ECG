from pathlib import Path
import pandas as pd

IN_DIR = Path("~/MCKI_Project/data/processed/sph").expanduser()
OUT_FILE = IN_DIR / "manifest_ptbxl5.csv"

PTBXL5 = ["NORM", "MI", "STTC", "CD", "HYP"]
ABNORMALS = {"MI", "STTC", "CD", "HYP"}

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).replace("\ufeff", "").strip().lower() for c in df.columns]
    return df

exp = pd.read_csv(IN_DIR / "exploded_labels.csv")
mp = pd.read_csv(IN_DIR / "manual_mapping.csv")

exp = clean_columns(exp)
mp = clean_columns(mp)

print("exploded_labels columns:", list(exp.columns))
print("manual_mapping columns:", list(mp.columns))

if "statement" not in exp.columns:
    raise ValueError(f"exploded_labels.csv 找不到 statement 列，现有列: {list(exp.columns)}")

if "statement" not in mp.columns:
    raise ValueError(f"manual_mapping.csv 找不到 statement 列，现有列: {list(mp.columns)}")

if "ptbxl5" not in mp.columns:
    raise ValueError(f"manual_mapping.csv 找不到 ptbxl5 列，现有列: {list(mp.columns)}")

exp["statement"] = exp["statement"].astype(str).str.strip().str.lower()
mp["statement"] = mp["statement"].astype(str).str.strip().str.lower()
mp["ptbxl5"] = mp["ptbxl5"].fillna("").astype(str).str.strip().str.upper()

valid = mp[mp["ptbxl5"].isin(PTBXL5)].copy()

df = exp.merge(
    valid[["statement", "ptbxl5"]],
    on="statement",
    how="left",
)

def normalize_labels(xs):
    xs = sorted(set(x for x in xs if x in PTBXL5))
    if any(x in ABNORMALS for x in xs):
        xs = [x for x in xs if x != "NORM"]
    return xs

group_cols = ["ecg_id", "patient_id", "path"]
missing_group_cols = [c for c in group_cols if c not in df.columns]
if missing_group_cols:
    raise ValueError(f"exploded_labels.csv 缺少分组列: {missing_group_cols}; 现有列: {list(df.columns)}")

agg = (
    df.groupby(group_cols)["ptbxl5"]
      .apply(lambda s: normalize_labels(s.dropna().tolist()))
      .reset_index()
      .rename(columns={"ptbxl5": "labels"})
)

for c in PTBXL5:
    agg[c] = agg["labels"].apply(lambda xs: int(c in xs))

agg["num_labels"] = agg[PTBXL5].sum(axis=1)
agg = agg[agg["num_labels"] > 0].copy()
agg = agg[agg["path"].astype(str).str.len() > 0].copy()

agg["labels_str"] = agg["labels"].apply(lambda xs: "|".join(xs))

cols = ["ecg_id", "patient_id", "path", "labels", "labels_str"] + PTBXL5 + ["num_labels"]
agg = agg[cols].copy()

agg.to_csv(OUT_FILE, index=False)

print("\\nsaved:", OUT_FILE)
print("num_records =", len(agg))
print("\\nclass counts:")
print(agg[PTBXL5].sum())
print("\\nmulti-label distribution:")
print(agg["num_labels"].value_counts().sort_index())

bad = agg[(agg["NORM"] == 1) & ((agg["MI"] + agg["STTC"] + agg["CD"] + agg["HYP"]) > 0)]
print("\\nNORM-with-abnormal count =", len(bad))
