from pathlib import Path
import pandas as pd
import re

BASE = Path("~/MCKI_Project/data/external_raw/sph").expanduser()
OUT = Path("~/MCKI_Project/data/processed/sph").expanduser()
OUT.mkdir(parents=True, exist_ok=True)

def norm_col(s):
    return re.sub(r'[^a-z0-9]+', '', str(s).lower())

def pick_col(cols, candidates):
    norm_map = {norm_col(c): c for c in cols}
    for cand in candidates:
        c = norm_col(cand)
        if c in norm_map:
            return norm_map[c]
    for cand in candidates:
        c = norm_col(cand)
        for k, v in norm_map.items():
            if c in k or k in c:
                return v
    return None

def split_items(x):
    if pd.isna(x):
        return []
    s = str(x).strip()
    if not s:
        return []
    return [p.strip() for p in re.split(r'[;|]', s) if p.strip()]

def parse_code_item(item: str):
    """
    例如:
      '30+145' -> primary_code='30', modifier_codes='145'
      '22'     -> primary_code='22', modifier_codes=''
    """
    s = str(item).strip()
    if '+' in s:
        parts = [p.strip() for p in s.split('+') if p.strip()]
        primary = parts[0] if parts else ""
        modifiers = '+'.join(parts[1:]) if len(parts) > 1 else ""
        return primary, modifiers
    return s, ""

def normalize_statement(x):
    s = str(x).strip().lower()
    s = re.sub(r'\s+', ' ', s)
    return s

def suggest_cls(statement: str) -> str:
    s = statement.lower().strip()

    if s == "normal ecg":
        return "NORM"

    if re.search(r'myocardial infarction|infarction|infarct|myocardial injury|old infarct', s):
        return "MI"

    if re.search(r'st deviation|st[- ]?t|t-wave|t wave|ischemi|repolar', s):
        return "STTC"

    if re.search(r'bundle-branch block|bundle branch block|fascicular|hemiblock|atrioventricular block|av block|conduction|wpw|pre-?excitation|prolonged pr interval|intraventricular', s):
        return "CD"

    if re.search(r'hypertrophy|enlargement|dilatation|dilation', s):
        return "HYP"

    return ""

meta = pd.read_csv(BASE / "metadata.csv")
code = pd.read_csv(BASE / "code.csv")

id_col = pick_col(meta.columns, ["ECG_ID", "ecg_id", "record_id", "id"])
patient_col = pick_col(meta.columns, ["Patient_ID", "patient_id", "subject_id", "pid"])
label_col = pick_col(meta.columns, ["AHA_Code", "aha_code", "diagnosis", "label", "labels"])

if id_col is None or label_col is None:
    raise ValueError(f"无法识别 metadata 关键列: {meta.columns.tolist()}")

code_col = pick_col(code.columns, ["Code", "code", "aha_code"])
stmt_col = pick_col(code.columns, ["Description", "description", "statement"])

if code_col is None or stmt_col is None:
    raise ValueError(f"无法识别 code.csv 关键列: {code.columns.tolist()}")

code_map = (
    code[[code_col, stmt_col]]
    .dropna()
    .assign(**{
        code_col: lambda df: df[code_col].astype(str).str.strip(),
        stmt_col: lambda df: df[stmt_col].astype(str).str.strip()
    })
)
code_map = dict(zip(code_map[code_col], code_map[stmt_col]))

record_paths = {}
for p in (BASE / "records").rglob("*.h5"):
    record_paths[p.stem] = str(p.resolve())

rows = []
for _, r in meta.iterrows():
    ecg_id = str(r[id_col]).strip()
    patient_id = str(r[patient_col]).strip() if patient_col is not None else ""
    raw_label = r[label_col]

    items = split_items(raw_label)
    if not items and pd.notna(raw_label):
        items = [str(raw_label).strip()]

    for it in items:
        primary_code, modifier_codes = parse_code_item(it)
        primary_desc = code_map.get(primary_code, primary_code)
        primary_desc = normalize_statement(primary_desc)

        if not primary_desc:
            continue

        rows.append({
            "ecg_id": ecg_id,
            "patient_id": patient_id,
            "path": record_paths.get(ecg_id, ""),
            "raw_item": str(it).strip(),
            "primary_code": primary_code,
            "modifier_codes": modifier_codes,
            "statement": primary_desc
        })

exp = pd.DataFrame(rows)
if exp.empty:
    raise ValueError("没有解析出任何 statement。")

exp.to_csv(OUT / "exploded_labels.csv", index=False)

freq = (
    exp.groupby("statement")
       .size()
       .reset_index(name="count")
       .sort_values("count", ascending=False)
       .reset_index(drop=True)
)
freq["suggested_ptbxl5"] = freq["statement"].apply(suggest_cls)
freq["final_ptbxl5"] = ""
freq.to_csv(OUT / "statement_freq.csv", index=False)

mapping_template = freq[["statement", "suggested_ptbxl5"]].copy()
mapping_template = mapping_template.rename(columns={"suggested_ptbxl5": "ptbxl5"})
mapping_template.to_csv(OUT / "manual_mapping_template.csv", index=False)

print("\nDone.")
print("saved:", OUT / "exploded_labels.csv")
print("saved:", OUT / "statement_freq.csv")
print("saved:", OUT / "manual_mapping_template.csv")
print("\nDetected columns:")
print("id_col     =", id_col)
print("patient_col=", patient_col)
print("label_col  =", label_col)
print("code_col   =", code_col)
print("stmt_col   =", stmt_col)
print("\nTop statements:")
print(freq.head(30).to_string(index=False))
