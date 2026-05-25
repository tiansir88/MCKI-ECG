from pathlib import Path
import re
import json
import pandas as pd

RAW_ROOT = Path("/root/gpufree-data/MCKI_Project/data/external_raw")
OUT_DIR = Path("/root/gpufree-data/MCKI_Project/data/processed/georgia")
OUT_DIR.mkdir(parents=True, exist_ok=True)

hea_files = sorted(RAW_ROOT.rglob("*.hea"))

rows = []
dx_rows = []

def try_parse_header_line(line: str):
    """
    WFDB header 第一行通常类似：
    JS00001 12 500 5000
    或
    JS00001 12 1000 10000
    """
    parts = line.strip().split()
    record_name = parts[0] if len(parts) > 0 else None

    n_sig = None
    fs = None
    sig_len = None

    if len(parts) > 1:
        try:
            n_sig = int(parts[1])
        except:
            pass

    if len(parts) > 2:
        # 有时 fs 会写成 500 或 500/...
        fs_token = parts[2].split("/")[0]
        try:
            fs = float(fs_token)
        except:
            pass

    if len(parts) > 3:
        try:
            sig_len = int(parts[3])
        except:
            pass

    return record_name, n_sig, fs, sig_len

for hea in hea_files:
    stem = hea.with_suffix("")
    rel_stem = stem.relative_to(RAW_ROOT)

    try:
        text = hea.read_text(encoding="utf-8", errors="ignore")
        lines = [x.rstrip("\n") for x in text.splitlines() if x.strip()]

        record_name, n_sig, fs, sig_len = None, None, None, None
        if len(lines) > 0:
            record_name, n_sig, fs, sig_len = try_parse_header_line(lines[0])

        comments = [x for x in lines if x.lstrip().startswith("#")]

        dx_raw = ""
        age = ""
        sex = ""

        for c in comments:
            c2 = c.lstrip("#").strip()

            m_dx = re.match(r"^Dx:\s*(.*)$", c2, flags=re.I)
            if m_dx:
                dx_raw = m_dx.group(1).strip()

            m_age = re.match(r"^Age:\s*(.*)$", c2, flags=re.I)
            if m_age:
                age = m_age.group(1).strip()

            m_sex = re.match(r"^Sex:\s*(.*)$", c2, flags=re.I)
            if m_sex:
                sex = m_sex.group(1).strip()

        rows.append({
            "record_path": str(rel_stem),
            "hea_path": str(hea),
            "mat_exists": stem.with_suffix(".mat").exists(),
            "record_name": record_name,
            "fs": fs,
            "sig_len": sig_len,
            "n_sig": n_sig,
            "age": age,
            "sex": sex,
            "dx_raw": dx_raw,
            "num_comments": len(comments),
        })

        if dx_raw:
            parts = [x.strip() for x in re.split(r"[;,]", dx_raw) if x.strip()]
            for p in parts:
                dx_rows.append({
                    "record_path": str(rel_stem),
                    "dx_code_or_text": p,
                })

    except Exception as e:
        rows.append({
            "record_path": str(rel_stem),
            "hea_path": str(hea),
            "mat_exists": stem.with_suffix(".mat").exists(),
            "record_name": "",
            "fs": None,
            "sig_len": None,
            "n_sig": None,
            "age": "",
            "sex": "",
            "dx_raw": "",
            "num_comments": None,
            "error": repr(e),
        })

audit_df = pd.DataFrame(rows)
audit_df.to_csv(OUT_DIR / "raw_audit.csv", index=False)

dx_df = pd.DataFrame(dx_rows)
dx_df.to_csv(OUT_DIR / "exploded_labels_raw.csv", index=False)

if len(dx_df) > 0:
    uniq = (
        dx_df["dx_code_or_text"]
        .astype(str)
        .value_counts()
        .rename_axis("dx_code_or_text")
        .reset_index(name="count")
    )
else:
    uniq = pd.DataFrame(columns=["dx_code_or_text", "count"])

uniq.to_csv(OUT_DIR / "unique_dx_raw.csv", index=False)

print("[OK] saved:", OUT_DIR / "raw_audit.csv")
print("[OK] saved:", OUT_DIR / "exploded_labels_raw.csv")
print("[OK] saved:", OUT_DIR / "unique_dx_raw.csv")

print("\n=== audit preview ===")
print(audit_df.head().to_string(index=False))

print("\n=== basic stats ===")
print("num_headers =", len(audit_df))
print("num_with_mat =", int(audit_df['mat_exists'].fillna(False).sum()))
print("num_with_dx =", int((audit_df['dx_raw'].astype(str).str.len() > 0).sum()))

if "fs" in audit_df.columns:
    print("\nfs value counts:")
    print(audit_df["fs"].value_counts(dropna=False).head(10).to_string())

if "sig_len" in audit_df.columns:
    print("\nsig_len describe:")
    print(audit_df["sig_len"].describe())

if len(uniq) > 0:
    print("\n=== top unique dx ===")
    print(uniq.head(30).to_string(index=False))
