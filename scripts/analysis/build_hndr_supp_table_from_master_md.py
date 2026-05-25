import argparse
import re
from pathlib import Path
import pandas as pd

PROTOCOLS = [
    'Full_Finetune',
    'Linear_Probing',
    'Few_Shot_1%',
    'Few_Shot_10%',
    'Few_Shot_25%',
]

HEADER_RE = re.compile(r"^###\s+.*?\[(.+?)\]\s*$")
ROW_RE = re.compile(r"^\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*$")
SEP_PREFIXES = ('| ---', '|---', '| ----------------')


def clean_cell(x: str) -> str:
    x = x.strip()
    x = x.replace('**', '').replace('`', '')
    return x


def parse_tables(md_text: str):
    lines = md_text.splitlines()
    current_protocol = None
    collecting = False
    rows = []

    for line in lines:
        m = HEADER_RE.match(line.strip())
        if m:
            proto = m.group(1).strip()
            current_protocol = proto if proto in PROTOCOLS else None
            collecting = current_protocol is not None
            continue

        if not collecting:
            continue

        s = line.strip()
        if not s.startswith('|'):
            continue
        if s.startswith(SEP_PREFIXES):
            continue
        if 'Method' in s and 'HNDR_Pair' in s:
            continue

        rm = ROW_RE.match(s)
        if not rm:
            continue
        method, macro_auc, auprc, f1, mi_f1, hndr_pair, hndr_inst = [clean_cell(g) for g in rm.groups()]
        rows.append({
            'Protocol': current_protocol,
            'Method': method,
            'Macro_AUC': macro_auc,
            'AUPRC': auprc,
            'F1': f1,
            'MI_F1': mi_f1,
            'HNDR_Pair': hndr_pair,
            'HNDR_Inst': hndr_inst,
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description='Build supplementary HNDR tables from merged markdown master tables.')
    ap.add_argument('--input-md', type=str, required=True, help='Path to merged result markdown, e.g. 表格结果.md')
    ap.add_argument('--output-dir', type=str, default='supp_hndr_tables')
    ap.add_argument('--highlight-method', type=str, default='MCKI(stage8)')
    args = ap.parse_args()

    md_path = Path(args.input_md)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    text = md_path.read_text(encoding='utf-8')
    df = parse_tables(text)
    if df.empty:
        raise RuntimeError('No in-domain master tables with HNDR columns were parsed. Check markdown format.')

    df.to_csv(out_dir / 'hndr_all_protocols_long.csv', index=False, encoding='utf-8-sig')

    for protocol, sub in df.groupby('Protocol', sort=False):
        keep = sub[['Method', 'HNDR_Pair', 'HNDR_Inst']].copy()
        keep.to_csv(out_dir / f'hndr_only_{protocol.replace("%", "pct")}.csv', index=False, encoding='utf-8-sig')

    lines = []
    lines.append('# Supplementary HNDR Tables\n')
    lines.append('These tables are extracted from the merged master result markdown and keep only HNDR-Pair / HNDR-Inst for supplementary presentation.\n')

    for protocol in PROTOCOLS:
        sub = df[df['Protocol'] == protocol][['Method', 'HNDR_Pair', 'HNDR_Inst']].copy()
        if sub.empty:
            continue
        lines.append(f'## {protocol}\n')
        lines.append('| Method | HNDR_Pair | HNDR_Inst |')
        lines.append('|---|---:|---:|')
        for _, row in sub.iterrows():
            method = row['Method']
            if method == args.highlight_method:
                method = f'**{method}**'
            lines.append(f"| {method} | {row['HNDR_Pair']} | {row['HNDR_Inst']} |")
        lines.append('')

    (out_dir / 'supplementary_hndr_tables.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'[OK] Saved -> {out_dir}')


if __name__ == '__main__':
    main()
