"""Sanity-check the CIC-Bell-DNS-EXF 2021 dataset after download.

Run after placing the downloaded files under data/raw/:

    uv run python scripts/verify_dataset.py

What this does
--------------
- Walks data/raw/ recursively
- Reports every CSV and pcap/pcapng file found, with sizes
- For each CSV: reads it, prints row count, column names, and if a Label
  (or Class / Target) column exists, prints the class distribution
- Does NOT assume specific filenames - UNB has reshuffled filenames in
  past releases, so this script just inspects whatever is there

This is a smoke test, not validation. The EDA notebook is where we
actually look at the data in depth.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
LABEL_CANDIDATES = {"label", "class", "target", "y"}


def human_size(num_bytes: int) -> str:
    """Format a byte count as KB / MB / GB for terminal output."""
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"


def inspect_csv(path: Path) -> None:
    """Print row count, columns, and class distribution for one CSV."""
    print(f"\n--- {path.relative_to(DATA_RAW)} ({human_size(path.stat().st_size)}) ---")
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as e:
        print(f"  could not read as CSV: {type(e).__name__}: {e}")
        return

    print(f"  rows: {len(df):,}")
    cols = list(df.columns)
    preview = cols[:8]
    overflow = f" ... ({len(cols) - 8} more)" if len(cols) > 8 else ""
    print(f"  columns ({len(cols)}): {preview}{overflow}")

    label_col = next((c for c in df.columns if c.strip().lower() in LABEL_CANDIDATES), None)
    if label_col is None:
        print("  (no obvious label column found - check columns above)")
        return

    print(f"  label column: {label_col!r}")
    counts = df[label_col].value_counts(dropna=False)
    for value, count in counts.items():
        pct = 100 * count / len(df)
        print(f"    {value!r:>30}  {count:>10,}  ({pct:.1f}%)")


def main() -> None:
    if not DATA_RAW.exists():
        print(f"ERROR: {DATA_RAW} does not exist.")
        print("Create the directory and place CIC-Bell-DNS-EXF 2021 files there.")
        raise SystemExit(1)

    csv_paths = sorted(p for p in DATA_RAW.rglob("*.csv") if p.is_file())
    pcap_paths = sorted(
        p for p in DATA_RAW.rglob("*") if p.is_file() and p.suffix.lower() in {".pcap", ".pcapng"}
    )

    print(f"Looking under: {DATA_RAW}")
    print(f"Found {len(csv_paths)} CSV file(s) and {len(pcap_paths)} pcap file(s).")

    if not csv_paths and not pcap_paths:
        print()
        print("No data files found. Steps to get the dataset:")
        print("  1. Register at https://www.unb.ca/cic/datasets/dns-exf-2021.html")
        print("  2. Download the files from the email link UNB sends")
        print(f"  3. Extract them under {DATA_RAW}")
        print("  4. Re-run this script")
        raise SystemExit(1)

    for csv_path in csv_paths:
        inspect_csv(csv_path)

    if pcap_paths:
        print("\n--- pcap files (not parsed in Week 1) ---")
        for p in pcap_paths:
            print(f"  {p.relative_to(DATA_RAW)}  ({human_size(p.stat().st_size)})")

    print("\nVerification complete.")


if __name__ == "__main__":
    main()
