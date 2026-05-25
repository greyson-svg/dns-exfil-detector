"""Load CIC-Bell-DNS-EXF 2021 CSVs and attach class / scenario metadata.

The dataset's labels live in the file paths, not in columns:

- `Attacks/*` and `Attacks (1)/*` are exfiltration captures (light and heavy)
- `Benign/*`, `Benign (1)/*`, `Benign (2)/*` are benign captures (matched-light,
  matched-heavy, and standalone respectively)

This module walks the dataset directory, reads every CSV, and attaches
metadata columns derived from the path. Use `load_dataset()` as the main
entry point. Use `classify_file()` if you only need to inspect path
semantics without reading the file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, TypedDict

import pandas as pd

FeatureView = Literal["stateless", "stateful"]

PAYLOAD_TYPES: frozenset[str] = frozenset(
    {"audio", "compressed", "exe", "image", "text", "video"}
)

_SCENARIO_MAP: dict[str, tuple[str, str]] = {
    "attacks": ("exfil", "light"),
    "attacks (1)": ("exfil", "heavy"),
    "benign": ("benign", "benign_matched_light"),
    "benign (1)": ("benign", "benign_matched_heavy"),
    "benign (2)": ("benign", "benign_standalone"),
}


class FileMetadata(TypedDict):
    """Metadata derived from a dataset CSV's file path."""

    label: str
    scenario: str
    payload_type: str | None
    feature_view: str
    source_file: str


def find_project_root(start: Path | None = None) -> Path:
    """Walk upward from `start` (or the current working directory) until a
    directory containing `pyproject.toml` is found.

    Raises:
        RuntimeError: if no parent contains pyproject.toml.
    """
    p = (start or Path.cwd()).resolve()
    while p != p.parent:
        if (p / "pyproject.toml").exists():
            return p
        p = p.parent
    raise RuntimeError("project root (pyproject.toml) not found")


def default_data_root() -> Path:
    """Return the default dataset location: data/raw/ under the project root."""
    return find_project_root() / "data" / "raw"


def classify_file(path: Path, data_root: Path) -> FileMetadata:
    """Derive label, scenario, payload_type, feature_view, and source_file
    from a CSV path under `data_root`.

    Unknown folder names or filenames map to `"unknown"` rather than raising,
    so the loader can keep walking past surprise files.

    Args:
        path: absolute path to a CSV file inside `data_root`.
        data_root: root directory of the raw dataset (typically data/raw/).

    Returns:
        A FileMetadata dict with five fields.
    """
    rel = path.relative_to(data_root)
    parts = [p.lower() for p in rel.parts]
    name = rel.name.lower()

    if "stateless" in name:
        feature_view = "stateless"
    elif "stateful" in name:
        feature_view = "stateful"
    else:
        feature_view = "unknown"

    label, scenario = _SCENARIO_MAP.get(parts[0], ("unknown", "unknown"))

    payload_type: str | None = None
    for ptype in PAYLOAD_TYPES:
        if f"_{ptype}." in name:
            payload_type = ptype
            break

    return FileMetadata(
        label=label,
        scenario=scenario,
        payload_type=payload_type,
        feature_view=feature_view,
        source_file=str(rel).replace("\\", "/"),
    )


def load_dataset(
    feature_view: FeatureView,
    data_root: Path | None = None,
) -> pd.DataFrame:
    """Load every CSV of the given feature view and concatenate them, attaching
    metadata columns to each row.

    Args:
        feature_view: `"stateless"` (per-query features, one row per DNS query)
            or `"stateful"` (aggregated flow-level features).
        data_root: dataset root. Defaults to data/raw/ under the project root.

    Returns:
        A pandas DataFrame with all data columns from the CSVs, plus five
        metadata columns: `label`, `scenario`, `payload_type`, `feature_view`,
        `source_file`.

    Raises:
        RuntimeError: if no matching CSVs are found.
    """
    if data_root is None:
        data_root = default_data_root()

    dfs: list[pd.DataFrame] = []
    for path in sorted(data_root.rglob("*.csv")):
        meta = classify_file(path, data_root)
        if meta["feature_view"] != feature_view:
            continue
        df = pd.read_csv(path, low_memory=False)
        for k, v in meta.items():
            df[k] = v
        dfs.append(df)
    if not dfs:
        raise RuntimeError(
            f"no CSVs found for feature_view={feature_view!r} under {data_root}"
        )
    return pd.concat(dfs, ignore_index=True)
