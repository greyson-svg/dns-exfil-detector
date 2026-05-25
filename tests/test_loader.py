"""Tests for dns_exfil.data.loader.

These tests cover `classify_file()` only; they use synthetic Path objects
and do not require the real dataset to be present. `load_dataset()` is
covered by manual EDA in the notebook (and would require fixtures or
the real dataset to test rigorously).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dns_exfil.data.loader import PAYLOAD_TYPES, classify_file


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    """A synthetic data root - no real files are created."""
    return tmp_path / "data" / "raw"


# ---- scenario / label mapping -------------------------------------------------


def test_light_attack(data_root: Path) -> None:
    p = data_root / "Attacks" / "stateless_features-light_audio.pcap.csv"
    meta = classify_file(p, data_root)
    assert meta["label"] == "exfil"
    assert meta["scenario"] == "light"
    assert meta["payload_type"] == "audio"
    assert meta["feature_view"] == "stateless"


def test_heavy_attack(data_root: Path) -> None:
    p = data_root / "Attacks (1)" / "stateful_features-heavy_video.pcap.csv"
    meta = classify_file(p, data_root)
    assert meta["label"] == "exfil"
    assert meta["scenario"] == "heavy"
    assert meta["payload_type"] == "video"
    assert meta["feature_view"] == "stateful"


def test_matched_benign_light(data_root: Path) -> None:
    p = data_root / "Benign" / "stateless_features-benign.pcap.csv"
    meta = classify_file(p, data_root)
    assert meta["label"] == "benign"
    assert meta["scenario"] == "benign_matched_light"
    assert meta["payload_type"] is None


def test_matched_benign_heavy(data_root: Path) -> None:
    p = data_root / "Benign (1)" / "stateful_features-benign.pcap.csv"
    meta = classify_file(p, data_root)
    assert meta["label"] == "benign"
    assert meta["scenario"] == "benign_matched_heavy"
    assert meta["payload_type"] is None


def test_standalone_benign(data_root: Path) -> None:
    p = data_root / "Benign (2)" / "stateless_features-benign.pcap.csv"
    meta = classify_file(p, data_root)
    assert meta["label"] == "benign"
    assert meta["scenario"] == "benign_standalone"


# ---- unknown / edge cases -----------------------------------------------------


def test_unknown_folder_does_not_raise(data_root: Path) -> None:
    p = data_root / "WhoKnows" / "stateless_features-something.csv"
    meta = classify_file(p, data_root)
    assert meta["label"] == "unknown"
    assert meta["scenario"] == "unknown"
    # feature_view should still be detected from the filename
    assert meta["feature_view"] == "stateless"


def test_unknown_feature_view(data_root: Path) -> None:
    p = data_root / "Attacks" / "weird_filename.csv"
    meta = classify_file(p, data_root)
    assert meta["feature_view"] == "unknown"


def test_attack_without_payload_type(data_root: Path) -> None:
    p = data_root / "Attacks" / "stateless_features-light.pcap.csv"
    meta = classify_file(p, data_root)
    assert meta["label"] == "exfil"
    assert meta["payload_type"] is None


# ---- payload type detection ---------------------------------------------------


@pytest.mark.parametrize("ptype", sorted(PAYLOAD_TYPES))
def test_each_payload_type_is_detected(data_root: Path, ptype: str) -> None:
    p = data_root / "Attacks" / f"stateless_features-light_{ptype}.pcap.csv"
    meta = classify_file(p, data_root)
    assert meta["payload_type"] == ptype


# ---- source_file formatting ---------------------------------------------------


def test_source_file_is_relative_to_data_root(data_root: Path) -> None:
    p = data_root / "Attacks" / "stateless_features-light_audio.pcap.csv"
    meta = classify_file(p, data_root)
    assert meta["source_file"].startswith("Attacks/")


def test_source_file_uses_forward_slashes(data_root: Path) -> None:
    """Even on Windows, source_file should use forward slashes for portability."""
    p = data_root / "Attacks" / "stateless_features-light_audio.pcap.csv"
    meta = classify_file(p, data_root)
    assert "\\" not in meta["source_file"]
