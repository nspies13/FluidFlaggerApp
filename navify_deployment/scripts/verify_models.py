#!/usr/bin/env python3
"""Verify baked BMP artifacts during local checks and Docker builds."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model_loader import (
    EXPECTED_BMP_MODEL_COUNT,
    EXPECTED_BMP_MODEL_KEYS,
    filename_for_key,
    load_required_models_from_dir,
)

MODELS_DIR = ROOT / "models"
CHECKSUM_FILE = ROOT / "checksums" / "models.sha256"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_manifest() -> dict[str, str]:
    manifest: dict[str, str] = {}
    for line in CHECKSUM_FILE.read_text().splitlines():
        if not line.strip():
            continue
        checksum, rel_path = line.split(maxsplit=1)
        manifest[rel_path.strip()] = checksum
    return manifest


def main() -> None:
    model_files = sorted(MODELS_DIR.glob("*.joblib"))
    if len(model_files) != EXPECTED_BMP_MODEL_COUNT:
        raise SystemExit(
            f"Expected {EXPECTED_BMP_MODEL_COUNT} BMP joblib files, found {len(model_files)}"
        )

    expected_rel_paths = {f"models/{filename_for_key(key)}" for key in EXPECTED_BMP_MODEL_KEYS}
    actual_rel_paths = {f"models/{path.name}" for path in model_files}
    if actual_rel_paths != expected_rel_paths:
        missing = sorted(expected_rel_paths - actual_rel_paths)
        extra = sorted(actual_rel_paths - expected_rel_paths)
        raise SystemExit(f"Model file mismatch. Missing={missing}; extra={extra}")

    manifest = _read_manifest()
    if set(manifest) != expected_rel_paths:
        missing = sorted(expected_rel_paths - set(manifest))
        extra = sorted(set(manifest) - expected_rel_paths)
        raise SystemExit(f"Checksum manifest mismatch. Missing={missing}; extra={extra}")

    for rel_path, expected_hash in sorted(manifest.items()):
        actual_hash = _sha256(ROOT / rel_path)
        if actual_hash != expected_hash:
            raise SystemExit(
                f"Checksum mismatch for {rel_path}: expected {expected_hash}, got {actual_hash}"
            )

    load_required_models_from_dir(MODELS_DIR)
    print(f"Verified {EXPECTED_BMP_MODEL_COUNT} BMP models and checksums.")


if __name__ == "__main__":
    main()
