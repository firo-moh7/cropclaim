#!/usr/bin/env python3
"""Prepare CropClaim public/private splits.

Shipd entrypoint:
    prepare(raw: Path, public: Path, private: Path) -> None

`raw` is the uploaded dataset root. Expected layout (either at root or one folder deep):

    metadata.csv
    crops/<sample_id>/crop_{0..3}.png
    LICENSE.txt   # optional
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20250917  # uncommon on purpose
N_TEST = 900
ID_PREFIX = "cc"


def _obfuscate_id(raw_id: str, salt: str = "cropclaim-v1") -> str:
    h = hashlib.sha1(f"{salt}:{raw_id}".encode()).hexdigest()[:12]
    return f"{ID_PREFIX}_{h}"


def _find_data_root(raw: Path) -> Path:
    """Locate directory that contains metadata.csv."""
    raw = Path(raw)
    direct = raw / "metadata.csv"
    if direct.is_file():
        return raw

    # Common zip layouts: single top-level folder wrapping the dataset
    matches = sorted(raw.rglob("metadata.csv"))
    # Prefer shallowest path
    matches = sorted(matches, key=lambda p: len(p.relative_to(raw).parts))
    for m in matches:
        # ignore accidental copies under public/private if present
        parts = {x.lower() for x in m.relative_to(raw).parts}
        if "public" in parts or "private" in parts:
            continue
        return m.parent

    tried = [str(direct)] + [str(p) for p in matches[:5]]
    raise FileNotFoundError(
        "Could not find metadata.csv under raw dataset. "
        f"Looked for: {tried}. Upload the RAW package "
        "(metadata.csv + crops/), not the prepared public/private split."
    )


def prepare(raw: Path, public: Path, private: Path) -> None:
    raw = Path(raw)
    public = Path(public)
    private = Path(private)
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)

    data_root = _find_data_root(raw)
    meta_path = data_root / "metadata.csv"
    meta = pd.read_csv(meta_path)
    if len(meta) <= N_TEST + 500:
        raise ValueError(f"raw pool too small: {len(meta)} rows in {meta_path}")

    rng = np.random.default_rng(SEED)
    idx = np.arange(len(meta))
    rng.shuffle(idx)
    test_idx = set(idx[:N_TEST].tolist())
    train_idx = idx[N_TEST:]

    train_rows = []
    test_rows = []
    answer_rows = []
    sample_rows = []

    train_img = public / "train_crops"
    test_img = public / "test_crops"
    if train_img.exists():
        shutil.rmtree(train_img)
    if test_img.exists():
        shutil.rmtree(test_img)
    train_img.mkdir(parents=True, exist_ok=True)
    test_img.mkdir(parents=True, exist_ok=True)

    def _copy_crops(row, dest_dir: Path, new_id: str) -> dict:
        out = {}
        for k in range(4):
            rel = str(row[f"crop_{k}"])
            src = data_root / rel
            if not src.is_file():
                # allow absolute-ish accidental paths
                alt = raw / rel
                if alt.is_file():
                    src = alt
                else:
                    raise FileNotFoundError(f"missing crop file: {src}")
            dst_name = f"{new_id}_c{k}.png"
            dst = dest_dir / dst_name
            shutil.copy2(src, dst)
            out[f"crop_{k}"] = f"{dest_dir.name}/{dst_name}"
        return out

    for i in train_idx:
        row = meta.iloc[int(i)]
        new_id = _obfuscate_id(str(row["sample_id"]))
        paths = _copy_crops(row, train_img, new_id)
        train_rows.append(
            {
                "id": new_id,
                "claim": row["claim"],
                "claim_type": row["claim_type"],
                **paths,
                "support_index": int(row["support_index"]),
            }
        )

    for i in sorted(test_idx):
        row = meta.iloc[int(i)]
        new_id = _obfuscate_id(str(row["sample_id"]))
        paths = _copy_crops(row, test_img, new_id)
        test_rows.append(
            {
                "id": new_id,
                "claim": row["claim"],
                "claim_type": row["claim_type"],
                **paths,
            }
        )
        answer_rows.append({"id": new_id, "support_index": int(row["support_index"])})
        # dummy sample_submission with >=0.01 quirk (never a valid index)
        sample_rows.append({"id": new_id, "support_index": 0.01})

    train_df = pd.DataFrame(train_rows).sort_values("id").reset_index(drop=True)
    test_df = pd.DataFrame(test_rows).sort_values("id").reset_index(drop=True)
    answers = pd.DataFrame(answer_rows).sort_values("id").reset_index(drop=True)
    sample = pd.DataFrame(sample_rows).sort_values("id").reset_index(drop=True)

    assert set(train_df["id"]).isdisjoint(set(test_df["id"]))
    assert answers["support_index"].between(-1, 3).all()

    train_df.to_csv(public / "train.csv", index=False)
    test_df.to_csv(public / "test.csv", index=False)
    sample.to_csv(public / "sample_submission.csv", index=False)
    answers.to_csv(private / "answers.csv", index=False)

    print(
        f"prepare done: data_root={data_root} train={len(train_df)} test={len(test_df)} "
        f"none_rate_train={(train_df.support_index == -1).mean():.3f} "
        f"none_rate_test={(answers.support_index == -1).mean():.3f}"
    )


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    prepare(root / "raw", root / "public", root / "private")
