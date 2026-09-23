#!/usr/bin/env python3
"""Prepare CropClaim public/private splits.

Shipd entrypoint:
    prepare(raw: Path, public: Path, private: Path) -> None
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


def prepare(raw: Path, public: Path, private: Path) -> None:
    raw = Path(raw)
    public = Path(public)
    private = Path(private)
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(raw / "metadata.csv")
    assert len(meta) > N_TEST + 500, "raw pool too small"

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
    train_img.mkdir(parents=True, exist_ok=True)
    test_img.mkdir(parents=True, exist_ok=True)

    def _copy_crops(row, dest_dir: Path, new_id: str) -> dict:
        out = {}
        for k in range(4):
            src = raw / str(row[f"crop_{k}"])
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

    # sanity: no id overlap, label range
    assert set(train_df["id"]).isdisjoint(set(test_df["id"]))
    assert answers["support_index"].between(-1, 3).all()

    train_df.to_csv(public / "train.csv", index=False)
    test_df.to_csv(public / "test.csv", index=False)
    sample.to_csv(public / "sample_submission.csv", index=False)
    answers.to_csv(private / "answers.csv", index=False)

    # drop raw-only meta from public train? claim_type is OK as a feature hint;
    # keep it — it's not a leak of the label.

    print(
        f"prepare done: train={len(train_df)} test={len(test_df)} "
        f"none_rate_train={(train_df.support_index == -1).mean():.3f} "
        f"none_rate_test={(answers.support_index == -1).mean():.3f}"
    )


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    prepare(root / "raw", root / "public", root / "private")
