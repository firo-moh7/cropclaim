#!/usr/bin/env python3
"""Prepare CropClaim public/private splits.

Shipd entrypoint:
    prepare(raw: Path, public: Path, private: Path) -> None

Accepts either:

1) RAW layout (preferred):
      metadata.csv
      crops/<sample_id>/crop_{0..3}.png

2) Already-prepared layout:
      public/train.csv, test.csv, sample_submission.csv, train_crops/, test_crops/
      private/answers.csv

Nested zip folders (one top-level directory) are supported.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20250917
N_TEST = 900
ID_PREFIX = "cc"


def _obfuscate_id(raw_id: str, salt: str = "cropclaim-v1") -> str:
    h = hashlib.sha1(f"{salt}:{raw_id}".encode()).hexdigest()[:12]
    return f"{ID_PREFIX}_{h}"


def _candidate_roots(raw: Path) -> list[Path]:
    """Shallow candidate roots only (avoid walking 20k PNGs)."""
    raw = Path(raw)
    out: list[Path] = []
    if raw.is_dir():
        out.append(raw)
        try:
            children = sorted([p for p in raw.iterdir() if p.is_dir()])
        except OSError:
            children = []
        # Prefer non-crop folder names first
        children = sorted(
            children,
            key=lambda p: (p.name.lower() in {"crops", "train_crops", "test_crops", "public", "private"}, p.name),
        )
        out.extend(children)
        # one more level for zip wrappers like CropClaim_Raw/
        for child in children[:20]:
            try:
                out.extend(sorted([p for p in child.iterdir() if p.is_dir()])[:30])
            except OSError:
                pass
    # de-dupe preserve order
    seen = set()
    uniq = []
    for p in out:
        rp = p.resolve() if p.exists() else p
        key = str(rp)
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    return uniq


def _list_top(raw: Path, n: int = 40) -> str:
    try:
        items = sorted(raw.iterdir(), key=lambda p: p.name)[:n]
        return ", ".join(p.name + ("/" if p.is_dir() else "") for p in items)
    except Exception as e:
        return f"<unlistable: {e}>"


def _find_prepared_root(raw: Path) -> Path | None:
    for root in _candidate_roots(raw):
        train = root / "public" / "train.csv"
        answers = root / "private" / "answers.csv"
        if train.is_file() and answers.is_file():
            return root
        # sometimes public/ is the raw root itself
        if (root / "train.csv").is_file() and (root.parent / "private" / "answers.csv").is_file():
            return root.parent
    return None


def _find_raw_root(raw: Path) -> Path | None:
    for root in _candidate_roots(raw):
        if (root / "metadata.csv").is_file():
            return root
    # last resort: bounded name search (files named metadata.csv only)
    try:
        for p in raw.rglob("metadata.csv"):
            parts = {x.lower() for x in p.relative_to(raw).parts}
            if "public" in parts or "private" in parts:
                continue
            return p.parent
    except Exception:
        pass
    return None


def _clear_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _copy_tree_contents(src: Path, dst: Path) -> None:
    _clear_dir(dst)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def _prepare_from_prepared(src: Path, public: Path, private: Path) -> None:
    pub_src = src / "public"
    priv_src = src / "private"
    required = [
        pub_src / "train.csv",
        pub_src / "test.csv",
        pub_src / "sample_submission.csv",
        priv_src / "answers.csv",
    ]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"prepared dataset missing files: {missing}")

    _copy_tree_contents(pub_src, public)
    _copy_tree_contents(priv_src, private)

    train = pd.read_csv(public / "train.csv")
    answers = pd.read_csv(private / "answers.csv")
    print(
        f"prepare done (prepared-copy): src={src} train={len(train)} test={len(answers)}"
    )


def _prepare_from_metadata(data_root: Path, raw: Path, public: Path, private: Path) -> None:
    meta_path = data_root / "metadata.csv"
    meta = pd.read_csv(meta_path)
    if len(meta) <= N_TEST + 500:
        raise ValueError(f"raw pool too small: {len(meta)} rows in {meta_path}")

    required_cols = {"sample_id", "claim", "claim_type", "support_index", "crop_0", "crop_1", "crop_2", "crop_3"}
    missing_cols = required_cols - set(meta.columns)
    if missing_cols:
        raise ValueError(f"metadata.csv missing columns: {sorted(missing_cols)}")

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
    _clear_dir(train_img)
    _clear_dir(test_img)

    def _resolve_crop(rel: str) -> Path:
        rel = str(rel).lstrip("./")
        candidates = [
            data_root / rel,
            raw / rel,
            data_root / "crops" / Path(rel).name,
        ]
        # also try if rel is just filename under any sample dir — skip expensive search
        for c in candidates:
            if c.is_file():
                return c
        raise FileNotFoundError(f"missing crop file for '{rel}' (tried {candidates[:2]})")

    def _copy_crops(row, dest_dir: Path, new_id: str) -> dict:
        out = {}
        for k in range(4):
            src = _resolve_crop(row[f"crop_{k}"])
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
        sample_rows.append({"id": new_id, "support_index": 0.01})

    train_df = pd.DataFrame(train_rows).sort_values("id").reset_index(drop=True)
    test_df = pd.DataFrame(test_rows).sort_values("id").reset_index(drop=True)
    answers = pd.DataFrame(answer_rows).sort_values("id").reset_index(drop=True)
    sample = pd.DataFrame(sample_rows).sort_values("id").reset_index(drop=True)

    if not set(train_df["id"]).isdisjoint(set(test_df["id"])):
        raise RuntimeError("train/test id overlap")
    if not answers["support_index"].between(-1, 3).all():
        raise RuntimeError("support_index out of range")

    train_df.to_csv(public / "train.csv", index=False)
    test_df.to_csv(public / "test.csv", index=False)
    sample.to_csv(public / "sample_submission.csv", index=False)
    answers.to_csv(private / "answers.csv", index=False)

    print(
        f"prepare done (from-metadata): data_root={data_root} "
        f"train={len(train_df)} test={len(test_df)} "
        f"none_rate_train={(train_df.support_index == -1).mean():.3f} "
        f"none_rate_test={(answers.support_index == -1).mean():.3f}"
    )


def prepare(raw: Path, public: Path, private: Path) -> None:
    raw = Path(raw)
    public = Path(public)
    private = Path(private)
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)

    if not raw.exists():
        raise FileNotFoundError(f"raw path does not exist: {raw}")

    # Mode 1: already prepared public/private package
    prepared = _find_prepared_root(raw)
    if prepared is not None:
        _prepare_from_prepared(prepared, public, private)
        return

    # Mode 2: RAW metadata.csv + crops/
    data_root = _find_raw_root(raw)
    if data_root is not None:
        _prepare_from_metadata(data_root, raw, public, private)
        return

    raise FileNotFoundError(
        "Could not find CropClaim data under raw. "
        f"raw={raw} top_entries=[{_list_top(raw)}]. "
        "Expected either metadata.csv+crops/ OR public/+private/."
    )


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    prepare(root / "raw", root / "public", root / "private")
