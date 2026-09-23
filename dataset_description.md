# Dataset Description

## Overview

**CropClaim** is a synthetic multimodal corpus for visual claim grounding. Each sample pairs one short English claim with four 96×96 RGB panel crops. Exactly one crop may visually support the claim, or none may (label `-1`).

All images and claims are generated programmatically by `generate_raw.py` (geometric panels, meters, dual-shape cards, count grids, dominant-color fills). No third-party photographs and **no LLM-generated text or labels**.

License: **CC0-1.0**.

## File Structure

```
raw/
├── LICENSE.txt
├── metadata.csv
└── crops/
    └── raw_XXXXXX/
        ├── crop_0.png
        ├── crop_1.png
        ├── crop_2.png
        └── crop_3.png
```

After `prepare.py`:

```
public/
├── train.csv
├── test.csv
├── sample_submission.csv
├── train_crops/*.png
└── test_crops/*.png

private/
└── answers.csv
```

## Features

### `metadata.csv` / `train.csv`

| Column | Type | Description |
|--------|------|-------------|
| sample_id / id | string | Unique sample id (obfuscated after prepare) |
| claim | string | English claim to ground |
| claim_type | string | One of `color_shape`, `meter`, `dual`, `count`, `dominant` |
| crop_0..crop_3 | string | Relative paths to the four panel PNGs |
| support_index | int | Gold index in `{-1,0,1,2,3}` (`-1` = unsupported). Present in raw + train only. |
| meta_json | string | Raw-only debug attributes (stripped from public test) |

### `answers.csv`

| Column | Type | Description |
|--------|------|-------------|
| id | string | Test id |
| support_index | int | Gold supporting crop index or `-1` |

## Notes

- ~22% of samples are **unsupported** (`support_index = -1`) to create an irreducible soft ceiling.
- Distractor crops often carry **watermark text** that repeats claim keywords (OCR / bag-of-words trap) while the pixels fail the claim.
- Claim families:
  - `color_shape` — patterned colored geometric shape present
  - `meter` — horizontal meter reading above a threshold
  - `dual` — two specific shapes both appear
  - `count` — exactly N shapes drawn
  - `dominant` — dominant fill color matches a named color
- Agents must fuse **vision + language**; text-only or OCR-only pipelines are intentionally weak.
