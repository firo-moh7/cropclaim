# CropClaim

Synthetic multimodal dataset for **visual claim grounding**: each sample pairs a short English claim with four 96×96 panel crops. The label is which crop supports the claim (`0`–`3`), or `-1` if none do.

## License

**[CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/)** — public domain dedication. Free for commercial use and redistribution.

## Dataset package (canonical download)

The redistributable dataset archive is:

**[CropClaim_Dataset.zip](https://github.com/firo-moh7/cropclaim/releases/latest/download/CropClaim_Dataset.zip)**

Also mirrored in this repository as [`CropClaim_Dataset.zip`](./CropClaim_Dataset.zip).

### Contents

```
CropClaim_Dataset/
├── public/
│   ├── train.csv
│   ├── test.csv
│   ├── sample_submission.csv
│   ├── train_crops/*.png
│   └── test_crops/*.png
└── private/
    └── answers.csv
```

- Train: 4300 labeled samples  
- Test: 900 samples  
- Images: 96×96 RGB panels (4 crops per sample)

## Origin

All images and claims are **original synthetic artifacts** produced by `generate_raw.py` (geometric panels, meters, dual-shape cards, count grids, dominant-color fills). No third-party photographs. **No LLM-generated labels or claim text.**

## Reproduce from scratch

```bash
pip install numpy pandas pillow
python generate_raw.py   # writes raw/metadata.csv + raw/crops/
python prepare.py        # writes public/ + private/
```

Fixed seeds: generation `91728364`, prepare `20250917`.

## Citation / attribution

`https://github.com/firo-moh7/cropclaim`
