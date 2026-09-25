# CropClaim

Synthetic multimodal dataset for **visual claim grounding**.

## License

**[CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/)**

## Shipd dataset upload (RAW — required)

Upload this archive (not a prepared public/private split):

**[CropClaim_Raw.zip](https://github.com/firo-moh7/cropclaim/releases/latest/download/CropClaim_Raw.zip)**

Layout:

```
CropClaim_Raw/
├── metadata.csv
├── LICENSE.txt
├── dataset_description.md
└── crops/<sample_id>/crop_{0,1,2,3}.png
```

`prepare.py` reads `metadata.csv` + `crops/` and writes `public/` + `private/`.

## Reproduce

```bash
pip install numpy pandas pillow
python generate_raw.py
python prepare.py
```

Seeds: generate `91728364`, prepare `20250917`.

## Attribution

`https://github.com/firo-moh7/cropclaim`
