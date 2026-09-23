# CropClaim

Synthetic multimodal dataset for **visual claim grounding**: each sample pairs a short English claim with four 96×96 panel crops. The label is which crop supports the claim (`0`–`3`), or `-1` if none do.

## License

**[CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/)** — public domain dedication. Free for commercial use and redistribution.

## Origin

All images and claims are **original synthetic artifacts** produced by `generate_raw.py` (geometric panels, meters, dual-shape cards, count grids, dominant-color fills). No third-party photographs. **No LLM-generated labels or claim text.**

## Reproduce

```bash
pip install numpy pandas pillow
python generate_raw.py   # writes raw/metadata.csv + raw/crops/
python prepare.py        # optional: public/private Shipd-style split
```

Fixed seeds: generation `91728364`, prepare `20250917`.

## Files

| Path | Description |
|------|-------------|
| `generate_raw.py` | Dataset generator (source of truth) |
| `prepare.py` | Deterministic train/test split helper |
| `dataset_description.md` | Schema and design notes |
| `LICENSE` | CC0 1.0 |

## Citation / attribution

If you use this dataset, link to this repository:

`https://github.com/firo-moh7/cropclaim`
