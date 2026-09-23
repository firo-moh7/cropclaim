# CropClaim

Synthetic multimodal dataset for **visual claim grounding**.

## License
**CC0 1.0 Universal** — https://creativecommons.org/publicdomain/zero/1.0/

## Shipd dataset upload (RAW)

Upload **CropClaim_Raw.zip** from Releases (not a prepared public/private split):

https://github.com/firo-moh7/cropclaim/releases/latest/download/CropClaim_Raw.zip

Root layout inside the zip:
```
metadata.csv
LICENSE.txt
crops/<sample_id>/crop_0.png … crop_3.png
```

`prepare.py` splits raw → `public/` + `private/`.

## Origin
Original synthetic data from `generate_raw.py`. No third-party media. No LLM labels.

Attribution: https://github.com/firo-moh7/cropclaim
