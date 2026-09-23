#!/usr/bin/env python3
"""Generate CropClaim raw multimodal corpus (synthetic, no LLM labels).

Creates:
  raw/metadata.csv
  raw/crops/<sample_id>/crop_{0..3}.png

Each sample: one English claim + four 96x96 panel crops.
Gold label: support_index in {-1,0,1,2,3} (-1 = none of the crops support the claim).

Distractors share lexical cues (color/shape words, watermark text) but fail visually
so OCR / bag-of-words shortcuts stay well below the visual ceiling.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

SEED = 91728364
N_TRAIN_POOL = 5200  # prepare will split; generate pooled labeled set
IMG = 96
OUT = Path(__file__).resolve().parent / "raw"

COLORS = {
    "red": (220, 60, 60),
    "blue": (50, 110, 220),
    "green": (40, 170, 90),
    "yellow": (230, 200, 40),
    "purple": (150, 70, 200),
    "orange": (230, 130, 40),
    "cyan": (40, 190, 200),
    "magenta": (210, 50, 160),
}
SHAPES = ["circle", "square", "triangle", "diamond", "hexagon", "star"]
PATTERNS = ["solid", "ringed", "striped"]


def _font(size: int = 11):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _bg(rng: np.random.Generator) -> tuple[int, int, int]:
    v = int(rng.integers(235, 250))
    return (v, v, v - int(rng.integers(0, 8)))


def _draw_shape(draw: ImageDraw.ImageDraw, shape: str, color, box, pattern: str):
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    r = min(x1 - x0, y1 - y0) / 2
    fill = color if pattern != "ringed" else None
    outline = color
    width = 3 if pattern == "ringed" else 2

    if shape == "circle":
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill, outline=outline, width=width)
    elif shape == "square":
        draw.rectangle([cx - r * 0.85, cy - r * 0.85, cx + r * 0.85, cy + r * 0.85], fill=fill, outline=outline, width=width)
    elif shape == "triangle":
        pts = [(cx, cy - r), (cx - r, cy + r * 0.85), (cx + r, cy + r * 0.85)]
        draw.polygon(pts, fill=fill, outline=outline)
    elif shape == "diamond":
        pts = [(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)]
        draw.polygon(pts, fill=fill, outline=outline)
    elif shape == "hexagon":
        pts = [
            (cx + r * math.cos(math.radians(a)), cy + r * math.sin(math.radians(a)))
            for a in range(0, 360, 60)
        ]
        draw.polygon(pts, fill=fill, outline=outline)
    elif shape == "star":
        pts = []
        for i in range(10):
            ang = math.radians(-90 + i * 36)
            rad = r if i % 2 == 0 else r * 0.45
            pts.append((cx + rad * math.cos(ang), cy + rad * math.sin(ang)))
        draw.polygon(pts, fill=fill, outline=outline)

    if pattern == "striped" and fill is not None:
        for i in range(int(x0), int(x1), 6):
            draw.line([(i, y0), (i, y1)], fill=(255, 255, 255), width=2)


def render_shape_panel(rng, color_name, shape, pattern="solid", watermark: str | None = None) -> Image.Image:
    img = Image.new("RGB", (IMG, IMG), _bg(rng))
    draw = ImageDraw.Draw(img)
    margin = 14
    _draw_shape(draw, shape, COLORS[color_name], (margin, margin, IMG - margin, IMG - margin), pattern)
    if watermark:
        draw.text((4, IMG - 14), watermark[:18], fill=(90, 90, 90), font=_font(10))
    return img


def render_dual_panel(rng, c1, s1, c2, s2, watermark: str | None = None) -> Image.Image:
    img = Image.new("RGB", (IMG, IMG), _bg(rng))
    draw = ImageDraw.Draw(img)
    _draw_shape(draw, s1, COLORS[c1], (8, 20, 44, 76), "solid")
    _draw_shape(draw, s2, COLORS[c2], (52, 20, 88, 76), "solid")
    if watermark:
        draw.text((4, IMG - 14), watermark[:18], fill=(90, 90, 90), font=_font(10))
    return img


def render_meter(rng, value: float, watermark: str | None = None) -> Image.Image:
    img = Image.new("RGB", (IMG, IMG), _bg(rng))
    draw = ImageDraw.Draw(img)
    draw.rectangle([12, 40, 84, 58], outline=(40, 40, 40), width=2)
    fill_w = 12 + int(72 * max(0.0, min(1.0, value / 100.0)))
    bar_color = (40, 160, 80) if value >= 50 else (200, 80, 60)
    draw.rectangle([12, 40, fill_w, 58], fill=bar_color)
    draw.text((12, 64), f"{value:.0f}", fill=(30, 30, 30), font=_font(14))
    if watermark:
        draw.text((4, 4), watermark[:18], fill=(90, 90, 90), font=_font(10))
    return img


def render_count_panel(rng, shapes_spec: list[tuple[str, str]], watermark: str | None = None) -> Image.Image:
    img = Image.new("RGB", (IMG, IMG), _bg(rng))
    draw = ImageDraw.Draw(img)
    slots = [(8, 8), (50, 8), (8, 50), (50, 50)]
    for (c, s), (x, y) in zip(shapes_spec, slots):
        _draw_shape(draw, s, COLORS[c], (x, y, x + 38, y + 38), "solid")
    if watermark:
        draw.text((4, IMG - 14), watermark[:18], fill=(90, 90, 90), font=_font(10))
    return img


def render_dominant_panel(rng, dominant: str, minority: str, watermark: str | None = None) -> Image.Image:
    img = Image.new("RGB", (IMG, IMG), _bg(rng))
    draw = ImageDraw.Draw(img)
    draw.rectangle([8, 8, 88, 70], fill=COLORS[dominant])
    draw.ellipse([60, 55, 88, 83], fill=COLORS[minority])
    if watermark:
        draw.text((4, IMG - 14), watermark[:18], fill=(40, 40, 40), font=_font(10))
    return img


def _other(rng, options, forbid):
    choices = [x for x in options if x != forbid]
    return choices[int(rng.integers(0, len(choices)))]


def _other_color(rng, forbid):
    return _other(rng, list(COLORS.keys()), forbid)


def _other_shape(rng, forbid):
    return _other(rng, SHAPES, forbid)


def build_sample(rng: np.random.Generator, sample_id: str):
    claim_type = ["color_shape", "meter", "dual", "count", "dominant"][int(rng.integers(0, 5))]
    none_case = bool(rng.random() < 0.22)
    support = -1 if none_case else int(rng.integers(0, 4))
    crops_meta = []
    images = [None, None, None, None]

    if claim_type == "color_shape":
        color = list(COLORS.keys())[int(rng.integers(0, len(COLORS)))]
        shape = SHAPES[int(rng.integers(0, len(SHAPES)))]
        pattern = PATTERNS[int(rng.integers(0, len(PATTERNS)))]
        claim = f"A {pattern} {color} {shape} is visible."
        for i in range(4):
            wm = f"{color} {shape}" if rng.random() < 0.55 else None  # OCR trap
            if i == support:
                img = render_shape_panel(rng, color, shape, pattern, watermark=None)
                crops_meta.append({"kind": "shape", "color": color, "shape": shape, "pattern": pattern, "supports": True})
            else:
                # lexical overlap, visual miss
                mode = int(rng.integers(0, 3))
                if mode == 0:
                    img = render_shape_panel(rng, color, _other_shape(rng, shape), pattern, watermark=wm)
                elif mode == 1:
                    img = render_shape_panel(rng, _other_color(rng, color), shape, pattern, watermark=wm)
                else:
                    img = render_shape_panel(
                        rng,
                        _other_color(rng, color),
                        _other_shape(rng, shape),
                        _other(rng, PATTERNS, pattern),
                        watermark=wm,
                    )
                crops_meta.append({"kind": "shape", "supports": False})
            images[i] = img

    elif claim_type == "meter":
        threshold = float([30, 40, 50, 60, 70][int(rng.integers(0, 5))])
        claim = f"The meter reading is above {threshold:.0f}."
        true_val = float(rng.uniform(threshold + 5, 98))
        for i in range(4):
            wm = f"above {threshold:.0f}" if rng.random() < 0.5 else None
            if i == support:
                val = true_val
                img = render_meter(rng, val, watermark=None)
                crops_meta.append({"kind": "meter", "value": val, "supports": True})
            else:
                val = float(rng.uniform(0, max(5.0, threshold - 3)))
                img = render_meter(rng, val, watermark=wm)
                crops_meta.append({"kind": "meter", "value": val, "supports": False})
            images[i] = img

    elif claim_type == "dual":
        pair = [SHAPES[i] for i in rng.choice(len(SHAPES), size=2, replace=False)]
        s1, s2 = pair[0], pair[1]
        c1 = list(COLORS.keys())[int(rng.integers(0, len(COLORS)))]
        c2 = _other_color(rng, c1)
        claim = f"Both a {s1} and a {s2} appear."
        for i in range(4):
            wm = f"{s1} {s2}" if rng.random() < 0.55 else None
            if i == support:
                img = render_dual_panel(rng, c1, s1, c2, s2, watermark=None)
                crops_meta.append({"kind": "dual", "shapes": [s1, s2], "supports": True})
            else:
                # one matching shape only, or wrong pair
                if rng.random() < 0.5:
                    img = render_dual_panel(rng, c1, s1, c2, _other_shape(rng, s2), watermark=wm)
                else:
                    img = render_shape_panel(rng, c1, s1, "solid", watermark=wm)
                crops_meta.append({"kind": "dual", "supports": False})
            images[i] = img

    elif claim_type == "count":
        n = int(rng.integers(2, 5))
        claim = f"Exactly {n} shapes are drawn."
        for i in range(4):
            wm = f"exactly {n}" if rng.random() < 0.5 else None
            if i == support:
                specs = [
                    (
                        list(COLORS.keys())[int(rng.integers(0, len(COLORS)))],
                        SHAPES[int(rng.integers(0, len(SHAPES)))],
                    )
                    for _ in range(n)
                ]
                img = render_count_panel(rng, specs, watermark=None)
                crops_meta.append({"kind": "count", "n": n, "supports": True})
            else:
                wrong_n = _other(rng, [1, 2, 3, 4], n)
                specs = [
                    (
                        list(COLORS.keys())[int(rng.integers(0, len(COLORS)))],
                        SHAPES[int(rng.integers(0, len(SHAPES)))],
                    )
                    for _ in range(wrong_n)
                ]
                img = render_count_panel(rng, specs, watermark=wm)
                crops_meta.append({"kind": "count", "n": wrong_n, "supports": False})
            images[i] = img

    else:  # dominant
        dom = list(COLORS.keys())[int(rng.integers(0, len(COLORS)))]
        minority = _other_color(rng, dom)
        claim = f"The dominant fill color is {dom}."
        for i in range(4):
            wm = f"dominant {dom}" if rng.random() < 0.55 else None
            if i == support:
                img = render_dominant_panel(rng, dom, minority, watermark=None)
                crops_meta.append({"kind": "dominant", "color": dom, "supports": True})
            else:
                # minority is actually dominant, or unrelated
                fake = _other_color(rng, dom)
                img = render_dominant_panel(rng, fake, dom if rng.random() < 0.6 else _other_color(rng, fake), watermark=wm)
                crops_meta.append({"kind": "dominant", "color": fake, "supports": False})
            images[i] = img

    # For none_case, support stays -1 and all crops are non-supporting (already).
    assert support == -1 or crops_meta[support]["supports"] is True
    return {
        "sample_id": sample_id,
        "claim": claim,
        "claim_type": claim_type,
        "support_index": support,
        "crops_meta": crops_meta,
        "images": images,
    }


def main():
    rng = np.random.default_rng(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    crops_root = OUT / "crops"
    if crops_root.exists():
        # rewrite cleanly
        import shutil

        shutil.rmtree(crops_root)
    crops_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for i in range(N_TRAIN_POOL):
        sid = f"raw_{i:06d}"
        sample = build_sample(rng, sid)
        d = crops_root / sid
        d.mkdir(parents=True, exist_ok=True)
        paths = []
        for k, im in enumerate(sample["images"]):
            p = d / f"crop_{k}.png"
            im.save(p, format="PNG", optimize=True)
            paths.append(str(p.relative_to(OUT)))
        rows.append(
            {
                "sample_id": sid,
                "claim": sample["claim"],
                "claim_type": sample["claim_type"],
                "support_index": sample["support_index"],
                "crop_0": paths[0],
                "crop_1": paths[1],
                "crop_2": paths[2],
                "crop_3": paths[3],
                "meta_json": json.dumps(sample["crops_meta"]),
            }
        )
        if (i + 1) % 500 == 0:
            print(f"generated {i + 1}/{N_TRAIN_POOL}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "metadata.csv", index=False)
    (OUT / "LICENSE.txt").write_text(
        "Synthetic CropClaim corpus generated by generate_raw.py.\n"
        "All images and claims are original synthetic artifacts (no third-party media).\n"
        "License: CC0-1.0 (public domain dedication).\n",
        encoding="utf-8",
    )
    print("wrote", OUT / "metadata.csv", "rows", len(df))
    print(df["support_index"].value_counts().sort_index().to_string())
    print(df["claim_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
