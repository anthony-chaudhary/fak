#!/usr/bin/env python3
"""Generate the repository social-preview / Open Graph card (1280x640 PNG).

This is the image GitHub shows when the repo link is shared on Twitter/X,
LinkedIn, Slack, Discord, etc. (Settings -> General -> Social preview), and the
`og:image` for the docs site. GitHub's recommended size is 1280x640 (2:1).

Run:  python tools/gen_social_preview.py
Out:  visuals/social-preview.png

Pure-Pillow (no SVG toolchain needed). Fonts: tries a few common families and
falls back to Pillow's bundled default so the script never hard-fails.
"""
from __future__ import annotations

import os
import sys

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover
    sys.exit("Pillow is required: pip install pillow")

W, H = 1280, 640
BG = (13, 17, 23)          # GitHub dark canvas
PANEL = (22, 27, 34)       # subtle inner panel
ACCENT = (88, 166, 255)    # GitHub blue
ACCENT2 = (63, 185, 80)    # GitHub green
FG = (230, 237, 243)       # near-white
MUTED = (139, 148, 158)    # gray

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "visuals", "social-preview.png")

FONT_DIRS = [
    r"C:\Windows\Fonts",
    "/usr/share/fonts",
    "/Library/Fonts",
    "/System/Library/Fonts",
]
BOLD_CANDIDATES = ["segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf", "Arial Bold.ttf"]
REG_CANDIDATES = ["segoeui.ttf", "arial.ttf", "DejaVuSans.ttf", "Arial.ttf"]
MONO_CANDIDATES = ["consola.ttf", "DejaVuSansMono.ttf", "cour.ttf", "Menlo.ttc"]


def _find(cands):
    for d in FONT_DIRS:
        for c in cands:
            p = os.path.join(d, c)
            if os.path.exists(p):
                return p
    return None


def font(cands, size):
    p = _find(cands)
    try:
        return ImageFont.truetype(p, size) if p else ImageFont.load_default()
    except Exception:
        return ImageFont.load_default()


def text_w(draw, s, f):
    return draw.textbbox((0, 0), s, font=f)[2]


def main():
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    # Accent bar down the left edge.
    d.rectangle([0, 0, 12, H], fill=ACCENT)

    f_kicker = font(MONO_CANDIDATES, 28)
    f_title = font(BOLD_CANDIDATES, 96)
    f_sub = font(REG_CANDIDATES, 36)
    f_pill = font(REG_CANDIDATES, 30)
    f_foot = font(REG_CANDIDATES, 28)

    pad = 80
    y = 70

    # Kicker
    d.text((pad, y), "the agent kernel", font=f_kicker, fill=ACCENT)
    y += 56

    # Title
    d.text((pad, y), "fak", font=f_title, fill=FG)
    y += 120

    # Subtitle (the thesis, in plain words)
    d.text((pad, y), "Treat the model like an untrusted program,",
           font=f_sub, fill=FG)
    y += 46
    d.text((pad, y), "and the tool call like a syscall.", font=f_sub, fill=FG)
    y += 84

    # Two capability pills
    pills = [
        ("default-deny permission gate", ACCENT),
        ("addressable bit-exact KV cache", ACCENT2),
    ]
    px = pad
    for label, color in pills:
        tw = text_w(d, label, f_pill)
        box_w = tw + 56
        d.rounded_rectangle([px, y, px + box_w, y + 56], radius=28,
                            outline=color, width=2, fill=PANEL)
        d.text((px + 28, y + 11), label, font=f_pill, fill=color)
        px += box_w + 24
    y += 110

    # Footer line
    d.text((pad, y),
           "prompt-injection containment + cache-efficient self-hosted LLM agent fleets",
           font=f_foot, fill=MUTED)
    y += 40
    d.text((pad, y), "Go  ·  Apache-2.0  ·  github.com/anthony-chaudhary/fak",
           font=f_foot, fill=MUTED)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    img.save(OUT, "PNG")
    print(f"wrote {OUT} ({W}x{H})")


if __name__ == "__main__":
    main()
