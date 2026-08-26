#!/usr/bin/env python3
"""Font-discovery and text-measurement tests for the social preview renderer."""

import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

import gen_social_preview as preview


def test_find_uses_candidate_order_within_font_directories():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "second.ttf").write_bytes(b"font-placeholder")
        saved = preview.FONT_DIRS
        preview.FONT_DIRS = [td]
        try:
            assert preview._find(["first.ttf", "second.ttf"]) == str(root / "second.ttf")
            assert preview._find(["missing.ttf"]) is None
        finally:
            preview.FONT_DIRS = saved


def test_text_width_reports_real_layout_extent():
    image = Image.new("RGB", (200, 50))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    assert preview.text_w(draw, "fak", font) > 0
    assert preview.text_w(draw, "fak fak", font) > preview.text_w(draw, "fak", font)

