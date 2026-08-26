#!/usr/bin/env python3
"""Image-layout tests for the offline info-overlay media generator."""

from PIL import Image

import info_overlay_gen as overlay


def test_normalize_heights_pads_frames_to_shared_canvas_without_scaling():
    small = Image.new("RGB", (10, 5), (1, 2, 3))
    tall = Image.new("RGB", (8, 12), (4, 5, 6))

    normalized = overlay.normalize_heights([small, tall])

    assert [frame.size for frame in normalized] == [(10, 12), (10, 12)]
    assert normalized[0].getpixel((0, 0)) == (1, 2, 3)
    assert normalized[0].getpixel((9, 11)) == overlay.BG
    assert normalized[1].getpixel((0, 0)) == (4, 5, 6)


def test_paint_trims_blank_rows_but_keeps_minimum_terminal_height():
    frame = overlay.paint(["overview «cache»", "value", "", ""], "cache")
    expected_height = overlay.PAD * 2 + 3 * overlay.CELL_H
    assert frame.width == overlay.PAD * 2 + overlay.COLS * overlay.CELL_W
    assert frame.height == expected_height


def test_find_fak_honors_explicit_path_without_filesystem_probe():
    assert overlay.find_fak("X:/custom/fak.exe") == "X:/custom/fak.exe"

