#!/usr/bin/env python3
"""Motion, sizing, and timeline tests for the hero video generator."""

from PIL import Image

import hero_video_gen as hero


def test_easing_clamps_and_multiplier_format_reaches_target():
    assert hero.smoothstep(-1.0) == 0.0
    assert hero.smoothstep(1.5) == 1.0
    assert hero.ease_out_cubic(0.0) == 0.0
    assert hero.ease_out_cubic(1.0) == 1.0
    assert hero._fmt_mult("9.7×", 1.0) == "9.7×"
    assert hero._fmt_mult("20,480x", 1.0) == "20,480×"


def test_contain_preserves_aspect_ratio_inside_box():
    wide = Image.new("RGB", (400, 200), "white")
    tall = Image.new("RGB", (100, 400), "white")
    assert hero.contain(wide, 100, 100).size == (100, 50)
    assert hero.contain(tall, 100, 100).size == (25, 100)


def test_autotrim_crops_uniform_margin_but_keeps_content():
    image = Image.new("RGB", (100, 80), "white")
    for x in range(30, 70):
        for y in range(20, 60):
            image.putpixel((x, y), (0, 0, 0))
    cropped = hero.autotrim(image, (255, 255, 255), tol=0, pad_frac=0)
    assert cropped.size == (40, 40)
    assert cropped.getpixel((0, 0)) == (0, 0, 0)
    assert cropped.getpixel((39, 39)) == (0, 0, 0)


def test_timeline_overlaps_each_scene_by_incoming_transition():
    class Stub:
        def __init__(self, dur, tin):
            self.dur = dur
            self.tin = tin

    starts, ends = hero.build_timeline([Stub(5.0, 0.0), Stub(4.0, 1.0), Stub(3.0, 0.5)])
    assert starts == [0.0, 4.0, 7.5]
    assert ends == [5.0, 8.0, 10.5]
