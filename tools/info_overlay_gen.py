#!/usr/bin/env python3
"""info_overlay_gen.py — render the `fak info` overlay to checked-in README media.

The `fak info` overlay is the tabbed status pane `fak guard --split` opens beside the
agent: cache savings, live agents, account seats, and the safety floor, each on its own
tab. It reads "value at a glance" for an end user, so it belongs in the README — but the
live overlay renders only from a running gateway, which is not reproducible. This tool
closes that gap by driving the OFFLINE renderer (`fak info --from-fixture`, the twin of
`fak console guard --journal`) over a checked-in, payload-free snapshot and turning each
tab's REAL rendered text into an image.

Pipeline (deterministic — the renderer is the source of truth, this only paints it):
  1. for each tab, run `fak info --from-fixture <json> --tab <t> --width W --height H`
     and capture the exact text the live overlay would draw for that snapshot;
  2. paint each frame onto a light terminal canvas (monospace, matching the reference
     look) -> visuals/info-overlay-<tab>.png, plus the cache tab as the hero
     visuals/info-overlay-screenshot.png;
  3. assemble the per-tab frames into a tab-cycle GIF + MP4 (+ poster) that walks through
     every tab, so a reader sees each tab's value in one short loop.

Usage:
  python tools/info_overlay_gen.py                 # build all PNGs + gif + mp4 + poster
  python tools/info_overlay_gen.py --fak PATH      # use a specific fak binary
  python tools/info_overlay_gen.py --no-video      # PNGs only
  python tools/info_overlay_gen.py --hold 1.6      # seconds each tab holds in the video

Fixture:  visuals/info-overlay-live.json   (regenerate: FAK_UPDATE_INFO_FIXTURE=1 go test
          ./cmd/fak/ -run TestGenerateOverlayFixture)
Outputs:  visuals/info-overlay-<tab>.png · info-overlay-screenshot.png ·
          info-overlay-video.gif · info-overlay-video.mp4 · info-overlay-video-poster.png
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover
    sys.exit("Pillow is required: pip install pillow")

ROOT = Path(__file__).resolve().parents[1]
VISUALS = ROOT / "visuals"
FIXTURE = VISUALS / "info-overlay-live.json"

TABS = ["overview", "agents", "accounts", "cache", "safety"]
HERO_TAB = "cache"  # the tab shown as the standalone README screenshot

# Render geometry (cells) — wide enough for the cache split line, tall enough for the
# overview to show every panel without the "+N more" fold.
COLS = 118
ROWS = 26

# Light terminal theme, matching the reference overlay look: a warm cream canvas, near-black
# text, a thin top rule under the tab bar. Kept brand-neutral.
BG = (251, 249, 240)       # warm cream canvas
PANEL = (255, 255, 255)    # inner panel (very subtly lighter)
FG = (32, 34, 38)          # near-black text
ACCENT = (176, 40, 52)     # the "bypass permissions on" red in the reference; used for the active tab
RULE = (208, 202, 186)     # hairline rules
MUTED = (120, 122, 128)

PAD = 28          # canvas padding
CELL_W = 11       # monospace cell width (px) at the chosen font size
CELL_H = 22       # line height (px)
FONT_SIZE = 18

FONT_DIRS = [
    r"C:\Windows\Fonts",
    "/usr/share/fonts",
    "/usr/share/fonts/truetype/dejavu",
    "/Library/Fonts",
    "/System/Library/Fonts",
]
# Prefer a glyph-rich monospace: the overlay uses ⊘ ● ○ « » and the block-elements ramp
# ▁▂▃▄▅▆▇█ / ░, which Consolas covers only partially (⊘ is tofu). DejaVu Sans Mono covers all
# of them, so it leads; Consolas etc. are the fallback so the tool never hard-fails on a box
# without DejaVu.
MONO_CANDIDATES = ["DejaVuSansMono.ttf", "consola.ttf", "CascadiaMono.ttf", "cour.ttf", "Menlo.ttc"]


def _matplotlib_font_dir():
    """matplotlib bundles DejaVu Sans Mono; if it is installed, add its font dir so the
    glyph-rich font is found even where the OS lacks it. Best-effort — no hard dependency."""
    try:
        import matplotlib
        return os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
    except Exception:
        return None


def _find_font(cands):
    dirs = list(FONT_DIRS)
    mpl = _matplotlib_font_dir()
    if mpl:
        dirs.insert(0, mpl)
    for c in cands:
        for d in dirs:
            p = os.path.join(d, c)
            if os.path.exists(p):
                return p
    return None


def mono(size):
    p = _find_font(MONO_CANDIDATES)
    try:
        return ImageFont.truetype(p, size) if p else ImageFont.load_default()
    except Exception:
        return ImageFont.load_default()


def find_fak(explicit: str | None) -> str:
    """Locate the fak binary. Prefer an explicit --fak, then a repo-built ./fak(.exe),
    then PATH. Never build here — the caller builds so a stale binary is obvious."""
    if explicit:
        return explicit
    for name in ("fak.exe", "fak"):
        cand = ROOT / name
        if cand.exists():
            return str(cand)
    from shutil import which
    got = which("fak")
    if got:
        return got
    sys.exit("fak binary not found: build it (`go build -o fak ./cmd/fak`) or pass --fak PATH")


def render_tab_text(fak: str, tab: str) -> list[str]:
    """Run the offline renderer for one tab and return its lines (the REAL overlay text)."""
    out = subprocess.run(
        [fak, "info", "--from-fixture", str(FIXTURE), "--tab", tab,
         "--width", str(COLS), "--height", str(ROWS)],
        capture_output=True, text=True, encoding="utf-8",
    )
    if out.returncode != 0:
        sys.exit(f"fak info --tab {tab} failed ({out.returncode}):\n{out.stderr}")
    return out.stdout.rstrip("\n").split("\n")


def paint(lines: list[str], active_tab: str) -> Image.Image:
    """Paint overlay text lines onto the terminal canvas. The first line is the tab bar; the
    active tab's «guillemet» chip is drawn in the accent color so the selection reads at a
    glance (the renderer already brackets it, we only colorize). The canvas height is cropped
    to the actual content (trailing blank rows dropped) so a short tab is not mostly whitespace,
    while a fixed width keeps every tab's frame the same size for a clean video."""
    content = [ln for ln in lines[:ROWS]]
    while len(content) > 1 and content[-1].strip() == "":
        content.pop()
    nrows = max(len(content), 3)
    w = PAD * 2 + COLS * CELL_W
    h = PAD * 2 + nrows * CELL_H
    img = Image.new("RGB", (w, h), BG)
    d = ImageDraw.Draw(img)
    f = mono(FONT_SIZE)

    # A subtle inner panel with a hairline border, like a terminal pane.
    d.rectangle([PAD - 10, PAD - 10, w - PAD + 10, h - PAD + 10], fill=PANEL, outline=RULE, width=1)

    y = PAD
    for i, line in enumerate(content):
        x = PAD
        if i == 0:
            # Tab bar: draw the active «...» chip in accent, the rest in FG. We colorize by
            # splitting on the guillemets the renderer emits around the active tab.
            _paint_tabbar(d, line, x, y, f)
            # hairline under the tab bar
            d.line([PAD, y + CELL_H - 2, w - PAD, y + CELL_H - 2], fill=RULE, width=1)
        else:
            d.text((x, y), line, font=f, fill=FG)
        y += CELL_H
    return img


def _paint_tabbar(d, line: str, x: int, y: int, f):
    """Draw the tab bar, tinting the active «chip» accent. Falls back to plain FG if the
    guillemets are absent (e.g. a future renderer change)."""
    lo = line.find("«")
    hi = line.find("»")
    if lo < 0 or hi < 0 or hi < lo:
        d.text((x, y), line, font=f, fill=FG)
        return
    before, chip, after = line[:lo], line[lo:hi + 1], line[hi + 1:]
    d.text((x, y), before, font=f, fill=FG)
    x += len(before) * CELL_W
    d.text((x, y), chip, font=f, fill=ACCENT)
    x += len(chip) * CELL_W
    d.text((x, y), after, font=f, fill=FG)


def normalize_heights(frames: list[Image.Image]) -> list[Image.Image]:
    """Pad every frame to the tallest frame's height (top-aligned on the canvas bg) so a video
    made from tabs of different content heights has uniform dimensions and a fixed tab bar."""
    w = max(f.width for f in frames)
    h = max(f.height for f in frames)
    out = []
    for f in frames:
        if f.size == (w, h):
            out.append(f)
            continue
        canvas = Image.new("RGB", (w, h), BG)
        canvas.paste(f, (0, 0))
        out.append(canvas)
    return out


def write_gif(frames: list[Image.Image], out: Path, hold_ms: int):
    frames = normalize_heights(frames)
    frames[0].save(
        out, save_all=True, append_images=frames[1:],
        duration=hold_ms, loop=0, optimize=True,
    )
    print(f"wrote {out}  ({len(frames)} tabs @ {hold_ms}ms)")


def write_mp4(frames: list[Image.Image], out: Path, poster: Path, fps: int, hold_s: float):
    try:
        import imageio_ffmpeg
        ff = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:  # pragma: no cover
        print(f"skip mp4: imageio-ffmpeg unavailable ({e})", file=sys.stderr)
        return
    frames = normalize_heights(frames)
    w, h = frames[0].size
    # even dimensions for yuv420p
    w -= w % 2
    h -= h % 2
    cmd = [
        ff, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
        "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "20", "-preset", "slow", "-movflags", "+faststart", str(out),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    hold_frames = max(1, round(hold_s * fps))
    for fr in frames:
        rgb = fr.convert("RGB").crop((0, 0, w, h))
        buf = rgb.tobytes()
        for _ in range(hold_frames):
            proc.stdin.write(buf)
    proc.stdin.close()
    proc.wait()
    print(f"wrote {out}  ({len(frames)} tabs x {hold_s}s @ {fps}fps)")
    # poster = the hero (cache) frame
    frames[TABS.index(HERO_TAB)].save(poster)
    print(f"wrote {poster}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fak", default=None, help="path to the fak binary (default: ./fak or PATH)")
    ap.add_argument("--no-video", action="store_true", help="PNGs only, skip gif/mp4")
    ap.add_argument("--hold", type=float, default=1.6, help="seconds each tab holds in the video")
    ap.add_argument("--fps", type=int, default=24)
    args = ap.parse_args()

    if not FIXTURE.exists():
        sys.exit(f"fixture not found: {FIXTURE} (regenerate with the go generator test)")
    fak = find_fak(args.fak)

    frames: list[Image.Image] = []
    for tab in TABS:
        lines = render_tab_text(fak, tab)
        img = paint(lines, tab)
        out = VISUALS / f"info-overlay-{tab}.png"
        img.save(out)
        print(f"wrote {out}")
        frames.append(img)
        if tab == HERO_TAB:
            hero = VISUALS / "info-overlay-screenshot.png"
            img.save(hero)
            print(f"wrote {hero}  (hero)")

    if not args.no_video:
        write_gif(frames, VISUALS / "info-overlay-video.gif", int(args.hold * 1000))
        write_mp4(frames, VISUALS / "info-overlay-video.mp4",
                  VISUALS / "info-overlay-video-poster.png", args.fps, args.hold)


if __name__ == "__main__":
    main()
