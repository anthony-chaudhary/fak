# fak brand assets

The fak mark is an **aperture** — an open ring at the center of a four-point spark.
It reads as a lens/shutter (the kernel *sees* every tool call) and as a compass/star
(one command in front of the agent you already use). The open center is the point:
fak wraps your agent without replacing it.

## Files

| File | Use |
|------|-----|
| `fak-mark.svg` | Primary mark, gradient. Transparent bg. |
| `fak-mark-white.svg` | Mark, solid white — dark or photographic backgrounds. |
| `fak-mark-mono.svg` | Mark, solid ink (`#0b1220`) — light backgrounds, print, stamps. |
| `fak-icon-badge.svg` | App/store icon — gradient mark on an ink rounded square. |
| `fak-favicon.svg` | Simplified mark (4 blades, wide-open ring) — stays legible ≤16px. |
| `fak-logo.svg` | Horizontal lockup, white wordmark — **for dark backgrounds**. |
| `fak-logo-ink.svg` | Horizontal lockup, ink — **for light backgrounds**. |
| `fak-mark-512.png` `fak-icon-512.png` `favicon-32.png` `favicon-16.png` | Raster exports (transparent). |

Prefer the SVGs everywhere they render; the PNGs are for contexts that can't take SVG.

## Which mark at which size

- **≥24px / hero / lockups** → `fak-mark.svg` (full 8-ray mark).
- **≤16px favicons** → `fak-favicon.svg`. The full mark's short diagonal rays and
  ring hole collapse below ~20px; the favicon variant drops the short rays and widens
  the aperture so it stays crisp.

## Color

| Token | Hex | Role |
|-------|-----|------|
| Ink | `#0b1220` | Backgrounds, mono mark, wordmark on light |
| Cyan | `#22d3ee` | Gradient start (lower-left) |
| Sky | `#38bdf8` | Gradient mid |
| Azure | `#5b9dff` | Gradient end (upper-right) |
| White | `#ffffff` | Wordmark on dark, mark on photos |

Gradient runs lower-left → upper-right across the mark. On backgrounds where the
gradient can't breathe (favicons under ~24px, single-color print, embroidery),
use `fak-mark-white.svg` or `fak-mark-mono.svg`.

## Clear space & don'ts

- Keep clear space around the mark ≥ the width of one long blade.
- Don't recolor the gradient, rotate the mark, or close the center ring.
- Don't set the wordmark in another typeface — it's hand-vectored monoline geometry,
  not a font; use the lockup SVGs as-is.

_Assets are generated; the source generator lives outside the repo. Edit the SVGs
directly for one-off tweaks, or regenerate the full set to change the system._
