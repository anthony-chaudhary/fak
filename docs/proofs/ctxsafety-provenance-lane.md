---
title: "C8 provenance lane render contract"
description: "Render-level contract for WITNESSED / OBSERVED / MODELED provenance as a visible, non-blended channel on context-safety visuals."
---

# C8 provenance lane render contract

Status: design contract for ctxsafety/visual (#1225). This is a render-level
requirement spec, not a new renderer.

Every rendered primitive that carries a number or point must carry exactly one
`provenance_lane`: `WITNESSED`, `OBSERVED`, or `MODELED`. The lane is a visible
channel, not metadata hidden in a tooltip. The renderer must encode the lane in
the primitive itself and in the legend.

## Lane definitions

| Lane | Data source | Required channel |
|---|---|---|
| `WITNESSED` | fak-authored facts: `max|Δ|=0` bit-exact comparisons and reuse-bit facts from fak's own cache path. | Green `#1a7f37`, solid fill, legend label starts with `WITNESSED`. |
| `OBSERVED` | External telemetry relayed by fak, such as provider `cache_read`. | Blue `#0969da`, diagonal hatch, legend label starts with `OBSERVED`. |
| `MODELED` | Projected page-down-tier values and other estimates until measured on-device. | Amber `#bf8700`, crosshatch, legend label starts with `MODELED`. |

Color and hatch are both load-bearing. A grayscale print, color-blind view, or
low-contrast embed must still distinguish the three lanes.

## Non-blending rule

No primitive may render a MODELED point identically to a WITNESSED point. A
series must not be promoted from `MODELED` to `WITNESSED` by sharing a color,
hatch, legend entry, or lane key with witnessed data.

Do not sum, average, stack, or smooth across provenance lanes into one rendered
series. If a chart needs all three values at the same x coordinate, render three
separate primitives or three faceted lanes. The same rule applies to segments:
a provider `cache_read` segment is `OBSERVED`; fak's reuse-bit segment is
`WITNESSED`; a projected page-down-tier segment is `MODELED`.

## Required honesty gate

The paired honesty test is `TestModeledPrimitiveCannotRenderOnWitnessedLane` in
`internal/cachewitness`. It constructs a projected page-down-tier primitive with
`MODELED` provenance and refuses it on the `WITNESSED` lane. The adjacent style
test asserts that `WITNESSED`, `OBSERVED`, and `MODELED` have distinct visible
encodings, so a modeled series cannot be emitted with the witnessed style by
accident.
