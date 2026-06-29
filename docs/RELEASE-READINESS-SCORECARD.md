---
title: "fak release-readiness scorecard — release-debt measuring stick"
description: "fak's deterministic release-readiness scorecard: KPIs across the release lifecycle — discover, automate, validate, trust — folded into a composite score and the headline release-debt metric, re-derived from git + the tracked tree + live release signals."
---

# Release-readiness scorecard — can fak release at agentic speed

<!-- release-readiness-scorecard: 2026-06-29 · process: tools/release_readiness_scorecard.py -->

The measuring stick for fak's **release velocity under truth**: a kernel that
writes hundreds of commits a day must be able to **cut, validate, publish, and
roll back** a release at the same speed — or `@latest` rots far behind HEAD. Every
number below is re-derived from git + the git-tracked tree + live release signals
by `tools/release_readiness_scorecard.py` — no hand-entry. The headline metric is
**release-debt**: the count of concrete, mechanical defects that keep fak from
releasing at agentic speed. The companion epic (#1354) retires the worst-first
defect by adding the missing release affordance.

> Regenerate: `python tools/release_readiness_scorecard.py --markdown --stamp DATE > docs/RELEASE-READINESS-SCORECARD.md`

## Headline

| Metric | Value |
|---|---|
| **Release-debt (total HARD defects)** | **8** |
| Composite score | 43.8/100 (grade F) |
| @latest staleness | v0.34.0 — 1919 commits / 4.1d behind HEAD → **VERY_STALE** |
| Release lifecycle | discover 62 · automate 20 · validate 75 · trust 17 |
| Documented gotcha count | 5 |
| Stable rollback anchors | 0 |
| Unwitnessed (offline) signals | 0 |

## The release lifecycle, band by band

### Discover — an agent can find & invoke the release path — 62/100 (debt 1)

| KPI | State | Fix if open |
|---|---|---|
| `fak release` is a dispatched verb | ❌ **debt** | Add a `fak release` subcommand (#1356) |
| AGENTS.md documents the release path | ✅ met |  |
| llms.txt points at the release path | ✅ met |  |
| `fak release-staleness` exists | ✅ met |  |

### Automate — the machine cuts on green, not a human — 20/100 (debt 3)

| KPI | State | Fix if open |
|---|---|---|
| cadence can cut on a scheduled tick | ❌ **debt** | Add guarded auto-cut to release-cadence.yml (#1355) |
| staleness signal wired into make/CI | ✅ met |  |
| @latest is not VERY_STALE | ❌ **debt** | Cut a release; automate the cadence (#1355) |
| @latest is FRESH vs HEAD | ❌ **debt** | Cut on green at agentic cadence (#1355) |

### Validate — the cut is gated and the publish verified — 75/100 (debt 1)

| KPI | State | Fix if open |
|---|---|---|
| release helpers carry unit tests | ✅ met |  |
| a published release is verified | ✅ met |  |
| single-writer release lock present | ✅ met |  |
| documented gotcha count <= 1 | ❌ **debt** | Eliminate the chicken-egg gotchas (#1368) |

### Trust — stable anchors, signed artifacts, rollback — 17/100 (debt 3)

| KPI | State | Fix if open |
|---|---|---|
| a stable/* rollback anchor exists | ❌ **debt** | Cut the first stable/* tag (#1370) |
| artifacts carry signing/provenance | ❌ **debt** | Sign artifacts with cosign/SLSA (#1372) |
| linux/arm64 leg actually shipped | ❌ **debt** | Ship the arm64 asset (#1371) |
| recent release carries archives+checksums | ✅ met |  |

**Next action:** Add guarded auto-cut to release-cadence.yml (#1355)

