---
title: "fak release-readiness scorecard — release-debt measuring stick"
description: "fak's deterministic release-readiness scorecard: KPIs across the release lifecycle — discover, automate, validate, trust — folded into a composite score and the headline release-debt metric, re-derived from git + the tracked tree + live release signals."
---

# Release-readiness scorecard — can fak release at agentic speed

<!-- release-readiness-scorecard: 2026-11-24 · process: tools/release_readiness_scorecard.py -->

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
| **Release-debt (total HARD defects)** | **2** |
| Composite score | 88.6/100 (grade B) |
| @latest staleness | v0.57.0 — 342 commits / 10.4d behind HEAD → **VERY_STALE** |
| Release lifecycle | discover 100 · automate 64 · validate 100 · trust 100 |
| Documented gotcha count | 1 |
| Stable rollback anchors | 1 |
| Unwitnessed (offline) signals | 0 |

## The release lifecycle, band by band

### Discover — an agent can find & invoke the release path — 100/100 (debt 0)

| KPI | State | Fix if open |
|---|---|---|
| `fak release` is a dispatched verb | ✅ met |  |
| AGENTS.md documents the release path | ✅ met |  |
| llms.txt points at the release path | ✅ met |  |
| `fak release-staleness` exists | ✅ met |  |

### Automate — the machine cuts on green, not a human — 64/100 (debt 2)

| KPI | State | Fix if open |
|---|---|---|
| cadence can cut on a scheduled tick | ✅ met |  |
| scheduled auto-cut is DEFAULT-ON (kill switch, not arm) | ✅ met |  |
| staleness signal wired into make/CI | ✅ met |  |
| @latest is not VERY_STALE | ❌ **debt** | Cut a release; automate the cadence (#1355) |
| @latest is FRESH vs HEAD | ❌ **debt** | Cut on green at agentic cadence (#1355) |

### Validate — the cut is gated and the publish verified — 100/100 (debt 0)

| KPI | State | Fix if open |
|---|---|---|
| release helpers carry unit tests | ✅ met |  |
| a published release is verified | ✅ met |  |
| failed published-release verification quarantines the release | ✅ met |  |
| single-writer release lock present | ✅ met |  |
| documented gotcha count <= 1 | ✅ met |  |

### Trust — stable anchors, signed artifacts, rollback — 100/100 (debt 0)

| KPI | State | Fix if open |
|---|---|---|
| a stable/* rollback anchor exists | ✅ met |  |
| artifacts carry signing/provenance | ✅ met |  |
| linux/arm64 leg actually shipped | ✅ met |  |
| recent release carries archives+checksums | ✅ met |  |

**Next action:** Cut a release; automate the cadence (#1355)

