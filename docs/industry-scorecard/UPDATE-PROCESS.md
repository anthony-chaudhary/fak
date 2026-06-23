---
title: "fak industry scorecard — the update process"
description: "How the industry scorecard stays current on both cadences: as the industry changes (new dimensions, moved SOTA bars) and as fak changes (a gap becomes a measured row)."
---

# Keeping the scorecard current — the two cadences

An outward scorecard rots in two directions. This is how each is caught and closed.

## 1. As the industry changes

New techniques appear and published SOTA bars move. Two mechanisms catch it:

- **New dimension → coverage drops.** When the field starts competing on something new (a new quant format, a new decoding trick), add it to `tools/industry_scorecard.data/_taxonomy.json`. It is immediately uncovered, so `coverage` falls and the dimension shows up in `--gaps` until fak is positioned on it.
- **Stale SOTA bar → industry-drift backlog.** Every dimension carries a `source_date` and every competitor a `last_reviewed`. `python tools/industry_scorecard.py --stale` lists the bars past the `industry_review_window_days` window — re-check them on the web and update the number + date. (Advisory, never parity-debt: a number does not become false the day it crosses the window, it wants a look.)

## 2. As fak changes

- **A benchmark lands → a gap becomes a measured row.** When a new fak number ships (traced in `BENCHMARK-AUTHORITY.md`), turn the relevant `no-claim` position into a `lead`/`parity`/`trails` row, citing the commit/artifact. `--gaps` lists the honest gaps that are candidates to measure.
- **A fak number ages → re-confirm.** `measured_on` drives the `freshness` KPI; old measurements are flagged (advisory) to re-confirm when a bench node is free.
- **A number changes → never hand-edit the doc.** Edit the data file, regenerate.

## The commands

```bash
python tools/industry_scorecard.py               # the scorecard + both work-lists
python tools/industry_scorecard.py --gaps        # coverage backlog (what to position/measure)
python tools/industry_scorecard.py --stale       # industry-drift backlog (SOTA bars to re-check)
python tools/industry_scorecard.py --verify-sources   # fak numbers still match their artifacts
python tools/industry_scorecard.py --markdown-dir docs/industry-scorecard  # regenerate this folder
```

## The one rule that overrides everything

**Never invent a number or a win.** A fak figure must already exist in `BENCHMARK-AUTHORITY.md` / a committed artifact; a competitor figure must cite a real source. If a fix would require manufacturing evidence, the honest row is a `no-claim` gap — not a guess. The `/industry-score` skill enforces this.

