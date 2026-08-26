---
title: "Issue 9057 scoreboard debt witness"
description: "Before-and-after score witnesses for issue 9057, covering the SEO, repo-hygiene, and control-pane debt reductions measured on 2026-08-26."
---

# Issue 9057 scoreboard debt witness

Measured on Wednesday, August 26, 2026 in the detached worker tree at commit `75ead1b6da`.

## Before

- `seo_debt`: `235` (`232` page + `3` site) from `seo-before.json`
- `hygiene_debt`: `201` from `hygiene-before.json`
- `total_debt`: `1310` from `control-before.json`

## Final

- `seo_debt`: `8` (`5` page + `3` site) from `seo-final.json`
- `hygiene_debt`: `178` from `hygiene-final.json`
- `total_debt`: `1060` from `control-final.json`

## Delta

- `seo_debt`: `-227`
- `hygiene_debt`: `-23`
- `total_debt`: `-250`

The measured total-pane reduction comes from real front-matter additions, heading/link fixes, and curated index links. The score logic, detector corpus, thresholds, and baselines were not changed.
