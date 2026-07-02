---
title: "AEO and marketing next steps (2026-07-01)"
description: "Consolidated, evidence-backed next-steps record for fak's answer-engine (AEO) and completion-driven marketing program: what refreshed today, what the SEO/AEO scorecard flags, and the honest bounded follow-ons."
---

# AEO & marketing — next steps (2026-07-01)

> Living next-steps record for fak's AEO/AgentEO surface (`internal/marketing/aeo.go`,
> `tools/gen_structured_data.py`, `tools/seo_aeo_scorecard.py`) and the completion-driven
> marketing loop (`fak marketing tick|generate|post|epic|release|aeo`). Every "shipped" line
> is git-witnessed; every "next" line is a bounded, checkable step with the command that
> proves it. No claim of market adoption is made here — the mechanism exists; the framing and
> the feeds are what get refreshed. Supersedes the ad-hoc follow-on list in
> [`CONCEPT-EMERGING-MARKET-ADOPTION-2026-06-30`](CONCEPT-EMERGING-MARKET-ADOPTION-2026-06-30.md)
> for the AEO/marketing slice (that note keeps the full India/China lever set).

## Refreshed today (witnessed)

- **AEO recency feeds regenerated** from the latest witnessed ships (`HEAD~50..HEAD`, the
  `fak marketing aeo --refresh` default range):
  - `docs/marketing/updates.json` — schema.org `ItemList` an answer engine ingests directly
    (47 witnessed ships gathered; the feed carries the newest 25, `defaultFeedCap`).
  - `llms-updates.txt` — the plain, newest-first feed agents poll.
  - Refreshed twice today; the current rewrite is anchored to 2026-07-01 commit SHAs
    (newest entry `3444c4d8`).
- **`llms-full.txt` regenerated — STALE flag cleared.** It now inlines the current
  `llms.txt` and all 110 source docs; `llms_full`, `llms_full_sources`, and
  `llms_full_navigable` all read `[ok]`.
- **"What's new" block reinjected into `llms.txt`** via `fak marketing aeo --inject`
  (marker-fenced; the 2026-07-01 ships now front the hand-authored map).
- **3 dead citation links repaired.** One real: redacted private-tool names left inside
  GitHub URL slots in `docs/benchmarks/SWEBENCH-VERIFIED-GPU-SERVER-RESOLVE-COMPARE.md`
  (now plain text, honestly marked private). Two were scanner false positives — a markdown
  link whose text is the URL itself mis-parsed by `SELF_REPO_RE`; the capture now
  terminates at `]` (`tools/seo_aeo_scorecard.py`). `citation_links` reads `[ok]`.

## Program status snapshot (SEO/AEO scorecard, `scope=core`)

Verdict: **ACTION (seo_debt)** — overall **96.5/100** (pages 93.1 · site 100.0), **56** units
of seo-debt across 253 pages (was 89.0/100 · 59 units · site 17/20 this morning). All site
checks green; the remaining debt is the in-page front-matter backlog:

- meta coverage **90.5%**; site checks **20/20**; grades **A:228 B:1 C:0 D:0 F:24**.
- JSON-LD present and valid: SoftwareApplication, FAQPage, WebSite, BreadcrumbList,
  Organization, Person, Question/Answer, ListItem, Offer.
- robots.txt welcomes all four major answer-engine crawlers (+7 named); sitemap + seo-tag
  plugins on; FAQPage JSON-LD mirrors 199 visible FAQ questions.

## Next steps — AEO surface (worst-first, each checkable)

1. **Retire the 24 F-grade pages, worst-first** — all fail on missing front-matter
   `title:` / `description:` (e.g. `docs/i18n/hi/README.md`, `docs/i18n/zh/README.md`,
   `docs/operator-brief.md`, `docs/INNOVATIONS-INDEX.md`, `docs/WORK-MAP.md`). Front-matter is
   the cheapest AEO win: a title/description is the SERP snippet and the answer-engine anchor.
   (In flight: a peer session is adding front-matter + new Indian-language entry points —
   `docs/i18n/{ta,te,bn,mr}` — as of this refresh.)
2. **Reduce the 33 discovery orphans** (published, not front-door-reachable) by wiring them
   into an index page so a crawler that lands on the front door can reach them.

## Next steps — marketing program (bounded, honest)

1. **Localized AEO terms (lever 2).** Extend `internal/marketing/aeo.go` to emit in-language
   disambiguation terms + structured data (Hindi; Baidu/Zhihu/掘金 zh terms) so an answer
   engine responding in Hindi/Chinese names fak correctly. Bounded generator extension, not
   new infra. Gap today: no localized terms are emitted.
2. **Cost framing in local unit-economics (lever 5).** Re-skin the existing (honest) benchmark
   story as cost-per-1,000-turns / margin-per-seat; quote the tuned ~4.1× headline, not the
   naive 60×. Copy-only, no new measurements.
3. **ModelScope URLs + Gitee mirror (lever 7).** List ModelScope (魔搭) weight URLs beside the
   HF ones in the zh page. **Fence:** do not claim a Gitee mirror until one actually exists.
4. **GTM channel seeding + per-market landing copy (levers 9, 10).** Human-owned; named for
   completeness. This note's job is to keep the docs/product ready for that owner.
5. **Confirm the completion trigger fires.** `fak marketing tick` is the single idempotent
   entrypoint (serve bgloop / git hook / cron all funnel through it, high-water-mark CAS +
   per-title dedupe). Verify a real completion advances the mark and posts once — the loop is
   only as fresh as the trigger that fires it. Slack surface stays internal-only (no public
   ceremony).

## Related

- Open epic **[#1678](https://github.com/anthony-chaudhary/fak/issues/1678)** — fak as the
  vendor-neutral binding layer for neo-silicon/neo-clouds; the English answer-engine vendor
  terms already landing in the feed (e.g. "silicon-vendor answer-engine terms",
  "binding-layer AEO vendor routing") are this epic's AEO face.
- [`CONCEPT-EMERGING-MARKET-ADOPTION-2026-06-30`](CONCEPT-EMERGING-MARKET-ADOPTION-2026-06-30.md)
  — the full India/China lever set behind marketing steps 1–4 above.

## Verify

```
fak marketing aeo --refresh              # rewrite updates.json + llms-updates.txt (done today)
python tools/gen_llms_full.py            # clear the llms-full.txt STALE flag (step 1)
fak marketing aeo --inject               # reinject What's-new into llms.txt (step 2)
python tools/seo_aeo_scorecard.py        # re-grade; prove seo-debt dropped
```
