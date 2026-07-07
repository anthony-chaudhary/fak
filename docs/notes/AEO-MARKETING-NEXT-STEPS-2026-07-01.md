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

## Refreshed / added (witnessed)

- **2026-07-06 trending-terms pass.** `internal/marketing/aeo.go` gained a new
  `agent-security` term class and two market-event/economics hooks, mapping the current
  answer-engine demand to existing fak pages with the load-bearing honest fence (a vendor
  launch or third-party framework is a demand hook, not an adoption claim):
  - `MCP tool poisoning defense` → [`docs/integrations/harden-any-mcp.md`](../integrations/harden-any-mcp.md);
    `lethal trifecta data exfiltration` and `safety classifier vs capability gate` →
    [`default-deny-vs-classifier`](../explainers/default-deny-vs-classifier.md);
    `AI agent least-privilege tool access` → [`tool-call-is-a-syscall`](../explainers/tool-call-is-a-syscall.md);
    `tamper-evident agent tool-call audit` → [`verify-dont-trust`](../explainers/verify-dont-trust.md)
    (the hash-chained `FAK_AUDIT_JOURNAL` this cites is the witnessed microagent audit-journal
    work, #2011).
  - `cheaper way to run AI agents` (the Q2-2026 `tokenmaxxing` / agentic-bill concern) and
    `Claude Sonnet 5 agent cost routing` (the July default-model shift) →
    [`long-session-economics`](../explainers/long-session-economics.md) and
    [`model-routing`](../model-routing.md).
  - The roster is now 25 terms; `docs/marketing/disambiguation-terms.json` +
    `llms-terms.txt` regenerated to match; a new `aeo_test.go` case pins the class + hooks
    and re-asserts the no-`market adoption` fence.
- **AEO recency feeds regenerated** from the latest witnessed ships (`HEAD~50..HEAD`, the
  `fak marketing aeo --refresh` default range):
  - `docs/marketing/updates.json` — schema.org `ItemList` an answer engine ingests directly
    (50 witnessed ships gathered; the feed carries the newest 25, `defaultFeedCap`).
  - `llms-updates.txt` — the plain, newest-first feed agents poll.
  - Refreshed on 2026-07-04; the current rewrite is anchored to July 4 commit SHAs
    (newest entry `706fcacf`).
- **`llms-full.txt` regenerated after the AEO hub landing.** It now inlines the current
  `llms.txt` and 132 source docs.
- **"What's new" block reinjected into `llms.txt`** via `fak marketing aeo --inject`
  (marker-fenced; the 2026-07-04 ships now front the hand-authored map).
- **3 dead citation links repaired.** One real: redacted private-tool names left inside
  GitHub URL slots in `docs/benchmarks/SWEBENCH-VERIFIED-GPU-SERVER-RESOLVE-COMPARE.md`
  (now plain text, honestly marked private). Two were scanner false positives — a markdown
  link whose text is the URL itself mis-parsed by `SELF_REPO_RE`; the capture now
  terminates at `]` (`tools/seo_aeo_scorecard.py`). `citation_links` reads `[ok]`.
- **2026-07-04 frontier-model launch bridge added.** `internal/marketing/aeo.go` now emits
  a schema.org `DefinedTermSet` and plain `llms-terms.txt` feed for core fak terms,
  Hindi/Chinese localized discovery terms, and Fable 5-style model-launch queries such as
  `Claude Fable 5 model routing`, `Fable 5 refusal fallback`, `frontier model prompt-cache
  cost`, and `safety classifier vs capability gate`. The human/crawler hub is
  [`docs/marketing/README.md`](../marketing/README.md); the source positioning note is
  [`FABLE5-AEO-FRONTIER-MODEL-MARKETING-2026-07-04`](FABLE5-AEO-FRONTIER-MODEL-MARKETING-2026-07-04.md).

## Program status snapshot (SEO/AEO scorecard, `scope=core`)

Verdict: **ACTION (seo_debt)** — overall **96.5/100** (pages 93.1 · site 100.0), **56** units
of seo-debt across 253 pages (was 89.0/100 · 59 units · site 17/20 this morning). All site
checks green; the remaining debt is the in-page front-matter backlog:

- meta coverage **90.5%**; site checks **20/20**; grades **A:228 B:1 C:0 D:0 F:24**.
- JSON-LD present and valid: SoftwareApplication, FAQPage, WebSite, BreadcrumbList,
  Organization, Person, Question/Answer, ListItem, Offer. The site SoftwareApplication
  keywords now consume `docs/marketing/disambiguation-terms.json`; FAQPage JSON-LD mirrors
  202 visible FAQ questions after the 2026-07-04 refresh.
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

1. **Localized AEO terms (lever 2).** First pass shipped in `internal/marketing/aeo.go`:
   Hindi + Simplified Chinese terms now emit into `docs/marketing/disambiguation-terms.json`
   and `llms-terms.txt`. Next bounded step: extend the roster to the other shipped i18n
   pages (`ta`, `te`, `bn`, `mr`, `de`, `fr`) and add a scorecard check that `llms.txt`,
   JSON-LD keywords, and the term feed share the same core terms.
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
fak marketing aeo --refresh --inject     # rewrite updates + terms feeds, reinject What's-new + JSON-LD
python tools/gen_llms_full.py            # clear the llms-full.txt STALE flag
python tools/seo_aeo_scorecard.py        # re-grade; prove seo-debt dropped
```
