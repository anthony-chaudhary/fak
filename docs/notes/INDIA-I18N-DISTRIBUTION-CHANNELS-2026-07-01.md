---
title: "India i18n distribution channels — where to share the Indian-language entry points (2026-07-01)"
description: "Vetted targets for sharing fak's Indian-language entry points: GitHub awesome-lists, Indic AI ecosystem surfaces, Indian dev communities. All not-yet-posted."
---

# India i18n distribution channels (2026-07-01)

> Companion to the [emerging-market adoption note](CONCEPT-EMERGING-MARKET-ADOPTION-2026-06-30.md)
> (levers 9–10, "community channels" and "per-market positioning") and the
> [i18n hub](../i18n/README.md). That note names channel seeding as **human-owned GTM**;
> this note does the research half: a vetted, checkable target list with what to post
> where and each list's acceptance path.
>
> **Honesty fence: every row below is a TARGET, status `not yet posted`.** Nothing in
> this note claims a submission, a merged PR, or any market adoption. Posting is an
> outward-facing action owned by a human operator — an agent must not mass-submit these.
> One honest, rule-following PR or post per venue, with affiliation disclosed where the
> venue requires it.

## What there is to share

Six localized entry points, each carrying the pitch, the 60-second proof, the install
path, and the market value props (INR cost framing, DPDP-Act residency, self-host,
zero payment rail), then handing off to the English docs:

- [हिन्दी / Hindi](../i18n/hi/README.md) · [தமிழ் / Tamil](../i18n/ta/README.md) ·
  [తెలుగు / Telugu](../i18n/te/README.md) · [বাংলা / Bengali](../i18n/bn/README.md) ·
  [मराठी / Marathi](../i18n/mr/README.md) · [简体中文 / Chinese](../i18n/zh/README.md)
- Hub with the honesty fence (machine-authored, native review pending): [`docs/i18n/`](../i18n/README.md)

The in-language pages are the *evidence* that makes fak worth listing for an Indian
audience — a submission should link the i18n hub alongside the English README.

## Tier 1 — GitHub awesome-lists (PR-able, each has its own bar)

| Target | Why it fits | What to submit | Status |
|---|---|---|---|
| [e2b-dev/awesome-ai-agents](https://github.com/e2b-dev/awesome-ai-agents) | Long-standing, widely-cited agent list | fak as agent kernel/gateway (guard + cache + audit), link README + i18n hub | not yet |
| [kyrolabs/awesome-agents](https://github.com/kyrolabs/awesome-agents) | Curated open-source agent tooling list | Same one-liner; open-source/self-host angle | not yet |
| [slavakurilyak/awesome-ai-agents](https://github.com/slavakurilyak/awesome-ai-agents) | 300+ agentic resources, tracks star counts | Same one-liner | not yet |
| [ashishpatel26/500-AI-Agents-Projects](https://github.com/ashishpatel26/500-AI-Agents-Projects) | Use-case-organized catalog, India-based maintainer, large reach | fak under an infra/governance use case, per its CONTRIBUTION.md metadata format | not yet |
| [awesome-selfhosted/awesome-selfhosted](https://github.com/awesome-selfhosted/awesome-selfhosted) | fak is genuinely self-hosted, Apache-2.0, single binary | Check their strict rules first (maturity window, category fit, affiliation disclosure required) | not yet |
| [avelino/awesome-go](https://github.com/avelino/awesome-go) | fak is a Go project; the list is the Go ecosystem front door | Read their quality bar first (CI, coverage, godoc); submit only if the bar is met — do not force it | not yet |

Rules that apply to all of tier 1: read the venue's `CONTRIBUTING.md` before the PR,
disclose that the submitter is the project author where the venue asks (awesome-selfhosted
does), one venue per PR, never resubmit a rejected entry without addressing the reason.

## Tier 2 — Indic AI / language ecosystem (consulted, mostly *not* posting targets)

- [AI4Bharat indicnlp_catalog](https://github.com/AI4Bharat/indicnlp_catalog) — the
  collaborative catalog of NLP resources *for Indic languages* (IndicBERT, IndicTrans2,
  iNLTK, L3Cube and friends live here). **Consulted and rejected as a posting target:**
  fak is agent infrastructure, not an Indic NLP resource — a fak entry there would be
  off-topic spam. It stays in this note as the map of the ecosystem whose developers the
  i18n pages serve, and as the place to *find* partners (e.g. teams running Indic models
  behind OpenAI-compatible servers, which fak fronts today).
- GitHub topic [`indian-languages`](https://github.com/topics/indian-languages) — same
  fence: fak must not tag itself into Indic-NLP discovery just for having entry pages.
  The honest hook is the reverse direction: fak governs/caches *any* OpenAI-compatible
  endpoint, which includes Indic-model servers (Sarvam-style endpoints, local Ollama
  running Indic fine-tunes). If a real Indic-model integration recipe ships someday,
  revisit.

## Tier 3 — Indian developer communities (human-owned GTM)

Named for completeness; these are posts a human makes, in-language where the venue is:

- **r/developersIndia** (the large India dev subreddit; self-promo rules apply — share as
  "I localized our agent-security tool into 5 Indian languages", not an ad).
- **dev.to / Hashnode** — both have large Indian developer bases; a Hindi or Hinglish
  walkthrough of the 60-second proof is the natural artifact.
- **IndiaAI ecosystem / campus + hackathon circuits** — the concept note's lever 9;
  the i18n pages are the handout.

The per-market positioning line (concept-note lever 10) to reuse verbatim:
**"cheaper agents on your stack, data in-country."**

## Suggested one-liner (EN + HI)

- EN: *fak — an Apache-2.0 agent kernel in one Go binary: checks every tool call before
  it runs, reuses cache across long sessions (~4.1× less work vs a tuned warm-cache
  stack), keeps data on your machine. Docs in हिन्दी, தமிழ், తెలుగు, বাংলা, मराठी, 中文.*
- HI: *fak — एक Go बाइनरी में agent kernel: हर tool call चलने से पहले जाँच, लंबी sessions
  में cache का दोबारा इस्तेमाल (~4.1× कम काम), डेटा आपकी मशीन पर ही। हिन्दी में शुरुआत:
  docs/i18n/hi/।*

Numbers discipline: quote the tuned ~4.1× figure, never the naive ~60× as headline —
same rule as [`BENCHMARK-AUTHORITY.md`](../../BENCHMARK-AUTHORITY.md).

## Machine-side complement (already tracked elsewhere)

The AEO program's localized-terms lever (emit Hindi/Indic disambiguation terms from
`internal/marketing/aeo.go` so answer engines name fak for "एजेंट कर्नेल") is tracked in
the [AEO next-steps note](AEO-MARKETING-NEXT-STEPS-2026-07-01.md) — discovery via answer
engines, distinct from the venue posting above.

## Verify

```
grep -ri "posted" docs/notes/INDIA-I18N-DISTRIBUTION-CHANNELS-2026-07-01.md   # every status must still read "not yet"
ls docs/i18n/hi docs/i18n/ta docs/i18n/te docs/i18n/bn docs/i18n/mr           # the artifacts a submission links to
```

When a submission actually merges, flip that row's status to the merged PR URL — the
row's evidence is the upstream PR, never this note's say-so.
