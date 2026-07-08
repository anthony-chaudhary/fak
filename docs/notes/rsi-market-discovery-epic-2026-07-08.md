---
title: "RSI market-discovery epic: file a ticket the moment the market moves (2026-07-08)"
description: "A reviewable seed of 12 broad research tickets for the self-improving work-discovery loop — freshly crawl trending GitHub/HN/Reddit/X and new-release firehoses, deep-index and cross-source dedup, expose domain topic packs, and drive it all from an operator-steerable superloop with a concept-disambiguation + AEO honesty gate. Anchored on the shipped arXiv+GitHub+Hacker News idea-scout spine so every ticket is disjoint and independently workable."
slug: rsi-market-discovery-epic-2026-07-08
date: 2026-07-08
---

# RSI market-discovery epic (2026-07-08)

**Goal.** Determine new work from *market conditions*: every few hours, crawl the
absolute-latest trending sources, dedup against what we already know, and file a
triage-ready ticket for any public idea worth working — ideally within moments of
its release. Then walk that backlog from an operator-steerable superloop and keep
the concept-disambiguation / QA / docs / AEO surface honest as it grows.

This note is the **reviewable seed** for that epic: 12 broad, self-contained
research tickets. Each is a real GitHub issue candidate — file with `gh issue
create` or fold into `fak idea-scout` topic packs. The tickets are deliberately
*broad research* units (the goal asked for breadth), but each names a concrete
first deliverable and an honest witness so it can't quietly become an overclaim.

## Working spine (what already exists — do not rebuild)

- `internal/ideascout` — the discovery kernel. Per-topic sources are pure folds
  over wire bytes (`ParseArxivAtom`, `ParseGitHubRepos`, **`ParseHackerNewsJSON`**),
  scored transparently (`ScoreCandidate`), deduped against a seen-cache + the
  existing issue backlog, and rendered to triage-ready plans. Only `LiveFetcher`
  touches the network; `--live` gates all issue creation.
- **arXiv + GitHub + Hacker News** sources ship today (HN via the public Algolia
  `search_by_date` API, newest-first — a trending front-page story becomes a
  candidate within moments of release). This is the anchor: every source ticket
  below is "add one more `Fetch*`/`Parse*` pair the same way."
- `internal/dogfoodissues` — files/updates issues idempotently (dedup by title,
  update-not-refile). `internal/superloop*` + `cmd/fak/superloop_drive.go` — the
  operator loop that walks work. `docs/concept-disambiguation-scorecard/` — the
  concept-confusion QA surface. `tools/seo_aeo_scorecard.py` — the AEO gate.

Because the spine is source-pluggable and dedup/scoring/filing are shared, the
tickets are disjoint: a new source is one lane, the scheduler is another, the
steering surface another. One worker can own any ticket end to end.

---

## Axis 1 — Fresh market-condition discovery (crawl + deep index)

### T1 — Reddit trending source adapter
**Deliverable:** `ParseRedditJSON` + `FetchReddit` (public `.json` listings for
`r/LocalLLaMA`, `r/MachineLearning`, `r/rust`, `r/programming`), a `Topic.Reddit`
query field, and an upvote/comment engagement signal reusing the `points` path.
**Why now:** Reddit is where a release is *discussed* first; the HN adapter proves
the pattern is cheap to extend. **Witness:** fixture-JSON parse test + a gather
test that filters by a min-score, mirroring the HN tests. **Lane:** `ideascout`.

### T2 — X/Twitter release-signal adapter (research)
**Deliverable:** research + a spike adapter for surfacing trending model/tool
releases from X. **Why now:** highest-velocity release channel, but auth/ToS-gated
— so this ticket is explicitly *research*: evaluate API tiers vs. nitter-style
mirrors vs. curated list polling, land whichever is legal + hermetically testable
as a pure parser. **Witness:** a decision note + a fixture-parse test for the
chosen wire shape; no live credential in the tree. **Lane:** `ideascout`.

### T3 — New-release firehose adapter (Product Hunt / npm / PyPI / HF-trending)
**Deliverable:** a `releases` source that polls package/model registries for
*brand-new* high-velocity releases (npm/PyPI new-and-trending, Hugging Face
trending models, GitHub `created:>=today` stars-velocity, Product Hunt). **Why
now:** "within moments of release" is best served by registries that timestamp
releases directly. **Witness:** per-registry fixture parse tests; a velocity score
(stars or downloads per hour) feeding `ScoreCandidate`. **Lane:** `ideascout`.

### T4 — Cross-source deep dedup & clustering
**Deliverable:** cluster candidates that describe *the same* idea across arXiv +
GitHub + HN + Reddit into one ticket (lexical shingle + optional local-embedding
similarity), so a release trending in three places files once, not thrice. **Why
now:** more sources multiply duplicates; the seen-cache dedups within a source,
not across them. **Witness:** a clustering test over a mixed fixture set asserting
N candidates → M<N tickets with a provenance list. **Lane:** `ideascout`.

### T5 — Deep index of the discovered corpus
**Deliverable:** a durable, queryable index of everything discovery has *seen*
(not just filed) — source, first-seen time, score trajectory, triage outcome — so
the loop can answer "what's rising?" and "did we miss X?". **Why now:** the
seen-cache is filed-only and lossy; a real index makes the loop auditable and
enables the meta-loop (T9). **Witness:** an append-only store + a `fak idea-scout
index` read verb with a golden query test. **Lane:** `ideascout`.

## Axis 2 — Minimum working spine to iterate

### T6 — Source-adapter contract + conformance suite
**Deliverable:** promote the implicit `Fetch*`/`Parse*` convention into an explicit
`Source` interface (`Name`, `Fetch`, `Parse`) + a conformance test any adapter
must pass (SourceID stability, URL fallback, empty-input safety, timestamp parse).
**Why now:** T1–T3 each re-derive the pattern; a contract makes "iterate separate
thoughts on this" cheap and keeps new sources honest. **Witness:** the existing
arXiv/GitHub/HN adapters pass the conformance suite unchanged. **Lane:** `ideascout`.

## Axis 3 — Support domain-specific agent builders

### T7 — Domain topic packs
**Deliverable:** shippable topic bundles per vertical (security, legal, bio/health,
robotics, data-eng) as `--config` presets, so someone building a domain-specific
agent points `fak idea-scout` at their field and gets a live domain backlog. **Why
now:** discovery is only interesting if it's *your* market; packs turn the generic
kernel into a per-domain scout. **Witness:** each pack is a validated config with a
dry-run smoke test over fixtures. **Lane:** `ideascout`.

### T8 — "Scout my domain" onramp recipe
**Deliverable:** a docs recipe + example showing a domain-agent builder how to
define topics, run the scout on a schedule, and wire filed issues into their own
tracker — the interesting/supportive front door for outside builders. **Why now:**
item (3) — be genuinely useful to people building domain-specific agents. **Witness:**
a runnable example under `examples/` + an INDEX-linked docs page passing the AEO
scorecard. **Lane:** `docs` / `examples`.

## Axis 4 — Superloops & operator-steerable meta-loops

### T9 — Discovery superloop + outcome-feedback meta-loop
**Deliverable:** a scheduled superloop that walks all sources every few hours, and
a *super-meta* loop above it that reweights topics from triage outcomes (issues
that got worked = signal up; `wontfix`/`duplicate` = signal down), closing the RSI
feedback loop. **Why now:** items (1)+(4) — discovery must self-tune, not just
fan out. **Witness:** a deterministic reweight function with a table test over
labeled outcomes; the schedule is a dry-runnable plan. **Lane:** `superloop` /
`ideascout`.

### T10 — Operator steering surface
**Deliverable:** one higher-level control to pause/resume, reweight, re-scope, and
budget the discovery loop, with an audit trail of every steer — so an operator
drives from above without editing code. **Why now:** item (4) — "operator
steerable from a higher level." **Witness:** a steering-state store + a `fak
idea-scout steer` verb with an audit-log round-trip test. **Lane:** `ideascout` /
`cmd`.

## Axis 5 — Concept disambiguation, QA, docs & AEO

### T11 — Concept-disambiguation of discovered ideas
**Deliverable:** map each candidate onto the fak/DOS concept taxonomy and flag
confusable neighbors before filing, extending the concept-disambiguation scorecard
to the discovery stream. **Why now:** item (5) — keep the growing backlog from
conflating distinct concepts. **Witness:** a scorecard row per source with a
labeled confusion fixture. **Lane:** `conceptbench` / `docs`.

### T12 — Honesty gate + AEO explainer for the discovery loop
**Deliverable:** (a) a QA gate that every auto-filed issue body is honest — no
market-adoption claim, no unrun benchmark, source-stamped and triage-labeled; and
(b) an explainer "how fak finds its own work" for the AEO surface. **Why now:**
item (5) — the loop's own output must pass the same honesty bar we hold docs to.
**Witness:** a linter over rendered issue bodies + an INDEX-linked explainer that
does not regress `tools/seo_aeo_scorecard.py`. **Lane:** `docs` / `ideascout`.

---

## Honest scope

- **Shipped today:** the arXiv + GitHub + **Hacker News** spine (T0 anchor). Every
  other ticket is open research/build, not done.
- No source ticket claims live coverage until its `Fetch*` lands with a witness;
  auth/ToS-gated sources (T2) stay research until a legal, testable path exists.
- Filing is always `--live`-gated and dedup-guarded; this epic files *candidates
  for tickets*, not adoption or novelty claims.

## Filing

File these 12 as `popularization`-style research issues (`gh issue create`, one per
ticket, deduped by title) or fold T1/T3/T7 directly into `fak idea-scout` topic
packs. Track landings by the `(fak <leaf>)` ship stamp on each closing commit.
