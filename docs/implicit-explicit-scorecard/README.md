---
title: "fak implicit-explicit scorecard - is every relied-on concept directly named"
description: "Inward naming scorecard: detects concepts that are assumed or hinted at but never directly named (hedged prose, magic literals, code-only identifiers, doc-only headings) and drives naming them. Two driven numbers: coverage of the discovered implicit-signal space and implicitness-debt."
---

# Implicit-explicit scorecard - naming what the system only assumes

<!-- implicit-explicit-scorecard: 2026-07-03 seed - process: tools/implicit_explicit_scorecard.py - data: tools/implicit_explicit_scorecard.data/ -->

The sibling [concept-disambiguation scorecard](../concept-disambiguation-scorecard/) grades the names that EXIST - of the similar-sounding names, is each distinct concept crystal-clear? This card asks the question that comes BEFORE it: **does each concept the system relies on have a direct name at all, or is it merely implicit - assumed by convention, hedged in prose ("the so-called warm window"), encoded as a magic literal repeated across files, named for the compiler but invisible in the docs, or named in prose with no code symbol behind it?** Every number below is re-derived by `tools/implicit_explicit_scorecard.py` and cross-checked against the real tree (the evidence must appear; a claimed named_symbol must resolve; a doc_anchor must exist). No verdict is hand-typed.

> Regenerate: `python tools/implicit_explicit_scorecard.py --markdown-dir docs/implicit-explicit-scorecard`.

## Headline

| Metric | Value |
|---|---|
| **Score** | **43.4/100** (grade F) = 4.3/10 |
| **Coverage** | **12.9%** (29/225 implicit-concept signals positioned) |
| **Implicitness-debt** | **196** (naming 0 + coverage 196) |
| Explicit concepts | 0 of 20 positioned |
| As of | 2026-07-03 (fak dev) |

> **Read this right.** The score is deliberately LOW at birth: it grades the WHOLE implicit-signal space discovered in the tree, not the few concepts already catalogued. A low coverage number is the honest statement that most assumed/hinted concepts are not yet named - which is exactly the debt this scorecard exists to retire.

## The explicitness ladder

| Verdict | Means |
|---|---|
| * explicit | a code identifier resolves AND a definition is written at a doc anchor that exists - named in both worlds |
| c named-code | a real identifier exists, but no written definition anywhere |
| d named-doc | defined in prose at a real anchor, but no code symbol behind it |
| ~ hinted | only hedges/patterns refer to it, but a proposed_name exists - naming planned |
| . latent | pure pattern; no name anywhere, no plan yet |

## Standing at a glance

```text
implicit-explicit chart - 20 concepts - score 43.4/100 (grade F) - implicitness-debt 196

explicitness ladder (count of concepts, named -> fog):
  * explicit     ............................ 0
  c named-code   #######################..... 9
  d named-doc    ............................ 0
  ~ hinted       ############################ 11
  . latent       ............................ 0

explicitness mix by signal (each cell = one concept):
  code-only       cccccccc             (8 concept(s); 0 explicit)
  doc-only        ~~~~~~~~             (8 concept(s); 0 explicit)
  hinted          ~                    (1 concept(s); 0 explicit)
  latent-literal  c~~                  (3 concept(s); 0 explicit)

coverage by signal kind (positioned / discovered):
  doc-only        ###......................... 13/138
  code-only       ######...................... 11/49
  latent-literal  ###......................... 4/37
  hinted          ############################ 1/1

signal coverage  [####............................] 12.9%  (29/225 implicit signals positioned)

legend: * explicit   c named-code   d named-doc   ~ hinted   . latent
```

## The concepts (most explicit first)

| | Verdict | Signal | Canonical - definition | Proposed / named |
|---|---|---|---|---|
| c | named-code | code-only | **RealRunner test seam** - the repo-wide convention of a RealRunner function - the production process-runner injected where tests substitute a fake - independently re-declared in mergepreview, modver, releasestale and more (19 files), never documented as the convention it is | `RealRunner` |
| c | named-code | code-only | **background-command window gate** - the internal/windowgate helper every spawned subprocess passes through so background commands never flash a console window on Windows (no-op elsewhere) - called from 77 production files, invisible in the docs | `ConfigureBackgroundCommand` |
| c | named-code | code-only | **cache admission verdict** - the cachemeta verdict string recording whether an entry was admitted to the cache and why (AdmissionFromVerdict maps abi verdict kinds onto it) - in 12 production files, never mentioned in the cache docs | `AdmissionVerdict` |
| c | named-code | code-only | **evidence ref** - the struct that points a witness at an artifact it can read back (a file path, a git object) - the unit of witnessable evidence in taskmgr/loopmgr, in 17 production files and zero docs despite the witness pipeline being a headline concept | `EvidenceRef` |
| c | named-code | code-only | **indented-JSON emitter** - the helper (declared per package, 59 production files) that emits the canonical two-space-indented JSON for --json verb output, with a no-escape variant - the de-facto JSON output contract of the CLI, stated in no doc | `writeIndentedJSON` |
| c | named-code | code-only | **per-verb ParseFlags convention** - the convention that every CLI verb declares its own ParseFlags/parseFlags helper (139 production files) with reject-args and or-help variants - the single most repeated declared name in the tree, described nowhere | `ParseFlags` |
| c | named-code | latent-literal | **provider overload status 529** - the non-standard HTTP 529 the Anthropic API returns when overloaded, load-bearing in stream-retry and account-rotation paths - internal/agent/retry.go names it (const statusOverloaded = 529) but ~8 other production files (gateway, attemptbudget, agent siblings) still use the bare literal, and no doc mentions either form | `statusOverloaded` |
| c | named-code | code-only | **tilde expansion helper** - the internal/pathutil helper that expands a leading ~ in user-supplied paths (54 production files) - the reason ~/foo works in every fak flag, a behavior users rely on that no doc promises | `ExpandTilde` |
| c | named-code | code-only | **weight-bearing engine** - the engine-interface predicate saying whether an engine actually carries model weights (real inference: DynamoEngine true) or is a stub/proxy (readEngine, localEngine false) - a load-bearing trust distinction that appears in 13 production files and zero docs | `WeightBearing` |
| ~ | hinted | doc-only | **ground truth** - the section (29 docs) pointing at the artifact/command a reader can run to verify a claim themselves - fak's evidence-over-assertion convention, unnamed and undefined | `GroundTruth` |
| ~ | hinted | doc-only | **honest fence** - the repo's core rhetorical device - an explicit statement of what a claim does NOT cover, placed next to the claim so it cannot be overread - used as a heading in 74+ docs under four drifting spellings (honest fence / honesty fence / honest boundary / honesty gate) with no definition anywhere and no code symbol | `HonestFence` |
| ~ | hinted | doc-only | **honest scope** - the recurring section (54 docs) that states what a feature deliberately does not attempt - the scope-declaration counterpart of the honest fence (which guards a specific claim) - never defined as a term | `HonestScope` |
| ~ | hinted | doc-only | **honesty ledger** - the recurring doc section (31 docs) recording which claims are proven vs open for a feature - readers meet it constantly, but which ledger file it denotes (if any) is never stated and no code writes a thing by this name | `HonestyLedger` |
| ~ | hinted | doc-only | **hot path** - used in 45 docs for the latency-critical code route (decode loop, gateway serve path) - borrowed jargon everyone assumes, but WHICH paths are the hot paths is never enumerated, so the term cannot be checked | `HotPath` |
| ~ | hinted | latent-literal | **in-repo deterministic LCG** - the textbook linear-congruential PRNG (state*1103515245 + 12345, masked to 31 bits) re-implemented inline in 14+ files (bench, benchids, cmdutil, agent) for deterministic jitter/ids - the multiplier and increment are never named, so the shared PRNG convention is invisible | `lcgMultiplier` |
| ~ | hinted | doc-only | **promotion gate** - the criteria a claim/feature must meet to move up a maturity rung - a heading in 22 docs (plus 'Promotion rule' in 10) with no code symbol and no definition, despite gating being fak's favorite mechanism | `PromotionGate` |
| ~ | hinted | latent-literal | **the conflated 300 literal** - a bare 300 on non-const lines in ~40 production files that conflates at least four distinct concepts: 300ms settle sleeps, the HTTP redirect boundary (StatusCode >= 300), a compaction-shed token budget, and sweep sizes - none of them named | `settleSleepMillis` |
| ~ | hinted | doc-only | **trust boundary** - the line between what the kernel verifies and what it merely receives from an untrusted agent - a heading in 33 docs and the load-bearing idea of the whole system, with no code identifier and no glossary definition | `TrustBoundary` |
| ~ | hinted | hinted | **unmeasured-containment residual** - the standing residual that fak's privacy/prompt-injection containment is a design posture, not a measured leakage number - repeated across research triages (MIRROR #1007) as an italicized hedge, never as a named, trackable residual | `ContainmentUnmeasured` |
| ~ | hinted | doc-only | **verdict ladder** - the ordered best-to-worst verdict scale every scorecard defines for its rows (crystal..undocumented, explicit..latent) - the shared pattern is used by 8+ docs as a heading but is named nowhere as the reusable concept it is | `VerdictLadder` |

## Per-KPI (implicitness-debt = naming hygiene of the rows that exist)

| Group | KPI | Score | Debt | Detail |
|---|---|---:|:--:|---|
| well-formed | `well_formed` | 100 | 0 | all 20 rows well-formed |
| grounded | `evidenced` | 100 | 0 | every row's evidence appears in the tree |
| named | `named_resolves` | 100 | 0 | every claimed name resolves in the code |
| named | `anchored` | 100 | 0 | every explicit concept's definition is anchored on disk |
| named | `naming_planned` | 100 | 0 | every implicit concept has a naming plan |
| honesty | `explicitness_consistent` | 100 | 0 | every verdict matches its evidence |
| honesty | `name_quality_soft` | 100 | 0 | proposed names are name-shaped |

## Coverage by signal kind (how much of each implicit space is positioned)

| Signal | Positioned | Discovered | Unpositioned |
|---|---:|---:|---:|
| doc-only | 13 | 138 | 125 |
| code-only | 11 | 49 | 38 |
| latent-literal | 4 | 37 | 33 |
| hinted | 1 | 1 | 0 |

