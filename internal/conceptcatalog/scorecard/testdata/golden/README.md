---
title: "fak concept-disambiguation scorecard - is every similar-sounding concept crystal-clear"
description: "Inward naming scorecard: each confusable fak concept positioned on the grounded / defined / disambiguated / anchored axes, with one clarity verdict per concept. Two driven numbers: coverage (of the confusable concept space discovered in the tree) and disambiguation-debt."
---

# Concept-disambiguation scorecard - crystal clarity across similar-sounding names

<!-- concept-disambiguation-scorecard: fixture - process: tools/concept_disambiguation_scorecard.py - data: tools/concept_disambiguation_scorecard.data/ -->

The sibling scorecards grade fak's code, docs, and competitive standing. This one asks the question that bites a reader as the system grows: **of the massive, growing set of similar-sounding names (cache, vCache, KV cache, cachemeta, the provider prompt-cache), is each distinct concept crystal-clear - one canonical name, a written definition, and an explicit line drawn against the siblings it is confused with?** Every number below is re-derived by `tools/concept_disambiguation_scorecard.py` and cross-checked against the real tree (the grounding token must appear in the production corpus; the glossary anchor must exist; a `distinct_from` reference must resolve). No verdict is hand-typed.

> Regenerate: `fak concept generate` (the canonical writer preserves classifications while refreshing every generated page).

## Headline

The driver is the UNBOUNDED disambiguation-debt (drive it to 0) plus the positive counters that climb without a ceiling. The old bounded /100 score SATURATES near the top - it hides the real, still-open work - so it is demoted to a labeled legacy line below and is NOT the metric to optimize.

| Metric (primary = unbounded driver) | Value |
|---|---|
| **Disambiguation-debt (drive to 0)** | **63** (clarity 56 + coverage 7) |
| **Crystal-clear concepts (and climbing)** | **5** crystal of 27 positioned |
| **Confusable tokens positioned (covered / discovered)** | **16 / 23** (69.6% of the discovered confusable space) |
| **Undrawn twin-pairs (drive to 0)** | **1** of 10 confusable name-pairs |
| **Ambiguous lookup names (drive to 0)** | **5** of 31 indexed names |
| As of | 2026-11-24 (fak 0.0.0-fixture) |
| Legacy bounded score (saturates; not the driver) | 61.1/100 (grade D) |

> **Read this right.** The metric to optimize is the UNBOUNDED disambiguation-debt (drive it toward 0) and the counters that climb without a ceiling (crystal concepts, confusable tokens positioned). The bounded /100 score SATURATES - once the catalogued namespace is clean it sits near 100 and can no longer tell you how much confusable space is still un-disambiguated - so it is kept only as a labeled legacy line, not the driver.

## Standing at a glance

```text
concept-disambiguation chart - 27 concepts - score 61.1/100 (grade D) - disambiguation-debt 63

clarity ladder (count of concepts, best -> fog):
  * crystal       #########................... 5
  o defined       ############################ 16
  ~ drifting      ##.......................... 1
  = entangled     ####........................ 2
  x colliding     ##.......................... 1
  . undocumented  ##.......................... 1

clarity mix by family (each cell = one concept):
  cache            **oooo.            (7 concept(s); 2 crystal)
  gate             ***oox             (6 concept(s); 3 crystal)
  loop             ooo~               (4 concept(s); 0 crystal)
  nofamily                            (1 concept(s); 0 crystal)
  session          ooooooo==          (9 concept(s); 0 crystal)

coverage by family (positioned / discovered):
  cache            ################............ 7/12
  gate             #####################....... 3/4
  session          ########################.... 6/7
  loop             ............................ 0/0

namespace coverage  [######################..........] 69.6%  (16/23 confusable tokens positioned)

pairwise separation (of the name-pairs a reader cannot keep apart):
  mutual       ##############.............. 5
  one-sided    ###########................. 4
  undrawn      ###......................... 1
  pairs separated   [#############################...] 9/10

name index: 31 lookup name(s) -> 27 concept(s), 5 ambiguous

legend: * crystal   o defined   ~ drifting   = entangled   x colliding   . undocumented
```

## The clarity ladder

| Verdict | Means |
|---|---|
| * crystal | defined + a line drawn against siblings + that line anchored in a doc that exists |
| o defined | defined + a distinction line, but the line is not written in a discoverable doc |
| ~ drifting | defined, but no line drawn against its siblings (you know what it is, not what it is NOT) |
| = entangled | defined and draws SOME line, but not against the twin its own NAME is confusable with - per-row clean, pairwise still fog |
| x colliding | shares a canonical name with another concept - a true ambiguity, fixable only by a rename |
| . undocumented | the catalog gives no definition |

## Separation - is each concept disambiguated FROM THE OTHERS?

Per-concept clarity is not the same question as pairwise separation. A concept in a 250-member family satisfies `disambiguated` by naming **any one** sibling - while the twin its name is genuinely mistakable for goes undrawn. So the scorecard discovers, from the catalog's own names, the pairs a reader cannot keep apart - **permuted** (`witnessPath` / `PathWitness`: the same words in a different order) and **near** (`SessionRef` / `SessionRow`: a couple of edits apart) - and asks whether the line is drawn between *those two specifically*. Boundaries are directed, so for a confusable pair the line must be drawn from **both** sides: `A.distinct_from = [B]` does nothing for a reader who arrived at B. A concept that does not separate from its own twin is `entangled` - clean per row, still fog pairwise.

| Separation metric | Value |
|---|---|
| Confusable name-pairs discovered | 10 |
| **Separated from each other (drive to all)** | **9 / 10** (5 mutual, 4 one-sided) |
| **Undrawn twin-pairs (drive to 0)** | **1** |
| Entangled concepts (own twin undrawn) | 9 |
| Boundaries drawn (mutual / total) | 14 / 23 |
| Dangling `distinct_from` references (drive to 0) | 1 |

## Indexing - can a reader who meets a NAME find the concept?

The catalog is organised by concept; a reader arrives with a **spelling**. [`INDEX.md`](INDEX.md) turns it around: every surface a name can arrive as - the canonical, every alias, the raw grounding token - is a lookup key pointing at the concept it denotes, carrying that concept's contrast set so *which one is this?* and *how is it not the other one?* are answered together. A key that lands on two concepts is an **index** defect the canonical names cannot show: both canonicals may be unique while one row quietly claims the other's name as an alias.

| Index metric | Value |
|---|---|
| Lookup names indexed | 31 over 27 concepts |
| Lookup names landing on several concepts | 5 |
| **Shared names whose concepts stay unseparated (drive to 0)** | **4** |
| Concepts carrying a contrast set | 25 |

## The concepts (best verdict first)

| | Verdict | Kind | Family | Canonical - definition |
|---|---|---|---|---|
| * | crystal | symbol | cache | **KVCache (engine)** - Engine-side paged attention state, keyed by <layer> & position. |
| * | crystal | concept | cache | **PromptCache** - The provider-side prompt cache. |
| * | crystal | symbol | gate | **Decision (kernel)** - The kernel's allow/deny decision. |
| * | crystal | cli-verb | gate | **Decision (scheduler)** - The scheduler's placement decision. |
| * | crystal | cli-verb | gate | **AdmissionGate** - Admits a request. |
| o | defined | config | cache | **CacheTTL5m** - Five-minute cache lifetime. |
| o | defined | config | cache | **CacheTTL1h** - One-hour cache lifetime. |
| o | defined | symbol | cache | **MetaCache** - Metadata about cached pages. |
| o | defined | symbol | cache | **KV Cache Duplicate** - Duplicate id row. |
| o | defined | metric | gate | **RunGate** - Counts run admissions. |
| o | defined | symbol | gate | **GatePolicy** - Policy one. |
| o | defined | concept | loop | **CycleA** - First of a cycle. |
| o | defined | concept | loop | **CycleB** - Second of a cycle. |
| o | defined | concept | loop | **SelfParent** - Points at itself. |
| o | defined | symbol | session | **SessionRef** - A reference to a session. |
| o | defined | symbol | session | **SessionRow** - A stored session row. |
| o | defined | doc-term | session | **SessionÉtat 🙂** - État d'une session — with 'single' and "double" quotes. |
| o | defined | concept | session | **Session'Q (a)** - Apostrophe one | piped. |
| o | defined | concept | session | **Session'Q (b)** - Apostrophe two. |
| o | defined | concept | session | **Q"uo'te (one)** - Both quotes. |
| o | defined | concept | session | **Q"uo'te (two)** - Both quotes again. |
| ~ | drifting | concept | loop | **BadParent** - Dangling parent. |
| = | entangled | symbol | session | **SessionKey** - One session key. |
| = | entangled | symbol | session | **SessionKeys** - Many session keys. |
| x | colliding | symbol | gate | **gate_policy** - Policy two. |
| . | undocumented | subsystem | cache | **Cache Meta (store)** -  |
|   | bogus | weird | nofamily | **Orphan Thing** - None |

## Concept roll-up (the namespace at its abstraction heads)

Most readers cannot hold every concept at once. The optional `parent` forest lets the catalog roll concepts up to the abstraction that HEADS them - and the roll-up is **weakest-link**: an abstraction reads as crystal only when *every* concept beneath it is. A single `defined` leaf keeps the whole head from rolling up to crystal, so the collapsed view can never hide fog it contains. `!` marks a head whose declared verdict reads clearer than its subtree supports.

```text
concept-disambiguation roll-up: 2 top-level abstraction(s), 4 head(s) total, max depth 5 (6 concepts in the forest)

Each abstraction rolls up WEAKEST-LINK: only as crystal-clear as its foggiest
descendant. '!' = the head verdict reads clearer than the subtree supports.

     rolled       head         size debt  mix                  family / canonical
   * crystal     crystal         2    2  **                   gate / Decision (kernel)
   o defined     defined         2    1  oo                   cache / KV Cache Duplicate

abstraction overclaims: none - every head verdict is backed by its whole subtree

(weakest-link: one 'defined' leaf keeps its abstraction from rolling up to crystal - anchor the leaf, or accept the honest lower roll-up.)
```

| | Abstraction | Rolled | Head declares | Subtree | Debt | Weakest descendant |
|---|---|---|---|---:|---:|---|
| * | **Decision (kernel)** (`decisionkernel`) | crystal | crystal | 2 | 2 | admissiongate = crystal |
| o | **KV Cache Duplicate** (`kvcache`) | defined | defined | 2 | 1 | kvcache = defined |

## Per-KPI (disambiguation-debt = clarity of the rows that exist)

| Group | KPI | Score | Debt | Detail |
|---|---|---:|:--:|---|
| well-formed | `well_formed` | 0 | 18 | 18 malformed field(s) |
| honesty | `clarity_consistent` | 0 | 10 | 10 verdict overclaim(s) |
| grounded | `grounded` | 4 | 6 | 6 ungrounded concept(s) |
| distinctness | `disambiguated` | 20 | 4 | 4 undisambiguated confusable concept(s) |
| indexed | `index_resolves` | 28 | 4 | 4 unresolvable shared lookup name(s) |
| separation | `pair_mutual` | 60 | 4 | 4 one-sided boundary on a confusable pair |
| grounded | `alias_resolved` | 64 | 3 | 3 unresolvable alias(es) |
| distinctness | `canonical_unique` | 50 | 2 | 2 canonical-name collision(s) |
| distinctness | `defined` | 68 | 2 | 2 undefined concept(s) |
| separation | `pair_separated` | 80 | 1 | 1 confusable pair(s) with no line drawn |
| separation | `reference_resolves` | 86 | 1 | 1 dangling distinct_from reference(s) |
| grounded | `anchored` | 88 | 1 | 1 missing/dangling anchor(s) |
| honesty | `hierarchy_soft` | 76 | 0 | 4 hierarchy issue(s) |
| honesty | `kind_grounding_soft` | 88 | 0 | 2 kind/grounding mismatch |
| separation | `mutuality_soft` | 100 | 0 | 9/23 boundaries drawn one-way only |

## Coverage by family (how much of each confusable space is positioned)

| Family | Positioned | Discovered | Unpositioned |
|---|---:|---:|---:|
| cache | 7 | 12 | 5 |
| gate | 3 | 4 | 1 |
| session | 6 | 7 | 1 |
| loop | 0 | 0 | 0 |

