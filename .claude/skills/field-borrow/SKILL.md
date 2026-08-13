---
name: field-borrow
description: High-priority default "inspired by" workflow. Invoke proactively whenever an external product, repository, paper, standard, benchmark, release, issue, PR, roadmap, design discussion, or field practice could improve fak—not only on explicit borrow requests. Mine code, tests, docs, history, releases, open/closed issues, PRs, discussions, roadmaps, and provenance; date observations and source events; directly port/adapt when exact-source licensing permits; explore the spirit of promising proposed or incomplete ideas; then dogfood fak self-query to classify PRESENT/PARTIAL/ABSENT before shipping or filing witnessed gaps. Assign each witnessed capability DEFAULT, OPTIONAL-MODULE, RECIPE, WATCH, or EXCLUDE so fak combines strong industry defaults with bounded, supportable user coverage.
---

# field-borrow — borrow from the field, but witness the gap against yourself first

## High-priority inspired-by default

This is the repository's high-priority default "inspired by" workflow. Invoke it proactively
whenever a named product, repository, paper, standard, benchmark, release, issue, PR, roadmap,
design discussion, or field practice could improve fak. Do not wait for an explicit research or
borrowing request: run it before bespoke design when credible prior art may change the choice.
External evidence does not override user scope, security, legal obligations, or coherent shipping.

Six laws govern every pass:

1. **Priority:** consider external prior art early, not as optional cleanup after designing.
2. **Licensed reuse:** copy or adapt directly when exact-source licensing and technical fit permit;
   fresh reimplementation is not inherently better.
3. **Exhaustiveness:** all relevant source classes were checked or explicitly marked unavailable.
4. **Dating:** every observation carries event time, observation time, source state, and refresh cue.
5. **Breadth:** mine implementation, docs, tests, history, releases, issues, PRs, discussions,
   roadmaps, and provenance—not only README claims.
6. **Spirit:** explore transferable principles and responsible extensions even when upstream has
   only a partial prototype; label inference separately from shipped fact.

## Exhaustive, dated source ledger

For a repository-shaped source inspect canonical docs/design/API examples; code, tests, fixtures,
and configuration at a pinned revision; releases/tags/changelog and relevant history/blame; open
**and closed** issues; merged, closed, and open PRs plus review discussion; discussions, RFCs,
ADRs, roadmaps, and TODOs; exact-revision LICENSE, per-file headers, NOTICE, vendored/generated
provenance, submodules, and contribution terms. For papers/products/standards use equivalent
primary sources (artifact/appendix, official release history, normative revision/errata). Follow
citations to the mechanism's origin. Inaccessible surfaces are omissions, not empty results.

Record every material observation with:

- `observed_at`: full ISO date/time for the research observation;
- `source_event_at`: commit, merge, release, publication, or update date;
- source state: released/shipped, merged-unreleased, proposed/open, rejected/closed,
  experimental, or unclear;
- immutable anchor: `path:line@full-sha`, release/tag+digest, paper version/DOI, or standards rev;
- platform context: relevant version, API generation, provider, hardware, plan/tier, or environment;
- refresh trigger: release, PR merge, API rollover, benchmark rerun, or `none known`.

Never write timeless “X does Y” for a changing platform. Open issues, PRs, and roadmaps prove
interest/direction, not shipped behavior.

## License disposition: copy when allowed

Read the license at the exact studied revision and check per-file/NOTICE/provenance boundaries.
Classify every candidate before implementation:

- **DIRECT-PORT:** copying/modification is permitted and obligations fit. Prefer copying the
  smallest coherent implementation or test, preserving notices and source `path@sha`, then adapt.
- **ADAPT:** reuse is permitted but fak interfaces, dependencies, or constraints require material
  changes. Identify direct versus rewritten portions and retain attribution.
- **INSPIRE-ONLY:** terms are absent, unclear, incompatible, proprietary, or behavior-only. Build
  independently; do not copy expressive code, tests, comments, or assets.
- **DO-NOT-USE:** provenance or terms are unsafe; exclude and record why.

Do **not** default to inspire-only because clean-room rewriting feels cleaner. Public visibility
also does not grant permission. When obligations are ambiguous, use INSPIRE-ONLY or seek review.

## Expansive candidate extraction

Extract all of these before ranking:

- direct mechanisms: algorithms, schemas, tests, UX, operations, and documentation;
- negative knowledge: reverts, rejected proposals, recurring issue failures, regressions, reviews;
- emerging direction: open PRs, issues, RFCs, roadmaps, TODOs, and design sketches, marked proposed;
- **Spirit extensions:** infer the broader principle and adjacent fak applications, using
  `source fact -> inferred principle -> fak opportunity -> disconfirming check`;
- combinations across independent sources that produce a stronger design.

Be expansive in discovery and conservative in claims. A promising unimplemented idea is a valid
**INFERENCE** when paired with a cheap experiment/falsifier; it is not parity evidence.

## Bounded-superset portfolio rule

The target is not "copy the winner" and not "accumulate every feature." It is a
**bounded capability superset**: within fak's declared scope and support budget, users should
be able to trust that fak carries the strongest reasonable industry default **and** preserves
materially useful alternatives for cohorts whose constraints differ. fak is egoless about
where those capabilities originated. Its modular leaf/adapter/integration boundaries are an
advantage: a useful alternative need not displace the default or contaminate the kernel core.

For every witnessed capability, record one portfolio disposition:

| Disposition | Use when | Required form |
|---|---|---|
| `DEFAULT` | strongest net-true choice for the common supported case | integrated, documented default with a witness against the real alternative |
| `OPTIONAL-MODULE` | useful for a named user/job/constraint but not the broad default | isolated leaf, adapter, policy, backend, or integration; explicit opt-in and support owner |
| `RECIPE` | useful and supportable, but too narrow or environment-specific to ship as maintained code | tested composition/integration guide with prerequisites and dated witness |
| `WATCH` | promising, moment-in-time, immature, or evidence is not strong enough to support yet | dated source + review trigger; no shipped claim |
| `EXCLUDE` | outside scope, unsafe, license-incompatible, operationally unaffordable, or genuinely dominated | named reason and the evidence/disconfirming condition that could reopen it |

`DEFAULT` is not the only successful outcome. Do not reject a capability merely because a
different technique wins overall, is newer, or is already fak's default. Ask whether a real
supported cohort still benefits under a constraint such as hardware, provider, latency,
quality, privacy, deployment topology, compatibility, cost, or operator preference. If yes,
prefer an `OPTIONAL-MODULE` or `RECIPE` seam over deletion. Conversely, age, popularity, or
"someone might want it" is not enough: an obviously superseded approach with no surviving
cohort goes to `EXCLUDE`, and a fast-moving snapshot goes to `WATCH` until revalidated.

Each non-excluded row must name: capability axis; user/job/constraint served; observed evidence
and date; relationship to the default; proposed modular seam; incremental build, test,
security, documentation, and maintenance cost; owner or review trigger; and a witness that
would disconfirm usefulness. This is the **reasonable-coverage budget**. Once marginal support
cost exceeds witnessed user value, stop expanding and register the omission instead of **boiling the ocean**.

A study is coverage-complete only when it reports both:

1. **Default frontier** — where fak's common-case default should lead or change.
2. **Coverage frontier** — useful supported cohorts still absent even though the default may
   already be best overall.

## Required completion evidence

In addition to the PRESENT/PARTIAL/ABSENT self-query pass below, completion requires a coverage
checklist, dated source ledger, license disposition, candidate matrix (fact versus inference and
shipped versus proposed), exact fak seam, and either shipped proof or contract-ready filed work.
Prefer shipping a safe in-scope minimal spine now over reflexively creating backlog. A long link
list is not exhaustive research: state what each source changed in the conclusion.


## Why this skill exists

fak researches the outward field well — the `docs/notes/RESEARCH-*` corpus, `industry-score`,
`idea-scout`, `sota-check` all point *outward*. But applying a field idea inward has a silent
failure mode: an agent reads "Cognee does GraphRAG" or "Zep has bi-temporal memory" and files
"fak should add graph memory" **without first asking fak whether it already has it** — so the
tracker accretes gaps fak already closed, or gaps described one abstraction away from the real
seam. The research was outward-complete but the *witness step* was manual and skipped.

The law, in one line: **the default answer to "does fak already have this field-standard
capability?" is _ask fak, don't assume_ — dogfood the self-query surface to witness
present / partial / absent, then file only the witnessed gaps, each grounded in a seam.**

This is the loop that produced #3161 / #3162 / #3163 (edge-aware query, retrieval-eval,
freshness rung) under epic #1494: three retrieval-*quality* gaps borrowed from Cognee/Zep,
LongMemEval/LOCOMO, and Zep/Mem0 — each witnessed by querying `fak_feature_query` against
fak's own `internal/selfquery` before filing, each grounded at `internal/selfquery/selfquery.go`
(the flat lexical ranker at `:566`). See
[`docs/notes/CONCEPT-FIELD-BORROW-QUERY-QUALITY-2026-07-08.md`](../../../docs/notes/CONCEPT-FIELD-BORROW-QUERY-QUALITY-2026-07-08.md)
for that worked instance.

### How this differs from its siblings (do not duplicate them)

| Skill / tool | Direction | Mechanism | Files issues? | Dogfoods the self-query witness? |
|---|---|---|---|---|
| `sota-check` | inward, **kernels only** | `fak sota <op>` prior-art matrix | no | no |
| `industry-score` | outward | scorecard: coverage + parity-debt over a taxonomy | no | no |
| `idea-scout` (`tools/idea_scout.py`) | outward, **automated** | arXiv/GitHub scan, dedup vs **existing issues** (seen-cache · body-stamp · title-Jaccard) | yes (triage queue) | **no** (issue-vs-issue only, never capability-vs-codebase) |
| **field-borrow** (this) | outward→inward | **dogfood `fak_feature_query` to witness the gap**, then file epic-anchored | yes (grounded, under the right epic) | **yes — this is the point** |

field-borrow composes with all three: it can be the human-in-the-loop pass over an
`idea-scout` hit, it feeds witnessed coverage rows to `industry-score`, and once a spine
ships it hands off to `spine-fanout`.

## The pass

### 1 — Name the source and the capability axis (dated)

Pick ONE capability from a **named** external system or a dated source: an agent-memory
product (Letta/MemGPT, Mem0, Zep/Graphiti, Cognee, Supermemory), a serving system
(vLLM/SGLang/TensorRT-LLM), a competitor feature, a benchmark (LongMemEval, LOCOMO), or a
row from the `RESEARCH-*` / industry field scans. Write it down with the source, so the
borrow is falsifiable: *"Zep/Graphiti — bi-temporal edges: facts carry validity windows;
a newer fact invalidates the old one."* One capability per pass; do not batch.

### 2 — Dogfood the self-query WITNESS (the heart of this skill)

Ask fak, three ways, whether it already has the capability — do **not** reason about it
from memory:

- **`fak_feature_query`** (MCP) / `fak capabilities <intent>` — intent → ranked capability
  cards (name, summary, `witness` content-address, the call to make).
- **`fak index docs|leaves|verbs|claims <terms>`** — the dev self-index over docs, packages,
  CLI verbs, and the CLAIMS ledger.
- Also a plain `Grep` for the concept's obvious symbol, to catch a capability the lexical
  ranker missed (see Honest limits).

Classify the result, and **paste the top cards as the witness**:

- **PRESENT** — a card / verb / leaf directly does it. *Read it and confirm* it truly does
  (not just a name collision). PRESENT is a valid, complete, **good** outcome — "fak already
  has this, here is the card" ends the pass; do not file.
- **PARTIAL** — an adjacent surface exists but not this capability (e.g. `fak_index_freshness`
  covers index↔tree drift but not semantic supersession).
- **ABSENT** — the query surfaces nothing on-point. An ABSENT claim **must** show the query
  that returned nothing on-point — that null result is the falsifiable witness.

### 3 — Ground the PARTIAL/ABSENT in a seam (file:line)

The capability cards point at the code. Find the **exact file:line** the capability would
touch and quote **one** dogfood result as the in-issue evidence — e.g. the flat lexical
ranker `score()` at `internal/selfquery/selfquery.go:566`, or the `FeatureCard` struct at
`:37` that carries no freshness/edge field. **No file:line seam, no issue** — a gap you
cannot locate is not ready to file; keep witnessing until you can point at the code.

### 4 — Dedup, then file grounded (under the right epic)

Search existing issues **and** the natural parent epic before filing (avoid re-filing what
`idea-scout` or a human already opened). File under the epic with, in order:

1. the pain, in fak's terms, quoting the seam from step 3;
2. the **named external inspiration** (dated — a fabricated "system X does Y" is the exact
   citation failure `dos_citation_resolve` exists to catch);
3. the **dogfood witness** from step 2 (the query + what it returned);
4. a **concrete first checkable step** (the smallest thing that proves the gap is real and
   the fix is buildable);
5. **honest fences** (deterministic / advisory / read-only where they apply).

Set milestone + labels at creation. On this host, run `gh` via PowerShell or the Bash tool
deliberately (quote bodies through `--body-file`, not inline heredocs that the shell mangles).

### 5 — Register the trail (or it lives only in a transcript)

Record the instance in a dated `docs/notes/CONCEPT-<topic>-<date>.md`: source → witness →
seam → filed issues, with a `Companions:` block. Add its **`INDEX.md`** line in the
*Notes & research* section (an unindexed dated note is an orphan `fak index freshness`
flags). Link the children from the parent epic. **This registration — not the issues alone —
is what makes the research path durable.**

### 6 — (optional) Feed the map

A witnessed PRESENT/PARTIAL/ABSENT is a coverage row `industry-score` can absorb; a genuinely
novel, spine-shaped gap can seed `spine-fanout` once a working spine ships.

## What "done" proves

- Every filed gap carries a **dogfood witness** (the query + its top cards), a **file:line
  seam**, a **named dated external source**, and a **first checkable step**.
- No gap was filed that fak already ships (step 2 caught the PRESENT cases).
- The instance is registered: a dated `docs/notes` entry, its `INDEX.md` line, and the epic
  links its children. `python tools/... ` freshness / `fak index freshness` reads clean for
  the new note.

## The anti-gaming rule (specific to this surface)

**Never file a capability gap you did not dogfood-witness, and never mark ABSENT without the
null query.** The witness is the falsifiable record — the way `sota-check`'s `Prior-art:`
trailer is. Two failure modes this rule kills:

- **Assumed-absent.** Filing "fak lacks X" from memory when `fak_feature_query` would have
  returned the card. The fix is the query, pasted.
- **Fabricated borrow.** "Zep does bi-temporal invalidation" when it does not, or citing a
  system/paper you did not read. Name the source, dated; if you could find no prior art, say
  *that* so the next person can disprove it.

PRESENT is not a failed pass — a witnessed "we already have this" is a real result.

## Where this misleads / honest limits

- **The self-query score is lexical.** `rankCards`/`score()`
  (`internal/selfquery/selfquery.go:540,566`) is `strings.Contains` substring matching over
  name/tags/ref/summary — a poorly-phrased query can miss a card fak *does* have, producing a
  **false-ABSENT**. Read the top cards, vary the phrasing, and cross-check with a raw `Grep`
  before trusting an ABSENT. (This limit is itself issue #3162 — the ranker is ungraded.)
- **ABSENT means "fak's own index doesn't surface it", not "it doesn't exist."** The index can
  be stale or the capability can live under a name you didn't guess.
- **The witness is a snapshot.** A card added tomorrow changes the verdict (the freshness
  problem is issue #3163). Re-witness before acting on an old pass.
- **Composes with, does not replace:** `idea-scout` (automated outward feeder), `industry-score`
  (the competitive coverage map), `spine-fanout` (post-spine fan-out), `sota-check` (kernel
  prior-art). If the capability is a **compute kernel**, use `sota-check` instead — this pass
  is for product / agent / memory / serving capabilities.

## When to run this

- Reading a `RESEARCH-*` / SOTA / competitor note and wondering *"do we already have this?"*
- Before filing any "fak should add X" capability-gap issue.
- After a competitor or a paper ships a capability worth borrowing.
- On a `/loop` cadence over the `docs/notes/RESEARCH-*` corpus, to convert inert research
  into witnessed, grounded, epic-anchored backlog — the inward, human-curated complement to
  the daily `idea-scout`.
