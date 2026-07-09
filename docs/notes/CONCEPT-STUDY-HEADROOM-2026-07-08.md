# Field-borrow study: headroom → fak (2026-07-08)

A `field-borrow` pass: study one external repo, witness each candidate technique
against what fak already ships, and file **only** the real gaps as small,
dispatchable leaves. This note is the durable trail — source pin, per-candidate
verdict with evidence, and what was filed vs. dropped and why.

## Source (pinned, falsifiable)

- **Repo:** [headroomlabs-ai/headroom](https://github.com/headroomlabs-ai/headroom) — an LLM gateway/proxy + library ("60–95% fewer tokens for JSON, 15–20% for coding agents · proxy · MCP · content-aware compressors · local-first · reversible").
- **License:** Apache-2.0 (`NOTICE`: "Copyright 2025 Headroom Contributors").
- **Pinned commit:** `38074888ac871b8b44418066d66b6a37159978ed` (2026-07-08, "fix(docker): report source build version (#1862)"), cloned to scratch and read at that SHA.
- **Borrow route:** **INSPIRE** — clean-room reimplement in Go idiom (source is Python). No bytes copied; techniques cited at `path:line@38074888`. Apache-2.0 permits copying too, but the seams differ enough that a Go rewrite is cleaner than a port.

## What was read

`proxy/output_shaper.py` (turn classification, effort routing, verbosity steering), `proxy/handlers/anthropic.py` (wire-up), `proxy/output_savings.py` (the honest "we can't directly observe output savings" caveat), plus the cache/compaction and memory-ranker surfaces for the witness comparisons.

## Candidates & verdicts

| # | Technique (source) | Witness vs. fak | Verdict | Disposition |
|---|---|---|---|---|
| C1 | Lower reasoning **effort** on mechanical tool-result continuations — `classify_turn`/`route_effort` (`output_shaper.py:209,316`) | `Grep` of `internal/gateway/{messages,http,completions}.go` for `Effort\|Thinking\|output_config\|budget_tokens` → **nothing**; passthrough forwards effort verbatim. `fak_feature_query` returned no on-point card. | **ABSENT** | **Filed → #3307** (parent epic **#745**) |
| C2 | **Cache-miss attribution** to a closed cause set — `classify_cache_miss` (hit/ttl_expiry/prefix_change/unknown/cold_start) | `internal/compactcohere/compactcohere.go:127` `Classify` already attributes each turn to `stable`/`fak_cut`/`fak_world_break`/`harness_rewrite`/`cold_ttl`, driven by **both** the prospective idle-vs-TTL signal (`coldByIdle`) **and** the confirmed observed-counters signal (`coldByCounters`: `cache_creation>0 && cache_read==0`). A strict **superset** — folds headroom's "unknown/early-evicted" into `cold_ttl`. | **PRESENT (better)** | **Dropped** — filing would duplicate `compactcohere`. |
| C3 | Normalize `cache_control` breakpoints to **≤4** to avoid a provider 400 | fak's offensive placement splices **≤1** breakpoint and **only when the caller sent none** (`already_set → identity`, `internal/gateway/debug_stats.go:163`); compaction is subtractive/byte-identical, never overlay-replay, so it can't accumulate >4. Capping a client's own breakpoints would violate passthrough fidelity. | **ABSENT but out-of-scope** | **Dropped** — not a fak-caused failure mode; a client cap changes semantics. |
| C4 | Cache-safe **verbosity/terseness steering** appended after the last system block — `apply_verbosity_steering` (`output_shaper.py:280`) | `internal/syspromptmmu/overlay.go` already ships the after-breakpoint overlay **mechanism** (cache-safe append, `doc.go:15`), filled by capability *query*; the leveled terseness **content** is absent. | **PARTIAL** | **Filed → #3308** (parent epic **#1258**) |

## Filed

- **#3307** — `feat(gateway)`: lower reasoning effort on mechanical tool-result continuations (guard/serve passthrough). Output-token cost lever; the sibling of `compactcohere`'s cache-prefix sensor. First rung = a pure content-free turn classifier + tests; actuator (lower-only, non-injecting, never touches `thinking.type`, shadow-first, opt-in) is the follow-on.
- **#3308** — `feat(syspromptmmu)`: cache-safe verbosity/terseness steering as an after-breakpoint overlay segment. Rides the existing `overlay.go` seam. First rung = a `steeringSegment(level)` producer + cache-safety/idempotence tests. Honest fence: output-token cost/UX lever, opt-in, no token-savings claim without a holdout (`output_savings.py:4` says the same).

## Router caveat

`tools/issue_lane_router.py --issues` routed both #3307 and #3308 to the **`docs`** lane — a false positive: it path-confirmed on the `docs/notes/…` + `INDEX.md` strings cited in the issue bodies, not the actual code lanes (`internal/gateway`, `internal/syspromptmmu`). Lane labels were **not** applied; dispatch/triage should route these to their code lanes. The topical labels + parent-epic links on each issue already point a human correctly.

## Takeaway

4 candidates → 2 filed, 2 dropped. The witness step earned its keep: C2 looked like a clean borrow but `compactcohere` already implements a superset, and C3 dissolved once fak's ≤1-breakpoint placement rule was checked. The two survivors are both **output-token** levers (the axis fak's cache work doesn't touch) that ride existing seams rather than adding new ones.

---

# Second pass: the input-side cache/context token-economics axis (2026-07-08)

A second `/study-repo` run over the **same** pinned commit (`38074888`), reading a
different subtree — `headroom/transforms/*` (the content-aware compressor pipeline),
`headroom/ccr/*` (the reversible Compress-Cache-Retrieve store), and
`crates/headroom-core/src/signals/*` (line-importance signals). The first pass took the
**output-token** levers; this one takes the **input-side / cached-prefix** axis, which is
exactly fak's gateway cache-economics territory — so every candidate had to survive a
witness against fak's *own* elision / compaction / dedup / breakpoint machinery. Same
license verdict: **INSPIRE** (source is Python + Rust; target is Go — no bytes copied,
techniques cited at `path:line@38074888`).

## What was read (pass 2)

`transforms/read_lifecycle.py`, `transforms/read_maturation.py`,
`transforms/cross_turn_dedup.py`, `transforms/cache_aligner.py`,
`transforms/smart_crusher.py`, `transforms/code_compressor.py`,
`transforms/content_router.py` (the routing spine), plus `learn/*` and the Rust
`signals/line_importance.rs` for context. Each candidate was witnessed against fak by a
focused code-reading pass (five parallel witnesses) **and** dogfooded via
`fak_feature_query`, guarded with raw `Grep`.

## Candidates & verdicts (pass 2)

| # | Technique (source `path:line@38074888`) | Witness vs. fak | Verdict | Disposition |
|---|---|---|---|---|
| C5 | **Read-lifecycle**: mark a Read STALE (file Edited after) / SUPERSEDED (covered by later Read) → reversible marker (`transforms/read_lifecycle.py:315,464`) | The rule already exists **offline** in fak's own `tools/ctxwin.py:341` and is **measured at 13.37% stale Read bytes** (`docs/notes/CTXWIN-CONTEXT-WINDOW-BASELINE-2026-06-23.md`), but is **never wired live** (`internal/pythongate/baseline.go` only lists it). Live retention keys on age/byte-size, not edit-after-read causality. Reversible CAS store already ships (`ctxmmu`/`ctxrestore`). | **PARTIAL** (live-ABSENT) | **Filed → #3339** (epic **#2393**) |
| C6 | **Cross-turn verbatim-span dedup**, prefix-monotonic, earliest-preserved (`transforms/cross_turn_dedup.py:151,216`) | `ctxplan` dedups only byte-identical **whole cells** (`plan.go:552`), keeps the **pinned** rep (`plan.go:598`, violating earliest-preserved), and `maybePlanMessages` **bails on tool-use turns** (`gateway.go:1937`) so the coding stream is never deduped. `anthropic_elide.go` shrinks a single result, no cross-block compare. | **PARTIAL** (core ABSENT) | **Filed → #3340** (epic **#2503**) |
| C7 | **Volatile-token classifier + operator warning** — name UUID/ISO/JWT/hash busting the prefix (`transforms/cache_aligner.py:194,155`) | `anthropic_cachebp.go:480` detects only UUID+ISO via **regex** as a **bare bool** used to *reorder* blocks; `sysprompt_fingerprint.go:76` hashes for drift. No JWT/hex-hash, no named class, no user-facing warning (only a `volatile_head` metric). | **PARTIAL** | **Filed → #3341** (epic **#2783**) |
| C8 | **Activity-based Read quiesce** maturation — mature after N quiet turns, not by fixed position (`transforms/read_maturation.py:105,322`) | fak matures by **fixed position** (`anthropic_elide.go:41` `elideRecentKeepMsgs=4`); the only quiet-turn counter is `ctxplan/refcount.go:126` `falseRetainK=3`, **advisory** and unwired to cache-writing. fak's stance is the inverse (keep recent warm). | **ABSENT** | **Filed shadow-first → #3342** (epic **#2393**) |
| C9 | **Budget-gated lossy TOON row-drop** with reversible CAS retrieval (`transforms/smart_crusher.py`) | `internal/toon` already does the columnar header+rows factoring (`toon.go:113`) but **lossless only** (`decide.go:239`, `roundTrips`), wired to fak's own MCP results only, **default-off**. No budget row-drop, no hash retrieval. | **PARTIAL** | **Filed → #3343** (epic **#3064**) |
| C10 | **AST-aware code-body compression** — tree-sitter, preserve imports/signatures/types, guaranteed-valid output (`transforms/code_compressor.py`; ref LongCodeZip) | No tree-sitter dependency in `go.mod`; code tool-results get only content-blind head+tail elision. Genuinely absent — but a **large new subsystem** (new dependency + multi-language), not a ship-alone leaf. | **ABSENT** | **Deferred** — recorded here, not filed: the tree-sitter-dependency decision is its own call. Sibling of the existing Compressor seam (#3204). |

## Filed (pass 2)

- **#3339** — `feat(agent)`: wire the stale/superseded Read-lifecycle classification to the live elision seam, reversible via the existing CAS store. First rung = STALE (edited-after) in Go with a cache-prefix-identity test; partial-range **coverage** superset logic is a named follow-on.
- **#3340** — `feat(agent)`: cross-turn verbatim-span dedup on the tool-output stream, with the `is_prefix_monotonic` cache-safety invariant as its witness.
- **#3341** — `feat(cachemeta)`: name the volatile token classes (add JWT + hex-hash) and surface an operator warning; read-only diagnostic, does not change breakpoint placement.
- **#3342** — `feat(ctxplan)`: **shadow-measure** an activity-based per-file quiesce clock vs. the fixed-window rule — forwarded bytes byte-identical, measurement only; the live cache-hold flip is the gated follow-on.
- **#3343** — `feat(toon)`: budget-gated lossy row-drop with CAS retrieval, keeping the lossless mode intact and under the epic's auto-fire gate.

## Takeaway (pass 2)

6 candidates → 5 filed, 1 deferred, 0 dropped-as-present. The strongest borrow (C5) is one
fak **already validated on its own traffic** (13.37% stale) but left stranded in an offline
Python reducer — the witness turned "borrow an idea" into "wire a proven-on-us signal to the
live seam with a store that already exists." C10 is the honest deferral: a real absence whose
cost (a tree-sitter dependency + a multi-language subsystem) makes it a decision, not a leaf.
Together with pass 1's output-token levers, headroom's whole compression stack is now mapped
onto fak: **output-side** (#3307/#3308) and **input/cache-side** (#3339–#3343), each grounded
at a fak seam.

> **Companions:** [`/study-repo`](../../.claude/skills/study-repo/SKILL.md) (this pass) →
> [`/field-borrow`](../../.claude/skills/field-borrow/SKILL.md) (the witness+file back-half).
> Parent epics: [#2393](https://github.com/anthony-chaudhary/fak/issues/2393) (ctxplan),
> [#2503](https://github.com/anthony-chaudhary/fak/issues/2503) (dedup),
> [#2783](https://github.com/anthony-chaudhary/fak/issues/2783) (cache-value),
> [#3064](https://github.com/anthony-chaudhary/fak/issues/3064) (toon).
