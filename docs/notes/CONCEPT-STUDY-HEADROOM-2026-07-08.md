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
