# Concept study — rtk (Rust Token Killer: a hook-time tool-output token compressor for coding agents) → witnessed borrows for fak

- **Source:** https://github.com/TaKO8Ki/rtk (a.k.a. Rust Token Killer), pinned `@5d32d0736f686b69d1e8b9dc45c007d4eb77a0a2` (2026-07-09), **Apache-2.0** (read `LICENSE` — permissive, compatible; all borrows below are nonetheless **INSPIRE / clean-room Go**, none vendor bytes).
- **Filed:** **#5011** (tool-aware distiller behind the Compressor seam) · **#5012** (per-tool opportunity lens) · **#5014** (error→correction extractor). Sibling already on the seam: **#3204** (LLMLingua-2 plugin).
- **Method (deep, fanned-out, witnessed):** parallel subsystem readers over the pinned clone — the TOML filter engine + 8-stage pipeline · the streaming/BlockHandler line model · the built-in filter corpus (`src/filters/*.toml`) · `discover` (session opportunity scan) · `learn` (error→correction miner) · the hook **trust/integrity/permissions** subsystem · token-measurement + `never_worse` · the worldview (README non-goals, defaults) — then a completeness critic. Each candidate **witnessed on-axis** against fak (`fak_feature_query`/`fak index` + raw `Grep` + reading the fak seam), classified at the property grain, not the capability name.

## The decisive tension — same job, opposite recovery model

rtk and fak both intercept an oversized tool_result before it reaches the model, but recover lost bytes oppositely. **rtk is lossy-by-default with human-only recovery:** its filters *drop* known-noise lines (300 `--- PASS` → a count) and tee the raw output to a file the **human** recovers with `tail -n +N`; the model never sees the dropped bytes again that turn. **fak is lossless-or-opaque with model recovery:** `internal/headroom/native.go` only does reversible structural transforms (never drops signal), and oversized results page out to an opaque CAS pointer the **model** itself re-fetches via `fak_context_restore`. rtk saves *more* on the happy path (it drops, native can't); fak keeps *reversibility + audit* (nothing is unrecoverable, and recovery is in-loop).

These compose rather than compete, which is the whole borrow: fak can adopt rtk's lossy per-tool distillation **with a stronger safety net than rtk has** — the error-preservation `unless` guard keeps failures inline, and fak's model-callable restore (not rtk's human-only tee) recovers anything the drop was wrong about, mid-turn. The gap is real because fak built the exact seam for it and left it unrouted: `headroom.go:41` documents `Input.Tool` as "a router hint", and **no compressor reads it.**

## Candidate table — FILED borrows (all `inspire`)

| Borrow | Source `@5d32d073` | Axis | fak seam | Witness | Filed |
|---|---|---|---|---|---|
| Tool-shape-aware line distiller: drop known-noise lines for a known tool, keep failures inline (zero round-trip, zero inference) | `src/core/toml_filter.rs` `apply_filter_with_info`; `src/filters/*.toml` `match_output.unless` | lossy per-tool *line*-distillation keyed on tool identity — distinct from #3204's token-level info-theoretic drop and from opaque page-out | `internal/headroom/headroom.go:40-45` (`Input.Tool`/`Kind` "router hint", unrouted); `native.go:33` ignores it | PARTIAL | **#5011** |
| Per-command output-volume **opportunity** scan over real sessions (which tools are the top uncompressed-token sources; which lack a filter) | `src/discover/mod.rs` `run`; `report.rs` | per-tool *counterfactual* savings + filter-coverage gap — not realized savings, not aggregate tokens | `internal/sessionaudit/*` (behavioral + aggregate only; `confusion.go:5`); `internal/headroom/status.go` (realized KPI) | PARTIAL | **#5012** |
| Deterministic **error→correction pair** miner: pair a failed command with its later successful fix, emit "use X not Y (seen N×)" | `src/learn/detector.rs` `find_corrections`; `report.rs` `write_rules_file` | no-model extraction of concrete wrong→right *pairs* — not a friction scalar, not a model-review trigger | `internal/sessionaudit/behavior.go` `RepeatFailures` (error side only); `internal/nightrun/learningnudge.go` (scalar gate) | PARTIAL | **#5014** |

## DIVERGENT ledger — earned dismissals (recorded, not filed)

- **Hook command-rewrite permission model** (`src/hooks/permissions.rs`: Deny>Ask>Allow>Default, per-segment allow, unattestable-construct → never-auto-allow, `rewrite_cmd.rs` Default→exit-3). rtk is a **proxy that rewrites the agent's command** to route through itself, so it needs a static per-command allow/deny lexer with a fail-closed "can't decompose → ask" rung. fak does **not** rewrite commands — it *adjudicates in-loop* (guard/kernel intercepts every call with full context), and that governance surface was already deep-witnessed mature by the openshell study. Their model exists to make a blind textual rewrite safe; fak's users get the same safety from in-loop adjudication that never needed the rewrite. *(One transferable discipline — "every segment of a compound command must independently clear; an unparseable construct defers, never auto-allows" — is worth a later field-borrow against fak's adjudicator, but that is a different subsystem than this compression-focused study and is left unwitnessed rather than asserted.)*
- **Filter-file trust gate** (`src/hooks/trust.rs`: SHA-256-over-exact-bytes, human-review, content-change-invalidation, untrusted→**skipped** not warned, CI-gated env override). This is the right design for *user/project-supplied* transform rules — a repo-committed filter that suppresses a scanner's error output is an output-integrity attack. But fak's #5011 scope is **built-in filters only** (compiled in Go, always trusted, like rtk's own built-ins), which need no trust gate. Recorded as the named prerequisite *inside #5011* for the day user-supplied filters are added — not filed now (respecting "we don't do everything" until the built-in path proves value). Distinct from fak's existing *content* poison-skip, which screens malicious bytes, not malicious rules.

## PRESENT ledger — witnessed "already have it on-axis" (dropped)

- **Provider-measured token accounting + never-worse floor.** rtk estimates tokens as `chars/4` and guards with `never_worse` / `saturating_sub` (`src/core/guard.rs`). fak is *ahead* on this axis: OBSERVED/WITNESSED provider-measured savings via the count_tokens probe (#3349), and `native.go`'s `len(body) >= len(orig)` passthrough is already a never-worse floor plus `skippedNotWorth`. PRESENT — drop.
- **Lossy-drop recovery.** rtk tees raw output to a file for human `tail` recovery. fak's CAS-preserve + model-callable `fak_context_restore` is strictly stronger (in-loop, not human-only). PRESENT/ahead — drop. (The *inline drop-hint* — telling the model "N lines dropped, restore handle X" — is folded into #5011, not separate.)
- **Generic structural compression** (ANSI strip, CR-redraw collapse, dedup, json-min). Exactly `native.go`'s remit. PRESENT — drop.

## Honest limits

- The witness is lexical + a snapshot (fak_feature_query is substring matching; verdicts true as of 2026-07-16). The three PARTIALs were each confirmed by reading the fak seam, not just a ranker miss.
- rtk's clone is `--depth 1`; "they removed X" history was not deepened (not needed — all borrows are from present code).
- License read is good-faith (Apache-2.0 → compatible), but every borrow is INSPIRE clean-room Go regardless, so no vendor decision rides on it.

## Companions

- Skill: [`study-repo`](../../.claude/skills/study-repo/SKILL.md) (this pass) → hands off to [`field-borrow`](../../.claude/skills/field-borrow/SKILL.md) for the per-capability witness.
- Seam sibling already filed: **#3204** (LLMLingua-2 token-level compressor behind the same `Compressor` seam) — #5011 is its rule-based, tool-aware complement.
