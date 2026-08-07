---
title: "Repeated shell tool-result spans: the dedup mechanism decision (2026-07-19)"
description: "Decides the mechanism for #5254 — port the already-shipped cross-turn verbatim-span dedup from the Anthropic byte-splice wire to the decoded []Message wire, because the decoded elider is size-gated and structurally blind to duplication."
---

# Repeated shell tool-result spans: the dedup mechanism decision (2026-07-19)

Status: MECHANISM DECIDED (this note = DoD item 1 of
[#5254](https://github.com/anthony-chaudhary/fak/issues/5254)) and PORTED (item 2,
`internal/agent/message_elide.go`). The dogfood witness (item 3) is **not reachable as
the DoD specifies it** — the corpus-provenance check below measured why, and it is the
one result in this note that changes what #5254 can claim.

**2026-08-06 update.** The `--guarded-only` provenance filter this note named as the
remaining blocker on item 3 now exists (`internal/session/compactaudit_provenance.go`,
`fak session compact-audit --guarded-only`), and the retargeted witness was run:
`internal/session/testdata/compactaudit/guarded-cohort-witness-2026-08-06.md`. Item 3
is still **not** satisfied, but the blocker has moved from a missing capability to a
missing population — only **2** fak-guarded Codex sessions exist since the port landed
(2026-07-27), and they carry **0** compaction fires, so no post-port window is
measurable. See "Retargeted witness" below for what the run did and did not license.

Child of [#4768](https://github.com/anthony-chaudhary/fak/issues/4768)
(attribution parent), grandchild of #4763. Sibling classes stay where they are:
`compaction_summary` with #3071, cache-creation pricing with #2785.

## What the evidence says

`internal/session/testdata/compactaudit/regrowth-witness-2026-07-18.md` decomposed
1.55 GB of transcript rows appended across 1,510 post-compaction windows in native
Codex sessions:

| quantity | value |
|---|---|
| `tool_result/shell_command` bytes | 1,032,156,451 (66.4% of all regrowth), 223,643 rows |
| duplicated inside that class | 12,458,846 B across 1,587 rows |
| windows carrying `REPEATED_TOOL_RESULT` | 296 / 1,510 (19.6%) |
| median tool calls, fast vs slow rebound cohort | 170 vs 172 |

The cohort line is the one that picks the mechanism: fast-rebounding windows run
the *same* tool volume as slow ones. The waste is retention of repeated result
bytes, not excess tool use — so the lever must be **dedup**, never throttling.

## The finding: the mechanism already exists, on one wire only

fak already ships exactly the transform #5254 asks for. `internal/agent/anthropic_elide_crossturn.go`
(#3340) folds a contiguous LINE run in a later `tool_result` that appeared verbatim
in a strictly-earlier one down to a one-line in-band pointer:

```
…[fak dedup: 42 lines identical to output shown earlier, turn 7, lines 118-159]…
```

It is default-ON (`gateway.DefaultElideResultBytes` = 16384, armed from
`cmd/fak/guard.go` and `cmd/fak/serve.go`), and it is cache-safe by construction:
strictly-earlier-only matching plus keep-earliest gives prefix monotonicity
(`dedupBlockLines(blocks[:k]) == dedupBlockLines(blocks)[:k]`), so appending a turn
never mutates an earlier turn's folded bytes.

But it is reachable from **one call site only** —
`internal/gateway/messages_transform.go:123` → `maybeElideAnthropicRaw` →
`agent.ElideAnthropicResultsWithOutcome` → `collectCrossTurnDedupEdits`. That is the
real-Anthropic `/v1/messages` byte-splice passthrough.

## The gap: the decoded wire is size-gated, and a size gate cannot see duplication

Every other wire — chat completions, the in-kernel local-model path, and the OpenAI
**Responses** wire that the Codex CLI speaks (`internal/gateway/responses.go`) —
reaches elision through `maybeElideMessages` (`internal/gateway/gateway.go:2628`,
also `internal/gateway/messages_stream_planner.go:128`) → `agent.ElideMessages`
(`internal/agent/message_elide.go:29`).

`ElideMessages` is head+tail only. Its entire eligibility test is:

```go
if m.Role != "tool" || len(m.Content) <= threshold { continue }
```

There is no cross-block comparison anywhere on that path. Two consequences, and the
second is the whole issue:

1. **No dedup level exists on the decoded wire at all.** A body repeated verbatim
   twenty times is twenty independent head+tail decisions.
2. **The 16 KB threshold is below the class.** #5254's own numbers put the duplicated
   `tool_result/shell_command` rows at 12,458,846 B / 1,587 rows ≈ **7.85 KB per row**
   — comfortably *under* the threshold. Those rows are skipped by the `continue`
   above no matter how many times they recur. A size gate is structurally blind to
   duplication: repetition is a property of the *set*, and the gate only ever looks
   at one member.

That is a sufficient explanation for why this class is 66% on *native Codex*
sessions specifically. Codex is a Responses-API client; the dedup pass that is
default-on for a Claude session is absent on the wire the 66% was measured on.

## Decision

**Port cross-turn verbatim-span dedup to the decoded `[]Message` path** — the
issue's option (c), "shedder-side dedup" — by reusing the already-proven pure core
`dedupBlockLines(blocks [][]string, turns []int)` rather than writing a second
matcher.

Shape (small, because the hard part is already written and witnessed):

- collect tool-role `Content` in wire order, splitting with `splitLinesKeepNL`; index
  **every** message as a possible source, including recent ones;
- run `dedupBlockLines`;
- write back the folded string only for messages strictly inside the existing
  eligible band (`i < len(messages) - elideRecentKeepMsgs`), copy-on-write, and only
  when the fold is genuinely shorter — the same fail-safe posture `ElideMessages`
  already takes;
- trigger on `crossTurnMinDupLines` (8 lines) / `crossTurnMinDupBytes` (240 bytes),
  **not** on the 16 KB size threshold. Size independence is the point of the change.

### Rejected alternatives

- **Hash-referenced bodies in a side store.** The DoD forbids persisting tool-output
  bodies in receipts, and a hash the model cannot resolve in-band is a deletion
  wearing a pointer's clothes. The existing in-band `turn N, lines A-B` pointer
  already gives reference semantics with zero persistence — the bytes stay reachable
  at their earliest occurrence, inside the same window.
- **Bounded re-emission** (cap how often a result may re-enter). It cannot separate a
  genuinely-needed re-read from a wasteful repeat, so it violates the issue's own
  "does not touch useful work" constraint. Dedup is lossless-by-relocation; an
  emission cap is lossy.

## The cache argument does NOT transfer for free

On the Anthropic wire, safety rests on byte-splicing after a `cache_control`
breakpoint so the protected prefix stays byte-identical. The decoded wire has no
breakpoint to anchor on; `message_elide.go` instead argues that the shrink is
*deterministic*, so the rebuilt prefix is byte-stable turn over turn and a local
backend's radix prefix cache keeps hitting.

Cross-turn dedup makes message *i*'s rendering depend on messages `< i`, which is a
stronger claim than head+tail's per-message determinism. It still holds — that is
exactly prefix monotonicity, so appending turn N+1 cannot change message *i* — but on
the decoded path it must be **asserted as its own test**, not inherited from the
Anthropic witness. This is the single review-worthy risk in the port.

## Corpus provenance: MEASURED, and the assumption FIRES

The note's own precondition — *confirm the corpus crossed fak's wire before writing
code* — was run. It does not hold.

The naive check is misleading in fak's favour and must not be used: `~/.codex/config.toml`
carries **no** `[model_providers.*]` table and no `base_url`. That is *not* evidence of
non-routing, because `fak guard -- codex` never writes one — it injects the provider via
per-invocation `-c model_providers.fak.*` argv overrides on the child command
(`cmd/fak/guard_codex.go`), which leave no durable config trace. Config absence is
therefore uninformative in both directions.

The actual discriminator is the guard witness ledger `fak guard` writes per session
(`cmd/fak/sessions_codex_loop.go` → `~/.codex/fak-guarded-sessions/<session-id>.json`,
schema `fak.codex_guard_witness.v1`). Intersecting it with the rollout corpus the
2026-07-18 witness mined:

| quantity | value |
|---|---|
| rollouts modified since 2026-06-15 (the measured corpus) | 2,448 |
| distinct fak-guarded session records on the box | 154 |
| guarded sessions **present in that corpus** | **120 (4.9%)** |
| `guarded_at` span of the ledger | 2026-07-14 → 2026-07-18 (5 days) |
| span of the corpus | since 2026-06-15 (~6 weeks) |

**≈95% of the measured corpus never transited fak.** The guard ledger covers only the
last 5 days of a 6-week window. So no gateway-side transform — this port or any other —
could have reduced the great majority of the measured 1,032,156,451 bytes.

### What that does and does not invalidate

It does **not** invalidate the mechanism. The gap is real and independently verified on
the wire fak *does* own: `handleResponses` → `completeServed` → `s.complete` →
`maybeElideMessages` → `ElideMessages`, which was head+tail-only and size-gated. For the
120 guarded sessions, and for every future guarded session, the port is a genuine cut.

It **does** invalidate the DoD's item-3 measurement as written. Re-running
`compact-audit --since <date>` over `~/.codex/sessions` measures a population that is
~95% traffic fak never saw; even a perfect dedup caps at ~5% of the measured bytes, well
inside run-to-run corpus drift. A "materially reduced" reading off that corpus would be
**unfalsifiable, not passing** — it would move for reasons unrelated to the change.

### Retargeted witness (the smallest thing that would actually prove it)

Scope the audit to sessions whose ids appear in the guard witness ledger, and compare
guarded-before vs guarded-after. That needs one new capability `compact-audit` does not
have today: a provenance filter (e.g. `--guarded-only`, joining
`~/.codex/fak-guarded-sessions/*.json`). That filter — not the port — is the remaining
blocker on item 3, and it is a `session`-lane change, not an `agent`-lane one.

**BUILT AND RUN (2026-08-06).** `--guarded-only` / `--guard-witness-dir` landed on
`fak session compact-audit`; the sweep fails closed when the ledger is absent or empty,
because an empty guarded cohort renders identically to "the class is gone". Full numbers:
`internal/session/testdata/compactaudit/guarded-cohort-witness-2026-08-06.md`. Three
results, none of them the reduction the DoD asked for:

1. The mixed-provenance problem is confirmed at scale: **122 of 3,206** rollouts (3.8%)
   join the ledger, stable against this note's 120/2,448 (4.9%) on a smaller slice.
2. The corpus-wide headline reproduces (`REPEATED_TOOL_RESULT` 300/1,587 = 18.90%;
   `tool_result/shell_command` 67.6% of regrowth bytes), so the class is not shrinking
   on the box as a whole.
3. The guarded cohort carries 2.80% (3/107) vs bare 20.07% (297/1,480) — but **154 of
   158** ledger records predate the port, so that gap is *pre-port*, and the guarded
   cohort is unmatched (0/107 windows rebounded; all 107 are censored). Observation-length
   skew and a real fak effect are not separable from this run.

Post-port (`--guarded-only --since 2026-07-28`): **2 guarded rollouts, 0 fires, no
regrowth block**. Item 3 is blocked on accumulating guarded sessions that fire
compaction, not on tooling.

The unit-level witness that DID land with the port, in
`internal/agent/message_elide_crossturn_test.go`: a decoded-path prefix-monotonicity test
(the review-worthy risk above, asserted directly rather than inherited) plus a fixture
asserting a sub-16 KB body repeated N times folds to one verbatim copy and N−1 pointers —
the case today's threshold provably misses.

## Generation classification

**`gen/next`** (`Generation G1 - Next Gen`). Not `gen/now`: the mechanism is not an
architecture bet — it is a port of shipped, default-on code — but the DoD's own
acceptance is a **dogfood / default-exposure proof**, which is precisely what
`docs/generation.md` puts in the G1 stream. Not `gen/second-next`: no simulation,
compatibility policy, or cross-generation dependency is required; the wire ABI is
untouched (the pointer is in-band text inside an existing string field).

- **Promotion evidence** (→ `gen/now`): a `compact-audit --guarded-only` re-run showing
  `REPEATED_TOOL_RESULT` window share and `tool_result/*` `dup_bytes` down on guarded
  sessions created **after** the port, with no accuracy regression. The filter now
  exists; as of 2026-08-06 the run returns 2 guarded rollouts and 0 fires since the
  port, so promotion is blocked on population, not tooling. The concrete bar: enough
  post-2026-07-27 guarded sessions to produce a double-digit count of post-fire windows.
- **Demotion / retirement evidence — PARTIALLY FIRED (2026-08-06).** This note
  pre-registered: "if a future measurement shows guarded sessions carry a materially
  *lower* `REPEATED_TOOL_RESULT` share than bare ones even BEFORE this port, the 66%
  headline is a property of un-guarded Codex". Measured: guarded 2.80% (3/107) vs bare
  20.07% (297/1,480), with 154 of 158 ledger records predating the port. That is the
  demotion branch — but it does **not** complete, because the guarded cohort is
  unmatched (0/107 windows rebounded; median in-window growth 72,127 tokens vs the
  corpus's 82,717 slow / 215,942 fast). Retirement into #4768 requires separating
  observation-length skew from a real fak effect; this run cannot.
- **Invalidating assumption — MEASURED, and it FIRED.** The load-bearing assumption was
  *that the 1,510 measured windows came from Codex sessions whose traffic crossed fak's
  `/v1/responses` wire.* Measured: only **120 of 2,448** corpus sessions (4.9%) appear
  in the guard witness ledger, re-measured 2026-08-06 as **122 of 3,206** (3.8%) on the
  whole corpus. The mechanism survives (the wire gap is real and verified); the DoD's
  item-3 measurement does not.
- **Still-unverified assumption — now the load-bearing one.** That the guarded cohort is
  *representative* of the un-guarded one. The 2026-08-06 run makes the skew visible
  rather than hypothetical: guarded windows never rebound (0/107) and start from a
  median 92,903 pre-fire resident tokens against the corpus's 237,820. Guarded runs are
  shorter and lighter, so any guarded-vs-bare delta under-reads the class by an unknown
  factor. Whoever gets a post-port population must report the cohort shape (rebound
  count, median growth, median pre-fire tokens) beside the dedup delta.

## Routing note

#5254 was dispatched to the `compute` lane. `compute` is `internal/compute/**`
(CUDA / GPU kernels) and has no bearing on this issue. The real file-trees are
`agent` (`internal/agent/message_elide.go`, `anthropic_elide_crossturn.go`),
`gateway` (`internal/gateway/gateway.go`, `responses.go`), and `session`
(the audit that measures it). This note lands in `docs`.
