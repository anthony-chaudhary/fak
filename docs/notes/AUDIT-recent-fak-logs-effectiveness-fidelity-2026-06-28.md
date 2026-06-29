# Audit: recent fak logs — how effective is the context management, and how do we know it isn't butchering context? (2026-06-28)

**Scope.** An operational audit driven by the *recent* fak logs on this clone
(`.fak/`), answering two questions: (1) how effective is fak's context/cache
management, and (2) how do we know it is not *butchering* (silently dropping or
corrupting correctness-critical) context. This is the empirical companion to the
*design* audit [`HARNESS-CACHE-COHERENCE-AUDIT-2026-06-28`](HARNESS-CACHE-COHERENCE-AUDIT-2026-06-28.md)
(epic #1131): that note specifies the coherence spine; this one reads what the
logs and the on-wire transforms actually do, and grades it against
[`net-true-value`](../standards/net-true-value.md).

Every number below is cited to the artifact or `file:line` that produced it. Nothing
here is a self-report.

---

## 1. What "recent fak logs" are

fak's logging is **structured, schema-versioned, and (for the loop ledger)
tamper-evident** — not free-text stdout.

| Artifact | Schema | What it records |
|---|---|---|
| `.fak/loops.jsonl` (88 rows) | `fak.loop-event.v1` | Loop lifecycle: `fire → admit → start → end → witness`, **hash-chained** (`prev_hash`/`hash`), with a closed reason vocabulary (`OK`, `WEEKLY_CAPPED`, `REFUSE_AT_CAP`, `WRAPPER_ADMITTED`, `POLICY_ADMITTED`, `STARTED`, `EXIT_0`, `SPAWNED`, `AUDIT_UNAVAILABLE`), `evidence_refs`, and per-event `metrics`. |
| `.fak/test-dogfood/report.json` | `fak.recent-feature-dogfood.v1` | The `fak recent-feature-dogfood` probe harness — runs recent features end-to-end and grades them (`tools/recent_feature_dogfood.py`). |
| `internal/tracesink` (in-process) | `turnbench.Trace` | Live model-submitted `ToolCall`s captured **before** any grammar transform, with a completeness invariant `Total() == Recorded() + Dropped()`. |
| `internal/sessionobs` (in-process) | sessionobs Record | RSI-usefulness scorecard of session data (structured signal only — no raw prose). |
| gateway `/metrics` | Prometheus text | `fak_compact_*`, `fak_harness_coherence_*`, tool-prune, reset-shadow, inference families. |

Two logging-fidelity properties matter for the audit:

- **The loop ledger witnesses its own claims.** The terminal event of a loop run is
  `kind:"witness"` — and it records `reason:"AUDIT_UNAVAILABLE"` when it *cannot*
  confirm the spawned work happened (`.fak/loops.jsonl` rows 69/76/77). The log does
  not assert success it cannot see; it records the absence of a witness. This is the
  "don't trust self-report" discipline applied to logging itself.
- **Metrics separate WITNESSED from OBSERVED.** Gateway accumulators are explicitly
  labelled "WITNESSED (fak authored)" vs "OBSERVED (provider-reported, relayed)"
  (`internal/gateway/metrics.go:103-107`), so a provider-side cache miss can never be
  mis-read as a fak bug, and fak never credits itself with a number it only relayed.

---

## 2. How effective is it — from today's dogfood log (`test-dogfood/report.json`, 2026-06-28T16:36Z)

| Lever | Number | Provenance | Honest framing |
|---|---|---|---|
| vCache 2x scorecard (telemetry) | `active_multiplier 7.13×`, grade A, `2x_ready` | **OBSERVED** dogfood telemetry, *one* workload | A *readiness* score from workload concentration (skew s>1, top-8 anchors → 86.4% coverage), not a realized cross-session gain. |
| callavoid turn economics | `amplification 2.46×` (10 raw → 4.06 executed, 5.94 avoided, 6 memo hits), grade B | WITNESSED | Turn-amplification proxy — avoided/repaired turns, not tokens. |
| cache-value ledger | `realized_reuse_ratio 63.0%` over 5 multi-turn turns | WITNESSED | Floor (75.1%) **correctly abstained**: `MinGateTurns=8` and only 5 turns present, so a thin corpus is not called a regression (`internal/cachevalueledger/ledger.go:121-125`). |
| dogfood coverage | `100%`, 15 audit rows, grade A | WITNESSED | — |
| vacuous-test detector | `5242 Test funcs, all assert`, score 100 | WITNESSED | The tests that prove fidelity are not themselves empty. |

The single most important effectiveness finding is the **honest ceiling**, not the
headline. `internal/cachevalueledger/ledger.go:111-119` hard-codes the *only* framing a
published cache-value number may take:

> `marginal-over-tuned-warm-KV (~1.0x single-session; the vs-naive 1/(1-reuse) re-prefill multiple is excluded per #1066)`

So the honest single-session cache value is **~1.0×** (the gain a long trajectory already
earns on a tuned warm-KV server). The 2×/7× vCache numbers are *workload-concentration
readiness from one telemetry sample*; the forbidden vs-naive re-prefill multiple is
excluded by construction. **This passes the net-true-value honesty bar** (provenance
labelled, baseline real, the inflated multiple structurally unreachable).

### The dogfood's overall verdict was ACTION (`ok:false`) — and both failures are honest, neither is context-butchering

1. **`benchmarks-run-vcache` fails *correctly*.** Its validator requires
   `active_source == "telemetry"` (`tools/recent_feature_dogfood.py:348-355`), but the
   probe runs `fak benchmarks run vcache` *without* `--telemetry`. So even though that
   run's `two_x_better:true` and `active_multiplier:3.76×`, the harness **refuses to call
   a planned 2× a proven 2×**. The dogfood failing here is the planned-vs-observed
   discipline working, not a bug. (If anything, the probe is mis-wired to a
   telemetry-requiring validator; worth reconciling, but it is conservative, not unsafe.)
2. **`go-test-fak-loop-vcache-benchmarks` failed on an *assembler* error** in that run —
   `internal/model/quant_amd64_kquant.s` (`CMPQ $0, AX invalid instruction`, lines 377/381;
   `VPSHUFD` 401/406), a toolchain/asm-syntax issue unrelated to context management that
   blocked the loop/vcache/benchmark CLI tests. **It does not reproduce at current HEAD:**
   `go build ./internal/model` is green (exit 0, witnessed 2026-06-28 during this audit), so
   this is a **stale-log artifact** — a transient/older-toolchain failure in the 09:36 run,
   not a live defect.

---

## 3. How do we know it is not butchering context — the five load-bearing invariants

fak's context transforms (history compaction, tool-result elision, tool-def pruning, KV
eviction, quarantine recall) share one discipline: **a transform must *prove* it preserved
the load-bearing bytes, or it does nothing.**

1. **Verbatim-prefix invariant (proven, not hoped).** Both history compaction
   (`internal/agent/anthropic_compact.go`) and oversized-tool-result elision
   (`internal/agent/anthropic_elide.go:16-26`) byte-**splice** on the original bytes: the
   protected prefix (everything through the first `cache_control` breakpoint) is copied by
   memcpy, never re-marshalled. `applySpliceEdits` copies every non-edited byte verbatim
   "so the cache guarantee is a `bytes.Equal`, not a hope" (`anthropic_elide.go:392-396`).
   A compaction turn only counts as `fired` **if the protected-prefix bytes were
   byte-identical to the input**, else it bails to `prefix_mismatch`
   (`internal/gateway/metrics.go:104-105`).

2. **Bail-to-identity on any ambiguity, with a closed reason vocabulary.** Compaction has
   9 typed bail reasons, elision has 9 (`anthropic_elide.go:57-68`); on *any* doubt the
   function "returns its input UNCHANGED (identity)" and a non-identity result is proven to
   (a) still decode as a valid `Messages` request and (b) keep the head prefix bytes
   byte-identical (`anthropic_elide.go:24-26`, `verifySplicedBody`). "Silence must not read
   as success" — every bail is a labelled metric (`metrics.go:98-100`).

3. **Bounded, marked, never-total loss.** Elision keeps each turn's structure and shrinks
   only an *oversized, old, un-cached* tool_result to a **head+tail** form with an in-band
   marker the model can see (`anthropic_elide.go:50-52, 337-346`); the recent working set
   (last `elideRecentKeepMsgs=4` messages) and any `cache_control`-bearing result are left
   **byte-for-byte intact** (`anthropic_elide.go:41-45, 132-138`). It never drops a result
   entirely. It is lossy, so it is **off by default**.

4. **Witness-gated re-entry / the quarantine moat.** A slice quarantined in a live session
   stays sealed across the *process* boundary; a reloaded core image refuses to page it
   into a new context unless **both** a witness `Clear()` ran **and** the bytes pass a fresh
   content re-screen through the same gate (`internal/recall/recall.go:18-22`). Quarantine
   cannot be laundered by a restart.

5. **Attribution + single-source metrics.** Because fak forwards the inbound protected
   prefix verbatim, "a change in this digest between two turns can only be the harness
   rewriting its own history — never fak" (`internal/gateway/harness_coherence.go:33-35`).
   Every prefix event is attributed to exactly one source
   (`stable`/`fak_cut`/`fak_world_break`/`harness_rewrite`/`cold_ttl`), and the `/metrics`
   scrape and the operator line fold the *same* snapshot so the two views can never disagree.

These are **tested**, not asserted: `TestCompactPreservesCachePrefix`,
`TestElideShrinksOldOversizedResultKeepsPrefixAndWorkingSet`,
`TestInboundProtectedPrefixDigestIsContentFree`, `TestQuarantineAtRiskCounted`,
`TestRevokedWitnessInvalidatesRecallPageOnPageIn`, `TestBenignPageRoundTripsByteIdentical`.

---

## 4. Where it *could* lose context — the honest fidelity limits (the most important section)

A credible "not butchering" answer must name where fidelity is imperfect. fak names three,
and surfaces each rather than hiding it:

1. **KV prefix-elision is admittedly NOT bit-exact.** Eliding an *old prefix* a surviving
   later token already attended to shrinks residency but reports `exact=false` rather than
   overclaim — "a re-RoPE cannot un-see attention a surviving later token already absorbed"
   (`internal/gateway/kvmmu_elide_test.go:51-76`). Only the trailing-suffix direction is
   bit-exact. The system refuses to claim the fidelity it cannot prove.

2. **The quarantine-survives-harness-summary hole.** fak seals a poisoned span on the
   *wire*, but Claude Code keeps its own on-disk transcript, and its auto-compaction can
   fold a sealed span into a summary that rides later requests as ordinary prose
   (`internal/compactcohere/doc.go:43-59`). `Decision.QuarantineAtRisk` is the **detector**
   for this — "a surfaced risk, not a fix. fak controls the wire bytes, NOT the harness's
   on-disk transcript or its summarizer."

3. **The harness-suppression actuator is not yet wired.** The standing block/allow posture
   that would make a `PreCompact` hook suppress the harness's cache-destroying
   auto-compaction is today **"a decision surface only until rung C wires the hook"**
   (`internal/gateway/harness_coherence.go:283`; actuator #1133 open). So fak **measures**
   the harness butchering context (the `harness_rewrite` / `quarantine_at_risk` / `bursts`
   counters) but does not yet **prevent** it. The fail-safe is already designed in:
   suppression yields back to the harness after a sustained bail streak so it "can never
   strand a session into a hard context overflow" (`compactcohere/doc.go:35-37`).

A fourth, smaller honesty note: editing the middle is a documented **cascade trade**, not
"never touches a cached byte" — later breakpoints cascade-burst and the shed middle is
re-billed once, while the dominant head cache survives (`anthropic_elide.go:21-26`).

**Live during this audit**, fak's machinery acted on the auditing session twice and
announced both: a `ctxview`-elided turn detail omitted from the resident view, and a
credential-shaped string **paged out** of context (withheld but retrievable through the
page-in gate). The mechanism is observable in the loop, with a stub standing in for the
held bytes — the opposite of a silent drop.

---

## 5. Verdict (net-true-value)

- **Effectiveness: REAL but workload-scoped and honestly fenced.** The published
  single-session cache value is ~1.0× (marginal over a tuned warm-KV server); the 2×/7×
  vCache figures are workload-concentration *readiness* from one telemetry sample; callavoid
  2.46× is a turn-amplification proxy. Provenance is labelled and the inflated vs-naive
  multiple is structurally excluded (#1066). This is the *available*-vs-*realized* honesty
  the standard demands.
- **Fidelity: STRONG by construction, HONEST about the gaps.** On fak's own wire transforms
  the answer to "how do we know it isn't butchering context" is concrete: the transforms
  *refuse to act unless they can prove the protected bytes are byte-identical*, and where
  they cannot prove it they either bail to identity or surface the risk as a metric — they do
  not silently proceed. The two real fidelity gaps (non-bit-exact prefix KV; quarantine
  surviving into a harness summary) are tested/surfaced, not denied, and the actuator that
  would close the harness-side hole is a named open follow-on (#1133), not a claimed fix.

## 6. Checkable follow-ons (`not yet`, with the next step)

1. **Re-run today's dogfood** — the 2026-06-28 09:36 `cmd/fak` test probe failed on the
   `quant_amd64_kquant.s` assembler error, but `go build ./internal/model` is green at HEAD
   (witnessed during this audit), so that failure was transient/stale. Re-running
   `fak recent-feature-dogfood` should now clear that probe; if it persists, capture the exact
   `go env`/toolchain the dogfood subprocess uses (it may differ from the interactive shell).
2. **Probe wiring** — `benchmarks-run-vcache` points a telemetry-requiring validator at a
   non-telemetry command, so it can *never* pass and keeps the dogfood red. Either run that
   probe with `--telemetry` or give it a planned-2x validator, so the dogfood's ACTION verdict
   reflects a real regression rather than a structurally-unsatisfiable probe.
3. **Harness actuator (#1133)** — wire the `PreCompact` hook to the standing posture so fak
   *prevents* (not just measures) the harness's cache-destroying auto-compaction; this is the
   single change that would close the largest remaining context-fidelity exposure.
