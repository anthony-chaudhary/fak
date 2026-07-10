---
title: "Borrow scout: zml → fak (2026-07-08)"
description: "Study of zml — a production ML-inference stack in Zig (compile a model to an accelerator-specific executable, then run it) — for techniques worth porting."
---

# Borrow scout: zml → fak (2026-07-08)

Study of **zml** — a production ML-inference stack in Zig (compile a model to an accelerator-specific
executable, then run it) — for techniques worth porting into fak. Apache-2.0; pinned SHA
`5cacbe606e2e4c4d15db9b61aab0f2cfbcc70b99` (HEAD subject: `platforms/intel: full hermetic sandboxing (#638)`,
2026-07-08), cloned read-only to session scratch. Every candidate below is **INSPIRE**, not INTEGRATE:
zml is Zig, fak is Go, so any port is a clean-room reimplementation — no source is copied, and the
Apache-2.0 + TigerBeetle attribution note on candidate G is courtesy, not obligation.

Scope note: zml's core product (MLIR/XLA lowering, PJRT device buffers, sharding, kernel codegen, DMA
weight streaming) is **out of scope** for fak — fak orchestrates coding agents and adjudicates their
tool calls; it does not compile or run neural nets. So the candidates below are zml's *robustness*,
*measurement-honesty*, and *type-discipline* techniques — the parts that generalize past the ML domain
— witnessed against fak's existing `adjudicator` / `conceptbench` / `bench` / `commitintent` machinery.

## Scorecard (7 candidate techniques, one technique each)

| # | Technique (zml anchor @ SHA 5cacbe6) | fak witness | Verdict |
|---|---|---|---|
| A | **Scheme-keyed VFS backend registry** — one uniform IO interface over many backends registered by URI scheme (`file`/`http`/`hf`/`s3`/`gcs`); a caller opens `s3://…` exactly like a local path and never knows which backend answered. `zml/io/vfs.zig` (register/`stripScheme`/scheme-dispatched lookup) | **PRESENT, fak ahead**: fak has a *named, documented* Agent Virtual Filesystem — "one uniform `open/read/write/stat` interface over many different backends … the kernel deciding, per call, what the program is even allowed to see" (`docs/explainers/agent-virtual-filesystem.md:33-42`). fak adds the reference-monitor seam (adjudicator) that zml's pure-dispatch VFS does not. | PRESENT — no borrow |
| B | **Exact-percentile distributional comparison** — compare over raw samples with p50/p90/p99/p999 (never histogram buckets), because bucketing quantizes the tail you are trying to see. `zml/testing.zig:164-228` (`compareSlices`/`CompareReport`) | **PRESENT, fak house style**: `internal/bench/tailload.go:17-21` — "the per-call latency DISTRIBUTION (exact percentiles over raw samples, not histogram buckets — a bucketed p99 would quantize exactly the tail this arm exists to see)"; plus the 100 µs `adjudication_latency_test.go` distribution gate. | PRESENT — no borrow |
| C | **Claim-vs-materialized separation** — distinct types for "described but not realized" (`Shape`/`Tensor`, metadata only, owns no memory) vs "materialized" (`Buffer`, on-device); the type system forbids treating a description as a result. `docs/learn/concepts.md:44-74,117-146` | **PRESENT, fak ahead**: fak enforces the same wall at runtime with a closed witness vocabulary + honesty gate — `nonWitness = {"", replay, none, scaffold, -, placeholder, self_report}` and `result_claim_allowed` computed, not trusted, so "a scaffold can never masquerade as a measured result" (`internal/conceptbench/report.go:1-69`); `abi.WitnessPayload{Claim}` throughout `internal/adjudicator/decide.go`. zml enforces at compile time via types; fak enforces at runtime across processes via a closed set — appropriate to fak's dynamic, cross-process reality. | PRESENT — no borrow |
| D | **Injection-safe token grammar** — when encoding metadata into one delimited string (`name#k=v,k2=v2#`), *panic* if any token contains the reserved delimiter, and size the buffer at comptime to avoid hot-path alloc. `zml/profiling/tracer.zig:104-108,114-150` | **PRESENT by different means**: fak avoids the vulnerable grammar class entirely — stamp text is built from a regex-constrained leaf (`leafRE`, `internal/commitintent/commitintent.go:483-514`) so `(fak <leaf>)` delimiters can't be injected, and structured metadata is JSON (quoted/escaped by construction), with `\r\n` rejected on free-text fields (`commitintent.go:516-544`). No hand-rolled `k=v` delimiter grammar exists to harden. | PRESENT (diff. means) — latent note below |
| E | **Overlap two independent startup bottlenecks** — compile the executable and stream weights from disk *concurrently* via `std.Io`, since both dominate startup and neither depends on the other. `docs/learn/concepts.md:36-37` | **N-A / out of domain**: the two heavyweight independent phases zml overlaps (graph compile + multi-GB weight load) live *inside the inference engines fak dispatches to* (vLLM/sglang), which already overlap them; fak itself has no such pair. fak's own preflight (`cmd/fak/dispatch_tick_preflight.go:35-60`) gathers cheap local probes and is deliberately serial+deterministic (an admission path that folds caps DOWN monotonically — determinism is the feature, not a bottleneck to parallelize). | N-A — no ticket |
| F | **Element-wise tolerance verdict** — compare two value arrays with combined absolute+relative tolerance and pass iff `close_fraction ≥ minimum_close_fraction` (default 0.999). `zml/testing.zig:43-114` (`expectClose`) | **LATENT**: fak's comparisons are pass/fail fidelity (`conceptbench`) or single-distribution percentile gates (`tailload`, latency); there is no site today that compares two continuous *golden vectors* element-wise where a "≥99.9% within tolerance" verdict would beat exact-equality. | LATENT — revisit-if (below) |
| G | **Operator-robust flag discipline** — no silent defaults in production, reject duplicate flags, `--key=value` only, precise fatal messages ("make operator errors harder to make"). `zml/stdx/flags.zig` (itself ported from TigerBeetle, Apache-2.0) | **PRESENT / N-A**: fak's CLI conventions are long-established (`cmd/fak`), and this is a diffuse guideline set rather than a crisp code borrow. No single actionable seam. | PRESENT/diffuse — no borrow |

**Outcome: 0 borrows of 7.** Witnessing prevented 7 duplicate / N-A / out-of-domain tickets. The
headline finding is *convergence*: a production ML-inference compiler in Zig, from a completely
unrelated domain, independently arrived at fak's load-bearing disciplines — VFS-over-backends,
exact-percentile (anti-quantization) measurement, and the claim-vs-materialized honesty wall — with
fak generally **ahead** on the trust dimension (reference monitor on the VFS seam; a runtime closed
witness set + honesty gate; regex+JSON token safety). That cross-domain convergence is external
validation of the architecture, and is the deliverable here.

## Not filed: candidates F and D — latent, revisit-if trigger

Do **not** file today.

- **F (element-wise tolerance + `close_fraction` verdict).** fak has no golden-*vector* comparison
  site where this beats what it already does. **Trigger to revisit:** if fak ever grades a continuous
  numeric output against a stored golden array (e.g. a scoring-model calibration vector, an embedding
  regression, a per-token logprob replay), port `expectClose`: per-element `abs_tol + rel_tol·scale`
  with a `close_fraction ≥ threshold` pass and a percentile error report — it resists single-outlier
  flapping that exact-equality and mean-error both mishandle. Anchor to re-read then: `zml/testing.zig:43-228`.

- **D (reserved-delimiter panic).** fak is safe today because it never hand-rolls a delimited
  single-line token grammar from caller-supplied values. **Trigger to revisit:** if fak ever builds
  such a grammar (e.g. a compact single-line trace-span/label encoding like `span#k=v,k2=v2#` on a
  hot path where JSON is too heavy), adopt zml's rule — validate every token against the reserved
  delimiter and fail loud, don't silently corrupt the grammar. Anchor: `zml/profiling/tracer.zig:104-108`.

## Trail

- zml pinned: `5cacbe606e2e4c4d15db9b61aab0f2cfbcc70b99` (Apache-2.0), read-only clone in session scratch.
- fak witness sites cited inline at `path:line` (HEAD of `main` at study time).
- Method: zml `docs/learn/concepts.md` + load-bearing modules (`io/vfs.zig`, `testing.zig`,
  `profiling/tracer.zig`, `stdx/{flags,debug}.zig`) → 7 candidates, one technique each → witnessed each
  against fak by reading the decisive fak site (`conceptbench/report.go` honesty gate, `bench/tailload.go`
  percentile discipline, `agent-virtual-filesystem.md` VFS seam, `commitintent.go` stamp grammar,
  `dispatch_tick_preflight.go` admission path) rather than trusting a lexical catalog match — guarding
  against false-ABSENT exactly as the sibling [study-pxpipe-borrow-scout-2026-07-08.md](./study-pxpipe-borrow-scout-2026-07-08.md) note prescribes.
