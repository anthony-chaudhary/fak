---
title: "compress default-on witness: the native context-compressor is safe to ship enabled (2026-07-06)"
description: "The no-model, re-runnable evidence that flipping fak manage's --compress from off-by-default to ON is a net-true, do-no-harm default: a WITNESSED 81.6% aggregate corpus saving (recoverable), a deterministic pure-function compressor, a poison-skip + Quarantine-outranked security order, a clear one-flag opt-out, and a cache-safe wire no-op on the flagship Anthropic passthrough. Answers the six net-true-value questions."
slug: compress-default-on-witness
date: 2026-07-06
---

# compress default-on witness (2026-07-06)

`fak manage`'s efficiency levers already ship ON by default — `managed-cache=auto`,
`ctx-view-budget`, `compact-history-budget`, `elide-result-bytes`, `mcp-register`. The
**native context-compressor** (`--compress`, `internal/headroom`) was the last one still
default-OFF. This note is the evidence that flipping it ON is a **net-true, do-no-harm**
default — the [net-true-value](../standards/net-true-value.md) rubric answered in full, plus
the safety case for the "first, do no harm" bar in [`AGENTS.md`](../../AGENTS.md) ("Ship the
value ON by default — safely, or gate it honestly").

## The six net-true-value questions

1. **Baseline — the real alternative.** Against `--compress=false`: raw tool results reach the
   model verbatim. That *is* the alternative a competent operator runs today (it was the
   default). No strawman.

2. **Net — after the cost the change adds.** WITNESSED **81.6% aggregate byte saving** on the
   representative corpus (below), net of: the worth-it floor (a marginal saving on a small
   result is left RAW — nothing spent), the content-addressed preserve write (idempotent —
   the same original dedups to an existence check), and the codec annotation the model reads.
   Where the net goes **negative or zero**: an incompressible/single-use result (auto-no-op,
   admitted as-is — see `plain-prose` below), and the Anthropic passthrough wire (a no-op —
   the saving does not reach that wire; only a bounded µs–ms/turn compress scan remains,
   negligible against a multi-second frontier turn).

3. **Scope — where it holds, where it vanishes.** The gate rewrites the DECODED messages the
   kernel adjudicates (`req.Messages`), so its saving reaches the model **only on wires that
   re-serialize those messages** — codex / OpenAI / local / dual. On the `fak manage -- claude`
   **Anthropic passthrough the client's raw bytes are forwarded verbatim (`req.Raw`)**, so the
   compressor is a **wire no-op** there: it cannot shrink the forwarded tokens, and equally
   cannot disturb the provider prompt-cache. The scope *is* the claim: savings on the
   re-marshal wires, cache-safe no-op on the flagship passthrough.

4. **Provenance — measured, labeled.** WITNESSED. fak owns the corpus and the compressor; the
   bench stamps `owner=fak dependency=in_process fidelity=recoverable evidence=witnessed`.

5. **Witness — a third party can re-derive it.**
   - `fak headroom bench` (below) — the aggregate + per-sample savings, no model, no GPU.
   - `go test ./internal/headroom` — pins the corpus aggregate `Saved ≥ 0.5`
     (`bench_test.go`) and the compressor's **determinism** (`a.Render()==b.Render()`).
   - `go test ./cmd/fak -run Compress` — `TestCompressActivates` (opt-out semantics) and
     `TestCompressDefaultsOn` (the flag's registered default is `true`, source-pinned).

6. **Realized — on by default, honestly.** Now ON by default in `fak manage`. Opt out with
   `--compress=false` **or** `FAK_COMPRESSOR=noop` (an explicit `FAK_COMPRESSOR` always wins).
   Auto-disables (admits the original as-is) on incompressible input, on poison, and on the
   passthrough wire.

## What was run on this host (win32, no GPU) — `fak headroom bench`

```
compressor: native
status:     measured owner=fak dependency=in_process fidelity=recoverable evidence=witnessed

sample               kind   status      codec               orig   new   saved
colorized-test       text   saved       ansi-strip           668   578   13.5%
progress-bar         text   saved       cr-collapse+trim    5111    27   99.5%
scattered-warnings   text   saved       line-fold           1171   483   58.8%
pretty-json          json   saved       json-min             262   156   40.5%
repeated-retries     text   saved       line-dedup          1833   116   93.7%
crlf-build-log       text   saved       trim                 124    83   33.1%
plain-prose          text   no_effect   (none)               303   303    0.0%   <- auto-no-op, admitted as-is
TOTAL                       measured                        9472  1746   81.6%
```

## Why this clears the "first, do no harm" bar

- **Reversible.** The original bytes are preserved in the shared content-addressed store
  before the payload is rewritten (`admit.go` `preserveOriginal`); a wrong compress costs one
  demand-page, never a lost fact. `fidelity=recoverable`.
- **Security-preserving.** The gate sits at rank 8 — after the de-obfuscation rescan
  (normgate, 5), and its `Transform` verdict is **outranked by `Quarantine`** in the fold. It
  screens the raw bytes itself (`ctxmmu.ScreenBytes`) and **declines to touch poison**, so it
  can never hide an injection from the downstream security gates.
- **Auto-disable when it can't help.** The worth-it floor leaves a marginal/incompressible
  result RAW (the `plain-prose` sample); the compressor is a deterministic pure function of
  its input, so it is replay-stable and never introduces turn-to-turn variance.
- **Clear opt-out, minimal surface.** One flag (`--compress=false`) or one env
  (`FAK_COMPRESSOR=noop`) disables it; the explicit value always wins over the default.
- **Cache-safe on the flagship.** Wire no-op on the Anthropic passthrough (see scope) — the
  provider prompt-cache is untouched.

## Honest residuals

- **Not yet under the default-value scorecard.** `--compress` is not a tracked
  `internal/defaultvaluescore` value-flag token, so the flip is not machine-pinned by
  [`DEFAULT-VALUE-SCORECARD.md`](../DEFAULT-VALUE-SCORECARD.md) — and that scorecard currently
  has **no regeneration verb** (`defaultvaluescore.Build` has no non-test caller; there is no
  `cmd/fak/defaultvaluescore.go`, unlike `conflationscore`/`propagationscore`/`unwiredscore`).
  Adding the token now would strand a stale, do-not-hand-edit committed doc. Follow-up: add the
  missing generator verb, then add `"compress"` to `valueFlagTokens`. The regression is pinned
  meanwhile by `TestCompressDefaultsOn` (source-scan) here.
- **Passthrough compute.** On the Anthropic passthrough the discarded compress pass still runs
  (bounded, negligible). It can be skipped by mirroring the existing `suppressDecodedCtxViewKey`
  ctx flag (`internal/gateway/gateway.go`) — noted for completeness, not blocking.
- **Not listed in the token-defaults scorecard** (`docs/serving/token-defaults-scorecard.md`)
  by design: that card measures the *Anthropic passthrough* token economy, where the compressor
  is a no-op — listing it there would be a false passthrough-savings claim.
