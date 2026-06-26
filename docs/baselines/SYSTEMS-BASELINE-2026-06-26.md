---
title: "Systems baseline — the runnable usable-today surfaces (2026-06-26)"
description: "A fresh, reproducible baseline of fak's newly-surfaced runnable systems (fak route, fak ablate, fak vcache prove/prove-recall/prove-telemetry/score, fak task sample, a2ademo). Each system was run fresh, its output committed as a baseline artifact with a sha256 digest, and its determinism classified, so a future regression is visible by re-running and diffing. Pinned to fak v0.34.0."
---

# Systems baseline — 2026-06-26

This is a fresh scoring and baseline for the runnable, "good enough to use today"
systems that the [product map](../PRODUCT-STATUS.md) surfaces. Each system was run
fresh, its output captured as a committed baseline artifact under
[`2026-06-26/`](2026-06-26/), digested with sha256, and classified by whether it
reproduces. A future run re-runs the same command and diffs against the artifact (or
re-hashes it); a changed deterministic field is a visible regression.

The machine-readable form is [`2026-06-26/manifest.json`](2026-06-26/manifest.json)
(`schema: fak.systems-baseline/1`): per system it carries the command, expected exit,
determinism class, artifact path, sha256, headline metrics, what it pins, the honest
fence, and how to compare.

## Baseline source (pinned, reproducible)

| Field | Value |
|---|---|
| Binary | `fak` v0.34.0 |
| Commit | `5239698` (the `v0.34.0` release tag, an ancestor of `main`) |
| Build | `git checkout v0.34.0 && go build -o fak ./cmd/fak && go build -o a2ademo ./cmd/a2ademo` |
| Host | windows / amd64 |

The baseline is pinned to the `v0.34.0` release tag rather than `HEAD` on purpose: on
2026-06-26 `HEAD` was transiently un-buildable (two in-flight peer commits left
`internal/model` and `internal/loopmgr` with a missing method and a duplicate symbol).
`v0.34.0` is green by construction (a release tag) and is an ancestor of `HEAD`, so the
baseline is reproducible by anyone with `git checkout v0.34.0 && go build`. Every system
here shipped in v0.34.0; re-pin to the next green release as the systems evolve.

## The systems

| System | Command | Exit | Determinism | Artifact | Headline |
|---|---|---|---|---|---|
| `fak route` (fail-closed default) | `fak route --json` | 0 | yes | [`route.json`](2026-06-26/route.json) | matched=false → fail-closed default model |
| `fak route` (matched vote-ensemble) | `fak route --manifest examples/model-routing.example.json --aspect tool_call --tool refund_payment --json` | 0 | yes | [`route-ensemble-refund.json`](2026-06-26/route-ensemble-refund.json) | matched vote-ensemble: guard-a+guard-b (reduce=vote) |
| `fak ablate` (vDSO on/off) | `fak ablate --sweep vdso --json` | 0 | counters yes / latency varies | [`ablate-vdso.json`](2026-06-26/ablate-vdso.json) | vDSO 7/12 hits; engine_calls 12→5; hash 9f1701415fb4a360 |
| `fak vcache prove` (M5 planned floor) | `fak vcache prove --json` | 0 (PROVEN) | yes | [`vcache-prove.json`](2026-06-26/vcache-prove.json) | PROVEN — saved 73.39% (21,094.4/28,742 tok-eq) |
| `fak vcache prove-recall` (M4 cost gate) | `fak vcache prove-recall --json` | 1 (REFUTED by design) | yes | [`vcache-prove-recall.json`](2026-06-26/vcache-prove-recall.json) | REFUTED — replay 3000 vs cold; break-even 301 siblings |
| `fak vcache prove-telemetry` (real Codex usage) | `fak vcache prove-telemetry --file experiments/agent-live/vcache-codex-token-count-proof-2026-06-25.jsonl --json` | 0 (PROVEN) | yes | [`vcache-prove-telemetry-codex.json`](2026-06-26/vcache-prove-telemetry-codex.json) | PROVEN — 85.98% realized (9,147,340.8 tok-eq, 68 events) |
| `fak vcache score` (2x readiness gate) | `fak vcache score --telemetry experiments/agent-live/vcache-codex-token-count-proof-2026-06-25.jsonl --json` | 0 | yes | [`vcache-score-codex.json`](2026-06-26/vcache-score-codex.json) | grade A, score 100, 7.13x ≥ 2x → 2x_ready |
| `fak task sample` (resource snapshot) | `fak task sample --json` | 0 | shape only | [`task-sample.json`](2026-06-26/task-sample.json) | SHAPE: schema/process/resource/tasks/concepts |
| `a2ademo` (in-kernel A2A) | `go run ./cmd/a2ademo` | 0 | yes | [`a2ademo.txt`](2026-06-26/a2ademo.txt) | ALLOW + DENY(TRUST_VIOLATION) + pub/sub + handoff |

Every artifact was double-run during capture and confirmed to reproduce its
deterministic fields. Two carry environment-measured fields that legitimately vary:

- **`fak ablate`** — the kernel COUNTERS (`engine_calls`, `vdso_hits`, `calls`, the
  per-arm deltas) and `workload_hash` are the stable pin. The `wall_seconds` and
  `p50/p99/mean_ns` latency fields are measured per run and vary; ignore them when
  comparing.
- **`fak task sample`** — a SHAPE baseline only. The resource values (`wall`/`heap`/
  `sys`/`goroutines`, `*_unix_nano`) sample live runtime state and vary every run; pin
  the JSON key set / `schema`, not the values.

## What each baseline pins, and its honest fence

- **`fak route`** pins the deterministic policy decisions: a no-subject request hits the
  fail-closed Default (the oracle never fails open), and a `refund_payment` write routes
  to a two-model vote ensemble (`guard-a` primary + `guard-b` verifier, `reduce: vote`).
  Fence: the oracle decides and reduces stand-in outputs; binding members to live model
  calls is the caller's integration.
- **`fak ablate`** pins the measured ablation delta: on `tau2-airline-smoke` the vDSO
  turns 7 of 12 tool calls into local cache hits, so `engine_calls` drops 12 → 5, bound
  to `workload_hash 9f1701415fb4a360`. Fence: rung 1 sweeps only the runtime vDSO knob.
- **`fak vcache prove`** pins the GUARANTEED planned floor: a star anchor saves 73.39%
  token-equivalents (21,094.4 of 28,742) across 7 sibling requests, with no provider in
  the loop. This is the only fak-guaranteed vcache number.
- **`fak vcache prove-recall`** pins the cost gate REFUSING a single-unit recall rebuild
  (replay 3000 vs sending cold; break-even = 301 siblings → `decision: cold_prefill`).
  Exit 1 is the correct designed outcome (the ~300× loss caught), and the engine is OFF
  by default until the live provider recall loop flips it on.
- **`fak vcache prove-telemetry`** / **`score`** pin the REALIZED savings replayed from
  committed Codex CLI usage (68 events): 85.98% saved, grade A, observed multiplier
  7.13× ≥ the 2× threshold → `2x_ready`. Fence: these are OBSERVED numbers relayed from
  the provider's own cache counters (a realized rebate), not a fak-caused effect; the
  73.4% planned floor is the only number fak guarantees without a provider.
- **`fak task sample`** pins the snapshot shape a long-running fak process embeds.
  Fence: it is a process-local reference fold, not a durable scheduler or cross-PID
  monitor.
- **`a2ademo`** pins the deterministic trace proving an in-kernel mailbox delivers an
  addressed value under the same default-deny floor a tool call rides (a cross-channel
  private send is `DENY (TRUST_VIOLATION)`). Fence: in-process delivery is real; durable
  cross-process delivery is the labeled-stub next rung.

## Reproduce / re-baseline

```bash
git checkout v0.34.0
go build -o fak ./cmd/fak && go build -o a2ademo ./cmd/a2ademo
# run any row's command (from the repo root) and diff against the committed artifact:
./fak vcache prove --json | diff - docs/baselines/2026-06-26/vcache-prove.json
sha256sum -c <(cd docs/baselines/2026-06-26 && sha256sum *.json *.txt)   # digest check
```

To re-baseline on a future green release: rebuild from the new tag, re-run each command,
overwrite `2026-06-26/` with a new dated folder, and regenerate `manifest.json`. Compare
the deterministic fields (and the vcache exit codes / `saved_pct` / `grade`); ignore the
ablate latency and task-sample resource values.
