---
title: "DeepSeek V4 Pro/Flash TTFT/TPOT/context-scaling scorecard"
description: "The checked-in `fak deepseekbench` harness: a keyless dry-run fixture that locks the JSONL row schema (TTFT/TPOT/E2E/tok-s/token+cache counters) across the model × context-bucket × output × reasoning × stream matrix, plus the opt-in live-run recipe behind DEEPSEEK_API_KEY + an explicit spend flag. Reports OBSERVED provider speed only — never a fak-authored saving."
---

# DeepSeek V4 Pro/Flash TTFT/TPOT/context-scaling scorecard

> **Status: harness + keyless fixture shipped; live latency numbers unwitnessed.**
> The `fak deepseekbench` command, its locked row schema, and the speedup-refusal
> gate are checked in and CI-green with **no key**. No latency headline is claimed:
> every dry-run number is a labelled placeholder, and the live arm has not been run
> on a keyed node yet. This is the same honesty posture as the self-host smoke in
> [`DEEPSEEK-V4-SELFHOST-BASELINE-RUNBOOK.md`](DEEPSEEK-V4-SELFHOST-BASELINE-RUNBOOK.md),
> which is "wired and skips cleanly, ready to run."

Resolves [#3014](https://github.com/anthony-chaudhary/fak/issues/3014). Parent
program: [#3006](https://github.com/anthony-chaudhary/fak/issues/3006). Complements
the self-host baseline [#3013](https://github.com/anthony-chaudhary/fak/issues/3013).

## What it measures

One JSONL row per scorecard cell across the matrix the issue locks:

- **model** — `deepseek-v4-pro` **and** `deepseek-v4-flash`, always side by side
  (Flash behavior is never inferred from Pro).
- **context bucket** — `4K / 32K / 128K / 512K / 1M`.
- **output target** — `short / 1K / 8K / long-reasoning`.
- **reasoning mode** — `non-thinking / high / max`.
- **stream vs non-stream**.

Each row carries: `ttft_ms`, `tpot_ms`, `e2e_ms`, `output_toks_per_s`,
`prompt_tokens`, `completion_tokens`, `reasoning_tokens`,
`prompt_cache_hit_tokens`, `prompt_cache_miss_tokens`, `cache_attribution`, plus the
route identity (`model_id`, `provider_route`, `engine_route`, `hosting`) and the two
comparability keys (`prompt_shape`, `quality_parity`). The full locked list is
`RequiredFields()` in [`internal/deepseekbench/deepseekbench.go`](../../internal/deepseekbench/deepseekbench.go);
the field-lock test `TestRequiredFields` fails if any row drifts from it. The
`fak deepseekbench` command in [`cmd/fak/deepseekbench.go`](../../cmd/fak/deepseekbench.go)
is only the thin flag/I-O wire over that pure, isolation-testable package.

## The keyless dry-run fixture (CI-safe, no key, no network)

```bash
fak deepseekbench                 # JSONL rows -> stdout; coverage + honesty -> stderr
fak deepseekbench --out rows.jsonl
```

Every dry-run row is labelled `"measurement":"dry-run-fixture"` and
`"speed_provenance":"fixture-placeholder-not-measured"`. The numbers are a pure
deterministic function of the axes — they exist to **lock the schema and prove the
harness runs keyless**, exactly like a fixture. They are **not** measurements and must
never be quoted as performance.

## The opt-in live run (behind a key + an explicit spend ack)

```bash
# Hosted DeepSeek API:
DEEPSEEK_API_KEY=sk-… fak deepseekbench --live --spend --model deepseek-v4-pro

# Self-hosted OpenAI-compatible endpoint (vLLM/SGLang, per the #3013 runbook):
DEEPSEEK_API_KEY=… fak deepseekbench --live --spend \
  --base-url http://host:8000/v1 --model deepseek-ai/DeepSeek-V4-Pro
```

The live arm **refuses before any network call** unless `DEEPSEEK_API_KEY` is set
**and** `--spend` is passed (an explicit acknowledgement that the run costs money), so
**default CI never touches the network**. A live row — and only a live row — carries
`"measurement":"live"` and `"speed_provenance":"provider-observed"`, with TTFT timed to
the first content delta, TPOT as the mean inter-delta gap, and the token + prompt-cache
counters read from the provider's final `usage` block.

## OBSERVED provider speed is NOT a fak-authored saving

This scorecard reports **what the provider's endpoint did** — TTFT/TPOT/E2E/tok-s and
the provider's own prompt-cache hit/miss split. None of it is a fak-authored saving:

- **OBSERVED provider speed** — the latency/throughput numbers here. Attributable to
  DeepSeek's serving stack (and, for a self-hosted row, to vLLM/SGLang), never to fak.
- **DeepSeek's prompt-cache hit rate** — provider-observed economics. DeepSeek's context
  caching is on by default; a hit is the provider's relayed counter, booked exactly as
  [`deepseek_pricing.go`](../../internal/gateway/deepseek_pricing.go) books it — **never**
  as a fak-authored saving unless a separate fak mechanism demonstrably shaped the request.
- **fak-authored savings** — compaction/dedup/tool-floor pruning etc. — live in the
  Track-2 P&L ledger (`internal/cachevaluereport`), a *different* artifact. This
  scorecard does not touch them.

### The speedup-refusal gate

The report generator **refuses to print a speedup** unless both compared rows share a
`prompt_shape` **and** both carry `quality_parity:"verified"` **and** both are live
measurements. A dry-run fixture, a shape mismatch, or an unverified parity yields a
`[NOT COMPARABLE: …]` line instead of a number (see `CompareSpeedup`, tested by
`TestSpeedupRefusal`). A DeepSeek row is compared against an existing fak
route baseline **only** when that baseline was produced by this same harness under a
matching shape — otherwise the row is `[NOT COMPARABLE]` and says so.

## Cross-links

- Self-host wire-readiness runbook (the route this scorecard measures):
  [`DEEPSEEK-V4-SELFHOST-BASELINE-RUNBOOK.md`](DEEPSEEK-V4-SELFHOST-BASELINE-RUNBOOK.md).
- Provider pricing / cache-counter discipline:
  [`internal/gateway/deepseek_pricing.go`](../../internal/gateway/deepseek_pricing.go).
- Authority ledger (the only place a number becomes authoritative):
  [`../../BENCHMARK-AUTHORITY.md`](../../BENCHMARK-AUTHORITY.md).
- Parent program: [#3006](https://github.com/anthony-chaudhary/fak/issues/3006).
