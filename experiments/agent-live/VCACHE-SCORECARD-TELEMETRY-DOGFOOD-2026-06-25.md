# vCache Scorecard Telemetry Dogfood - 2026-06-25

Issue #789 (epic #788). Companion to
[VCACHE-CODEX-OPENAI-PROBE-2026-06-25.md](VCACHE-CODEX-OPENAI-PROBE-2026-06-25.md).

## Question

Does `fak vcache score` report 2x readiness from REAL provider telemetry -- not
only the synthetic Zipf workload or the planned star-anchor proof?

## Current status

**Proven from the committed Codex CLI session telemetry.** The sibling probe doc
proves the raw savings with `fak vcache prove-telemetry`; this dogfood runs the
full SCORECARD (`fak vcache score --telemetry`) over the same telemetry, so the
2x agent-dev gate is exercised on observed `cached_input_tokens`, not just the
deterministic planned proof. When telemetry is supplied, the scorecard's
`active_source` flips from `planned` to `telemetry` and the grade reflects the
cache reads the provider actually served.

No API key, no network: the telemetry is replayed from a committed JSONL
artifact, so the result is reproducible by anyone with the repo and the Go
toolchain.

## Command (reproducible)

```sh
go run ./cmd/fak vcache score \
  --telemetry experiments/agent-live/vcache-codex-token-count-proof-2026-06-25.jsonl \
  --json \
  --out experiments/agent-live/vcache-score-codex-telemetry-2026-06-25.json
```

- **Telemetry input (replayable):**
  `experiments/agent-live/vcache-codex-token-count-proof-2026-06-25.jsonl`
  (68 Codex CLI `token_count` rows; `last_token_usage.input_tokens` +
  `cached_input_tokens`, no API key, no PII).
- **Frozen score artifact:**
  `experiments/agent-live/vcache-score-codex-telemetry-2026-06-25.json`.
- **Threshold:** the default 2x gate (`--two-x 2.0`).
- **Environment assumption:** OpenAI/Codex read-discount accounting
  (`read_mult = 0.1`), the same model the sibling `prove-telemetry` proof uses.

## Result (PASS)

```text
status:            2x_ready
grade:             A (100/100)
active source:     telemetry          <- flipped from planned by --telemetry
active multiplier: 7.13x  (target 2.00x)
2x gate:           pass
observed proof:    PROVEN saved 85.98% (68 requests, first positive request 1)
planned proof:     PROVEN saved 73.4% (the deterministic floor, still reported)
correctness depends on cache hit: false
```

### Economics (hit / read / rebate / cost) — from the frozen artifact

These are the four economics acceptance bar #1 asks for, read straight off
`vcache-score-codex-telemetry-2026-06-25.json` (`observed` block); nothing here is
hand-entered.

| Economic | Value | Source field(s) |
|----------|-------|-----------------|
| **hit** — cache-read share of input | 95.53% (10,163,712 of 10,638,831 input token-equiv) | `cache_read_tokens / baseline_token_equiv` |
| **read** — cached input tokens served | 10,163,712 (cache *writes* 0) | `cache_read_tokens` / `cache_creation_tokens` |
| **rebate** — token-equivalents saved | 9,147,340.8 (85.98% of baseline) | `saved_token_equiv` / `saved_pct` |
| **cost** — token-equivalents actually paid | 1,491,490.2 vs 10,638,831 baseline | `actual_token_equiv` vs `baseline_token_equiv` |
| **2x readiness** | 7.13x ≥ 2.00x → pass | `active_multiplier` / `two_x_better` |

The cost is the realized read discount, not a trust claim: at `read_mult = 0.1` the
identity `actual = baseline − (1 − read_mult)·read` holds exactly —
`1,491,490.2 = 10,638,831 − 0.9 × 10,163,712`. The rebate is what the provider's
cache reads actually repaid; budgeting still happens at the uncached price.

The scorecard reports realized hit/read/rebate economics and 2x readiness from
the observed telemetry, and the planned star-anchor floor stays on its own line
-- so the artifact carries both the provider-witnessed number and the number fak
can guarantee without a provider.

## Acceptance (issue #789)

- [x] `fak vcache score` reports hit/read/rebate/cost AND 2x readiness from live
  telemetry -- `active source: telemetry`; hit 95.53%, read 10,163,712 cached
  tokens, rebate 9,147,340.8 token-equiv (85.98%), cost 1,491,490.2 vs 10,638,831
  baseline, multiplier 7.13x, gate pass (see the economics table above).
- [x] The artifact is regenerable by the documented command above.
- [x] Secrets and customer data are out of the artifacts -- the telemetry is
  token-count rows only (verified: 0 secret-shaped matches; the sibling probe doc
  documents the capture and redaction).
- [x] CLAIMS / BENCHMARK-AUTHORITY: no NEW public fak performance claim is made, so
  no new authority row is required. The observed 85.98% / 7.13x is OBSERVED
  (provider-relayed cache reads, not a fak-caused effect) and is already recorded
  in CLAIMS.md's M5 vCache Governor entry; the deterministic planned floor (73.4%)
  is the only fak-guaranteed number and is also in CLAIMS.md. BENCHMARK-AUTHORITY
  indexes fak's own measured throughput/reuse numbers, not a relayed provider
  counter.

## What this does and does not prove

This proves the SCORECARD reads observed Codex/OpenAI prompt-cache economics and
turns them into a 2x readiness verdict from a real session. It does **not** prove
fak's vCache layer caused those savings -- the Codex session cached the prefix on
its own; the scorecard is reading the provider's relayed counters (OBSERVED), not
claiming them as a fak effect. It supports the vCache design premise the sibling
doc lays out: if fak keeps prefixes stable and routes related turns together, the
scorecard's observed multiplier is what the provider will actually deliver.

The honest two-number split is the point. The planned proof (73.4%) is the
deterministic ceiling fak guarantees; the observed proof (85.98%) is what this
provider delivered on this thread. The repeatable gate
(`tools/vcache_scorecard_gate.py`, #791) asserts the deterministic planned floor;
this dogfood records the observed number against a frozen, replayable artifact.
