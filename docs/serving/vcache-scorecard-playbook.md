---
title: "vCache Scorecard: The Operator Playbook"
description: "How to run fak vcache score, read the 2x agent-dev gate, build the hot-anchor index it plans, and move a workload from planned to telemetry-proven savings."
---

# vCache scorecard: the operator playbook

`fak vcache score` folds the vCache proof leaves -- planned star-anchor savings,
observed provider telemetry, workload concentration, false-warm risk, recall
economics -- into one operator artifact that answers three questions at once:

1. **Is this workload at least 2x better** with the virtual provider-cache than the
   naive floor?
2. **What index should the agent build** to capture that win (the hot-anchor list)?
3. **What is the next action** that moves the workload closer to the target?

It is a pure, off-path decision layer (`internal/vcachescore`, tier 2). It issues
no provider calls, warms no prefix, and treats provider cache hits as rebates only:
the score never depends on a cache hit landing. So it is safe to run anywhere -- a
laptop, CI, a pre-deploy gate -- with no key, no GPU, and no network.

## The one command

```
fak vcache score
```

A run prints the grade, the 2x gate verdict, the active savings source, the
hot-anchor index plan, the prediction-error rates, the recall economics, and a
worst-first action list. A real run over the built-in synthetic Zipf workload:

```
status: 2x_ready
grade: A (100/100)
active source: planned
active multiplier: 3.76x (target 2.00x)
2x gate: pass
planned proof: PROVEN saved 21094.4 / 28742.0 (73.4%)
planes: provider=MISSING kernel=MISSING context=MISSING external=MISSING forecast=FORECAST
agentic activation: 0 events (kernel=0 context=0 provider-decisions=0 external=0)
default usefulness: not_ready (F 32/100) - no realized provider/kernel/context/external evidence supplied; score is mostly forecast
concentration: s=1.74 measured=true defeated=false
hot-anchor index: top 8 covers 86.4% (target 85.0%)
prediction errors: false-warm 0.00% false-cold 0.00% (0 samples)
recall proof: refuted decision=cold_prefill break-even siblings=301
actions:
- ship the star-anchor path behind telemetry; keep uncached-first budgeting and prove realized savings per run
- collect provider telemetry with fak vcache prove-telemetry, then re-score on observed cache_read counters
- build a hot-anchor index for the top 8 anchors; expected coverage 86.4%
- keep chain recall off for single units; batch at least 301 siblings before rebuild
correctness depends on cache hit: false
```

Add `--json` for the machine-readable `Report` (schema, grade, score, the typed
proofs, the index, and the action/risk lists) and `--out FILE` to persist it.

## Reading the verdict

| Field | What it tells the operator |
|-------|----------------------------|
| `status` | `2x_ready` when the active multiplier clears the threshold; otherwise the gap to close. |
| `grade` / `score` | A 0-100 fold across the multiplier, concentration, index coverage, and false-warm rate -- an A means the workload is both better-than-2x AND low-risk, not just better. |
| `active source` | `planned` (the deterministic star-anchor proof) until you feed real provider telemetry, then `observed`. The score prefers observed over planned the moment telemetry is present. |
| `active multiplier` | The savings ratio the gate judges, against `--two-x` (default 2.0). |
| `2x gate` | The one-bit ship signal: does this workload earn the virtual-cache path? |
| `planes.provider_observed` | Provider prompt-cache telemetry, relayed from upstream usage counters. This is a rebate witness, not fak-owned reuse. |
| `planes.kernel_witnessed` | Pure-fak KV-prefix reuse evidence, supplied by `--kernel-kv-prompt-tokens` / `--kernel-kv-reused-tokens` or the gateway cache observer. |
| `planes.context_witnessed` | O(1) context/query value, including gateway compaction shed-token evidence. Shed-only evidence is visible but needs a resident/baseline denominator for net-value credit. |
| `planes.external_engine_observed` | SGLang/vLLM/llama prefix-cache hit-rate evidence. Hit rate alone improves coverage, not token-value credit. |
| `agentic_activation` | Counts fak-authored mechanisms that fired. Provider cache counters alone do not increment it. |
| `default_usefulness` | A separate conservative score over realized value, activation, cold-path correctness, granularity, coverage, drift resistance, and actionability. |
| `concentration` | The Zipf `s` of the workload's anchor reuse. `defeated=true` means the workload is too flat for anchor caching to help -- the honest "do not bother" case. |
| `hot-anchor index` | The plan: how many top anchors to index and the coverage they buy. This is the artifact the agent persists (`--index-out FILE`). |
| `prediction errors` | False-warm / false-cold rates from any calibration samples you pass -- the cost of warming the wrong anchors. |
| `recall proof` | The chain-recall economics: for a single small unit recalled from a long warm prefix, the cost gate REFUSES (a net loss); rebuild wins only past the named sibling fan-out. |

The old 2x gate remains intentionally provider-compatible: a provider-only run can
print `2x gate: pass` while `default_usefulness` stays `partial` because fak did
not author any cache action. Supply fak-owned activity with
`--kernel-kv-events`, `--context-events`, `--provider-vcache-decisions`, and
`--external-engine-events`. For value witnesses, add
`--kernel-kv-prompt-tokens` / `--kernel-kv-reused-tokens` for pure-fak KV,
`--context-shed-tokens` / `--context-resident-tokens` for O(1) context, and
`--external-engine-hit-rate` for external serving engines. A running gateway also
exposes the same report at `GET /v1/fak/vcache/score`.

## The workflow: planned to proven

The scorecard is designed to be run twice -- once before you have provider data, once
after.

1. **Score the planned path (no telemetry).** `fak vcache score` runs the
   deterministic star-anchor proof over your workload shape. Tune the shape with
   `--zipf-s`, `--anchors`, `--anchors-file`, `--target-coverage`, and the cost
   multipliers (`--read-mult`, `--write-mult`, `--write-5m-mult`, `--write-1h-mult`).
   This tells you whether the workload *can* clear 2x before you spend a run on it.

2. **Build the hot-anchor index.** When the gate passes, persist the planned index
   with `--index-out anchors.json`. That provider-neutral artifact is what the agent
   loads to warm the top anchors the scorecard selected.

3. **Collect provider telemetry.** Run the workload behind `fak guard` / `fak serve`
   and capture the provider usage JSONL (Claude Code probe output, OpenAI
   Responses/Chat usage objects, Codex CLI `token_count` rows). Prove realized
   savings standalone with `fak vcache prove-telemetry --file usage.jsonl`.

4. **Re-score on observed counters.** `fak vcache score --telemetry usage.jsonl`.
   Now `active source` flips to `observed` and the grade reflects the cache reads the
   provider actually served, not the model's planned ceiling. This is the honest
   number to publish: realized, provider-witnessed, not modeled.

The split matters because fak only controls one half of a cache hit. The bytes it
ships are byte-identical by construction (WITNESSED); whether the provider reuses
them is the provider's call (OBSERVED, relayed verbatim). The planned proof is the
ceiling fak can guarantee; the telemetry proof is what the provider delivered.

## Using it as a gate

The score is deterministic, so it gates cleanly:

- **Pre-deploy:** run `fak vcache score --telemetry <last-run>.jsonl --two-x 2.0` and
  refuse a rollout that regressed below the target multiplier.
- **CI:** wire the same call into a repeatable dogfood step (see the companion CI
  gate) so a change that quietly defeats anchor concentration turns the build red.
- **Per-workload triage:** a `defeated=true` concentration is the scorecard telling
  you this workload is too flat for the virtual cache -- spend the warm budget
  elsewhere rather than forcing it.

## Related surfaces

- `fak vcache status` -- what is actually wired right now (the M5 governor is live and
  off-path; calibration/warming/recall stages are issue-tracked).
- `fak vcache prove` -- the deterministic star-anchor token-savings proof in isolation
  (exit 0 PROVEN, 1 REFUTED).
- `fak vcache prove-telemetry` -- realized savings from one provider usage JSONL.
- `fak vcache prove-recall` -- the chain-recall cost-gate proof (the single-unit
  recall is a net loss; rebuild wins only for amortized fan-out).
- `GET /v1/fak/vcache/score` -- the served API twin of `fak vcache score`, folding
  live provider telemetry, kernel KV observer data, compaction shed-token context
  evidence, and external serving-engine cache hit-rate rows.
- [Hardware-aware KV cache](hardware-aware-cache.md) -- where a warmed span physically
  lives across the memory tiers.
