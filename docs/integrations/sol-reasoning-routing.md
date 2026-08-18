# GPT-5.6 Sol execution routing for guarded workers

**Status (2026-08-17): FAK classifies every orchestration task as `standard`, `max`, `ultra`, or `pro` before a worker launch. `pro` is recorded but fails closed because Codex CLI 0.147.0 cannot transmit `reasoning.mode`.**

## The four names are not one scale

| FAK route | Provider mechanism | Default use | Cost / risk boundary |
|---|---|---|---|
| `standard` | one Codex worker, `reasoning_effort=high` | small, bounded implementation work | fastest and least expensive guarded route |
| `max` | one Codex worker, `reasoning_effort=xhigh` | correctness-heavy, uncertain, security, architecture, or adversarial work | more reasoning time and tokens; no extra independence |
| `ultra` | FAK's leased multi-worker orchestration, each at `high`, with independent effect readback | fleet waves, backlog work, fan-out, and genuinely independent issue work | multiplies worker usage; collision control and reconciliation are mandatory |
| `pro` | OpenAI Responses `reasoning.mode="pro"`, independent of effort | an explicit, separately metered consultation or adversarial review | potentially slow/expensive; not suitable for every tool-use turn |

OpenAI's reasoning guide says `reasoning.mode` chooses `standard` or `pro` execution and is independent of `reasoning.effort`; omitted effort defaults to `medium` in either mode. Do not spell Pro as `xhigh`, `max`, or a model slug. Product “Ultra” is likewise not an API reasoning mode: it is multi-agent delegation (or, on some surfaces, a UI label for maximum effort).

Sources, observed 2026-08-17:

- OpenAI reasoning guide: <https://developers.openai.com/api/docs/guides/reasoning#reasoning-mode>
- Codex does not yet expose `reasoning.mode`: <https://github.com/openai/codex/issues/32823>
- The proposed safe Pro shape is an asynchronous, explicit consultation, not every turn: <https://github.com/openai/codex/issues/35247>
- Codex Ultra uses a Responses `multi_agent` parameter and currently has provider/account constraints: <https://github.com/openai/codex/issues/37858>

## Current FAK and developer defaults

The managed Codex developer configuration selects `gpt-5.6-sol` with `model_reasoning_effort="xhigh"`. That means **maximum available effort is enabled for our interactive development sessions; Pro is not enabled**. The local Codex model cache advertises only `low`, `medium`, and `high` for Sol even though the managed override accepts `xhigh`, so a launch receipt must record the requested route rather than infer provider execution from the picker.

FAK orchestration now makes the decision explicit in `resolved.sol_route` and launch receipts:

```powershell
fak orchestration plan --profile auto --task-text "audit an uncertain security invariant" --json
fak orchestration plan --profile auto --task-text "run a fleet wave over independent issues" --json
```

The first resolves to `max/xhigh`; the second resolves to `ultra/high` and the ultracode profile. An explicit `consult pro` request resolves to `pro`, but `--launch` returns `SOL_ROUTE_PRO_CONSULT_ONLY` instead of silently launching ordinary reasoning.

## Default policy

- Keep ordinary issue workers on `standard/high`.
- Select `max/xhigh` when correctness or uncertainty dominates and parallelism would not provide independent evidence.
- Select `ultra/high` for fleet-wave, super-loop, backlog, unattended, or clearly decomposable independent work. Leases, bounded workers, independent witnesses, telemetry receipts, and reconciliation remain required.
- Select `pro` only when the issue explicitly asks for a Pro consultation. Until Codex supports `reasoning.mode`, fail closed and retain the planned question; never relabel xhigh as Pro.
- Do not enable Pro globally for FAK development. Its value is concentrated in scarce architectural decisions and final adversarial review, while its latency/cost and missing Codex wire support make default-on both inefficient and unverifiable.

The deterministic classifier lives in `internal/orchestration/solroute.go`; it is deliberately conservative and emits its decision reason so telemetry can later replace keyword policy with measured outcomes.
