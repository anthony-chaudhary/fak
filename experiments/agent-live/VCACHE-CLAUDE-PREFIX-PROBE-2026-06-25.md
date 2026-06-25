# vCache Claude Prefix Probe - 2026-06-25

## Question

Can the vCache star-anchor idea be proven or refuted with real Claude telemetry?

## Setup

- Runner: Claude Code CLI, headless JSON mode.
- Model reported by the CLI: `claude-opus-4-8[1m]`.
- Prompts: four one-turn requests with the same Claude Code/project/tool prefix and a tiny changing user suffix.
- Raw witness: `experiments/agent-live/vcache-claude-prefix-probe-2026-06-25.jsonl`.
- Pricing model: Anthropic prompt-caching multipliers from the Claude prompt-caching docs: 1h writes cost `2.0x` base input, cache reads cost `0.1x` base input. Anthropic's docs also define the usage fields `cache_creation_input_tokens`, `cache_read_input_tokens`, and the `cache_creation.ephemeral_1h_input_tokens` split.

## Commands

```powershell
$null | claude -p 'Reply with exactly: vcache-smoke-pong' --output-format json --dangerously-skip-permissions
$null | claude -p 'Reply with exactly: vcache-sibling-alpha' --output-format json --dangerously-skip-permissions
$null | claude -p 'Reply with exactly: vcache-sibling-beta' --output-format json --dangerously-skip-permissions
$null | claude -p 'Reply with exactly: vcache-sibling-gamma' --output-format json --dangerously-skip-permissions
```

## Observed telemetry

| turn | result | input | 1h cache write | cache read | cost |
|---|---:|---:|---:|---:|---:|
| create | `vcache-smoke-pong` | 10,098 | 59,400 | 0 | $0.64479 |
| sibling_alpha | `vcache-sibling-alpha` | 10,065 | 15,411 | 43,995 | $0.2267575 |
| sibling_beta | `vcache-sibling-beta` | 10,065 | 15,410 | 43,995 | $0.2267225 |
| sibling_gamma | `vcache-sibling-gamma` | 10,065 | 15,424 | 43,995 | $0.2268875 |

## Token-equivalent accounting

Re-run the verifier:

```powershell
go run ./cmd/fak vcache prove-telemetry --file experiments/agent-live/vcache-claude-prefix-probe-2026-06-25.jsonl
```

Baseline input-equivalent tokens assume the whole prompt would have been processed cold:

`baseline = input_tokens + cache_creation_input_tokens + cache_read_input_tokens`

Observed input-equivalent tokens apply the Claude prompt-cache multipliers:

`actual = input_tokens + 2.0 * ephemeral_1h_input_tokens + 0.1 * cache_read_input_tokens`

| turn | baseline input-equiv | actual input-equiv | delta |
|---|---:|---:|---:|
| create | 69,498.0 | 128,898.0 | -59,400.0 |
| sibling_alpha | 69,471.0 | 45,286.5 | 24,184.5 |
| sibling_beta | 69,470.0 | 45,284.5 | 24,185.5 |
| sibling_gamma | 69,484.0 | 45,312.5 | 24,171.5 |
| **total** | **277,923.0** | **264,781.5** | **13,141.5** |

Total realized saving over these four turns: **13,141.5 input-token equivalents**, or **4.73%**.

## Verdict

**Proven, conditionally:** Claude did reuse a shared prefix. The three sibling turns each reported `cache_read_input_tokens = 43,995`, so the vCache star-anchor premise is real on Claude Code traffic: a stable prefix can be warmed once and read by later sibling requests.

**Refuted for blanket claims:** this is not "the whole prompt is free after the first turn." Each sibling also wrote about 15.4k new 1h cache tokens, so the exact Claude Code layout only becomes net-positive after the fourth similar request in this run. Three total turns are still below aggregate break-even; four turns are positive.

Design implication: vCache must reconcile every request from provider telemetry and budget the cold/write path until `cache_read_input_tokens` proves the hit. The warm manifest cannot assume the entire prefix hit, and a plan that expects fewer than four similar Claude Code turns in this measured layout should not pre-credit savings.

Sources:

- Anthropic prompt caching docs: https://platform.claude.com/docs/en/build-with-claude/prompt-caching
- Anthropic pricing docs: https://platform.claude.com/docs/en/about-claude/pricing
