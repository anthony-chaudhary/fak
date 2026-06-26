---
title: "Agentic benchmark run packets for fak harness wins"
description: "Concrete run packets derived from the fresh agentic benchmark inventory: what can run now, what is hardware-gated, what adapter is missing, and what evidence would prove a fak harness win with Opus-class or GLM-5.2-class models."
---

# Agentic benchmark run packets for fak harness wins

Date: 2026-06-25.

Companion to
[`AGENTIC-BENCHMARK-INVENTORY-2026-06-25.md`](AGENTIC-BENCHMARK-INVENTORY-2026-06-25.md).
This file turns the inventory into execution packets tied to seams that exist in
this repo today. It is deliberately status-tagged: a packet that needs an
adapter is not presented as runnable.

## Status legend

| status | meaning |
|---|---|
| `runnable-now` | the command exists in this tree and does not need a live frontier model |
| `run-contract-shipped` | the pre-run contract artifact exists, but the benchmark result is still gated |
| `hardware-gated` | the run is scripted, but needs the named serving node or external harness |
| `adapter-gap` | the benchmark is high-value, but the repo lacks the adapter that would make the comparison honest |
| `negative-witness` | the command exists only to prove the gap is still real |

## Packet A: AgentDojo structural safety floor

Status: `runnable-now`.

Why first: the existing `cmd/agentdojoredteam` is already a deterministic
AgentDojo-style red-team that scores detection-only versus fak's full stack.
It directly targets the harness win: prompt-injected content may reach the
model, but the sink should still be barred by provenance and policy.

Command:

```powershell
go run ./cmd/agentdojoredteam -json > experiments/agent-live/agentdojo-fak-fullstack-20260625.json
```

Evidence that would count:

- JSON parses.
- `summary.fullstack_attack_successes == 0` or equivalent folded field.
- detection-only attack success is non-zero on adaptive/paraphrased cases, so
  the run proves the full stack, not just a weak fixture.
- every blocked attack carries a closed catch reason such as
  `TRUST_VIOLATION` or `MALFORMED`.

Win claim allowed:

> fak's in-kernel full-stack defense holds ASR at zero on the shipped dynamic
> red-team floor, while detection-only does not.

Win claim not allowed:

> fak beats the official external AgentDojo leaderboard.

That needs an external AgentDojo-compatible adapter and raw model arm.

Local witness from this turn:

```text
go run ./cmd/agentdojoredteam -json
total attacks: 38
detection-only successes: 29
full-stack successes: 0
gate: PASS
```

## Packet B: GLM-5.2 over vLLM live agentic battery

Status: `hardware-gated`.

Why second: this is the existing GLM-5.2 open-weight path. It compares a raw
vLLM GLM-5.2 endpoint against the same endpoint behind `fak serve`, then adds a
20-task SWE-bench Verified slice and fak-native floors.

Manifest command:

```powershell
python tools/glm52_vllm_agentic_battery.py `
  --out experiments/vllm/glm52-agentic-battery/manifest.json `
  --markdown experiments/vllm/glm52-agentic-battery/MANIFEST.md `
  --script experiments/vllm/glm52-agentic-battery/run.sh `
  --swebench-difficulty $env:FAK_SWEBENCH_DIFFICULTY `
  --allow-pending
```

Run command: use the generated
`experiments/vllm/glm52-agentic-battery/run.sh` on a GLM-5.2-capable node.

Evidence that would count:

- `experiments/vllm/glm52-agentic-battery/final-check.json` is complete.
- The final checker emits
  `experiments/vllm/glm52-agentic-battery/BENCHMARK-AUTHORITY-DRAFT.md`.
- The live serving witness is `PASS`.
- The adjudication-tax witness includes measured raw-vLLM and fak-gateway legs.
- SWE-bench compare has two arms, `raw-vllm` and `fak-gateway`, with matching
  selected instance ids and official harness grades.

Win claim allowed:

> same GLM-5.2 endpoint, fak gateway preserves resolve behavior while adding
> policy/result evidence and measured gateway tax.

Win claim not allowed:

> fak native GLM-5.2 is faster than vLLM.

That is a different native-kernel comparison and requires the apples-to-apples
ladder in
[`GLM52-NATIVE-THROUGHPUT-AND-BENCHMARK-PLAN-2026-06-25.md`](GLM52-NATIVE-THROUGHPUT-AND-BENCHMARK-PLAN-2026-06-25.md).

## Packet C: Opus-class SWE-bench smoke through fak

Status: `ready-for-remote-grading-contract-shipped`; the actual Opus raw/fak
run is still gated on executing the raw scaffold, the live Opus gateway, and
official SWE-bench grader output.

Why third: SWE-bench is not the best final headline, but it is the compatibility
control. The point is to show an Opus-class coding agent can run through fak
without losing solve behavior, while recording tool policy and evidence.

Prerequisite:

- `fak serve` or the equivalent gateway is running with an Opus-class model id
  such as `claude-opus-4-8`.
- `FAK_SWEBENCH_DIFFICULTY` points to the official Verified difficulty map, or
  `--dataset` points to a full Verified export.

Current pre-run contract witness:

```powershell
go run ./cmd/fak swebench smoke-contract `
  --difficulty testdata/swebench_smoke.json `
  --model claude-opus-4-8 `
  --out experiments/agent-live/swebench-opus-smoke-contract-20260626.json `
  --md experiments/agent-live/swebench-opus-smoke-contract-20260626.md
```

This fixes the five selected smoke task ids, the shared Opus-class model id, the
raw mini-swe-agent scaffold command, the fak arm command, and the exact
official-grader commands for both arms. The checked-in contract is now
`READY_FOR_REMOTE_GRADING`: both arm commands are present, but this box does not
have the `swebench` Python harness importable. `result_claim_allowed` remains
`false` until raw and fak `predictions.json` files exist and both are graded by
the official harness.

Smoke command:

```powershell
go run ./cmd/fak swebench run `
  --agent fleet `
  --filter smoke `
  --difficulty $env:FAK_SWEBENCH_DIFFICULTY `
  --gateway localhost:8080 `
  --model claude-opus-4-8 `
  --output experiments/agent-live/swebench-opus-fak-smoke-20260625
```

Comparison command after grading or when predictions exist:

```powershell
go run ./cmd/fak swebench compare `
  --difficulty $env:FAK_SWEBENCH_DIFFICULTY `
  --predictions experiments/agent-live/swebench-opus-fak-smoke-20260625/predictions.json `
  --with-adjudication `
  --out experiments/agent-live/swebench-opus-fak-smoke-20260625/compare.json `
  --md experiments/agent-live/swebench-opus-fak-smoke-20260625/COMPARE.md
```

Evidence that would count:

- Predictions are generated by the fleet runner, not `mock`.
- The official SWE-bench harness grades the predictions, or the result is
  explicitly gated with the exact grader command.
- Compare JSON includes adjudication cost, geometry, and resolve-rate family.
- A raw Opus arm on the same selected ids exists before any "fak improved
  solve rate" claim is made.

Win claim allowed:

> fak is compatible with an Opus-class coding-agent run and records the kernel
> metrics needed for safe-pass and cost comparison.

Win claim not allowed:

> fak improves SWE-bench resolve rate.

That requires the raw Opus arm on identical task ids and budget.

## Packet D: DeepSWE external coding benchmark

Status: `adapter-fixture-witness-shipped`; real DeepSWE/R2E-Gym execution is
still gated on an external adapter and task run.

Why it matters: DeepSWE is fresher than SWE-bench Verified and already reports
cost, output tokens, and steps. It is a clean place to test whether fak as a
harness improves safe pass per dollar with the same base model.

Current tree evidence:

- `internal/swebench.RunnerDeepSWE` exists.
- `deepSWERunner.RunInstance` shells to a configured adapter instead of returning
  the old `deepswe runner not wired` placeholder.
- Adapter selection is `FAK_DEEPSWE_RUNNER` plus optional
  `FAK_DEEPSWE_RUNNER_ARGS`, or a `fak-deepswe-runner` executable in
  `DeepSWERepo`.
- The adapter receives `fak.swebench.deepswe-request.v1` JSON on stdin and must
  emit canonical SWE-bench prediction JSON or a unified diff on stdout.
- `cmd/fak-deepswe-runner --fixture` is a checked-in contract fixture adapter.
  It is not a model runner; it exists so the `fak swebench run --agent deepswe`
  execution path can be witnessed end to end without fabricating a DeepSWE
  score.
- `experiments/agent-live/deepswe-adapter-smoke-20260626/` records that witness:
  two fixed SWE-bench smoke ids, `2/2` adapter invocations done, canonical
  `predictions.json` rows emitted, and official eval honestly gated on this box.

Configured witness command shape:

```powershell
$env:FAK_DEEPSWE_RUNNER = "C:\path\to\fak-deepswe-runner.exe"
$env:FAK_DEEPSWE_RUNNER_ARGS = "--single-pass"
go run ./cmd/fak swebench run `
  --agent deepswe `
  --filter smoke `
  --difficulty $env:FAK_SWEBENCH_DIFFICULTY `
  --model http://127.0.0.1:8000/v1 `
  --preds-only `
  --output experiments/agent-live/deepswe-gap-witness-20260625
```

Evidence that would count now:

- With no adapter configured, the run fails closed with `deepswe adapter not
  configured`, not a synthetic placeholder patch.
- With an adapter configured, the run emits `predictions.json` entries whose
  `instance_id` matches the requested SWE-bench task id and whose `model_patch`
  is adapter-authored.
- The fixture witness counts only for adapter plumbing and grader-consumable
  prediction shape. It does not count as pass@1, safe-pass, cost, output-token,
  or step evidence for DeepSWE itself.

Adapter completion bar:

- Point the contract at a real DeepSWE/R2E-Gym single-pass runner that accepts
  the model/base URL and emits SWE-bench-compatible predictions.
- Run the same task ids through raw DeepSWE and `fak serve`.
- Report pass@1, cost, output tokens, steps, safe-pass, and kernel verdicts.

Win claim allowed after adapter:

> with the same DeepSWE task ids and model budget, fak improves safe pass per
> dollar or evidence coverage without lowering pass@1.

## Packet E: tau3 and ToolSandbox policy/state benchmarks

Status: `adapter-smoke-shipped`; official tau3/ToolSandbox harness execution is
still gated.

Why high priority: these benchmarks are closer to fak's core contract than
coding pass@1. They test stateful tool use, policy adherence, insufficient
information, and minefield avoidance.

Adapter completion bar:

- Map benchmark tools into fak's tool registry.
- Run raw tool-calling and fak-adjudicated tool-calling with the same base
  model and user simulator.
- Record pass^1/pass^k, policy breach rate, minefield hits, argument repairs,
  denied tool calls, and user turns.

Current local witness:

```powershell
go run ./cmd/toolsandboxbench `
  -suite testdata/toolsandbox/policy_state_smoke.json `
  -out experiments/agent-live/toolsandbox-policy-state-smoke-20260625.json `
  -md experiments/agent-live/toolsandbox-policy-state-smoke-20260625.md
```

This is a ToolSandbox-shaped adapter smoke, not an official Apple ToolSandbox or
tau3 leaderboard run. It preserves task ids, milestones, minefield labels, and a
shared trace while replaying the same calls through raw and fak arms. The shipped
smoke records raw safe pass^1 `0.500` versus fak safe pass^1 `1.000`, with one
fak-denied policy/minefield call and benign utility preserved on both tasks. The
regenerated witness also carries `evidence_class=SIMULATED_LOCAL_FIXTURE`, an
unavailable `official_harness` gate, explicit promotion requirements, and
`result_claim_allowed=false`.

Remaining official-harness bar:

- Replace the smoke suite with externally supplied tau3 or ToolSandbox task
  definitions/grader output under their licenses.
- Keep the raw and fak arms on the same model, simulator, task ids, and budgets.
- Promote only official-harness results into `BENCHMARK-AUTHORITY.md`.

First model arms:

- `claude-opus-4-8` for ceiling.
- `zai-org/GLM-5.2-FP8` behind vLLM for open-weight long-context control.

Win claim allowed:

> fak improves policy-correct completion or pass^k reliability at equal model
> and task ids by structurally denying off-policy tool calls and preserving
> state evidence.

## Packet F: Terminal-Bench command-policy benchmark

Status: `adapter-smoke-shipped`; official Terminal-Bench harness execution is
still gated.

Why useful: Terminal-Bench exercises the shell/file boundary that fak already
guards. Resolve rate is model-dominated, but safe-resolve and command evidence
are harness surfaces.

Adapter completion bar:

- Run Terminal-Bench inside its official environment.
- Insert fak at the command/tool boundary, not as a post-hoc log parser.
- Preserve the official task tests as the success oracle.
- Emit per-command verdicts, denied commands, filesystem scope, elapsed time,
  and task resolution.

Current local witness:

```powershell
go run ./cmd/terminalbench `
  -suite testdata/terminalbench/command_boundary_smoke.json `
  -out experiments/agent-live/terminalbench-command-boundary-smoke-20260625.json `
  -md experiments/agent-live/terminalbench-command-boundary-smoke-20260625.md
```

This is a Terminal-Bench-shaped adapter smoke, not an official Terminal-Bench
run. It replays recorded command traces as `terminal.exec` fak tool calls,
keeps recorded test-oracle fields in the report, and emits solve, safe-resolve,
blocked-dangerous-action, unnecessary-block, and command-evidence metrics. The
shipped smoke records raw safe resolve `0.500` versus fak safe resolve `1.000`,
with one fak-denied destructive command and no unnecessary fak blocks.

Remaining official-harness bar:

- Replace the smoke suite with benchmark-native Terminal-Bench task ids,
  images/environments, command logs, and official test output.
- Keep raw and fak arms on the same model, task ids, environment image, budget,
  and retry policy.
- Link the official test output to the fak per-command verdict/evidence log.

Win claim allowed:

> same model and task ids, fak preserves or improves resolve rate while
> reducing unsafe command execution and adding replayable command evidence.

## Packet G: Browser and computer-use safe success

Status: `adapter-smoke-shipped`; official browser/computer-use harness execution
is still gated.

Targets:

- OSWorld / OSWorld-Verified.
- WorkArena / BrowserGym.
- WebArena / VisualWebArena / VideoWebArena.
- BrowseComp for evidence-required research answers.

Why lower than safety/tool packets: the base model controls visual grounding and
UI planning. fak's first win is safe success, state checkpoints, evidence, and
action policy, not raw success-rate jumps.

Adapter completion bar:

- Browser/desktop actions become adjudicated tool calls.
- Mutating actions carry authority labels.
- Page/tool content can be quarantined before model-visible context.
- State checkpoints are saved per task.
- Final answers require evidence read-back where the benchmark permits it.

Current local witness:

```powershell
go run ./cmd/browseractionbench `
  -suite testdata/webbench/action_mediation_smoke.json `
  -out experiments/agent-live/browser-action-mediation-smoke-20260625.json `
  -md experiments/agent-live/browser-action-mediation-smoke-20260625.md
```

This is a browser-action adapter smoke, not an official WebArena, OSWorld,
WorkArena, BrowseComp, or BrowserGym run. It normalizes benchmark-style browser
actions into `browser.*` fak tool calls and emits per-action evidence
checkpoints while replaying the same action trace through raw and fak arms. The
shipped smoke records raw safe pass^1 `0.500` versus fak safe pass^1 `1.000`,
with one fak-denied destructive click and benign task utility preserved.

Remaining official-harness bar:

- Replace the smoke suite with benchmark-native task and action traces under the
  upstream benchmark's license.
- Keep raw and fak arms on the same model, browser state, task ids, and budgets.
- Link benchmark-native grader output to the fak action/evidence log.

Win claim allowed:

> fak adds safe-action and evidence guarantees to browser/computer-use runs,
> with raw success rate preserved or improved on identical task ids.

## Priority order

1. Run Packet A now and land the JSON as a local fak-owned structural safety
   floor.
2. Keep Packet B as the GLM-5.2 open-weight live-serving battery for the
   serving node.
3. Run Packet C once an Opus gateway is available.
4. Implement Packet D's DeepSWE adapter before using DeepSWE as a public
   coding headline.
5. Implement Packet E before chasing more SWE-bench variants, because policy
   and state are where the harness should win most clearly.
6. Add Packet F once the command-boundary adapter can live inside the official
   Terminal-Bench environment.
7. Add Packet G after browser/desktop action traces are normalized as fak tool
   calls.

## Commit fence

These packets are docs and run sheets. They are not results. A result is
quotable only when:

- the artifact is parseable JSON;
- the raw and fak arms use identical task ids and budgets;
- the benchmark's official grader or task oracle is preserved;
- `BENCHMARK-AUTHORITY.md` links the artifact and reproduce command;
- the claim says `safe-pass`, `policy-breach`, `cost`, `latency`, or
  `evidence` explicitly, rather than implying a generic model-quality win.
