# Tool-call control spine — 2026-08-17

## Value frame

- **Centrality:** Core. FAK handles every tool call; avoiding calls before execution is on its primary performance seam.
- **For:** long-running coding and research agents whose prompt is already large.
- **Problem:** a low-value tool call pays not only tool latency, but another model continuation over the accumulated context.
- **Today:** prompts can ask the model to be frugal, while existing runtime telemetry mostly explains calls after they happen.
- **Better because:** a deterministic prefilter can reuse exact fresh observations, coalesce independent reads, and defer weak speculative reads before execution.
- **Witness:** `go run ./cmd/toolcallcontroldemo -selfcheck -pretty` emits per-call decisions and a same-trace control/prefilter ablation.

All four problem checks apply: this reduces managed context replay (P1), reports net-true call and context-token effects without calling a proxy measured cost (P2), admits state changes and avoids suppressing mutations (P3), and emits machine-readable decisions plus arm metrics (P4).

## Minimal working spine

The model-side instruction is deliberately short: before calling, name the missing evidence and the decision it can change; reuse fresh results; batch independent reads; stop when evidence is sufficient. At 64k tokens and above it explicitly identifies the long-context replay penalty.

The prefilter then checks claims the kernel can witness:

1. **Exact fresh repeat:** `(tool, normalized args, state epoch)` matches a prior observation → `reuse`.
2. **Batchable read:** read-only proposals share a batch key → allow one deterministic leader and merge followers into it.
3. **Speculative read:** no evidence gap, no changed decision, and low stated information gain → `defer`.
4. **Safety boundary:** mutations are not deferred on weak model rationale; novel and required calls remain allowed.

This first spine intentionally avoids semantic similarity and learned gating. False-positive suppression is more expensive than a redundant call, so broader gates need trace-backed calibration first.

## Ablation contract

Every arm runs over the same independently labeled proposals. Reports keep these quantities separate:

- calls executed;
- unneeded calls avoided;
- needed calls suppressed (the false-positive guardrail);
- context tokens whose replay was avoided;
- `attention_proxy_avoided = Σ context_tokens²`, explicitly a quadratic exposure proxy—not provider FLOPs, latency, dollars, or measured savings.

The captured 128k-context demo executes 5 calls in control versus 2 continuations in the prefilter arm, avoids 2 independently labeled unnecessary calls, and suppresses 0 needed calls. Batching merges one needed read but does not count it as suppression.

## Recent GitHub field scan (2026-08-03 through 2026-08-17)

GitHub search was run for recently created/popular agent tool-calling and context-engineering repositories, plus recent commits/issues in established agent projects. Search results were noisy; popularity alone did not identify a shipped, general unnecessary-call gate. The strongest transferable work was in actively developed OpenAI Codex:

- [`openai/codex#38978`](https://github.com/openai/codex/pull/38978), merged 2026-08-17: caps skill-catalog context at a configurable budget (default remains 2% of window, hard cap 10k). **Borrow:** bound model-visible catalogs instead of assuming retrieval is free. **FAK disposition:** follow-on, not this gate.
- [`openai/codex#39008`](https://github.com/openai/codex/pull/39008), merged 2026-08-17: task-context fusion for short continuation requests, bounded to two prior substantive requests and recent relevant skills, evaluated as a shadow selector. **Borrow:** use bounded task state rather than full transcript, and shadow new selectors before enforcement. **FAK disposition:** default methodology for learned/semantic call gating.
- [`openai/codex#38993`](https://github.com/openai/codex/pull/38993), merged 2026-08-17: recent-skill and character-routing variants evaluated in shadow mode with deterministic ranking and truncation metadata. **Borrow:** compare multiple selectors on identical traffic and preserve truncation metadata. **FAK disposition:** follow-on ablation arm design.
- [`openai/codex#38980`](https://github.com/openai/codex/pull/38980), merged 2026-08-17: bounds parent-compaction context and fails closed when oversized. **Borrow:** context admission needs explicit byte/token limits and an observable oversized outcome. **FAK disposition:** follow-on for evidence/result reuse payloads.
- [`openai/codex#38921`](https://github.com/openai/codex/pull/38921), merged 2026-08-17: groups successful command activity while preserving the full transcript and flushes at failures/boundaries. **Borrow:** compact observability without deleting drill-down evidence. **FAK disposition:** report/UI follow-on.

New repositories with the largest stars in the exact two-week creation window were mostly application wrappers (for example `ysr666/dsh-vision-router`, 615 stars when observed) rather than evidence about unnecessary-call prevention. That negative result matters: this spine borrows tested mechanisms from active upstream changes, not popularity theater.

## Reproduction

```bash
go test ./internal/toolcallcontrol ./cmd/toolcallcontroldemo
go run ./cmd/toolcallcontroldemo -selfcheck -pretty
```

The demo is offline, deterministic, and performs no real tool execution.


---

# Turn-cost control: charge the thought, not only the tool

Date: 2026-08-17
Status: working spine shipped under #7150; expansion backlog is issue-backed

## Result first

A tool-call controller is useful only when the work of deciding costs less than the downstream work it prevents. A model instruction that asks the agent to deliberate can create the very continuation it is meant to save. Therefore FAK should prefer constant-time checks before model reflection, charge every controller invocation, preserve cost as a vector until an operator supplies an exchange rate, and optimize expected task utility over a horizon rather than raw call count.

The first spine adds controller and false-suppression recovery units to identical-trace replay. It computes:

```
gross_saved = sum(replay units for suppressed proposals)
net_replay_value = gross_saved - controller_units - false_suppression_recovery_units
break_even = net_replay_value >= 0
```

`cost_basis=observed|scenario` states whether the charges came from a trace or a counterfactual. Replay units and their square remain exposure proxies, never provider billing, latency, FLOPs, or dollars.

## The three costs that previous call-count accounting missed

1. **Control cost:** prompt replay, generated reasoning, latency, and model/provider charge used to reach the gate decision. A separate model turn is usually too expensive; deterministic checks at the existing tool-hook seam are preferred.
2. **Action cost:** tool runtime, API price, side effects, result ingestion, and the continuation that follows the tool result. Avoiding a cheap local read at the price of another 128k-context model continuation is a loss.
3. **Error and horizon cost:** a false suppression can cause retries, stale answers, task failure, or delayed evidence. Conversely, one apparently redundant call can reduce uncertainty and prevent many future turns. The objective is expected downstream utility, not minimum calls this turn.

Never add unlike dimensions without an explicit valuation. Preserve this cost vector first:

```
C = {
  billed_input_tokens, cached_input_tokens, output_tokens,
  wall_ms, tool_ms, tool_usd, provider_usd,
  replay_units_proxy, replay_square_proxy,
  recovery_units, task_success, evidence_quality
}
```

Then allow an operator- or workload-specific scalarization:

```
J(policy) = E[task_value]
          - lambda_usd * E[usd]
          - lambda_time * E[wall_ms]
          - lambda_risk * E[failure_or_recovery]
          - lambda_context * E[replay_exposure]
```

Without declared lambdas, report a Pareto frontier rather than one invented score.

## Short term: make the cheap path actually work

- Run exact fingerprint reuse, freshness/epoch checks, mutation invalidation, and closed-set read classification in the existing hook. These do not consume a model turn.
- Put the naive instruction in setup/context already being sent; do not ask the model for a separate tool-necessity essay.
- Charge instruction-only and learned/judge arms for their measured input/output usage. Charge deterministic arms for measured CPU/wall time when available, but do not pretend that microseconds and tokens are interchangeable.
- Add per-turn decision IDs so control telemetry joins the model request, proposed call, hook decision, eventual result, and retry/recovery chain.
- Gate enforcement on `needed_suppressed == 0` plus positive net value by context bucket. Gross avoided calls alone cannot promote a policy.
- Join authoritative provider cost through #6651 rather than estimating dollars from tokens.

## Middle ground: bounded adaptive control

- Learn or calibrate a tiny admission policy from shadow traces using features available before execution: exact-repeat fingerprint, freshness, state changes, tool class, context bucket, prior success, expected result size, and task phase.
- Use a cascade: deterministic allow/reuse first; a cheap classifier only for the ambiguous residue; expensive model deliberation only when expected value of information exceeds its own cost.
- Estimate counterfactuals with randomized shadow holdouts or paired replay. Avoid training only on the controller's own historical decisions, which creates selective-label bias.
- Calibrate per workload and tool family. A repo read, paid search API, destructive mutation, and GPU job have different loss surfaces.
- Use conformal/risk bounds or abstention: uncertainty should route to execution, not aggressive suppression.
- Optimize a constrained objective: minimize cost subject to task-success/evidence-quality floors and a hard false-suppression budget.

## Long term: turn economics as a research program

The natural unit is not a tool call but a partially observed decision process over a trajectory. At state `s`, choose among answer, stop, tool, batch, reuse, compress, delegate, or think. Each action changes context, information, cacheability, risk, and future action space.

Research directions:

- **Metareasoning / value of computation:** invoke more reasoning only when expected decision improvement exceeds computation cost.
- **Value of information:** price a tool by how much it can change the final decision, not by novelty of its output.
- **Semi-Markov control:** actions have variable latency and duration; optimize cost-to-evidence and cost-to-success over whole trajectories.
- **Budgeted POMDPs:** uncertainty, hidden world state, and partial observations make one-turn heuristics insufficient.
- **Optimal stopping:** decide when evidence is sufficient and quantify the option value of one more observation.
- **Cache-aware scheduling:** account for prompt-cache survival, compaction boundaries, and the nonlinear cost of a continuation at long context.
- **Multi-agent economics:** compare delegation overhead, duplicated setup, communication, and independent-witness value against one-agent continuation cost.
- **Causal evaluation:** estimate what would have happened under the call that was suppressed; replay alone cannot observe this.
- **Preference and task value:** model correctness, proof quality, safety, and operator time alongside provider spend.
- **Mechanism design:** prevent agents from gaming call-count or token targets by degrading evidence or splitting work differently.

A mature turn-cost function will be workload-conditioned, horizon-aware, uncertainty-calibrated, and provenance-honest. It should produce policy frontiers and confidence intervals, not a universal magic scalar.

## External bearings and local boundaries

The 2026-08-17 field scan in `TOOL-CALL-CONTROL-SPINE-2026-08-17.md` found the strongest recent transferable mechanisms in active OpenAI Codex work: bounded context budgets, shadow-evaluated selectors, deterministic ranking/truncation metadata, bounded compaction context, and compact display with full transcript retention. Those support a bounded cascade and honest shadow evaluation; they do not establish a universal turn-cost function.

The broader intellectual lineage is metareasoning/value-of-computation, value-of-information, optimal stopping, and budgeted sequential decision-making. FAK should borrow the structure while grounding every quantity in its own request, hook, usage, billing, and outcome ledgers.

Local dependency boundaries:

- #6651 owns authoritative session-keyed provider billing.
- #6874 owns spend attribution to work class and witnessed outcome.
- `internal/toolcallcontrol` owns pre-execution decisions and same-trace ablation.
- This program must join those surfaces; it must not duplicate their ledgers or call proxies money.

## Working witness

The #7150 fixture has two 100-unit turns. Exact reuse avoids 100 replay units on the second proposal, but its controller is charged 80 units on each turn. Gross savings are `100`; controller overhead is `160`; net replay value is `-60`; break-even is `false`. This is the failure mode the operator identified: avoiding a call is not a gain when thinking about avoidance costs more than the call path.
