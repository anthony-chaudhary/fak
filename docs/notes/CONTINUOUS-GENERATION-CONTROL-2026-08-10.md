# Continuous generation control: trajectory, epoch, and steering point

**Status:** working spine (#6219)  
**Date:** 2026-08-10

The request/response boundary is a transport convenience, not the natural unit
of agent work. As sessions become continuous, generation should be modeled as a
sequence of replaceable decoding spans inside one durable trajectory.

## Durable vocabulary

| Concept | Meaning | Durable home |
|---|---|---|
| **Trajectory** | Long-lived intent plus admitted state/effects. It survives turns, model calls, workers, and devices. | `internal/trajectory` for records and outcome analysis; session journal for durable identity. |
| **Generation epoch** | One contiguous decode span owned by one micro-agent and placed on one worker/model/device. A request usually carries an epoch today, but they are not synonymous. | `internal/generationctl.Epoch`. |
| **Steering point** | An ordered boundary where output observed so far is classified, admitted prefix is checkpointed, and a typed directive is applied. Token boundaries are the theoretical lower bound; provider/tool syntax and effect admission determine the safe practical boundary. | `internal/generationctl.Transition`; later an additive ABI event when wired into the kernel event stream. |
| **Directive** | `continue`, `redirect`, `fork`, `yield`, or `stop`. These describe trajectory control rather than provider-specific cancellation mechanics. | `internal/generationctl.DirectiveKind`. |
| **Checkpoint** | Durable admitted prefix and trajectory identity at an epoch boundary. Speculative tool arguments and unadmitted effects do not belong in it. | `internal/generationctl.Checkpoint`; later journal persistence. |
| **Compute placement** | Replaceable worker/model/device executing an epoch. It is not the owner or the trajectory. | `internal/generationctl.Compute`; later the flexible-compute scheduler contract. |
| **Micro-agent owner** | The narrow role accountable for an epoch. Ownership may remain stable across compute migration or change at a handoff. | `Epoch.Owner`; role registries remain outside this leaf. |

This separation is the key connection: **micro-agents are control-plane owners;
compute is placement; generation epochs are schedulable execution; the
trajectory is continuity.** Treating all four as a “request” makes live steering,
model escalation, migration, and safe forks unnecessarily coarse.

## Semantics

An epoch emits ordered deltas. Deltas can be observed at token granularity, but
not every token is a safe commit point. Text accepted into the trajectory prefix
is durable. Partial tool arguments remain speculative until admitted. A live
steer therefore follows:

1. observe a delta;
2. classify it against current policy, operator input, budgets, and trajectory;
3. continue, or stop at the next safe boundary;
4. checkpoint only admitted prefix/effects;
5. redirect/fork/yield/stop;
6. optionally resume as a new epoch under another owner or compute placement.

Already-sampled tokens are history, not mutable state. “Live” means bounded
control latency and cheap epoch replacement, not retroactive token editing.
Provider capabilities set the realized granularity:

- token/logit hooks: potentially token-level steering;
- streaming cancellation: prefix-level stop and restart;
- tool-call delta hooks: argument-level interruption before an effect;
- non-streaming APIs: request boundary only, while preserving the same contract.

The policy is provider-neutral; adapters report their steering resolution rather
than silently weakening the meaning.

## Working spine

`internal/streamrules` already recognizes rule matches over streaming tool-call
argument deltas. `internal/generationctl` now composes that primitive with epoch
identity, accepted-prefix checkpoints, typed directives, micro-agent ownership,
and compute handoff.

Run the captured local witness:

```bash
go run ./cmd/generationcontroldemo -selfcheck
```

It starts a planner epoch on CPU, observes a destructive shell tool call while
its arguments stream, redirects before the partial call becomes an effect,
checkpoints the accepted text, and resumes the same trajectory as a safety
micro-agent on an L4 placement. The trace ends with
`SELF_CHECK_PASS continuous_generation_redirect_handoff`.

This is intentionally a deterministic spine, not a claim that a live provider
adapter is shipped. The next product seam is to connect gateway/provider stream
callbacks and the journal to this contract, then measure steer latency and
wasted speculative tokens by adapter capability.

## Placement rules for future named concepts

- If it changes the provider-neutral control state machine, it belongs in
  `internal/generationctl`.
- If it recognizes incremental tool arguments, it belongs in
  `internal/streamrules`.
- If it measures or stores whole-run outcomes, it belongs in
  `internal/trajectory`.
- If it is an event every kernel observer must understand, add it to
  `internal/abi` only when the live kernel emits it; do not reserve aspirational
  ABI names.
- If it chooses worker/model/device, it belongs in the compute scheduler and
  consumes `Compute`; it must not own trajectory semantics.
- Product promises and capability-resolution tables belong in `docs/supported/`;
  dated design reasoning and experiment results belong in `docs/notes/`.

## Non-conflations

- Continuous session does not require one immortal provider request.
- Token-level observability does not imply token-level mutation.
- Cancellation is a mechanism; redirect/fork/yield/stop are semantics.
- A micro-agent is not necessarily a process or model call.
- Compute migration must not create a new trajectory.
- A generated tool call is not an effect until the kernel admits and executes it.
