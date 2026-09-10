<!-- fak-native-context-key: up-context-plan-propagation-v1 -->

# feat(up): propagate the turnkey context plan into native admission

## Current state

`fak up --context` and `macfit.ConfigureTurnkey` resolve
`TurnkeyProfile.ContextBudgetTokens`, and `startTurnkeyServer` passes that number into the
loader memory fit. It constructs the planner through `NewInKernelPlanner`, however, so the
resolved plan does not reach `InKernelPlannerConfig.ContextTokens`. The turnkey
`/v1/models` response omits context capacity, and every planner failure is returned as HTTP
500 text, including the typed model-window overflow introduced by #12663.

## Classification

- Portfolio tier: 1 — one-touch native serving correctness.
- Centrality: Core.
- Work unit: S1 leaf.
- Priority: P1.

## Problem frame

- P1: advanced - one-touch users receive the context capacity the turnkey planner selected.
- P2: advanced - `fak up` stops loading with one cap while enforcing and advertising another.
- P3: preserved - reuse `TurnkeyProfile.ContextBudgetTokens` and the #12663 planner/error contract.
- P4: preserved - deterministic turnkey fixtures prove propagation and HTTP behavior without
  hardware or performance claims.

## Parent context

Depends on #12663. Coordinates with the separate `fak serve` native context configuration leaf;
neither is start-blocked on the other.

## Why this is next

The turnkey plan already owns a concrete selected context value. Passing it through closes the
smallest one-touch gap without adding another flag or sizing algorithm.

## Working spine

Turnkey hardware/profile plan -> model load estimate -> planner config -> exact request
admission -> OpenAI error mapping and model discovery.

## Core through-line

Use `TurnkeyProfile.ContextBudgetTokens` as the requested ceiling for one `fak up` server
instance, then use the planner's model-clamped `ContextWindow()` as the effective cap reported
to the catalog, ready output, and REPL.

## Scope

1. Convert `TurnkeyProfile.ContextBudgetTokens` to `int` with an explicit overflow check, then
   construct the turnkey native planner with `InKernelPlannerConfig.ContextTokens` set to the
   checked value.
2. Publish the planner's known effective capacity as `data[].context_length` from turnkey
   `/v1/models`; omit it for mock/unknown planners.
3. Map the #12663 typed overflow to HTTP 400 with code `context_length_exceeded` in the
   turnkey OpenAI error envelope; keep unrelated inference failures at 500.
4. Add focused conversion, propagation, catalog, exact-limit, and one-token-over tests around
   `fak up`.

## Gold-plating boundary

Do not change `--context` selection math, `macfit`, `fak serve`, session-lifetime budgets,
loader formats, model kernels, streaming, or scheduler policy. Do not claim performance or
hardware qualification.

## Done condition

- [x] Auto and explicit turnkey plan ceilings reach the planner without integer wrap; the
      model declaration may tighten the effective cap.
- [x] Turnkey `/v1/models` publishes only the known effective context capacity.
- [x] A typed overflow is HTTP 400 `context_length_exceeded`; other failures remain 500.
- [x] Exact-limit and one-token-over turnkey fixtures pass with no token execution on refusal.
- [x] A plan value above the platform `int` range is refused before model loading.
- [x] Focused `cmd/fak` tests pass and the changed files are format/diff clean.
- [ ] Managed landing build and boundary validation pass.

## Definition of Done

- [x] The turnkey load, planner, admission, and model catalog agree on one context cap.
- [x] Mock and unknown-capacity paths remain truthful.

## Acceptance gate

Focused turnkey fixtures prove cap propagation, catalog output, exact-limit admission, and
one-token-over HTTP refusal; scoped validation exits zero.

## Closure binding

The DCO-signed resolving commit closes the created public issue and carries `(fak up)`.

## Witness

The following exact command exercises the offline process bootstrap and every new turnkey
context contract in one deterministic software witness.

### Verifiable Witness

```bash
go test ./cmd/fak -run '^(TestUp(HelpUsesServeSurface|BootsUnifiedAgentRuntime|Bootstrap|InKernelModelExecution|Models.*|Context.*|NonContext.*)|TestTurnkey.*)$' -count=1 -v
fak validate --mine cmd/fak/up.go --mine cmd/fak/up_context_test.go --mine cmd/fak/up_test.go --mine docs/tickets/native-long-context/TICKET-03-up-context.md
```

## Verification result

- The exact isolated command above passes all 11 selected top-level turnkey tests on base
  `96375c472713df95efb73ffc7eadd740c42e8b12`; `.fak/up-context-test.log` has SHA-256
  `ff6558d6b7c7346fef3f7f5a667fcd0c094483ec162c024c002035704923037a`.
- All seven focused context contracts pass: checked conversion, planner propagation,
  model-declaration clamping, truthful known/unknown/mock catalog rows, exact-limit execution,
  one-token-over HTTP 400 refusal, and preservation of HTTP 500 for unrelated inference
  errors.
- The broader turnkey process witness initially reproduced an unchanged-main fixture failure:
  its serve-delegated child supplied `--engine mock --native` but omitted serve's explicit
  `--mock`, so it exited before readiness with the existing no-model refusal. The independent
  test author added only the missing `--mock`; the process witness and the combined command
  then pass.
- The fixture correction does not change production flags or delegation behavior.
- The concept catalog distinguishes the native context refusal from the context
  window and allocator, covering its typed error and HTTP error-code spellings.
- `fak validate --mine` checked all three overlays, then reached its four-minute timeout while
  running the broader test phase. The exact eight-test witness above passes in 15.5 seconds;
  managed landing owns the remaining build and boundary gates.

## Likely files

- `cmd/fak/up.go:366-550` (`startTurnkeyServer`, models and completion handlers)
- `cmd/fak/up_context_test.go`
- `cmd/fak/up_test.go:75` (`TestUpBootsUnifiedAgentRuntime` offline fixture)
- `tools/concept_disambiguation_scorecard.data/rows-context-ctx.json`
- `docs/tickets/native-long-context/TICKET-03-up-context.md`

## Blast radius and affected lanes

- Affected: turnkey native planner construction and OpenAI endpoints.
- Unaffected: `fak serve`, macfit selection, proxy serving, kernels, and session budgets.
- Fallback: mock mode remains execution-free and does not advertise a fabricated native cap.

## Lane

cmd

## Expected steps

4

```routing
lane: cmd
paths: cmd/fak/up.go, cmd/fak/up_context_test.go, cmd/fak/up_test.go, docs/tickets/native-long-context/TICKET-03-up-context.md, tools/concept_disambiguation_scorecard.data/rows-context-ctx.json
expected_steps: 4
```

## Work estimate

Estimate: 2 points.

## Overall completion contribution

Contribution: 2/2 points.

## Completion standard

development
