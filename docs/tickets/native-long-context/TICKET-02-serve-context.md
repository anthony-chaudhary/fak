<!-- fak-native-context-key: serve-resolved-context-window-v1 -->

# feat(serve): resolve one native context window before model load

## Current state

`fak serve` has three distinct token controls but currently crosses their meanings.
`--context-budget-tokens` is a managed-session lifetime budget and
`--native-admission-token-budget` is a concurrent scheduler resource cap. Yet
`serveFlags.effectiveAdmissionTokenBudget` prefers the session budget and its result also
sizes the GGUF KV plan, CUDA graph capacity, and `--plan-json` artifact. The emitted
`context_budget_tokens` remains `0` when the box auto-sizer chooses a concrete limit.
There is no first-class native model-window flag, no pre-payload validation against the
GGUF model declaration, and no single resolved limit passed to the planner contract in
#12663.

## Classification

- Portfolio tier: 1 — one-touch native serving correctness.
- Centrality: Core.
- Work unit: S1 leaf.
- Priority: P1.

## Problem frame

- P1: advanced - operators can state or automatically obtain one truthful native window.
- P2: advanced - the loader, planner, scheduler, and catalog stop inferring different limits
  from unrelated resource and session-budget knobs.
- P3: preserved - reuse GGUF header metadata, `compute.AutoSizeContextPlan`, selected load
  arms, and the #12663 planner contract.
- P4: preserved - header-only and deterministic memory fixtures prove configuration flow;
  no hardware performance or maximum-context claim is made.

## Parent context

Depends on #12663 for planner enforcement and catalog readback. The closed #9079 scheduler
budget remains a separate resource contract.

## Why this is next

#12663 makes the native planner enforce one context window. This leaf gives `fak serve` a
single pre-load source for that window and keeps the existing scheduler and session knobs
semantically independent.

## Working spine

GGUF model declaration plus selected load arm and memory headroom, or a SafeTensors
`config.json` declaration -> auto or explicit native context resolution -> pre-payload
validation -> KV/load plan -> planner config -> truthful dry-run and effective-config
readback.

## Core through-line

Resolve exactly one native context capacity before reading weight payload bytes and pass the
same value through every model-window consumer. A known model resolves a positive value;
unknown metadata stays unknown rather than fabricating capacity.

## Scope

1. Add `--native-context-tokens N` to `fak serve`; `0` means auto and a positive value is an
   explicit ceiling.
2. From the GGUF header, resolve the model-declared maximum and the existing selected-arm
   `AutoSizeContextPlan` result with current memory headroom, including the ordinary device
   arm's configured device/host weight split. For a SafeTensors directory, read only
   `config.json` and label the result `model-metadata` because it is not memory-qualified.
   Refuse a positive explicit value above a known model maximum before weight payload load.
   Unknown metadata stays explicit.
3. Feed the resolved value to the loader/KV memory estimate, CUDA graph capacity, and
   `agent.InKernelPlannerConfig.ContextTokens` from #12663. When Metal selects the resident
   Q4_K arm, keep the host fit check on that chosen arm instead of reselecting the CPU Q8 arm.
4. Preserve explicitly supplied `--native-admission-token-budget` as the independent
   scheduler resource cap. Preserve `--context-budget-tokens` as the managed-session lifetime
   budget and keep `--ctx` as its session-lifetime alias; document their compatibility and
   stop using either as the model window. Preserve the local measured-admission pipeline:
   automatic local admission starts from the resolved native window while explicit scheduler
   declarations retain precedence.
5. Extend `--plan-json` with the model-declared limit, resolved limit, and source (`auto`,
   `explicit`, or SafeTensors `model-metadata`). Keep `--print-effective-config` non-loading
   and non-networked: it reports the requested value and marks model resolution unresolved.
6. Publish `docs/native-long-context.md` with concise `serve` auto/explicit usage, `up`
   usage, the distinction between model-window, scheduler-admission, and session-lifetime
   controls, and the rule that configured capacity is not hardware qualification.
7. Add focused flag, pre-load refusal, auto-size, propagation, and compatibility tests.
8. Add the existing native-resolution and memory-plan-projection terms to the context
   disambiguation catalog and refresh its generated README and index.

## Gold-plating boundary

Do not modify `fak up`, planner enforcement, gateway catalog formatting, provider proxy
limits, model formats, compute kernels, context compaction, or scheduler policy. Do not add
backend-specific defaults or claim measured speed, memory capacity, or quality.

## Done condition

- [x] Auto mode resolves a positive known limit from model metadata and current sizing inputs.
- [x] An explicit value at or below the model maximum is preserved; one above it is refused
      before payload load.
- [x] Loader sizing, planner config, and CUDA graph capacity consume the same resolved limit.
- [x] Scheduler admission and managed-session lifetime budgets retain independent semantics.
- [x] `--plan-json` identifies model, resolved value, and source; effective config identifies
      the requested value and truthfully marks header resolution unresolved.
- [x] Focused `cmd/fak` tests and scoped boundary/leak checks pass.

## Definition of Done

- [x] One resolved native model window governs every `fak serve` native consumer.
- [x] Existing explicit scheduler and session-budget behavior remains covered by regression tests.
- [x] Unknown model metadata never produces a fabricated advertised limit.

## Acceptance gate

Header fixtures cover auto, explicit-valid, explicit-too-large, and unknown model limits;
the too-large case proves no payload read; plan/config fixtures prove all three token controls
remain distinct. Request-envelope fixtures prove `--max-total-tokens` and
`--max-batch-prefill-tokens` validate against the effective scheduler cap, with explicit
invalid envelopes rejected immediately and local-auto envelopes checked after header
resolution but before payload load. Focused tests and the scoped boundary/leak checks exit zero.

## Closure binding

The DCO-signed resolving commit closes the created public issue and carries `(fak serve)`.

## Witness

Run the focused native-context regression and scoped changed-path validator from
the repository root. Both commands must exit zero.

### Verifiable Witness

```bash
go test -v -count=1 -timeout=120s ./cmd/fak -run '^(TestResolveServeNativeContext|TestResolveServeNativeContextRejectsInvalidExplicitLimits|TestResolveServeNativeContextUnknownMetadataStaysTruthful|TestResolveServeNativeContextDirectoryUsesConfigOnly|TestResolveServeNativeContextAutoCanExceedSchedulerDefault|TestServeNativeContextAutoAdmitsLongNativeHTTP|TestServeNativeContextDeviceSizingMatchesWeightBudgetedLoadPlan|TestServeNativeContextMetalHostFitUsesSelectedResidentArm|TestServeNativeContextKeepsThreeTokenControlsIndependent|TestServePrintEffectiveConfigKeepsNativeContextUnresolvedWithoutModelIO|TestServeNativeContextUnknownAutoFallsBackToSchedulerDefault|TestServeSizingArtifactReportsResolvedNativeContext|TestServeEffectiveAdmissionTokenBudget|TestServeAdmissionTokenBudgetDefault|TestServeAdmissionTokenBudgetExplicitEnvelope|TestServeAdmissionTokenBudgetRejectsNonPositiveAtStartup|TestServeAdmissionMaxTotalTokensOverBudgetFailsFast|TestServeAdmissionTokenBudgetMaterializesResolvedModelWindow|TestServeSizingArtifactCPUArm|TestServeSizingArtifactDeviceArmRefusalBecomesWarning|TestServeSizingArtifactJSONShape|TestServeGGUFMemoryPlanAutoSizesContextToFitWhenNoBudget|TestFitServeGGUFOnDeviceAutoSizeStillRefusesWeightsOverflow|TestApplyDeviceWeightBudgetSplitsWeightsAndLeavesRuntimeDeviceScoped)$'
go run ./cmd/fak-dev gate check --staged
go run ./cmd/fak-dev audit-leak --staged
```

The independent receipt in `.fak/test-witness.json`, produced from base
`7a62f603d13d9ce5f40abe8aafd5bddd68d5610e`, records 24 passing top-level tests for
the focused command (command SHA-256
`ff242ef54f7c2393632ae7d5f85076b2632f58e92d21aefe7e03c663c583c47a`; log SHA-256
`6768bcd706a20309fa30151f676e999e2a67646411167c5244548b1efdd3b842`; test-list
SHA-256 `fe46561155a819f602ee989c79322f5b3f0349e50fc97a102d516077a070b7c7`). Its
`TestServeNativeContextAutoAdmitsLongNativeHTTP` request resolved and admitted a 65,536-token
window, then returned HTTP 200 after 9,019 prompt tokens plus a one-token decode reserve
(9,020 total positions requested) traversed real attention and KV allocation in a one-layer
synthetic CPU-reference model. This is a deterministic software integration witness; it is
not physical-hardware context, throughput, or latency qualification.

The broader scoped `fak validate --mine` attempt checked all 11 overlays and skipped none,
then returned `PARTIAL` when its test phase reached the fixed four-minute timeout. The final
fresh-base landing therefore retains the managed build and boundary gates; the ticket does
not treat that partial run as a passing receipt.

## Likely files

- `cmd/fak/serve.go` (flag registration and token-control separation)
- `cmd/fak/serve_stages.go` (pre-load validation and planner propagation)
- `cmd/fak/serve_load_helpers.go` (selected Metal-arm host fit)
- `cmd/fak/serve_model_fit.go` (resolved context plan)
- `cmd/fak/serve_sizing_json.go` (resolved/model/source fields)
- `cmd/fak/serve_config.go` (effective configuration readback)
- `cmd/fak/serve_help.go` (context-help discovery)
- `cmd/fak/serve_admission_budget_test.go`
- `cmd/fak/serve_context_test.go`
- `docs/native-long-context.md`
- `docs/tickets/native-long-context/TICKET-02-serve-context.md`
- `tools/concept_disambiguation_scorecard.data/rows-context-ctx.json`
- `docs/concept-disambiguation-scorecard/README.md`
- `docs/concept-disambiguation-scorecard/INDEX.md`

## Blast radius and affected lanes

- Affected: native `fak serve` configuration, header-only sizing, and planner construction.
- Unaffected: proxy serves, `fak up`, scheduler algorithm, session budget debit, model kernels,
  and provider context limits.
- Fallback: unknown model metadata remains explicitly unknown; no constant limit is injected.
  SafeTensors auto resolution is metadata-derived and is labeled as not memory-qualified.

## Lane

cmd

## Expected steps

8

```routing
lane: cmd
paths: cmd/fak/serve.go, cmd/fak/serve_stages.go, cmd/fak/serve_load_helpers.go, cmd/fak/serve_model_fit.go, cmd/fak/serve_sizing_json.go, cmd/fak/serve_config.go, cmd/fak/serve_help.go, cmd/fak/serve_admission_budget_test.go, cmd/fak/serve_context_test.go, docs/native-long-context.md, docs/tickets/native-long-context/TICKET-02-serve-context.md, tools/concept_disambiguation_scorecard.data/rows-context-ctx.json, docs/concept-disambiguation-scorecard/README.md, docs/concept-disambiguation-scorecard/INDEX.md
expected_steps: 8
```

## Work estimate

Estimate: 4 points.

## Overall completion contribution

Contribution: 4/4 points.

## Completion standard

development
