# Turn avoidance as a first-class outcome

**Outcome:** count a whole model turn as avoided only when a candidate commits fewer
model turns than its control, reaches the same independently observed required effects,
and records the lifecycle as realized. Faster retained turns, fewer tool calls, parallel
work, and speculative overlap remain valuable, but they stay in separate accounts.

**Observed at:** `2026-08-24T13:08:51-07:00`

**Issues:** [#8791](https://github.com/anthony-chaudhary/fak/issues/8791) defines the
first-class outcome and longer program. [#8796](https://github.com/anthony-chaudhary/fak/issues/8796)
is its offline replay/ledger spine. This note records the evidence and terminology for
that spine; it does not claim live prediction or enforcement.

## Product frame

- **For:** operators of long-lived, tool-using agents whose model round trips compound
  latency, spend, context growth, and failure exposure.
- **Problem:** adjacent counters cannot show that a complete model/tool round trip became
  unnecessary while the task still reached its required effects.
- **Today:** application-specific workflow logic is combined with token, cache, latency,
  and tool-call metrics that are useful individually but easy to conflate.
- **Better because:** one versioned replay ledger compares identical immutable rows,
  preserves mechanism and lifecycle provenance, charges overhead, and refuses realized
  credit without an independent equivalent-effect witness.
- **Witness:** replay a labeled trace through control, exact-reuse, and fused-batch arms;
  prove fewer committed model turns only for equivalent realized outcomes, zero credit for
  a cheaper retained turn or unsafe suppression, and net cost after all overhead.

## Accounting boundary

| Account | What changed | Whole model turns avoided? |
|---|---|---:|
| Realized whole-model-turn avoidance | The candidate commits fewer model inferences than control, lifecycle is `realized`, and independently observed required effects are equivalent. | Yes: `max(control_committed - candidate_committed, 0)`. |
| Retained-turn acceleration | The same model turn uses fewer tokens or less latency through prompt/KV/provider-cache reuse, routing, or transport improvements. | No. |
| Tool-call avoidance | An external tool invocation is locally answered, reused, denied, or made unnecessary. | No, unless a paired trace also proves that a model turn disappeared. |
| Fusion or batching | Tools share one assistant turn, execute as a dependency-aware batch, or have lower serial depth. | Only the observed reduction in committed model turns; tool calls and effects remain separately counted. |
| Speculation | Work starts before the authoritative decision and may later commit or be discarded. | No by itself. Charge prediction, validation, rejected work, recovery, and rollback; credit only a separately witnessed later model turn that never executes. |
| Counterfactual opportunity | Replay says a different arm could have saved work, but that arm did not realize the outcome. | No. |

An avoided public API request is also not necessarily an avoided model turn: accounting
belongs below transport and orchestration boundaries. An unknown mechanism or lifecycle
state remains unknown or is rejected explicitly; it is never silently coerced to realized.

## Dated primary-source ledger

All source prose below is paraphrased. `source_event_at` is the publication, revision, or
commit time, not the observation time above.

| Source | `source_event_at` · state · anchor | Platform and verified fact | Refresh trigger |
|---|---|---|---|
| LLMCompiler paper | 2024-07-21..27 · published ICML 2024 · [PMLR 235:24370–24391](https://proceedings.mlr.press/v235/kim24y.html) | Its planner, task-fetching unit, and executor parallelize function calls; the official abstract reports latency and cost gains against ReAct. This proves a scheduling result, not a universal reduction in calls or model turns. | None known; revisit if a corrected proceedings version appears. |
| LLMCompiler implementation | 2024-07-10 · shipped repository state · [`task_fetching_unit.py:124-135@a00c9d35507507da70e8c637eee64efc8c1857ae`](https://github.com/SqueezeAILab/LLMCompiler/blob/a00c9d35507507da70e8c637eee64efc8c1857ae/src/llm_compiler/task_fetching_unit.py#L124-L135) | Python/`asyncio`; the scheduler launches every dependency-ready task concurrently. The pinned code proves dependency-aware concurrency, not model-turn removal. | A new repository commit that changes scheduler semantics. |
| CodeAct | 2024-07-21..27 · published ICML 2024 · [paper](https://proceedings.mlr.press/v235/wang24h.html), [author poster](https://xwang.dev/assets/pdf/posters/2024-codeact.pdf) | Executable Python consolidates and composes actions. The author poster reports up to 30% fewer turns on M3ToolEval; that is benchmark-specific evidence that programmatic execution can collapse model/tool alternation, not a generic saving. | None known; rerun on fak workloads before making a product claim. |
| GPT Semantic Cache | 2024-12-09 · revised preprint v3 · [arXiv:2411.05276v3](https://arxiv.org/abs/2411.05276v3) | Redis-backed semantic matching can return a pre-generated response without a redundant LLM API call. Its self-reported hit quality is not an independent required-effect witness, so it supports a bypass mechanism, not automatic realized credit. | A new paper revision or an independent outcome-equivalence evaluation. |
| Prompt Cache | 2024 · published MLSys 2024 · [conference paper](https://proceedings.mlsys.org/paper_files/paper/2024/hash/a66caa1703fe34705a4368c3014c1966-Abstract-Conference.html) | Reusing attention state improves time to first token while inference still runs. This is retained-turn acceleration and receives zero whole-turn credit. | None known; implementation changes affect performance, not the accounting class. |
| Speculative Actions | 2026-04-23 · revised preprint v2 · [arXiv:2510.04371v2](https://arxiv.org/abs/2510.04371v2) | A faster model predicts future actions, executes them in parallel, and commits matching predictions. The source reports latency reduction, but the predicted action still executes; speculation is overlap until a later operation is independently proven absent. | A new paper revision or changed commit/rollback semantics. |
| τ-bench | 2024-06-17 · preprint v1 · [arXiv:2406.12045v1](https://arxiv.org/abs/2406.12045v1) | Tool-agent conversations are judged by final database state against an annotated goal, with repeated-run reliability. Borrow the independent goal-state witness: a shorter dialogue is not successful when its required effects or policy compliance differ. | A benchmark revision that changes its goal-state or reliability protocol. |
| OpenAI programmatic tool calling | source update time not exposed · current shipped documentation, observed at the time above · [vendor guide](https://developers.openai.com/api/docs/guides/tools-programmatic-tool-calling) (mutable) | Generated JavaScript runs in an isolated V8 runtime and can coordinate loops, conditions, and parallel tools. The guide favors direct calls when fresh semantic judgment, writes/approval, or final citations matter. | Documentation, runtime, or model-availability change. |
| Anthropic programmatic tool calling | source update time not exposed · current shipped documentation, observed at the time above · [vendor guide](https://platform.claude.com/docs/en/agents-and-tools/tool-use/programmatic-tool-calling) (mutable) | Code can call several tools without re-sampling the model between them; the vendor gives an `N`-to-one model-round-trip example and vendor-authored evaluation. Nested tools still execute, and workload-specific vendor results are not fak evidence. | Documentation, code-execution runtime, or model-availability change. |

## Fact, inference, and disposition

| Source fact | fak inference | Spine disposition |
|---|---|---|
| An accepted cached response can bypass an LLM API call. | Exact reuse is the cleanest initial bypass arm; semantic reuse needs stronger identity, freshness, policy, state, and outcome witnesses. | **Borrow:** exact-reuse arm. **Watch:** semantic reuse. |
| Programmatic code can collapse repeated model/tool alternation. | Fusion may remove model turns while retaining all nested tool calls and effects. | **Borrow:** fused-batch arm, crediting only the committed-model-turn delta. |
| A dependency scheduler runs calls concurrently. | Lower serial depth is valuable but is not proof that any logical call or model turn disappeared. | **Reject as turn credit:** report batching/latency separately. |
| Attention or provider-cache state accelerates a model invocation. | A cheaper retained inference remains a committed model turn. | **Reject as turn credit:** preserve retained-turn token/latency fields. |
| Speculation executes predicted work before authoritative validation. | Hidden or discarded work can improve latency while increasing total cost and risk. | **Reject as turn credit by itself:** charge all overhead and waste. |
| A final external state can be compared with an independently annotated goal. | Required-effect equivalence can gate realized credit without letting the candidate grade itself. | **Borrow:** independent control/candidate effect labels and suppressed-effect accounting. |

## Boundary from existing fak surfaces

Repository claims below were independently reread at
`56e90bf37ba043b69eb6c58c90defcea75665653`; transient module revision labels from an
earlier research checkout are intentionally omitted.

- `internal/toolcallcontrol/replay.go:14-35,104-225` already supplies ordered,
  independently labeled rows, decoding/validation, identical-input replay arms, and the
  earlier-turn-only observation rule that prevents same-turn result leakage. It measures
  tool calls, needed suppression, and replay units—not committed model-turn equivalence.
- `internal/callavoid/doc.go:1-47` and `internal/callavoid/fold.go:27-74` account for
  avoiding local tool dispatch, memo economics, repair, and witnessed productive denies.
  They are the tool-call dual, not the paired whole-model-turn ledger.
- `internal/turnbench/turnbench.go:1-54,148-220` replays frozen tool-call traces through
  the real kernel and prices specialized grammar/vDSO turn tax against a transparent
  baseline. It lacks the candidate/control required-effect label pair this spine needs.
- `internal/cachemeta/provider.go:98-130` makes provider-cache evidence non-serveable and
  explicitly `cost_latency_only`. Provider-cache hits therefore remain retained-turn
  savings.
- `internal/savingsvector/doc.go:1-35` reprojects an existing `turns_saved` value across
  separate resource budgets; it does not independently prove that a model turn vanished.
- `internal/agent/turnbatch.go:3-58` measures tool calls per assistant turn and batched-turn
  rate. That structural KPI is adjacent evidence, not an equivalent-effect witness.

The shipped `internal/turnavoid` leaf reuses the pure replay shape and leakage guard
without importing these adjacent proxies into its realized-turn numerator.

## Shipped offline spine

`fak.turnavoid.trace/v1` is JSONL with exactly one `control`, `exact-reuse`, and
`fused-batch` row for every `(trace_id, unit_id, turn_index)` immutable input. Cross-arm
validation requires the same SHA-256 input digest, earlier-turn decision basis, control
turn count, control gross work, independent observer, and control required-effect labels.
Rows are ordered within each trace/unit/arm sequence; a decision basis at or beyond its
own turn is rejected as same-turn result leakage. The closed mechanism, lifecycle, and
reason vocabularies include an explicit `unknown`; other spellings are rejected instead
of being coerced.

The pure fold emits `fak.turnavoid.report/v1` arm-by-arm rather than adding mutually
exclusive experimental arms together. For each arm it reports committed model turns,
realized and withheld turn deltas, lifecycle counts, retained-turn reductions, preserved
and suppressed required effects, gross control/candidate work, validation/speculation/
retry/recovery overhead, net latency and cost, and sorted mechanism/reason attribution.
`HONEST_WITHHELD_CREDIT` means an opportunity remained visible but did not enter the
realized numerator; it is an honest report, not a gate failure.

Run the captured six-input witness (the issue's five cases include both unsafe and
counterfactual no-credit rows) with:

```bash
fak turnavoid replay --in cmd/fak/testdata/turnavoid-replay.jsonl
fak turnavoid replay --in cmd/fak/testdata/turnavoid-replay.jsonl --json
go test ./internal/turnavoid ./cmd/fak -run Turnavoid
```

This remains offline and additive: it does not predict, enforce, route inference, call a
model, or execute a tool. Rollback is deletion of the leaf, bounded CLI dispatch arm,
fixture/tests, and these two documentation entries; there are no live schema consumers.

## License and research limits

No source code, tests, comments, or assets were copied. The pinned LLMCompiler revision is
MIT-licensed at [`LICENSE@a00c9d35507507da70e8c637eee64efc8c1857ae`](https://github.com/SqueezeAILab/LLMCompiler/blob/a00c9d35507507da70e8c637eee64efc8c1857ae/LICENSE),
so a future direct port could be permitted if its notice is preserved; this work is
**INSPIRE-ONLY** because it adopts no expressive bytes. Paper and vendor mechanisms are
likewise paraphrased rather than implemented from source.

The supplied field report was used as a discovery map and then checked against the
primary sources above. Companion scratch reports named `github.md` and `internal.md` were
absent. Sources and figures not independently verified in this pass are intentionally
excluded. The accounting classes in this note are fak's proposed operational vocabulary,
not a claim that the field has standardized these names.
