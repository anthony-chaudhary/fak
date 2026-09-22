# Three context budgets, not one

`fak` has **three independent context budgets**. They are routinely conflated —
including in-review, including by agents. This page is the disambiguator. Read it
before changing any "context budget" number.

The failure mode this prevents is concrete: a change made against budget **B**
got justified by citing a defect in budget **C**, and a proposed fix to **C**
would have regressed hosted-provider economics while doing nothing for the
native engine.

## The three budgets

| | **A. Model window** | **B. Native shed-line** | **C. Worker envelope** |
|---|---|---|---|
| **Governs** | what the engine can *represent* — RoPE positions, KV allocation | how much *this engine* keeps resident as compactible history | how big a *hosted* worker's conversation budget may be |
| **Unit** | tokens the model accepts | tokens of resident history | tokens the provider bills / you budget |
| **Keyed on** | the loaded model's `max_position_embeddings` | the *resolved* window | a **glob on the model name string** |
| **Authority** | `model.Config.MaxPositionEmbeddings` → `agent.InKernelPlanner.ContextWindow()` | `agent.DeriveCompactHistoryBudget` | `ctxplan.EnvelopeForModel` |
| **Consumer** | admission (`refuseContextLength`), KV sizing, `/v1/models` | `ApplyPromptShrink` → `CompactMessagesWithOptions` | `cmd/dispatchworker` (`fak guard` fleet launch) |
| **Overflow behavior** | HTTP 400 `context_length_exceeded` | old turns compacted pre-admission | budget → `BUDGET_CONTEXT_EXHAUSTED` |
| **Provenance** | the checkpoint's `config.json` | `MODELED` (60% of window − 32k reserve) | `MODELED` (`#3611` worker-model envelope) |

### A — the model window

Set by the checkpoint. This is the only budget that describes *the engine*. It is
**not** a resident target — see `docs/long-context-defaults.md` ("the advertised
context window is a hard cap, not a target").

Declared windows for the checkpoints in-tree (read from their `config.json`, not
assumed — this is the file a model change must update):

| model | declared window | RoPE scaling |
|---|---:|---|
| DeepSeek V4.1 Flash | 1,048,576 | YaRN ×16 from 65,536 |
| DeepSeek V4 Flash | 1,048,576 | see `deepseek_v4_flash_config.json` |
| GLM-5 Next | 1,048,576 | see `glm5next/config.json` |
| Qwen 3.8 / Flash-Next | 262,144 | see `qwen4exp_flash_next_config.json` |

**The models differ by 4× and that is the point.** A shed-line policy keyed on a
constant (48k, 96k, or even 128k) is wrong for at least one of these: too small
for the 1M models, and potentially *larger than the whole window* for a small
model. That is why **B is derived from A per launch** rather than fixed. A single
hardcoded resident budget cannot serve a 262k Qwen and a 1M DeepSeek correctly.

Two consequences that follow from the table:

- **A 150k agent window is ~14% of a 1M model and ~57% of a 262k model.** The same
  absolute token target is a very different fraction of each model's capacity. A
  "150k context" claim is only meaningful alongside the model it was made on.
- **Small-window models must not be given a large shed-line.** `DeriveCompactHistoryBudget`
  returns `0` (compaction off) when the window is at or below the 32k reserve, and
  floors at the reserve above it. That branch exists for exactly the model
  diversity in this table.

DeepSeek V4.1's window and YaRN parameters are asserted by
`admitDeepSeekV41Published` (witnessed by
`TestDeepSeekV41RejectsShrunkContextWindow`). The other three are not yet asserted —
a config that silently shrank one of them would be admitted today. That is an open
gap, not a shipped guarantee.

### B — the native shed-line

Derived from A, per launch:

```text
shed_line = (resolved_window - 32000 output reserve) * 60%
```

Consumed only by the native in-kernel prompt path. This is what makes a 150k
window *usable*: without it the path had no resident line at all and refused an
over-window transcript instead of compacting it. See
`docs/native-long-context.md` and the witness at
`docs/_witnesses/native-context-shedline-2026-09-22/`.

Being *derived* is the design: the 60% share is one number that produces a
correct-shaped line for every window in the A table, from 262,144 (→ 138,086)
to 1,048,576 (→ 609,945), and correctly yields `0` for a model too small to
spare any history.

### C — the worker envelope

A **name-pattern table for hosted provider models**, plus a wildcard fallback.
Probed values at HEAD:

| model string | pattern hit | hard cap | effective ceiling |
|---|---|---:|---:|
| `gpt-5.3-codex` | `*codex*` | 272,000 | 96,000 |
| `claude-haiku-4` | `*haiku*` | 96,000 | 48,000 |
| `fable-3` | `*fable*` | 64,000 | 32,000 |
| `deepseek-v41-flash` | `*` | 200,000 | 32,000 |
| `qwen3.8-27b` | `*` | 200,000 | 32,000 |
| `hive-ai/deepseek-ai/DeepSeek-V4.1-Flash` | `*` | 200,000 | 32,000 |

## The rules that follow

1. **C does not touch the native engine.** Verified by `rg`: `EnvelopeForModel`
   has exactly three call sites, all in `cmd/dispatchworker/guard.go` (the fleet
   launcher). `DefaultEnvelopes` has three more: two scorecard renders in
   `cmd/fak/token_defaults.go` and `EnvelopeForModel` itself. **Zero** call sites
   exist in `internal/agent` or the native prompt path. Changing C cannot make a
   native turn longer; changing B cannot change what a hosted worker is allowed.
   They are disjoint.

   One near-miss worth naming, because it is the kind of thing this page exists to
   catch: `ctxplan.DefaultBudgetBounds` (`internal/ctxplan/adaptive.go:162`) does
   derive from `GenericTurnEnvelope()`, which returns `EnvelopeForModel("")` with
   `TargetResidentTokens` overwritten to a hardcoded **8192**. That looks like a
   native coupling. It is not: `DefaultBudgetBounds` / `RecommendBudget` /
   `RecommendBudgetForForecast` have **no callers outside `internal/ctxplan`**, so
   the adaptive-budget machinery is currently reachable only from `ctxplan`'s own
   query API and its tests. It is also labelled as a seed the caller overrides
   ("the obvious lever a caller overrides with the model's actual context
   window") — not a ceiling anyone enforces. Treat it as inert until something
   wires it; do not cite it as a native budget.

2. **A native/local model never matches a provider pattern.** No local model
   string contains `codex`, `haiku`, or `fable`; every one falls to `*`. The
   `272000` in `envelope.go` is therefore **not** a defect on the native path —
   it is a hosted-Codex-model prior. (An earlier review claimed otherwise; it was
   wrong.)

3. **C is deliberately a name glob, for a deliberate reason.** C budgets what you
   *pay a provider*. A 272k-capable hosted model with a real 32k output reserve is
   a money decision, not a capability discovery. Deferring C to "the resolved
   window" would be *wrong* for hosted models, where the cap is the provider's and
   the reserve is real spend.

4. **A and B must agree with the engine; C need not.** If A is wrong the engine
   allocates a KV store it cannot address. If B is wrong the turn is refused or
   the live task is shed. If C is wrong a worker gets a budget that is
   suboptimal-but-safe. Only A and B are correctness budgets.

## The one real latent mismatch (open, and it is not on the native path)

C's wildcard fallback ignores a model's *actual* declared window. A
`deepseek-v41-flash` launched as a fleet worker gets a 200k cap / 32k ceiling
from C even though its checkpoint declares 1M.

This is **open, not broken**, and deliberately not fixed here:

- It lives in the fleet launcher, not the engine.
- The conservative direction is intentional for hosted-worker economics, and its
  provenance is labeled `MODELED` with a witness note.
- The honest fix is *not* "C defers to A" — for a hosted model, A is the
  provider's number and the right resident line is an economics question. A
  correct fix would let a caller *supply* the declared window for a local model
  while leaving provider patterns authoritative for hosted ones, and would need a
  witness that a longer worker envelope actually pays.

Next checkable step: decide whether a local/native model launched through
`cmd/dispatchworker` should carry its declared window into C, and if so witness
the quality/cost tradeoff. Filed as a session flag, not silently absorbed.

## Provenance of this disambiguation

Probe: a temporary table test over `ctxplan.EnvelopeForModel` for twelve model
strings, run at `e7df4710c3`. The table above is that probe's output.

Call-site claims were verified with `rg` over non-test `.go` files, not inferred.
They were **corrected once during writing**: the first draft of rule 1 claimed only
`cmd/dispatchworker` + a scorecard render, and the `rg` sweep found
`ctxplan.DefaultBudgetBounds` as a fourth `GenericTurnEnvelope` consumer. That
consumer is documented above as reachable-only-within-`ctxplan`. Recorded because
the correction is the point of this page: an unverified "no other callers" claim is
exactly how the budgets got conflated in the first place.

This page was written after a review that cited a `*codex*` envelope defect
(`HardContextCap: 272000`) as an obstacle to native 150k context. That review was
wrong, and the misdiagnosis is preserved here rather than quietly dropped, because
it is the most likely way a future reader re-derives it.
