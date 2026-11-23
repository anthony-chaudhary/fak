---
title: "The precision machinery in fak"
description: "The three layers of precision machinery in fak: the metadata contract, the cross-support tensor, and the runtime policy and schedule that decide."
slug: the-precision-machinery
keywords:
  - precision contract
  - cross support tensor
  - dynamic precision policy
  - precision schedule
  - fail closed gate
  - typed reason codes
  - honest fence
  - per token precision
  - fak mixed precision ladder
date: 2026-11-18
---

# The precision machinery in fak

The precision machinery in fak is three layers: the metadata contract, the cross-support tensor, and the runtime policy and schedule that decide what runs. The wiring status of each is stated plainly.

*You are on **Level 4 of 5** of the [mixed-precision ladder](README.md).*

> **In one breath:** fak keeps precision in three layers of machinery.
> One checks whether a combination is allowed at all. One records which
> formats each backend implements. One decides what should run right now,
> and every unknown is a visible refusal.

**One line:** fak splits precision into a metadata contract that adjudicates whether a combination is allowed, a cross-support tensor that records which formats each backend implements, and a runtime policy plus schedule that decide what should run, and all three share one discipline: an unknown is a visible refusal, never a quiet default.

> **Audience.** You are a systems engineer. You are wiring precision into a
> runtime and you want to know what the pieces are, what each one decides, and
> how they fit together. You are comfortable reading a Go interface.

> **Short answer.** Three layers, four pieces. The **contract** answers "is
> this combination allowed at all" and is pure metadata. The **cross-support
> tensor** answers "does this backend implement this format" and turns every gap
> into an honest fence. The **dynamic policy** answers "should this token run
> quantized" and is the one that touches execution today; the **schedule**
> answers "which level for this tensor, layer, phase" and is designed but not
> yet wired into a live path. The unifying rule is fail-closed: an unknown is a
> visible refusal.

## Why are there three separate pieces of machinery?

Because they answer three different questions, and collapsing them is how
precision systems get quietly wrong.

The tempting design is one function that takes a model and a target and returns
a plan. That function would have to decide three unrelated things at once: is
this combination even legal, does the hardware support it, and what should run
on this particular token. Those questions have different inputs, different
failure modes, and different owners. fak keeps them apart.

```text
  +---------------------------------------------------------------+
  |  LAYER 1  THE CONTRACT            internal/mixedprecision/     |
  |           "is this combination    contract.go                  |
  |            allowed at all?"       pure metadata, no execution  |
  +---------------------------------------------------------------+
  |  LAYER 2  THE CROSS-SUPPORT       internal/covmatrix/          |
  |           TENSOR                  precision.go                 |
  |           "does this backend      a table, gaps become fences  |
  |            implement this format?"                             |
  +---------------------------------------------------------------+
  |  LAYER 3  THE RUNTIME             internal/model/              |
  |           dynamic_precision.go  + precision_schedule.go        |
  |           "what should run        the only layer that touches  |
  |            right now?"            execution                    |
  +---------------------------------------------------------------+
```

| Layer | Piece | Question it answers | File | Touches execution? |
|---|---|---|---|---|
| 1 | The contract | Is this combination allowed at all? | `internal/mixedprecision/contract.go` | No. Metadata only. |
| 2 | The cross-support tensor | Does this backend implement this format? | `internal/covmatrix/precision.go` | No. A table. |
| 3a | The dynamic policy | Should this token run quantized? | `internal/model/dynamic_precision.go` | Yes. Wired into `Prefill`/`Step`. |
| 3b | The schedule | Which level for this tensor, layer, phase? | `internal/model/precision_schedule.go` | Not yet. Designed, exercised by its own tests. |

The contract's own README is explicit about the boundary. `internal/mixedprecision/README.md`
states that the contract does not choose a quantization method, does not execute
a kernel, and does not claim that lower modeled weight bits preserve model
quality. It adjudicates metadata. That is the whole job.

## What does the contract actually decide?

One function: `Evaluate(d Descriptor, support Support) Result` in
`internal/mixedprecision/contract.go`. It takes a descriptor, which is a
proposed per-module precision map, and a support matrix, which is what the
caller says is allowed. It returns a `Result` with an `Outcome`, a typed
`ReasonCode`, the assignments it produced, and a budget.

The body is a sequence of **fail-closed gates**. Each gate either passes or
returns a typed reason and stops. Nothing falls through to a default.

| Gate | Condition that trips it | Reason code |
|---|---|---|
| Schema | `d.Schema` is not `mixedprecision/v1` | `unknown_schema` |
| Artifact pin | artifact id or version not in the support matrix | `unknown_artifact` / `unknown_artifact_version` |
| Recipe pin | recipe id or version not in the support matrix | `unknown_recipe` / `unknown_recipe_version` |
| Runtime pin | runtime id or version not in the support matrix | `unknown_runtime` / `unknown_runtime_version` |
| Combination | the (artifact, recipe, runtime) triple is not declared | `undeclared_combination` |
| Evidence | malformed evidence record | `invalid_evidence` |
| Descriptor | empty modules or empty rules | `invalid_descriptor` |
| Ambiguity | a module matches more than one rule | `ambiguous_module` |
| Coverage | a module matches no rule | decided by `Fallback.Mode` |
| Precision | a precision not in the support list | `unknown_precision` |
| Budget | weighted-bit arithmetic would overflow | `invalid_descriptor` with detail `budget_overflow` |

The `ReasonCode` type has sixteen constants, listed at the top of
`internal/mixedprecision/contract.go`. They are the vocabulary of refusal. A
caller never has to parse a string to know why something was rejected.

**The three fallback modes.** When a module matches no rule, the descriptor's
`Fallback.Mode` decides what happens, and there are exactly three answers:

| Mode | What happens | Outcome |
|---|---|---|
| `refuse` | the whole evaluation refuses with `unmatched_module` | `refused` |
| `handoff` | the decision is delegated to the runtime | `delegate` |
| `assign` | the declared fallback precision is used and counted as fallback | the declared combination outcome, with reason `accepted_fallback` |

The `handoff` mode is the honest one for a runtime-specific compatibility
matrix: rather than claim support it does not have, the contract says "this is
the runtime's call" and returns `delegate`.

**The budget.** `Budget` reports `ModulesTotal`, `ModulesMatched`,
`ModulesFallback`, `ParametersTotal`, `ParametersMatched`,
`ParametersFallback`, `WeightedBits`, `AverageBits`, and `Coverage`. The two
derived numbers are `AverageBits = WeightedBits / ParametersTotal` and
`Coverage = matched parameters over total parameters`. The bit table lives in
`precisionBits`: `fp32` is 32, `fp16` and `bf16` are 16, `fp8` and `int8` are 8,
`int6` is 6, `int4`, `nf4` and `fp4` are 4, `int3` is 3, `int2` is 2. Anything
else is not a known precision and refuses.

**What the budget excludes.** `internal/mixedprecision/README.md` is careful
here: the budget excludes scales, zero-points, padding, metadata, activation and
cache state, and runtime workspace. It is a weight-bit accounting, not a memory
prediction.

**The identity.** `canonicalID` is a sha256 over the canonical JSON of the
schema, the provenance, the sorted assignments, and the budget. The same
descriptor always yields the same id, so a plan can be referenced by hash.

The package's `doc.go` states the invariant in one line: mixed precision
operations are fail-closed and bounded.

## What is a fence, and why is it not a bug?

A fence is a cell in the cross-support tensor that says "this backend does not
implement this format." It is a deliberate, honest refusal, and the source says
so in as many words.

The tensor lives in `internal/covmatrix/precision.go`. Its envelope id is
`fak-cross-support-tensor/1`. The precision axis is the weight dtype the kernel
actually dispatches a matmul or upload path on, and there are three of them:
`Precisions = []string{"f32", "q8_0", "q4_k"}`.

| Backend | Weight precisions it implements |
|---|---|
| `cpu` | `f32`, `q8_0`, `q4_k` |
| `cuda` | `f32`, `q8_0`, `q4_k` |
| `metal` | `f32` |
| `vulkan` | `f32`, `q8_0` |

That table is `backendPrecisions`. The classification function is `classifyX`:
if the backend does not implement the precision, the cell is `Fenced`;
otherwise it inherits the (family, backend) support from the two-dimensional
grid. By construction `classifyX` never returns `Undefined`.

**Why a fence is safe.** Every backend ends its weight-dtype switch in an honest
panic. If a caller somehow reaches a path the backend does not implement, the
program stops rather than computing a wrong answer. The real panic sites are
`internal/compute/cpuref.go` at the `MatMul` and `BatchedMatMul` defaults,
`internal/compute/metal.go` at the "only F32 today" upload guards,
`internal/compute/vulkan.go` at the "only F32 today" upload guards, and
`internal/compute/cuda.go` at the `Upload` dtype guard. The comment at the top
of `precision.go` says it plainly: a fence is honest, it is not debt.

**The never-Undefined invariant.** The witness is
`internal/covmatrix/precision_test.go`, in `TestGridXNoSilentUndefined`. It
walks every cross cell and fails if any is `Undefined` or has an empty support
level. The invariant is not assumed; it is computed by the same machinery that
would catch a regression.

**What is deliberately out of the axis.** The comment in `precision.go` notes
that the wider dtype set (`f16`, `bf16`, `i8`, `i4`, `fp8`) and the orthogonal
cache-precision tier are not weight-matmul dispatch keys on the reference path,
so they are out of this axis by design. They are noted, not silently dropped.

## How does the runtime decide per token?

The dynamic policy is the first layer that touches execution. It lives in
`internal/model/dynamic_precision.go`, and its design point is that it is
**online and per-token**. It does not decide once for the model. It decides
again for every span, using the model's own confidence as the signal.

The type is `DynamicPrecisionPolicy`, with two fields: `MinTop1Margin` and
`MinTop1Prob`. `MinTop1Margin` is in raw logit units: the top logit minus the
second must be at least this value. `MinTop1Prob` gates on the softmax
probability of the top token.

The mechanism is `runDynamic`. Read it as a sequence:

```text
  runDynamic(tokens, f32, q8):
    if the model has no quantized weights:
        run f32, record, return
    before := clone the KV cache
    logits := run the quantized path
    Q8Attempts++
    margin, prob := logitConfidence(logits)
    if the policy accepts(margin, prob):
        record Q8, return the quantized logits
    else:
        restore the cloned cache
        logits := run f32
        record F32 fallback, return
```

Two details matter. First, the cache is **cloned before** the quantized run and
**restored** if the result is rejected. The rejected work is discarded, not
blended. Second, `logitConfidence` returns `margin = top1 - top2` and `topProb`,
the softmax probability of the top token. That is the margin argument from
Level 3 turned into a runtime controller.

`accepts` has two guards worth knowing. A nil receiver returns false, so a
session with no policy never gains a precision path it did not ask for. And a
gate of zero means "do not gate on this signal."

**The zero-value hazard, stated plainly.** The source comment on
`DynamicPrecisionPolicy` says that with both fields zero, the policy accepts
every quantized result and behaves like a fixed choice, except that it still
records statistics. There is no default constructor. A zero-valued policy is
not a conservative policy; it is a policy that gates on nothing. If you
construct one and forget to set the fields, you have silently turned the
adaptive path into a fixed quantized path. Set at least one field, or set both
deliberately.

`PrecisionStats` records what happened: `Q8Attempts`, `Q8Calls`, `Q8Tokens`,
`F32Calls`, `F32Tokens`, `Fallbacks`, `LastTier`, `LastAccepted`, `LastMargin`,
`LastTop1Prob`. The `Fallbacks` counter is the one to watch: it is the number of
times the quantized path was tried and rejected.

## What is a precision schedule, and why is a refusal better than a default?

The dynamic policy decides one tier per call. The contract is metadata. Neither
can answer the question the schedule exists for, and the file comment in
`internal/model/precision_schedule.go` states it: **which level should this
tensor run at, at this layer, in this phase?**

A `Point` names one decision site: `Layer`, `Role`, `Phase`. `Role` is a closed
vocabulary, from `weight` and the routed-expert roles through the KV roles to
`head`, `draft`, and `target`. `Phase` is `prefill` or `decode`. Both have
parsers that refuse an unknown token rather than silently becoming a zero value.

**The ordering is load-bearing.** `Level` is a `uint8` ordered from widest to
narrowest, and a lower number is a wider, more precise level:

| Level | Value | Meaning |
|---|---|---|
| `LevelF32` | 0 | widest |
| `LevelF16` | 1 | |
| `LevelBF16` | 2 | |
| `LevelFP8` | 3 | |
| `LevelQ8_0` | 4 | |
| `LevelQ4_K` | 5 | |
| `LevelQ4_0` | 6 | |
| `LevelFP4` | 7 | |
| `LevelI4` | 8 | narrowest |

Because `LevelF32` is 0, the zero value of `Level` is the widest level. That is
exactly why a `Level` returned without its `ok` flag must never be trusted as a
decision. The boolean is what distinguishes "we chose f32" from "we did not
choose at all." To make a refusal impossible to misread, the file defines
`LevelKVUnknown = 255`, deliberately outside the ordered enum, so a refusal can
never be mistaken for a widest-level widen.

**Two methods, two jobs.** The `Schedule` interface has `LevelAt`, the raw
decision, and `LevelFor`, the checked decision returning `(Level, bool)`. A
refusal is explicit, never a silent widen.

**The receipt.** `LevelReceipt` has two slices, `Resolved` and `Refused`.
`Record` appends an admitted decision; `Refuse` appends an explicit refusal
carrying the level that was **refused**. That is what lets an auditor tell "we
chose f32" from "we asked for int4 and were told no." `Levels()` returns the
distinct admitted levels in first-seen order, and `DistinctLevels()` counts
them. The zero value is usable.

**The policy-versus-kernel split.** `RunScheduledPass` consults the schedule per
point, executes that point at the decided level, and records every decision,
admitted or refused. The comment is the key sentence: the schedule owns the
**policy** and the caller owns the **kernel**. A refused point is never executed
and never quieted into a widest-level rerun. That is what keeps a refusal
auditable instead of a silent widen.

**Why a refusal beats a default.** A default is invisible. If a missing mapping
silently widened to f32, a tensor nobody asked about would quietly double its
byte cost, and no receipt would show it. A refusal is visible, typed, and
carries a reason. The concrete schedules are built to fail closed:
`UniformSchedule(level)` is total by construction, `StaticSchedule(map)` answers
only the points it was handed and refuses the rest, and
`ScheduleKVPrecision(kv, fallback)` answers the KV roles from the tier and
everything else from the fallback.

One honest caveat from the file: the levels are dispatch **hints** for the
existing kernels, not a claim that two levels are numerically identical.

**The wiring status, stated plainly.** The schedule is designed and tested, but
it is not yet on a live path. `Session.PrecisionSchedule` is assigned only in
`internal/model/precision_schedule_test.go`, and `RunScheduledPass` is called
only from that same test file. The dynamic policy, by contrast, is wired: a
non-nil `Session.PrecisionPolicy` routes `Prefill` and `Step` through the
dynamic path. So of the four layers in the table above, exactly one touches
execution today. Level 5 grades this properly.

## How do the pieces connect?

The bridge from the metadata contract to the runtime is
`ScheduleFromAssignments` in `internal/model/precision_schedule.go`. It takes
the `Assignment` values the contract produced and a function that maps a module
name to a `Role`, and returns a `Schedule`.

It does three things, and each one is a refusal waiting to happen:

1. It maps each assignment to a `Point`, recovering the layer ordinal from the
   module name, because the contract's `Assignment` carries no layer field.
2. It materializes **both** phases for each covered (layer, role) pair, because
   assignments carry no phase. The adapter is phase-invariant by construction
   rather than guessing a phase at lookup time.
3. It **refuses** any precision with no `Level`. `int6`, `int3`, and `int2` have
   no `Level`, so those points refuse explicitly. An unmapped (layer, role)
   point refuses too, and a module with no integer segment in its name refuses
   rather than landing on layer 0.

```text
  Descriptor + Support
        |
        v
  mixedprecision.Evaluate        ->  Result.Assignments
        |                                   |
        |                                   v
        |                        ScheduleFromAssignments
        |                                   |
        |                                   v
        |                            a Schedule
        |                                   |
        v                                   v
  (metadata: allowed?)          RunScheduledPass -> LevelReceipt
                                (runtime: what ran, and what refused)
```

So the contract decides what is legal, the adapter turns that into a per-point
plan, and the scheduled pass executes it and records every decision. The
cross-support tensor sits alongside as the table the support and scorecard
tooling folds to report which formats a backend implements. It is a reporting
input, not a runtime lookup: no code in `internal/model` or `internal/compute`
imports it.

## What one rule ties the pieces together?

Every layer refuses the same way: an unknown is a visible refusal, never a quiet
default. The shape of the refusal differs by layer, and that is fine. What
matters is that none of them guesses.

- The contract returns a typed `ReasonCode` for every gate it fails, and its
  `doc.go` states the invariant: fail-closed and bounded.
- The cross-support tensor turns every unimplemented (backend, precision) pair
  into a fence, and the backend's dtype switch panics rather than computing a
  wrong answer.
- The dynamic policy's nil receiver returns false, and a rejected quantized
  result is discarded, not blended.
- The schedule returns `(Level, bool)`, records refusals with the level that was
  refused, and never widens a refused point into a rerun.

The reason is the same in all four places. In a numeric system, the dangerous
failure is not the error you see. It is the error you did not notice. A visible
refusal is a failure you can see, count, and audit. A quiet default is a failure
that ships.

## Check yourself

1. What three questions do the three layers of machinery answer, and which
   layer touches execution today?
2. What does the contract's `Evaluate` return when a module matches no rule,
   and what decides the answer?
3. What is a fence, and why is it not a bug?
4. What is the zero-value hazard in `DynamicPrecisionPolicy`, and why does it
   matter?
5. Why is a refusal better than a default in a precision schedule?

<details>
<summary>Answers</summary>

1. The contract answers "is this combination allowed at all" and is pure
   metadata. The cross-support tensor answers "does this backend implement this
   format" and is a table. The dynamic policy answers "should this token run
   quantized" and is the one that touches execution today; the schedule answers
   "which level for this tensor, layer, phase" and is designed but not yet
   wired into a live path.
2. It depends on `Fallback.Mode`. `refuse` returns `refused` with
   `unmatched_module`; `handoff` returns `delegate`; `assign` uses the declared
   fallback precision and counts it as fallback.
3. A fence is a cross-support cell where the backend does not implement the
   format. It is not a bug because every backend ends its weight-dtype switch in
   an honest panic, so the gap is a refusal rather than a silently wrong result.
   `classifyX` never returns `Undefined`, and `TestGridXNoSilentUndefined`
   witnesses it.
4. With both `MinTop1Margin` and `MinTop1Prob` zero, the policy accepts every
   quantized result and behaves like a fixed choice, while still recording
   statistics. There is no default constructor, so a zero-valued policy silently
   turns the adaptive path into a fixed quantized path.
5. A default is invisible: a missing mapping that widened to f32 would quietly
   change the byte cost with no record. A refusal is typed, carries a reason,
   and is recorded in the receipt, so an auditor can tell "we chose f32" from
   "we asked for int4 and were told no."

</details>

## See also

- [Level 3 — What to quantize, and what to leave alone](level-3-what-to-quantize.md) — the margin argument this machinery acts on.
- [Level 5 — The evidence boundary](level-5-the-evidence-boundary.md) — which of these claims are measured, modeled, or designed-not-wired.
- [Level 4 of the inference ladder](../inference/level-4-prefill-vs-decode.md) — the same subject from the other side: the arithmetic-intensity argument that makes bytes per weight a speed decision, not only a memory one.
- [The fak mixed-precision ladder](README.md) — all five rungs.

---

*[Full ladder (README)](README.md) · prev ← [Level 3 — What to quantize, and what to leave alone](level-3-what-to-quantize.md) · next → [Level 5 — The evidence boundary](level-5-the-evidence-boundary.md)*
