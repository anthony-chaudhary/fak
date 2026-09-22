---
title: "Native agent context shed-line wiring — software witness"
description: "Software witness that the native in-kernel path derives a compaction shed-line from the resolved context window, so an over-window agent transcript is compacted and served instead of refused with HTTP 400."
---

# Native agent context shed-line wiring — software witness

## Verdict

- **Verdict:** `SW_VERIFIED_SHEDLINE_WIRING`
- **Provenance class:** `[SW-VERIFIED]` — this is a deterministic in-memory witness.
  It is **not** a physical hardware receipt and must never be cited as
  `[HW-WITNESSED]`.
- **Scope:** the admission/compaction wiring on the native (`fak serve` /
  `fak up`) wire. The attention kernel, RoPE scaling, and KV residency are out of
  scope.
- **Branch:** `fak/ticket-150k-agent-context`
- **Commits:** `e646e3837d` (shed-line wiring), `94af792ba9` (V4.1 window assertion)

## The defect this witness closes

The native planner was constructed without `InKernelPlannerConfig.CompactHistoryBudget`
by **every** production constructor:

- `cmd/fak/serve_native_config.go` `serveNativeControlConfig` — field absent.
- `cmd/fak/up.go` `newTurnkeyInKernelPlanner` — field absent.

`InKernelPlanner.ApplyPromptShrink` short-circuits on a zero budget:

```go
if compactBudget <= 0 && !elideStale && !deferTools {
    return messages, tools, TypedPromptShrinkOutcome{}
}
```

So the typed-message compactor (`CompactMessagesWithOptions`) — implemented and
unit-tested — was **never invoked on the native prompt path**. An agent
transcript that outgrew the resolved window reached
`refuseContextLength` at full size and was refused with HTTP 400
`context_length_exceeded`, with no second chance. Only the Anthropic passthrough
compactor was live.

## The fix

`agent.DeriveCompactHistoryBudget(windowTokens, requested)` is now the single
authority for the native shed-line:

```text
shed_line = (resolved_native_context_tokens - 32000 output reserve) * 60%
```

It is pure (no env, no clock, no I/O) and monotone. Both production
constructors spend it, and `--native-compact-history-budget` overrides it.
Because the shrink runs inside `preparePrompt` **before** tokenization and the
window check, the over-window case is compacted pre-admission and never reaches
the 400 — there is no retry loop and no duplicate refusal path.

## Witnesses (all green)

| Witness | What it proves |
|---|---|
| `agent.TestCompactBudgetWiring` | A wired planner sheds **178,504 est. tokens** (178,538 → 156) and **serves** the turn with no `InKernelContextLengthError`; the unwired production shape (budget 0) is a no-op. Asserts the outcome, not that a field was set. |
| `agent.TestDeriveCompactHistoryBudget` | The derivation's 9 boundary cases, including 150k → 70,800 and 1M → 609,945. |
| `agent.TestDeriveCompactHistoryBudgetIsMonotone` | The shed-line never regresses as the window grows (0 → 2M sweep). |
| `agent.TestDeriveCompactHistoryBudgetMatchesDocTable` | The worked-examples table in `docs/native-long-context.md` cannot drift from the code. |
| `cmd/fak.TestServePlannerWiresNativeCompactBudget` | The **production** constructor `serveNativePlannerConfigWithContext` derives the budget from the resolved window, honors the explicit flag, and stays inert on an unresolved window. |
| `model.TestDeepSeekV41RejectsShrunkContextWindow` | The published 1,048,576 window and YaRN x16-from-65536 are now asserted by admission; 4 tamper cases refuse. |

Run them:

```bash
go test ./internal/agent/ -run 'TestDeriveCompactHistoryBudget|TestCompactBudgetWiring' -count=1
go test ./cmd/fak/ -run 'TestServePlannerWiresNativeCompactBudget|TestResolveServeNativeContext|TestServeNativeContext' -count=1
go test ./internal/model/ -run 'TestDeepSeekV41RejectsShrunkContextWindow' -count=1
```

## Worked shed-line defaults

| Resolved window | Shed-line |
|---:|---:|
| 32,000 | 0 (at the reserve; compaction off) |
| 40,000 | 32,000 (floored at the reserve) |
| 150,000 | **70,800** |
| 1,048,576 | 609,945 |

## Honest fences

- **This is not an MECW claim.** `docs/long-context-defaults.md` defines a Measured
  Effective Context Window as the largest resident context that still meets a
  quality SLO under a *local fak witness*. The 60%/32K shape is a doctrine-derived
  prior, labelled `MODELED`. It does not claim a 150k-token native turn keeps
  answer quality within SLO.
- **This is not a throughput or latency claim.** Nothing here measures prefill or
  decode rate at 150k tokens.
- **The test ruler is a heuristic.** `CompactMessagesWithOptions` sizes messages
  with `EstimateMessageTokens` ((bytes+3)/4), not the tokenizer. The witness
  asserts on that same ruler, so it is internally consistent but does not prove
  tokenizer-exact shedding. A tokenizer-exact guarantee would need the shrink to
  run against real token counts.
- **Physical long-context behavior is witnessed elsewhere, for a different
  model.** `docs/_witnesses/issue-9665-strix-halo-long-context/` records
  Qwen3.8-27B fak-native at 35k/64k/128k/200k on 128GB Strix Halo (KV 6.84 →
  25.02 → 39.10 GiB). That is prior evidence the 150k class is *reachable* on
  that hardware; it is **not** a witness for this wiring, which is model- and
  hardware-independent.
- **Not yet done:** S2 in the goal spec (whether the `*codex*` ctxplan envelope
  should defer to a resolved native window for a local model). See
  `_scratch/goals/GOAL-agent-150k-context.md`.
