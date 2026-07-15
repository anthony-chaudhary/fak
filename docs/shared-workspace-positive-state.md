# Negframe and managed context are one positive-state pipeline

`internal/negframe` and managed context implement one rule on two data types: **before a model-visible emission, construct the positive prose and positive working state that downstream reasoning should reuse.** Negframe performs that construction on fak-authored sentences; managed context performs it on the session view. Both reduce stale operands in the same shared workspace.

## One principle, two surfaces

| Surface | Unmanaged form | Positive-state form | Checkable implementation |
|---|---|---|---|
| Prose | A denial or stale error remains salient and asks the model to invert it. | Emit the reason plus the available action or target state. | [`internal/negframe.Reframe`](../internal/negframe/reframe.go), [`internal/negframe.ReframeFakOnly`](../internal/negframe/fakonly.go) |
| Session state | A chronological transcript accumulates errors, abandoned plans, and repeated corrections. | Keep one originating-task pin and replace the current working state; page durable detail by content ID. | [`ManagedQuerySession`](../internal/gateway/query_not_chat.go), [`CtxRestoreTombstoneIDs`](../internal/gateway/ctxrestore_worker.go), [`AssembleInvariantReseed`](../internal/gateway/invariant_reseed.go) |

The mechanistic justification and authoring rule are canonical in [Positive-state construction](positive-state-construction.md). The append-versus-reseed decision is specified in [Query, not chat](query-not-chat.md).

## The shared gateway emit seam

The two transformations meet where `internal/gateway` assembles model-visible guidance from managed state:

1. [`adviseCtxStep`](../internal/gateway/ctxvalue.go) reads the current context-headroom state, derives a `sessionsteer.StepClass`, and builds an affordance for the next model step.
2. In that same symbol, `a.Affordance = negframe.Reframe(sessionsteer.StepAdviceAffordance(...))` turns the state-derived advice into positive fak-authored prose before it is returned in the managed-context view.
3. [`adjudicationNote`](../internal/gateway/refusal_notes.go) applies `negframe.ReframeFakOnly` at the result-admission boundary, preserving opaque user/tool bytes while reframing only fak-owned fragments.
4. [`journalReframePass`](../internal/gateway/reframe_journal.go) and [`journalReframeFragments`](../internal/gateway/reframe_journal.go) provide the observation seam: treatment rows run `ReframePass`, control rows preserve the text, and the bounded journal records content-free outcomes.

This is the named conjunction: managed context decides **which state and affordance** enter the request; negframe decides **how fak-owned guidance states that affordance**. They are not independent post-processing features.

## What is wired now

- `adviseCtxStep` emits reframed, context-sensitive step advice.
- `adjudicationNote` distinguishes fak-authored and opaque fragments before rewriting.
- [`SwapErrorKeepQuery`](../internal/gateway/swap_error_keep_query.go) replaces a failed-result frame while retaining the originating-task restore ID.
- [`AssembleInvariantReseed`](../internal/gateway/invariant_reseed.go) constructs a corrected goal plus required invariant, then checks that reframe retained every invariant token.
- [`QueryNotChatRegistry`](../internal/gateway/query_not_chat.go) detects append-style accumulation and gates it only when the soak flag is enabled.
- [`extractPositiveResidue`](../internal/agent/anthropic_positive_residue.go) is opt-in and keeps explicit current assertions while preserving dropped bytes by digest.

## What remains intentionally incomplete

- Emit-time reframing is not a license to transform arbitrary provider, user, or tool-result content. Only typed fak-owned fragments cross `ReframeFakOnly`; other bytes remain opaque.
- Query-not-chat append enforcement is observe-only unless `FAK_QUERY_NOT_CHAT_ENFORCE=true`.
- Positive-residue compaction is opt-in through `CompactOptions.PositiveResidue` while its conservative extractor soaks.
- Research tickets still need to validate that negation-tax and workspace-pressure proxies track measured model degradation; the current metrics are operational signals, not proof of causal model quality.

## Evidence commands

The prose-side contract is exercised by:

```text
go test ./internal/negframe/... -run Reframe
```

The state-side contract is exercised by:

```text
go test ./internal/gateway/... -run Context
```

Those selectors cover real symbols in the table above, but they prove implementation behavior only. They do not upgrade the open model-side validation work into a measured performance claim.
