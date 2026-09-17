# V4.1 selection reference — provenance and resolved unmasked top-k semantics

Pinned reference revision: `dba1be0a40aa45a94ad051997016db3960a90277`
(`deepseek-ai/DeepSeek-V4.1-Flash` `inference/model.py`, Compressor + lightning
indexer; MIT).

Fixture: `selection_reference.json` (schema
`fak-v41-oracle-selection-reference/1`, rung
`HAND-DERIVED-FROM-REFERENCE-EQUATIONS`).

Consumers: `internal/model/v41_compress_index_test.go`
(`TestV41SelectionReferenceVector`), which re-executes the independent
transcription `v41SelectionRefRowsExec` and requires the checked-in bytes to be
exactly reproduced, then asserts production `V41SelectIndexRows` matches.

## Resolved semantics: the unmasked arm is SCORE-RANKED

With no source mask (`candidates == nil`), the final top-k is a **score-ranked
selection over causal-visible positions**:

1. a position is causal-visible iff `position < compressLen`;
2. **every** `logits` position is ranked, with masked-out and causal-invisible
   positions ranked as `-Inf` so they sink to the tail of the descending-score
   sort;
3. among visible positions, an exact score tie keeps the **lower** position;
4. the top `min(topK, len(logits))` are kept, re-sorted by position ascending,
   then shifted by `offset`;
5. **the `-1` slots are not a fixed end-pad.** Because the result is re-sorted by
   position, a `-1` occupies the position-order slot of whichever selection was
   causal-invisible or masked-out. `-Inf` ranking sends invisible positions to
   the tail of the *score* sort, so they are only dropped from the kept set
   after the visible ones — meaning `-1` will appear *after* the visible rows
   whenever the invisible positions are the higher indices (the unmasked case,
   e.g. `[11 13 15]` has no pad; `[11 12 13 14]` has none either), and *before*
   them when the invisible positions are the lower indices (e.g. the masked case
   `[-1 -1 22]`, where positions 0-1 are masked out and position 2 is visible).

The independent transcription of this equation is
`v41RefIndexRowsExec` (`internal/model/v41_compress_index_ref_test.go:212-261`,
"top-k causal-visible positions **by score**, returned in position order and
shifted by offset, padded with `-1`") and, for this fixture,
`v41SelectionRefRowsExec` (`internal/model/v41_compress_index_test.go`). Both
reuse nothing from the production port, so a correlated mis-read shared with it
cannot pass silently.

## The disproven alternative, and why it cannot be the semantics

fak#13168 was opened on the premise that the unmasked arm should return a
contiguous run of the highest indices — the "latest-window"/positional reading —
e.g. `[11 12 13 14 15 16]` for `compressLen=5, offset=11`. That premise is
**disproven**:

- **Unrepresentable output.** The int32 row encoding is bounded by
  `offset + (compressLen-1)`. At `offset=11, compressLen=5` that bound is `15`;
  production writes `int32(offset + position)` only for `position < compressLen`
  (`internal/model/v41_compress_index.go:420-425`). Absolute row `16` would
  require `position 5 >= compressLen=5`, i.e. a causal-invisible row, which is
  emitted as `-1`.
- **Contradicted by the pinned oracle.** `compress_index_ref.json` pins
  `score=[1,4,2,9,2]`, `mask=[F,F,F,T,T]`, `topK=3` → `rows=[-1,103,104]`, which
  is scored/mask driven, not positional.
- **Contradicted by the reference-executed test.** The unmasked case at
  `compressLen=5, topK=6, offset=11` yields `[11 12 13 14 15 -1]`
  (`internal/model/v41/v41_indexer_candidates_test.go:60-67`), not `[11..16]`.

**Disposition: production was correct.** The claimed fixture expectation
`[... 16]` was the error. No production change was made. The real gap this
reconciliation closes is *coverage*: before this fixture, `internal/model` had no
pinned-vector case for the **unmasked** arm, the topK-shorter-than-visible arm,
or the tie-break arm — the sibling `v41_compress_index_ref_test.go` selection
controls all pass a mask with `topK=1`, under which a broken unmasked branch
still passed.

The disproven positional reading is carried in the fixture as
`naive_positional_rows` and asserted to DIFFER from the expected rows on at least
two cases. That is the non-vacuity witness: a positional implementation fails
this vector.

## Fixture cases

| case | arm | pins |
|---|---|---|
| `block_max_and_newest_pinned` | unmasked | score-ranked selection: the picked set `[11 13 15]` is not a contiguous tail, so the positional reading (`[13 14 15]`) fails |
| `block_max_tie_lower_index_wins` | unmasked | the lower-position tie rule under a three-way tie at the top-k boundary (`[11 12 13 14]`; a higher-index tie-break yields `[12 13 14 15]`) |
| `unmasked_topk_exceeds_visible_pads_after` | unmasked | `topK` larger than the visible count: invisible positions supply the trailing `-1` slots, and the highest raw scores sit outside the causal window |
| `topk_blocks_control` | masked | `topKBlocks=1` source-candidate control; the mask changes the picked set (`[11 12 15]` vs positional `[13 14 15]`) |
| `single_visible_row_masked` | masked | the masked representation and `-1` placement: `[-1 -1 22]` puts `-1` *before* a visible row, which a fixed end-pad implementation cannot produce |

## Scope

This note covers the **selection stage only** — the final top-k rows, masked and
unmasked. It does not extend the `V41SelectCandidateBlocks` mask, the compressor
pooling stage, or any forward/router/attention behavior.