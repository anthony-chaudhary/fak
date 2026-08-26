---
title: "attention names"
description: "Map of exact repository symbols that use the attention root, kept distinct from the broader family label and sibling operations."
---

# attention names

This map positions the current `attention` coverage backlog. Each entry names the exact repository symbol; the family label remains the broader domain and is not a substitute for the symbol.

## Follow-on discovered names

- **`externalinvalidateattentionindex`** — the exact `attention` symbol; distinct from the broader family label and sibling operations.
- **`planeattentionindex`** — the exact `attention` symbol; distinct from the broader family label and sibling operations.


### preparePrefillAttention (cache and output setup)

preparePrefillAttention appends one layer's raw key, transformed key, and value panels to the KV cache, resolves that layer's sliding window, and allocates the zeroed attention output panel for the incoming prefill rows.

**Distinct from:** It prepares cached inputs and output storage before attention math; it does not run the causal query-key and value-mixing loop performed by attnPrefillInto.


### fillAttentionScores (raw single-head score fill)

fillAttentionScores writes scaled query-key dot products for one query head across a contiguous key-position range without applying softmax or mixing values.

**Distinct from:** It produces raw scaled logits only; softmaxAttentionScores normalizes an existing score buffer, while preparePrefillAttention prepares cache and output storage.


### fillSoftmaxAttentionScores (single-head score and softmax)

fillSoftmaxAttentionScores fills one query head's scaled query-key scores and then normalizes that score row in place with softmax.

**Distinct from:** It combines score generation and normalization for one head; fillAttentionScores leaves raw logits, and softmaxAttentionScores only normalizes an already-filled buffer.


### fillSoftmaxAttentionScores3 (three-head score and softmax)

fillSoftmaxAttentionScores3 traverses the shared key range once through scoreDot3 for exactly three query heads and then normalizes the three resulting score rows independently.

**Distinct from:** It is the three-query shared-key score path; fillSoftmaxAttentionScores handles one head, and attnFdot3SIMD is only the optional three-way dot-product kernel it may receive.


### accumulateAttentionValues (single-head value reduction)

accumulateAttentionValues performs the in-order weighted value-vector accumulation for one query head by applying each normalized attention score to the matching KV value head.

**Distinct from:** It reduces values for one head after score normalization; fillAttentionScores generates logits, while accumulateAttentionGroup shares one value traversal across an entire GQA head group.
