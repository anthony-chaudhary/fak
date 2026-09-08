<!-- fak-qwen38-key: rawdecode-runner-owned-selected-logprobs-v3 -->
# Seal runner-owned selected-token logprobs into Qwen3.8 Vulkan receipts

```routing
lane: compute
paths: ["internal/rawdecode/executor.go", "internal/rawdecode/executor_test.go", "internal/compute/qwen38_vulkan_decode_receipt.go", "internal/compute/qwen38_vulkan_decode_receipt_test.go", "docs/tickets/strix-hardware-validation/TICKET-12176-RAWDECODE-SELECTED-LOGPROBS.md"]
expected_steps: 6
```

GitHub: [#12302](https://github.com/anthony-chaudhary/fak/issues/12302)

## Parent context

Bounded enabling child of #12096 and #12176. Receipt-v3 prerequisite #12294 is satisfied by `075b803c81`; measured-run identity is preserved from `7f1a50ac78`.

## Why this is next

#12176 cannot honestly emit a matched native candidate/reference pair until the native runner retains normalized chosen-token evidence. This is the smallest device-free producer leaf that advances that path without fabricating authority.

## Current state

The raw-decode loop selects each generated token while the full logit row is present, but retains only token ID, top-1/top-2 logits, and margin. The Qwen3.8 Vulkan receipt carries output IDs and a finite-logits boolean, but no normalized selected-token logprob sequence or binding digest. `model.NativeInferenceReceipt` is a distinct served-request schema and cannot substitute for this physical receipt.

## Working spine

Real model logits -> greedy token selection -> stable runner-owned chosen-token logprob -> typed receipt-v3 binding -> deterministic mutation refusal.

## Core through-line

Compute stable max-subtracted log-softmax during real greedy selection, keep the observation inaccessible to caller construction, and bind the ordered values to receipt-v3 token IDs through a canonical digest. Reject missing or mutated observations instead of inventing defaults.

## Gold-plating boundary

No modelbench CLI wiring, llama.cpp observer, HTTP/subprocess adapter, raw-logit retention, non-greedy sampling semantics, GPU work, SSH/service/lease changes, scoreboard admission, statistics, throughput ratio, or performance-win claim.

## Done condition

- [ ] The real rawdecode loop captures one finite non-positive selected-token logprob per generated token, including the prefill-selected token.
- [ ] Caller-constructed `rawdecode` values cannot claim an observed logprob.
- [ ] Receipt v3 binds ordered token/logprob evidence and a canonical digest; v2 bytes and validation remain compatible.
- [ ] Missing, non-finite, positive, length-mismatched, reordered, and mutated evidence fail closed.
- [ ] No physical, pair, comparison, or performance credit is emitted.

## Done condition / witness

Done when the focused rawdecode/compute tests prove real-loop capture, stable arithmetic, v3 binding, mutation refusal, and v2 compatibility.

## Witness

```text
go test ./internal/rawdecode ./internal/compute -run 'Test.*(SelectedTokenLogprob|Qwen38Vulkan.*Logprob)' -count=1
go vet ./internal/rawdecode ./internal/compute
fak validate --mine internal/rawdecode/executor.go --mine internal/rawdecode/executor_test.go --mine internal/compute/qwen38_vulkan_decode_receipt.go --mine internal/compute/qwen38_vulkan_decode_receipt_test.go
```

## Verifiable Witness

The witness is deterministic and device-free. It grants no Strix hardware, comparator, concurrency, throughput, or performance credit.

## Acceptance gate

All focused tests and vet exit zero, all mutations return a zero receipt, and existing v2 fixtures remain unchanged.

## Closure binding

The resolving commit cites #12302, includes this ticket, and carries `(fak compute)`.

## Likely files

- `internal/rawdecode/executor.go`
- `internal/rawdecode/executor_test.go`
- `internal/compute/qwen38_vulkan_decode_receipt.go`
- `internal/compute/qwen38_vulkan_decode_receipt_test.go`
- this ticket

## Lane

`compute`; two Go packages plus this ticket, six expected steps.

## Problem frame

- Centrality: Core.
- P1 Context: advanced - unlocks exact candidate quality evidence for #12176.
- P2 Net value: advanced - normalized token evidence is stronger than raw logits or decoded text.
- P3 Adaptation: preserved - extends the existing native runner and receipt-v3 seams.
- P4 Operations: advanced - evidence originates at selection and incomplete binding fails closed.

## Blast radius and fallback

Only rawdecode evidence and the v3 canonical receipt are in scope. V2 remains readable. Until #12176 wires this producer, modelbench pair capture remains unavailable and non-creditable.
