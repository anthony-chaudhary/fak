# Bounded vLLM beam search on the ordinary native CPU chat route

This is one complete, explicit search mode, not a new integer indirection helper.
The live/completed beam lifecycle, top-2B expansion and generated-length score
adapt vLLM `5c8ae7c74aaa99d1f7fbeea942fd7282e3413810`:

- [offline.py](https://github.com/vllm-project/vllm/blob/5c8ae7c74aaa99d1f7fbeea942fd7282e3413810/vllm/entrypoints/generate/beam_search/offline.py),
  Git `e458ba99282ff683593e7ddbedf726d0c8612857`;
- [utils.py](https://github.com/vllm-project/vllm/blob/5c8ae7c74aaa99d1f7fbeea942fd7282e3413810/vllm/entrypoints/generate/beam_search/utils.py),
  Git `d009008cb5a060939b26535b419ccfeea09d1673`.

The donor is Apache-2.0, copyright contributors to the vLLM project. The exact
root LICENSE (Git `261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64`) is already
reproduced in THIRD_PARTY_NOTICES.md; this port has its own attribution there.
The pinned donor has no root NOTICE. This Go adaptation changes execution,
floating-point reduction and deterministic ties; it does not inherit vLLM
device throughput, tokenizer/model qualification or KV-sharing behavior.

## Real consumer and opt-in

The ordinary `fak serve` native dense CPU planner accepts a buffered
`POST /v1/chat/completions` request such as:

```json
{
  "messages": [{"role": "user", "content": "A short answer, please."}],
  "max_tokens": 32,
  "fak": {"beam_search": {"width": 4, "length_penalty": 1}}
}
```

No flag, environment variable or global mutable selector is introduced.
Omitting `fak.beam_search` preserves the old sampler and response fields.
The internal API is `agent.WithNativeBeamSearch`; its options are captured by
value and copied on each application. The ordinary Complete entry calls the
same `preparePrompt`, context/deadline guards, host-memory admission and
session configuration used by this planner. It does not synthesize prompt IDs
from user text or call a standalone fixture-only model.

Each live beam is evaluated serially in a NEW owned CPU Session by replaying
the entire prepared prompt plus that beam's generated prefix through
`Session.PrefillContext`. The row is copied and the session is closed before
the search consumes it. No KV snapshot, shallow session clone, prefix-cache
loan, GPU stream, asynchronous consumer or existing BatchDriver indirection is
used. This incurs **extra prefill work**, explicitly reported as forward_calls
and replayed_tokens. It is a correctness-first CPU consumer, not an efficiency
claim. Prefix caching, speculative decoding and continuous batching are not
silently substituted.

The best sequence is the ordinary assistant message and passes through the
existing gateway adjudication exactly once. The public opt-in
`fak.beam_search` response is a separate, copied NON-CONTENT projection:
rank, generated-token count, cumulative log probability, normalized score,
closed finish reason and replay-work fields only. It NEVER contains alternative
messages, raw winner copies, token IDs, prompts or content hashes. Invalid or
unbounded internal metadata is omitted. This prevents the extension from
resurrecting a redacted or refused winner or publishing an unadjudicated
alternative.

The internal `agent.NativeBeamSearchReceipt` still owns ranked messages and
token IDs for trusted in-process search consumers; it is NOT a public response
and does not imply gateway admission. Ordinary usage counts the selected
visible sequence only, not all explored beams. A terminal EOS is omitted from
internal visible token IDs but INCLUDED in generated_tokens and the
length-penalty denominator. No ordinary native timing/measurement receipt is
fabricated for replay.

## Search and admission contract

- Width 1..8, generated steps 1..128, total context at most 4096, vocabulary at
  most 262144. The HTTP request requires an explicit output bound (max_tokens or its normal max_completion_tokens alias).
- Length penalty defaults to 1; explicit 0 disables normalization. Finite [0,2]
  only. Score is cumulative log probability divided by max(generated,1)^penalty.
- Every live parent expands its top min(2B,vocabulary) logits. EOS candidates
  complete, other candidates compete for the next topB live rows. Completed
  scores are immutable and their bestB are retained. There is no optimistic
  early-stopping heuristic.
- All logits must be finite. Log-softmax uses a float64 maximum-subtracted
  reduction. Any NaN, infinity, shape error, failed forward/close or cancellation
  returns no partial result. Stable ties preserve parent order and ascending
  token ID; final completed/live ties favor completed rows. This is deterministic
  FAK behavior, not a claim about Torch's tie order.
- Before the first forward, the conservative replay-token bound is
  P + B*((S-1)*P + (S-1)*S/2), limited to 1048576. The maximum-length replay is
  priced by the existing host session estimator, with a 1GiB session-plus-search
  admission ceiling. This is an analytical plan, not an RSS measurement.
- Search-owned token/header bookkeeping has a 32MiB conservative logical
  allowance; it is charged to the SAME armed host budget in addition to ordinary
  session admission and released on every exit. Model/session allocations and
  allocator overhead are not claimed to fit inside that 32MiB. The existing host
  budget's unknown-usage policy is preserved. No aggregate global request bound
  or zero-memory claim is implied.

Unsupported requests fail rather than downgrade: streaming, HTTP-provider or
turnkey `fak up` routes, legacy `/v1/completions`, tools/tool history, multimodal
or structured messages, guided constraints, raw prompts, stop strings, logit
bias/penalties/top-p/top-k/min-p/seed, nonzero explicit temperature, reasoning
controls/history, and ordinary decode/inference trace receipts. The native
planner additionally refuses device/Metal, MoE/DeepSeekV41, expert parallel,
speculative/DFlash/MTP/Engram, eligible device-cohort and Qwen Q4_K sequence/chunk
targets. The latter use a target-specific panel admission estimate, which must not
be silently substituted for this adapter's full-prefix replay. The ordinary serve
constructor arms BatchDecode and supplies a 512-token Qwen panel setting even for
unrelated dense models. Both structural settings are retained, not rejected or
switched off; eligibility checks the actual coalescer and chunk target predicates.
The normal-constructor consumer witness keeps BatchDecode:true and the 512-token
setting and proves that both remain unchanged on an eligible dense CPU model.
These are honest initial supported-source bounds,
not evidence that other models are inherently incompatible.

Generated tool-call candidates are refused across ALL retained alternatives.
Plain-text reasoning projection and normal tool normalization are applied
before any beam result is exposed. Existing gateway authentication, request
admission, model-route selection, deadline and winner adjudication remain on
the normal path.

## Evidence and acceptance still needed

Source fixtures cover known beam-vs-greedy divergence, EOS normalization,
stable ties, extreme finite logits, pre-forward bounds, cancellation/error
atomicity, input/result ownership, the real synthetic CPU Session consumer,
nil default, typed option capture, host-budget release, HTTP ingress/refusal,
non-content publication ownership, a genuinely pre-redacted winner, and the
normal handler's completed-edit body-refusal fold. Those wire fixtures preserve
raw internal alternatives while proving they cannot reappear in the response.
These are **authored but UNRUN** unless a later exact-source receipt says
otherwise. Compiler, full suite, real-model qualification, main integration,
hardware and quality/performance are separate debts.

A qualified software executor can run bounded focused checks at the exact
source SHA, without downloading models or launching hardware jobs:

```sh
go test ./internal/beam -run '^TestSearch' -count=1
go test ./internal/agent -run '^TestNativeBeam' -count=1
go test ./internal/gateway -run '^TestNativeBeam' -count=1
go test ./cmd/fak -run '^TestTurnkeyRefusesExplicitNativeBeam' -count=1
```

Later model evidence must bind source/build, tokenizer and model hashes,
CPU/quantization configuration, prepared prompt length, width/steps/penalty,
EOS IDs and the full replay denominator. Compare winner scores and token IDs
against a qualified pinned reference; record repeated-prefill cost and memory
rather than claiming a speedup from a launch or beam count. Rollback is simply
omitting the request option; the historical default path is unchanged.
