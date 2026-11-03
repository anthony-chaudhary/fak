# Packed heterogeneous host prefill (qualification only)

This coherent port adapts vLLM's packed request input preparation to Fak's existing
CPU projection/attention kernels and the existing native scheduler. It is not a
new scheduler, page allocator, asynchronous pipeline, or GPU backend.

## Selection and normal consumer

Build with `-tags fak_vllm_packed_prefill`. The untagged build remains off.
The selected native syscall `Engine.Admit` path arms its own NativeScheduler before
publication. Existing `cmd/fak serve --engine inkernel`/model-selected routing
reaches the registered `modelengine.Default`, then kernel Engine.Complete and
Engine.Admit. This is distinct from the chat InKernelPlanner route, which is not
changed. No environment variable, new endpoint, credential, or routing authority
is added. This is not a test-only setter.

Eligible CPU dense pre-norm requests enter the existing PREFILLING state. One
scheduler executor borrows a cohort of independently owned sessions behind
existing cache-free statistics shells. At most 32 actual prompt tokens TOTAL
are packed per iteration, with cumulative request offsets, each request's own
absolute positions and causal attention, shared projection panels, and one
final-logit row per request. No padding tokens enter the cache. Decode lanes
still advance at the existing iteration boundary.

The model API supports FP32 or explicit Q8 weights with FP32 KV. The current
normal native session constructor uses FP32; this port does not silently change
its weight precision. Quantized KV, Q4/K-quants, device backends, hybrid/recurrent,
MoE/MLA, explicit device-only execution policy, installed attention observers,
unsupported topology/position modes, and auxiliary hidden-state capture
are refused before mutation by the packed API. The scheduler leaves unsupported
sessions on historical synchronous admission. Existing in-flight prefix
coalescing or explicit scheduler prefix state also retains its original path.
Under recompute capacity admission, each unfinished packed owner reserves its full
prompt rather than only its current empty/partial KV. This preserves the existing
admission/promotion contract, not a new hard cap across legacy decode growth.
The API borrows sessions; callers exclusively own their mode/config/cache until
return and must not race Close or model configuration changes.

Cancel/Close only signals while model execution owns a session. At the completed
boundary the original owner is restored before terminal retirement; canceled
work publishes no logits/tokens. Panic/Goexit unwinding retires all borrowed
owners without claiming success. Existing CloseAndWait joins the scheduler.
PackedPrefillReceipt counts executed calls/sequences/rows and the maximum whole
cohort rows, not speed, accepted tokens, or memory/RSS.

## Donor and compatibility

vllm-project/vllm at `5c8ae7c74aaa99d1f7fbeea942fd7282e3413810`,
`vllm/v1/worker/gpu_model_runner.py`:
`_prepare_inputs` cumulative query offsets, per-request positions, and final-row
logits gather. Source blob `d98498a4f97bfd89700829fd36df71bbab6437f5`;
Apache-2.0 LICENSE blob `261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64`.
No upstream executable/GPU code or model weights are imported.

Equal-length existing PrefillEach routes retain their rectangular embedding and
attention kernels; the generalized wrapper adds a cumulative-offset slice whose
allocation overhead remains unmeasured. The packed entry uses the same per-layer norm/bias/QK norm,
Kraw, RoPE, gated MLP and logits scaling contracts. Q8 numerical acceptance is
the existing cosine>=0.999 plus greedy parity, not a new bit-equality claim.

## Required qualification (UNRUN)

```sh
go test ./internal/model -run '^TestPackedPrefill' -count=1
go test -tags fak_vllm_packed_prefill ./internal/modelengine -run '^TestNativePackedPrefill' -count=1
go test -race -tags fak_vllm_packed_prefill ./internal/modelengine -run '^TestNativePackedPrefill' -count=1
```

The ordinary Engine fixture must prove a multi-sequence call (Sequences>Calls),
exact actual token rows, total32 bound, serial-output parity, terminal retirement
and canceled/Close-in-flight cleanup. Model fixtures cover heterogeneous lengths,
different prefix lengths, repeated buffers, continuation, cache/lineage parity
and no-effect rejection.

Before promotion, run paired same-checkpoint/tokenizer/runtime default versus
selected serving traces with unequal cold prompts, one hot decode lane, and
bounded cancellations. Report prompt-token denominator, packed cohort counts,
prefill/TTFT and decode-tail distributions, peak allocated bytes and qualification
status. No throughput, latency, allocation, hardware or FULL claim follows from
source publication. No hardware job has been requested or launched.
