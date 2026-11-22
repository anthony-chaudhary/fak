<!-- fak-inkernel-prefix-cache-request-bypass-key: inkernel-prefix-cache-request-bypass -->
# In-kernel prefix cache: a per-request bypass (and a minimum-length control) for cold-path timing

```routing
lane: gateway
tracker: pub
```

Process cause: none

## Provenance

A token-time benchmark driver timed DeepSeek-V4.1-Flash on `fak serve --engine inkernel`
through `/v1/chat/completions` (stream + `include_usage`). Every timed run reported
`usage.prompt_tokens_details.cached_tokens == prompt_tokens` (32 of 32) and a 12-38 ms
client TTFT, while decode ran at about 4 s/token. Earlier untimed requests with the same
prompt, a calibration `max_tokens=1` probe and a `fak.native_inference_receipt` readback,
had admitted it to the radix cache. An untimed request with a different first content
token sent between runs did not evict it.

Read from source:

- `internal/radixkv` keeps many entries. It evicts LRU leaves only past a
  context-sized token budget (`resolveInKernelRadixBudgetTokens`), matches the longest
  token prefix, and on CUDA demotes to the host-L2 and SSD tiers instead of dropping.
- On a full-prompt hit, `lookupOwnedNativePrefix` returns the stored last logits, and
  `generateReusedContextWithBias` samples the first token with no forward pass.
- No request field turns reuse off. `cache_prompt` is not read, and chat/completions
  decode drops unknown fields silently (`decodeJSON` has no `DisallowUnknownFields`).
  The only switch is `FAK_INKERNEL_RADIX=off`, which applies to the whole serve and
  needs a restart. On a large offloaded model, a restart is minutes of boot and
  page-cache warmup.
- No `ignore_eos`, `min_tokens` or `min_new_tokens` reaches the in-kernel decode loop.
  `logit_bias` against EOS is only an additive bias clamped to ±100, and a non-empty bias
  leaves the device-greedy and speculative paths (`decodeLane.deviceGreedy`,
  `greedySpeculativeRequestEligible`), so a timing run that uses it measures a
  different decode path.

The private driver now works around both gaps: a per-request random system-message
nonce for cold prefill, and a prompt that naturally outruns `max_tokens`. These are
workarounds. The nonce changes the prompt shape, and the long-answer prompt depends on
the model following an instruction.

## Work items

1. **Per-request cache bypass.** Add a `fak` request extension, for example
   `"fak": {"prefix_cache": "bypass"}`, or honor llama.cpp's `cache_prompt: false`. It
   would skip lookup (always `matched = 0`, `cached_tokens = 0`) and, ideally, skip
   admission, so one request is timed cold without restarting the serve or touching the
   cache for other tenants. Witness: a gateway test where two identical requests, the
   second marked bypass, report `cached_tokens` = prompt_tokens and then 0.
2. **Minimum-length control.** Honor `ignore_eos` (or `min_tokens`) on the in-kernel
   decode loop: mask the stop-id set until `gen >= min_tokens`, while keeping the
   device-greedy and speculative paths eligible (the mask is a stop-set rule, not a
   logit bias). Witness: a decode-loop test whose model emits EOS at step 8 still
   returns `completion_tokens == max_tokens` with `finish_reason: length`.
3. Document both knobs beside `FAK_INKERNEL_RADIX` and refuse them where they cannot be
   honored, rather than dropping them silently.
