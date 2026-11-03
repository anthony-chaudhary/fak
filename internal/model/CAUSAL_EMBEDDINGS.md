# Explicit causal-model embeddings

This optional module adapts llama.cpp's final-hidden-output, pooling, and
LM-head bypass contract at `cd8427d018c07eefab6c110f7f115c31ec4c6157`
(`src/llama-graph.cpp`, MIT) into FAK's existing CPU resident forward.
It is one configured provider at the ordinary `POST /v1/embeddings` route,
not a detached pooling helper or a second inference engine.

## Selection and actual consumer

`fak serve --embedding-pooling mean` or `last` selects learned causal
representations from the already loaded local model and tokenizer.
An explicit local model name and compatible tokenizer are required.
The existing `--gguf` directory arm loads a safetensors snapshot through
`LoadSafetensorsQuantConfigDir`; its lean Q8 projection store is admitted
without expanding dropped F32 weights. F32 resident models are also admitted.
The loader, authentication, and existing fit/weight ownership rules are unchanged.
This change does not download a checkpoint or select a model for an operator.

The model must be standard dense Llama with the explicitly supported configuration
axes. Hybrid/recurrent, MoE, MLA, vision, custom normalization/attention topology,
nonstandard RoPE, adapters, observers, device backends, proxies, explicit device-only policy, and collective/EP
execution are refused. Selected errors do not fall back to hash embeddings.
An empty flag preserves the original hashing backend byte-for-byte.

## Forward and output contract

The provider retains a model weight-session hold throughout each computation.
It performs a fresh causal hidden forward with existing layer mathematics, applies
the final norm, then mean or last-position pooling and L2 normalization.
It never allocates the all-layer Activations history and never invokes the LM head.
This is not bidirectional BERT, a trained retrieval embedding checkpoint, CLS/rank
pooling, arbitrary dimension projection, or a quality/performance claim.

Text input uses the configured tokenizer's untrusted-text encoding and the declared
model BOS once. Integer arrays are real vocabulary IDs and are not textified,
prepended, or truncated. Successful response usage counts the input IDs processed, including any BOS. Failed entered rows are conservatively charged their complete input to existing session/token budgets; no partial embedding response is emitted.
Each sequence has 1..256 tokens and must fit the declared model context.
A batch has at most eight items/1024 tokens; text is at most 4096 bytes per item;
the serialized input field is at most 64 KiB. Numerical workspace/work admission
adds a finite model-dependent bound; it is not an allocator, RSS, or latency guarantee.
One provider computation is admitted at a time; concurrent work is refused 429.
Selected learned work uses the existing session/spend/memory and scheduler/principal/provider token reservation gates with exact input counts and zero generated tokens. Research-arm ownership is released synchronously. Every return releases the admitted turn and lease; chat history reset/splicing is deliberately not applied to immutable embedding IDs.
Cancellation is observed at admission and CPU layer boundaries, not inside a kernel.

Dimensions must be omitted/zero or exactly the model hidden width. Float and base64
responses use the existing wire serializer. Response headers carry tokenizer identity,
selected F32/Q8_0 projection precision, pooling mode, and authenticated artifact receipt
when the trusted loader supplied one. Missing artifact membership stays `unknown`;
a model alias is not an authenticated release claim.

## Qualification debt

Source fixtures cover final normalization, mean/last pooling, malformed geometry,
closed weights, cancellation, no LM head dependence, lean Q8 resident execution,
real token IDs, actual handler/batch/base64, selected fail-closed errors, EP refusal,
and unchanged hash default. They are authored but UNRUN.

Suggested scoped host checks, only through the qualified resource-admitted route:

```
go test ./internal/model -run '^TestCausalEmbedding' -count=1
go test ./internal/gateway -run '^TestCausalEmbedding' -count=1
```

Before any model-quality or performance claim, bind exact source/build flags/CPU,
checkpoint and tokenizer hashes, F32 versus selected Q8 precision, input IDs and
pooling, baseline, raw outputs, full-vector equality/tolerance and actual timing/memory.
No benchmark result, hardware job, semantic quality, main integration or FULL pass
is supplied by this source packet.
