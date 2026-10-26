# pdbench

`pdbench` is a bounded acceptance tracer for process-separated native prefill and
decode. Both processes load and hash the same GGUF; the producer captures the
complete host prefix plus final prefill logits before generating its reference
continuation, then transfers those frames over loopback TCP. The consumer does no
prefill, restores the prefix, greedily decodes, and returns every generated token
for parity checking.

Start with a small Qwen2.5 1.5B or 3B Q8 GGUF that fits each device. A 27B Q8
checkpoint is not an appropriate 16 GB eGPU acceptance envelope.

Build the CUDA-enabled binary explicitly; an ordinary `go run` does not register
the CUDA backend:

```text
fak-dev build-cuda binary -arch sm_120 -out-pkg ./cmd/pdbench -out-bin <OUTPUT>/pdbench -repo-root <CHECKOUT> -json
CUDA_VISIBLE_DEVICES=<DECODE-EGPU-UUID> <OUTPUT>/pdbench -mode consumer -gguf MODEL.gguf -backend cuda -address 127.0.0.1:19191 -prompt-ids 1,42,17 -steps 8
CUDA_VISIBLE_DEVICES=<PREFILL-EGPU-UUID> <OUTPUT>/pdbench -mode producer -gguf MODEL.gguf -backend cuda -address 127.0.0.1:19191 -prompt-ids 1,42,17 -steps 8
```

Replace the placeholders with two exact, distinct GPU UUIDs; ordinal `0`/`1` is
not sufficient device evidence. Run each role in a separate process. The receipt reports artifact/config identity,
backend, quant path, and declared `CUDA_VISIBLE_DEVICES`, exact connection bytes, separated prefill,
snapshot, reference-decode, restore, consumer-decode, and end-to-end handoff times, plus
all-token parity. It deliberately makes no performance claim. CPU protocol tests
must opt in with `-backend cpu-ref -software-only` and remain software evidence.
That path uses the existing f32 dequant-on-load GGUF loader and an unquantized
session; CUDA uses the Q8-on-load loader and Q8 session. Both choices are bound
into the handoff identity and receipt, so neither role can silently switch paths.
`-timeout` bounds socket setup/I/O and is checked between inference stages; the
current synchronous GGUF load, file hash, and individual model calls are not
preemptible and are excluded from that guarantee.
The UUIDs are validated declarations, not runtime CUDA device-attestation; a
passing receipt is process/device-declaration evidence and is not by itself a
hardware-qualified result.
