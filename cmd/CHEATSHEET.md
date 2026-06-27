# fak cmd/ diagnostic cheatsheet

A decision table for the 13 benchmark/diagnostic tools in `cmd/`. Each entry carries a copy‑pasteable example invocation that runs on the current codebase and an honest scope note.

## Decision table

| Tool | Use when you need… | Example invocation | Scope / notes |
|------|-------------------|-------------------|---------------|
| `cfgprobe` | Check MoE/dense FFN config axes from GGUF metadata | `go run ./cmd/cfgprobe <model.gguf>` | Reads only metadata shard; catches dimension bugs in seconds without full model load. |
| `diagtok` | Disambiguate broken forward pass vs broken tokenizer | `go run ./cmd/diagtok -gguf <model.gguf> -n 16` | Round-trips a known string and greedily decodes, printing token ids with decodes. |
| `gemma4diag` | Sweep uncertain forward axes for Gemma 4 bring-up | `go run ./cmd/gemma4diag -gguf <gemma4.gguf> -n 5` | Tests softmax scale, rope_freqs, decoder-norm; prints top-k predictions for each configuration. |
| `ggufprobe` | Dump GGUF architecture without llama.cpp | `go run ./cmd/ggufprobe <model.gguf>` | Prints version, metadata KVs, tensor-dtype histogram, and whether fak's model.Config can be derived. |
| `glmcfgdiag` | Verify GLM-5.2 MLA dims before ~1.5h full load | `go run ./cmd/glmcfgdiag <glm52-shard.gguf>` | Cheap no-reload witness; prints QKNopeHeadDim, VHeadDim, and latent ranks; exits non-zero on mismatch. |
| `gpucheck` | Witness GPU backend correctness against f32 reference | `go run -tags cuda ./cmd/gpucheck -hf <hf-dir> -backend cuda -n 12` | Loads HF checkpoint, decodes on f32 path and GPU backend; asserts greedy token streams agree. |
| `kpiprobe` | Measure RSI loop's deterministic LRU hit-rate | `go run ./cmd/kpiprobe` | Reports KPI=<float> line over fixed reference trace; wall-clock-free, reproducible across platforms. |
| `modelbench` | Measure in-kernel forward pass latency baseline | `go run ./cmd/modelbench -dir <export-dir> -quant` | Measures prefill and decode latency; supports FAK_WORKERS and FAK_BUDGET. |
| `q4kdiag` | Verify Q4_K_M quantized path correctness (FAK_Q4K=1) | `FAK_Q4K=1 go run ./cmd/q4kdiag -gguf <Qwen3.6-27B.q4_k_m.gguf>` | Loads Qwen3.6-27B-Q4_K_M, reports population/shapes, prints top-5 first-token logits against oracle. |
| `q8bench` | Independent verifier for int8-quantized path | `go run ./cmd/q8bench -dir <export-dir> -quant` | Confirms argmax-exact vs HF oracle and measures decode speedup; correctness + speed in one run. |
| `qwen35check` | Verify Qwen3.5/3-Next Gated-DeltaNet correctness | `go run ./cmd/qwen35check -hf <hf-dir>` | Loads hybrid HF snapshot, greedy-decodes tokens; compares against llama.cpp continuation. |
| `toktdiag` | Check if fak can extract embedded GGUF tokenizer | `go run ./cmd/toktdiag <model.gguf>` | Reports success/failure of tokenizer extraction from GGUF metadata. |
| `tpcheck` | Test tensor-parallel decomposition (simulated ranks) | `go run ./cmd/tpcheck` | Runs intra-layer sharding over simulated in-memory ranks; verifies Megatron identities. |

## Common patterns

- Most `-gguf` tools read GGUF checkpoints via the quant loader; path can be absolute or relative.
- `-hf` tools accept a HuggingFace directory containing `config.json` + `model.safetensors[.index.json]`.
- `-dir` tools accept a fak export directory (Python export from HF or local model path).
- `-quant` triggers the int8 quantized path where available; Q8 is default for most architectures.
- `FAK_Q4K=1` enables the Q4_K_M quantized kernel (experimental, correctness-checked by `q4kdiag`).
- `go run -tags cuda` is required for GPU backend tools (`gpucheck`).
- Most tools exit non-zero on failure/dismatch for CI gating.

## Detailed writeups

(Per‑tool `*-RESULTS.md` files will be linked here once produced.)