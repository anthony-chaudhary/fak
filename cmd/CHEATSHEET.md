# fak cmd/ diagnostic cheatsheet

A decision table for the 13 benchmark/diagnostic tools in `cmd/`. Each entry carries a copy‑pasteable example invocation that runs on the current codebase and an honest scope note.

## Decision table

| Tool | Use when you need… | Example invocation | Scope / notes |
|------|-------------------|-------------------|---------------|
| `batchbench` | Measure aggregate decode throughput with batch size B | `go run ./cmd/batchbench -hf ~/.cache/fak-models/qwen2.5-1.5b-instruct -quant -model qwen2.5-1.5b-instruct -batch 1,2,4,8,16 -tokens 1024` | Pure decode-speed benchmark; single-encoder, multi-decoder workloads. |
| `demorace` | Live demo: fak vs tuned warm-cache baseline | `go run ./cmd/demorace -addr 127.0.0.1:8147` | Side‑by‑side latency comparison; no吞吐 numbers. |
| `fleetserve` | Cross‑agent shared‑prefix fleet workload | `fleetserve -quant -prefix 1024 -turns 1,2,4 -decode 32 -result 128 -concurrency 1,8,16,32,64 [-out f.json]` | Simulates multi‑agent chat traffic; reports per‑agent latency and aggregate throughput. |
| `sessionbench` | Net value‑add of fused agent kernel on long multi‑agent session | `go run ./cmd/sessionbench -hf ~/.cache/fak-models/qwen2.5-1.5b-instruct -quant` | Compares three modes: naive‑stateless, per‑agent‑KV, and fak‑fused. |
| `modelbench` | In‑kernel forward pass latency (serial core + parity lane) | `go run ./cmd/modelbench -hf ~/.cache/fak-models/qwen2.5-1.5b-instruct -quant` | Measures prefill and decode latency at the model boundary. Supports `FAK_WORKERS` and `FAK_BUDGET`. |
| `modelprof` | Modular bottleneck profiler with roofline attribution | `go run ./cmd/modelprof -hf ~/.cache/fak-models/qwen2.5-1.5b-instruct -quant` | Emits a per‑layer breakdown with roofline‑style compute‑vs‑memory attribution. |
| `q8bench` | Independent verifier for int8‑quantized path | `go run ./cmd/q8bench -hf ~/.cache/fak-models/qwen2.5-1.5b-instruct -quant` | Pass‑by‑pass verification against f32 reference; exits non‑zero on divergence. |
| `q8kernel` | GEMV kernel microbenchmark (f32 vs int8×int8 vs int8×f32) | `go run ./cmd/q8kernel -hf ~/.cache/fak-models/qwen2.5-1.5b-instruct -quant` | Isolated kernel timing; not a full‑model benchmark. |
| `ggufprobe` | GGUF architecture dump | `ggufprobe [-full] [-dump name,...] <model.gguf>` | Read‑only inspector for GGUF metadata; no model execution. |
| `q4kdiag` | Q4_K_M correctness diagnostic (requires FAK_Q4K=1) | `FAK_Q4K=1 go run ./cmd/q4kdiag -gguf <Qwen3.6-27B.q4_k_m.gguf>` | Checks Q4_K_M quantized path against reference; exits non‑zero on mismatch. |
| `qwen35check` | Qwen3.5/Qwen3‑Next hybrid HF snapshot correctness witness | `go run ./cmd/qwen35check -hf ~/.cache/fak-models/qwen2.5-1.5b-instruct` | Verifies the hybrid GDN + Q8 path on a known‑good checkpoint. |
| `paritybench` | Assemble cross‑model parity artifact (local vs frontier) | `paritybench --local 'experiments/parity/local-*.json' --local-gpu 'experiments/parity/remote-*-7b.json' --reference-cards experiments/parity/reference-frontier.json --reference claude-sonnet --require-phase1 --out-json experiments/parity/parity.json --out-md experiments/parity/PARITY.md` | Requires pre‑collected A/B reports (from `tools/run_local_model.sh`). Emits a scored parity table. |
| `lensviz` | Visual next‑token debugger (logit lens) | `lensviz -hf ~/.cache/fak-models/qwen2.5-1.5b-instruct -tok ~/.cache/fak-models/tokenizers/qwen2.5 -p "The capital of France is" -raw -k 6 -html lens.html` | Two views: terminal (per‑layer top‑k) and HTML (`-html out.html`, full layer×position heatmap). |

## Common patterns

- Most `-hf` tools accept a HuggingFace directory containing `config.json` + `model.safetensors[.index.json]`.
- `-quant` triggers the int8 quantized path where available; Q8 is the default for Llama‑family, Qwen3.5 uses the hybrid GDN path.
- `-gguf` tools read GGUF checkpoints via the quant loader; `-tok` points to a `tokenizer.json` directory (defaults to `-hf` dir or GGUF sidecar).
- `FAK_Q4K=1` enables the Q4_K_M quantized kernel (experimental, correctness‑checked by `q4kdiag`).

## Detailed writeups

(Per‑tool `*-RESULTS.md` files will be linked here once produced.)