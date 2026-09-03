# Next-Gen OSS Performance Scout: Qwen 3.8 & GLM 5.3 Flash

- **Generated At**: 2026-09-03 18:36:59 UTC
- **Total Candidate Repos Scored**: 110
- **Retained Performance Repos**: 50
- **Qwen 3.8 Flash Repos**: 33
- **GLM 5.3 Flash Repos**: 18
- **Cross-Model Dual Repos**: 1

## Performance Inventory Table

| Repo | Stars | Model | Engine | Hardware | Quant | Proof / tok/s | Grade | Score |
|---|---|---|---|---|---|---|---|---|
| [hasso5703/dgx-spark-qwen38](https://github.com/hasso5703/dgx-spark-qwen38) | 154 | Qwen3.8-27B | SGLang | DGX Spark (GB10) | NVFP4 | 50 tok/s | MEASURED | 105 |
| [adrienbrault/qwen3.8-27b-...](https://github.com/adrienbrault/qwen3.8-27b-rtx5090) | 8 | Qwen3.8-27B | vLLM | RTX 5090 | NVFP4 | 300 t/s, 262K ctx | MEASURED | 105 |
| [Azhu9701/ninfer-4090d](https://github.com/Azhu9701/ninfer-4090d) | 0 | Qwen3.8-27B | NInfer | RTX 4090 | Unspecified / FP16 | 195 tok/s | MEASURED | 100 |
| [marksunner/dgx-spark-glm52](https://github.com/marksunner/dgx-spark-glm52) | 2 | GLM-5.3 | Custom OSS Runtime | DGX Spark (GB10) | Unspecified / FP16 | 26 tok/s, 200K context | MEASURED | 95 |
| [albond/SingleSpark-Qwen3....](https://github.com/albond/SingleSpark-Qwen3.8-Flash-Next) | 2 | Qwen3.8-Flash-Next | vLLM | DGX Spark (GB10) | NVFP4, FP8, INT4 | 44 tok/s | MEASURED | 95 |
| [thadreber-web/llama.cpp-q...](https://github.com/thadreber-web/llama.cpp-qwen38-flash-next) | 0 | Qwen3.8-Flash-Next | llama.cpp | DGX Spark (GB10) | Unspecified / FP16 | Empirical Benchmark... | EVALUATION | 95 |
| [vcruz305/GLM-5.3-Flash-EX...](https://github.com/vcruz305/GLM-5.3-Flash-EXL3-K2-DGX-Spark-recipe) | 22 | GLM-5.3-Flash | vLLM | DGX Spark (GB10) | EXL3 | Empirical Benchmark... | EVALUATION | 90 |
| [gitcommit90/glm-5.3-one-s...](https://github.com/gitcommit90/glm-5.3-one-spark) | 12 | GLM-5.3-Flash | Custom OSS Runtime | DGX Spark (GB10) | EXL3, Sub-3bpw Quant | 64 tok/s | MEASURED | 90 |
| [shyringo/qwen3.8-flash-ne...](https://github.com/shyringo/qwen3.8-flash-next-in-c) | 0 | Qwen3.8-Flash-Next | Native C | Host CPU | Unspecified / FP16 | 5.03 token/s | MEASURED | 90 |
| [airawatraj/dgx-spark-qwen...](https://github.com/airawatraj/dgx-spark-qwen38-flash-agent) | 6 | Qwen3.8-Flash-Next | SGLang | DGX Spark (GB10) | Unspecified / FP16 | 36.8 tok/s, 262K co... | MEASURED | 90 |
| [cglab-public/dgx-spark-fl...](https://github.com/cglab-public/dgx-spark-flashnext) | 0 | Qwen3.8-Flash-Next | SGLang | DGX Spark (GB10) | NVFP4 | Empirical Benchmark... | EVALUATION | 90 |
| [Gr33n93/llama.cpp-qwen3.8...](https://github.com/Gr33n93/llama.cpp-qwen3.8-flash-next-mtp-vulkan) | 0 | Qwen3.8-Flash-Next | llama.cpp, Vulkan/RADV | AMD Strix Halo (gfx1151) | Unspecified / FP16 | Operational Recipe | KERNEL | 85 |
| [MindLab-Research/ferrite](https://github.com/MindLab-Research/ferrite) | 0 | GLM-5.3-Flash | ferrite (Rust) | Host CPU | Unspecified / FP16 | Operational Recipe | KERNEL | 85 |
| [feifeidu-max/Qwen3.8-Flas...](https://github.com/feifeidu-max/Qwen3.8-FlashNext-Optimization) | 0 | Qwen3.8-Flash-Next | llama.cpp | RTX Pro Enterprise | Unspecified / FP16 | Empirical Benchmark... | KERNEL | 85 |
| [halt95/qwen38-flash-next-...](https://github.com/halt95/qwen38-flash-next-3090s) | 0 | Qwen3.8-Flash-Next | vLLM | RTX 3090 | Unspecified / FP16 | 118 tok/s, 262K con... | MEASURED | 80 |
| [punkjazz-labs/glm-5.3-fla...](https://github.com/punkjazz-labs/glm-5.3-flash-exl3-4x-dgx-spark) | 0 | GLM-5.3-Flash | vLLM | DGX Spark (GB10) | EXL3, INT4 | Empirical Benchmark... | EVALUATION | 80 |
| [ormandj/sglang-glm53-flas...](https://github.com/ormandj/sglang-glm53-flash-sm120) | 15 | GLM-5.3-Flash | SGLang | RTX Pro Enterprise, NVIDIA B200 (Blackwell) | FP8 | Operational Recipe | RECIPE | 80 |
| [himorishige/glm53-flash-2...](https://github.com/himorishige/glm53-flash-2x-dgx-spark-recipe) | 3 | GLM-5.3-Flash | vLLM | DGX Spark (GB10) | NVFP4 | Empirical Benchmark... | EVALUATION | 80 |
| [pangoleen/qwen3.8-27b-dgx...](https://github.com/pangoleen/qwen3.8-27b-dgx-spark-dflash2) | 22 | Qwen3.8-27B | SGLang | DGX Spark (GB10) | Unspecified / FP16 | Empirical Benchmark... | EVALUATION | 75 |
| [PieBru/Qwen-3.8-27B_Strix...](https://github.com/PieBru/Qwen-3.8-27B_Strix-Halo_gfx1151) | 12 | Qwen3.8-27B | llama.cpp | AMD Strix Halo (gfx1151) | Unspecified / FP16 | Empirical Benchmark... | EVALUATION | 75 |
| [carloslfu/slotstream](https://github.com/carloslfu/slotstream) | 275 | Qwen3.8-Flash-Next | MLX | Apple Silicon | INT4 | Operational Recipe | RECIPE | 70 |
| [Emmanue4888/qwen38-3090-s...](https://github.com/Emmanue4888/qwen38-3090-sglang) | 0 | Qwen3.8-27B | SGLang | RTX 3090 | AWQ INT4, INT4 | Operational Recipe | RECIPE | 70 |
| [rittim/qwen38-flash-next-...](https://github.com/rittim/qwen38-flash-next-thor-vllm) | 0 | Qwen3.8-Flash-Next | vLLM | DGX Spark (GB10) | NVFP4 | Operational Recipe | RECIPE | 70 |
| [SRSWTI/knivesysl](https://github.com/SRSWTI/knivesysl) | 0 | Qwen3.8-27B | Custom OSS Runtime | RTX 5090 | Unspecified / FP16 | 256k context | KERNEL | 70 |
| [amasu/dgx-spark-qwen38](https://github.com/amasu/dgx-spark-qwen38) | 11 | Qwen3.8-27B | vLLM, SGLang | DGX Spark (GB10) | FP8 | Operational Recipe | RECIPE | 70 |
| [ARahim3/kaggle-tpu-lab](https://github.com/ARahim3/kaggle-tpu-lab) | 0 | Qwen3.8-27B | Custom OSS Runtime | Kaggle TPU | Unspecified / FP16 | 130 tok/s, 262k con... | MEASURED | 65 |
| [Bluff-pretend532/qwen3.8-...](https://github.com/Bluff-pretend532/qwen3.8-Flash-DGX) | 0 | Qwen3.8-Flash-Next | vLLM | DGX Spark (GB10) | Unspecified / FP16 | Operational Recipe | RECIPE | 65 |
| [jkingston/qwen3.8-27b-dgx...](https://github.com/jkingston/qwen3.8-27b-dgx-spark) | 0 | Qwen3.8-27B | vLLM | DGX Spark (GB10) | Unspecified / FP16 | Empirical Benchmark... | EVALUATION | 65 |
| [coffeegrind123/instantcoffee](https://github.com/coffeegrind123/instantcoffee) | 0 | Qwen3.8-27B | llama.cpp | RTX 4090, Kaggle TPU | Unspecified / FP16 | Operational Recipe | RECIPE | 65 |
| [michele1967lux/flash-next...](https://github.com/michele1967lux/flash-next-3090ti-r9700) | 0 | Qwen3.8-Flash-Next | llama.cpp, Vulkan/RADV | AMD RDNA4, RTX 3090 | Unspecified / FP16 | Empirical Benchmark... | EVALUATION | 65 |
| [alexellis/glm-5.3-flash-4...](https://github.com/alexellis/glm-5.3-flash-4x-dgx-spark-switchless) | 52 | GLM-5.3-Flash | Custom OSS Runtime | DGX Spark (GB10) | NVFP4 | Operational Recipe | RECIPE | 65 |
| [lynseyaggregate8337/Qwen3...](https://github.com/lynseyaggregate8337/Qwen3.8-Flash-Next-Dual-DGX-Sparks) | 0 | Qwen3.8-Flash-Next | SGLang | DGX Spark (GB10) | NVFP4 | Operational Recipe | RECIPE | 60 |
| [dg1kjd/sglang-v100-sxm2-q...](https://github.com/dg1kjd/sglang-v100-sxm2-qwen3.8-flash-next) | 0 | Qwen3.8-Flash-Next | SGLang | Tesla V100 | NVFP4 | 262K context | RECIPE | 60 |
| [davidcanar/vllm-strix-halo](https://github.com/davidcanar/vllm-strix-halo) | 0 | GLM-5.3-Flash | vLLM | AMD Strix Halo (gfx1151), Apple Silicon | Unspecified / FP16 | Operational Recipe | RECIPE | 55 |
| [mratsim/sglang-qwen38fn-s...](https://github.com/mratsim/sglang-qwen38fn-sm120-turbo) | 4 | Qwen3.8-Flash-Next | SGLang | RTX Pro Enterprise | Unspecified / FP16 | Operational Recipe | RECIPE | 55 |
| [Reederey87/glm53-flash-ex...](https://github.com/Reederey87/glm53-flash-exl3-2x-dgx-spark) | 49 | GLM-5.3-Flash | Custom OSS Runtime | DGX Spark (GB10) | EXL3 | Operational Recipe | RECIPE | 55 |
| [ciffelia/llama.cpp-glm-5....](https://github.com/ciffelia/llama.cpp-glm-5.3-flash) | 0 | GLM-5.3-Flash | llama.cpp | NVIDIA B200 (Blackwell) | Unspecified / FP16 | Operational Recipe | RECIPE | 55 |
| [Oasisincoherence641/glm-5.3](https://github.com/Oasisincoherence641/glm-5.3) | 0 | GLM-5.3 | Custom OSS Runtime | Apple Silicon | Unspecified / FP16 | Empirical Benchmark... | EVALUATION | 50 |
| [WeZZard/dgx-spark-bench](https://github.com/WeZZard/dgx-spark-bench) | 0 | Qwen3.8-Flash-Next | Custom OSS Runtime | DGX Spark (GB10) | Unspecified / FP16 | Empirical Benchmark... | EVALUATION | 50 |
| [Dyluhn/R9V](https://github.com/Dyluhn/R9V) | 17 | Qwen3.8-Flash-Next | ROCm | AMD RDNA4 | Unspecified / FP16 | Operational Recipe | RECIPE | 45 |
| [Apzoldek/GLM-5.3-Flash-NV...](https://github.com/Apzoldek/GLM-5.3-Flash-NVFP4-Dual-DGX-Spark) | 0 | GLM-5.3-Flash | Custom OSS Runtime | DGX Spark (GB10) | NVFP4 | Operational Recipe | RECIPE | 45 |
| [Lhworl5977/glm-5.3-flash-...](https://github.com/Lhworl5977/glm-5.3-flash-exl3-2x-spark) | 0 | GLM-5.3-Flash | Custom OSS Runtime | DGX Spark (GB10) | EXL3 | Operational Recipe | RECIPE | 45 |
| [yanun0323/Whallm](https://github.com/yanun0323/Whallm) | 69 | Qwen3.8 | Custom OSS Runtime | Apple Silicon | FP8 | Operational Recipe | RECIPE | 45 |
| [Douda/qwen3.8-vllm-single...](https://github.com/Douda/qwen3.8-vllm-single-r9700) | 0 | Qwen3.8-27B | vLLM | AMD RDNA4 | Unspecified / FP16 | Operational Recipe | RECIPE | 45 |
| [kiojuvr/glm53-flash-mlx](https://github.com/kiojuvr/glm53-flash-mlx) | 0 | GLM-5.3-Flash | MLX | Apple Silicon | Unspecified / FP16 | Operational Recipe | RECIPE | 45 |
| [pctablet505/glm53-flash-s...](https://github.com/pctablet505/glm53-flash-single-gpu) | 0 | GLM-5.3-Flash | Custom OSS Runtime | RTX Pro Enterprise, NVIDIA B200 (Blackwell) | NVFP4 | Operational Recipe | RECIPE | 45 |
| [constricted-astronavigati...](https://github.com/constricted-astronavigation5515/qwen38-flash-next-spark) | 0 | Qwen3.8-Flash-Next | Custom OSS Runtime | DGX Spark (GB10) | Unspecified / FP16 | 262K context | RECIPE | 40 |
| [HaberstrohSystems/qwen3.8...](https://github.com/HaberstrohSystems/qwen3.8-flash-next-24gb-sglang) | 0 | Qwen3.8-Flash-Next | SGLang | Custom Hardware | Sub-3bpw Quant | Operational Recipe | RECIPE | 40 |
| [markldn/llama.cpp-qwen4ex...](https://github.com/markldn/llama.cpp-qwen4exp-lru-async) | 0 | Qwen3.8-Flash-Next | llama.cpp | Custom Hardware | Unspecified / FP16 | Operational Recipe | RECIPE | 35 |
| [brandonmmusic-max/glm-5.3...](https://github.com/brandonmmusic-max/glm-5.3-exl3-k3-dcp4) | 0 | GLM-5.3 | Custom OSS Runtime | Custom Hardware | EXL3 | Operational Recipe | RECIPE | 35 |

## Subagent Cohort Breakdown

### Cohort 1 (13 Repositories)

- **[hasso5703/dgx-spark-qwen38](https://github.com/hasso5703/dgx-spark-qwen38)** (`Qwen3.8-27B`): Fastest measured Qwen3.8-27B config for DGX Spark (GB10): SGLang + NVFP4 + DFlash2, deterministic boots, 50 tok/s greedy median, 148 tok/s at 8 streams, 258 at 32. One command, everything pinned, lossless.
  - *Hardware*: DGX Spark (GB10) | *Engine*: SGLang | *Quant*: NVFP4 | *Proof*: 50 tok/s
  - *Mechanisms*: DFlash2 Speculative Decoding
- **[albond/SingleSpark-Qwen3.8-Flash-Next](https://github.com/albond/SingleSpark-Qwen3.8-Flash-Next)** (`Qwen3.8-Flash-Next`): Experimental: Qwen3.8-Flash-Next (180B/6B MoE) on one NVIDIA DGX Spark at 44 tok/s — 4-bit NVFP4 + FP8 hybrid under vLLM, with the quality cost measured and published
  - *Hardware*: DGX Spark (GB10) | *Engine*: vLLM | *Quant*: NVFP4, FP8, INT4 | *Proof*: 44 tok/s
- **[shyringo/qwen3.8-flash-next-in-c](https://github.com/shyringo/qwen3.8-flash-next-in-c)** (`Qwen3.8-Flash-Next`): Run Qwen3.8-Flash-Next (125B-A6B + 51B PLE) on a single laptop CPU at 5.03 token/s. Native C, no GPU or Python. | 单颗笔记本 CPU 运行 Qwen3.8-Flash-Next（125B-A6B + 51B PLE），常驻聊天 5.03 token/s；原生 C 语言，无需 GPU 或 Python。
  - *Hardware*: Host CPU | *Engine*: Native C | *Quant*: Unspecified / FP16 | *Proof*: 5.03 token/s
  - *Mechanisms*: PLE / N-gram Embedding Accelerator
- **[MindLab-Research/ferrite](https://github.com/MindLab-Research/ferrite)** (`GLM-5.3-Flash`): ferrite — native GLM-5.3-Flash inference engine in Rust: PDAF disaggregation, composable TP/CP/DCP axis algebra, exact MHC, WYF chunkwise GatedDeltaNet, sm_100a CUDA kernels. CPU golden-standard verified (74 tests).
  - *Hardware*: Host CPU | *Engine*: ferrite (Rust) | *Quant*: Unspecified / FP16 | *Proof*: Operational Recipe
  - *Mechanisms*: Next-Gen Arch sm_100/sm_120 Tuning
- **[ormandj/sglang-glm53-flash-sm120](https://github.com/ormandj/sglang-glm53-flash-sm120)** (`GLM-5.3-Flash`): SGLang GLM-5.3-Flash on 2x RTX PRO 6000 Blackwell (SM120, TP2): C4 @ 507k-token KV pool, W4A16+FP8-mix quant, HiCache, native MTP — reproducible image, producers, bench harness, receipts
  - *Hardware*: RTX Pro Enterprise, NVIDIA B200 (Blackwell) | *Engine*: SGLang | *Quant*: FP8 | *Proof*: Operational Recipe
  - *Mechanisms*: Multi-Token Prediction (MTP), Next-Gen Arch sm_100/sm_120 Tuning
- **[carloslfu/slotstream](https://github.com/carloslfu/slotstream)** (`Qwen3.8-Flash-Next`): Run Qwen3.8-Flash-Next (125B MoE, 104 GB at 4-bit) on Macs with a fraction of that RAM by streaming experts from SSD. MLX + Swift, Ollama-compatible API.
  - *Hardware*: Apple Silicon | *Engine*: MLX | *Quant*: INT4 | *Proof*: Operational Recipe
  - *Mechanisms*: NVMe/SSD Tiered Weight & Cache Streaming
- **[amasu/dgx-spark-qwen38](https://github.com/amasu/dgx-spark-qwen38)** (`Qwen3.8-27B`): Docker compose configs for serving Qwen3.8-27B on a DGX Spark (GB10): vLLM+MTP current, SGLang+DSPARK + FP8 rollback stacks
  - *Hardware*: DGX Spark (GB10) | *Engine*: vLLM, SGLang | *Quant*: FP8 | *Proof*: Operational Recipe
  - *Mechanisms*: Multi-Token Prediction (MTP)
- **[coffeegrind123/instantcoffee](https://github.com/coffeegrind123/instantcoffee)** (`Qwen3.8-27B`): Reproducible Docker Compose stack running Qwen3.8-27B on a single RTX 4090: llama.cpp with MTP + n-gram speculative decoding behind the forge guardrail proxy, driven by the pi coding agent. Plus rtk bash-output filtering, MCP + browser tools, a Matrix channel, and switchable coding/prose modes.
  - *Hardware*: RTX 4090, Kaggle TPU | *Engine*: llama.cpp | *Quant*: Unspecified / FP16 | *Proof*: Operational Recipe
  - *Mechanisms*: Multi-Token Prediction (MTP), PLE / N-gram Embedding Accelerator
- **[dg1kjd/sglang-v100-sxm2-qwen3.8-flash-next](https://github.com/dg1kjd/sglang-v100-sxm2-qwen3.8-flash-next)** (`Qwen3.8-Flash-Next`): Serve Qwen3.8-Flash-Next (125B MoE, NVFP4) at 262K context on 4x Tesla V100-SXM2-32GB — a Volta (sm70) port of SGLang for agentic coding. Native OpenAI + Anthropic APIs.
  - *Hardware*: Tesla V100 | *Engine*: SGLang | *Quant*: NVFP4 | *Proof*: 262K context
- **[ciffelia/llama.cpp-glm-5.3-flash](https://github.com/ciffelia/llama.cpp-glm-5.3-flash)** (`GLM-5.3-Flash`): llama.cpp container image for running GLM-5.3-Flash on SM120 Blackwell GPUs with vision support.
  - *Hardware*: NVIDIA B200 (Blackwell) | *Engine*: llama.cpp | *Quant*: Unspecified / FP16 | *Proof*: Operational Recipe
  - *Mechanisms*: Next-Gen Arch sm_100/sm_120 Tuning
- **[Apzoldek/GLM-5.3-Flash-NVFP4-Dual-DGX-Spark](https://github.com/Apzoldek/GLM-5.3-Flash-NVFP4-Dual-DGX-Spark)** (`GLM-5.3-Flash`): Deploy GLM-5.3-Flash NVFP4 on dual DGX Spark nodes for accelerated inference, scaling large language model serving with reduced latency and optimized throughput.
  - *Hardware*: DGX Spark (GB10) | *Engine*: Custom OSS Runtime | *Quant*: NVFP4 | *Proof*: Operational Recipe
- **[kiojuvr/glm53-flash-mlx](https://github.com/kiojuvr/glm53-flash-mlx)** (`GLM-5.3-Flash`): Run GLM-5.3-Flash on M3Ultra512GB
  - *Hardware*: Apple Silicon | *Engine*: MLX | *Quant*: Unspecified / FP16 | *Proof*: Operational Recipe
- **[markldn/llama.cpp-qwen4exp-lru-async](https://github.com/markldn/llama.cpp-qwen4exp-lru-async)** (`Qwen3.8-Flash-Next`): llama.cpp fork: Qwen3.8-Flash-Next (qwen4exp) with an async, device-side GPU-resident LRU MoE expert cache
  - *Hardware*: Custom Hardware | *Engine*: llama.cpp | *Quant*: Unspecified / FP16 | *Proof*: Operational Recipe
  - *Mechanisms*: Device-Side GPU LRU MoE Expert Cache

### Cohort 2 (13 Repositories)

- **[adrienbrault/qwen3.8-27b-rtx5090](https://github.com/adrienbrault/qwen3.8-27b-rtx5090)** (`Qwen3.8-27B`): Qwen3.8-27B on RTX 5090s — 262K ctx, 1.5M-token KV pool, ~300 t/s code decode. NVFP4 + vLLM + sm120 patches, reproducible.
  - *Hardware*: RTX 5090 | *Engine*: vLLM | *Quant*: NVFP4 | *Proof*: 300 t/s, 262K ctx
  - *Mechanisms*: Next-Gen Arch sm_100/sm_120 Tuning
- **[thadreber-web/llama.cpp-qwen38-flash-next](https://github.com/thadreber-web/llama.cpp-qwen38-flash-next)** (`Qwen3.8-Flash-Next`): Qwen3.8-Flash-Next on NVIDIA GB10: CUDA graph cache fix (+12-14% generation), on-disk PLE, MTP speculative decode, TurboQuant KV. Measured results including the failures.
  - *Hardware*: DGX Spark (GB10) | *Engine*: llama.cpp | *Quant*: Unspecified / FP16 | *Proof*: Empirical Benchmarks Documented
  - *Mechanisms*: Multi-Token Prediction (MTP), PLE / N-gram Embedding Accelerator, CUDA Graph Cache Optimization
- **[airawatraj/dgx-spark-qwen38-flash-agent](https://github.com/airawatraj/dgx-spark-qwen38-flash-agent)** (`Qwen3.8-Flash-Next`): Qwen3.8-Flash-Next as Cogni-Brain on NVIDIA DGX Spark (GB10): HashK GPU PLE + SGLang NEXTN, 36.8 tok/s code, 100/100 tool-eval, 262K context.
  - *Hardware*: DGX Spark (GB10) | *Engine*: SGLang | *Quant*: Unspecified / FP16 | *Proof*: 36.8 tok/s, 262K context
  - *Mechanisms*: PLE / N-gram Embedding Accelerator
- **[feifeidu-max/Qwen3.8-FlashNext-Optimization](https://github.com/feifeidu-max/Qwen3.8-FlashNext-Optimization)** (`Qwen3.8-Flash-Next`): ik_llama.cpp qwen4exp (Qwen3.8-Flash-Next) hybrid GPU/RAM inference optimization on 2x RTX 8000: profiling, patches (PLE gather hoist, index clamp, ABS/SGN registration) and benchmarks
  - *Hardware*: RTX Pro Enterprise | *Engine*: llama.cpp | *Quant*: Unspecified / FP16 | *Proof*: Empirical Benchmarks Documented
  - *Mechanisms*: PLE / N-gram Embedding Accelerator
- **[himorishige/glm53-flash-2x-dgx-spark-recipe](https://github.com/himorishige/glm53-flash-2x-dgx-spark-recipe)** (`GLM-5.3-Flash`): GLM-5.3-Flash (320B-A18B, NVFP4) on 2x NVIDIA DGX Spark with vLLM TP=2: config-as-code, measured numbers, pitfalls
  - *Hardware*: DGX Spark (GB10) | *Engine*: vLLM | *Quant*: NVFP4 | *Proof*: Empirical Benchmarks Documented
- **[Emmanue4888/qwen38-3090-sglang](https://github.com/Emmanue4888/qwen38-3090-sglang)** (`Qwen3.8-27B`): Run Qwen3.8-27B on RTX 3090s with one-command SGLang container, INT4 AWQ, speculative decoding, and multimodal support.
  - *Hardware*: RTX 3090 | *Engine*: SGLang | *Quant*: AWQ INT4, INT4 | *Proof*: Operational Recipe
  - *Mechanisms*: Speculative Decoding
- **[ARahim3/kaggle-tpu-lab](https://github.com/ARahim3/kaggle-tpu-lab)** (`Qwen3.8-27B`): Qwen3.8-27B on a free Kaggle TPU: OpenAI-compatible endpoint, 262k context, ~130 tok/s, works with Claude Code, Codex, Opencode and Pi.
  - *Hardware*: Kaggle TPU | *Engine*: Custom OSS Runtime | *Quant*: Unspecified / FP16 | *Proof*: 130 tok/s, 262k context
- **[michele1967lux/flash-next-3090ti-r9700](https://github.com/michele1967lux/flash-next-3090ti-r9700)** (`Qwen3.8-Flash-Next`): Qwen3.8-Flash-Next (125B MoE, qwen4exp) on llama.cpp with one RTX 3090 Ti + one Radeon AI PRO R9700 and 40GB of RAM: measured numbers, the CUDA+Vulkan recipe, and the traps
  - *Hardware*: AMD RDNA4, RTX 3090 | *Engine*: llama.cpp, Vulkan/RADV | *Quant*: Unspecified / FP16 | *Proof*: Empirical Benchmarks Documented
- **[davidcanar/vllm-strix-halo](https://github.com/davidcanar/vllm-strix-halo)** (`GLM-5.3-Flash`): Run vLLM with GLM-5.3-Flash and DeepSeek-V4-Flash on 2x AMD Strix Halo (gfx1151) machines, tensor-parallel over Thunderbolt RoCE-RDMA
  - *Hardware*: AMD Strix Halo (gfx1151), Apple Silicon | *Engine*: vLLM | *Quant*: Unspecified / FP16 | *Proof*: Operational Recipe
  - *Mechanisms*: Thunderbolt RoCE-RDMA TP Fabric
- **[Oasisincoherence641/glm-5.3](https://github.com/Oasisincoherence641/glm-5.3)** (`GLM-5.3`): Audit Z.AI's GLM-5.3 cyber-engine with the official desktop tool for Windows and macOS—fast, precise, benchmark-breaking.
  - *Hardware*: Apple Silicon | *Engine*: Custom OSS Runtime | *Quant*: Unspecified / FP16 | *Proof*: Empirical Benchmarks Documented
- **[Lhworl5977/glm-5.3-flash-exl3-2x-spark](https://github.com/Lhworl5977/glm-5.3-flash-exl3-2x-spark)** (`GLM-5.3-Flash`): Run GLM-5.3-Flash, a 320B-parameter model, on two NVIDIA DGX Spark desktops as a private OpenAI-compatible API with 1.3M-token context.
  - *Hardware*: DGX Spark (GB10) | *Engine*: Custom OSS Runtime | *Quant*: EXL3 | *Proof*: Operational Recipe
- **[pctablet505/glm53-flash-single-gpu](https://github.com/pctablet505/glm53-flash-single-gpu)** (`GLM-5.3-Flash`): Running GLM-5.3-Flash (320B/18B-active, NVFP4) on ONE RTX PRO 6000 Blackwell: a 181 GiB model on a 96 GiB card. Measurements, the gather kernel, and the negative results.
  - *Hardware*: RTX Pro Enterprise, NVIDIA B200 (Blackwell) | *Engine*: Custom OSS Runtime | *Quant*: NVFP4 | *Proof*: Operational Recipe
- **[brandonmmusic-max/glm-5.3-exl3-k3-dcp4](https://github.com/brandonmmusic-max/glm-5.3-exl3-k3-dcp4)** (`GLM-5.3`): Research-only full GLM-5.3 EXL3 uniform-K3 TP4/DCP4/MTP3 Jovian runtime recipe
  - *Hardware*: Custom Hardware | *Engine*: Custom OSS Runtime | *Quant*: EXL3 | *Proof*: Operational Recipe
  - *Mechanisms*: Multi-Token Prediction (MTP)

### Cohort 3 (12 Repositories)

- **[Azhu9701/ninfer-4090d](https://github.com/Azhu9701/ninfer-4090d)** (`Qwen3.8-27B`): Qwen3.8-27B on RTX 4090 D (48GB): production deployment of NInfer with MTP7 + E8 KV + NVMe disk cache, 195 tok/s decode, crash forensics for WDDM desktop GPUs
  - *Hardware*: RTX 4090 | *Engine*: NInfer | *Quant*: Unspecified / FP16 | *Proof*: 195 tok/s
  - *Mechanisms*: Multi-Token Prediction (MTP), NVMe/SSD Tiered Weight & Cache Streaming
- **[vcruz305/GLM-5.3-Flash-EXL3-K2-DGX-Spark-recipe](https://github.com/vcruz305/GLM-5.3-Flash-EXL3-K2-DGX-Spark-recipe)** (`GLM-5.3-Flash`): vLLM recipe: GLM-5.3-Flash EXL3 K2 on one DGX Spark GB10. Native MTP k=2. Measured tok/s.
  - *Hardware*: DGX Spark (GB10) | *Engine*: vLLM | *Quant*: EXL3 | *Proof*: Empirical Benchmarks Documented
  - *Mechanisms*: Multi-Token Prediction (MTP)
- **[cglab-public/dgx-spark-flashnext](https://github.com/cglab-public/dgx-spark-flashnext)** (`Qwen3.8-Flash-Next`): Serving Qwen3.8-Flash-Next (NVFP4) on DGX Spark (GB10 / sm_121) with SGLang — field notes, traps, and measured concurrency
  - *Hardware*: DGX Spark (GB10) | *Engine*: SGLang | *Quant*: NVFP4 | *Proof*: Empirical Benchmarks Documented
  - *Mechanisms*: Next-Gen Arch sm_100/sm_120 Tuning
- **[halt95/qwen38-flash-next-3090s](https://github.com/halt95/qwen38-flash-next-3090s)** (`Qwen3.8-Flash-Next`): [WIP] Qwen3.8-Flash-Next at 262K context on 4x RTX 3090 with vLLM: 43 -> ~118 tok/s single-stream. Patches and harness to follow.
  - *Hardware*: RTX 3090 | *Engine*: vLLM | *Quant*: Unspecified / FP16 | *Proof*: 118 tok/s, 262K context
- **[pangoleen/qwen3.8-27b-dgx-spark-dflash2](https://github.com/pangoleen/qwen3.8-27b-dgx-spark-dflash2)** (`Qwen3.8-27B`): Qwen3.8-27B on one DGX Spark: SGLang + DFlash2 recipe, engine image, benchmarks, data and charts. By @redp314
  - *Hardware*: DGX Spark (GB10) | *Engine*: SGLang | *Quant*: Unspecified / FP16 | *Proof*: Empirical Benchmarks Documented
  - *Mechanisms*: DFlash2 Speculative Decoding
- **[rittim/qwen38-flash-next-thor-vllm](https://github.com/rittim/qwen38-flash-next-thor-vllm)** (`Qwen3.8-Flash-Next`): Running Qwen3.8-Flash-Next NVFP4 under vLLM on a Jetson AGX Thor (sm_110): the delta over the DGX Spark recipe — concurrency + vision
  - *Hardware*: DGX Spark (GB10) | *Engine*: vLLM | *Quant*: NVFP4 | *Proof*: Operational Recipe
  - *Mechanisms*: Next-Gen Arch sm_100/sm_120 Tuning
- **[Bluff-pretend532/qwen3.8-Flash-DGX](https://github.com/Bluff-pretend532/qwen3.8-Flash-DGX)** (`Qwen3.8-Flash-Next`): Run Qwen3.8-Flash-Next on a single DGX Spark with vLLM, streaming n-gram tables from NVMe to unlock 500k-token context.
  - *Hardware*: DGX Spark (GB10) | *Engine*: vLLM | *Quant*: Unspecified / FP16 | *Proof*: Operational Recipe
  - *Mechanisms*: NVMe/SSD Tiered Weight & Cache Streaming, PLE / N-gram Embedding Accelerator
- **[alexellis/glm-5.3-flash-4x-dgx-spark-switchless](https://github.com/alexellis/glm-5.3-flash-4x-dgx-spark-switchless)** (`GLM-5.3-Flash`): GLM-5.3-Flash (NVFP4) at TP4 across 4x DGX Spark via a switchless RoCE ring + DFlash2 — reproducible recipe
  - *Hardware*: DGX Spark (GB10) | *Engine*: Custom OSS Runtime | *Quant*: NVFP4 | *Proof*: Operational Recipe
  - *Mechanisms*: DFlash2 Speculative Decoding, Thunderbolt RoCE-RDMA TP Fabric
- **[mratsim/sglang-qwen38fn-sm120-turbo](https://github.com/mratsim/sglang-qwen38fn-sm120-turbo)** (`Qwen3.8-Flash-Next`): Qwen3.8-Flash-Next optimized for 1x RTX Pro 6000
  - *Hardware*: RTX Pro Enterprise | *Engine*: SGLang | *Quant*: Unspecified / FP16 | *Proof*: Operational Recipe
  - *Mechanisms*: Next-Gen Arch sm_100/sm_120 Tuning
- **[WeZZard/dgx-spark-bench](https://github.com/WeZZard/dgx-spark-bench)** (`Qwen3.8-Flash-Next`): Benchmark & tuning lab for LLM serving on a 2-node NVIDIA DGX Spark cluster (Qwen3.8-Flash-Next, DeepSeek-V4-Flash-0731, GLM-5.3-Flash)
  - *Hardware*: DGX Spark (GB10) | *Engine*: Custom OSS Runtime | *Quant*: Unspecified / FP16 | *Proof*: Empirical Benchmarks Documented
- **[yanun0323/Whallm](https://github.com/yanun0323/Whallm)** (`Qwen3.8`): DeepSeek-V4-Flash-0731  284B inference in ~30 GB of RAM / Qwen3.8-Next-Flash-FP8 inference in ~20 GB of RAM on any M-series MacBook
  - *Hardware*: Apple Silicon | *Engine*: Custom OSS Runtime | *Quant*: FP8 | *Proof*: Operational Recipe
- **[constricted-astronavigation5515/qwen38-flash-next-spark](https://github.com/constricted-astronavigation5515/qwen38-flash-next-spark)** (`Qwen3.8-Flash-Next`): Run 180B-parameter Qwen3.8-Flash-Next on a DGX Spark with full 262K context using SSD-backed memory for ultimate desktop AI performance.
  - *Hardware*: DGX Spark (GB10) | *Engine*: Custom OSS Runtime | *Quant*: Unspecified / FP16 | *Proof*: 262K context
  - *Mechanisms*: NVMe/SSD Tiered Weight & Cache Streaming

### Cohort 4 (12 Repositories)

- **[marksunner/dgx-spark-glm52](https://github.com/marksunner/dgx-spark-glm52)** (`GLM-5.3`): GLM 5.2 → 5.3 on 4× DGX Spark — the complete journey from unboxing to first inference, now upgraded to GLM 5.3 (Sept 2026). 753B params, 200K context, MTP, ~26 tok/s. Plus: What Is Fabric? (lossless RoCE). Built on tonyd2wild's QuantTrio recipe.
  - *Hardware*: DGX Spark (GB10) | *Engine*: Custom OSS Runtime | *Quant*: Unspecified / FP16 | *Proof*: 26 tok/s, 200K context
  - *Mechanisms*: Multi-Token Prediction (MTP), PLE / N-gram Embedding Accelerator, Thunderbolt RoCE-RDMA TP Fabric
- **[gitcommit90/glm-5.3-one-spark](https://github.com/gitcommit90/glm-5.3-one-spark)** (`GLM-5.3-Flash`): GLM-5.3-Flash at 64 tok/s on one DGX Spark — TP1 EXL3 2.05 + DFlash2 K7
  - *Hardware*: DGX Spark (GB10) | *Engine*: Custom OSS Runtime | *Quant*: EXL3, Sub-3bpw Quant | *Proof*: 64 tok/s
  - *Mechanisms*: DFlash2 Speculative Decoding
- **[Gr33n93/llama.cpp-qwen3.8-flash-next-mtp-vulkan](https://github.com/Gr33n93/llama.cpp-qwen3.8-flash-next-mtp-vulkan)** (`Qwen3.8-Flash-Next`): Validated Qwen3.8 Flash Next MTP patch set for llama.cpp on AMD Strix Halo Vulkan/RADV
  - *Hardware*: AMD Strix Halo (gfx1151) | *Engine*: llama.cpp, Vulkan/RADV | *Quant*: Unspecified / FP16 | *Proof*: Operational Recipe
  - *Mechanisms*: Multi-Token Prediction (MTP)
- **[punkjazz-labs/glm-5.3-flash-exl3-4x-dgx-spark](https://github.com/punkjazz-labs/glm-5.3-flash-exl3-4x-dgx-spark)** (`GLM-5.3-Flash`): GLM-5.3-Flash EXL3 (4 bpw) on four NVIDIA DGX Sparks with vLLM TP4: measured production recipe, autoresearch tuning, watchdog, benchmark and every receipt
  - *Hardware*: DGX Spark (GB10) | *Engine*: vLLM | *Quant*: EXL3, INT4 | *Proof*: Empirical Benchmarks Documented
- **[PieBru/Qwen-3.8-27B_Strix-Halo_gfx1151](https://github.com/PieBru/Qwen-3.8-27B_Strix-Halo_gfx1151)** (`Qwen3.8-27B`): Qwen3.8-27B + DFlash2 spec-decode on Strix Halo (gfx1151) via the strix-halo llama.cpp fork — quick start, configs, benchmarks,  ...
  - *Hardware*: AMD Strix Halo (gfx1151) | *Engine*: llama.cpp | *Quant*: Unspecified / FP16 | *Proof*: Empirical Benchmarks Documented
  - *Mechanisms*: DFlash2 Speculative Decoding
- **[SRSWTI/knivesysl](https://github.com/SRSWTI/knivesysl)** (`Qwen3.8-27B`): bare-cuda fp6 inference engine for qwen3.8-27b on one rtx 5090 — sm120 tensor-core kernels, 256k context, near-lossless perf too!
  - *Hardware*: RTX 5090 | *Engine*: Custom OSS Runtime | *Quant*: Unspecified / FP16 | *Proof*: 256k context
  - *Mechanisms*: Next-Gen Arch sm_100/sm_120 Tuning
- **[jkingston/qwen3.8-27b-dgx-spark](https://github.com/jkingston/qwen3.8-27b-dgx-spark)** (`Qwen3.8-27B`): Pinned vLLM runtime and benchmarks for Qwen3.8-27B on NVIDIA DGX Spark
  - *Hardware*: DGX Spark (GB10) | *Engine*: vLLM | *Quant*: Unspecified / FP16 | *Proof*: Empirical Benchmarks Documented
- **[lynseyaggregate8337/Qwen3.8-Flash-Next-Dual-DGX-Sparks](https://github.com/lynseyaggregate8337/Qwen3.8-Flash-Next-Dual-DGX-Sparks)** (`Qwen3.8-Flash-Next`): Deploy a 176B-parameter NVFP4 MoE model across two DGX Sparks with SGLang TP2 for ultra-fast inference.
  - *Hardware*: DGX Spark (GB10) | *Engine*: SGLang | *Quant*: NVFP4 | *Proof*: Operational Recipe
- **[Reederey87/glm53-flash-exl3-2x-dgx-spark](https://github.com/Reederey87/glm53-flash-exl3-2x-dgx-spark)** (`GLM-5.3-Flash`): GLM-5.3-Flash EXL3 (320B MoE) on 2x NVIDIA DGX Spark — production serving kit, 1M context, 97%+ multi-session prefix caching, DFlash2 spec decode
  - *Hardware*: DGX Spark (GB10) | *Engine*: Custom OSS Runtime | *Quant*: EXL3 | *Proof*: Operational Recipe
  - *Mechanisms*: DFlash2 Speculative Decoding
- **[Dyluhn/R9V](https://github.com/Dyluhn/R9V)** (`Qwen3.8-Flash-Next`): Shape-specialized ROCm inference for Qwen3.8 Flash Next on RDNA4
  - *Hardware*: AMD RDNA4 | *Engine*: ROCm | *Quant*: Unspecified / FP16 | *Proof*: Operational Recipe
- **[Douda/qwen3.8-vllm-single-r9700](https://github.com/Douda/qwen3.8-vllm-single-r9700)** (`Qwen3.8-27B`): Qwen3.8-27B via vLLM on a single AMD Radeon AI PRO R9700 (gfx1201, 32 GiB) — bare, script-free setup guide, written to be executed by an AI agent or by hand
  - *Hardware*: AMD RDNA4 | *Engine*: vLLM | *Quant*: Unspecified / FP16 | *Proof*: Operational Recipe
- **[HaberstrohSystems/qwen3.8-flash-next-24gb-sglang](https://github.com/HaberstrohSystems/qwen3.8-flash-next-24gb-sglang)** (`Qwen3.8-Flash-Next`): Qwen3.8-Flash-Next (176B MoE, 6B active) at 2.5 bpw on one 24 GB GPU with 32 GB RAM: SGLang serving patch, kernels, measurement tools and engineering log
  - *Hardware*: Custom Hardware | *Engine*: SGLang | *Quant*: Sub-3bpw Quant | *Proof*: Operational Recipe

