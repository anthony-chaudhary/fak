# Hardware Catalog

Track all benchmark machines with full specs, onboarding procedures, and baseline runs.

## Hardware Catalog

### Machine 1: Apple M3 Pro (`node-macos-a`)

- **CPU**: Apple M3 Pro (6P+6E)
- **GPU**: 18-core GPU (Metal)
- **RAM**: 36 GB unified
- **OS**: macOS Darwin arm64
- **Status**: ✅ Active
- **Benchmarks**:
  - M3-LLAMACPP-RESULTS.md
  - SESSION-VALUE-STACK-RESULTS.md
  - QWEN36-PARITY-RESULTS.md

### Machine 2: AMD Ryzen 9 9950X + Radeon RX 7600 (`desktop`)

- **CPU**: AMD Ryzen 9 9950X — 16 cores / 32 threads, x86_64, AVX-512
- **GPU**: AMD Radeon RX 7600 (8 GB, Vulkan 1.4)
- **RAM**: 256 GB
- **OS**: Windows 11 (native Vulkan)
- **Status**: ✅ Active
- **Benchmarks**:
  - GPU-QWEN-RESULTS.md
  - Vulkan backend work

### Machine 3: Intel + NVIDIA RTX 4070 Laptop (`workstation-a`)

- **CPU**: Intel, x86_64, AVX2/AVX-512
- **GPU**: NVIDIA RTX 4070 Laptop (8GB, Ada sm_89)
- **RAM**: ~15 GB (WSL2 capped)
- **OS**: Windows 11 + WSL2 Ubuntu
- **CUDA**: 12.6
- **Status**: ✅ Active
- **Benchmarks**:
  - GPU-QWEN-RESULTS.md
  - CUDA backend work

### Machine 4: GCP L4 (`gcp-g2-l4`)

- **CPU**: x86_64, 8 cores
- **GPU**: NVIDIA L4
- **RAM**: TBD
- **OS**: GCP Deep Learning VM (CUDA)
- **Status**: ✅ Active
- **Benchmarks**:
  - GCP benchmark runs

### Machine 5: A100 Datacenter Server (`a100`)

- **CPU**: x86_64, 256 cores
- **GPU**: NVIDIA A100 (spec TBD)
- **RAM**: TBD
- **OS**: Linux
- **Status**: 🕸️ Spec needed
- **Benchmarks**:
  - Multi-GPU serving work planned

## Onboarding Checklist

For each new machine:

1. Capture full hardware specs (CPU, GPU, RAM, OS)
2. Run baseline sanity tests (`fak turntax --suite turntax-happy`)
3. Run model compatibility tests (`go test ./internal/model`)
4. Document backend capabilities (CUDA, Metal, Vulkan, CPU-only)
5. Run first benchmark: SmolLM2-135M Q8 (baseline)
6. Run first benchmark: Realistic model (Qwen2.5-1.5B or larger)
7. Add to catalog (`tools/bench_catalog.py add-machine`)
8. Create machine-specific results doc

## Baseline Runs

Every machine must run:

- SmolLM2-135M Q8 (fast sanity check)
- RadixAttention benchmark (agents workload)
- Session value-stack (multi-agent session)
- Turn-tax safety floor (1→0 injections, 1→0 destructive)

## Automation

- `fak bench onboard` - One-time machine setup
- `fak bench baseline` - Run all baseline tests
- `fak bench specs` - Auto-detect and report hardware

## Related Documentation

- **Hardware Matrix**: See [`docs/HARDWARE-MATRIX.md`](../HARDWARE-MATRIX.md) for detailed hardware coverage and benchmark results
- **Benchmark Authority**: See [`BENCHMARK-AUTHORITY.md`](../../BENCHMARK-AUTHORITY.md) for single source of truth on benchmark claims
- **Benchmark Governance**: See [`BENCHMARK-GOVERNANCE.md`](../../BENCHMARK-GOVERNANCE.md) for DOS-centric benchmark process
- **Cross-Machine Infrastructure**: See [`docs/benchmark/CROSS-MACHINE-INFRASTRUCTURE.md`](benchmark/CROSS-MACHINE-INFRASTRUCTURE.md) for multi-node benchmark design

## Machine Registry

The live machine-readable catalog is maintained at [`experiments/benchmark/catalog.json`](../../experiments/benchmark/catalog.json) and managed via [`tools/bench_catalog.py`](../../tools/bench_catalog.py).