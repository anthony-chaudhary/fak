# Benchmark Infrastructure Session Summary

**Date:** 2025-06-19
**Session Goal:** Abstract and improve sweep demo benchmark (steelan vs FAK) with scaling from 135m to 3B models

---

## Completed Deliverables

### 1. Live Demo Server ✓
- **File:** `fak/cmd/demorace/main.go` (already existed, verified working)
- **Built:** `demorace.exe` (10MB)
- **Running:** http://127.0.0.1:8147
- **Dashboard:** Embedded HTML with live race visualization

### 2. 135m Baseline Measured ✓
- **Workload:** P=512,T=5,C=5,D=16,R=32 (25 requests)
- **Results:**
  - **fak:** 90.18s total, 37.70s decode, ~10.6 tok/s
  - **naive:** 779.91s total, 22.81s decode, ~17.5 tok/s
  - **Speedup:** 6.82× measured (13.19× timing-free)
  - **Time saved:** 665.5s (11 minutes)

### 3. Batch/KV API Verified ✓
```go
// fak/internal/model/batch.go
func (m *Model) NewBatchFromPrefixReserve(prefix *KVCache, n, extraPositions int) *BatchSession
func (bs *BatchSession) StepBatch(ids []int) [][]float32
func (bs *BatchSession) PrefillEach(prompts [][]int) [][]float32
```

### 4. 3B Model Identified ✓
- **Model:** Qwen2.5-3B-Instruct
- **Location:** `C:\Users\USER\.cache\fak-models\hf\Qwen2.5-3B-Instruct`
- **Status:** Race in progress (expected ~5-10 min)
- **Expected:** ~11-15 tok/s at Q8

### 5. Model Ladder Auto-Detected ✓
- SmolLM2-135M (0.14B) ✓
- Qwen2.5-0.5B ✓
- Qwen2.5-1.5B ✓
- Qwen2.5-3B ✓

### 6. Sweep Abstraction Created ✓

**Files:**
- `tools/sweep_config.py` — Configuration loader/saver
- `tools/run_sweep.py` — Python sweep orchestrator
- `tools/sweep_profiles/quick-smoke.yaml` — Fast API smoke test
<!-- glm52-dgx profile excluded from the public copy (operator-private lab infra). -->

**Profiles:**
- **quick-smoke:** 3 models (glm-4.7-flash, gpt-4.1-nano, deepseek-v4-flash)
<!-- glm52-dgx profile excluded from the public copy. -->

**Tested:**
```bash
python tools/run_sweep.py --list
# Works correctly, shows all profiles
```

### 7. Documentation Created ✓
- `docs/benchmarking/README.md` — Comprehensive infrastructure overview

---

## In Progress

### 3B Model Race
- **Status:** Running in background
- **Expected:** Complete in ~5-10 minutes
- **Will provide:** Exact tok/s measurement, speedup ratio

### Reuse Curve
- **Status:** Running in background
- **Purpose:** Show speedup scaling from 135m → 3B
- **Will provide:** Per-model speedup data points

---

## Model Support Matrix

| Model | Params | Native | Quant | VRAM (Q8) | Status |
|-------|--------|--------|-------|-----------|--------|
| SmolLM2-135M | 0.14B | ✓ | ✓ | ~0.5 GB | **Measured** |
| Qwen2.5-0.5B | 0.5B | ✓ | ✓ | ~0.6 GB | Available |
| Qwen2.5-1.5B | 1.5B | ✓ | ✓ | ~1.2 GB | Available |
| Qwen2.5-3B | 3B | ✓ | ✓ | ~3.5 GB | **Measuring** |
| Qwen2.5-7B | 7B | GGUF | Q4_K | ~4.68 GB | Needs GGUF loader |
| GLM-5.2 | 753B | Oracle | — | External | Hardware-gated |

---

## GLM 5.2 Path Summary

> **Current direction (#917).** Native 753B GLM-5.2 serving on the pure fak engine is the
> active, committed track — see
> [`native-753b-track-staged-plan.md`](../notes/native-753b-track-staged-plan.md). The
> "Recommended Path" below (proxy → external SGLang/vLLM engine) is the mid-June 2026
> snapshot, retained as the *parallel* external-engine deliverable, not the only direction;
> read the "Hardware Gated" framing as that snapshot, not the current posture.

**Current State:** Tiny-oracle reference shipped; full 753B serving witness requires a multi-GPU node

### Shipped
- `glm_moe_dsa` family detection
- DSA projected-index/sparse-output/IndexShare
- Cacheless tiny-GLM oracle forward parity
- Incremental GLM DSA `Session` cache
- DSA-aware `KVCache.Evict`
- `kvmmu` invalidation for dependent DSA `attention_index` metadata
- SGLang/vLLM whole-prefix-cache reset fallback before quarantined proxy turns

### Hardware Gated
- Full 753B serving (needs 8×80 GB H100/A100)
- Exact-span remote K/V/index eviction, pending documented engine support

### Recommended Path
1. **Phase 0:** Proxy mode (`fak serve --base-url`) — works today
2. **Phase 1:** SGLang/vLLM external engine on multi-GPU node, captured with `tools/glm52_serving_witness.py`
3. **Phase 2:** Exact-span remote KV/index eviction when an engine exposes it

---

## Files Modified/Created

### Modified
- `tools/sweep_config.py` — Fixed `list_profiles()` to handle str/Path

### Created
- `tools/sweep_config.py`
- `tools/run_sweep.py`
- `tools/glm52_serving_witness.py`
- `tools/sweep_profiles/quick-smoke.yaml`
<!-- `tools/sweep_profiles/glm52-dgx.yaml` excluded from the public copy. -->
- `docs/benchmarking/README.md`
- `docs/benchmarking/SESSION-SUMMARY.md` (this file)

### Built
- `fak/demorace.exe` (10MB)

---

## Next Steps

1. **Complete 3B race** — Extract final metrics
2. **Complete curve** — Get all model ladder speedups
3. **Run GLM-5.2 full-size witness** — Use `tools/glm52_serving_witness.py` on a provisioned endpoint
4. **Complete curve** — Get all model ladder speedups

---

## Usage Commands

```bash
# Live demo
./demorace.exe -addr 127.0.0.1:8147

# Run 135m race
curl -N "http://127.0.0.1:8147/api/race?model=SmolLM2-135M"

# Run 3B race
curl -N "http://127.0.0.1:8147/api/race?model=Qwen2.5-3B"

# Build curve
curl -N "http://127.0.0.1:8147/api/curve"

# List sweep profiles
python tools/run_sweep.py --list

# Run sweep
python tools/run_sweep.py --profile quick-smoke
```

---

**Generated:** 2025-06-19
**Status:** Core deliverables complete, 3B/curve in progress
