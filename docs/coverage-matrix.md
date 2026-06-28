# Model × Backend Coverage Matrix

> Generated from the source tree by `fak coverage-matrix`.  
> Last updated: 2026-06-27 (commit SHA: dynamically sourced)

This matrix tracks the support status of every model family on every backend. A cell is **honestly supported** when it either (a) panics on disallowed inputs (a fence is present) or (b) has a proven conformance path. The `growth_debt` metric counts cells that are **silently undefined** — reachable by the dispatch but without a fence or proven behavior.

## Summary

- **Models**: 20
- **Backends**: 4
- **Total cells**: 80
- **Growth debt** (silently undefined cells): **59**

| Status | Count |
|--------|-------|
| SUPPORTED | 0 |
| PANICS | 21 |
| UNDEFINED | 59 |

## Matrix

| Model | CPU | CUDA | Metal | Vulkan |
|-------|-----|------|-------|--------|
| llama | ○ | ○ | ○ | ○ |
| smolLM2 | ○ | ○ | ○ | ○ |
| qwen | ○ | ○ | ○ | ○ |
| qwen2.5 | ○ | ○ | ○ | ○ |
| qwen3 | ○ | ○ | ○ | ○ |
| gptneox | ✗ | ✗ | ✗ | ✗ |
| falcon | ✗ | ✗ | ✗ | ✗ |
| mpt | ○ | ○ | ○ | ○ |
| stablelm | ○ | ○ | ○ | ○ |
| olmo2 | ✗ | ✗ | ✗ | ✗ |
| cohere | ✗ | ✗ | ✗ | ✗ |
| gemma | ✗ | ✗ | ✗ | ✗ |
| mixtral | ○ | ○ | ○ | ○ |
| gptoss | ○ | ○ | ○ | ○ |
| deepseek | ○ | ○ | ○ | ○ |
| mistral | ○ | ○ | ○ | ○ |
| glm | ○ | ○ | ✗ | ○ |
| minimax_m3 | ○ | ○ | ○ | ○ |
| phi3 | ○ | ○ | ○ | ○ |
| ornith | ○ | ○ | ○ | ○ |

## Legend

- **•** SUPPORTED — Has a proven conformance path (oracle or synthetic fixture)
- **✗** PANICS — Has an honest panic fence; fails loudly on disallowed inputs
- **?** UNTESTED — No fence, no proven path (treated as UNDEFINED for debt)
- **○** UNDEFINED — Silently undefined: dispatch can reach this cell with no fence

## Key Fences

### PreNorm Block Topology Fence (`internal/model/kv.go:153`)

Affects: gptneox, falcon, olmo2, cohere, gemma (PostNorm topologies)

These models use non-PreNorm block topologies (PostNorm, NoNorm, etc.). The HAL decode path, Metal prefill, and quant-batch all hardcode Llama's PreNorm wiring as a seam optimization (SEAM-0). This fence turns a silent wrong-result path into a loud panic.

Message: `"model: <path> does not yet implement BlockTopology <topology> (only PreNorm); see MODEL-ARCH-SEAM SEAM-0"`

### GLM-MoE-DSA Metal Fence (`internal/model/kv.go:195`)

Affects: glm on metal

The GLM-MoE-DSA sparse attention implementation does not yet support Metal/PrecisionPolicy paths. Only the CPU f32 resident path is wired today (compute.Backend GEMM offload is allowed).

Message: `"model: GLM-MoE-DSA Session: Metal/PrecisionPolicy paths are unwired (CPU resident DSA cache; compute.Backend GEMM offload is allowed)"`

## Oracle Presence

| Model | Oracle Status |
|-------|---------------|
| llama | ABSENT |
| smolLM2 | ABSENT |
| qwen | ABSENT |
| qwen2.5 | ABSENT |
| qwen3 | ABSENT |
| gptneox | ABSENT |
| falcon | ABSENT |
| mpt | ABSENT |
| stablelm | ABSENT |
| olmo2 | ABSENT |
| cohere | ABSENT |
| gemma | ABSENT |
| mixtral | ABSENT |
| gptoss | ABSENT |
| deepseek | ABSENT |
| mistral | ABSENT |
| glm | ABSENT |
| minimax_m3 | ABSENT |
| phi3 | ABSENT |
| ornith | ABSENT |

Note: Oracle presence scanning is currently stubbed. The actual implementation would parse `TestOptional*Oracle` test functions in `internal/model/oracle_test.go` and related files to determine which models have weight-backed HF parity tests.

## Regenerating

```bash
# View human-readable matrix
fak coverage-matrix

# View JSON payload
fak coverage-matrix --json

# Write snapshot to file
fak coverage-matrix --json --out tools/coverage_matrix.snapshot.json
```

## Integration

The `growth_debt` metric (59 cells) is wired into the unified scorecard control pane:

```bash
# View full control pane
python tools/scorecard_control_pane.py

# Re-pin baseline after debt improvements
python tools/scorecard_control_pane.py --pin
```

The metric appears in `tools/scorecard_baseline.json` under the `growth` key and is folded into the portfolio total debt. A regression in `growth_debt` (newly undefined cells) will trigger the early-warning lens even if the overall portfolio remains green.