# Model/backend/precision coverage alternatives — 2026-08-10

Status: **INCOMPLETE**. Issue [#6196](https://github.com/anthony-chaudhary/fak/issues/6196) tracks real runtime and external-framework runs with independent resource and cost witnesses.

## Capability and same workload

`internal/covmatrix` exhaustively classifies every declared model family across CPU, CUDA, Metal, and Vulkan, then crosses those cells with each declared precision and surfaces stale declarations whose support claim lacks the required numeric oracle. The frozen workload is the same family roster, backend roster, precision roster, support oracle, and stale-declaration oracle for every arm.

Correctness requires exact classification of every family/backend and family/backend/precision cell, zero silently undefined cells, and exact stale-cell reporting. A compatibility page or successful load alone is not equivalent evidence. Scorecard rendering and source-drift guard tests remain separate package capabilities and are not claimed by this contract.

## Arms

| Arm | Class | Local status |
|---|---|---:|
| fak native model/backend/precision coverage matrix | native | available |
| hand-maintained support-table lookup | tuned no-feature baseline | available, misses stale declarations |
| fak + CUDA runtime witness | first-class integration | unavailable |
| fak + Metal runtime witness | first-class integration | unavailable |
| fak + Vulkan runtime witness | first-class integration | unavailable |
| vLLM supported-model matrix | external | unavailable |
| llama.cpp backend and quantization matrix | external | unavailable |
| Hugging Face Optimum hardware compatibility | external | unavailable |
| ONNX Runtime execution-provider matrix | external | unavailable |
| TensorRT-LLM support matrix | external | unavailable |

The three `fak + runtime` rows are separate because the native matrix contains distinct first-class backend paths and each must execute on its real device. A mocked adapter, copied documentation table, or CPU substitute is not a runtime witness.

## Completion evidence

Every complete arm must report cell-level precision/recall against an independently authored oracle, undefined/stale/false cells, latency and cells/second, CPU, peak RSS, input/storage/network bytes, setup and operator time, and total cost. Pin versions, hardware, drivers, model fixtures, commands, raw reports, and independent read-back.

`TestCompareLocalKeepsCoverageAlternativesExplicit` locks the arm inventory, native cross-product completeness, the baseline's missed stale check, and measurement-zero honesty for unavailable arms. `BenchmarkModelBackendPrecisionCoverageMatrix` executes the real matrix and stale scan. Five local Windows/amd64 samples were 7,705, 7,962, 7,560, 8,016, and 8,201 ns/op (median 7,962 ns/op; 27,960 B/op; 12 allocs/op). Local timing is not a cross-system ranking; no external or integration arm is ranked before the open issue carries real witnesses.
