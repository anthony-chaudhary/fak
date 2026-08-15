# Native harness stack-resolution spine witness

**Date:** 2026-08-15  
**Issue:** [#6891](https://github.com/anthony-chaudhary/fak/issues/6891)  
**Surface:** `go run ./cmd/stackresolvedemo -selfcheck`

## Witness and boundary

The selfcheck drives the real `fak-stack-manifest/1` parser and resolver twice. A coding stack rooted at `harness:ponytail@r8` plus an L4/CUDA fixture resolves eight transitive components. An AWQ stack paired with an SM75 fixture refuses because its selected kernel requires `device.cuda.sm80`.

The allow receipt records a deterministic portable-kernel substitute, an unmet SM90 recommendation that does not block, and an unavailable browser tool that remains optional. The refusal traces harness -> model -> backend -> kernel -> device capability and retains the kernel authority and device-decode proof tier.

The machine-readable output is captured in [`internal/stackresolve/testdata/selfcheck-witness.json`](../../internal/stackresolve/testdata/selfcheck-witness.json). `TestCapturedSelfcheckWitnessMatches` fails on semantic drift.

This is an assembly-resolution witness over normalized provider facts. It does **not** claim that this workstation has an L4, that the fixture proves real AWQ execution, or that `ponytail` is fit outside the declared coding workload. Live evidence ingestion/preflight remain #6896/#6897; workload fitness remains #6893/#6894.

## Captured result

```text
ALLOW stack for coding-change@r1
selected: 8 components
backend:awq@1.4.2 requires -> kernel:awq-portable@0.9 via substitute
WARN RECOMMENDATION_UNMET: wants infra.gpu.sm90 — recommendation is not a launch requirement
WARN OPTIONAL_UNAVAILABLE: wants tool.browser — optional component is unavailable

REFUSE stack for coding-change@r1
blocker: device.cuda.sm80 (UNSATISFIED_REQUIREMENT)
chain: harness:ponytail@r8 -> model:coder-awq@sha256:111 -> backend:awq@1.4.2 -> kernel:awq-fast@0.9 -> device.cuda.sm80
authority: kernel; source: awq-fast@0.9; proof: device-decode
SELFCHECK PASS
```

## Comparison with manual preflight

| Question | Manual path | Spine receipt |
|---|---|---|
| What was selected? | join several manifests mentally | sorted identities plus source evidence |
| Can artifact reach a kernel on the baseline? | cross-read quant/backend/kernel/hardware docs | one transitive chain |
| Is SM90 mandatory? | infer wording from prose | typed non-blocking recommendation |
| Is fallback active? | inspect backend behavior | explicit substitute decision |
| Why does SM75 fail? | reconstruct four documents | deterministic decisive chain |
| Does this prove a live host? | easy to overread a matrix | no; fixture authority is explicit and preflight remains required |

Providers retain domain authority. `internal/stackresolve` imports no fak domain package; an architecture test parses every resolver source import to preserve that boundary.

## Reproduce

```bash
go test -count=1 ./internal/stackresolve ./cmd/stackresolvedemo
go vet ./internal/stackresolve ./cmd/stackresolvedemo
go run ./cmd/stackresolvedemo -selfcheck
go run ./cmd/stackresolvedemo -selfcheck -json
```

Manifest refusal exits `3`, validation/runtime errors exit `1`, and usage errors exit `2`. The standalone command is the smallest working spine while the shared `cmd/fak` harness surface is active peer WIP; first-class integration must land through that owner rather than collide with it.
