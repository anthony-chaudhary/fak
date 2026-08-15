# First-class harness stack resolution witness — 2026-08-15

**Issue:** #6907  
**Surface:** `fak harness stack resolve --manifest PATH [--json]`  
**Contracts reused:** `fak-stack-manifest/1` and `fak-stack-receipt/1` from `internal/stackresolve`.

The first-class harness CLI now delegates to the same parser, backtracking resolver, receipt schema, and formatter as the #6891 spine. It adds no resolver semantics. Exit `0` means the requested stack composes, exit `3` means a deterministic refusal receipt was produced, exit `2` is CLI misuse, and exit `1` is malformed/unreadable input or an internal error.

## Captured allow path

```text
$ fak harness stack resolve --manifest internal/stackresolve/testdata/coding-stack.json
ALLOW stack for coding-change@r1
selected: 8 components
  - backend:awq@1.4.2 (serving-backend; awq-backend@1.4.2)
  - harness:ponytail@r8 (harness; ponytail@r8)
  - infra:l4-cuda12@r3 (infrastructure-baseline; l4-cuda12@r3)
  - kernel:awq-portable@0.9 (kernel-path; awq-portable@0.9)
  - model:coder-awq@sha256:111 (model-artifact; artifact:sha256:111)
  - policy:repo-reviewed@r3 (policy; repo-reviewed@r3)
  - runtime:cuda@12.8 (runtime; cuda@12.8)
  - tool:repo@r4 (tool-provider; repo-tool@r4)
WARN RECOMMENDATION_UNMET: harness:ponytail@r8 wants infra.gpu.sm90 — recommendation is not a launch requirement [coding-latency-2026-08-15]
WARN OPTIONAL_UNAVAILABLE: harness:ponytail@r8 wants tool.browser — optional component is unavailable [harness-manifest@r8]
```

## Captured refusal path

```text
$ fak harness stack resolve --manifest internal/stackresolve/testdata/awq-sm75-unsat.json
REFUSE stack for coding-change@r1
blocker: device.cuda.sm80 (UNSATISFIED_REQUIREMENT)
chain: harness:ponytail@r8 -> model:coder-awq@sha256:111 -> backend:awq@1.4.2 -> kernel:awq-fast@0.9 -> device.cuda.sm80
authority: kernel; source: awq-fast@0.9; proof: device-decode
remediation: add a provider for device.cuda.sm80
remediation: collect or refresh evidence for device.cuda.sm80
```

The binary returns exit `3` for this refusal. `go run` wraps that child status and therefore reports shell exit `1` plus `exit status 3`; the end-to-end CLI test invokes the router directly and asserts the actual `3` contract.

## Verification

```text
go test -count=1 ./cmd/fak -run '^TestHarnessStackResolve'
go vet ./cmd/fak
```

Tests cover allow text, refusal and exit `3`, malformed schema, JSON receipt schema, and nested-command usage. `cmd/stackresolvedemo` remains the stable embedded selfcheck witness; the first-class command is the operator surface.
