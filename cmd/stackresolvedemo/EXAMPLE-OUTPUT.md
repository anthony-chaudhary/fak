# Harness stack resolution demo captured output

Command, run from the repository root:

```console
go run ./cmd/stackresolvedemo -selfcheck
```

The block below is the complete deterministic stdout capture consumed by the package regression test.

<!-- BEGIN SELFCHECK OUTPUT -->
```text
STACK RESOLUTION SELFCHECK
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
  backend:awq@1.4.2 requires -> kernel:awq-portable@0.9 via substitute [awq-backend@1.4.2]
  harness:ponytail@r8 requires -> model:coder-awq@sha256:111 [harness-manifest@r8]
  harness:ponytail@r8 requires -> policy:repo-reviewed@r3 [coding-change@r1]
  harness:ponytail@r8 requires -> tool:repo@r4 [harness-manifest@r8]
  kernel:awq-portable@0.9 requires -> runtime:cuda@12.8 [awq-portable@0.9]
  model:coder-awq@sha256:111 requires -> backend:awq@1.4.2 [artifact:sha256:111]
  runtime:cuda@12.8 requires -> infra:l4-cuda12@r3 [cuda@12.8]
WARN RECOMMENDATION_UNMET: harness:ponytail@r8 wants infra.gpu.sm90 — recommendation is not a launch requirement [coding-latency-2026-08-15]
WARN OPTIONAL_UNAVAILABLE: harness:ponytail@r8 wants tool.browser — optional component is unavailable [harness-manifest@r8]
REFUSE stack for coding-change@r1
blocker: device.cuda.sm80 (UNSATISFIED_REQUIREMENT)
chain: harness:ponytail@r8 -> model:coder-awq@sha256:111 -> backend:awq@1.4.2 -> kernel:awq-fast@0.9 -> device.cuda.sm80
authority: kernel; source: awq-fast@0.9; proof: device-decode
remediation: add a provider for device.cuda.sm80
remediation: collect or refresh evidence for device.cuda.sm80
SELFCHECK PASS: satisfiable stack allowed; transitive hardware dependency refused
```
<!-- END SELFCHECK OUTPUT -->
