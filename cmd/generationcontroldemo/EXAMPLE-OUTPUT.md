# Continuous generation control demo captured output

Command, run from the repository root:

```console
go run ./cmd/generationcontroldemo -selfcheck
```

The block below is the complete deterministic stdout capture consumed by the package regression test.

<!-- BEGIN SELFCHECK OUTPUT -->
```text
{"event":"epoch_started","epoch":{"trajectory_id":"trajectory-demo-1","number":1,"owner":"planner-micro-agent","compute":{"worker":"worker-cpu-1","model":"fast-model","device":"cpu"}}}
{"event":"steering_point","directive":{"kind":"redirect","reason":"stream-rule:no-shell-delete","action":"Inspect the target with the read-only inventory tool."},"checkpoint":{"trajectory_id":"trajectory-demo-1","after_epoch":1,"accepted":"I will inspect the workspace. "}}
{"event":"epoch_started","epoch":{"trajectory_id":"trajectory-demo-1","number":2,"owner":"safety-micro-agent","compute":{"worker":"worker-gpu-7","model":"deep-model","device":"L4"}}}
{"event":"trajectory_checkpoint","checkpoint":{"trajectory_id":"trajectory-demo-1","after_epoch":2,"accepted":"I will inspect the workspace. Inventory complete; no destructive action ran."}}
SELF_CHECK_PASS continuous_generation_redirect_handoff
```
<!-- END SELFCHECK OUTPUT -->
