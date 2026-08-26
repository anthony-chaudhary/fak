# Task manager snapshot

[← Claims index](../../CLAIMS.md)


- [SHIPPED] Process-local task manager (`internal/taskmgr`): a stdlib-only `Manager` records running tasks and steps, samples the current process' Go runtime resource state (wall seconds, runtime CPU seconds when exposed, heap/sys memory, goroutines), reports per-task/per-step resource deltas, aggregates step runtime by concept bucket, and emits ETA only when a running task or step has positive progress against a known total. `fak task sample` exposes the same JSON snapshot shape for the current command process. Honest fence: this is not a durable scheduler, cross-PID monitor, or fleet oracle; it is the embeddable in-process reference fold. Witness: `go test ./internal/taskmgr`; `go test ./cmd/fak -run TestTask`; `go test ./internal/architest -run TestEveryPackageDeclaresTier`.

