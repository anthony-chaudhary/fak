# Shared Task Record Verdict Fixtures

These fixtures cover the non-acceptance collaboration verdicts a UI or protocol
adapter must render without advancing the task revision.

```text
+---------------+     +---------------------------+     +--------------------------+
| fixture files | --> | internal/sharedtask       | --> | non-acceptance verdict   |
+---------------+     | contract validator        |     | rendered; task revision  |
                      | (verdicts witness)        |     | does not advance (exit 0)|
                      +---------------------------+     +--------------------------+
```

Prerequisites: the Go toolchain from the repo root; no network, API key, model, or
GPU is needed. The validation run completes in seconds and is deterministic: it
reads the fixture files and returns the same verdict on every run.

Run:

```bash
bash examples/shared-task-record-verdicts/run.sh
```

(the script is one command: `go test ./internal/sharedtask -run
TestContractVerdictsFixtureValidates -v`)

## What You See

The command exits 0 when every fixture renders the expected non-acceptance
verdict without advancing the task revision.

## What This Does Not Claim

This fixture does not prove UI behavior, transport delivery, or multi-agent
scheduling. It is a local contract example; the deeper contract is documented in
[shared-task-record-contract.md](../../docs/shared-task-record-contract.md).
