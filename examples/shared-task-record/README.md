# Shared Task Record Fixtures

These fixtures are runnable examples for the shared task record contract. They
cover an initial task, title edits, open-decision edits, tenant-scoped notes with
external body refs, accepted/conflict patch results, disaggregated artifact refs,
body-ref updates, and materialized journals.

```text
+---------------+     +---------------------------+     +---------------------------+
| fixture files | --> | internal/sharedtask       | --> | task record sequence,     |
+---------------+     | contract validator        |     | revision chain, journal,  |
                      | (sequence witness)        |     | artifact refs validate    |
                      +---------------------------+     | (exit 0)                  |
                                                        +---------------------------+
```

Prerequisites: the Go toolchain from the repo root; no network, API key, model, or
GPU is needed. The validation run completes in seconds and is deterministic: it
reads the fixture files and returns the same verdict on every run.

Run:

```bash
go test ./internal/sharedtask -run TestContractSequenceFixtureValidates -v
```

## What You See

The command exits 0 when the task record sequence, revision chain, journal, and
artifact references validate against the shared schemas.

## What This Does Not Claim

This fixture does not prove UI behavior, transport delivery, or multi-agent
scheduling. It is a local contract example; the deeper contract is documented in
[shared-task-record-contract.md](../../docs/shared-task-record-contract.md).
