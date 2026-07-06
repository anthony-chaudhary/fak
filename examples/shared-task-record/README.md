# Shared Task Record Fixtures

These fixtures are runnable examples for the shared task record contract. They
cover an initial task, title edits, open-decision edits, tenant-scoped notes with
external body refs, accepted/conflict patch results, disaggregated artifact refs,
body-ref updates, and materialized journals.

```text
+---------------+     +---------------------------+     +---------------------------+
| fixture files | --> | tools/schemas/shared-*    | --> | each fixture validates,   |
+---------------+     | JSON envelope schemas     |     | and its id-stable core    |
                      | + shared-item read parity |     | fields render the same on |
                      | (validate_shared_items.py)|     | sidecar and Slack (exit 0)|
                      +---------------------------+     +---------------------------+
```

Prerequisites: Python 3 from the repo root; no network, API key, model, GPU, or
third-party package is needed. The validation run completes in seconds and is
deterministic: it reads the fixture files and returns the same verdict on every
run.

Run:

```bash
bash examples/shared-task-record/run.sh
```

(the script is one command: `python3 examples/shared-task-record/validate_shared_items.py
examples/shared-task-record`)

The Go contract fold (`internal/sharedtask`) that used to run this witness was
retired as an unwired package (`faa9a66b8`, issue #2743); the live validation
authority is now the JSON envelope schemas under `tools/schemas/shared-*.json`,
which is what `validate_shared_items.py` reads.

## What You See

The command exits 0 when every fixture validates against the schema it names
(the write gate), and each item's id-stable core fields render identically on
both surfaces (the shared-item read-parity property from #2216). A fixture that
fails validation is REFUSED with a typed reason, never rendered best-effort.

## What This Does Not Claim

This fixture does not prove UI behavior, transport delivery, or multi-agent
scheduling. It is a local contract example; the deeper contract is documented in
[shared-task-record-contract.md](../../docs/shared-task-record-contract.md).
