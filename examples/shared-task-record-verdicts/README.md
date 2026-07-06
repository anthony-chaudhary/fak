# Shared Task Record Verdict Fixtures

These fixtures cover the non-acceptance collaboration verdicts a UI or protocol
adapter must render without advancing the task revision.

```text
+---------------+     +---------------------------+     +--------------------------+
| fixture files | --> | tools/schemas/shared-*    | --> | each verdict validates,  |
+---------------+     | JSON envelope schemas     |     | and renders id-stable on |
                      | + shared-item read parity |     | sidecar and Slack; task  |
                      | (validate_shared_items.py)|     | rev does not advance (0) |
                      +---------------------------+     +--------------------------+
```

Prerequisites: Python 3 from the repo root; no network, API key, model, GPU, or
third-party package is needed. The validation run completes in seconds and is
deterministic: it reads the fixture files and returns the same verdict on every
run.

Run:

```bash
bash examples/shared-task-record-verdicts/run.sh
```

(the script is one command: `python3 examples/shared-task-record/validate_shared_items.py
examples/shared-task-record-verdicts`)

The Go contract fold (`internal/sharedtask`) that used to run this witness was
retired as an unwired package (`faa9a66b8`, issue #2743); the live validation
authority is now the JSON envelope schemas under `tools/schemas/shared-*.json`,
which is what `validate_shared_items.py` reads.

## What You See

The command exits 0 when every verdict fixture validates against the schema it
names and renders id-stable on both surfaces (the shared-item read-parity
property from #2216), without advancing the task revision. A fixture that fails
validation is REFUSED with a typed reason, never rendered best-effort.

## What This Does Not Claim

This fixture does not prove UI behavior, transport delivery, or multi-agent
scheduling. It is a local contract example; the deeper contract is documented in
[shared-task-record-contract.md](../../docs/shared-task-record-contract.md).
