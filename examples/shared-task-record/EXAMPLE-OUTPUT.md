# Example Output

Command:

```bash
python3 examples/shared-task-record/validate_shared_items.py examples/shared-task-record
```

Output on this checkout:

```text
ok: 18 fixture(s) pass the schema write gate; 18 render id-stable on sidecar and slack
per-schema: fak.shared-artifact-ref.v1=1, fak.shared-event.v1=1, fak.shared-patch-result.v1=7, fak.shared-patch.v1=6, fak.shared-task-journal.v1=2, fak.shared-task.v1=1
```

The validator envelope-validates every fixture file in this directory against
the JSON schema it names (`tools/schemas/shared-*.json`) — a fixture that fails
is REFUSED with a typed reason (`MISSING_REQUIRED_FIELD`, `UNKNOWN_SCHEMA`,
`MALFORMED_JSON`, …), never rendered best-effort — and then witnesses the
shared-item read-parity property from #2216: the id-stable core-field
projection handed to a sidecar pane is byte-identical to the one handed to a
Slack card. The per-schema counts above are the same counts the retired
`internal/sharedtask` go-test witness pinned, over the same fixture files.

Exit code: `0`.
