# Example Output

Command:

```bash
python3 examples/shared-task-record/validate_shared_items.py examples/shared-task-record-verdicts
```

Output on this checkout:

```text
ok: 5 fixture(s) pass the schema write gate; 5 render id-stable on sidecar and slack
per-schema: fak.shared-patch-result.v1=5
```

The validator envelope-validates every non-acceptance verdict fixture in this
directory against the JSON schema it names (`tools/schemas/shared-*.json`) — a
fixture that fails is REFUSED with a typed reason, never rendered best-effort —
and then witnesses the shared-item read-parity property from #2216: the
id-stable core-field projection handed to a sidecar pane is byte-identical to
the one handed to a Slack card. Every verdict carries `base_rev` equal to
`current_rev` (the task revision does not advance), and the per-schema count
above is the same count the retired `internal/sharedtask` go-test witness
pinned, over the same fixture files.

Exit code: `0`.
