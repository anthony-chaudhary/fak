# Example Output

Command:

```bash
python tools/shared_task_contract.py validate-dir examples/shared-task-record
```

Output on this checkout:

```text
examples\shared-task-record: OK (fak.shared-artifact-ref.v1=1, fak.shared-event.v1=1, fak.shared-patch-result.v1=7, fak.shared-patch.v1=6, fak.shared-task-journal.v1=2, fak.shared-task.v1=1)
```

Command:

```bash
python tools/shared_task_contract.py validate-sequence examples/shared-task-record
```

Output on this checkout:

```text
examples\shared-task-record: OK shared sequence (fak.shared-artifact-ref.v1=1, fak.shared-event.v1=1, fak.shared-patch-result.v1=7, fak.shared-patch.v1=6, fak.shared-task-journal.v1=2, fak.shared-task.v1=1)
```

Exit code: `0`.
