# Example Output

Command:

```bash
python tools/shared_task_contract.py validate-dir examples/shared-task-record-verdicts
```

Output on this checkout:

```text
examples\shared-task-record-verdicts: OK (fak.shared-patch-result.v1=5)
```

Command:

```bash
python tools/shared_task_contract.py validate-verdicts examples/shared-task-record-verdicts
```

Output on this checkout:

```text
examples\shared-task-record-verdicts: OK collaboration verdicts (fak.shared-patch-result.v1=5)
```

Exit code: `0`.
