# Example Output

Command:

```bash
go test ./internal/sharedtask -run TestContractSequenceFixtureValidates -v
```

Output on this checkout:

```text
=== RUN   TestContractSequenceFixtureValidates
--- PASS: TestContractSequenceFixtureValidates (0.00s)
PASS
ok  	github.com/anthony-chaudhary/fak/internal/sharedtask	0.237s
```

The test envelope-validates every fixture file in this directory and then checks
the sequence shape (task record, bound materialized journals, an accepted patch
result, the title-replacement patch), pinning the exact per-schema counts:

```text
fak.shared-artifact-ref.v1=1, fak.shared-event.v1=1, fak.shared-patch-result.v1=7, fak.shared-patch.v1=6, fak.shared-task-journal.v1=2, fak.shared-task.v1=1
```

Exit code: `0`.
