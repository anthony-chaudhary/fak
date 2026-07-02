# Example Output

Command:

```bash
go test ./internal/sharedtask -run TestContractVerdictsFixtureValidates -v
```

Output on this checkout:

```text
=== RUN   TestContractVerdictsFixtureValidates
--- PASS: TestContractVerdictsFixtureValidates (0.00s)
PASS
ok  	github.com/anthony-chaudhary/fak/internal/sharedtask	0.148s
```

The test envelope-validates every fixture file in this directory and then checks
the non-acceptance shape (needs_approval, denied, and quarantined verdicts all
present, each carrying a reason, none advancing the task revision), pinning the
exact per-schema count:

```text
fak.shared-patch-result.v1=5
```

Exit code: `0`.
