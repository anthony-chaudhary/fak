# Session intent demo captured output

Command, run from the repository root:

```console
go run ./cmd/sessionintentdemo -selfcheck
```

The block below is the complete deterministic stdout capture consumed by the package regression test.

<!-- BEGIN SELFCHECK OUTPUT -->
```text
{
  "version": "fak.session-intent/v1alpha1",
  "objective": "research structured session needs",
  "trigger": {
    "kind": "immediate"
  },
  "effort": [
    {
      "kind": "minimum",
      "clock": "active",
      "duration": "2h0m0s"
    },
    {
      "kind": "target",
      "clock": "active",
      "duration": "4h0m0s"
    },
    {
      "kind": "maximum",
      "clock": "elapsed",
      "duration": "10h0m0s"
    }
  ]
}
DECISIONS: 1h_active=continue 2h_active=eligible 10h_elapsed=timeout
SELFCHECK PASS: minimum and target govern stop eligibility/planning; maximum governs forced timeout
```
<!-- END SELFCHECK OUTPUT -->
