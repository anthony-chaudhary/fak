# Workload-fit selection demo captured output

Command, run from the repository root:

```console
go run ./cmd/workloadfitdemo -selfcheck
```

The block below is the complete deterministic stdout capture consumed by the package regression test.

<!-- BEGIN SELFCHECK OUTPUT -->
```text
CODING fit: choose ponytail@r8
LEGAL REVIEW fit: choose legal-review-harness@r4
PONYTAIL for legal: refuse
  citations: missing — capability is not evaluated for this candidate [legal-review@r1]
  confidential: missing — capability is not evaluated for this candidate [matter-policy@r3]
  human-review: unsupported — candidate evidence says capability is unsupported [ponytail@r8]
  jurisdiction: missing — capability is not evaluated for this candidate [legal-review@r1]
BOUNDARY: fitness fixture is not legal certification; domain review remains required
SELFCHECK PASS: compatibility and purpose-specific fitness remain separate
```
<!-- END SELFCHECK OUTPUT -->
