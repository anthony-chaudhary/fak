# Architecture report demo

Prerequisite: Go 1.26+ (the repository toolchain setting can fetch it automatically). No key, network service, Git metadata, or GPU is required.

Run the enforced architecture-graph report through its production Go seam:

```bash
go run ./cmd/archreportdemo -selfcheck
```

## Sample run output

```text
fak architecture report demo
schema: fak-architecture/1
healthy leaves: 4 across 3 tiers
upward violations: 1 (primitive -> composite)
direct fan-in hotspots: abi=2, policy=2
diagnostic: retired has a stale tier declaration
selfcheck: PASS (real archreport seam, deterministic fixture)
```

## What this proves

- The production `internal/archreport` parser derives tier counts and import direction.
- A primitive-to-composite edge is named as an upward violation.
- Reverse edges produce deterministic direct fan-in hotspots.
- A stale tier declaration becomes a recovery-bearing diagnostic without suppressing healthy leaves.

## What this does not claim

This deterministic miniature graph proves report semantics, not the architecture quality of an arbitrary checkout. It does not execute package code, inspect test-only imports, run Git, mutate the workspace, or claim that the current repository has no violations. Use `fak architecture` against a real checkout for that report.

The repository-wide evidence and current limitations are recorded in [the architecture report dogfood note](../../docs/notes/ARCHITECTURE-REPORT-DOGFOOD-2026-08-09.md).
