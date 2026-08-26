# Captured micro-context selfcheck output

Command, run from the repository root:

```sh
go run ./cmd/microcontextdemo -captured-selfcheck
```

Captured stdout:

```text
PASS fak-microcontext-spine/1 synthetic fixture
PASS logical_contexts=32 physical_workers=4 completed=32 failed=0
PASS shared_base_installs=1 turns=32 bounded_peak<=4
CLAIM bounded fan-out and shared-base accounting only; not model quality, throughput, KV residency, or GPU evidence
```
