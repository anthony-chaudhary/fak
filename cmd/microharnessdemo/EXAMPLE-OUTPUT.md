# Captured microharness selfcheck output

Command, run from the repository root:

```sh
go run ./cmd/microharnessdemo -selfcheck -ledger off
```

Captured stdout:

```text
FAK MICROHARNESS — bounded microagents construct a harness
goal: Build a local coding harness that can edit this repository and prove its work.
  receipt architecture depth=1 turns=2 -> compose a local coding harness from bounded tool and proof profiles
  receipt proof        depth=2 turns=3 -> require build plus affected tests before completion
  receipt tools        depth=2 turns=1 -> profile=repo-read-write; shell=workspace-only
  task class one_turn case=capability-selection turns=1 outcome=completed
  task class bounded_correction case=witness-correction turns=3 outcome=completed
  task class root_only case=irreversible-goal turns=0 outcome=refused-delegation
harness: runtime=fak-native; tools=repo-read-write/workspace-only; completion=build+affected-tests
context boundary: root retained 877 receipt bytes; child transcript bytes=777; full child transcripts in root=false
benchmark monolith quality=3/3 wall_ms=18 tokens=1678 cache_read=0 root_bytes=1654 cost_microusd=1678
benchmark receipt_only quality=3/3 wall_ms=8 tokens=901 cache_read=877 root_bytes=877 cost_microusd=901 reduction=root:46% tokens:46%
recursion boundary: depth<=2; turns/child<=3; child requests are re-admitted by the host
PASS — go run ./cmd/microharnessdemo -selfcheck
```
