# macOS default guard child-memory containment demo

Demonstrates macOS default child process-tree memory containment in `fak guard`: host-derived resident set size (RSS) thresholds, metric typing, leak containment, and structured receipt emission under schema `fak.guard.child-resource.v1`.

On macOS (Darwin), `fak guard` monitors and bounds wrapped agent child process trees to prevent runaway memory leaks from destabilizing the host. The default threshold is derived from physical memory as `clamp(physical/4, 1GiB, 64GiB)` resident RSS, with zero system headroom required (as physical capacity is not current system commit pressure).

## Prerequisites

- **Go 1.26+** (the repository toolchain). Zero external dependencies, no API key, no network, no model, and no GPU.
- **Cross-platform support:** On macOS (Darwin), the demo directly queries host physical memory and validates the live process-tree RSS census via `ps`. On Linux and Windows, it evaluates the identical Darwin RSS threshold clamp logic, leak containment decisions, and receipt schemas deterministically.

## Run it

Run the self-contained demo or headless acceptance check with one command:

```bash
go run ./examples/macos-guard-child-memory-demo -selfcheck
```

Or run the full walkthrough or JSON output:

```bash
go run ./examples/macos-guard-child-memory-demo
go run ./examples/macos-guard-child-memory-demo -json
./examples/macos-guard-child-memory-demo/run.sh
```

The demo completes in under 1 second. Determinism: every run produces deterministic, byte-identical decisions and receipts across invocations.

## What you'll see

The walkthrough evaluates both compliant and runaway child process trees against the Darwin RSS containment policy and emits structured receipts:

```text
== macOS Default Guard Child-Memory Containment Demo ==
Platform: darwin/arm64
Host physical RAM: 36.00 GiB
Default guard child RSS limit: 9.00 GiB (clamp(physical/4, 1GiB, 64GiB))
Active memory metric: rss

Scenario 1: compliant-child-tree
  Policy threshold: 48 MiB RSS
  Process tree: 2 processes, 32 MiB RSS
    PID 1000 (claude): 12 MiB
    PID 1001 (worker): 20 MiB
  Decision: stop=false (compliant; no containment action)

Scenario 2: runaway-child-tree
  Policy threshold: 48 MiB RSS
  Process tree: 3 processes, 75 MiB RSS (BREACH)
    PID 2000 (claude): 15 MiB
    PID 2001 (leaking-worker): 45 MiB [OFFENDER]
    PID 2002 (sub-helper): 15 MiB
  Decision: stop=true reason=CHILD_TREE_RSS_LIMIT
  Action: reap_tree (descendants_survive=false)
  Emitted receipt schema: fak.guard.child-resource.v1 (metric=rss, tree_rss_bytes=78643200)

Live Darwin Process Probe:
  PID 91057: verified live snapshot (metric=rss, tree_rss=13631488 bytes) · consistency=ok

selfcheck: PASS (all macOS default guard child-memory containment invariants verified)
```

Captured output details are preserved in [`EXAMPLE-OUTPUT.md`](EXAMPLE-OUTPUT.md).

## What this demo does not claim

What this demo does not claim: it does not claim to replace OS kernel-level cgroups, hypervisor boundaries, or kernel sandbox profiles; it demonstrates application-level child process-tree RSS accounting, host-derived threshold enforcement, and fail-closed receipt emission. For broader guard capabilities, see [`CLAIMS.md`](../../CLAIMS.md) and [`docs/cli-reference.md`](../../docs/cli-reference.md).
