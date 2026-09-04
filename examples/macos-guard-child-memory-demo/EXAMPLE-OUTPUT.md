# macOS Default Guard Child-Memory Containment Demo — Example Output

Captured output from running the deterministic macOS default guard child-memory containment demo:

```text
$ go run ./examples/macos-guard-child-memory-demo
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

Running in headless selfcheck mode:

```text
$ go run ./examples/macos-guard-child-memory-demo -selfcheck
selfcheck: PASS (macOS child RSS limit clamp, metric typing, breach containment, receipt schema, and live Darwin probe verified)
```
