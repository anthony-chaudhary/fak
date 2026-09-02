# Captured run — `examples/release-steward`

Real run, offline, no key/model/GPU. Verified byte-identical across consecutive
runs — the receipt is a pure function of the demo's fixed six-month script.

```console
$ go run ./examples/release-steward
release-steward lifecycle: 6 simulated months
stable identity: macro/release-steward
sessions=3 restarts=2 inbox=2 delegation=1 micro_fleet=1
selective promotion: 2 facts; raw child history retained: no
retirement: exported and retired
{"schema":"fak.release-steward-demo/1","macro_id":"macro/release-steward","address":"macro://release-steward/inbox","sessions":3,"restarts":2,"inbox_delivered":2,"delegations":1,"micro_operations":1,"durable_memory":["release cadence: monthly","rollback rule: retain last-known-good"],"raw_child_history_retained":false,"receipts":[{"month":1,"kind":"baseline_session","model":"frontier","fleet":"interactive","outcome":"plan_saved"},{"month":3,"kind":"delegated_child","model":"small","fleet":"single","outcome":"verified_summary_promoted"},{"month":5,"kind":"micro_fleet","model":"small","fleet":"100k-fanout","outcome":"accepted_aggregate_promoted"},{"month":6,"kind":"retirement","model":"none","fleet":"none","outcome":"exported"}],"state":"retired"}
```

## What the capture proves

- **One identity across every boundary.** `macro/release-steward` survives 3
  sessions and 2 restarts, receives 2 mailbox deliveries, absorbs 1 delegated
  child and 1 micro-fleet operation — and stays the same addressable macro.
- **Promotion is selective by construction.** Exactly 2 facts are promoted to
  `durable_memory`; `raw_child_history_retained: false` is the demo's point —
  raw child history is not durable by default.
- **The boundaries are inspectable.** Each `receipts[]` row names the month,
  kind (baseline_session / delegated_child / micro_fleet / retirement), the
  model class, the fleet shape, and the outcome — so a reader can trace what
  happened at each session/sub-agent/micro-operation boundary.
- **Termination is explicit.** The lifecycle ends `exported and retired` with
  `"state":"retired"` — a closed arc, not an open loop.
