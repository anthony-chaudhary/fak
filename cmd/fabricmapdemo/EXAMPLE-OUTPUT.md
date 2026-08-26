# Captured selfcheck

Captured from the repository root on Windows with Go 1.26.7. The JSON is stdout;
the final PASS line is stderr. The command exits 0 only after asserting that the
single selected edge is `ssd-to-gpu-direct`.

```console
$ go run ./cmd/fabricmapdemo -selfcheck
{
  "from": "L3",
  "to": "L1",
  "links": [
    {
      "id": "ssd-to-gpu-direct",
      "from": "L3",
      "to": "L1",
      "transport": "gds-rdma",
      "cost": 1,
      "cpu_path": "bypass",
      "labels": {
        "gpu-direct": "yes"
      }
    }
  ],
  "bytes": 0,
  "objective": "static_cost",
  "total_cost": 1,
  "total_latency_nanos": 0
}
PASS: L3 is a user label, and the directed L3 -> L1 CPU-bypass link was selected
```

Observed warm-run wall time: 2.7 seconds. A cold Go toolchain download or first
compile is outside that timing.
