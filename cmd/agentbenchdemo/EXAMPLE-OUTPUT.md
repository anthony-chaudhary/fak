# Example output — agentbenchdemo

Captured from `go run ./cmd/agentbenchdemo` and `-selfcheck`. The allow/deny **counts**
are deterministic and identical every run; the **timing** numbers (per-call cost,
throughput, the ×-cheaper ratio) are measured on the box and will differ from the
sample below — that is the point of a benchmark.

## `go run ./cmd/agentbenchdemo` (the self-tax table)

```text
agentbenchdemo · the self-tax: what the kernel costs per tool call
hardware: 32 cores · 32 matmul workers · pure-Go Q8 CPU (no GPU backend in this build)

  iterations:        4000   (×3 calls = 12000 adjudicated tool calls)
  allowed / denied:  4000 / 8000   (1 allow + 2 deny per iteration)
  per call:          ~1.90 µs   (mean, this box)
  throughput:        525,743 adjudicated calls/sec
  total wall-clock:  22.8 ms

  net: a single LLM tool-calling round-trip is ~1–3 s. The kernel adjudicates a
       tool call in ~1.90 µs — about 788,643× cheaper, so the safety floor is effectively
       free on the agent's critical path (the floor never gates the LLM, only the call).
```

The kernel folds **12,000 real tool calls** — one allowed read plus two refused
destructive calls per iteration — in about 23 ms. At ~1.9 µs per adjudicated call, the
safety floor's cost is roughly six orders of magnitude below a single LLM round-trip.

## `go run ./cmd/agentbenchdemo -selfcheck` (the CI gate)

```text
agentbenchdemo -selfcheck: the self-tax invariants hold (900 calls · 300 allowed · 600 denied · ~1.74 µs/call on this box)
```

`-selfcheck` re-runs a small batch and asserts the deterministic counts (1 allow + 2
deny per iteration) and a generous anti-hang ceiling on the per-call cost, then exits
non-zero on any drift. It asserts structure, not an absolute latency, so it gates
cross-platform without flaking on a loaded box.
