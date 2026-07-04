# Example output

A representative captured run of [`run.sh`](run.sh) on this repository — no key, no
model download, no GPU, no network. The two witnesses run offline on a **synthetic**
model through the real quarantine gate, so the numbers below reproduce deterministically
on any machine with Go.

```text
== 1/2 write-time evict == never-saw (the headline: max|Δ| = 0) ==
   internal/kvmmu.TestWriteTimeEvictEqualsNeverSaw
=== RUN   TestWriteTimeEvictEqualsNeverSaw
    kvmmu_test.go:115: max|Δ| evict-vs-never = 0.000e+00 (want 0) ; poison-vs-never = 3.257e-01 (want >0)
--- PASS: TestWriteTimeEvictEqualsNeverSaw (0.00s)
ok  	github.com/anthony-chaudhary/fak/internal/kvmmu	0.33s

== 2/2 paged (non-contiguous) evict == contiguous evict, bit-for-bit ==
   internal/model.TestPagedEvictBitIdenticalToContiguous
=== RUN   TestPagedEvictBitIdenticalToContiguous
--- PASS: TestPagedEvictBitIdenticalToContiguous (0.00s)
ok  	github.com/anthony-chaudhary/fak/internal/model	0.20s
```

The one line to read is:

```text
max|Δ| evict-vs-never = 0.000e+00 (want 0) ; poison-vs-never = 3.257e-01 (want >0)
```

- **`evict-vs-never = 0.000e+00`** — after the poisoned span is evicted write-time, the
  next-token distribution is **bit-identical** to a cache that never saw the span.
- **`poison-vs-never = 3.257e-01`** — the non-vacuity control: keeping the poison *does*
  move the distribution (by 0.326), so the `= 0` above is a real forgetting, not a
  cache that ignores its input.

## The real-weights rung (recorded, requires the oracle export)

The synthetic witness above proves the mechanism (RoPE-linear reposition + write-time
eviction). The **token-for-token vs Hugging Face** rung is
`internal/model.TestKVQuarantineEqualsNeverSaw`, which greedily continues the evicted
cache and asserts it equals HF's never-saw-poison run, plus a reposition invariant of
`max|Δ| = 0`. It is **fixture-gated**: it `SKIP`s cleanly unless the gitignored ~538 MB
f32 export is present.

```text
$ go test ./internal/model -run 'TestKVQuarantineEqualsNeverSaw$' -count=1 -v
    evict_test.go:87: no exported weights in .cache/smollm2-135m; run: python internal/model/export_oracle.py --out .cache/smollm2-135m
--- SKIP: TestKVQuarantineEqualsNeverSaw (0.00s)
```

To run it for real, export the weights first (needs Python + the model), then re-run:

```bash
python internal/model/export_oracle.py --out .cache/smollm2-135m
go test ./internal/model -run 'TestKVQuarantineEqualsNeverSaw$' -count=1 -v
```

The recorded numbers for that rung are published in
[`docs/benchmarks/KV-QUARANTINE-BRIDGE-RESULTS.md`](../../docs/benchmarks/KV-QUARANTINE-BRIDGE-RESULTS.md)
(`evict-vs-never max|Δ| = 0`; poison-vs-never `= 3.257e-01`).
