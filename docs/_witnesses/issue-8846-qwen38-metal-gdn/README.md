# Issue #8846 — Qwen3.8 Metal GDN resident decode packet

Verdict: **REJECT pending hardware readback**. The code spine owns and seeds one
native Metal GDN state per linear layer and reports the explicit fak-native path,
but this checkout has no immutable same-source M3 Pro P=32/T=64 A/B receipts.
The packet therefore cannot claim a performance KEEP.

Run the exact independent readback:

```console
go test ./docs/_witnesses/issue-8846-qwen38-metal-gdn -run '^TestQwen38MetalGDNWitness$' -count=1 -v
```

The sanctioned hardware operator must replace the two empty throughput arrays
with three sequential same-binary repetitions per arm only after acquiring
machine-wide GPU ownership. KEEP additionally requires CPU-oracle, exact greedy
token, lifecycle, native identity, and no-fallback gates to remain green.
