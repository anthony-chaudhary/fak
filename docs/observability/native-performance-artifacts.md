---
title: "Native performance artifact index"
description: "Index of scrubbed fak-native performance evidence, mapping opaque correlation identifiers to bounded artifacts for dashboard investigation."
---

# Native performance artifact index

The native artifact index is the bounded drill-down surface from a Grafana point to the exact
scrubbed evidence behind it. It accepts only the opaque `npc1_<32 lowercase hex>` correlation
key defined by [`native-performance-correlation.md`](native-performance-correlation.md); it never
joins on request IDs, run IDs, paths, or Grafana labels.

## Contract

- `engine` is always `fak-native`. There is no llama.cpp fallback.
- One record contains at most five artifacts: benchmark receipt, Metal profile bundle, CUDA
  profile bundle, kernel trace, and comparison report.
- Ready artifacts use HTTPS, a lowercase SHA-256 digest, and an optional expiry. URLs with
  credentials, query strings, fragments, loopback/private hostnames, or private/raw-log path
  material are rejected before indexing.
- Broken and redacted entries retain only their typed state. They publish no locator or digest.
- The in-memory index has an explicit capacity and evicts the oldest correlation record.
- Resolution is exact on `(correlation_key, artifact_kind)`. Missing, broken, expired, private,
  untrusted-scheme, and redacted artifacts return distinct failures; no neighboring artifact or
  engine is substituted.

The package contract is `internal/nativeperfartifact` (`fak-native-performance-artifacts/1`). A
production adapter may persist or serve its scrubbed snapshot, but must preserve these rejection
and capacity rules.

## Grafana status

`fak-native-artifacts` does **not** pretend that the in-memory index is already exported through
Prometheus. The live dashboard shows only the bounded native receipt streams and freshness
families that the gateway actually emits. It carries no correlation-key variable, artifact
annotation, or clickable resolver URL, because the binary does not yet expose an
`artifact_info` metric or a public resolver adapter.

The checked-in provisioning fixture still uses documentation-only `.invalid` URLs and synthetic
keys to prove the future query, label, link, and failure-state shape. It is not live telemetry or
benchmark evidence. When a production adapter lands, it must wire the scrubbed index into the
binary, preserve the exact-key rejection rules above, and only then enable per-artifact Grafana
links.
