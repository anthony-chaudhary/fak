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

## Grafana links

`fak-native-artifacts` provides the operator-facing index. Panels and annotations query only
`fak_native_artifact_info{engine="fak-native", correlation_key=~"$correlation_key"}` and link via
`https://artifacts.example.invalid/native/${__field.labels.correlation_key}/${__field.labels.kind}`. The endpoint is a
public-safe artifact resolver: it must validate the exact key, re-run expiry/state checks, and
return an honest error page for unavailable evidence.

The checked-in fixture uses documentation-only `.invalid` URLs and synthetic keys. It proves
query, label, link, and failure-state shape; it is not benchmark evidence and cannot expose a
credential, private path, or raw log.
