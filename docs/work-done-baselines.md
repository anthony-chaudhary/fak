# Work-done baselines

`fak info` reports work avoided relative to a named, versioned alternative. The current default is `direct-provider/v1`: the same observed session and provider workload, with fak-local response reuse and inline serving disabled on the baseline arm. The candidate arm keeps provider caching and the fak mechanisms that were actually configured for the session.

This is not a timeless “without fak” estimate. Each `fak.info.work-done/1` record carries the baseline ID, revision, effective date, comparison scope, both arm descriptions, and a SHA-256 configuration fingerprint. Every token, call, and time metric repeats the baseline ID and identifies its evidence class:

- `observed`: provider-reported cache usage converted to token-equivalent work against the declared arm;
- `witnessed`: a fak-local engine call was actually skipped;
- `modeled`: avoided calls multiplied by the current session's observed mean end-to-end latency;
- `unavailable`: the source evidence has not been emitted, which is not the same as zero.

The TUI shows the compact baseline revision in WORK DONE. Open the **Cache** tab to inspect both arms, scope, effective date, and fingerprint. Machine consumers can capture the same descriptor with:

```console
fak info --gateway-url http://127.0.0.1:PORT --json
```

Do not combine trends whose baseline ID and configuration fingerprint differ. The live TUI restarts its trend ring and emits a baseline-change row when that compatibility key changes. Historical consumers must preserve the same boundary rather than rewriting old savings under a newer provider alternative.

Claims derived from this record must follow [`net-true-value`](standards/net-true-value.md): compare against the real tuned alternative, include added costs, state scope, retain provenance, and report `not yet` when evidence is unavailable.
