# Issue #9094 — Qwen full-attention KV reservation REJECT

Verdict: **REJECT**. No candidate code shipped and the prepared candidate was
never executed. The exact pinned control could not finish the native
P=32,800/T=8 request before the final predeclared Mac safety gate.

The terminal control run used base
`36d56da5a70075c90013c930be0f12545f328713`, the exact Qwen3.8-27B Q4_K_M
artifact, 512-token chunks, 12 load workers, and 48,368,448,672 bytes of
per-arm cache displacement. It crossed the 12 GiB swap-growth ceiling after
three consecutive samples. At controlled termination it had reached:

- 13,287,492,157 bytes (12,671.94 MiB) new swap;
- 44,023,414,784 bytes sampled peak footprint, with 44,914,704,520 bytes from
  `/usr/bin/time -l`;
- 18,144,788,480 bytes peak RSS;
- 27% minimum system-free memory; and
- no completed long-request response or native receipt.

That 12 GiB ceiling remains below #8972's observed +13,591.19 MiB unsafe
boundary. A control that cannot complete inside the admitted envelope cannot
support a matched candidate claim, so running the candidate would add risk
without producing admissible comparative evidence.

Isolation was clean: one persistent machine-wide GPU lease, exact bootout of
the `com.fak.qwen36-model` launchd job, zero watcher matches or unmatched
signals, exact PID-scoped arm teardown, then restoration of `qwen3.6-27b` with
the expected command hash for 90 seconds followed by 30 stale-helper-free
samples.

Artifacts:

- `reject.json` — verdict, identities, envelope, measurements, and hashes;
- `control-arm.json` — scrubbed structured terminal control record;
- `control-memory-samples.tsv` — all 89 load/request pressure samples;
- `safety-trip.txt` — exact terminal crossing;
- `watcher.log` — zero-overlap watcher close record;
- `restore-samples.tsv` — exact command/health/model restoration samples; and
- `stale-helper-samples.tsv` — 30 post-restore helper checks.

The deterministic candidate tests passed before hardware admission, but they
are intentionally not a shipment claim. The candidate diff and binary are
identified in `reject.json`; both were discarded after this control refusal.
