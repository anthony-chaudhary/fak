# Apple unified-memory counter provenance — 2026-08-27

## What issue #9427 ships

`fak model-observe bandwidth collect` can import
`fak-apple-memory-counter-import/1`, a provider-neutral normalized JSON
envelope for Apple unified-memory counter evidence. The envelope accepts either
paired direct byte rates or paired monotonic byte-counter snapshots. It is a
stable importer contract, not a raw `powermetrics` parser and not a live Apple
collector.

The importer accepts only package or system scope. Those observations may be
compared with a host-memory roofline, but they must not be labeled as process,
request, model, or device traffic. A request phase records temporal overlap
only. Host rooflines applied to these observations remain isolated from AMD and
NVIDIA device observations.

No live Apple capture was made for this issue. The fixtures are synthetic
contract data, and this note does not claim that an undocumented field or
command is available on any current Apple release.

## Historical `powermetrics` field evidence

The historical evidence is third-party source code, not an Apple-published
schema. At pinned asitop commit
[`9595593c7eafa76395d817c4ac6a34abbe5e21fd`](https://github.com/tlkh/asitop/tree/9595593c7eafa76395d817c4ac6a34abbe5e21fd):

- [the parser](https://github.com/tlkh/asitop/blob/9595593c7eafa76395d817c4ac6a34abbe5e21fd/asitop/parsers.py#L3-L34)
  reads a `bandwidth_counters` collection and enumerates historical names that
  include `DCS RD` and `DCS WR`; and
- [the launcher](https://github.com/tlkh/asitop/blob/9595593c7eafa76395d817c4ac6a34abbe5e21fd/asitop/utils.py#L41-L58)
  requests the historical `bandwidth` sampler and plist output at a configured
  interval.

That code establishes that the project expected those names in the output it
consumed. It does **not** establish a documented Apple unit, reset/wrap rule,
scope contract, stable raw schema, or current availability. In particular, fak
does not copy asitop's scaling as a unit definition.

The next asitop commit,
[`74ebe2cbc23d5b1eec874aebb1b9bacfe0e670cd`](https://github.com/tlkh/asitop/commit/74ebe2cbc23d5b1eec874aebb1b9bacfe0e670cd),
is titled “Updated for Ventura,” disables the bandwidth path, and changes that
project's README to say that Apple removed memory bandwidth from
`powermetrics`. This is the historical removal boundary reported by asitop; it
is not evidence about undocumented behavior on later releases.

Because the historical sources do not provide the complete contract required
for a safe conversion, raw `powermetrics` plist remains unsupported. A provider
must establish byte units, timing, reset state, and package/system scope before
normalizing evidence for fak.

## Normalized import contract

The top-level object rejects unknown and case-variant fields and contains:

- `schema`: exactly `fak-apple-memory-counter-import/1`;
- `provider` and `provider_version`: exact values repeated by the caller;
- `scope`: exactly `system` or `package`, with no scope ID;
- `capture.status`: `ok`; `permission-denied` and `unsupported` are explicit
  refusals and produce no observation;
- `capture.started_at`, `capture.ended_at`, and `capture.interval_ns`: a
  positive, mutually consistent interval;
- non-empty `capture.metadata.os_version`, `hardware_model`, and
  `capture_method` values;
- `raw_provider_fields`: the retained provider values; and
- exactly one read counter and one write counter with identical scope and byte
  provenance.

Each normalized counter names its exact retained raw field. The importer checks
that the retained raw value matches the normalized rate or snapshot pair rather
than accepting a derived value with no evidence.

Two counter modes are accepted:

1. **Direct rate:** `unit: "bytes_per_second"` with an unsigned integer
   `rate_bytes_per_second` for each direction.
2. **Monotonic delta:** `unit: "bytes"` with exactly two unsigned snapshots for
   each direction. Snapshot times must equal the capture bounds, every snapshot
   must state `reset_observed: false`, and the end value must not decrease.

The importer rejects resets, ambiguous reset state, decreases without a wrap
contract, unsupported units, missing or duplicate direction pairs, mixed rate
and delta provenance, mixed scopes, scope IDs, process/device scope, mismatched
raw values, malformed metadata, provider/version mismatches, duplicate JSON
names, and unknown fields. Unavailable values remain absent rather than being
serialized as zero.

## Fixture boundary

- `internal/modelperfobs/testdata/apple-memory-direct-rate.json` covers a
  package-scoped direct-rate pair.
- `internal/modelperfobs/testdata/apple-memory-byte-delta.json` covers a
  system-scoped monotonic-delta pair.

Both fixtures produce 85 GB/s read plus 17.5 GB/s write from deterministic
integers. They are schema witnesses only—not captures, benchmark results, or
claims about Apple hardware. The tests mutate those fixtures to prove the
refusal and host/device-isolation boundaries above while retaining the existing
Darwin memory-pressure fields independently.

## 2026-08-30 current-provider audit

**Result: Apple unified-memory bandwidth remains explicitly unavailable on the
current sanctioned Apple Silicon probe.** The implementation now probes the
bundled `/usr/bin/powermetrics --help` contract and emits typed evidence instead
of silently returning zero or deriving traffic from an adjacent signal.

The scrubbed arm64 capture in
`docs/_witnesses/modelperfobs/apple-unified-memory-2026-08-30.json` records macOS
26.6.2 (build 25G83). Its advertised samplers are tasks, battery, network, disk,
interrupts, cpu_power, thermal, sfi, gpu_power, and ane_power. None is a
documented package/system unified-memory read/write byte-rate or monotonic byte
counter. A one-shot `gpu_power` probe was permission-gated and, regardless,
power/frequency is not byte traffic. The provider result is therefore
`unsupported`; permission failures, malformed help/output, and a future but
insufficiently specified memory sampler have distinct `permission-denied`,
`malformed`, and `ambiguous` results.

The audit also examined SiliconScope v4.1.3 as a current reproducible IOReport
consumer. It is useful prior art, but its public `BandwidthSample` reports
requestor totals in GB/s rather than package read and write separately, and it
can fall back to a PMP DCS bandwidth residency histogram marked `isEstimated`.
That fallback is an estimate from residency buckets, not accepted byte-counter
evidence under this contract. Consequently fak does not ingest SiliconScope's
combined/estimated signal as read/write memory bandwidth.

A future provider becomes admissible only when a fixture binds all of the
following to a current version and machine-readable output:

- exact package/system read and write raw field names;
- byte or byte-per-second units, without capacity or roofline conversion;
- the measured interval and whether values are direct rates or counter deltas;
- monotonic counter reset/wrap semantics, or explicit direct-rate semantics;
- package/system scope rather than process, storage, or device-only traffic.

When those facts exist, provider output must still be translated into
`fak-apple-memory-counter-import/1` and accepted by the existing strict importer.
Raw powermetrics plist/text remains unsupported; the current-provider seam does
not weaken the importer or infer telemetry from capacity, pressure, residency,
process/storage I/O, power, utilization, GPU occupancy, or theoretical
rooflines.
