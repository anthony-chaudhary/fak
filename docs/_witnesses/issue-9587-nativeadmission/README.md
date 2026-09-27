# Issue #9587: native Metal admission witnesses

## Live lifecycle capture (2026-09-26): `live-2026-09-26.json`

This is a real capture on the pinned Apple M3 Pro host (36 GiB unified memory, 18-core GPU). The binary was `fak` built from the #9587 admission-lifecycle change *before* the review fixes that landed with it in `0fbaa0516`. Two behaviors differ in the landed code:

- A teardown that hits `WeightSessionsActiveError` now keeps the reservation until process exit (`cleanup: deferred_to_exit`). In this capture those releases were recorded as `released` with a `teardown_error`.
- The in-process peak-RSS probe now reads `getrusage`. It was empty on darwin in this capture, so these receipts measure only swap, and the `time_l_*` fields carry RSS.

Every `receipts[]` entry is a JSON line that `fak serve` wrote to stderr (`fak native admission receipt: ...`). The `time_l_*` values come from `/usr/bin/time -l` wrapped around each serve process. The ledger snapshots are `reservations.json` before, during and after each run.

- **Small-model co-residency (`FAK_NATIVE_ADMISSION=aggregate`)**
  - SmolLM2-135M Q8_0 and Qwen2.5-Coder-3B Q4_K_M were admitted, reached `steady`, and served at the same time. The ledger held two `steady` rows.
  - One row left by a dead owner (a killed earlier service) was reaped on admission.
  - After SIGTERM, each serve emitted a `release` receipt with `cleanup: released`, and the ledger returned to empty.
- **Qwen3.8-27B-Q4_K_M (17,106,775,008 bytes), default mode: refused before the loader**
  - Reason: `aggregate_capacity`. The planned startup peak of 21,026,703,360 B was measured against 5,937,441,792 B allocatable, with host pressure `warning` caused by other workloads on the shared host.
  - The refused process peaked at an 86,524,888 B physical footprint, so no weights were loaded.
- **Topology**
  - Every receipt reports `topology: apple-unified-memory` and `host_unified: true`, probed from the Metal device.
  - Every receipt reports `host_addressable: false`. A unified pool never makes device buffers host-dereferenceable.

### Findings from this capture (open work)

- **Planned peak is below measured peak.** The planned startup peak does not bound the whole process. The `/usr/bin/time` peak footprint exceeded the plan for both small models:
  - SmolLM2: 1.19 GB measured vs 0.93 GB planned.
  - Qwen2.5-Coder-3B: 6.15 GB measured vs 4.43 GB planned.

  The plan covers model bytes only. Process baseline and runtime transients (including #12958's whole-sequence graph working set) are not yet charged.
- **Teardown can't free the weights.** At teardown, `CloseWeights` reported `1 session(s) remain`, because legacy host sessions are never closed. Since `0fbaa0516` the reservation is therefore held until the process exits (`cleanup: deferred_to_exit`) instead of being handed back while the weights are still resident. The next step is to close host sessions at shutdown so teardown can actually free the weights.
- **Still owed:** an admit/steady receipt for the 27B on a quiet host.

## `receipt.json` (2026-09-03): not a live capture

`receipt.json` predates this capture and was assembled by hand, not measured.

- Its `peak_rss_bytes` and `reserved_bytes` equal the planned startup peak. That value is the #8971 refusal constant, not a measurement.
- It reports `reserved_bytes` 22,754,885,632 B as admitted against `allocatable_bytes` of 10,911,449,088 B, which the reservation store would refuse.

It is kept for history only. Do not cite it as evidence.
