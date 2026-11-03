// Package computetune turns replayable workload profiles into compatibility-bound
// kernel selections and deterministic storage-compute arbitration decisions.
//
// Arbitration is fail-closed and deterministic: missing profiles, incompatible
// devices, non-positive parameters, or candidate failures trigger immediate
// fallback or recomputation rather than speculative unverified execution.
//
//   - Manifest lookups require exact environment compatibility (backend, device, revisions).
//   - Candidate outputs must match the reference implementation before timing is admitted.
//   - Offline arbitration evaluates unified memory bandwidth against NVMe SSD flash wear.
//
// Wiring status (updated 2026-11-03, flow ticket debtwave2-internal-computetune):
// the production call path is the `fak computetune` verb (cmd/fak/computetune.go),
// which invokes Decide with the baseline Strix platform profiles or an
// operator-supplied JSON PlatformProfile. Tune and DispatchMatMul remain offline
// seams: no engine-runtime caller exists yet, and manifest-driven kernel
// selection awaits a follow-on wave authorized to touch the caller side.
package computetune
