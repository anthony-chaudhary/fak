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
package computetune
