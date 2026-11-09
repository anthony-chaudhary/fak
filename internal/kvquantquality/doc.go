// Package kvquantquality evaluates version-pinned KV-cache quantization quality
// evidence against an unquantized tuned baseline.
//
// The package is runtime-neutral: it evaluates independently supplied attention,
// output, and task measurements and never claims that a model/runtime combination
// is universally suitable. Reports preserve whether evidence is modeled or was
// observed on identified hardware.
//
// Beyond the budget evaluation, the package records per-scheme rounding-floor
// receipts (fak.kvquant-quality.floor/v1): the intrinsic round-trip error a
// precision/block/scale combination pays on real data, attributed per role
// (window KV, compressed latent, indexer). A floor-relative verdict separates
// scheme-inherent divergence (at or below the floor: not a defect) from genuine
// regressions (above the floor). Quoted upstream floors stay labeled
// upstream_quoted_unverified and are never claimed as appliance measurements.
package kvquantquality
