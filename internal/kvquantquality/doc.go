// Package kvquantquality evaluates version-pinned KV-cache quantization quality
// evidence against an unquantized tuned baseline.
//
// The package is runtime-neutral: it evaluates independently supplied attention,
// output, and task measurements and never claims that a model/runtime combination
// is universally suitable. Reports preserve whether evidence is modeled or was
// observed on identified hardware.
package kvquantquality
