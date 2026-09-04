// Package rotationmeta defines a neutral contract for rotation-based low-bit
// quantization transform provenance and runtime-fusion requirements.
//
// Invariant: rotation metadata validation is fail-closed and deterministic.
// Guard: unknown contract versions, unpinned recipe versions, missing transforms,
// and undeclared runtime fusions must be rejected or delegated explicitly without fallback.
package rotationmeta
