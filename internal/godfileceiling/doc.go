// Package godfileceiling enforces the ratcheting god-file LOC ceiling gate (issue #2898).
//
// Invariant: Non-baseline tracked source files must never exceed HardCeiling, and baseline entries may only ratchet downward.
// Contract: Evaluate and Repin operate as pure, deterministic mapping functions with no file I/O or ambient state.
// Precondition: LineCount expects physical UTF-8 or ASCII text buffers; empty input returns zero lines.
//
// # Why
//
// Unbounded file growth creates unmaintainable monoliths. This package provides a hard
// build gate: a Go test invariant failing the build when tracked .go files exceed the ceiling.
//
// # The two rules
//
// Evaluate checks measured physical line counts against Baseline:
//   - NO NEW GOD-FILE: tracked .go files not in Baseline must not exceed HardCeiling (1500 lines).
//   - RATCHET DOWN: pinned baseline files may only shrink; caps ratchet monotonically downward.
//
// # LOC definition
//
// Physical line count matches code_quality_scorecard.py line counting over tracked Go files,
// excluding vendor, testdata, agent checkouts, and *_test.go files.
//
// # Regenerating the baseline
//
// Baseline is regenerated only to tighten caps after a file shrinks via MeasureTree and Repin.
//
// Tier: foundation (1) — see internal/architest. Stdlib-only; off the request path.
package godfileceiling
