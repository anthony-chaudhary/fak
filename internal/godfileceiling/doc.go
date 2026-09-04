// Package godfileceiling is the ratcheting god-file LOC ceiling gate (issue #2898,
// Hermes-inspiration epic #2871).
//
// Invariant: Non-baseline tracked source files must never exceed HardCeiling, and baseline entries may only ratchet downward.
// Contract: Evaluate and Repin operate as pure, deterministic mapping functions with no file I/O or ambient state.
// Precondition: LineCount expects physical UTF-8 or ASCII text buffers; empty input returns zero lines.
//
// # Why
//
// Hermes' gateway/run.go-equivalent (gateway/run.py) grew to 20,320 lines because
// nothing on merge said "no". fak's code-quality scorecard already FLAGS god-files
// (> 1500 lines) and /modularize splits them, but a flag is advisory — a new monolith
// can still land. This package is the HARD gate: a Go test invariant that fails the
// build when any tracked .go file crosses the ceiling, turning "we should refactor"
// into "you cannot land a new monolith". It is the LOC sibling of internal/pythongate
// (which ratchets the count of Python tools down) and follows the same shape: a frozen
// baseline that can only tighten, enforced by a Test in this package that go test runs
// as part of `make ci`.
//
// # The two rules
//
// Evaluate applies two rules to a measured {path: lines} tree against the pinned
// Baseline (baseline.go):
//
//   - NO NEW GOD-FILE: every tracked .go file NOT in the baseline must stay at or below
//     HardCeiling (1500 lines). A brand-new 1600-line file fails here. This is the
//     load-bearing rule — it stops the debt from ever growing a new head.
//   - RATCHET DOWN: every file that IS in the baseline (today's worst offenders, pinned
//     at their current line count) may only SHRINK. If a pinned god-file grows past its
//     cap the gate fails; when it drops, an operator re-pins it lower (Repin) so the
//     ceiling ratchets monotonically down and never back up. A cap is never hand-raised.
//
// # LOC definition
//
// A file's size is its physical line count (len of the "\n"-split contents), matching
// tools/code_quality_scorecard.py's n_lines so the gate and the scorecard agree on what
// a god-file is. The measured set is `git ls-files '*.go'` minus the same non-first-party
// trees the scorecard excludes (testdata, vendor, and the agent-machinery checkout dirs
// that hold full repo COPIES) — see ExcludeDirs — and minus *_test.go, which the
// architecture corpus grades under the tests KPI rather than as god-files. Test files
// churn on every new leaf (a shared *_test.go gains a row per package), so pinning them
// would red the gate on growth unrelated to any monolith; internal/hooks/gate_godfile.go
// (the god-file/function GROWTH ratchet, #2868) excludes them for the same reason.
//
// # Regenerating the baseline
//
// Regenerate baseline.go only to TIGHTEN it after a file shrinks — never to raise a cap
// or admit a new offender. Repin refuses a proposal that would raise any cap or pin a
// brand-new over-ceiling file (that would defeat the ratchet); a legitimate re-pin is
// always a subset-or-lower of the previous baseline. To regenerate, feed a fresh
// MeasureTree through Repin (against the current Baseline) and write FormatBaseline's
// output to baseline.go — TestRepinRatchetsDownOnly proves the refusals hold.
//
// Tier: foundation (1) — see internal/architest. Stdlib-only; off the request path.
package godfileceiling
