// Package testquality is the Go TEST-QUALITY ratchet: the Go-side twin of
// internal/pythongate, pointed at tests that pass whether or not the code under
// test works.
//
// # Why fak needs it
//
// fak's known failure family is a green gate that measured nothing: a suite
// hidden behind `//go:build ignore`, a `go test -overlay` run reporting a false
// `ok`, a green test that never validated the header comment it was named for. A
// test that asserts nothing is that same failure with no build tag to give it
// away — it executes code and then reports success unconditionally. fak already
// scores test quality (internal/brittleness, internal/mutationefficacy,
// internal/mutationbudget) but had no LANDING-TIME floor on it, and the one
// landing-time floor of this shape it does run (pythongate's NEW_PYTHON_TOOL)
// covers Python only.
//
// # The ratchet contract, in four clauses
//
// Every clause here is load-bearing; dropping any one turns the gate into either
// noise or a silent pass.
//
//  1. KEYS STABLE UNDER EDITING. A finding's identity is (code, file, test
//     function) — never the line. Reformat the file, insert an assertion above the
//     finding, move the function: the key survives. A line-keyed baseline goes
//     stale on any edit above a finding and re-reports work nobody touched.
//
//  2. COUNTS, NOT LINES. The baseline stores how MANY findings of each key the
//     tree already carries. Two findings of the same code in the same function
//     have no stable distinguishing identity — the second one is only "the second
//     one", and inserting a line above it would renumber it. Counting means fixing
//     one of two still tightens the floor on regeneration, and adding a third is
//     still caught.
//
//  3. HARD-FAIL ON AN UNPARSEABLE BASELINE ROW. Blank lines and `#` comments are
//     skipped; everything else that does not parse is an ERROR naming the line
//     number, never a skipped line. A lenient parser that dropped a malformed row
//     would read that key's floor as ZERO, so the ratchet's own bug would present
//     as a fresh finding in somebody else's diff — and, on the other side of the
//     same coin, a typo'd key would permanently absorb a real finding.
//
//  4. ONLY EVER CLAIM "NOT GROWING". The verdict is about the DELTA. This package
//     never says a tree is clean, because it cannot: a finding is a CANDIDATE, and
//     some candidates are correct as written (see the deliberate rows in
//     baseline.txt). A tool that refused on any candidate would be wrong most of
//     the time and would simply be switched off — and a switched-off checker
//     reports zero findings, which is byte-identical to a clean tree.
//
// # What it reports
//
//	TESTQ_NO_ASSERTION         a TestXxx with no reachable failure call at all: no
//	                           t.Error/t.Fatal, no t.Skip, and the *testing.T is
//	                           never handed to a helper to assert on its behalf.
//	TESTQ_SELF_COMPARISON      an assertion comparing a value to itself (`got ==
//	                           got`, `reflect.DeepEqual(want, want)`) — true by
//	                           construction, so it cannot fail.
//	TESTQ_UNCHECKED_ERR        an error captured and then never inspected before it
//	                           is overwritten or the function ends: the failure that
//	                           call can return cannot fail the test.
//	TESTQ_UNREAD_EXPECTATION   a table-test row field named like an expectation
//	                           (want…/expect…/golden…) that no code in the test ever
//	                           reads — the table documents an assertion it does not
//	                           make.
//
// # Why it under-reports on purpose
//
// Every judgement call is resolved toward silence, because a checker with false
// positives gets disabled and then reports nothing at all — the same output as a
// clean tree. The costs, named so they are not mistaken for coverage:
//
//   - Errors are recognised by IDENTIFIER NAME (case-insensitive `err`, or the
//     exact name `e`), not by type. A test that calls its error `problem` is
//     invisible. Fixing that needs go/types and a full package load.
//   - Handing the *testing.T to any function counts as delegating the assertion,
//     so `wantRefusal(t, err, "…")` is trusted rather than followed. A helper that
//     asserts nothing therefore launders every caller.
//   - An expectation field is read if the row variable's field is selected
//     ANYWHERE in the test, and the whole rule stands down if the row is ever
//     passed somewhere as a value.
//   - `!=` self-comparison is not reported in a function that mentions NaN, which
//     is the one place `x != x` is a real check.
//
// # Scope
//
// The corpus is the git-TRACKED `*_test.go` set, not a filesystem walk. On fak's
// shared many-session checkout an untracked peer scratch file is not part of
// anybody's commit, and a gate whose verdict is a function of peer WIP is a gate
// that refuses work nobody did.
//
// # Regenerating the baseline
//
//	fak test-quality --write-baseline
//
// Regeneration only ever records TODAY's tree. Because a fixed finding is simply
// absent from the new scan, the regenerated floor is at or below the old one for
// every key that was fixed — the ratchet tightens, and `fak test-quality` names
// each loose key so the tightening is visible rather than silent.
package testquality
