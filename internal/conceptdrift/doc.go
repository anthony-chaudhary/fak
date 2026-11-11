// Package conceptdrift is the concept-coverage drift watcher, ported from the
// retired tools/concept_drift_watcher.py pair (NEW_PYTHON_TOOL ratchet).
//
// The concept-disambiguation scorecard (tools/concept_disambiguation_scorecard.py,
// grandfathered Python that STAYS) computes coverage_debt against the
// confusable-token universe it discovers in the tree. That number is a SNAPSHOT:
// it says nothing about whether it has MOVED since the last time anyone looked.
// And it moves - proven 2026-11-07/08, where merges re-opened debt 2 twice
// (discovered 4170 -> 4173). This package closes that loop:
//
//	RecordBaseline   run the scorecard once and durably record its numbers (git
//	                 head + discovered / covered / coverage-debt + standing) to
//	                 the baseline file. Run this after retiring a drift batch.
//	Check            run the scorecard again, diff against the baseline, write a
//	                 drift receipt next to the baseline. Exit 0 when debt did not
//	                 grow; exit 3 when coverage-debt RE-OPENED (delta > 0),
//	                 printing the unpositioned tokens per family.
//	Simulate         prove the mechanism without touching real data: run the
//	                 scorecard on the intact catalog, then again on a temp copy
//	                 with the rows file covering the most tokens removed, and
//	                 require the debt to grow with every newly-uncovered token
//	                 belonging to the removed file. Exit 0 when the self-witness
//	                 holds; exit 4 when it does not.
//	AutoTicket       on a real drift, best-effort create a positioning ticket via
//	                 `fak-run fak-ticket create concept-coverage-drift-<ts> ...`.
//
// The scorecard runs as a SUBPROCESS (python tools/concept_disambiguation_scorecard.py
// --json) so the watcher never re-implements discovery; its stdout is read through a
// pipe (clean bytes), and a BOM / UTF-16 header is tolerated for manually redirected
// output. Read-only over the tree; the only writes are the baseline + receipt JSONs.
// Deterministic, ASCII.
//
// # Port notes (behavior preserved from the Python original)
//
//   - Baseline/receipt paths and JSON formats are preserved EXACTLY so existing
//     recorded baselines remain readable: schema strings, key sets, 2-space
//     indentation, sorted keys, trailing newline, and the drift-receipt-<ts>.json
//     naming next to the baseline.
//   - The Python watcher anchored the scorecard script to its OWN install location
//     (the repo carrying tools/). The Go equivalent anchors it to the local repo
//     root (the git toplevel of the invoking cwd, resolved by the cmd/fak shell);
//     Options.Scorecard overrides it, and an empty value falls back to
//     <workspace>/tools/concept_disambiguation_scorecard.py - the layout when the
//     workspace IS the fak repo root, the only supported production shape.
//   - Likewise the Python simulate default data dir anchored to the script's repo
//     root; the Go version anchors it to the resolved workspace (identical for the
//     supported production shape).
//   - All spawned processes (git, python, fak-run) use the house window-suppressed
//     spawn (pkg/sysproc ConfigureBackground: HideWindow + CREATE_NO_WINDOW on
//     Windows) so the watcher can run from background automation without flashing
//     console windows (windowgate UNSUPPRESSED_SPAWN posture).
package conceptdrift
