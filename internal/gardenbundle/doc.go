// Package gardenbundle is the garden bundle -- one default-on fold over the
// repo's read-only gardening passes.
//
// fak already measures itself with several orchestrators, each folding a
// different slice of repo health, and each run by hand or in CI but never as
// ONE thing:
//
//   - tools/scorecard_control_pane.py folds the scorecard family into one
//     portfolio total_debt with a pinned-baseline ratchet (--check).
//   - tools/fresh_status.py folds the four top-level domains (git, benchmarks,
//     work, industry) into one cross-domain snapshot.
//   - tools/fleet_control_pane.py loop-audit runs the gardening loop catalog and
//     buckets each loop healthy / action / broken.
//
// This is the fold over those folds -- the single read-only "is the garden
// tended?" verdict, so "run the gardening" is one command instead of three. It
// is deliberately READ-ONLY: it runs each member (which stay grandfathered
// Python tools), reads each member's control-pane JSON payload, and folds one
// schema/ok/verdict/finding/reason/next_action envelope. It mutates nothing and
// fixes nothing -- auto-fix is a later, witness-gated rung.
//
// --check is the bundle's CI contract: it exits non-zero when any gating member
// reports a hard regression (today only the scorecard ratchet gates;
// fresh-status and loop-audit are advisory panes that surface conditions without
// failing the bundle). A member that fails to RUN (errored) always trips the
// gate, so a silently-broken pass can't masquerade as a clean garden.
//
// The bundle is skipped entirely when FAK_GARDEN is set to an off value
// (0/off/false/no/disable/disabled) -- the env-side half of the governor brake.
//
// The pure, tested surface is Interpret (fold one member payload into a uniform
// row), Fold (fold member rows into one envelope) and CheckGate (the CI gate
// decision over a folded payload). The live runner shells out to
// python3 tools/<x>.py exactly as the Python version did.
//
// Full design: docs/notes/GARDENING-BUNDLE-DEFAULT-ON-2026-06-25.md.
package gardenbundle
