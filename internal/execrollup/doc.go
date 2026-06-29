// Package execrollup is the executive activity roll-up: one read-only fold that
// turns the firehose of agentic-fleet signals into a single signal-dense page a
// human can read in a glance — the answer to "how does one person keep up with a
// city of agents".
//
// It aggregates the already-emitted JSON of the per-plane folds —
// tools/dispatch_status.py (throughput + closure honesty), fak loop health (dark
// loops), fak cadence (quality-debt trend + work-done), and fak fleet status
// (box liveness) — into one control-pane envelope: a GREEN/WATCH/RED fleet
// verdict, the marquee signal-to-noise ratio (witnessed-resolved vs claimed),
// and a ranked "what needs you" list.
//
// Signal-to-noise is the design rule. A quiet plane contributes no line; only
// deviations surface. An unmeasured plane is a WATCH gap, never a silent GREEN.
// Every surfaced number carries a provenance label (WITNESSED / OBSERVED /
// CLAIMED / UNVERIFIED).
//
// The package is PURE — it reads no clock, runs no tool, and touches no disk. The
// live collectors that shell the per-plane folds live in cmd/fak/rollup.go, the
// same pure-core / impure-shell split as internal/cadencereport.
package execrollup
