// Package dispatchconservation is the worker-unit conservation ledger for the
// dispatch fleet: over a window, units_spent = accounted + leaked, ensuring that
// every worker-unit that dies ungraded is accounted as a leak rather than silence.
//
// Invariant: dispatch conservation accounting is fail-closed and monotonic.
// Over any evaluation window, total spent units must equal the exact sum of
// witnessed, unwitnessed, no-commit refusals, spawn failures, and leaked units.
//
// Guard: untracked or unverified processes default to live status to prevent false
// leak detections, while sidecar witness verdicts strictly take precedence over
// live process-table probes.
package dispatchconservation
