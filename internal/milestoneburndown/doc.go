// Package milestoneburndown is the GitHub-milestone SCHEDULE dimension the
// milestone report never had: it reads the live milestones' own due dates,
// open/closed counts, and trailing closure velocity, then classifies each into a
// closed at-risk verdict (ON_TRACK / AT_RISK / OVERDUE / NO_DUE_DATE / DONE) with
// a projected drain date compared against the due date.
//
// It is deliberately ORTHOGONAL to internal/milestonereport (the maturity CLIMB +
// epic ROADMAP fold) and to internal/metrics generation horizons: a "generation"
// is a maturity horizon, NOT a deadline (see docs/generation.md), so this package
// never redefines what a generation means. It reports the schedule truth of the
// GitHub milestone objects — the fact that G0 is due 2026-07-13 with N open, or
// that a milestone carries no due date at all — which no other fold surfaces.
//
// Shape mirrors the sibling reports: a pure, unit-testable fold (Interpret / Fold /
// Render / the ledger row + trend) with the live `gh` shell isolated in collect.go
// behind an injectable Runner, so the whole verdict path is testable without a
// process or a network. The report embeds trendreport.Envelope so it emits the
// same schema/ok/verdict/finding control-pane envelope the other trend reports do,
// and its advisory gate fails ONLY when the milestones could not be read — a
// report is a MIRROR of schedule truth, never a second quality gate.
package milestoneburndown
