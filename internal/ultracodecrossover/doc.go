// Package ultracodecrossover evaluates the bounded task-complexity crossover
// where micro-context scoping stops preserving accepted outcomes.
//
// Invariant: UltraCode crossover campaigns are fail-closed and bounded;
// evaluation halts immediately when consecutive quality failures occur,
// refusing to extrapolate token avoidance past degraded or divergent results.
//
// Guard: Every campaign rung requires strict frozen checks and matching cell receipts.
package ultracodecrossover
