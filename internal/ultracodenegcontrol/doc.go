// Package ultracodenegcontrol evaluates predeclared negative controls for the
// frozen managed-context campaign. It refuses to turn unequal outcomes or
// incomplete telemetry into token savings.
//
// Invariant: UltraCode negative control evaluations are fail-closed and reproducible.
// Any missing telemetry, altered outcomes, or unverified gains force abstention or contradiction.
// Guard: Placebo and adversarial perturbations must produce zero credited savings under all conditions.
//
// Tier: primitive (1) - see internal/architest. This package may import only
// packages whose tier is <= 1; an upward import fails the architest gate.
package ultracodenegcontrol
