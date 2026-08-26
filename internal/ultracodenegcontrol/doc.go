// Package ultracodenegcontrol evaluates predeclared negative controls for the
// frozen managed-context campaign. It refuses to turn unequal outcomes or
// incomplete telemetry into token savings.
//
// Tier: primitive (1) - see internal/architest. This package may import only
// packages whose tier is <= 1; an upward import fails the architest gate.
package ultracodenegcontrol
