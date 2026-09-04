// Package devexmeter is dev-ex friction meter and RSI close gate.
//
// Tier: foundation (1) — see internal/architest. This package may import only
// packages whose tier is <= 1; an upward import fails the architest gate.
// See AGENTS.md and internal/architest for the layering contract.
//
// Invariant: developer experience metering is fail-closed and bounded.
// Unparsed rows, invalid numeric values, or unverified friction claims are
// rejected immediately rather than silently ignored or approximated.
//
// Guard: GateIssue returns PASS only when a tagged dev-ex friction issue provides
// witnessed before and after meter windows with a strictly decreasing after value.
// Any missing windows, equal values, or increasing friction yield NOT_YET verdicts.
package devexmeter
