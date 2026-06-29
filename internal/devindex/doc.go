// Package devindex is queryable self-index over fak's own dev facts (lanes/leaves + doc map): query, don't survey.
//
// Tier: foundation (1) — see internal/architest. This package may import only
// packages whose tier is <= 1; an upward import fails the architest gate.
// See AGENTS.md and internal/architest for the layering contract.
package devindex
