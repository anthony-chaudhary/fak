// Package orgdebt grades organizational health and shift-left maturity across
// backlog readiness, task scope, lane contention, merge hygiene, and spine fan-out.
//
// Tier: primitive (1) - see internal/architest. This package may import only
// packages whose tier is <= 1; an upward import fails the architest check.
// See AGENTS.md and internal/architest for the layering contract.
package orgdebt
