// Package codelint is language-server packs: lint agent-written code (Go/Python/CUDA/JSON) off the hot path.
//
// Tier: foundation (1) — see internal/architest. This package may import only
// packages whose tier is <= 1; an upward import fails the architest gate.
// See AGENTS.md and internal/architest for the layering contract.
package codelint
