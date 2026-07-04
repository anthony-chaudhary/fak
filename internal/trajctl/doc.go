// Package trajctl is the trajectory-control objective, score-row, witness-rung,
// and JSONL ledger model.
//
// Tier: foundation (1) - see internal/architest. This package may import only
// packages whose tier is <= 1; an upward import fails the architest gate.
// See AGENTS.md and internal/architest for the layering contract.
//
// The package is a data-plane leaf plus the pure scoring fold. It does not steer,
// spawn, or shell out: a Scorer folds an Objective and an injected evidence
// window into ScoreRows, and the impure evidence resolvers (git, transcript,
// verdict-hash) are injected from the call site so the fold stays deterministic
// and tier-1. Producers append Objective and ScoreRow records to the ledger;
// later steering leaves fold those witnessed rows.
package trajctl
