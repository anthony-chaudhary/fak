// Package trajctl is the trajectory-control objective, score-row, witness-rung,
// and JSONL ledger model.
//
// Tier: foundation (1) - see internal/architest. This package may import only
// packages whose tier is <= 1; an upward import fails the architest gate.
// See AGENTS.md and internal/architest for the layering contract.
//
// The package is deliberately a data-plane leaf. It does not score, steer, spawn,
// or shell out. Producers append Objective and ScoreRow records to the ledger;
// later scorer and steering leaves fold those witnessed rows.
package trajctl
