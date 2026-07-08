// Package l3kv is durable disk-backed L3 KV residency backend: StageSpan/RestoreSpan persist a demoted span to blobfs by digest (#1472).
//
// Tier: foundation (1) - see internal/architest. This package may import only
// packages whose tier is <= 1; an upward import fails the architest gate.
// See AGENTS.md and internal/architest for the layering contract.
package l3kv
