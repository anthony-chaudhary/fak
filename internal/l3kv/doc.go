// Package l3kv is the durable L3 KV residency backend: StageSpan/RestoreSpan persist a demoted span by digest through the storedrv router (blobfs local stand-in + optional blobhttp remote pool) behind a durable span→content manifest (#1472).
//
// Tier: mechanism (2) - see internal/architest. This package may import only
// packages whose tier is <= 2; an upward import fails the architest gate.
// See AGENTS.md and internal/architest for the layering contract.
package l3kv
