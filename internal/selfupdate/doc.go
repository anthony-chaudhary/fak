// Package selfupdate owns updater orchestration decisions and automation receipts.
//
// Tier: mechanism (3) - see internal/architest. This package may import only
// packages whose tier is <= 3; an upward import fails the architest gate.
// See AGENTS.md and internal/architest for the layering contract.
package selfupdate
