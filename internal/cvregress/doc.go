// Package cvregress is per-session cache-efficiency (hit% + write-amp) regression flagging over the cache-savings ledger axes.
//
// Tier: mechanism (2) - see internal/architest. This package may import only
// packages whose tier is <= 2; an upward import fails the architest gate.
// See AGENTS.md and internal/architest for the layering contract.
package cvregress
