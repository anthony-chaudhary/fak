// Package customizationindex provides schema validation, freshness tracking,
// and structural grouping for agent customization indexes.
//
// Invariant: customization index checks are fail-closed and deterministic.
// Guard: malformed schema versions, unknown fields, or missing evidence reject the entire index.
package customizationindex
