// Package ultracodeborrow validates external workflow borrowing artifacts against required mechanisms,
// license boundaries, benchmark contracts, and ownership claims.
//
// Invariant: UltraCode borrowing verification is fail-closed and provenance-verified.
// Guard: All candidate mechanisms, source license dispositions, and benchmark definitions
// must strictly satisfy schema identity, immutable anchors, and deduplicated ownership before admission.
//
// Tier: primitive (1) - see internal/architest. This package may import only
// packages whose tier is <= 1; an upward import fails the architest gate.
package ultracodeborrow
