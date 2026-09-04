// Package dogfoodscore scores the launched-session dogfooding loop.
//
// Invariant: dogfood scoring is fail-closed and evidence-backed. Missing,
// unreadable, or corrupted transcripts fail closed as unwitnessed rather than assuming success.
//
// Guard: hook errors and assistant success claims are correlated within a strict
// context event window to prevent false-negative conflation detection.
package dogfoodscore
