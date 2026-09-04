// Package ultracodetokenizer evaluates tokenizer-portable Ultracode context-omission receipts.
//
// It keeps semantic omission (bytes and messages), model input tokens, and
// runtime cache tokens in separate provenance classes. Cross-tokenizer shares
// are emitted only for byte-identical canonical inputs with equal outcomes.
//
// Invariant: UltraCode tokenizer evaluation is fail-closed, deterministic, and requires at least three distinct tokenizer families.
// Guard: Comparisons abstain when canonical inputs, digest receipts, or accepted outcomes mismatch across measurements.
package ultracodetokenizer
