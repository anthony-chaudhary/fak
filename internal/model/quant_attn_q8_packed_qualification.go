//go:build fak_qattn_q8_packed_qualification

package model

// q8PackedDecodeExperimental is an immutable build-time qualification posture.
// Ordinary builds remain off. Enabling the qualification tag changes only this
// admission gate, not QSA/serial precedence, eligibility, fallback or arithmetic.
const q8PackedDecodeExperimental = true
