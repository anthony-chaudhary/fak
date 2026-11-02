//go:build !fak_qwen35_prefill_tiled_attention_qualification

package model

// The online reduction has no default runtime or measured-performance approval.
const qwen35PrefillTiledAttentionQualification = false
