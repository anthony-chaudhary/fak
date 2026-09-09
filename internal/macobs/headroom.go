package macobs

// HeadroomConfig parameterizes model architecture and memory reservations for headroom modeling.
type HeadroomConfig struct {
	Layers              uint64 `json:"layers"`
	FullAttnLayers      uint64 `json:"full_attn_layers,omitempty"`
	RecurrentLayers     uint64 `json:"recurrent_layers,omitempty"`
	RecurrentStateBytes uint64 `json:"recurrent_state_bytes,omitempty"`
	KVHeads             uint64 `json:"kv_heads"`
	HeadDim             uint64 `json:"head_dim"`
	KVBytesPerElement   uint64 `json:"kv_bytes_per_element"` // e.g. 2 for fp16/bf16, 1 for fp8
	ModelWeightBytes    uint64 `json:"model_weight_bytes"`
	ContextTokens       uint64 `json:"context_tokens"`
	SharedPrefixTokens  uint64 `json:"shared_prefix_tokens"`
	PrivateTailTokens   uint64 `json:"private_tail_tokens"`
	OSReserveBytes      uint64 `json:"os_reserve_bytes"`
}

// EffectiveKVLayers returns the number of layers maintaining token-indexed KV cache rows.
// For hybrid GDN architectures (like Qwen3.8 3:1 GDN), this returns FullAttnLayers.
func (cfg HeadroomConfig) EffectiveKVLayers() uint64 {
	if cfg.FullAttnLayers > 0 {
		return cfg.FullAttnLayers
	}
	if cfg.Layers > 0 {
		return cfg.Layers
	}
	return 28
}

// HybridRatio returns the ratio of full-attention layers to total layers (e.g. 0.25 for 3:1 GDN).
func (cfg HeadroomConfig) HybridRatio() float64 {
	total := cfg.Layers
	if total == 0 {
		total = 28
	}
	return float64(cfg.EffectiveKVLayers()) / float64(total)
}

const (
	// DefaultQwen38RecurrentStatePerAgentBytes is the fixed O(1) recurrent state across 48 linear-attention
	// layers in Qwen3.8-27B (~3.2 MB/agent, specifically 3,355,443 bytes).
	DefaultQwen38RecurrentStatePerAgentBytes uint64 = 3355443
)

// DefaultHeadroomConfig returns representative defaults for Qwen3.8 27B 3:1 GDN on Apple Silicon (36GB unified memory).
func DefaultHeadroomConfig() HeadroomConfig {
	return Qwen38GDNHeadroomConfig()
}

// Qwen38GDNHeadroomConfig returns the 3:1 GDN hybrid architecture configuration for Qwen3.8-27B
// (16 full-attention layers + 48 recurrent linear-attention layers) on 36GB Apple Silicon hardware.
func Qwen38GDNHeadroomConfig() HeadroomConfig {
	return HeadroomConfig{
		Layers:              64,
		FullAttnLayers:      16,
		RecurrentLayers:     48,
		RecurrentStateBytes: DefaultQwen38RecurrentStatePerAgentBytes,
		KVHeads:             8,
		HeadDim:             128,
		KVBytesPerElement:   2,
		ModelWeightBytes:    16 * 1024 * 1024 * 1024, // ~16GB Q4_K_M weights
		ContextTokens:       8192,
		SharedPrefixTokens:  4096,               // Global RadixAttention preamble (0.25 GB)
		PrivateTailTokens:   1024,               // Private agent reasoning tail
		OSReserveBytes:      7680 * 1024 * 1024, // ~7.5GB macOS system reserve
	}
}

// Standard7BHeadroomConfig returns representative defaults for a 7B/8B GQA standard transformer model.
func Standard7BHeadroomConfig() HeadroomConfig {
	return HeadroomConfig{
		Layers:             28,                     // e.g. Qwen2.5 7B
		KVHeads:            4,                      // Grouped Query Attention
		HeadDim:            128,                    // Standard head dimension
		KVBytesPerElement:  2,                      // fp16/bf16
		ModelWeightBytes:   5 * 1024 * 1024 * 1024, // ~5GB 4-bit quantized weights
		ContextTokens:      8192,                   // Full context window
		SharedPrefixTokens: 4096,                   // System prompt + tool preamble
		PrivateTailTokens:  2048,                   // Per-agent reasoning and tool tail
		OSReserveBytes:     3 * 1024 * 1024 * 1024, // 3GB macOS system reserve
	}
}

func safeMul(a, b uint64) (uint64, bool) {
	if a == 0 || b == 0 {
		return 0, false
	}
	c := a * b
	if c/a != b {
		return ^uint64(0), true
	}
	return c, false
}

// ComputeHeadroom calculates unified memory headroom, KV bytes, and agent concurrency limits.
func ComputeHeadroom(hw HardwareTelemetry, cfg HeadroomConfig) HeadroomTelemetry {
	// Sanitize and apply default fallbacks for zero values
	effLayers := cfg.EffectiveKVLayers()
	if effLayers == 0 {
		effLayers = 28
	}
	if cfg.KVHeads == 0 {
		cfg.KVHeads = 4
	}
	if cfg.HeadDim == 0 {
		cfg.HeadDim = 128
	}
	if cfg.KVBytesPerElement == 0 {
		cfg.KVBytesPerElement = 2
	}
	if cfg.ContextTokens == 0 {
		cfg.ContextTokens = 8192
	}
	if cfg.PrivateTailTokens == 0 {
		cfg.PrivateTailTokens = 2048
	}
	if cfg.SharedPrefixTokens >= cfg.ContextTokens {
		cfg.SharedPrefixTokens = cfg.ContextTokens / 2
	}

	// 2 * EffectiveKVLayers * KVHeads * HeadDim * KVBytesPerElement (2 for Key + Value)
	var overflow bool
	kvBytesPerToken := uint64(2)
	for _, f := range []uint64{effLayers, cfg.KVHeads, cfg.HeadDim, cfg.KVBytesPerElement} {
		var o bool
		kvBytesPerToken, o = safeMul(kvBytesPerToken, f)
		if o {
			overflow = true
			break
		}
	}
	if overflow {
		kvBytesPerToken = ^uint64(0)
	}

	wiredLimit := hw.WiredMemoryLimitBytes
	if wiredLimit == 0 && hw.TotalSystemMemoryBytes > 0 {
		wiredLimit = (hw.TotalSystemMemoryBytes * 3) / 4
	}

	requiredBase := cfg.ModelWeightBytes + cfg.OSReserveBytes
	if requiredBase < cfg.ModelWeightBytes { // uint64 overflow protection
		requiredBase = ^uint64(0)
	}

	var availableKVPool uint64
	if wiredLimit > requiredBase {
		availableKVPool = wiredLimit - requiredBase
	}

	// Max isolated agents (each requiring full context KV allocation + recurrent state)
	isolatedKVPerAgent, isoOverflow := safeMul(cfg.ContextTokens, kvBytesPerToken)
	isolatedPerAgent := isolatedKVPerAgent + cfg.RecurrentStateBytes
	var maxIsolated int
	if !isoOverflow && !overflow && isolatedPerAgent > 0 && availableKVPool > 0 {
		maxIsolated = int(availableKVPool / isolatedPerAgent)
	}

	// Max shared prefix agents (1 shared prefix + private tail + recurrent state per agent)
	sharedPrefixKV, sharedOverflow := safeMul(cfg.SharedPrefixTokens, kvBytesPerToken)
	tailKVPerAgent, tailOverflow := safeMul(cfg.PrivateTailTokens, kvBytesPerToken)
	tailPerAgent := tailKVPerAgent + cfg.RecurrentStateBytes
	var maxShared int
	if !sharedOverflow && !tailOverflow && !overflow && availableKVPool >= sharedPrefixKV && tailPerAgent > 0 {
		rem := availableKVPool - sharedPrefixKV
		maxShared = int(rem / tailPerAgent)
	}

	var concurrencyAdv float64
	if maxIsolated > 0 {
		concurrencyAdv = float64(maxShared) / float64(maxIsolated)
	} else if maxShared > 0 {
		concurrencyAdv = float64(maxShared)
	} else {
		concurrencyAdv = 1.0
	}

	return HeadroomTelemetry{
		ModelKVBytesPerToken: kvBytesPerToken,
		AvailableKVPoolBytes: availableKVPool,
		MaxSharedAgents:      maxShared,
		MaxIsolatedAgents:    maxIsolated,
		ConcurrencyAdvantage: concurrencyAdv,
		SharedPrefixTokens:   cfg.SharedPrefixTokens,
		PrivateTailTokens:    cfg.PrivateTailTokens,
		ModelWeightBytes:     cfg.ModelWeightBytes,
		FullAttnLayers:       cfg.FullAttnLayers,
		RecurrentLayers:      cfg.RecurrentLayers,
		RecurrentStateBytes:  cfg.RecurrentStateBytes,
		SharedPrefixBytes:    sharedPrefixKV,
		TailKVBytesPerAgent:  tailKVPerAgent,
		Available:            (hw.Available || wiredLimit > 0) && availableKVPool > 0,
	}
}
