package macobs

// HeadroomConfig parameterizes model architecture and memory reservations for headroom modeling.
type HeadroomConfig struct {
	Layers             uint64 `json:"layers"`
	KVHeads            uint64 `json:"kv_heads"`
	HeadDim            uint64 `json:"head_dim"`
	KVBytesPerElement  uint64 `json:"kv_bytes_per_element"` // e.g. 2 for fp16/bf16, 1 for fp8
	ModelWeightBytes   uint64 `json:"model_weight_bytes"`
	ContextTokens      uint64 `json:"context_tokens"`
	SharedPrefixTokens uint64 `json:"shared_prefix_tokens"`
	PrivateTailTokens  uint64 `json:"private_tail_tokens"`
	OSReserveBytes     uint64 `json:"os_reserve_bytes"`
}

// DefaultHeadroomConfig returns representative defaults for a 7B/8B GQA model on Apple Silicon.
func DefaultHeadroomConfig() HeadroomConfig {
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
	if cfg.Layers == 0 {
		cfg.Layers = 28
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

	// 2 * Layers * KVHeads * HeadDim * KVBytesPerElement (2 for Key + Value)
	var overflow bool
	kvBytesPerToken := uint64(2)
	for _, f := range []uint64{cfg.Layers, cfg.KVHeads, cfg.HeadDim, cfg.KVBytesPerElement} {
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

	// Max isolated agents (each requiring full context KV allocation)
	isolatedKVPerAgent, isoOverflow := safeMul(cfg.ContextTokens, kvBytesPerToken)
	var maxIsolated int
	if !isoOverflow && !overflow && isolatedKVPerAgent > 0 && availableKVPool > 0 {
		maxIsolated = int(availableKVPool / isolatedKVPerAgent)
	}

	// Max shared prefix agents (1 shared prefix + private tail per agent)
	sharedPrefixKV, sharedOverflow := safeMul(cfg.SharedPrefixTokens, kvBytesPerToken)
	tailKVPerAgent, tailOverflow := safeMul(cfg.PrivateTailTokens, kvBytesPerToken)
	var maxShared int
	if !sharedOverflow && !tailOverflow && !overflow && availableKVPool >= sharedPrefixKV && tailKVPerAgent > 0 {
		rem := availableKVPool - sharedPrefixKV
		maxShared = int(rem / tailKVPerAgent)
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
		Available:            (hw.Available || wiredLimit > 0) && availableKVPool > 0,
	}
}
