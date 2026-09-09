package model

import "fmt"

// GLM5NextContextStage defines evaluated context length tiers.
type GLM5NextContextStage struct {
	Name               string `json:"name"`              // "short", "32k", "128k", "256k", "1m"
	ContextTokens      int    `json:"context_tokens"`    // 4096, 32768, 131072, 262144, 1048576
	KDABallastBytes    int64  `json:"kda_ballast_bytes"` // fixed O(1) state: 34 layers * 64 heads * 128 * 128 * 4
	DSAKVBytes         int64  `json:"dsa_kv_bytes"`      // O(N): 11 layers * 512 latent * 2 bytes * N
	ActiveMoEBytes     int64  `json:"active_moe_bytes"`  // 18B active parameters (~18.2 GB in FP8)
	TotalRequiredBytes int64  `json:"total_required_bytes"`
	FeasibleOn96GB     bool   `json:"feasible_on_96gb"`
	FeasibleOn128GB    bool   `json:"feasible_on_128gb"`
}

// GLM5NextOperatingEnvelope models and checks the capacity envelope for GLM-5.3-Flash.
type GLM5NextOperatingEnvelope struct {
	Model              string                 `json:"model"`
	TotalParameters    int64                  `json:"total_parameters"`
	ActiveParameters   int64                  `json:"active_parameters"`
	LayersTotal        int                    `json:"layers_total"`
	KDARecurrentLayers int                    `json:"kda_recurrent_layers"`
	DSASparseLayers    int                    `json:"dsa_sparse_layers"`
	Stages             []GLM5NextContextStage `json:"stages"`
}

// DefaultGLM5NextOperatingEnvelope returns the verified operating envelope parameters.
func DefaultGLM5NextOperatingEnvelope() *GLM5NextOperatingEnvelope {
	const kdaLayers = 34
	const dsaLayers = 11
	const numHeads = 64
	const headDim = 128
	const dsaLatentRank = 512

	// Fixed KDA recurrent state: 34 * 64 * 128 * 128 * 4 bytes = 142,606,336 bytes (~142.6 MB)
	kdaFixedBytes := int64(kdaLayers) * int64(numHeads) * int64(headDim) * int64(headDim) * 4

	// Active weights in FP8 (~18.2 GB)
	activeWeightBytes := int64(18_200_000_000)

	tiers := []struct {
		name   string
		tokens int
	}{
		{"short", 4096},
		{"32k", 32768},
		{"128k", 131072},
		{"256k", 262144},
		{"1m", 1048576},
	}

	stages := make([]GLM5NextContextStage, len(tiers))
	for i, t := range tiers {
		// DSA KV cache: 11 layers * 512 latent * 2 bytes/token * tokens
		dsaBytes := int64(dsaLayers) * int64(dsaLatentRank) * 2 * int64(t.tokens)
		total := kdaFixedBytes + dsaBytes + activeWeightBytes
		stages[i] = GLM5NextContextStage{
			Name:               t.name,
			ContextTokens:      t.tokens,
			KDABallastBytes:    kdaFixedBytes,
			DSAKVBytes:         dsaBytes,
			ActiveMoEBytes:     activeWeightBytes,
			TotalRequiredBytes: total,
			FeasibleOn96GB:     total <= 96*1024*1024*1024,
			FeasibleOn128GB:    total <= 128*1024*1024*1024,
		}
	}

	return &GLM5NextOperatingEnvelope{
		Model:              "zai-org/GLM-5.3-Flash",
		TotalParameters:    320_000_000_000,
		ActiveParameters:   18_000_000_000,
		LayersTotal:        45,
		KDARecurrentLayers: kdaLayers,
		DSASparseLayers:    dsaLayers,
		Stages:             stages,
	}
}

// AdmitGLM5NextContext checks if a requested context length fits within available VRAM.
func (env *GLM5NextOperatingEnvelope) AdmitGLM5NextContext(contextTokens int, availableVRAMBytes int64) error {
	if contextTokens <= 0 {
		return fmt.Errorf("model: invalid contextTokens %d", contextTokens)
	}
	if contextTokens > 1048576 {
		return fmt.Errorf("model: context length %d exceeds 1M architecture maximum", contextTokens)
	}
	kdaBytes := int64(env.KDARecurrentLayers) * 64 * 128 * 128 * 4
	dsaBytes := int64(env.DSASparseLayers) * 512 * 2 * int64(contextTokens)
	activeWeightBytes := int64(18_200_000_000)
	required := kdaBytes + dsaBytes + activeWeightBytes

	if required > availableVRAMBytes {
		return fmt.Errorf("model: capacity refused: required %d bytes > available %d bytes for context %d",
			required, availableVRAMBytes, contextTokens)
	}
	return nil
}
