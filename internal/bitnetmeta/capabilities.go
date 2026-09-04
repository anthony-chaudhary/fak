package bitnetmeta

// DefaultCapabilities returns the baseline recognized schemas, formats, activations,
// packings, recipes, runtimes, and hardware envelopes for BitNet metadata adjudication.
func DefaultCapabilities() Capabilities {
	return Capabilities{
		Schemas: []string{
			SchemaV1,
		},
		Formats: []string{
			"safetensors@1",
			"gguf@3",
		},
		Activations: []string{
			"integer/8",
			"bfloat/16",
			"float/16",
		},
		Packings: []string{
			"bitplane-lsb",
			"i2_s-pair",
			"two-bit-codes",
		},
		Recipes: []string{
			"native-bitnet@1",
			"absmean-ternarize@2",
			"int2-quantize@1",
		},
		Runtimes: []string{
			"bitnet.cpp@2026.08",
		},
		Hardware: []string{
			"cpu/x86-64-avx2",
		},
	}
}
