package fp4meta

// DefaultCapabilities returns the baseline recognized variants, scale formats,
// accumulators, hardware, and runtime support for FP4 artifact adjudication.
func DefaultCapabilities() Capabilities {
	return Capabilities{
		Variants:     []Variant{VariantE2M1, VariantNVFP4, VariantMXFP4},
		ScaleFormats: []ScaleEncoding{ScaleBinary32, ScaleE4M3, ScaleUE8M0},
		Accumulators: []Accumulator{AccumulatorFP16, AccumulatorFP32},
		Hardware:     []string{"nvidia/sm100"},
		Runtime:      true,
	}
}
