package macfit

// Inkling2MiniMinContextTokens is the single-session context the Inkling-2
// launcher admits, which a host must hold to count as fitting the model.
const Inkling2MiniMinContextTokens = 8192

// Inkling2MiniTier is Inkling-2-Mini UD-Q3_K_XL on the Apple GPU. WeightBytes is
// the resident plan of the 13,631,779,296-byte GGUF rather than its file size:
// packed target images, the DFlash2 draft images, prefill and decode scratch,
// and session state. Layers, KVHeads and HeadDim are the KV-equivalent of the
// looped forward (24 layers visited three times), chosen so the FP16 estimate
// equals the session's 101,376 KV bytes per token.
var Inkling2MiniTier = ModelTier{
	Name:           "Inkling-2-Mini",
	ModelID:        "inkling2-mini",
	QuantTier:      "Q3_K_XL",
	WeightBytes:    15_534_497_792,
	Layers:         72,
	KVHeads:        11,
	HeadDim:        32,
	MinMemoryBytes: 24 * GiB,

	FixedContextTokens: Inkling2MiniMinContextTokens,
}

// Inkling2MiniFits reports whether memoryBytes of unified memory holds the
// resident model plus one minimum-context session inside the 20% reserve.
func Inkling2MiniFits(memoryBytes uint64) bool {
	tier := Inkling2MiniTier
	if memoryBytes < tier.MinMemoryBytes {
		return false
	}
	kv, err := mul(2, tier.Layers, tier.KVHeads, tier.HeadDim, 2, Inkling2MiniMinContextTokens)
	if err != nil {
		return false
	}
	usable := memoryBytes - (memoryBytes*20)/100
	return usable >= tier.WeightBytes && usable-tier.WeightBytes >= kv
}

// MacDefaultTier returns the first-run default tier for an Apple Silicon host:
// Inkling-2-Mini when it fits. It reports false on every other platform, when
// the model does not fit, and when servable is false, leaving the caller on the
// StandardTiers ladder.
func MacDefaultTier(goos, goarch string, memoryBytes uint64, servable bool) (ModelTier, bool) {
	if goos != "darwin" || goarch != "arm64" || !servable || !Inkling2MiniFits(memoryBytes) {
		return ModelTier{}, false
	}
	return Inkling2MiniTier, true
}
