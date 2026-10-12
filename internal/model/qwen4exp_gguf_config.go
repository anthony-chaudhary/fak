package model

// Qwen4ExpHParams carries the Qwen3.8-Flash-Next ("qwen4exp") GGUF axes that have
// no generic Config field: hyper-connections, the QSA indexer, mrope sections and
// the PLE n-gram hash. Nil for every other arch.
type Qwen4ExpHParams struct {
	HCCount         int
	HCLowRank       int
	RopeSections    []int
	IndexNHeads     int
	IndexHeadDim    int
	IndexTopK       int
	CompressRatios  []int
	SharedExpertFFN int
	PLE             *Qwen4ExpPLE
}

// Qwen4ExpPLE is the per-layer n-gram hash embedding read from the GGUF ple.* keys.
type Qwen4ExpPLE struct {
	Layers           []int
	NgramSize        int
	HeadsPerNgram    int
	NumHeads         int
	ConvKernel       int
	EOSTokenID       int
	ImageTokenID     int
	EmbedPerLayer    int
	LayerMultipliers []uint64
	HeadOffsets      []uint64
	HeadVocabSizes   []uint64
}

// qwen4ExpGGUFModelType is the canonical ModelType a qwen4exp GGUF loads as.
const qwen4ExpGGUFModelType = "qwen4exp"

// IsQwen4Exp reports whether c is a Qwen3.8-Flash-Next config, from a GGUF or HF.
func (c Config) IsQwen4Exp() bool {
	return c.ModelType == qwen4ExpGGUFModelType || c.ModelType == qwen4ExpModelType
}

// refuseQwen4ExpNative keeps a qwen4exp config off every generic forward: its
// layer_types mark it a GDN hybrid, but the qwen35 forward has no
// hyper-connections, QSA indexer or PLE, so it would run and return garbage.
func refuseQwen4ExpNative(c Config) error {
	if !c.IsQwen4Exp() {
		return nil
	}
	return &UnsupportedQwen4ExpExecutionError{Engine: "fak-native", ModelType: c.ModelType, Architecture: qwen4ExpArchitecture}
}
