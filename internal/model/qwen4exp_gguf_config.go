package model

import (
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/qwen4exp"
)

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
	// DType is the PLE table's resolved dtype. It is the checkpoint-declared
	// dtype when the metadata carries one, else the model default. DTypeSource
	// records which of the two it is, so a caller can tell "declared" from
	// "assumed" rather than trusting a silently-defaulted value.
	DType       string
	DTypeSource PLEDTypeSource
}

// PLEDTypeSource records how a PLE dtype was resolved.
type PLEDTypeSource string

const (
	// PLEDTypeDeclared means the checkpoint metadata declared the PLE dtype.
	PLEDTypeDeclared PLEDTypeSource = "declared"
	// PLEDTypeAssumed means the metadata omitted the PLE dtype and the model
	// default was used. Upstream TensorRT-LLM (ple.py:69) shows that assuming
	// the wrong dtype for the PLE table silently produces a wrong-precision
	// embedding lookup, so an assumed dtype is a distinct, reportable state.
	PLEDTypeAssumed PLEDTypeSource = "assumed"
)

// PLEDTypeResolution is the outcome of resolving the PLE table's dtype.
type PLEDTypeResolution struct {
	DType  string
	Source PLEDTypeSource
}

// Declared reports whether the PLE dtype came from checkpoint metadata rather
// than the model default.
func (r PLEDTypeResolution) Declared() bool { return r.Source == PLEDTypeDeclared }

// ResolvePLEDType resolves the PLE table's dtype from raw checkpoint metadata,
// falling back to modelDefault only when the metadata genuinely omits the PLE
// dtype. A metadata-declared dtype always wins over the model default, and the
// returned Source records which path was taken.
func ResolvePLEDType(metadata []byte, modelDefault string) (PLEDTypeResolution, error) {
	meta, err := qwen4exp.PLEDTypeFromManifest(metadata)
	if err != nil {
		return PLEDTypeResolution{}, err
	}
	if meta.Declared {
		return PLEDTypeResolution{DType: meta.DType, Source: PLEDTypeDeclared}, nil
	}
	fallback := strings.TrimSpace(modelDefault)
	if fallback == "" {
		return PLEDTypeResolution{}, fmt.Errorf("model: PLE dtype absent from metadata and no model default to fall back to")
	}
	return PLEDTypeResolution{DType: fallback, Source: PLEDTypeAssumed}, nil
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
