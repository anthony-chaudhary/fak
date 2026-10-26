package quantmeta

import (
	"errors"
	"fmt"
)

// DtypeDescriptor describes one fixed storage layout, not a kernel capability or
// conversion. Bits counts payload code bits; BytesPerElement includes block
// metadata. Symmetric means no affine minimum/zero-point, not equal endpoint
// magnitudes. A negative GGUF/Dtype means that vocabulary has no fixed-layout
// equivalent. Empty Format means the neutral vocabulary has no matching term.
// Numeric enum IDs keep this neutral package independent of compute/ggufload.
type DtypeDescriptor struct {
	Token           string
	GGUF            int64
	Dtype           int16
	Format          Format
	Bits            int
	BlockSize       int
	BlockBytes      int
	Symmetric       bool
	BytesPerElement float64
}

var (
	ErrUnknownFormat     = errors.New("quantmeta: unknown enum value")
	ErrUnsupportedFormat = errors.New("quantmeta: storage layout unsupported")
	ErrAmbiguousFormat   = errors.New("quantmeta: ambiguous storage layout")
)

func storage(token string, gguf int64, dtype int16, format Format, bits, block, bytes int, symmetric bool) DtypeDescriptor {
	return DtypeDescriptor{token, gguf, dtype, format, bits, block, bytes, symmetric, float64(bytes) / float64(block)}
}

// Geometry follows ggufload/gguf_type_traits.go and its block constants. HAL
// K/IQ formats retain these raw blocks. In contrast HAL Q8_0 and Q2_0 use
// runtime block sizes and separate float32 scales: they are NOT GGUF layouts.
var dtypeRegistry = []DtypeDescriptor{
	storage("f32", 0, 0, FormatF32, 32, 1, 4, true),
	storage("f16", 1, 1, FormatF16, 16, 1, 2, true),
	storage("bf16", 30, 2, FormatBF16, 16, 1, 2, true),
	storage("gguf:q4_0", 2, -1, FormatInt4, 4, 32, 18, true),
	storage("gguf:q4_1", 3, -1, FormatInt4, 4, 32, 20, false),
	storage("gguf:q5_0", 6, -1, "", 5, 32, 22, true),
	storage("gguf:q5_1", 7, -1, "", 5, 32, 24, false),
	storage("gguf:q8_0", 8, -1, FormatInt8, 8, 32, 34, true),
	storage("gguf:q2_k", 10, 11, FormatInt2, 2, 256, 84, false),
	storage("gguf:q3_k", 11, 12, FormatInt3, 3, 256, 110, true),
	storage("gguf:q4_k", 12, 7, FormatInt4, 4, 256, 144, false),
	storage("gguf:q5_k", 13, 8, "", 5, 256, 176, false),
	storage("gguf:q6_k", 14, 9, FormatInt6, 6, 256, 210, true),
	storage("gguf:iq2_xxs", 16, 15, FormatCodebook, 2, 256, 66, true),
	storage("gguf:iq2_xs", 17, 19, FormatCodebook, 2, 256, 74, true),
	storage("gguf:iq3_xxs", 18, 13, FormatCodebook, 3, 256, 98, true),
	storage("gguf:iq4_nl", 20, -1, FormatCodebook, 4, 32, 18, true),
	storage("gguf:iq3_s", 21, 14, FormatCodebook, 3, 256, 110, true),
	storage("gguf:iq2_s", 22, 18, FormatCodebook, 2, 256, 82, true),
	storage("gguf:iq4_xs", 23, 17, FormatCodebook, 4, 256, 136, true),
	storage("gguf:mxfp4", 39, -1, FormatMXFP4, 4, 32, 17, true),
	storage("gguf:q2_0", 42, -1, FormatTernary, 2, 128, 34, true),
	storage("gguf:pq2_0", 142, -1, FormatTernary, 2, 128, 34, true),
}

var ggufEnum = map[string]uint32{
	"TensorF32": 0, "TensorF16": 1, "TensorQ4_0": 2, "TensorQ4_1": 3, "TensorQ5_0": 6, "TensorQ5_1": 7, "TensorQ8_0": 8,
	"TensorQ2_K": 10, "TensorQ3_K": 11, "TensorQ4_K": 12, "TensorQ5_K": 13, "TensorQ6_K": 14,
	"TensorIQ2_XXS": 16, "TensorIQ2_XS": 17, "TensorIQ3_XXS": 18, "TensorIQ1_S": 19, "TensorIQ4_NL": 20, "TensorIQ3_S": 21, "TensorIQ2_S": 22, "TensorIQ4_XS": 23, "v41EngramRawI8Type": 24, "TensorIQ1_M": 29, "TensorBF16": 30, "TensorMXFP4": 39, "TensorQ2_0": 42, "TensorPQ2_0": 142, "TensorQ1_0": 43, "TensorHQQ4": 44,
}
var dtypeEnum = map[string]uint8{
	"F32": 0, "F16": 1, "BF16": 2, "Q8_0": 3, "I8": 4, "I4": 5, "FP8": 6, "Q4_K": 7, "Q5_K": 8, "Q6_K": 9, "Q2_0": 10, "Q2_K": 11, "Q3_K": 12, "IQ3_XXS": 13, "IQ3_S": 14, "IQ2_XXS": 15, "FP4": 16, "IQ4_XS": 17, "IQ2_S": 18, "IQ2_XS": 19, "IQ1_S": 20, "NVFP4": 21,
}
var unsupportedGGUF = map[uint32]string{
	24: "private packed V4.1 Engram row encoding (264 bytes), not an ordinary GGUF tensor block",
	19: "IQ1 offset/grid semantics are not represented by this descriptor",
	29: "IQ1 offset/grid semantics are not represented by this descriptor",
	43: "Q1_0 numeric tag is provisional pending a real file header",
	44: "HQQ4 numeric tag is provisional pending a real file header",
}
var unsupportedDtype = map[uint8]string{
	3:  "Q8_0 block size and separate float32 scales require QuantSpec; not packed GGUF Q8_0",
	4:  "generic int8 layout requires QuantSpec",
	5:  "generic int4 layout requires QuantSpec",
	6:  "FP8 variant and scale layout require QuantSpec",
	10: "Q2_0 block size and separate float32 scales require QuantSpec; not packed GGUF Q2_0",
	16: "FP4 permits different scale types; requires QuantSpec",
	20: "IQ1 offset/grid semantics are not represented by this descriptor",
	21: "NVFP4 block and tensor scale planes require a richer descriptor; not MXFP4",
}
var unsupportedFormat = map[Format]string{
	FormatFP8E4M3: "numeric format alone does not specify scale/layout",
	FormatFP8E5M2: "numeric format alone does not specify scale/layout",
	FormatFP4E2M1: "numeric format alone does not specify scale/layout",
	FormatMXFP6:   "no fixed storage identity in current GGUF/Dtype vocabulary",
	FormatMXFP8:   "no fixed storage identity in current GGUF/Dtype vocabulary",
	FormatNF4:     "no fixed storage identity in current GGUF/Dtype vocabulary",
	FormatInt8:    "numeric width does not imply GGUF block/scales",
	FormatInt6:    "numeric width does not imply GGUF block/scales",
	FormatInt3:    "numeric width does not imply GGUF block/scales",
	FormatInt2:    "numeric width does not imply GGUF block/scales",
	FormatBinary:  "no verified fixed storage identity in current vocabulary",
}

// Entries returns an independent snapshot; callers cannot mutate the registry.
func Entries() []DtypeDescriptor { return append([]DtypeDescriptor(nil), dtypeRegistry...) }

// FromGGUF resolves the GGUF numeric domain only. Named TensorType values can
// be explicitly converted to uint32 without importing ggufload here.
func FromGGUF(id uint32) (DtypeDescriptor, error) {
	for _, e := range dtypeRegistry {
		if e.GGUF == int64(id) {
			return e, nil
		}
	}
	if reason, ok := unsupportedGGUF[id]; ok {
		return DtypeDescriptor{}, fmt.Errorf("%w: GGUF %d: %s", ErrUnsupportedFormat, id, reason)
	}
	return DtypeDescriptor{}, fmt.Errorf("%w: GGUF %d", ErrUnknownFormat, id)
}

// FromDtype resolves the HAL numeric domain only, never a GGUF numeric fallback.
func FromDtype(id uint8) (DtypeDescriptor, error) {
	for _, e := range dtypeRegistry {
		if e.Dtype == int16(id) {
			return e, nil
		}
	}
	if reason, ok := unsupportedDtype[id]; ok {
		return DtypeDescriptor{}, fmt.Errorf("%w: Dtype %d: %s", ErrUnsupportedFormat, id, reason)
	}
	return DtypeDescriptor{}, fmt.Errorf("%w: Dtype %d", ErrUnknownFormat, id)
}

// FromFormat refuses coarse/ambiguous identities rather than choosing the first
// layout sharing a code width. Format is not a storage-layout bijection.
func FromFormat(f Format) (DtypeDescriptor, error) {
	if !f.Known() {
		return DtypeDescriptor{}, fmt.Errorf("%w: Format %q", ErrUnknownFormat, f)
	}
	var found DtypeDescriptor
	n := 0
	for _, e := range dtypeRegistry {
		if e.Format == f {
			found = e
			n++
		}
	}
	if n > 1 {
		return DtypeDescriptor{}, fmt.Errorf("%w: Format %q has %d layouts", ErrAmbiguousFormat, f, n)
	}
	if reason, ok := unsupportedFormat[f]; ok {
		return DtypeDescriptor{}, fmt.Errorf("%w: Format %q: %s", ErrUnsupportedFormat, f, reason)
	}
	if n == 1 {
		return found, nil
	}
	return DtypeDescriptor{}, fmt.Errorf("%w: Format %q has no layout", ErrUnsupportedFormat, f)
}

// Validate checks registry consistency and mapped-or-explicitly-unsupported
// coverage. The source-bound enum test independently detects added constants,
// aliases and changed values in all three defining packages.
func Validate() error { return validateRegistry(dtypeRegistry) }

func validateRegistry(entries []DtypeDescriptor) error {
	tokens := map[string]bool{}
	gg := map[uint32]bool{}
	dt := map[uint8]bool{}
	formats := map[Format]bool{}
	knownGG := map[uint32]bool{}
	for _, v := range ggufEnum {
		knownGG[v] = true
	}
	knownDT := map[uint8]bool{}
	for _, v := range dtypeEnum {
		knownDT[v] = true
	}
	for _, e := range entries {
		if e.GGUF < -1 || e.Dtype < -1 || (e.GGUF < 0 && e.Dtype < 0) || e.Token == "" || tokens[e.Token] || e.Bits <= 0 || e.BlockSize <= 0 || e.BlockBytes <= 0 || e.BytesPerElement != float64(e.BlockBytes)/float64(e.BlockSize) {
			return fmt.Errorf("invalid descriptor %q", e.Token)
		}
		tokens[e.Token] = true
		if e.GGUF >= 0 {
			if e.GGUF > 1<<32-1 || !knownGG[uint32(e.GGUF)] || gg[uint32(e.GGUF)] || unsupportedGGUF[uint32(e.GGUF)] != "" {
				return fmt.Errorf("invalid GGUF binding %q", e.Token)
			}
			gg[uint32(e.GGUF)] = true
		}
		if e.Dtype >= 0 {
			if e.Dtype > 255 || !knownDT[uint8(e.Dtype)] || dt[uint8(e.Dtype)] || unsupportedDtype[uint8(e.Dtype)] != "" {
				return fmt.Errorf("invalid Dtype binding %q", e.Token)
			}
			dt[uint8(e.Dtype)] = true
		}
		if e.Format != "" {
			if !e.Format.Known() {
				return fmt.Errorf("unknown Format %q", e.Format)
			}
			formats[e.Format] = true
		}
	}
	for v, reason := range unsupportedGGUF {
		if !knownGG[v] || reason == "" {
			return fmt.Errorf("invalid GGUF unsupported %d", v)
		}
	}
	for v, reason := range unsupportedDtype {
		if !knownDT[v] || reason == "" {
			return fmt.Errorf("invalid Dtype unsupported %d", v)
		}
	}
	for f, reason := range unsupportedFormat {
		if !f.Known() || reason == "" {
			return fmt.Errorf("invalid Format unsupported %q", f)
		}
	}
	for name, v := range ggufEnum {
		if !gg[v] && unsupportedGGUF[v] == "" {
			return fmt.Errorf("unmapped GGUF %s", name)
		}
	}
	for name, v := range dtypeEnum {
		if !dt[v] && unsupportedDtype[v] == "" {
			return fmt.Errorf("unmapped Dtype %s", name)
		}
	}
	for f := range knownFormats {
		if !formats[f] && unsupportedFormat[f] == "" {
			return fmt.Errorf("unmapped Format %q", f)
		}
	}
	return nil
}
