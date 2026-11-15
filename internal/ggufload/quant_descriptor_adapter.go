package ggufload

// quant_descriptor_adapter.go ports the vLLM W4AFP8 checkpoint adapter
// (vllm/model_executor/layers/quantization/w4afp8.py:1@05737d01eb8d74b52c8478c6885dfb603808f335,
// Apache-2.0) into the fak weight loader as a pure host-side, loader-component
// translator.
//
// SGLang-style W4AFP8 MoE checkpoints serialize their routed experts as
// INT4 group-128 weights with FP8 (per-token, dynamic) activations. The
// canonical runtime path is the compressed-tensors W4A8 FP8 MoE method, which
// expects a different checkpoint *spelling*: the same numeric precision, but a
// different scheme identity and a different routed-expert tensor layout
// (weight -> weight_packed, weight_scale_inv -> weight_scale, and the static
// input_scale dropped because the W4A8 kernel quantizes activations per token).
//
// This file is that load-time translation: it rewrites foreign quant metadata
// and routed-expert tensor names onto the canonical scheme BEFORE weight
// binding, so one runtime kernel set serves every checkpoint spelling of the
// quantization class. It is metadata-only: no new kernels, no GPU, no network,
// and no model weights are read or written.
//
// Fail-closed contract: any spelling that is not the known W4AFP8 foreign
// descriptor, or any routed-expert tensor name that is not one of the three
// remapped forms, is refused with the ErrUnknownQuantSpelling sentinel rather
// than silently bound to the wrong kernel.

import (
	"errors"
	"fmt"
	"strings"
)

// ErrUnknownQuantSpelling is the fail-closed sentinel returned when a foreign
// quant descriptor or routed-expert tensor name does not match the known
// W4AFP8 spelling the adapter translates. Callers must treat it as a hard
// refusal: the checkpoint must not be bound to the canonical W4A8 FP8 MoE
// kernel on an unrecognized spelling.
var ErrUnknownQuantSpelling = errors.New("gguf: quant descriptor: unknown spelling")

// Foreign and canonical scheme identities. The two spellings carry identical
// numeric precision (INT4 group-128 weights, FP8 per-token dynamic
// activations); what differs is the scheme identity and the routed-expert
// tensor layout, which is exactly what the adapter rewrites.
const (
	// foreignW4AFP8Scheme is the SGLang-style foreign scheme identity.
	foreignW4AFP8Scheme = "w4afp8"

	// canonicalW4A8Fp8MoEScheme is the canonical compressed-tensors W4A8 FP8
	// MoE scheme identity the foreign descriptor is translated onto.
	canonicalW4A8Fp8MoEScheme = "w4a8-fp8-moe"

	// w4afp8GroupSize is the fixed INT4 quantization group size the foreign
	// spelling carries.
	w4afp8GroupSize = 128
)

// ForeignQuantDescriptor is the load-time quant metadata of a SGLang-style
// W4AFP8 MoE checkpoint, exactly as the foreign spelling records it.
type ForeignQuantDescriptor struct {
	// Scheme is the foreign scheme identity (must be "w4afp8").
	Scheme string
	// WeightBits is the weight precision in bits (must be 4).
	WeightBits int
	// WeightType is the weight numeric type (must be "int").
	WeightType string
	// Symmetric reports whether the weight quantization is symmetric.
	Symmetric bool
	// Strategy is the weight quantization strategy (must be "group").
	Strategy string
	// GroupSize is the weight quantization group size (must be 128).
	GroupSize int
	// ActBits is the activation precision in bits (must be 8).
	ActBits int
	// ActType is the activation numeric type (must be "float").
	ActType string
	// ActStrategy is the activation quantization strategy (must be "token").
	ActStrategy string
	// ActDynamic reports whether activations are quantized dynamically per
	// token (must be true).
	ActDynamic bool
}

// CanonicalQuantDescriptor is the quant metadata of the canonical W4A8 FP8 MoE
// scheme, the target of the translation. It is the spelling the runtime
// kernel set binds against.
type CanonicalQuantDescriptor struct {
	// Scheme is the canonical scheme identity ("w4a8-fp8-moe").
	Scheme string
	// WeightBits is the weight precision in bits.
	WeightBits int
	// WeightType is the weight numeric type.
	WeightType string
	// Symmetric reports whether the weight quantization is symmetric.
	Symmetric bool
	// Strategy is the weight quantization strategy.
	Strategy string
	// GroupSize is the weight quantization group size.
	GroupSize int
	// ActBits is the activation precision in bits.
	ActBits int
	// ActType is the activation numeric type.
	ActType string
	// ActStrategy is the activation quantization strategy.
	ActStrategy string
	// ActDynamic reports whether activations are quantized dynamically per
	// token.
	ActDynamic bool
}

// isKnownForeignW4AFP8 reports whether d is exactly the known W4AFP8 foreign
// spelling. Every field is checked so a single drifted field (a different
// group size, an asymmetric weight, a static activation) is refused rather
// than translated.
func isKnownForeignW4AFP8(d ForeignQuantDescriptor) bool {
	return d.Scheme == foreignW4AFP8Scheme &&
		d.WeightBits == 4 &&
		d.WeightType == "int" &&
		d.Symmetric &&
		d.Strategy == "group" &&
		d.GroupSize == w4afp8GroupSize &&
		d.ActBits == 8 &&
		d.ActType == "float" &&
		d.ActStrategy == "token" &&
		d.ActDynamic
}

// TranslateQuantDescriptor rewrites a foreign W4AFP8 checkpoint descriptor
// onto the canonical W4A8 FP8 MoE scheme. It is the metadata half of the
// adapter: the numeric precision is preserved, the scheme identity is
// rewritten to the canonical one, and the result is what the runtime kernel
// set binds against.
//
// A descriptor that is not the known W4AFP8 foreign spelling is refused with
// ErrUnknownQuantSpelling (wrapped, so callers can use errors.Is). The
// returned descriptor is the zero value on refusal.
func TranslateQuantDescriptor(d ForeignQuantDescriptor) (CanonicalQuantDescriptor, error) {
	if !isKnownForeignW4AFP8(d) {
		return CanonicalQuantDescriptor{}, fmt.Errorf("%w: scheme=%q weight=%dbits/%s/%s/group=%d act=%dbits/%s/%s/dynamic=%v",
			ErrUnknownQuantSpelling, d.Scheme, d.WeightBits, d.WeightType, d.Strategy, d.GroupSize,
			d.ActBits, d.ActType, d.ActStrategy, d.ActDynamic)
	}
	return CanonicalQuantDescriptor{
		Scheme:      canonicalW4A8Fp8MoEScheme,
		WeightBits:  4,
		WeightType:  "int",
		Symmetric:   true,
		Strategy:    "group",
		GroupSize:   w4afp8GroupSize,
		ActBits:     8,
		ActType:     "float",
		ActStrategy: "token",
		ActDynamic:  true,
	}, nil
}

// MapQuantTensorName translates a foreign W4AFP8 routed-expert tensor name
// onto the canonical W4A8 FP8 MoE layout. It is the name half of the adapter
// and mirrors the upstream checkpoint weight mapper:
//
//   - "<expert>.weight"          -> "<expert>.weight_packed"
//   - "<expert>.weight_scale_inv" -> "<expert>.weight_scale"
//   - "<expert>.input_scale"      -> dropped (ok=false): the W4A8 kernel
//     quantizes activations per token, so the static input scale is not bound.
//
// The caller applies this only to routed-expert tensors; dense layers already
// use the native FP8 serialization and are not remapped. A name that is not
// one of the three remapped forms is refused with ErrUnknownQuantSpelling.
//
// The returned ok is true when the name maps to a canonical tensor, false when
// the name is dropped (the input scale). On refusal the returned name is
// empty and ok is false.
func MapQuantTensorName(name string) (string, bool, error) {
	// The static input scale is dropped, not remapped.
	if strings.HasSuffix(name, ".input_scale") {
		return "", false, nil
	}
	if rest, found := strings.CutSuffix(name, ".weight_scale_inv"); found {
		return rest + ".weight_scale", true, nil
	}
	if rest, found := strings.CutSuffix(name, ".weight"); found {
		return rest + ".weight_packed", true, nil
	}
	return "", false, fmt.Errorf("%w: tensor %q", ErrUnknownQuantSpelling, name)
}

// RepackSignedInt4ToUint4B8 applies the W4AFP8 load-time byte transform to a
// routed-expert weight payload: each int8 byte (one signed INT4 per byte) is
// XORed with 0x88. This mirrors the upstream
// `tensor.view(torch.uint8).bitwise_xor(0x88).view(torch.int32)` repack, whose
// net byte effect is the per-byte XOR; the int32 view only reinterprets the
// result, so the byte count is unchanged.
//
// The input length must be a multiple of 4 (the upstream transform views the
// result as int32). A mis-sized payload is refused rather than partially
// repacked. The input slice is not mutated.
func RepackSignedInt4ToUint4B8(raw []byte) ([]byte, error) {
	if len(raw)%4 != 0 {
		return nil, fmt.Errorf("gguf: quant descriptor repack: payload %d bytes is not a multiple of 4", len(raw))
	}
	out := make([]byte, len(raw))
	for i, b := range raw {
		out[i] = b ^ 0x88
	}
	return out, nil
}
