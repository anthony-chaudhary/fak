// Package fp4meta defines neutral metadata and capability decisions for FP4 artifacts.
//
// Invariants: fp4 metadata contracts ensure deterministic quantization encoding and decoding bounds.
// Metadata parsing and capability adjudication are fail-closed: unknown schemas, unverified hardware,
// and malformed bit patterns are strictly rejected or abstained upon.
package fp4meta

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
)

// SchemaV1 is the canonical schema identifier for version 1 of the FP4 metadata contract.
const SchemaV1 = "fak.fp4meta/v1"

// Outcome represents the coarse adjudication result of an FP4 artifact evaluation.
type Outcome string

const (
	// OutcomeAccept indicates the descriptor is valid and supported by local capabilities.
	OutcomeAccept Outcome = "accept"
	// OutcomeDelegate indicates the descriptor requires an external or specialized runtime.
	OutcomeDelegate Outcome = "delegate"
	// OutcomeAbstain indicates the descriptor cannot be judged locally due to unknown schemas or unverified hardware.
	OutcomeAbstain Outcome = "abstain"
	// OutcomeRefuse indicates the descriptor is malformed or violates format invariants.
	OutcomeRefuse Outcome = "refuse"
)

// ReasonCode provides a fine-grained, closed-set explanation for an adjudication outcome.
type ReasonCode string

const (
	// ReasonSupported indicates full support for the metadata and capability envelope.
	ReasonSupported ReasonCode = "supported"
	// ReasonRuntimeDelegation indicates delegation to a runtime is required.
	ReasonRuntimeDelegation ReasonCode = "runtime_delegation_required"
	// ReasonUnknownSchema indicates an unrecognised schema version.
	ReasonUnknownSchema ReasonCode = "unknown_schema"
	// ReasonUnknownVariant indicates an unrecognised FP4 variant.
	ReasonUnknownVariant ReasonCode = "unknown_variant"
	// ReasonInvalidDescriptor indicates semantic or structural validation failure.
	ReasonInvalidDescriptor ReasonCode = "invalid_descriptor"
	// ReasonIncompatibleEncoding indicates the float encoding is not locally supported.
	ReasonIncompatibleEncoding ReasonCode = "incompatible_encoding"
	// ReasonIncompatibleScale indicates the scale encoding format is not locally supported.
	ReasonIncompatibleScale ReasonCode = "incompatible_scale"
	// ReasonIncompatibleAccumulator indicates the accumulator precision is not locally supported.
	ReasonIncompatibleAccumulator ReasonCode = "incompatible_accumulator"
	// ReasonHardwareUnverified indicates lack of a recognized witness for measured hardware.
	ReasonHardwareUnverified ReasonCode = "hardware_unverified"
)

// Variant identifies the specific FP4 quantization and microscaling specification.
type Variant string

const (
	// VariantE2M1 represents standard 4-bit float with 1 sign bit, 2 exponent bits, and 1 mantissa bit.
	VariantE2M1 Variant = "e2m1"
	// VariantNVFP4 represents NVIDIA NVFP4 with 16-element block scaling using E4M3.
	VariantNVFP4 Variant = "nvfp4"
	// VariantMXFP4 represents OCP MXFP4 microscaling with 32-element blocks using UE8M0.
	VariantMXFP4 Variant = "mxfp4"
)

// ScaleEncoding identifies the numerical representation used for block or tensor scales.
type ScaleEncoding string

const (
	// ScaleNone indicates no scaling factor is applied.
	ScaleNone ScaleEncoding = "none"
	// ScaleE4M3 indicates 8-bit FP format with 4 exponent bits and 3 mantissa bits.
	ScaleE4M3 ScaleEncoding = "e4m3"
	// ScaleUE8M0 indicates 8-bit unsigned exponent-only power-of-two scaling.
	ScaleUE8M0 ScaleEncoding = "ue8m0"
	// ScaleBinary32 indicates IEEE 754 single-precision 32-bit floating point.
	ScaleBinary32 ScaleEncoding = "binary32"
)

// ScaleScope defines the granularity at which a scale factor applies.
type ScaleScope string

const (
	// ScalePerTensor indicates a single scale factor applies to the entire tensor.
	ScalePerTensor ScaleScope = "per_tensor"
	// ScalePerBlock indicates scale factors are shared across small fixed-size blocks.
	ScalePerBlock ScaleScope = "per_block"
)

// Accumulator specifies the target precision for intermediate dot-product accumulations.
type Accumulator string

const (
	// AccumulatorFP16 specifies half-precision 16-bit floating point accumulation.
	AccumulatorFP16 Accumulator = "fp16"
	// AccumulatorFP32 specifies single-precision 32-bit floating point accumulation.
	AccumulatorFP32 Accumulator = "fp32"
)

// FloatEncoding describes stored element bits independently of a named recipe.
type FloatEncoding struct {
	Bits         int  `json:"bits"`
	SignBits     int  `json:"sign_bits"`
	ExponentBits int  `json:"exponent_bits"`
	MantissaBits int  `json:"mantissa_bits"`
	ExponentBias int  `json:"exponent_bias"`
	FiniteOnly   bool `json:"finite_only"`
}

// BlockScale records how one scale is shared and encoded. ExponentOnly makes
// power-of-two scale semantics explicit instead of treating UE8M0 as a float.
type BlockScale struct {
	Scope        ScaleScope    `json:"scope"`
	BlockSize    int           `json:"block_size,omitempty"`
	Encoding     ScaleEncoding `json:"encoding"`
	ExponentBits int           `json:"exponent_bits,omitempty"`
	ExponentBias int           `json:"exponent_bias,omitempty"`
	ExponentOnly bool          `json:"exponent_only,omitempty"`
}

// Artifact records container and packaging metadata for the quantized weights.
type Artifact struct {
	Format  string `json:"format"`
	Version string `json:"version"`
	Digest  string `json:"digest,omitempty"`
}

// Recipe identifies the quantization pipeline or transformation recipe used.
type Recipe struct {
	ID      string `json:"id"`
	Version string `json:"version"`
}

// Runtime identifies the execution engine required when delegation is needed.
type Runtime struct {
	ID      string `json:"id"`
	Version string `json:"version"`
}

// HardwareEnvelope records target device vendor, architecture, and verification witness.
type HardwareEnvelope struct {
	Vendor       string `json:"vendor"`
	Architecture string `json:"architecture"`
	Measured     bool   `json:"measured"`
	Witness      string `json:"witness,omitempty"`
}

// Descriptor represents the complete self-describing metadata of an FP4 artifact.
type Descriptor struct {
	Schema      string           `json:"schema"`
	Variant     Variant          `json:"variant"`
	Encoding    FloatEncoding    `json:"encoding"`
	Scale       BlockScale       `json:"scale"`
	Accumulator Accumulator      `json:"accumulator"`
	Artifact    Artifact         `json:"artifact"`
	Recipe      Recipe           `json:"recipe"`
	Runtime     *Runtime         `json:"runtime,omitempty"`
	Hardware    HardwareEnvelope `json:"hardware"`
}

// Capabilities declares the local host or engine feature set against which descriptors are evaluated.
type Capabilities struct {
	Variants     []Variant       `json:"variants"`
	ScaleFormats []ScaleEncoding `json:"scale_formats"`
	Accumulators []Accumulator   `json:"accumulators"`
	Hardware     []string        `json:"hardware,omitempty"`
	Runtime      bool            `json:"runtime"`
}

// Result contains the adjudication outcome, reason code, and diagnostic detail.
type Result struct {
	Outcome Outcome    `json:"outcome"`
	Reason  ReasonCode `json:"reason"`
	Detail  string     `json:"detail"`
}

// Parse strictly decodes a descriptor and returns typed decisions for valid JSON,
// including descriptors from unknown schema versions or variants.
func Parse(raw []byte, capabilities Capabilities) (Descriptor, Result, error) {
	var descriptor Descriptor
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&descriptor); err != nil {
		return Descriptor{}, Result{Outcome: OutcomeRefuse, Reason: ReasonInvalidDescriptor, Detail: err.Error()}, err
	}
	var extra any
	if err := decoder.Decode(&extra); err != io.EOF {
		if err == nil {
			err = errors.New("multiple JSON values")
		}
		return Descriptor{}, Result{Outcome: OutcomeRefuse, Reason: ReasonInvalidDescriptor, Detail: err.Error()}, err
	}
	return descriptor, Adjudicate(descriptor, capabilities), nil
}

// Adjudicate separates metadata recognition from runtime and hardware support.
func Adjudicate(d Descriptor, c Capabilities) Result {
	if d.Schema != SchemaV1 {
		return result(OutcomeAbstain, ReasonUnknownSchema, fmt.Sprintf("schema %q is not recognized", d.Schema))
	}
	if !knownVariant(d.Variant) {
		return result(OutcomeAbstain, ReasonUnknownVariant, fmt.Sprintf("variant %q is not recognized", d.Variant))
	}
	if err := validate(d); err != nil {
		return result(OutcomeRefuse, ReasonInvalidDescriptor, err.Error())
	}
	if !contains(c.Variants, d.Variant) {
		return result(OutcomeDelegate, ReasonIncompatibleEncoding, fmt.Sprintf("variant %q is not locally supported", d.Variant))
	}
	if !contains(c.ScaleFormats, d.Scale.Encoding) {
		return result(OutcomeDelegate, ReasonIncompatibleScale, fmt.Sprintf("scale encoding %q is not locally supported", d.Scale.Encoding))
	}
	if !contains(c.Accumulators, d.Accumulator) {
		return result(OutcomeDelegate, ReasonIncompatibleAccumulator, fmt.Sprintf("accumulator %q is not locally supported", d.Accumulator))
	}
	if d.Runtime != nil && !c.Runtime {
		return result(OutcomeDelegate, ReasonRuntimeDelegation, fmt.Sprintf("runtime %q must execute the recipe", d.Runtime.ID))
	}
	if d.Hardware.Measured {
		key := d.Hardware.Vendor + "/" + d.Hardware.Architecture
		if d.Hardware.Witness == "" || !contains(c.Hardware, key) {
			return result(OutcomeAbstain, ReasonHardwareUnverified, fmt.Sprintf("measured hardware envelope %q lacks a recognized witness", key))
		}
	}
	return result(OutcomeAccept, ReasonSupported, "metadata and declared capability envelope are supported")
}

// DecodeE2M1 decodes one low nibble using finite E2M1 semantics. The high nibble
// is rejected so callers cannot accidentally pass a packed byte without selecting an element.
func DecodeE2M1(bits byte) (float64, error) {
	if bits > 0x0f {
		return 0, fmt.Errorf("E2M1 value 0x%02x exceeds four bits", bits)
	}
	sign := 1.0
	if bits&0x8 != 0 {
		sign = -1
	}
	exponent := (bits >> 1) & 0x3
	mantissa := bits & 0x1
	if exponent == 0 {
		if mantissa == 0 {
			return math.Copysign(0, sign), nil
		}
		return sign * 0.5, nil
	}
	return sign * math.Ldexp(1+float64(mantissa)/2, int(exponent)-1), nil
}

// MarshalCanonical serializes a Descriptor to formatted JSON adhering to the canonical schema indentation.
func MarshalCanonical(d Descriptor) ([]byte, error) { return json.MarshalIndent(d, "", "  ") }

func validate(d Descriptor) error {
	if d.Artifact.Format == "" || d.Artifact.Version == "" || d.Recipe.ID == "" || d.Recipe.Version == "" {
		return errors.New("artifact format/version and recipe id/version are required")
	}
	if d.Encoding != (FloatEncoding{Bits: 4, SignBits: 1, ExponentBits: 2, MantissaBits: 1, ExponentBias: 1, FiniteOnly: true}) {
		return errors.New("encoding must be finite E2M1 (4 bits, sign 1, exponent 2, mantissa 1, bias 1)")
	}
	if d.Accumulator != AccumulatorFP16 && d.Accumulator != AccumulatorFP32 {
		return fmt.Errorf("accumulator %q is invalid", d.Accumulator)
	}
	switch d.Variant {
	case VariantE2M1:
		if d.Scale.Scope != ScalePerTensor || d.Scale.BlockSize != 0 || d.Scale.Encoding != ScaleBinary32 || d.Scale.ExponentOnly {
			return errors.New("e2m1 requires a per-tensor binary32 scale")
		}
	case VariantNVFP4:
		if d.Scale.Scope != ScalePerBlock || d.Scale.BlockSize != 16 || d.Scale.Encoding != ScaleE4M3 || d.Scale.ExponentBits != 4 || d.Scale.ExponentBias != 7 || d.Scale.ExponentOnly {
			return errors.New("nvfp4 requires 16-element blocks with E4M3 scales (4 exponent bits, bias 7)")
		}
	case VariantMXFP4:
		if d.Scale.Scope != ScalePerBlock || d.Scale.BlockSize != 32 || d.Scale.Encoding != ScaleUE8M0 || d.Scale.ExponentBits != 8 || d.Scale.ExponentBias != 127 || !d.Scale.ExponentOnly {
			return errors.New("mxfp4 requires 32-element blocks with exponent-only UE8M0 scales (8 exponent bits, bias 127)")
		}
	}
	if d.Hardware.Measured && (d.Hardware.Vendor == "" || d.Hardware.Architecture == "") {
		return errors.New("measured hardware requires vendor and architecture")
	}
	if d.Runtime != nil && (d.Runtime.ID == "" || d.Runtime.Version == "") {
		return errors.New("runtime id and version are required when runtime is declared")
	}
	return nil
}

func knownVariant(v Variant) bool { return v == VariantE2M1 || v == VariantNVFP4 || v == VariantMXFP4 }

func contains[T comparable](values []T, target T) bool {
	for _, value := range values {
		if value == target {
			return true
		}
	}
	return false
}

func result(outcome Outcome, reason ReasonCode, detail string) Result {
	return Result{Outcome: outcome, Reason: reason, Detail: detail}
}
