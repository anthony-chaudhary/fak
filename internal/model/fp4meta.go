package model

// fp4meta.go reads the self-describing metadata an FP4 / microscaling checkpoint ships with
// and answers ONE question: can this loader decode these bytes without guessing?
//
// It is a metadata contract, not a kernel and not a quantizer. Nothing here scores a format,
// converts an artifact, or claims a speedup. The four dispositions are kept distinct because
// collapsing them is how a loader ends up silently pretending:
//
//   - ACCEPT   — every field is readable, self-consistent, and names an envelope this build
//     can decode in-kernel.
//   - DELEGATE — readable and consistent, but execution belongs to someone else (the producer
//     said so, or the declared hardware has no native FP4 decode/GEMM). fak routes, not claims.
//   - ABSTAIN  — fak cannot READ it: an unknown schema version, or a vocabulary word (element
//     format, scale encoding, accumulator) outside this build. Not an error — the artifact may
//     be perfectly valid and merely newer than this binary, and treating "new" as "broken"
//     would make the loader a brake on the ecosystem.
//   - REFUSE   — fak CAN read it and it is wrong: the document contradicts itself
//     (FP4_MALFORMED), or the tuple contradicts the fixed definition of the format it names
//     (FP4_UNSUPPORTED_COMBINATION). No future version makes "mxfp4 with 16-element blocks"
//     mean something.
//
// The abstain/refuse split is the load-bearing one, and it is the same split
// internal/quantmeta draws for the neutral quantization descriptor.
//
// The fixed definitions this file adjudicates against are the published ones, not fak's
// preferences: NVFP4 is E2M1 elements in 16-element blocks with an E4M3 per-block scale plus a
// per-tensor scale (two scale levels); MXFP4 (OCP microscaling) is E2M1 elements in 32-element
// blocks with a single E8M0 (power-of-two) per-block scale. Both accumulate in FP32 — the same
// stance internal/fp4runtime's compatibility matrix takes, where an fp16-accumulate profile is
// a refusal, because summing 4-bit products in fp16 throws away the block dynamic range the
// shared scale was chosen to preserve.
//
// Sibling packages, and why this one is separate: internal/fp4runtime negotiates a pinned
// artifact/runtime/GPU/accumulator REQUEST against an external compatibility MATRIX (delegating
// to TensorRT-LLM and friends); this file adjudicates a single self-describing metadata blob at
// the point the model loader has the bytes in hand and no matrix.

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
)

const FP4MetadataSchema = "fp4meta/v1"

// FP4Format names the serialization identity of the packed 4-bit payload. NVFP4 and MXFP4 are
// never aliases merely because both carry E2M1 elements — their block size and scale encoding
// differ, so a decoder that confused them would read the scales at the wrong stride.
type FP4Format string

const (
	FP4FormatE2M1    FP4Format = "e2m1"
	FP4FormatNVFP4   FP4Format = "nvfp4"
	FP4FormatMXFP4   FP4Format = "mxfp4"
	FP4FormatROCmFP4 FP4Format = "rocmfp4"
)

// FP4ScaleEncoding is the numeric format of the per-block shared scale. E8M0 is a pure
// power-of-two exponent (the OCP microscaling scale); E4M3 is an fp8 scale with a mantissa;
// FP16 is an IEEE binary16 float scale (used by ROCmFP4_FAST).
type FP4ScaleEncoding string

const (
	FP4ScaleNone FP4ScaleEncoding = "none"
	FP4ScaleE4M3 FP4ScaleEncoding = "e4m3"
	FP4ScaleE8M0 FP4ScaleEncoding = "e8m0"
	FP4ScaleFP16 FP4ScaleEncoding = "fp16"
)

// FP4Accumulator is the type the GEMM accumulates products in — not the storage type of the
// weights, and not the output type.
type FP4Accumulator string

const (
	FP4AccumulatorFP16 FP4Accumulator = "fp16"
	FP4AccumulatorBF16 FP4Accumulator = "bf16"
	FP4AccumulatorFP32 FP4Accumulator = "fp32"
)

// FP4ClaimScope names what kind of user-facing claim the metadata licenses. The scopes stay
// separate so an artifact's mere existence is never reported as a measured hardware result.
type FP4ClaimScope string

const (
	FP4ClaimArtifact         FP4ClaimScope = "artifact"
	FP4ClaimRecipe           FP4ClaimScope = "recipe"
	FP4ClaimRuntimeDelegated FP4ClaimScope = "runtime_delegated"
	FP4ClaimMeasuredHardware FP4ClaimScope = "measured_hardware"
)

// FP4BlockScale describes the shared scale: how many elements it covers, how it is encoded,
// and how many scale LEVELS the format applies. Levels is 1 for a single per-block scale
// (MXFP4) and 2 for a per-block scale under a second per-tensor scale (NVFP4); it is 0 exactly
// when there is no scale at all.
type FP4BlockScale struct {
	Elements uint32           `json:"elements"`
	Encoding FP4ScaleEncoding `json:"encoding"`
	Levels   uint8            `json:"levels"`
}

// FP4Exponent is the exponent field of one 4-bit ELEMENT (not of the scale).
type FP4Exponent struct {
	Bits uint8 `json:"bits"`
	Bias int8  `json:"bias"`
}

// FP4HardwareCapability is what the producer says the target can do natively. NativeDecode and
// NativeGEMM are separate because a device can unpack FP4 into a wider type without having an
// FP4 tensor-core GEMM, and only the pair licenses in-kernel execution.
type FP4HardwareCapability struct {
	Runtime      string `json:"runtime,omitempty"`
	Accelerator  string `json:"accelerator,omitempty"`
	NativeDecode bool   `json:"native_decode"`
	NativeGEMM   bool   `json:"native_gemm"`
}

// FP4Metadata is the whole self-describing document.
type FP4Metadata struct {
	Schema      string                `json:"schema"`
	Format      FP4Format             `json:"format"`
	Encoding    string                `json:"encoding"`
	BlockScale  FP4BlockScale         `json:"block_scale"`
	Exponent    FP4Exponent           `json:"exponent"`
	Accumulator FP4Accumulator        `json:"accumulator"`
	Hardware    FP4HardwareCapability `json:"hardware"`
	ClaimScope  FP4ClaimScope         `json:"claim_scope"`
}

// FP4Disposition is the typed verdict. There is no fifth, implicit "assume it is fine".
type FP4Disposition string

const (
	FP4Accept   FP4Disposition = "accept"
	FP4Abstain  FP4Disposition = "abstain"
	FP4Refuse   FP4Disposition = "refuse"
	FP4Delegate FP4Disposition = "delegate"
)

// FP4Reason is the stable machine-readable reason code carried by every verdict, so a caller
// never has to parse the prose in Detail.
type FP4Reason string

const (
	FP4ReasonSupported              FP4Reason = "FP4_SUPPORTED"
	FP4ReasonUnknownSchema          FP4Reason = "FP4_UNKNOWN_SCHEMA"
	FP4ReasonUnknownFormat          FP4Reason = "FP4_UNKNOWN_FORMAT"
	FP4ReasonUnsupportedCombination FP4Reason = "FP4_UNSUPPORTED_COMBINATION"
	FP4ReasonRuntimeDelegation      FP4Reason = "FP4_RUNTIME_DELEGATION_REQUIRED"
	FP4ReasonMalformed              FP4Reason = "FP4_MALFORMED"
)

// FP4Result is the adjudication of one document. Metadata is echoed back only on the paths
// where the document was fully readable and coherent (accept and delegate) — an abstain or a
// refusal has nothing trustworthy to hand on.
type FP4Result struct {
	Disposition FP4Disposition `json:"disposition"`
	Reason      FP4Reason      `json:"reason"`
	Metadata    *FP4Metadata   `json:"metadata,omitempty"`
	Detail      string         `json:"detail,omitempty"`
}

// ParseFP4Metadata reads one FP4 metadata document and adjudicates it. Parsing is strict: an
// unknown field, a truncated document, or trailing JSON is a refusal rather than a silently
// partial read, because a field this build drops on the floor is exactly the field that
// changes how the payload must be decoded.
func ParseFP4Metadata(data []byte) FP4Result {
	var m FP4Metadata
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&m); err != nil {
		return FP4Result{Disposition: FP4Refuse, Reason: FP4ReasonMalformed, Detail: err.Error()}
	}
	if err := ensureJSONEOF(dec); err != nil {
		return FP4Result{Disposition: FP4Refuse, Reason: FP4ReasonMalformed, Detail: err.Error()}
	}
	if m.Schema != FP4MetadataSchema {
		return FP4Result{Disposition: FP4Abstain, Reason: FP4ReasonUnknownSchema, Detail: fmt.Sprintf("unsupported schema %q", m.Schema)}
	}
	return AdjudicateFP4Metadata(m)
}

// AdjudicateFP4Metadata classifies an already-parsed document. It is a pure function of the
// metadata: no file name, producer, or tensor content changes the verdict for a document of
// the same shape, which is what keeps this a contract rather than a policy knob.
//
// The order below is deliberate — read it before narrowing it. Readability comes first
// (schema, then vocabulary), so an artifact from a newer build abstains instead of being
// refused on a rule this binary derived from a format it does not know. Self-consistency comes
// next, since a document that contradicts itself is wrong no matter which format it names.
// Only then do the per-format definitions and the execution envelope apply.
func AdjudicateFP4Metadata(m FP4Metadata) FP4Result {
	// Re-checked here and not only in ParseFP4Metadata: this is exported, and a caller that
	// hand-built the struct must not get a v1 reading of a v2 document.
	if m.Schema != FP4MetadataSchema {
		return fp4Verdict(FP4Abstain, FP4ReasonUnknownSchema, nil, fmt.Sprintf("unsupported schema %q", m.Schema))
	}

	// Unreadable vocabulary abstains. All three fields spell the same failure — this build has
	// no meaning for that word — so they share one reason code and separate details.
	if !m.Format.known() {
		return fp4Verdict(FP4Abstain, FP4ReasonUnknownFormat, nil, fmt.Sprintf("unknown fp4 format %q", m.Format))
	}
	if !m.BlockScale.Encoding.known() {
		return fp4Verdict(FP4Abstain, FP4ReasonUnknownFormat, nil, fmt.Sprintf("unknown block-scale encoding %q", m.BlockScale.Encoding))
	}
	if !m.Accumulator.known() {
		return fp4Verdict(FP4Abstain, FP4ReasonUnknownFormat, nil, fmt.Sprintf("unknown accumulator %q", m.Accumulator))
	}
	if !m.ClaimScope.known() {
		return fp4Verdict(FP4Abstain, FP4ReasonUnknownFormat, nil, fmt.Sprintf("unknown claim scope %q", m.ClaimScope))
	}

	if detail, ok := fp4SelfContradiction(m); !ok {
		return fp4Verdict(FP4Refuse, FP4ReasonMalformed, nil, detail)
	}
	if detail, ok := fp4FormatDefinition(m); !ok {
		return fp4Verdict(FP4Refuse, FP4ReasonUnsupportedCombination, nil, detail)
	}

	// The accumulator is judged against the format, not against taste: a shared block scale
	// exists to hold the block's dynamic range, and summing the products in fp16/bf16 discards
	// the range it bought. internal/fp4runtime refuses the same combination from its matrix.
	if m.Accumulator != FP4AccumulatorFP32 {
		return fp4Verdict(FP4Refuse, FP4ReasonUnsupportedCombination, nil,
			fmt.Sprintf("%s blocks accumulate in fp32, got %q: a 4-bit product summed in %s discards the dynamic range the shared scale was chosen to preserve", m.Format, m.Accumulator, m.Accumulator))
	}

	// A measured-hardware claim is the one scope that asserts something about a real device, so
	// it must name one. Without that, the document is claiming more than it carries.
	if m.ClaimScope == FP4ClaimMeasuredHardware && (m.Hardware.Accelerator == "" || m.Hardware.Runtime == "") {
		return fp4Verdict(FP4Refuse, FP4ReasonMalformed, nil,
			"claim scope measured_hardware requires hardware.accelerator and hardware.runtime to name the device and runtime that were measured")
	}

	// Execution ownership. The producer can hand it away outright, and a target without BOTH
	// native decode and a native FP4 GEMM cannot take it here either — in both cases fak routes
	// rather than claiming it can run the artifact in-kernel.
	if m.ClaimScope == FP4ClaimRuntimeDelegated {
		return fp4Verdict(FP4Delegate, FP4ReasonRuntimeDelegation, &m,
			fmt.Sprintf("metadata delegates execution to runtime %q", m.Hardware.Runtime))
	}
	if !m.Hardware.NativeDecode || !m.Hardware.NativeGEMM {
		return fp4Verdict(FP4Delegate, FP4ReasonRuntimeDelegation, &m,
			fmt.Sprintf("declared target (accelerator %q, runtime %q) reports native_decode=%t native_gemm=%t; execution belongs to a runtime that has both",
				m.Hardware.Accelerator, m.Hardware.Runtime, m.Hardware.NativeDecode, m.Hardware.NativeGEMM))
	}

	return fp4Verdict(FP4Accept, FP4ReasonSupported, &m,
		fmt.Sprintf("%s: %s elements in %d-element blocks, %s scale x%d, fp32 accumulate",
			m.Format, m.Encoding, m.BlockScale.Elements, m.BlockScale.Encoding, m.BlockScale.Levels))
}

// fp4SelfContradiction checks the document against ITSELF — no format definition is consulted,
// so every rule here holds for an FP4 format this build has never heard of. It returns the
// detail for the first contradiction found.
func fp4SelfContradiction(m FP4Metadata) (string, bool) {
	if m.Encoding == "" {
		return "element encoding is empty: the metadata does not say how the 4-bit payload is laid out", false
	}
	// A 4-bit element spends one bit on the sign, so the exponent field cannot reach 4 bits and
	// an exponent-less 4-bit float is not a float.
	if m.Exponent.Bits == 0 || m.Exponent.Bits > 3 {
		return fmt.Sprintf("exponent.bits=%d does not fit a 4-bit element (1 sign bit leaves at most 3)", m.Exponent.Bits), false
	}
	switch {
	case m.BlockScale.Encoding == FP4ScaleNone:
		// No scale means no block: an element count or a level count here describes a scale the
		// document just said does not exist.
		if m.BlockScale.Elements != 0 || m.BlockScale.Levels != 0 {
			return fmt.Sprintf("block_scale.encoding is %q but declares elements=%d levels=%d", FP4ScaleNone, m.BlockScale.Elements, m.BlockScale.Levels), false
		}
	default:
		if m.BlockScale.Elements == 0 {
			return fmt.Sprintf("block_scale.encoding %q covers 0 elements", m.BlockScale.Encoding), false
		}
		// Microscaling blocks are power-of-two so a block index is a shift, not a divide; a
		// non-power-of-two count means the decoder and the producer disagree on every stride.
		if m.BlockScale.Elements&(m.BlockScale.Elements-1) != 0 {
			return fmt.Sprintf("block_scale.elements=%d is not a power of two", m.BlockScale.Elements), false
		}
		if m.BlockScale.Levels == 0 {
			return fmt.Sprintf("block_scale.encoding %q declares 0 scale levels", m.BlockScale.Encoding), false
		}
	}
	return "", true
}

// fp4FormatDefinition holds a named format to its PUBLISHED definition. These are not
// preferences: an NVFP4 reader that accepted 32-element blocks would read every scale at the
// wrong stride and return plausible garbage.
func fp4FormatDefinition(m FP4Metadata) (string, bool) {
	// E2M1 is the element encoding under all three formats, and its exponent field is fixed:
	// 2 bits, bias 2^(2-1)-1 = 1.
	if m.Encoding == string(FP4FormatE2M1) {
		if m.Exponent.Bits != 2 || m.Exponent.Bias != 1 {
			return fmt.Sprintf("e2m1 elements have a 2-bit exponent with bias 1, got bits=%d bias=%d", m.Exponent.Bits, m.Exponent.Bias), false
		}
	}
	switch m.Format {
	case FP4FormatNVFP4:
		return fp4RequireBlock(m, FP4ScaleE4M3, 16, 2)
	case FP4FormatMXFP4:
		return fp4RequireBlock(m, FP4ScaleE8M0, 32, 1)
	case FP4FormatROCmFP4:
		return fp4RequireBlock(m, FP4ScaleFP16, 32, 1)
	case FP4FormatE2M1:
		// The bare element format pins no block geometry of its own; any coherent scale is valid.
		return "", true
	default:
		return fmt.Sprintf("unknown fp4 format %q", m.Format), false
	}
}

func fp4RequireBlock(m FP4Metadata, enc FP4ScaleEncoding, elements uint32, levels uint8) (string, bool) {
	if m.Encoding != string(FP4FormatE2M1) {
		return fmt.Sprintf("%s carries e2m1 elements, got encoding %q", m.Format, m.Encoding), false
	}
	if m.BlockScale.Encoding != enc || m.BlockScale.Elements != elements || m.BlockScale.Levels != levels {
		return fmt.Sprintf("%s is defined as a %s scale over %d-element blocks with %d scale level(s), got %s over %d with %d",
			m.Format, enc, elements, levels, m.BlockScale.Encoding, m.BlockScale.Elements, m.BlockScale.Levels), false
	}
	return "", true
}

func fp4Verdict(d FP4Disposition, r FP4Reason, m *FP4Metadata, detail string) FP4Result {
	return FP4Result{Disposition: d, Reason: r, Metadata: m, Detail: detail}
}

func (f FP4Format) known() bool {
	switch f {
	case FP4FormatE2M1, FP4FormatNVFP4, FP4FormatMXFP4, FP4FormatROCmFP4:
		return true
	}
	return false
}

func (e FP4ScaleEncoding) known() bool {
	switch e {
	case FP4ScaleNone, FP4ScaleE4M3, FP4ScaleE8M0, FP4ScaleFP16:
		return true
	}
	return false
}

// ROCmFP4MetadataPreset returns the self-describing metadata for an AMD RDNA 3/3.5
// ROCmFP4_FAST checkpoint: E2M1 elements in 32-element blocks (aligned to RDNA 3/3.5
// half-wave SIMD strides) with a single FP16 scale per block and FP32 accumulation.
func ROCmFP4MetadataPreset() FP4Metadata {
	return FP4Metadata{
		Schema:      FP4MetadataSchema,
		Format:      FP4FormatROCmFP4,
		Encoding:    "e2m1",
		BlockScale:  FP4BlockScale{Elements: 32, Encoding: FP4ScaleFP16, Levels: 1},
		Exponent:    FP4Exponent{Bits: 2, Bias: 1},
		Accumulator: FP4AccumulatorFP32,
		Hardware: FP4HardwareCapability{
			Runtime:      "rocm",
			Accelerator:  "gfx1151",
			NativeDecode: true,
			NativeGEMM:   true,
		},
		ClaimScope: FP4ClaimArtifact,
	}
}

func (a FP4Accumulator) known() bool {
	switch a {
	case FP4AccumulatorFP16, FP4AccumulatorBF16, FP4AccumulatorFP32:
		return true
	}
	return false
}

func (c FP4ClaimScope) known() bool {
	switch c {
	case FP4ClaimArtifact, FP4ClaimRecipe, FP4ClaimRuntimeDelegated, FP4ClaimMeasuredHardware:
		return true
	}
	return false
}

func ensureJSONEOF(dec *json.Decoder) error {
	var extra any
	if err := dec.Decode(&extra); err != io.EOF {
		if err == nil {
			return fmt.Errorf("multiple JSON values")
		}
		return err
	}
	return nil
}
