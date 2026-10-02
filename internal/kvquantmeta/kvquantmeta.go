package kvquantmeta

import (
	"fmt"
	"strings"
)

// Precision names an independently declared K- or V-cache storage scheme.
type Precision string

const (
	// PrecisionFP16 stores 16-bit IEEE floating point values in cache buffers.
	PrecisionFP16 Precision = "fp16"
	// PrecisionBF16 stores 16-bit Brain floating point values with wide dynamic range.
	PrecisionBF16 Precision = "bf16"
	// PrecisionFP8 stores 8-bit floating point values for compact tensor representations.
	PrecisionFP8 Precision = "fp8"
	// PrecisionINT8 stores 8-bit signed integer quantized representations.
	PrecisionINT8 Precision = "int8"
	// PrecisionINT4 stores 4-bit packed integer quantized representations.
	PrecisionINT4 Precision = "int4"
	// PrecisionINT2 stores 2-bit aggressive integer quantized representations.
	PrecisionINT2 Precision = "int2"
)

// Grouping describes where cache scales are shared.
type Grouping string

const (
	// GroupingPerToken normalizes quantization parameters per sequence position.
	GroupingPerToken Grouping = "per-token"
	// GroupingPerChannel normalizes quantization parameters per feature channel.
	GroupingPerChannel Grouping = "per-channel"
	// GroupingPerTokenChannel normalizes quantization parameters across both tokens and channels.
	GroupingPerTokenChannel Grouping = "per-token-channel"
)

// Recoverability states whether a lower tier can return to its source tier.
type Recoverability string

const (
	// RecoverableExact guarantees lossless reconstruction back to the baseline tier.
	RecoverableExact Recoverability = "exact"
	// RecoverableApproximate permits bounded loss reconstruction back to the baseline tier.
	RecoverableApproximate Recoverability = "approximate"
	// RecoverableNone marks one-way irreversible quantization without an inversion path.
	RecoverableNone Recoverability = "none"
)

// ReasonCode is stable machine-readable adjudication detail.
type ReasonCode string

const (
	// ReasonSupported marks valid descriptors and allowed tier transitions.
	ReasonSupported ReasonCode = "KVQUANT_SUPPORTED"
	// ReasonUnknownScheme marks unsupported schemes, precisions, groupings, or transforms.
	ReasonUnknownScheme ReasonCode = "KVQUANT_UNKNOWN_SCHEME"
	// ReasonInvalidDescriptor marks missing or inconsistent descriptor configuration fields.
	ReasonInvalidDescriptor ReasonCode = "KVQUANT_INVALID_DESCRIPTOR"
	// ReasonUnsupportedTransition marks disallowed directed tier movement edges.
	ReasonUnsupportedTransition ReasonCode = "KVQUANT_UNSUPPORTED_TRANSITION"
)

// Adapted from vllm-project/vllm-metal PR #891, commit 77dd9480 (Apache-2.0):
// a constant (zero-variance) cache block has max == min, so its block scale is
// 0, the quantize step divides by 0, and dequantization returns NaN instead of
// the constant. Fak expresses the upstream property as a scalar helper rather
// than copying the upstream quantization implementation.
const minScale float32 = 1e-8

// FloorScale prevents a zero-variance cache block from producing a zero scale,
// keeping the quantize divisor nonzero so the block stays finite and exactly
// recoverable. The floor is a sane epsilon, not float32's smallest normal
// (1.1754944e-38): that would leave q = (x-min)/scale large enough to overflow
// float32 once x-min exceeds a few units. Scales already above the floor are
// returned unchanged, so the helper cannot alter a well-conditioned block.
func FloorScale(scale float32) float32 {
	if scale < minScale {
		return minScale
	}
	return scale
}

// Descriptor is a runtime-neutral KV-cache quantization contract. K and V are
// separate on purpose and cannot be confused with model-weight precision.
type Descriptor struct {
	ID                   string         `json:"id"`
	Version              string         `json:"version"`
	KeyPrecision         Precision      `json:"key_precision"`
	ValuePrecision       Precision      `json:"value_precision"`
	Grouping             Grouping       `json:"grouping"`
	GroupSize            int            `json:"group_size,omitempty"`
	ResidualWindowTokens int            `json:"residual_window_tokens,omitempty"`
	Transform            string         `json:"transform,omitempty"`
	Tier                 string         `json:"tier"`
	Recoverability       Recoverability `json:"recoverability"`
}

// Transition declares an explicit cache-tier change.
type Transition struct {
	From Descriptor `json:"from"`
	To   Descriptor `json:"to"`
}

// Support is the caller-declared runtime envelope.
type Support struct {
	Schemes     map[string][]string
	Precisions  []Precision
	Groupings   []Grouping
	Transforms  []string
	Transitions map[string][]string
}

// Result is returned for every descriptor and transition decision.
type Result struct {
	Supported bool       `json:"supported"`
	Reason    ReasonCode `json:"reason"`
	Detail    string     `json:"detail,omitempty"`
}

// Validate checks one descriptor without silently selecting a fallback scheme.
func Validate(d Descriptor, s Support) Result {
	if field := missing(d); field != "" {
		return Result{Reason: ReasonInvalidDescriptor, Detail: field}
	}
	versions, ok := s.Schemes[d.ID]
	if !ok || !containsString(versions, d.Version) {
		return Result{Reason: ReasonUnknownScheme, Detail: d.ID + "@" + d.Version}
	}
	if !containsPrecision(s.Precisions, d.KeyPrecision) || !containsPrecision(s.Precisions, d.ValuePrecision) {
		return Result{Reason: ReasonUnknownScheme, Detail: fmt.Sprintf("K=%s,V=%s", d.KeyPrecision, d.ValuePrecision)}
	}
	if !containsGrouping(s.Groupings, d.Grouping) {
		return Result{Reason: ReasonUnknownScheme, Detail: string(d.Grouping)}
	}
	if d.Transform != "" && !containsString(s.Transforms, d.Transform) {
		return Result{Reason: ReasonUnknownScheme, Detail: d.Transform}
	}
	return Result{Supported: true, Reason: ReasonSupported}
}

// ValidateTransition checks both tiers and then requires a declared directed edge.
func ValidateTransition(t Transition, s Support) Result {
	if result := Validate(t.From, s); !result.Supported {
		return result
	}
	if result := Validate(t.To, s); !result.Supported {
		return result
	}
	if !containsString(s.Transitions[t.From.Tier], t.To.Tier) {
		return Result{Reason: ReasonUnsupportedTransition, Detail: t.From.Tier + "->" + t.To.Tier}
	}
	return Result{Supported: true, Reason: ReasonSupported}
}

func missing(d Descriptor) string {
	if strings.TrimSpace(d.ID) == "" {
		return "id"
	}
	if strings.TrimSpace(d.Version) == "" {
		return "version"
	}
	if d.KeyPrecision == "" {
		return "key_precision"
	}
	if d.ValuePrecision == "" {
		return "value_precision"
	}
	if d.Grouping == "" {
		return "grouping"
	}
	if d.Grouping != GroupingPerToken && d.GroupSize <= 0 {
		return "group_size"
	}
	if d.ResidualWindowTokens < 0 {
		return "residual_window_tokens"
	}
	if strings.TrimSpace(d.Tier) == "" {
		return "tier"
	}
	if d.Recoverability == "" {
		return "recoverability"
	}
	return ""
}

func containsString(values []string, target string) bool {
	for _, v := range values {
		if v == target {
			return true
		}
	}
	return false
}
func containsPrecision(values []Precision, target Precision) bool {
	for _, v := range values {
		if v == target {
			return true
		}
	}
	return false
}
func containsGrouping(values []Grouping, target Grouping) bool {
	for _, v := range values {
		if v == target {
			return true
		}
	}
	return false
}
