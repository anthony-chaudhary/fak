package vcachewarm

import "math"

// Provider names the provider surface whose cache semantics are being planned.
// The known constants cover the M3 rules; other string values are treated as an
// implicit-cache provider where a dedicated decode-1 warm is the only primitive.
type Provider string

const (
	ProviderAnthropic Provider = "anthropic"
	ProviderOpenAI    Provider = "openai"
	ProviderDeepSeek  Provider = "deepseek"
)

// ToolChoice is the request's tool_choice mode for the Anthropic max_tokens:0
// rejected-combination guard.
type ToolChoice string

const (
	ToolChoiceAuto ToolChoice = "auto"
	ToolChoiceNone ToolChoice = "none"
	ToolChoiceTool ToolChoice = "tool"
	ToolChoiceAny  ToolChoice = "any"
)

// Primitive is the warming path selected for a request.
type Primitive string

const (
	PrimitiveNone                Primitive = ""
	PrimitiveAnthropicMaxTokens0 Primitive = "anthropic_max_tokens_0"
	PrimitiveDecode1             Primitive = "decode_1"
	PrimitiveOrderFirstReal      Primitive = "order_first_real"
)

// ActiveCacheCapability is the caller's witness that this provider/engine can
// honor the active warming primitive the planner may select.
type ActiveCacheCapability string

const (
	ActiveCacheCapabilityUnknown     ActiveCacheCapability = ""
	ActiveCacheCapabilitySupported   ActiveCacheCapability = "supported"
	ActiveCacheCapabilityUnsupported ActiveCacheCapability = "unsupported"
)

// Reason explains why a primitive was selected or refused.
type Reason string

const (
	ReasonAnthropicExplicitWarm            Reason = "anthropic_explicit_warm"
	ReasonExplicitRejectedFallback         Reason = "explicit_rejected_fallback_decode_1"
	ReasonImplicitDecode1                  Reason = "implicit_decode_1"
	ReasonAutoCacheNaturalWarm             Reason = "auto_cache_order_first_real"
	ReasonUnsupportedActiveCacheCapability Reason = "unsupported_active_cache_capability"
	ReasonBelowBreakEven                   Reason = "below_break_even"
	ReasonPrefixFingerprintMismatch        Reason = "prefix_fingerprint_mismatch"
	ReasonInvalidDiscount                  Reason = "invalid_read_discount"
	ReasonNoExpectedReuse                  Reason = "no_expected_reuse"
	ReasonNoSharedAnchor                   Reason = "no_shared_anchor"
)

// PrefixFingerprint is the byte-stability witness M2 hands M3. Digest should be
// the hash of the exact serialized prefix bytes that the provider hashes for
// prompt-cache matching; the other fields scope the identity so a serializer or
// tokenization change cannot masquerade as byte identity.
type PrefixFingerprint struct {
	SerializerID string
	Digest       string
	Bytes        int
	Tokens       int
}

// Equal reports byte-identical prefix identity at the M3 boundary.
func (f PrefixFingerprint) Equal(g PrefixFingerprint) bool {
	return f.SerializerID == g.SerializerID &&
		f.Digest == g.Digest &&
		f.Bytes == g.Bytes &&
		f.Tokens == g.Tokens
}

// Valid reports whether the fingerprint carries enough evidence to count a warm.
func (f PrefixFingerprint) Valid() bool {
	return f.SerializerID != "" && f.Digest != "" && f.Bytes > 0 && f.Tokens > 0
}

// Request is the pure decision input for one candidate warm.
type Request struct {
	Provider Provider

	// ActiveCapability must be Supported before Plan may choose any active
	// warming path. Unknown and Unsupported both fail closed with a named
	// reason, so default-on callers cannot silently spend on an unproved path.
	ActiveCapability ActiveCacheCapability

	// ExpectedReuseBeforeTTL is k: the expected number of later reads before the
	// provider TTL expires. Dedicated warms are refused below the strict
	// decode-1 break-even gate.
	ExpectedReuseBeforeTTL int
	// ReadDiscount is r in k > 1/(1-r). If zero, the provider default is used:
	// Anthropic 0.1, OpenAI/DeepSeek 0.5, and 0.5 for unknown implicit providers.
	ReadDiscount float64

	// WarmPrefix is the prefix bytes the warm would target. RealPrefix is the
	// shared prefix the later real request will send. Decode-1 warms are counted
	// only when these are byte-identical.
	WarmPrefix PrefixFingerprint
	RealPrefix PrefixFingerprint

	// SharedBlockCount is the count of serialized content blocks shared with the
	// real request. For Anthropic max_tokens:0, the cache_control breakpoint is
	// placed on SharedBlockCount-1, never on a placeholder user turn.
	SharedBlockCount int

	Stream           bool
	ExtendedThinking bool
	StructuredOutput bool
	ToolChoice       ToolChoice
	Batch            bool
}

// Decision is the primitive and guard evidence a caller may act on.
type Decision struct {
	Primitive Primitive
	Reason    Reason

	Dedicated bool
	MaxTokens int

	RequiredReuse          int
	ExpectedReuseBeforeTTL int
	ReadDiscount           float64

	BreakpointBlockIndex int
	FallbackFromExplicit bool
	RejectedExplicit     []Reason
	Fingerprint          PrefixFingerprint
}

// Plan chooses the M3 warming primitive without issuing it.
func Plan(req Request) Decision {
	discount := req.ReadDiscount
	if discount == 0 {
		discount = DefaultReadDiscount(req.Provider)
	}
	required, ok := DedicatedDecode1ReuseFloor(discount)
	base := Decision{
		RequiredReuse:          required,
		ExpectedReuseBeforeTTL: req.ExpectedReuseBeforeTTL,
		ReadDiscount:           discount,
		BreakpointBlockIndex:   -1,
		Fingerprint:            req.WarmPrefix,
	}
	if !ok {
		base.Reason = ReasonInvalidDiscount
		return base
	}
	if req.ActiveCapability != ActiveCacheCapabilitySupported {
		base.Reason = ReasonUnsupportedActiveCacheCapability
		return base
	}
	if req.ExpectedReuseBeforeTTL <= 0 {
		base.Reason = ReasonNoExpectedReuse
		return base
	}
	if IsAutoCacheProvider(req.Provider) {
		base.Primitive = PrimitiveOrderFirstReal
		base.Reason = ReasonAutoCacheNaturalWarm
		return base
	}
	if req.ExpectedReuseBeforeTTL < required {
		base.Reason = ReasonBelowBreakEven
		return base
	}
	if !req.WarmPrefix.Valid() || !req.RealPrefix.Valid() || !req.WarmPrefix.Equal(req.RealPrefix) {
		base.Reason = ReasonPrefixFingerprintMismatch
		return base
	}

	if req.Provider == ProviderAnthropic {
		rejections := ExplicitRejections(req)
		if len(rejections) == 0 {
			if req.SharedBlockCount <= 0 {
				base.Reason = ReasonNoSharedAnchor
				return base
			}
			base.Primitive = PrimitiveAnthropicMaxTokens0
			base.Reason = ReasonAnthropicExplicitWarm
			base.Dedicated = true
			base.MaxTokens = 0
			base.BreakpointBlockIndex = req.SharedBlockCount - 1
			return base
		}
		base.RejectedExplicit = rejections
		base.FallbackFromExplicit = true
		base.Primitive = PrimitiveDecode1
		base.Reason = ReasonExplicitRejectedFallback
		base.Dedicated = true
		base.MaxTokens = 1
		return base
	}

	base.Primitive = PrimitiveDecode1
	base.Reason = ReasonImplicitDecode1
	base.Dedicated = true
	base.MaxTokens = 1
	return base
}

// DefaultReadDiscount returns the design-note default r when calibration has not
// provided one.
func DefaultReadDiscount(provider Provider) float64 {
	if provider == ProviderAnthropic {
		return 0.1
	}
	return 0.5
}

// DedicatedDecode1ReuseFloor returns the minimum integer k that satisfies the
// strict break-even inequality k > 1/(1-r).
func DedicatedDecode1ReuseFloor(readDiscount float64) (int, bool) {
	if readDiscount < 0 || readDiscount >= 1 {
		return 0, false
	}
	return int(math.Floor(1/(1-readDiscount))) + 1, true
}

// IsAutoCacheProvider reports providers where the first real request warms for
// free, so M3 must never spend a dedicated decode-1 warm.
func IsAutoCacheProvider(provider Provider) bool {
	return provider == ProviderOpenAI || provider == ProviderDeepSeek
}

// ExplicitRejections returns the Anthropic max_tokens:0 combinations that must
// fall back to a real decode-1 warm.
func ExplicitRejections(req Request) []Reason {
	var out []Reason
	if req.Stream {
		out = append(out, "stream")
	}
	if req.ExtendedThinking {
		out = append(out, "extended_thinking")
	}
	if req.StructuredOutput {
		out = append(out, "structured_output")
	}
	if req.ToolChoice == ToolChoiceTool || req.ToolChoice == ToolChoiceAny {
		out = append(out, "tool_choice_"+Reason(req.ToolChoice))
	}
	if req.Batch {
		out = append(out, "batch")
	}
	return out
}

// StreamEventKind is the minimal event vocabulary needed by the send-one-then-fan
// barrier. Only a content delta proves the provider has started streaming content.
type StreamEventKind string

const (
	StreamEventHTTPStatus    StreamEventKind = "http_status"
	StreamEventMessageStart  StreamEventKind = "message_start"
	StreamEventContentDelta  StreamEventKind = "content_delta"
	StreamEventMessageDelta  StreamEventKind = "message_delta"
	StreamEventMessageStop   StreamEventKind = "message_stop"
	StreamEventTransportDone StreamEventKind = "transport_done"
)

// FanoutGate releases dependents only after the first streamed content delta.
type FanoutGate struct {
	released bool
}

// Observe records one stream event and reports whether dependents may run.
func (g *FanoutGate) Observe(kind StreamEventKind) bool {
	if kind == StreamEventContentDelta {
		g.released = true
	}
	return g.released
}

// Released reports whether the barrier has opened.
func (g FanoutGate) Released() bool { return g.released }

// CacheReadback is one later real request's provider-cache telemetry.
type CacheReadback struct {
	CacheReadTokens int64
}

// WarmStatus is the accounting verdict for a dedicated warm.
type WarmStatus string

const (
	WarmNotDedicated WarmStatus = "not_dedicated"
	WarmPending      WarmStatus = "pending"
	WarmConfirmed    WarmStatus = "confirmed"
	WarmWasted       WarmStatus = "wasted"
)

// WarmAccounting records whether a dedicated warm later read back from provider
// cache or became wasted spend.
type WarmAccounting struct {
	Status          WarmStatus
	Wasted          bool
	CacheReadTokens int64
	LaterCalls      int
}

// ReconcileWarm accounts for a completed fanout. A dedicated warm with no later
// cache_read tokens is recorded as wasted spend; savings are confirmed only from
// telemetry that the warmer did not author.
func ReconcileWarm(dec Decision, fanoutClosed bool, readbacks []CacheReadback) WarmAccounting {
	if !dec.Dedicated {
		return WarmAccounting{Status: WarmNotDedicated}
	}
	var tokens int64
	for _, rb := range readbacks {
		if rb.CacheReadTokens > 0 {
			tokens += rb.CacheReadTokens
		}
	}
	acc := WarmAccounting{CacheReadTokens: tokens, LaterCalls: len(readbacks)}
	if !fanoutClosed {
		acc.Status = WarmPending
		return acc
	}
	if tokens == 0 {
		acc.Status = WarmWasted
		acc.Wasted = true
		return acc
	}
	acc.Status = WarmConfirmed
	return acc
}
