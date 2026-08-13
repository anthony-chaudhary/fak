package toon

import (
	"encoding/json"
	"fmt"
	"reflect"
)

// SkipReason is one token from a CLOSED vocabulary: the only values Decide may return in
// Decision.Reason when it refuses to fire. An unknown reason is a bug — the set is fixed
// here as Go constants and enumerated by KnownSkipReasons, mirroring the dos_* / internal
// ablate fail-loud posture (a refusal names a checkable condition, never free text).
//
// NOTE (#3066): each token here is registered in the repo-root dos.toml [reasons] table, so
// `dos man wedge <TOKEN> --explain` resolves it (valid, refusable) and `dos refuse-reasons` lists it.
// This file is the producer of record; the Go set and the dos.toml registration are bound by
// TestKnownSkipReasonsRegisteredInDosToml so neither can drift from the other.
type SkipReason string

const (
	// ReasonTabularEligibilityLow: TabularEligibility(payload) < τ. Nested / non-uniform
	// data collapses TOON accuracy (independent nested-data benches rank TOON last) because
	// it removes the repeated field-name anchors the model leans on — not a saving.
	ReasonTabularEligibilityLow SkipReason = "TABULAR_ELIGIBILITY_LOW"
	// ReasonPayloadTooSmall: rows < R_min or bytes < B_min. The fixed header (+ any primer)
	// overhead is not amortized on a small payload, so the net token delta is ≤ 0.
	ReasonPayloadTooSmall SkipReason = "PAYLOAD_TOO_SMALL"
	// ReasonNetTokensNonPositive: TOONTokens ≥ JSONTokens − margin, measured by the caller's
	// tokenizer. This is THE witness: if the real tokenizer shows no win, there is no win.
	// This gate makes "never fire at a loss" a proof, not a hope.
	ReasonNetTokensNonPositive SkipReason = "NET_TOKENS_NONPOSITIVE"
	// ReasonCachePrefixResident: the span is already inside a cached prefix. Re-encoding turns
	// a 0.1× cache read into a full-price recompute — a cost-multiplier loss dressed up as a
	// token saving. The fak-specific killer. (Signal supplied by the wiring layer, #3067.)
	ReasonCachePrefixResident SkipReason = "CACHE_PREFIX_RESIDENT"
	// ReasonVolatileSpan: the span head is volatile (the anthropic_cachebp.go
	// headValueIsVolatile notion — a per-turn-rewritten head). Encoding it thrashes the
	// stable prefix. (Signal supplied by the wiring layer, #3067.)
	ReasonVolatileSpan SkipReason = "VOLATILE_SPAN"
	// ReasonModelToonUnfit: the target model's TOON fitness is below φ and a primer won't
	// close the gap within budget. Small/untrained models degrade on the format; the primer
	// cost is counted against the saving. (Signal supplied by the wiring layer, #3067.)
	ReasonModelToonUnfit SkipReason = "MODEL_TOON_UNFIT"
	// ReasonRoundtripLossy: Decode(Encode(payload)) ≠ payload. The correctness spine — an
	// encoding we cannot prove reversible is never emitted.
	ReasonRoundtripLossy SkipReason = "ROUNDTRIP_LOSSY"
	// ReasonOutputDirection: the payload is a response schema the model must PRODUCE. Model
	// TOON *output* support is weak, so keep JSON — TOON here is input-only.
	ReasonOutputDirection SkipReason = "OUTPUT_DIRECTION"
)

// KnownSkipReasons returns the closed skip-reason vocabulary in a stable order. A reason
// outside this set is a bug; callers (and the eventual dos_refuse_reasons registration)
// enumerate the legal set from here rather than hard-coding it.
func KnownSkipReasons() []SkipReason {
	return []SkipReason{
		ReasonTabularEligibilityLow,
		ReasonPayloadTooSmall,
		ReasonNetTokensNonPositive,
		ReasonCachePrefixResident,
		ReasonVolatileSpan,
		ReasonModelToonUnfit,
		ReasonRoundtripLossy,
		ReasonOutputDirection,
	}
}

// Known reports whether r is a member of the closed vocabulary.
func (r SkipReason) Known() bool {
	switch r {
	case ReasonTabularEligibilityLow, ReasonPayloadTooSmall, ReasonNetTokensNonPositive,
		ReasonCachePrefixResident, ReasonVolatileSpan, ReasonModelToonUnfit,
		ReasonRoundtripLossy, ReasonOutputDirection:
		return true
	default:
		return false
	}
}

// Default thresholds. Principled starting points from issue #3066's research; NOT final —
// issue #3068 (the fak-native TOON scorecard) empirically CALIBRATES these against OBSERVED
// token deltas and WITNESSED accuracy on real fak payloads. A caller overrides any of them
// via DecideInput; a zero value there means "use the default below".
const (
	// DefaultTabularEligibilityMin (τ): fraction of the payload's scalar leaves that must live
	// in uniform tabular arrays for TOON's field-name-once win to apply. ~0.8: below it,
	// accuracy collapses off the tabular happy path (#3064's nested-data evidence).
	DefaultTabularEligibilityMin = 0.8
	// DefaultMinRows (R_min): a table needs a few rows before the one-time header cost is
	// out-earned by the per-row field-name savings.
	DefaultMinRows = 4
	// DefaultMinBytes (B_min): canonical-JSON bytes below which the fixed ~5–10% structural
	// (+ primer) overhead cannot be amortized.
	DefaultMinBytes = 160
	// DefaultNetTokenMargin (margin): minimum net tokens TOON must save vs JSON to fire.
	// Small on purpose — the comprehension tax is handled by MODEL_TOON_UNFIT — but > 0 so a
	// literal tie never fires. Clamped to ≥ 0 so the "never fire at a loss" invariant holds
	// even if a caller passes a negative margin.
	DefaultNetTokenMargin = 4
	// DefaultModelFitnessMin (φ): target-model TOON fitness below which the format degrades
	// enough to refuse (unless a primer closes the gap).
	DefaultModelFitnessMin = 0.5
)

// DecideInput carries every caller-supplied signal Decide consumes. Decide is PURE: it does
// no I/O and imports nothing from the gateway/agent layers. The cache/model/direction
// signals (CacheResident, Volatile, OutputDirection, ModelFitness*) are INPUTS that the
// wiring layer (#3067) will source from the cache-layout state, anthropic_cachebp.go's
// volatility check, and the model registry; THIS issue does not wire them.
type DecideInput struct {
	// Tokenizer measures a byte slice in the target model's tokens — the witness behind the
	// net-token gate. When nil, Decide falls back to the codec's bytes/4 yardstick
	// (memview.tokenEstimate) so the gate is always computable; a real caller supplies the
	// model's tokenizer for a true measurement.
	Tokenizer func([]byte) int

	// EncodeOptions threads through to Encode for the round-trip and net-token gates.
	EncodeOptions Options

	// --- cache / model / direction signals (the wiring layer #3067 supplies these) ---

	// CacheResident: the span already sits inside a cached prefix → CACHE_PREFIX_RESIDENT.
	CacheResident bool
	// Volatile: the span head is rewritten each turn (anthropic_cachebp headValueIsVolatile)
	// → VOLATILE_SPAN.
	Volatile bool
	// OutputDirection: the payload is a response schema the model must PRODUCE (not consume)
	// → OUTPUT_DIRECTION.
	OutputDirection bool
	// ModelFitnessKnown gates whether the model-fitness signal participates at all. When
	// false (no signal yet — the common standalone case before #3067/#3068 wire it), the
	// MODEL_TOON_UNFIT gate is inert; a fabricated unfitness would be dishonest.
	ModelFitnessKnown bool
	// ModelFitness is the target model's TOON fitness in [0,1] (target.TOONFitness).
	ModelFitness float64
	// PrimerLift is the fitness a short in-context primer can add; counted toward closing the
	// φ gap. Default 0 (no primer).
	PrimerLift float64

	// --- tunable thresholds (0 ⇒ the documented Default* above; #3068 calibrates) ---

	TabularEligibilityMin float64 // τ
	MinRows               int     // R_min
	MinBytes              int     // B_min
	NetTokenMargin        int     // margin (clamped ≥ 0)
	ModelFitnessMin       float64 // φ
}

// withDefaults returns a copy of in with every zero-valued threshold replaced by its
// documented default and the net-token margin clamped to ≥ 0 (so the safety invariant
// cannot be defeated by a negative margin). The tokenizer default is applied at call time
// by tokenize, not here.
func (in DecideInput) withDefaults() DecideInput {
	if in.TabularEligibilityMin == 0 {
		in.TabularEligibilityMin = DefaultTabularEligibilityMin
	}
	if in.MinRows == 0 {
		in.MinRows = DefaultMinRows
	}
	if in.MinBytes == 0 {
		in.MinBytes = DefaultMinBytes
	}
	if in.NetTokenMargin == 0 {
		in.NetTokenMargin = DefaultNetTokenMargin
	}
	if in.NetTokenMargin < 0 {
		in.NetTokenMargin = 0
	}
	if in.ModelFitnessMin == 0 {
		in.ModelFitnessMin = DefaultModelFitnessMin
	}
	return in
}

// tokenize measures b with the caller's Tokenizer, falling back to the bytes/4 yardstick
// (memview.tokenEstimate) when none is supplied.
func (in DecideInput) tokenize(b []byte) int {
	if in.Tokenizer != nil {
		return in.Tokenizer(b)
	}
	if len(b) <= 0 {
		return 0
	}
	return (len(b) + 3) / 4
}

// Decision is the result of Decide: either a fire (Encode == true, Reason == "") or a skip
// (Encode == false, Reason a single member of the closed vocabulary). It also carries the
// observed inputs that produced the verdict — Eligibility and the two token counts — so a
// run can render WHY TOON did or did not fire (decisions are auditable, not silent).
type Decision struct {
	Encode      bool
	Reason      SkipReason // "" iff Encode; otherwise exactly one KnownSkipReasons() member
	Eligibility float64    // TabularEligibility(payload)
	JSONTokens  int        // tokens of the canonical JSON encoding
	TOONTokens  int        // tokens of the TOON encoding (0 when the payload could not encode)
}

// String renders a one-line audit form of the decision, e.g.
// "toon: FIRE elig=1.00 json=42 toon=18" or "toon: SKIP(PAYLOAD_TOO_SMALL) elig=1.00 …".
func (d Decision) String() string {
	head := "SKIP(" + string(d.Reason) + ")"
	if d.Encode {
		head = "FIRE"
	}
	return fmt.Sprintf("toon: %s elig=%.2f json=%d toon=%d", head, d.Eligibility, d.JSONTokens, d.TOONTokens)
}

// skip is the single construction point for a refusal, so a Decision that skips can never
// accidentally carry Encode == true.
func (d Decision) skip(r SkipReason) Decision {
	d.Encode = false
	d.Reason = r
	return d
}

// Decide is the pure, governed TOON auto-fire decision. It returns Encode:true ONLY when
// every gate passes; otherwise it returns exactly one SkipReason from the closed set. It
// does no I/O and imports nothing from the gateway/agent layers — the cache/model/direction
// inputs arrive via DecideInput.
//
// The gates are checked cheapest-and-most-decisive first, with the two encode-dependent
// witnesses (round-trip, then the net-token measurement) LAST:
//
//  1. OUTPUT_DIRECTION      — TOON is input-only.
//  2. TABULAR_ELIGIBILITY_LOW — shape must be uniform-tabular.
//  3. PAYLOAD_TOO_SMALL     — big enough to amortize the header.
//  4. CACHE_PREFIX_RESIDENT — don't recompute a cached prefix.
//  5. VOLATILE_SPAN         — don't thrash a per-turn-rewritten head.
//  6. MODEL_TOON_UNFIT      — the model can read TOON (or a primer closes the gap).
//  7. ROUNDTRIP_LOSSY       — Decode(Encode(payload)) == payload.
//  8. NET_TOKENS_NONPOSITIVE — the tokenizer shows a real win beyond the margin.
//
// SAFETY INVARIANT ("never fire at a loss"): Decide never returns Encode:true when
// TOONTokens ≥ JSONTokens. Gate 8 is the proof: it fires (skip) whenever
// TOONTokens + margin ≥ JSONTokens, and margin is clamped ≥ 0, so passing it implies
// TOONTokens < JSONTokens. Gate 8 is the last gate before a fire, so the property holds for
// EVERY path that returns Encode:true. This is proven by the property test over a randomized
// corpus (decide_test.go, TestDecideNeverFiresAtALoss).
func Decide(payload any, in DecideInput) Decision {
	in = in.withDefaults()

	d := Decision{Eligibility: TabularEligibility(payload)}

	// Encode ONCE up front. The result feeds the round-trip and net-token gates and, for the
	// audit surface, the observed token counts on the Decision regardless of which gate fires.
	enc, encErr := Encode(payload, in.EncodeOptions)
	if jsonBytes, jErr := json.Marshal(payload); jErr == nil {
		d.JSONTokens = in.tokenize(jsonBytes)
	}
	if encErr == nil {
		d.TOONTokens = in.tokenize(enc)
	}

	// 1. Direction: TOON is input-only; a response schema the model must emit stays JSON.
	if in.OutputDirection {
		return d.skip(ReasonOutputDirection)
	}

	// 2. Shape: below τ the payload is nested/non-uniform and TOON's accuracy collapses.
	if d.Eligibility < in.TabularEligibilityMin {
		return d.skip(ReasonTabularEligibilityLow)
	}

	// 3. Size: too few rows or too few bytes to amortize the fixed header/primer overhead.
	if tabularRowCount(payload) < in.MinRows || payloadBytes(payload) < in.MinBytes {
		return d.skip(ReasonPayloadTooSmall)
	}

	// 4. Cache: re-encoding a cache-resident span is a full-price recompute (the fak killer).
	if in.CacheResident {
		return d.skip(ReasonCachePrefixResident)
	}

	// 5. Volatility: encoding a per-turn-rewritten head thrashes the stable prefix.
	if in.Volatile {
		return d.skip(ReasonVolatileSpan)
	}

	// 6. Model fitness: an unfit model that a primer can't lift over φ degrades on TOON.
	if in.ModelFitnessKnown && (in.ModelFitness+in.PrimerLift) < in.ModelFitnessMin {
		return d.skip(ReasonModelToonUnfit)
	}

	// 7. Round-trip: an encoding we can't prove reversible is never emitted. An encode error
	// is the degenerate "cannot produce a reversible encoding" case and skips here too.
	if encErr != nil || !roundTrips(payload, enc) {
		return d.skip(ReasonRoundtripLossy)
	}

	// 8. Net tokens: the witness. Skip unless the tokenizer shows a saving beyond the margin.
	// This is the last gate, so a fire ALWAYS implies TOONTokens < JSONTokens (margin ≥ 0).
	if d.TOONTokens+in.NetTokenMargin >= d.JSONTokens {
		return d.skip(ReasonNetTokensNonPositive)
	}

	d.Encode = true
	return d
}

// roundTrips reports whether Decode(enc) deep-equals payload. The round-trip witness holds
// for encoding/json-native values (map[string]any, []any, float64, string, bool, nil — the
// exact shape a tool result carries on the wire); a payload outside that domain (e.g. Go ints
// that Decode normalizes to float64) conservatively fails here, yielding a SKIP, never a
// wrong fire.
func roundTrips(payload any, enc []byte) bool {
	got, err := Decode(enc)
	if err != nil {
		return false
	}
	return reflect.DeepEqual(got, payload)
}

// payloadBytes is the canonical-JSON byte size used by the size gate. json.Marshal is
// deterministic (sorted map keys) so this is stable. An unmarshalable payload reports 0,
// which trips PAYLOAD_TOO_SMALL (and, later, ROUNDTRIP_LOSSY) — a skip, never a fire.
func payloadBytes(v any) int {
	b, err := json.Marshal(v)
	if err != nil {
		return 0
	}
	return len(b)
}

// tabularRowCount counts the rows that live in uniform, flat, tabular-eligible arrays
// anywhere in the payload — the count the header-amortization size gate reasons about. A
// non-tabular payload reports 0 (and would already have tripped the eligibility gate).
func tabularRowCount(v any) int {
	switch x := v.(type) {
	case []any:
		if _, ok := uniformFlatFields(x); ok {
			return len(x)
		}
		n := 0
		for _, e := range x {
			n += tabularRowCount(e)
		}
		return n
	case map[string]any:
		n := 0
		for _, val := range x {
			n += tabularRowCount(val)
		}
		return n
	default:
		return 0
	}
}
