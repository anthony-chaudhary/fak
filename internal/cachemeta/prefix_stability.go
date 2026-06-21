package cachemeta

import "bytes"

// prefix_stability.go — GLM52-HOSTED-CACHE-COHERENCE §A3 (the prefix-stability
// linter, CPU-only, no provider call).
//
// Provider prompt caching (GLM-5.2 / Z.AI, OpenAI, Anthropic) only rewards a turn
// whose prompt PREFIX is byte-stable versus the previous turn: the provider caches
// the longest matching prefix and re-bills everything from the first divergence
// onward at full price. A volatile span (timestamp, request id, a reordered tool
// schema) placed AHEAD of the stable bulk silently destroys the hit rate — the cache
// still "works", it just never hits, and a blended hit-rate number hides it (refusal
// rule 4). This analyzer makes that failure VISIBLE offline from recorded turns: it
// computes the first-divergence token offset (the same offset the §A2
// ProviderCache.FirstDivergeAt field records), scores cacheable vs lost prompt tokens,
// honors the sealed-span rule (refusal rule 3), flags the fixable
// volatile-ahead-of-stable ordering bug, and emits the recommended layout.

// SegmentKind classifies a prompt part for layout advice. Only SegSealed changes the
// divergence math (it stops the cacheable prefix); the rest inform the lint.
type SegmentKind string

const (
	SegStable     SegmentKind = "stable"      // system prompt, static instructions — should be byte-identical every turn
	SegToolSchema SegmentKind = "tool_schema" // tool/function definitions — stable, belongs in the prefix
	SegVolatile   SegmentKind = "volatile"    // timestamp, request id, nonce — known to change every turn
	SegToolResult SegmentKind = "tool_result" // an inbound tool result — stable once appended, but tail content
	SegMessage    SegmentKind = "message"     // a conversation message — tail content
	SegSealed     SegmentKind = "sealed"      // a span fak quarantined — MUST NOT be re-served via the provider cache
)

// PromptSegment is one ordered piece of a turn's prompt. Content is the exact
// serialized bytes the provider hashes for prefix matching; Tokens is the count the
// provider bills for the segment.
type PromptSegment struct {
	Kind    SegmentKind
	Tokens  int64
	Content []byte
	// Witness is the OPTIONAL external trust witness this segment's content depends on
	// (a git SHA / blob hash / lease epoch carried from the recorded page that produced
	// it). It lets §A4 break the prefix at the SEGMENT level when the witness is revoked,
	// avoiding any page-vs-prompt token-coordinate alignment — the witness travels with
	// the segment. Empty for segments with no external dependency. Ignored by the §A3
	// prefix-matching math (which compares only Content).
	Witness string
}

// TurnDivergence reports where turn `next`'s prompt prefix stops matching turn
// `prev`'s, and the provider-cache consequence in billed tokens.
type TurnDivergence struct {
	// FirstDivergeSeg is the index into `next` of the first segment that is NOT part of
	// the cacheable common prefix (a content mismatch, a length overrun, or a sealed
	// segment). It equals len(next) when every segment of next is a cacheable
	// continuation of prev (a pure append — the whole overlap hit).
	FirstDivergeSeg int
	// StableTokens is the provider-cacheable prefix length in tokens. This is the
	// FirstDivergeAt token offset the §A2 ProviderCache entry records.
	StableTokens int64
	// LostTokens is next's prompt tokens at/after the break — re-billed at full price.
	LostTokens int64
	// Identical is true when next is byte-identical to prev.
	Identical bool
	// SealedStop is true when the cacheable prefix was cut short by a sealed segment
	// rather than a content divergence (refusal rule 3 forbids re-serving it).
	SealedStop bool
}

// Diverge computes the cacheable common prefix between two consecutive turns' prompts.
// It walks segments in order: while next[i] is byte-identical to prev[i] AND not
// sealed, the segment's tokens stay cacheable; the first content mismatch, length
// overrun, or sealed segment ends the cacheable prefix. A sealed segment stops the
// prefix even when its bytes match (refusal rule 3).
func Diverge(prev, next []PromptSegment) TurnDivergence {
	d := TurnDivergence{}
	i := 0
	for i < len(next) {
		if next[i].Kind == SegSealed {
			d.SealedStop = true
			break
		}
		if i >= len(prev) {
			break // next is longer: the extra tail segments are new (lost) tokens.
		}
		if prev[i].Kind == SegSealed || !bytes.Equal(prev[i].Content, next[i].Content) {
			break
		}
		d.StableTokens += next[i].Tokens
		i++
	}
	d.FirstDivergeSeg = i
	for j := i; j < len(next); j++ {
		d.LostTokens += next[j].Tokens
	}
	d.Identical = !d.SealedStop && i == len(next) && len(next) == len(prev)
	return d
}

// FirstDivergeTokenOffset is the token offset where the provider prefix first breaks —
// exactly the value the §A2 ProviderCache.FirstDivergeAt field carries. It returns 0
// when the whole prompt is cacheable (no divergence in next's span).
func (d TurnDivergence) FirstDivergeTokenOffset() int64 {
	if d.LostTokens == 0 {
		return 0
	}
	return d.StableTokens
}

// LayoutLint flags the single most impactful fixable ordering bug in one turn: a
// volatile segment sitting AHEAD of stable/tool-schema content. Everything after the
// earliest volatile segment is uncacheable purely because of ordering.
type LayoutLint struct {
	EarliestVolatileSeg  int   // index of the first volatile segment, or -1 if none
	StrandedStableTokens int64 // stable/tool-schema tokens after the earliest volatile segment
	Fixable              bool  // StrandedStableTokens > 0
}

// LintPrefixLayout reports the volatile-ahead-of-stable ordering bug for one turn.
func LintPrefixLayout(turn []PromptSegment) LayoutLint {
	lint := LayoutLint{EarliestVolatileSeg: -1}
	for i, seg := range turn {
		if seg.Kind == SegVolatile {
			lint.EarliestVolatileSeg = i
			break
		}
	}
	if lint.EarliestVolatileSeg < 0 {
		return lint
	}
	for _, seg := range turn[lint.EarliestVolatileSeg+1:] {
		if seg.Kind == SegStable || seg.Kind == SegToolSchema {
			lint.StrandedStableTokens += seg.Tokens
		}
	}
	lint.Fixable = lint.StrandedStableTokens > 0
	return lint
}

// LayoutRecommendation is the §A3 "recommended layout" output for one turn: the
// cache-optimal re-ordering plus the predicted cacheable-token uplift.
type LayoutRecommendation struct {
	Reordered       []PromptSegment // volatile segments hoisted to the tail; others keep original order
	MovedVolatile   int
	BeforeCacheable int64
	AfterCacheable  int64
	PredictedUplift int64
	Changed         bool
}

// frontCacheableRun is the contiguous leading run a provider could cache for this
// turn: leading segments until the first volatile (breaks the prefix) or sealed
// (refusal rule 3), exclusive.
func frontCacheableRun(segs []PromptSegment) int64 {
	var n int64
	for _, s := range segs {
		if s.Kind == SegVolatile || s.Kind == SegSealed {
			break
		}
		n += s.Tokens
	}
	return n
}

// RecommendLayout produces the cache-optimal re-ordering for one turn: every volatile
// segment is hoisted to the tail (volatile content is position-independent metadata —
// a timestamp / request id — so moving it preserves semantics), while every
// non-volatile segment keeps its original relative order (so conversation messages,
// tool results, and sealed spans are never reordered among themselves). The predicted
// uplift is the extra contiguous front-prefix tokens the provider can then cache. A
// sealed segment still caps the run wherever it sits (refusal rule 3 stays enforced).
func RecommendLayout(turn []PromptSegment) LayoutRecommendation {
	rec := LayoutRecommendation{BeforeCacheable: frontCacheableRun(turn)}
	nonVol := make([]PromptSegment, 0, len(turn))
	var vol []PromptSegment
	for _, s := range turn {
		if s.Kind == SegVolatile {
			vol = append(vol, s)
		} else {
			nonVol = append(nonVol, s)
		}
	}
	rec.MovedVolatile = len(vol)
	rec.Reordered = append(nonVol, vol...)
	rec.AfterCacheable = frontCacheableRun(rec.Reordered)
	rec.PredictedUplift = rec.AfterCacheable - rec.BeforeCacheable
	rec.Changed = rec.MovedVolatile > 0 && rec.PredictedUplift > 0
	return rec
}

// StabilityReport rolls a recorded session (one prompt per turn) into the §A3
// headline: how many provider-cacheable tokens survive across the session, how many
// are re-billed, and the SPECIFIC turn where the prefix first broke.
type StabilityReport struct {
	Turns             int
	CacheableTokens   int64
	LostTokens        int64
	BrokeAtTurn       int   // first turn (>=1) whose prefix diverged within the shared span; -1 if never
	RecoverableTokens int64 // sum of LintPrefixLayout.StrandedStableTokens — predicted reorder uplift
}

// AnalyzeStability computes the §A3 report over an ordered list of turns (each turn is
// the full serialized prompt for that request).
func AnalyzeStability(turns [][]PromptSegment) StabilityReport {
	r := StabilityReport{Turns: len(turns), BrokeAtTurn: -1}
	for _, turn := range turns {
		r.RecoverableTokens += LintPrefixLayout(turn).StrandedStableTokens
	}
	for i := 1; i < len(turns); i++ {
		d := Diverge(turns[i-1], turns[i])
		r.CacheableTokens += d.StableTokens
		r.LostTokens += d.LostTokens
		// A break is a divergence WITHIN the previous turn's span that strands tokens:
		// not a pure tail-append (FirstDivergeSeg==len(prev)) AND tokens were re-billed
		// (a clean truncation, LostTokens==0, is not a break).
		if r.BrokeAtTurn < 0 && d.LostTokens > 0 && d.FirstDivergeSeg < len(turns[i-1]) {
			r.BrokeAtTurn = i
		}
	}
	return r
}
