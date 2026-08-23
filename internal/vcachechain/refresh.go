package vcachechain

import "sort"

// FactRecord is one keyed fact in a cached base or immutable correction segment.
// Keys identify facts across segments; later records with the same key supersede
// earlier values when the effective fact set is resolved.
type FactRecord struct {
	Key   string `json:"key"`
	Value string `json:"value"`
}

// CorrectionSegment is one append-only delta after an immutable cached base.
// ByteLen is the serialized segment size used by RefreshPolicy. When it is not
// positive, CorrectionBytes derives a deterministic lower bound from fact bytes.
type CorrectionSegment struct {
	ID      string       `json:"id"`
	Facts   []FactRecord `json:"facts,omitempty"`
	ByteLen int          `json:"byte_len"`
}

// CorrectionChain holds one immutable base plus its correction segments in append
// order. Its fields stay private so callers cannot rewrite the cached base or an
// earlier correction through an aliased slice.
type CorrectionChain struct {
	base        []FactRecord
	corrections []CorrectionSegment
}

// NewCorrectionChain snapshots base into an empty correction chain.
func NewCorrectionChain(base []FactRecord) CorrectionChain {
	return CorrectionChain{base: cloneFacts(base)}
}

// AppendCorrection returns a new chain with segment appended. The receiver, its
// base, earlier correction versions, and the caller's segment remain unchanged.
func (c CorrectionChain) AppendCorrection(segment CorrectionSegment) CorrectionChain {
	next := CorrectionChain{
		base:        cloneFacts(c.base),
		corrections: cloneCorrections(c.corrections),
	}
	next.corrections = append(next.corrections, cloneCorrection(segment))
	return next
}

// BaseFacts returns a snapshot of the immutable base facts.
func (c CorrectionChain) BaseFacts() []FactRecord {
	return cloneFacts(c.base)
}

// Corrections returns a deep snapshot of correction segments in append order.
func (c CorrectionChain) Corrections() []CorrectionSegment {
	return cloneCorrections(c.corrections)
}

// EffectiveFacts resolves the newest value for every fact key without rewriting
// the base. Corrections are applied in append order and records within a segment
// in record order, so the last occurrence wins. Results are sorted by key to make
// the resolution byte-stable across runs.
func (c CorrectionChain) EffectiveFacts() []FactRecord {
	latest := make(map[string]FactRecord, len(c.base))
	for _, fact := range c.base {
		latest[fact.Key] = fact
	}
	for _, segment := range c.corrections {
		for _, fact := range segment.Facts {
			latest[fact.Key] = fact
		}
	}

	keys := make([]string, 0, len(latest))
	for key := range latest {
		keys = append(keys, key)
	}
	sort.Strings(keys)

	effective := make([]FactRecord, 0, len(keys))
	for _, key := range keys {
		effective = append(effective, latest[key])
	}
	return effective
}

// CorrectionBytes returns the cumulative serialized size of appended corrections.
// A positive ByteLen is authoritative because the caller owns serialization. For
// an unset length, the UTF-8 bytes of each fact key and value provide a deterministic
// size instead of silently treating a populated segment as free.
func (c CorrectionChain) CorrectionBytes() int {
	total := 0
	for _, segment := range c.corrections {
		if segment.ByteLen > 0 {
			total += segment.ByteLen
			continue
		}
		for _, fact := range segment.Facts {
			total += len(fact.Key) + len(fact.Value)
		}
	}
	return total
}

// RefreshPolicy closes the two correction-growth budgets. A non-positive limit is
// unconfigured; a configured limit is reached at equality, not only after overflow.
type RefreshPolicy struct {
	MaxCorrectionCount int `json:"max_correction_count,omitempty"`
	MaxCorrectionBytes int `json:"max_correction_bytes,omitempty"`
}

// RefreshAction is the closed base-refresh action vocabulary.
type RefreshAction string

const (
	RefreshKeepBase RefreshAction = "keep_base"
	RefreshBase     RefreshAction = "refresh_base"
)

// RefreshReason records which configured budget produced the action.
type RefreshReason string

const (
	RefreshWithinBudget                 RefreshReason = "within_budget"
	RefreshCorrectionCountLimit         RefreshReason = "correction_count_limit"
	RefreshCorrectionByteLimit          RefreshReason = "correction_byte_limit"
	RefreshCorrectionCountAndByteLimits RefreshReason = "correction_count_and_byte_limits"
)

// RefreshEvaluation is the typed receipt for evaluating correction growth.
type RefreshEvaluation struct {
	Action          RefreshAction `json:"action"`
	Reason          RefreshReason `json:"reason"`
	CorrectionCount int           `json:"correction_count"`
	CorrectionBytes int           `json:"correction_bytes"`
}

// NeedsRefresh reports whether the immutable base should be rebuilt from the
// effective facts and the correction chain reset.
func (d RefreshEvaluation) NeedsRefresh() bool {
	return d.Action == RefreshBase
}

// EvaluateRefresh evaluates both configured budgets and returns one closed decision.
func (c CorrectionChain) EvaluateRefresh(policy RefreshPolicy) RefreshEvaluation {
	count := len(c.corrections)
	bytes := c.CorrectionBytes()
	countReached := policy.MaxCorrectionCount > 0 && count >= policy.MaxCorrectionCount
	bytesReached := policy.MaxCorrectionBytes > 0 && bytes >= policy.MaxCorrectionBytes

	decision := RefreshEvaluation{
		Action:          RefreshKeepBase,
		Reason:          RefreshWithinBudget,
		CorrectionCount: count,
		CorrectionBytes: bytes,
	}
	switch {
	case countReached && bytesReached:
		decision.Action = RefreshBase
		decision.Reason = RefreshCorrectionCountAndByteLimits
	case countReached:
		decision.Action = RefreshBase
		decision.Reason = RefreshCorrectionCountLimit
	case bytesReached:
		decision.Action = RefreshBase
		decision.Reason = RefreshCorrectionByteLimit
	}
	return decision
}

func cloneFacts(facts []FactRecord) []FactRecord {
	return append([]FactRecord(nil), facts...)
}

func cloneCorrection(segment CorrectionSegment) CorrectionSegment {
	segment.Facts = cloneFacts(segment.Facts)
	return segment
}

func cloneCorrections(segments []CorrectionSegment) []CorrectionSegment {
	if len(segments) == 0 {
		return nil
	}
	out := make([]CorrectionSegment, len(segments))
	for i, segment := range segments {
		out[i] = cloneCorrection(segment)
	}
	return out
}
