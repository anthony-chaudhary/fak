package ctxplan

import (
	"sort"
	"strconv"
	"strings"
)

// AssumptionSource is the closed provenance vocabulary for context assumptions.
// The order is deliberate: a caller should be able to distinguish user-stated
// facts, witnessed facts, inferred guesses, stale recall, and unknowns before
// an effectful action consumes them.
type AssumptionSource string

const (
	AssumptionUserStated AssumptionSource = "user_stated"
	AssumptionWitnessed  AssumptionSource = "witnessed"
	AssumptionInferred   AssumptionSource = "inferred"
	AssumptionStale      AssumptionSource = "stale"
	AssumptionUnknown    AssumptionSource = "unknown"
)

// AssumptionAction is the managed-context decision for one assumption.
type AssumptionAction string

const (
	AssumptionUse     AssumptionAction = "use"
	AssumptionQuery   AssumptionAction = "query"
	AssumptionRefresh AssumptionAction = "refresh"
)

// Assumption is one fact-like item the context plan may rely on. Confidence is
// normalized to [0,1]; when omitted (0), the source class supplies a conservative
// default so stale/unknown entries cannot accidentally pass as high confidence.
type Assumption struct {
	Key        string           `json:"key"`
	Statement  string           `json:"statement,omitempty"`
	Source     AssumptionSource `json:"source"`
	Confidence float64          `json:"confidence,omitempty"`
	SourceRef  string           `json:"source_ref,omitempty"`
}

// AssumptionPolicy holds the thresholds for converting provenance+confidence
// into an action. Inferred assumptions have a higher bar than user-stated or
// witnessed facts because they were not directly supplied by a strong source.
type AssumptionPolicy struct {
	MinConfidence      float64 `json:"min_confidence"`
	InferredConfidence float64 `json:"inferred_confidence"`
}

// DefaultAssumptionPolicy is intentionally conservative for effect contexts.
func DefaultAssumptionPolicy() AssumptionPolicy {
	return AssumptionPolicy{MinConfidence: 0.65, InferredConfidence: 0.80}
}

// AssumptionAssessment is the scored, actioned form of one assumption.
type AssumptionAssessment struct {
	Key        string           `json:"key"`
	Statement  string           `json:"statement,omitempty"`
	Source     AssumptionSource `json:"source"`
	Confidence float64          `json:"confidence"`
	Action     AssumptionAction `json:"action"`
	Reason     string           `json:"reason"`
	SourceRef  string           `json:"source_ref,omitempty"`
}

// AssumptionSummary counts the action classes so a caller can gate effects with
// one field while still preserving per-assumption detail.
type AssumptionSummary struct {
	Use     int `json:"use"`
	Query   int `json:"query"`
	Refresh int `json:"refresh"`
}

// AssumptionReport is the plan-level gate for assumptions. EffectSafe is true
// only when every assumption is usable as-is; query/refresh items must be
// resolved before acting on the plan as if the assumptions were fresh facts.
type AssumptionReport struct {
	EffectSafe  bool                   `json:"effect_safe"`
	Summary     AssumptionSummary      `json:"summary"`
	Assessments []AssumptionAssessment `json:"assessments,omitempty"`
}

// AssessAssumptions scores assumptions into use/query/refresh actions with a
// deterministic key order. It is pure and host-agnostic so selfquery, gateway,
// and debug surfaces can consume the same fold without importing each other.
func AssessAssumptions(in []Assumption, policy AssumptionPolicy) AssumptionReport {
	policy = normalizeAssumptionPolicy(policy)
	items := append([]Assumption(nil), in...)
	sort.SliceStable(items, func(i, j int) bool {
		ki, kj := assumptionKey(items[i], i), assumptionKey(items[j], j)
		if ki != kj {
			return ki < kj
		}
		return items[i].Statement < items[j].Statement
	})
	out := AssumptionReport{EffectSafe: true}
	for i, a := range items {
		assessment := assessAssumption(a, policy, i)
		switch assessment.Action {
		case AssumptionRefresh:
			out.Summary.Refresh++
			out.EffectSafe = false
		case AssumptionQuery:
			out.Summary.Query++
			out.EffectSafe = false
		default:
			out.Summary.Use++
		}
		out.Assessments = append(out.Assessments, assessment)
	}
	return out
}

func assessAssumption(a Assumption, policy AssumptionPolicy, idx int) AssumptionAssessment {
	source := normalizeAssumptionSource(a.Source)
	confidence := normalizeAssumptionConfidence(a.Confidence, source)
	out := AssumptionAssessment{
		Key:        assumptionKey(a, idx),
		Statement:  strings.TrimSpace(a.Statement),
		Source:     source,
		Confidence: confidence,
		SourceRef:  strings.TrimSpace(a.SourceRef),
	}
	switch source {
	case AssumptionStale:
		out.Action = AssumptionRefresh
		out.Reason = "stale assumption must refresh its source before effects"
	case AssumptionUnknown:
		out.Action = AssumptionQuery
		out.Reason = "unknown assumption must be queried before effects"
	case AssumptionInferred:
		if confidence < policy.InferredConfidence {
			out.Action = AssumptionQuery
			out.Reason = "inferred assumption below confidence threshold"
		} else {
			out.Action = AssumptionUse
			out.Reason = "inferred assumption cleared confidence threshold"
		}
	default:
		if confidence < policy.MinConfidence {
			out.Action = AssumptionQuery
			out.Reason = "assumption below confidence threshold"
		} else {
			out.Action = AssumptionUse
			out.Reason = "assumption has direct source and sufficient confidence"
		}
	}
	return out
}

func normalizeAssumptionPolicy(p AssumptionPolicy) AssumptionPolicy {
	if p.MinConfidence <= 0 || p.MinConfidence > 1 {
		p.MinConfidence = DefaultAssumptionPolicy().MinConfidence
	}
	if p.InferredConfidence <= 0 || p.InferredConfidence > 1 {
		p.InferredConfidence = DefaultAssumptionPolicy().InferredConfidence
	}
	if p.InferredConfidence < p.MinConfidence {
		p.InferredConfidence = p.MinConfidence
	}
	return p
}

func normalizeAssumptionSource(source AssumptionSource) AssumptionSource {
	switch source {
	case AssumptionUserStated, AssumptionWitnessed, AssumptionInferred, AssumptionStale, AssumptionUnknown:
		return source
	default:
		return AssumptionUnknown
	}
}

func normalizeAssumptionConfidence(v float64, source AssumptionSource) float64 {
	if v > 0 {
		if v > 1 {
			return 1
		}
		return v
	}
	switch source {
	case AssumptionUserStated, AssumptionWitnessed:
		return 1
	case AssumptionInferred:
		return 0.5
	case AssumptionStale:
		return 0.2
	default:
		return 0
	}
}

func assumptionKey(a Assumption, idx int) string {
	key := strings.TrimSpace(a.Key)
	if key != "" {
		return key
	}
	return "assumption:" + strconv.Itoa(idx)
}
