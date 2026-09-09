package amdgpu

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"regexp"
	"strings"
	"time"
)

const (
	StrixValidationSchemaV1 = "fak.strix.validation/v1"
	StrixValidationSchemaV2 = "fak.strix.validation/v2"
	StrixValidationSchema   = StrixValidationSchemaV2
)

var fullGitTipRE = regexp.MustCompile(`^[0-9a-fA-F]{40}$`)
var sha256RE = regexp.MustCompile(`^sha256:[0-9a-f]{64}$`)

// StrixOracleKind represents a closed typed-oracle kind for Strix subkernel parity.
type StrixOracleKind string

const (
	StrixOracleExactArgmax     StrixOracleKind = "exact_argmax"
	StrixOracleCosineMaxAbs    StrixOracleKind = "cosine_max_abs"
	StrixOracleCosineArgmax    StrixOracleKind = "cosine_argmax"
	StrixOracleMaxAbs          StrixOracleKind = "max_abs"
	StrixOracleStateContinuity StrixOracleKind = "state_continuity"
	StrixOracleHostContract    StrixOracleKind = "host_contract"
)

// Valid returns true if the oracle kind belongs to the closed vocabulary.
func (k StrixOracleKind) Valid() bool {
	switch k {
	case StrixOracleExactArgmax,
		StrixOracleCosineMaxAbs,
		StrixOracleCosineArgmax,
		StrixOracleMaxAbs,
		StrixOracleStateContinuity,
		StrixOracleHostContract:
		return true
	default:
		return false
	}
}

// BoolPtr returns a pointer to a bool value.
func BoolPtr(b bool) *bool { return &b }

// Float64Ptr returns a pointer to a float64 value.
func Float64Ptr(f float64) *float64 { return &f }

// StrixParityMetrics captures observed numerical or functional metrics for an oracle event.
type StrixParityMetrics struct {
	// exact_argmax
	ArgmaxExact *bool `json:"argmax_exact,omitempty"`

	// cosine / numeric metrics
	CosineSimilarity *float64 `json:"cosine_similarity,omitempty"`
	MaxAbsoluteDelta *float64 `json:"max_absolute_delta,omitempty"`
	RelativeL2       *float64 `json:"relative_l2,omitempty"`
	MaxSourceDelta   *float64 `json:"max_source_delta,omitempty"`
	StateIdentity    *bool    `json:"state_identity,omitempty"`
	FiniteOutput     *bool    `json:"finite_output,omitempty"`

	// state_continuity
	StateCosine     *float64 `json:"state_cosine,omitempty"`
	StateDelta      *float64 `json:"state_delta,omitempty"`
	StateContinuous *bool    `json:"state_continuous,omitempty"`

	// host_contract
	ContractHolds *bool  `json:"contract_holds,omitempty"`
	ContractName  string `json:"contract_name,omitempty"`
}

// StrixParityBounds specifies canonical bounds and comparison semantics for an oracle event.
type StrixParityBounds struct {
	Comparison string `json:"comparison,omitempty"`

	// exact_argmax
	ExactMatch *bool `json:"exact_match,omitempty"`

	// cosine_max_abs
	MinCosine             *float64 `json:"min_cosine,omitempty"`
	CosineComparison      string   `json:"cosine_comparison,omitempty"`
	MaxAbsDelta           *float64 `json:"max_abs_delta,omitempty"`
	MaxAbsComparison      string   `json:"max_abs_comparison,omitempty"`
	MaxSourceDelta        *float64 `json:"max_source_delta,omitempty"`
	SourceDeltaComparison string   `json:"source_delta_comparison,omitempty"`
	StateIdentity         *bool    `json:"state_identity,omitempty"`
	FiniteOutput          *bool    `json:"finite_output,omitempty"`

	// state_continuity
	MinStateCosine        *float64 `json:"min_state_cosine,omitempty"`
	StateCosineComparison string   `json:"state_cosine_comparison,omitempty"`
	MaxStateDelta         *float64 `json:"max_state_delta,omitempty"`
	StateDeltaComparison  string   `json:"state_delta_comparison,omitempty"`
	StateComparison       string   `json:"state_comparison,omitempty"`
	StateContinuous       *bool    `json:"state_continuous,omitempty"`

	// host_contract
	ContractExpected *bool `json:"contract_expected,omitempty"`
}

// StrixParityEvent binds observed metrics, canonical bounds, case count, device observation,
// engine identity, and passed state for one typed oracle evaluation.
type StrixParityEvent struct {
	OracleKind     StrixOracleKind    `json:"oracle_kind"`
	CaseCount      int                `json:"case_count"`
	DeviceObserved bool               `json:"device_observed"`
	Engine         string             `json:"engine"`
	Passed         bool               `json:"passed"`
	Observed       StrixParityMetrics `json:"observed"`
	Bounds         StrixParityBounds  `json:"bounds"`
	Detail         string             `json:"detail,omitempty"`
}

// PhysicalParityCredit reports whether this event earns physical-device parity credit.
// Host contracts and unobserved events never earn physical-device parity credit.
func (e StrixParityEvent) PhysicalParityCredit() bool {
	if e.OracleKind == StrixOracleHostContract {
		return false
	}
	if !e.DeviceObserved {
		return false
	}
	return e.Passed
}

// Validate checks that the event adheres to closed vocabulary, valid metrics, finite values,
// proper comparison operators, bounded canonical thresholds, and no exact/cosine conflation.
func (e StrixParityEvent) Validate() error {
	if !e.OracleKind.Valid() {
		return fmt.Errorf("amdgpu: unsupported or unknown oracle kind %q", e.OracleKind)
	}
	if e.CaseCount <= 0 {
		return fmt.Errorf("amdgpu: oracle %s requires positive case_count, got %d", e.OracleKind, e.CaseCount)
	}
	if strings.TrimSpace(e.Engine) == "" {
		return fmt.Errorf("amdgpu: oracle %s requires non-empty engine identity", e.OracleKind)
	}

	checkFinite := func(name string, f *float64) error {
		if f == nil {
			return nil
		}
		if math.IsNaN(*f) || math.IsInf(*f, 0) {
			return fmt.Errorf("amdgpu: oracle %s metric/bound %s cannot be NaN or Inf", e.OracleKind, name)
		}
		return nil
	}

	for name, f := range map[string]*float64{
		"observed.cosine_similarity":  e.Observed.CosineSimilarity,
		"observed.max_absolute_delta": e.Observed.MaxAbsoluteDelta,
		"observed.relative_l2":        e.Observed.RelativeL2,
		"observed.max_source_delta":   e.Observed.MaxSourceDelta,
		"observed.state_cosine":       e.Observed.StateCosine,
		"observed.state_delta":        e.Observed.StateDelta,
		"bounds.min_cosine":           e.Bounds.MinCosine,
		"bounds.max_abs_delta":        e.Bounds.MaxAbsDelta,
		"bounds.max_source_delta":     e.Bounds.MaxSourceDelta,
		"bounds.min_state_cosine":     e.Bounds.MinStateCosine,
		"bounds.max_state_delta":      e.Bounds.MaxStateDelta,
	} {
		if err := checkFinite(name, f); err != nil {
			return err
		}
	}
	if e.Observed.MaxSourceDelta != nil {
		if e.Bounds.MaxSourceDelta == nil {
			return fmt.Errorf("amdgpu: oracle %s reports max_source_delta without a bound", e.OracleKind)
		}
		cmp := e.Bounds.SourceDeltaComparison
		if cmp == "" {
			cmp = "<="
		}
		if cmp != "<=" && cmp != "<" {
			return fmt.Errorf("amdgpu: wrong comparison %q for max_source_delta", cmp)
		}
		if e.Passed && ((cmp == "<=" && *e.Observed.MaxSourceDelta > *e.Bounds.MaxSourceDelta) || (cmp == "<" && *e.Observed.MaxSourceDelta >= *e.Bounds.MaxSourceDelta)) {
			return fmt.Errorf("amdgpu: false pass: observed max_source_delta exceeds bound")
		}
	}
	if e.Observed.StateIdentity != nil {
		if e.Bounds.StateIdentity == nil {
			return fmt.Errorf("amdgpu: oracle %s reports state_identity without a bound", e.OracleKind)
		}
		if e.Passed && *e.Observed.StateIdentity != *e.Bounds.StateIdentity {
			return fmt.Errorf("amdgpu: false pass: state_identity does not match bound")
		}
	}
	if e.Observed.FiniteOutput != nil {
		if e.Bounds.FiniteOutput == nil {
			return fmt.Errorf("amdgpu: oracle %s reports finite_output without a bound", e.OracleKind)
		}
		if e.Passed && *e.Observed.FiniteOutput != *e.Bounds.FiniteOutput {
			return fmt.Errorf("amdgpu: false pass: finite_output does not match bound")
		}
	}

	switch e.OracleKind {
	case StrixOracleExactArgmax:
		if e.Observed.CosineSimilarity != nil {
			return fmt.Errorf("amdgpu: exact/cosine conflation: exact_argmax evidence must not include cosine similarity")
		}
		if e.Bounds.MinCosine != nil {
			return fmt.Errorf("amdgpu: exact/cosine conflation: exact_argmax must not specify cosine bounds")
		}
		if e.Observed.ArgmaxExact == nil {
			return fmt.Errorf("amdgpu: exact_argmax requires observed argmax_exact metric")
		}
		cmp := e.Bounds.Comparison
		if cmp == "" {
			cmp = "=="
		}
		if cmp != "==" && cmp != "=" && cmp != "exact" {
			return fmt.Errorf("amdgpu: wrong comparison %q for exact_argmax (want == or exact)", cmp)
		}
		if e.Bounds.ExactMatch != nil && !*e.Bounds.ExactMatch {
			return fmt.Errorf("amdgpu: loosened bounds: exact_argmax requires exact_match=true")
		}
		if e.Passed && !*e.Observed.ArgmaxExact {
			return fmt.Errorf("amdgpu: false pass: passed is true but observed argmax_exact is false")
		}

	case StrixOracleCosineArgmax:
		if e.Observed.CosineSimilarity == nil || e.Observed.ArgmaxExact == nil {
			return fmt.Errorf("amdgpu: cosine_argmax requires cosine_similarity and argmax_exact")
		}
		if e.Bounds.MinCosine == nil || e.Bounds.ExactMatch == nil || !*e.Bounds.ExactMatch {
			return fmt.Errorf("amdgpu: cosine_argmax requires min_cosine and exact_match=true bounds")
		}
		if *e.Bounds.MinCosine < 0.90 || *e.Bounds.MinCosine > 1.0 {
			return fmt.Errorf("amdgpu: loosened bounds: min_cosine %.6f outside canonical range [0.90, 1.00]", *e.Bounds.MinCosine)
		}
		if e.Passed && (*e.Observed.CosineSimilarity < *e.Bounds.MinCosine || !*e.Observed.ArgmaxExact) {
			return fmt.Errorf("amdgpu: false pass: cosine_argmax observation violates bounds")
		}

	case StrixOracleCosineMaxAbs:
		if e.Observed.CosineSimilarity == nil {
			return fmt.Errorf("amdgpu: cosine_max_abs requires observed cosine_similarity metric")
		}
		if e.Observed.MaxAbsoluteDelta == nil {
			return fmt.Errorf("amdgpu: cosine_max_abs requires observed max_absolute_delta metric")
		}
		if e.Bounds.MinCosine == nil {
			return fmt.Errorf("amdgpu: cosine_max_abs requires min_cosine bound")
		}
		if e.Bounds.MaxAbsDelta == nil {
			return fmt.Errorf("amdgpu: cosine_max_abs requires max_abs_delta bound")
		}

		cosCmp := e.Bounds.CosineComparison
		if cosCmp == "" {
			cosCmp = e.Bounds.Comparison
		}
		if cosCmp == "" {
			cosCmp = ">="
		}
		if cosCmp != ">=" && cosCmp != ">" {
			return fmt.Errorf("amdgpu: wrong comparison %q for cosine in cosine_max_abs (want >= or >)", cosCmp)
		}

		deltaCmp := e.Bounds.MaxAbsComparison
		if deltaCmp == "" {
			deltaCmp = "<="
		}
		if deltaCmp != "<=" && deltaCmp != "<" {
			return fmt.Errorf("amdgpu: wrong comparison %q for max_abs in cosine_max_abs (want <= or <)", deltaCmp)
		}

		if *e.Bounds.MinCosine < 0.90 || *e.Bounds.MinCosine > 1.0 {
			return fmt.Errorf("amdgpu: loosened bounds: min_cosine %.6f outside canonical range [0.90, 1.00]", *e.Bounds.MinCosine)
		}
		if *e.Bounds.MaxAbsDelta < 0.0 || *e.Bounds.MaxAbsDelta > 1.0 {
			return fmt.Errorf("amdgpu: loosened bounds: max_abs_delta %.6f outside canonical range [0.00, 1.00]", *e.Bounds.MaxAbsDelta)
		}

		if e.Passed {
			if cosCmp == ">=" && *e.Observed.CosineSimilarity < *e.Bounds.MinCosine {
				return fmt.Errorf("amdgpu: false pass: passed is true but observed cosine %.6f < bound %.6f", *e.Observed.CosineSimilarity, *e.Bounds.MinCosine)
			}
			if cosCmp == ">" && *e.Observed.CosineSimilarity <= *e.Bounds.MinCosine {
				return fmt.Errorf("amdgpu: false pass: passed is true but observed cosine %.6f <= bound %.6f", *e.Observed.CosineSimilarity, *e.Bounds.MinCosine)
			}
			if deltaCmp == "<=" && *e.Observed.MaxAbsoluteDelta > *e.Bounds.MaxAbsDelta {
				return fmt.Errorf("amdgpu: false pass: passed is true but observed max_abs %.6f > bound %.6f", *e.Observed.MaxAbsoluteDelta, *e.Bounds.MaxAbsDelta)
			}
			if deltaCmp == "<" && *e.Observed.MaxAbsoluteDelta >= *e.Bounds.MaxAbsDelta {
				return fmt.Errorf("amdgpu: false pass: passed is true but observed max_abs %.6f >= bound %.6f", *e.Observed.MaxAbsoluteDelta, *e.Bounds.MaxAbsDelta)
			}
		}

	case StrixOracleMaxAbs:
		if e.Observed.CosineSimilarity != nil {
			return fmt.Errorf("amdgpu: exact/cosine conflation: max_abs evidence must not include cosine similarity")
		}
		if e.Bounds.MinCosine != nil {
			return fmt.Errorf("amdgpu: exact/cosine conflation: max_abs must not specify cosine bounds")
		}
		if e.Observed.MaxAbsoluteDelta == nil {
			return fmt.Errorf("amdgpu: max_abs requires observed max_absolute_delta metric")
		}
		if e.Bounds.MaxAbsDelta == nil {
			return fmt.Errorf("amdgpu: max_abs requires max_abs_delta bound")
		}
		cmp := e.Bounds.Comparison
		if cmp == "" {
			cmp = e.Bounds.MaxAbsComparison
		}
		if cmp == "" {
			cmp = "<="
		}
		if cmp != "<=" && cmp != "<" {
			return fmt.Errorf("amdgpu: wrong comparison %q for max_abs (want <= or <)", cmp)
		}
		if *e.Bounds.MaxAbsDelta < 0.0 || *e.Bounds.MaxAbsDelta > 1.0 {
			return fmt.Errorf("amdgpu: loosened bounds: max_abs_delta %.6f outside canonical range [0.00, 1.00]", *e.Bounds.MaxAbsDelta)
		}
		if e.Passed {
			if cmp == "<=" && *e.Observed.MaxAbsoluteDelta > *e.Bounds.MaxAbsDelta {
				return fmt.Errorf("amdgpu: false pass: passed is true but observed max_abs %.6f > bound %.6f", *e.Observed.MaxAbsoluteDelta, *e.Bounds.MaxAbsDelta)
			}
			if cmp == "<" && *e.Observed.MaxAbsoluteDelta >= *e.Bounds.MaxAbsDelta {
				return fmt.Errorf("amdgpu: false pass: passed is true but observed max_abs %.6f >= bound %.6f", *e.Observed.MaxAbsoluteDelta, *e.Bounds.MaxAbsDelta)
			}
		}

	case StrixOracleStateContinuity:
		hasMetric := e.Observed.StateContinuous != nil || e.Observed.StateCosine != nil || e.Observed.StateDelta != nil || e.Observed.StateIdentity != nil || e.Observed.FiniteOutput != nil
		if !hasMetric {
			return fmt.Errorf("amdgpu: state_continuity requires observed state continuity metric")
		}
		if e.Observed.StateContinuous != nil {
			if e.Bounds.StateContinuous == nil {
				return fmt.Errorf("amdgpu: state_continuity requires state_continuous bound")
			}
			cmp := e.Bounds.Comparison
			if cmp == "" {
				cmp = "=="
			}
			if cmp != "==" && cmp != "=" && cmp != "exact" {
				return fmt.Errorf("amdgpu: wrong comparison %q for state_continuous (want == or exact)", cmp)
			}
			if e.Passed && *e.Observed.StateContinuous != *e.Bounds.StateContinuous {
				return fmt.Errorf("amdgpu: false pass: passed is true but state_continuous does not match bound")
			}
		}
		if e.Observed.StateCosine != nil {
			if e.Bounds.MinStateCosine == nil {
				return fmt.Errorf("amdgpu: state_continuity requires min_state_cosine bound")
			}
			cmp := e.Bounds.StateCosineComparison
			if cmp == "" {
				cmp = e.Bounds.StateComparison
			}
			if cmp == "" {
				cmp = ">="
			}
			if cmp != ">=" && cmp != ">" {
				return fmt.Errorf("amdgpu: wrong comparison %q for state_cosine (want >= or >)", cmp)
			}
			if *e.Bounds.MinStateCosine < 0.90 || *e.Bounds.MinStateCosine > 1.0 {
				return fmt.Errorf("amdgpu: loosened bounds: min_state_cosine %.6f outside canonical range [0.90, 1.00]", *e.Bounds.MinStateCosine)
			}
			if e.Passed {
				if cmp == ">=" && *e.Observed.StateCosine < *e.Bounds.MinStateCosine {
					return fmt.Errorf("amdgpu: false pass: passed is true but observed state_cosine %.6f < bound %.6f", *e.Observed.StateCosine, *e.Bounds.MinStateCosine)
				}
				if cmp == ">" && *e.Observed.StateCosine <= *e.Bounds.MinStateCosine {
					return fmt.Errorf("amdgpu: false pass: passed is true but observed state_cosine %.6f <= bound %.6f", *e.Observed.StateCosine, *e.Bounds.MinStateCosine)
				}
			}
		}
		if e.Observed.StateDelta != nil {
			if e.Bounds.MaxStateDelta == nil {
				return fmt.Errorf("amdgpu: state_continuity requires max_state_delta bound")
			}
			cmp := e.Bounds.StateDeltaComparison
			if cmp == "" {
				cmp = e.Bounds.StateComparison
			}
			if cmp == "" {
				cmp = "<="
			}
			if cmp != "<=" && cmp != "<" {
				return fmt.Errorf("amdgpu: wrong comparison %q for state_delta (want <= or <)", cmp)
			}
			if *e.Bounds.MaxStateDelta < 0.0 || *e.Bounds.MaxStateDelta > 1.0 {
				return fmt.Errorf("amdgpu: loosened bounds: max_state_delta %.6f outside canonical range [0.00, 1.00]", *e.Bounds.MaxStateDelta)
			}
			if e.Passed {
				if cmp == "<=" && *e.Observed.StateDelta > *e.Bounds.MaxStateDelta {
					return fmt.Errorf("amdgpu: false pass: passed is true but observed state_delta %.6f > bound %.6f", *e.Observed.StateDelta, *e.Bounds.MaxStateDelta)
				}
				if cmp == "<" && *e.Observed.StateDelta >= *e.Bounds.MaxStateDelta {
					return fmt.Errorf("amdgpu: false pass: passed is true but observed state_delta %.6f >= bound %.6f", *e.Observed.StateDelta, *e.Bounds.MaxStateDelta)
				}
			}
		}

	case StrixOracleHostContract:
		if e.DeviceObserved {
			return fmt.Errorf("amdgpu: host_contract cannot claim device_observed=true (host contracts are non-physical)")
		}
		if e.Observed.CosineSimilarity != nil {
			return fmt.Errorf("amdgpu: exact/cosine conflation: host_contract evidence must not fabricate cosine similarity")
		}
		if e.Observed.ContractHolds == nil {
			return fmt.Errorf("amdgpu: host_contract requires observed contract_holds metric")
		}
		cmp := e.Bounds.Comparison
		if cmp == "" {
			cmp = "=="
		}
		if cmp != "==" && cmp != "=" && cmp != "holds" {
			return fmt.Errorf("amdgpu: wrong comparison %q for host_contract (want == or holds)", cmp)
		}
		expected := true
		if e.Bounds.ContractExpected != nil {
			expected = *e.Bounds.ContractExpected
		}
		if e.Passed && *e.Observed.ContractHolds != expected {
			return fmt.Errorf("amdgpu: false pass: passed is true but contract_holds is %v (expected %v)", *e.Observed.ContractHolds, expected)
		}
	}

	return nil
}

// NewExactArgmaxParityEvent creates a verified exact_argmax parity event.
func NewExactArgmaxParityEvent(engine string, caseCount int, deviceObserved bool, exact bool) StrixParityEvent {
	return StrixParityEvent{
		OracleKind:     StrixOracleExactArgmax,
		CaseCount:      caseCount,
		DeviceObserved: deviceObserved,
		Engine:         engine,
		Passed:         exact,
		Observed: StrixParityMetrics{
			ArgmaxExact: BoolPtr(exact),
		},
		Bounds: StrixParityBounds{
			Comparison: "==",
			ExactMatch: BoolPtr(true),
		},
	}
}

// NewCosineMaxAbsParityEvent creates a verified cosine_max_abs parity event.
func NewCosineMaxAbsParityEvent(engine string, caseCount int, deviceObserved bool, cosine, minCosine, maxAbs, maxAbsBound float64) StrixParityEvent {
	passed := cosine >= minCosine && maxAbs <= maxAbsBound
	return StrixParityEvent{
		OracleKind:     StrixOracleCosineMaxAbs,
		CaseCount:      caseCount,
		DeviceObserved: deviceObserved,
		Engine:         engine,
		Passed:         passed,
		Observed: StrixParityMetrics{
			CosineSimilarity: Float64Ptr(cosine),
			MaxAbsoluteDelta: Float64Ptr(maxAbs),
		},
		Bounds: StrixParityBounds{
			MinCosine:        Float64Ptr(minCosine),
			CosineComparison: ">=",
			MaxAbsDelta:      Float64Ptr(maxAbsBound),
			MaxAbsComparison: "<=",
		},
	}
}

// NewMaxAbsParityEvent creates a verified max_abs parity event.
func NewMaxAbsParityEvent(engine string, caseCount int, deviceObserved bool, maxAbs, maxAbsBound float64) StrixParityEvent {
	passed := maxAbs <= maxAbsBound
	return StrixParityEvent{
		OracleKind:     StrixOracleMaxAbs,
		CaseCount:      caseCount,
		DeviceObserved: deviceObserved,
		Engine:         engine,
		Passed:         passed,
		Observed: StrixParityMetrics{
			MaxAbsoluteDelta: Float64Ptr(maxAbs),
		},
		Bounds: StrixParityBounds{
			Comparison:       "<=",
			MaxAbsDelta:      Float64Ptr(maxAbsBound),
			MaxAbsComparison: "<=",
		},
	}
}

// NewStateContinuityParityEvent creates a verified state_continuity parity event.
func NewStateContinuityParityEvent(engine string, caseCount int, deviceObserved bool, stateCosine, minCosine, stateDelta, maxDelta float64) StrixParityEvent {
	passed := stateCosine >= minCosine && stateDelta <= maxDelta
	return StrixParityEvent{
		OracleKind:     StrixOracleStateContinuity,
		CaseCount:      caseCount,
		DeviceObserved: deviceObserved,
		Engine:         engine,
		Passed:         passed,
		Observed: StrixParityMetrics{
			StateCosine: Float64Ptr(stateCosine),
			StateDelta:  Float64Ptr(stateDelta),
		},
		Bounds: StrixParityBounds{
			MinStateCosine:        Float64Ptr(minCosine),
			StateCosineComparison: ">=",
			MaxStateDelta:         Float64Ptr(maxDelta),
			StateDeltaComparison:  "<=",
		},
	}
}

// NewHostContractParityEvent creates a verified host_contract parity event.
func NewHostContractParityEvent(engine string, caseCount int, contractName string, holds bool) StrixParityEvent {
	return StrixParityEvent{
		OracleKind:     StrixOracleHostContract,
		CaseCount:      caseCount,
		DeviceObserved: false,
		Engine:         engine,
		Passed:         holds,
		Observed: StrixParityMetrics{
			ContractHolds: BoolPtr(holds),
			ContractName:  contractName,
		},
		Bounds: StrixParityBounds{
			Comparison:       "==",
			ContractExpected: BoolPtr(true),
		},
	}
}

// StrixValidationReceipt represents a verified hardware execution artifact on AMD Strix Halo.
type StrixValidationReceipt struct {
	Schema             string                 `json:"schema"`
	Timestamp          string                 `json:"timestamp"`
	Verdict            string                 `json:"verdict"` // PASS | FAIL | SKIPPED
	Target             StrixTarget            `json:"target"`
	Provenance         StrixProvenance        `json:"provenance"`
	SelectedCount      int                    `json:"selected_count,omitempty"`
	ExecutedCount      int                    `json:"executed_count,omitempty"`
	SelectedSubkernels int                    `json:"selected_subkernels,omitempty"`
	ExecutedSubkernels int                    `json:"executed_subkernels,omitempty"`
	SelectedAblations  int                    `json:"selected_ablations,omitempty"`
	ExecutedAblations  int                    `json:"executed_ablations,omitempty"`
	Subkernels         []StrixSubkernelResult `json:"subkernels,omitempty"`
	Ablations          []StrixAblationResult  `json:"ablations,omitempty"`
	ParityEvents       []StrixParityEvent     `json:"parity_events,omitempty"`
	Failures           []string               `json:"failures,omitempty"`
	Digest             string                 `json:"digest,omitempty"`
	Verified           bool                   `json:"verified"`
	authority          strixReceiptAuthority
}

// UnmarshalJSON clears in-process physical authority before decoding public
// receipt fields, including when decoding fails after making partial progress.
func (r *StrixValidationReceipt) UnmarshalJSON(data []byte) error {
	r.authority = strixReceiptAuthority{}
	type wireReceipt StrixValidationReceipt
	return json.Unmarshal(data, (*wireReceipt)(r))
}

// strixReceiptAuthority is deliberately absent from the wire format. Its seal
// is created only after the verifier has observed the complete execution and
// cleanup path; hashing the public receipt is integrity, not authority.
type strixReceiptAuthority struct {
	seal    *strixReceiptAuthoritySeal
	binding string
}

type strixReceiptAuthoritySeal struct{}

var strixReceiptAuthoritySealValue strixReceiptAuthoritySeal

func (a strixReceiptAuthority) validFor(r *StrixValidationReceipt) bool {
	if a.seal != &strixReceiptAuthoritySealValue || a.binding == "" || r == nil {
		return false
	}
	binding, err := r.ComputeDigest()
	return err == nil && a.binding == binding
}

// StrixProvenance records the software revision, command, and run mode.
type StrixProvenance struct {
	GitRef                  string   `json:"git_ref,omitempty"`
	GitTip                  string   `json:"git_tip,omitempty"`
	Command                 string   `json:"command,omitempty"`
	GeneratedBy             string   `json:"generated_by"`
	Transport               string   `json:"transport"` // "local" | "ssh"
	SourceArchiveSHA256     string   `json:"source_archive_sha256,omitempty"`
	BinarySHA256            string   `json:"binary_sha256,omitempty"`
	ShaderBundleSHA256      string   `json:"shader_bundle_sha256,omitempty"`
	BuildCommandSHA256      string   `json:"build_command_sha256,omitempty"`
	ExecutionManifestSHA256 string   `json:"execution_manifest_sha256,omitempty"`
	EngineIdentity          string   `json:"engine_identity,omitempty"`
	Trace                   []string `json:"trace,omitempty"`
	CleanupObserved         bool     `json:"cleanup_observed"`
}

type StrixExecutionEvidence struct {
	SourceArchiveSHA256 string `json:"source_archive_sha256"`
	BinarySHA256        string `json:"binary_sha256"`
	ShaderBundleSHA256  string `json:"shader_bundle_sha256"`
	CommandSHA256       string `json:"command_sha256"`
	DeviceIdentity      string `json:"device_identity"`
	EngineIdentity      string `json:"engine_identity"`
	ArtifactRehashed    bool   `json:"artifact_rehashed"`
	DeviceTimeoutMS     int64  `json:"device_timeout_ms"`
	LeasePathSHA256     string `json:"lease_path_sha256"`
	AdmissionWaitMS     int64  `json:"admission_wait_ms"`
	Acquired            bool   `json:"acquired"`
	Released            bool   `json:"released"`
	AcquireOrdinal      int    `json:"acquire_ordinal"`
	ReleaseOrdinal      int    `json:"release_ordinal"`
	ExitCode            *int   `json:"exit_code,omitempty"`
	RawOutputSHA256     string `json:"raw_output_sha256"`
	RawOutputBytes      int    `json:"raw_output_bytes"`
}

// StrixSubkernelResult records the physical device execution of one compute sub-kernel.
type StrixSubkernelResult struct {
	Name         string                 `json:"name"`        // e.g. "argmax", "matmul_f32", "q4k_matmul"
	Status       string                 `json:"status"`      // PASS | FAIL | SKIPPED
	DurationUS   int64                  `json:"duration_us"` // latency in microseconds
	Iterations   int                    `json:"iterations"`
	Parity       StrixParityVerdict     `json:"parity"`
	ParityEvents []StrixParityEvent     `json:"parity_events,omitempty"`
	Evidence     StrixExecutionEvidence `json:"evidence"`
	Metrics      map[string]any         `json:"metrics,omitempty"`
	Error        string                 `json:"error,omitempty"`
}

// StrixParityVerdict captures numerical and functional agreement against CPU reference.
type StrixParityVerdict struct {
	ReferenceGEMV         string             `json:"reference_gemv"`
	LogitCosineSimilarity float64            `json:"logit_cosine_similarity"`
	MaxAbsoluteDelta      float64            `json:"max_absolute_delta"`
	RelativeL2            float64            `json:"relative_l2"`
	ArgmaxExact           bool               `json:"argmax_exact"`
	Passed                bool               `json:"passed"`
	OracleKind            StrixOracleKind    `json:"oracle_kind,omitempty"`
	Events                []StrixParityEvent `json:"events,omitempty"`
}

// AllParityEvents aggregates only explicit typed parity events. Historical
// legacy fields remain readable but never become v2 credit by inference.
func (s *StrixSubkernelResult) AllParityEvents() []StrixParityEvent {
	events := make([]StrixParityEvent, 0, len(s.ParityEvents)+len(s.Parity.Events))
	events = append(events, s.ParityEvents...)
	events = append(events, s.Parity.Events...)
	return events
}

// AllParityEvents returns all parity events from receipt level and all subkernels.
func (r *StrixValidationReceipt) AllParityEvents() []StrixParityEvent {
	events := make([]StrixParityEvent, 0, len(r.ParityEvents))
	events = append(events, r.ParityEvents...)
	for _, sk := range r.Subkernels {
		events = append(events, sk.AllParityEvents()...)
	}
	return events
}

// CreditEligible reports whether the receipt qualifies for physical Strix Halo parity credit.
// Historical v1 receipts and host contracts are non-credit. Current physical validation credit
// is deliberately narrower than general v2 validity: exactly one argmax subkernel and no ablations.
func (r *StrixValidationReceipt) CreditEligible() bool {
	if err := r.Validate(); err != nil {
		return false
	}
	if !r.authenticatedPass() {
		return false
	}
	if !r.Target.Reachable {
		return false
	}
	if r.SelectedCount != 1 || r.ExecutedCount != 1 ||
		r.SelectedSubkernels != 1 || r.ExecutedSubkernels != 1 ||
		len(r.Subkernels) != 1 || r.Subkernels[0].Name != "argmax" {
		return false
	}
	if r.SelectedAblations != 0 || r.ExecutedAblations != 0 || len(r.Ablations) != 0 {
		return false
	}
	events := r.AllParityEvents()
	if len(events) == 0 {
		return false
	}
	hasPhysicalCredit := false
	for _, ev := range events {
		if !ev.Passed {
			return false
		}
		if ev.PhysicalParityCredit() {
			hasPhysicalCredit = true
		}
	}
	return hasPhysicalCredit
}

// PhysicalParityCredit reports whether the receipt has at least one passed physical-device parity event.
func (r *StrixValidationReceipt) PhysicalParityCredit() bool {
	return r.CreditEligible()
}

// StrixAblationResult records a differential comparison across architectural or execution arms.
type StrixAblationResult struct {
	Dimension    string                 `json:"dimension"`     // "target" | "topology" | "quantization" | "residency" | "batch"
	Feature      string                 `json:"feature"`       // e.g. "f16_contiguize", "q4k_vs_f32", "fused_vs_discrete"
	BaselineArm  StrixArmResult         `json:"baseline_arm"`  // control
	CandidateArm StrixArmResult         `json:"candidate_arm"` // treatment
	Speedup      float64                `json:"speedup"`       // baseline_latency / candidate_latency
	LiftRatio    float64                `json:"lift_ratio"`    // candidate_throughput / baseline_throughput
	CosineParity float64                `json:"cosine_parity"` // numerical parity between arms
	Evidence     StrixExecutionEvidence `json:"evidence"`
	Verdict      string                 `json:"verdict"` // VERIFIED_LIFT | PARITY_MATCH | REGRESSION
}

// StrixArmResult captures throughput, latency, and memory for one ablation arm.
type StrixArmResult struct {
	Name            string  `json:"name"`
	LatencyUS       int64   `json:"latency_us"`
	Samples         int     `json:"samples"`
	ThroughputTokS  float64 `json:"throughput_tok_s,omitempty"`
	DRAMBandwidthGB float64 `json:"dram_bandwidth_gbps,omitempty"`
	AllocatedBytes  int64   `json:"allocated_bytes,omitempty"`
	Argmax          int     `json:"argmax,omitempty"`
}

// ComputeDigest computes a deterministic SHA-256 integrity digest over the
// public receipt. It does not mint physical execution authority.
func (r *StrixValidationReceipt) ComputeDigest() (string, error) {
	if r.Schema == StrixValidationSchemaV1 {
		return r.computeV1Digest()
	}
	copyReceipt := *r
	copyReceipt.Digest = ""

	raw, err := json.Marshal(copyReceipt)
	if err != nil {
		return "", fmt.Errorf("amdgpu: marshal receipt for digest: %w", err)
	}
	hash := sha256.Sum256(raw)
	return "sha256:" + hex.EncodeToString(hash[:]), nil
}

func (r *StrixValidationReceipt) computeV1Digest() (string, error) {
	type p struct {
		ReferenceGEMV         string  `json:"reference_gemv"`
		LogitCosineSimilarity float64 `json:"logit_cosine_similarity"`
		MaxAbsoluteDelta      float64 `json:"max_absolute_delta"`
		RelativeL2            float64 `json:"relative_l2"`
		ArgmaxExact           bool    `json:"argmax_exact"`
		Passed                bool    `json:"passed"`
	}
	type sk struct {
		Name       string         `json:"name"`
		Status     string         `json:"status"`
		DurationUS int64          `json:"duration_us"`
		Iterations int            `json:"iterations"`
		Parity     p              `json:"parity"`
		Metrics    map[string]any `json:"metrics,omitempty"`
		Error      string         `json:"error,omitempty"`
	}
	type arm struct {
		Name            string  `json:"name"`
		LatencyUS       int64   `json:"latency_us"`
		ThroughputTokS  float64 `json:"throughput_tok_s,omitempty"`
		DRAMBandwidthGB float64 `json:"dram_bandwidth_gbps,omitempty"`
		AllocatedBytes  int64   `json:"allocated_bytes,omitempty"`
		Argmax          int     `json:"argmax,omitempty"`
	}
	type ab struct {
		Dimension    string  `json:"dimension"`
		Feature      string  `json:"feature"`
		BaselineArm  arm     `json:"baseline_arm"`
		CandidateArm arm     `json:"candidate_arm"`
		Speedup      float64 `json:"speedup"`
		LiftRatio    float64 `json:"lift_ratio"`
		CosineParity float64 `json:"cosine_parity"`
		Verdict      string  `json:"verdict"`
	}
	type prov struct {
		GitRef      string `json:"git_ref,omitempty"`
		GitTip      string `json:"git_tip,omitempty"`
		Command     string `json:"command,omitempty"`
		GeneratedBy string `json:"generated_by"`
		Transport   string `json:"transport"`
	}
	type rec struct {
		Schema             string      `json:"schema"`
		Timestamp          string      `json:"timestamp"`
		Verdict            string      `json:"verdict"`
		Target             StrixTarget `json:"target"`
		Provenance         prov        `json:"provenance"`
		SelectedCount      int         `json:"selected_count,omitempty"`
		ExecutedCount      int         `json:"executed_count,omitempty"`
		SelectedSubkernels int         `json:"selected_subkernels,omitempty"`
		ExecutedSubkernels int         `json:"executed_subkernels,omitempty"`
		Subkernels         []sk        `json:"subkernels,omitempty"`
		Ablations          []ab        `json:"ablations,omitempty"`
		Failures           []string    `json:"failures,omitempty"`
		Digest             string      `json:"digest,omitempty"`
		Verified           bool        `json:"verified"`
	}
	x := rec{Schema: r.Schema, Timestamp: r.Timestamp, Verdict: r.Verdict, Target: r.Target, Provenance: prov{r.Provenance.GitRef, r.Provenance.GitTip, r.Provenance.Command, r.Provenance.GeneratedBy, r.Provenance.Transport}, SelectedCount: r.SelectedCount, ExecutedCount: r.ExecutedCount, SelectedSubkernels: r.SelectedSubkernels, ExecutedSubkernels: r.ExecutedSubkernels, Failures: r.Failures}
	for _, s := range r.Subkernels {
		x.Subkernels = append(x.Subkernels, sk{Name: s.Name, Status: s.Status, DurationUS: s.DurationUS, Iterations: s.Iterations, Parity: p{s.Parity.ReferenceGEMV, s.Parity.LogitCosineSimilarity, s.Parity.MaxAbsoluteDelta, s.Parity.RelativeL2, s.Parity.ArgmaxExact, s.Parity.Passed}, Metrics: s.Metrics, Error: s.Error})
	}
	for _, a := range r.Ablations {
		x.Ablations = append(x.Ablations, ab{a.Dimension, a.Feature, arm{a.BaselineArm.Name, a.BaselineArm.LatencyUS, a.BaselineArm.ThroughputTokS, a.BaselineArm.DRAMBandwidthGB, a.BaselineArm.AllocatedBytes, a.BaselineArm.Argmax}, arm{a.CandidateArm.Name, a.CandidateArm.LatencyUS, a.CandidateArm.ThroughputTokS, a.CandidateArm.DRAMBandwidthGB, a.CandidateArm.AllocatedBytes, a.CandidateArm.Argmax}, a.Speedup, a.LiftRatio, a.CosineParity, a.Verdict})
	}
	raw, err := json.Marshal(x)
	if err != nil {
		return "", err
	}
	h := sha256.Sum256(raw)
	return "sha256:" + hex.EncodeToString(h[:]), nil
}

// Validate checks the public receipt's schema, integrity, and structural
// consistency. Opaque in-process execution authority is intentionally checked
// only by credit consumers, so serialized receipts remain readable.
func (r *StrixValidationReceipt) Validate() error {
	if r.Schema != StrixValidationSchemaV1 && r.Schema != StrixValidationSchemaV2 {
		return fmt.Errorf("invalid schema %q (want %q or %q)", r.Schema, StrixValidationSchemaV2, StrixValidationSchemaV1)
	}
	if r.Verdict != "PASS" && r.Verdict != "FAIL" && r.Verdict != "SKIPPED" {
		return fmt.Errorf("invalid verdict %q (want PASS, FAIL, or SKIPPED)", r.Verdict)
	}
	if r.Digest == "" {
		return fmt.Errorf("receipt digest is required")
	}
	expectedDigest, err := r.ComputeDigest()
	if err != nil {
		return fmt.Errorf("cannot compute digest for verification: %w", err)
	}
	if r.Digest != expectedDigest {
		return fmt.Errorf("digest mismatch (recorded %s != computed %s)", r.Digest, expectedDigest)
	}
	if r.Schema == StrixValidationSchemaV1 {
		return nil
	}
	if r.Verdict != "PASS" {
		if r.Verified {
			return fmt.Errorf("non-PASS v2 receipt cannot be Verified")
		}
		return nil
	}
	if !r.Verified {
		return fmt.Errorf("PASS v2 receipt is not Verified")
	}
	if r.Verdict == "PASS" {
		if !r.Target.Reachable {
			return fmt.Errorf("verdict is PASS but target is not reachable")
		}
		if !strings.Contains(strings.ToLower(r.Target.GPUName), "8060s") &&
			!strings.Contains(strings.ToLower(r.Target.GPUName), "strix") &&
			!strings.Contains(strings.ToLower(r.Target.TargetISA), "gfx1151") {
			return fmt.Errorf("verdict is PASS but GPU %q / ISA %q is not Strix Halo", r.Target.GPUName, r.Target.TargetISA)
		}
		for _, sk := range r.Subkernels {
			if sk.Status == "FAIL" {
				return fmt.Errorf("verdict is PASS but subkernel %q failed: %s", sk.Name, sk.Error)
			}
		}
		if r.Schema == StrixValidationSchemaV2 {
			for _, sk := range r.Subkernels {
				events := sk.AllParityEvents()
				for _, ev := range events {
					if err := ev.Validate(); err != nil {
						return fmt.Errorf("subkernel %q parity event: %w", sk.Name, err)
					}
					if !ev.Passed {
						return fmt.Errorf("verdict is PASS but subkernel %q parity event failed", sk.Name)
					}
				}
			}
			for _, ev := range r.ParityEvents {
				if err := ev.Validate(); err != nil {
					return fmt.Errorf("receipt parity event: %w", err)
				}
				if !ev.Passed {
					return fmt.Errorf("verdict is PASS but receipt parity event failed")
				}
			}
		}
		if (r.SelectedCount > 0 && r.ExecutedCount == 0) || (r.SelectedSubkernels > 0 && r.ExecutedSubkernels == 0) {
			return fmt.Errorf("verdict is PASS but subkernels were selected and 0 were executed")
		}
		if len(r.Failures) > 0 {
			return fmt.Errorf("verdict is PASS but receipt contains failures: %s", strings.Join(r.Failures, "; "))
		}
		for _, ab := range r.Ablations {
			if ab.Verdict == "REGRESSION" {
				return fmt.Errorf("verdict is PASS but ablation %q suffered regression (speedup=%.2fx)", ab.Feature, ab.Speedup)
			}
		}
		if err := r.validateExecutionEvidence(); err != nil {
			return err
		}
	}
	return nil
}

func validExecutionEvidence(e StrixExecutionEvidence) error {
	for n, v := range map[string]string{"source archive": e.SourceArchiveSHA256, "binary": e.BinarySHA256, "shader bundle": e.ShaderBundleSHA256, "command": e.CommandSHA256, "lease path": e.LeasePathSHA256, "raw output": e.RawOutputSHA256} {
		if !sha256RE.MatchString(v) {
			return fmt.Errorf("missing or invalid %s digest", n)
		}
	}
	if e.EngineIdentity != "fak-native/vulkan" || strings.TrimSpace(e.DeviceIdentity) == "" {
		return fmt.Errorf("missing device or fak-native engine identity")
	}
	if !e.ArtifactRehashed {
		return fmt.Errorf("missing in-admission artifact rehash evidence")
	}
	if e.DeviceTimeoutMS <= 0 || e.DeviceTimeoutMS > 60000 {
		return fmt.Errorf("missing or invalid host-local device timeout")
	}
	if e.AdmissionWaitMS <= 0 || e.AdmissionWaitMS > 60000 {
		return fmt.Errorf("invalid admission wait")
	}
	if !e.Acquired || !e.Released || e.AcquireOrdinal <= 0 || e.ReleaseOrdinal <= e.AcquireOrdinal {
		return fmt.Errorf("incomplete or unordered admission evidence")
	}
	if e.ExitCode == nil || *e.ExitCode != 0 {
		return fmt.Errorf("missing or nonzero exit evidence")
	}
	if e.RawOutputBytes <= 0 {
		return fmt.Errorf("missing raw output evidence")
	}
	return nil
}

func (r *StrixValidationReceipt) crossBindEvidence(e StrixExecutionEvidence) error {
	if e.SourceArchiveSHA256 != r.Provenance.SourceArchiveSHA256 || e.BinarySHA256 != r.Provenance.BinarySHA256 || e.ShaderBundleSHA256 != r.Provenance.ShaderBundleSHA256 || e.DeviceIdentity != r.Target.GPUName+"|"+r.Target.TargetISA || e.EngineIdentity != r.Provenance.EngineIdentity {
		return fmt.Errorf("execution evidence contradicts top-level source/binary/shader/device/engine provenance")
	}
	return nil
}

func sameOptionalFloat(got, want *float64) bool {
	if got == nil || want == nil {
		return got == nil && want == nil
	}
	return *got == *want
}

func validateReceiptEventContract(selector string, event StrixParityEvent) error {
	contract, ok := LookupSubkernelParityContract(selector)
	if !ok {
		return fmt.Errorf("unknown subkernel selector %q", selector)
	}
	if event.OracleKind != StrixOracleKind(contract.OracleKind) || event.DeviceObserved != contract.DeviceObserved {
		return fmt.Errorf("typed event contradicts registered oracle/device contract")
	}
	if contract.DeviceObserved && event.Engine != StrixVulkanEngine {
		return fmt.Errorf("typed event contradicts registered engine contract")
	}
	if !sameOptionalFloat(event.Bounds.MinCosine, contract.Bounds.MinCosine) || !sameOptionalFloat(event.Bounds.MaxAbsDelta, contract.Bounds.MaxAbsDelta) || !sameOptionalFloat(event.Bounds.MaxSourceDelta, contract.Bounds.MaxSourceDelta) {
		return fmt.Errorf("typed event bounds contradict registered selector bounds")
	}
	if contract.Bounds.RequireArgmaxExact != (event.Bounds.ExactMatch != nil && *event.Bounds.ExactMatch) || contract.Bounds.RequireStateIdentity != (event.Bounds.StateIdentity != nil && *event.Bounds.StateIdentity) || contract.Bounds.RequireFinite != (event.Bounds.FiniteOutput != nil && *event.Bounds.FiniteOutput) {
		return fmt.Errorf("typed event boolean bounds contradict registered selector bounds")
	}
	if contract.Bounds.MinCosine != nil && (event.Observed.CosineSimilarity == nil || *event.Observed.CosineSimilarity < *contract.Bounds.MinCosine) {
		return fmt.Errorf("typed event lacks required in-bound cosine observation")
	}
	if contract.Bounds.MaxAbsDelta != nil && (event.Observed.MaxAbsoluteDelta == nil || *event.Observed.MaxAbsoluteDelta > *contract.Bounds.MaxAbsDelta) {
		return fmt.Errorf("typed event lacks required in-bound max-absolute observation")
	}
	if contract.Bounds.RequireSourceMutationCheck && (event.Observed.MaxSourceDelta == nil || contract.Bounds.MaxSourceDelta == nil || *event.Observed.MaxSourceDelta > *contract.Bounds.MaxSourceDelta) {
		return fmt.Errorf("typed event lacks required source-mutation observation")
	}
	if contract.Bounds.RequireArgmaxExact && (event.Observed.ArgmaxExact == nil || !*event.Observed.ArgmaxExact) {
		return fmt.Errorf("typed event lacks required exact argmax observation")
	}
	if contract.Bounds.RequireStateIdentity && (event.Observed.StateIdentity == nil || !*event.Observed.StateIdentity) {
		return fmt.Errorf("typed event lacks required state identity observation")
	}
	if contract.Bounds.RequireFinite && (event.Observed.FiniteOutput == nil || !*event.Observed.FiniteOutput) {
		return fmt.Errorf("typed event lacks required finite-output observation")
	}
	return nil
}
func executionManifestDigest(r *StrixValidationReceipt) string {
	var b strings.Builder
	for _, s := range r.Subkernels {
		fmt.Fprintf(&b, "subkernel:%s:%s\n", s.Name, s.Evidence.CommandSHA256)
	}
	for _, a := range r.Ablations {
		fmt.Fprintf(&b, "ablation:%s:%s\n", a.Feature, a.Evidence.CommandSHA256)
	}
	h := sha256.Sum256([]byte(b.String()))
	return "sha256:" + hex.EncodeToString(h[:])
}
func withinRatioTolerance(got, want, tol float64) bool {
	return want > 0 && !math.IsNaN(got) && !math.IsInf(got, 0) && math.Abs(got-want)/want <= tol
}
func (r *StrixValidationReceipt) authenticatedPass() bool {
	return r != nil && r.Schema == StrixValidationSchemaV2 && r.Verdict == "PASS" &&
		r.Verified && len(r.Failures) == 0 && r.authority.validFor(r)
}

// authorizePhysicalCredit is package-private so serialized data and external
// callers cannot mint physical credit. The production validation flow invokes
// it only after source/build artifacts, device execution, raw output, and
// cleanup have all been observed and cross-bound by validateExecutionEvidence.
func (r *StrixValidationReceipt) authorizePhysicalCredit() error {
	if err := r.validateExecutionEvidence(); err != nil {
		return err
	}
	binding, err := r.ComputeDigest()
	if err != nil {
		return fmt.Errorf("compute physical execution authority binding: %w", err)
	}
	r.authority = strixReceiptAuthority{seal: &strixReceiptAuthoritySealValue, binding: binding}
	return nil
}

func (r *StrixValidationReceipt) validateExecutionEvidence() error {
	if !fullGitTipRE.MatchString(r.Provenance.GitTip) || !sha256RE.MatchString(r.Provenance.SourceArchiveSHA256) || !sha256RE.MatchString(r.Provenance.BinarySHA256) || !sha256RE.MatchString(r.Provenance.ShaderBundleSHA256) || !sha256RE.MatchString(r.Provenance.BuildCommandSHA256) || !sha256RE.MatchString(r.Provenance.ExecutionManifestSHA256) || r.Provenance.EngineIdentity != "fak-native/vulkan" || strings.TrimSpace(r.Provenance.Command) == "" {
		return fmt.Errorf("PASS v2 receipt has incomplete immutable source/build/command provenance")
	}
	if !r.Provenance.CleanupObserved {
		return fmt.Errorf("PASS v2 receipt lacks cleanup evidence")
	}
	if len(r.Failures) > 0 || len(r.Subkernels)+len(r.Ablations) == 0 {
		return fmt.Errorf("PASS v2 receipt has failures or no execution evidence")
	}
	if r.SelectedCount != r.ExecutedCount || r.SelectedSubkernels != r.ExecutedSubkernels || r.ExecutedSubkernels != len(r.Subkernels) {
		return fmt.Errorf("PASS v2 receipt has partial subkernel execution")
	}
	if r.SelectedAblations != r.ExecutedAblations || r.ExecutedAblations != len(r.Ablations) {
		return fmt.Errorf("PASS v2 receipt has partial ablation execution")
	}
	for _, s := range r.Subkernels {
		if s.Status != "PASS" || s.DurationUS <= 0 || s.Iterations <= 0 {
			return fmt.Errorf("subkernel %q has incomplete execution", s.Name)
		}
		events := s.AllParityEvents()
		if len(events) == 0 {
			return fmt.Errorf("subkernel %q lacks typed parity", s.Name)
		}
		physical := false
		for _, ev := range events {
			if err := ev.Validate(); err != nil {
				return fmt.Errorf("subkernel %q parity: %w", s.Name, err)
			}
			if err := validateReceiptEventContract(s.Name, ev); err != nil {
				return fmt.Errorf("subkernel %q parity contract: %w", s.Name, err)
			}
			if !ev.Passed {
				return fmt.Errorf("subkernel %q parity failed", s.Name)
			}
			physical = physical || ev.PhysicalParityCredit()
		}
		if !physical {
			return fmt.Errorf("subkernel %q lacks physical parity credit", s.Name)
		}
		if err := validExecutionEvidence(s.Evidence); err != nil {
			return fmt.Errorf("subkernel %q: %w", s.Name, err)
		}
		if err := r.crossBindEvidence(s.Evidence); err != nil {
			return fmt.Errorf("subkernel %q: %w", s.Name, err)
		}
	}
	for _, a := range r.Ablations {
		if a.Dimension == "" || a.Feature == "" || a.BaselineArm.Name == "" || a.CandidateArm.Name == "" || a.BaselineArm.LatencyUS <= 0 || a.CandidateArm.LatencyUS <= 0 || a.BaselineArm.Samples <= 0 || a.CandidateArm.Samples <= 0 || math.IsNaN(a.CosineParity) || math.IsInf(a.CosineParity, 0) || a.CosineParity < .999 || (a.Verdict != "VERIFIED_LIFT" && a.Verdict != "PARITY_MATCH") {
			return fmt.Errorf("ablation %q has incomplete or invalid evidence", a.Feature)
		}
		speed := float64(a.BaselineArm.LatencyUS) / float64(a.CandidateArm.LatencyUS)
		lift := speed
		if a.BaselineArm.ThroughputTokS > 0 && a.CandidateArm.ThroughputTokS > 0 {
			lift = a.CandidateArm.ThroughputTokS / a.BaselineArm.ThroughputTokS
		}
		if !withinRatioTolerance(a.Speedup, speed, .01) || !withinRatioTolerance(a.LiftRatio, lift, .01) {
			return fmt.Errorf("ablation %q contradicts recomputed speedup/lift", a.Feature)
		}
		if a.Verdict == "VERIFIED_LIFT" && speed <= 1 {
			return fmt.Errorf("ablation %q claims lift without faster candidate", a.Feature)
		}
		if a.Verdict == "PARITY_MATCH" && (speed < .99 || speed > 1) {
			return fmt.Errorf("ablation %q claims parity outside the [0.99, 1.00] latency ratio band", a.Feature)
		}
		if err := validExecutionEvidence(a.Evidence); err != nil {
			return fmt.Errorf("ablation %q: %w", a.Feature, err)
		}
		if err := r.crossBindEvidence(a.Evidence); err != nil {
			return fmt.Errorf("ablation %q: %w", a.Feature, err)
		}
	}
	if r.Provenance.ExecutionManifestSHA256 != executionManifestDigest(r) {
		return fmt.Errorf("execution command manifest mismatch")
	}
	return nil
}

// NewStrixValidationReceipt initializes a new receipt with default provenance.
func NewStrixValidationReceipt(target StrixTarget, gitRef, gitTip, command string) *StrixValidationReceipt {
	transport := "ssh"
	if target.Mode == "local" {
		transport = "local"
	}
	return &StrixValidationReceipt{
		Schema:    StrixValidationSchema,
		Timestamp: time.Now().UTC().Format(time.RFC3339),
		Verdict:   "PASS",
		Target:    target,
		Provenance: StrixProvenance{
			GitRef:      gitRef,
			GitTip:      gitTip,
			Command:     command,
			GeneratedBy: "fak/internal/amdgpu",
			Transport:   transport,
		},
		Subkernels:   make([]StrixSubkernelResult, 0),
		Ablations:    make([]StrixAblationResult, 0),
		ParityEvents: make([]StrixParityEvent, 0),
		Failures:     make([]string, 0),
		Verified:     true,
	}
}
