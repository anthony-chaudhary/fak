package mixedprecision

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"sort"
	"strings"
)

const SchemaV1 = "mixedprecision/v1"

type Outcome string

const (
	OutcomeSupported Outcome = "supported"
	OutcomeRefused   Outcome = "refused"
	OutcomeDelegate  Outcome = "delegate"
)

type ReasonCode string

const (
	ReasonSupported              ReasonCode = "supported"
	ReasonAcceptedFallback       ReasonCode = "accepted_fallback"
	ReasonRuntimeHandoff         ReasonCode = "runtime_handoff_required"
	ReasonInvalidDescriptor      ReasonCode = "invalid_descriptor"
	ReasonUnknownSchema          ReasonCode = "unknown_schema"
	ReasonUnknownArtifact        ReasonCode = "unknown_artifact"
	ReasonUnknownArtifactVersion ReasonCode = "unknown_artifact_version"
	ReasonUnknownRecipe          ReasonCode = "unknown_recipe"
	ReasonUnknownRecipeVersion   ReasonCode = "unknown_recipe_version"
	ReasonUnknownRuntime         ReasonCode = "unknown_runtime"
	ReasonUnknownRuntimeVersion  ReasonCode = "unknown_runtime_version"
	ReasonUndeclaredCombination  ReasonCode = "undeclared_combination"
	ReasonUnknownPrecision       ReasonCode = "unknown_precision"
	ReasonAmbiguousModule        ReasonCode = "ambiguous_module"
	ReasonUnmatchedModule        ReasonCode = "unmatched_module"
	ReasonInvalidEvidence        ReasonCode = "invalid_evidence"
)

type FallbackMode string

const (
	FallbackRefuse  FallbackMode = "refuse"
	FallbackHandoff FallbackMode = "handoff"
	FallbackAssign  FallbackMode = "assign"
)

type EvidenceKind string

const (
	EvidenceModeled  EvidenceKind = "modeled"
	EvidenceObserved EvidenceKind = "observed"
)

// PinnedRef identifies immutable input or implementation provenance. Version and
// SHA256 are both mandatory: a mutable tag or a digest without recipe semantics
// is not sufficient provenance.
type PinnedRef struct {
	ID      string `json:"id"`
	Version string `json:"version"`
	SHA256  string `json:"sha256"`
	URI     string `json:"uri,omitempty"`
}

type Provenance struct {
	Artifact PinnedRef `json:"artifact"`
	Recipe   PinnedRef `json:"recipe"`
	Runtime  PinnedRef `json:"runtime"`
}

type Module struct {
	Name       string `json:"name"`
	Parameters uint64 `json:"parameters"`
}

type Rule struct {
	// Pattern is either an exact module name or a prefix ending in '*'.
	// Overlapping matches are refused rather than resolved by hidden priority.
	Pattern   string `json:"pattern"`
	Precision string `json:"precision"`
}

type Fallback struct {
	Mode      FallbackMode `json:"mode"`
	Precision string       `json:"precision,omitempty"`
}

type HardwareEnvelope struct {
	Accelerator string `json:"accelerator"`
	Runtime     string `json:"runtime"`
	Driver      string `json:"driver"`
	OS          string `json:"os"`
}

// EvaluationEvidence keeps predictive/model evidence separate from observed
// hardware measurements. No values are synthesized by this package.
type EvaluationEvidence struct {
	Kind       EvidenceKind      `json:"kind"`
	Metric     string            `json:"metric"`
	Dataset    PinnedRef         `json:"dataset"`
	Value      float64           `json:"value"`
	Baseline   *float64          `json:"baseline,omitempty"`
	ModelBasis string            `json:"model_basis,omitempty"`
	Hardware   *HardwareEnvelope `json:"hardware,omitempty"`
	Samples    uint64            `json:"samples,omitempty"`
}

type Descriptor struct {
	Schema     string               `json:"schema"`
	Provenance Provenance           `json:"provenance"`
	Modules    []Module             `json:"modules"`
	Rules      []Rule               `json:"rules"`
	Fallback   Fallback             `json:"fallback"`
	Evidence   []EvaluationEvidence `json:"evidence,omitempty"`
}

type Combination struct {
	Artifact string  `json:"artifact"`
	Recipe   string  `json:"recipe"`
	Runtime  string  `json:"runtime"`
	Outcome  Outcome `json:"outcome"`
}

type Support struct {
	Artifacts    map[string][]string `json:"artifacts"`
	Recipes      map[string][]string `json:"recipes"`
	Runtimes     map[string][]string `json:"runtimes"`
	Precisions   []string            `json:"precisions"`
	Combinations []Combination       `json:"combinations"`
}

type Assignment struct {
	Module     string `json:"module"`
	Precision  string `json:"precision"`
	Rule       string `json:"rule,omitempty"`
	Fallback   bool   `json:"fallback"`
	Parameters uint64 `json:"parameters"`
}

type Budget struct {
	ModulesTotal       int     `json:"modules_total"`
	ModulesMatched     int     `json:"modules_matched"`
	ModulesFallback    int     `json:"modules_fallback"`
	ParametersTotal    uint64  `json:"parameters_total"`
	ParametersMatched  uint64  `json:"parameters_matched"`
	ParametersFallback uint64  `json:"parameters_fallback"`
	WeightedBits       uint64  `json:"weighted_bits"`
	AverageBits        float64 `json:"average_bits"`
	Coverage           float64 `json:"coverage"`
}

type Result struct {
	Outcome     Outcome              `json:"outcome"`
	Reason      ReasonCode           `json:"reason"`
	Detail      string               `json:"detail,omitempty"`
	Assignments []Assignment         `json:"assignments,omitempty"`
	Budget      Budget               `json:"budget"`
	CanonicalID string               `json:"canonical_id,omitempty"`
	Provenance  Provenance           `json:"provenance"`
	Evidence    []EvaluationEvidence `json:"evidence,omitempty"`
}

func Evaluate(d Descriptor, support Support) Result {
	r := Result{Outcome: OutcomeRefused, Provenance: d.Provenance}
	if d.Schema != SchemaV1 {
		r.Reason, r.Detail = ReasonUnknownSchema, d.Schema
		return r
	}
	if reason, detail := validateRef(d.Provenance.Artifact, support.Artifacts, ReasonUnknownArtifact, ReasonUnknownArtifactVersion); reason != "" {
		r.Reason, r.Detail = reason, detail
		return r
	}
	if reason, detail := validateRef(d.Provenance.Recipe, support.Recipes, ReasonUnknownRecipe, ReasonUnknownRecipeVersion); reason != "" {
		r.Reason, r.Detail = reason, detail
		return r
	}
	if reason, detail := validateRef(d.Provenance.Runtime, support.Runtimes, ReasonUnknownRuntime, ReasonUnknownRuntimeVersion); reason != "" {
		r.Reason, r.Detail = reason, detail
		return r
	}
	combo := combinationOutcome(d.Provenance, support.Combinations)
	if combo == "" {
		r.Reason = ReasonUndeclaredCombination
		return r
	}
	if detail := validateEvidence(d.Evidence); detail != "" {
		r.Reason, r.Detail = ReasonInvalidEvidence, detail
		return r
	}
	precisions := stringSet(support.Precisions)
	seen := map[string]bool{}
	modules := append([]Module(nil), d.Modules...)
	sort.Slice(modules, func(i, j int) bool { return modules[i].Name < modules[j].Name })
	if len(modules) == 0 || len(d.Rules) == 0 {
		r.Reason, r.Detail = ReasonInvalidDescriptor, "modules/rules"
		return r
	}
	for _, m := range modules {
		name := canonicalName(m.Name)
		if name == "" || m.Parameters == 0 || seen[name] {
			r.Reason, r.Detail = ReasonInvalidDescriptor, m.Name
			return r
		}
		seen[name] = true
		r.Budget.ModulesTotal++
		r.Budget.ParametersTotal += m.Parameters
		var matches []Rule
		for _, rule := range d.Rules {
			if match(canonicalName(rule.Pattern), name) {
				matches = append(matches, rule)
			}
		}
		if len(matches) > 1 {
			r.Reason, r.Detail = ReasonAmbiguousModule, name
			return r
		}
		assignment := Assignment{Module: name, Parameters: m.Parameters}
		if len(matches) == 0 {
			switch d.Fallback.Mode {
			case FallbackHandoff:
				r.Outcome, r.Reason, r.Detail = OutcomeDelegate, ReasonUnmatchedModule, name
				return r
			case FallbackAssign:
				assignment.Precision, assignment.Fallback = canonicalPrecision(d.Fallback.Precision), true
				r.Budget.ModulesFallback++
				r.Budget.ParametersFallback += m.Parameters
			case FallbackRefuse:
				r.Reason, r.Detail = ReasonUnmatchedModule, name
				return r
			default:
				r.Reason, r.Detail = ReasonInvalidDescriptor, "fallback.mode"
				return r
			}
		} else {
			assignment.Precision = canonicalPrecision(matches[0].Precision)
			assignment.Rule = canonicalName(matches[0].Pattern)
			r.Budget.ModulesMatched++
			r.Budget.ParametersMatched += m.Parameters
		}
		bits, ok := precisionBits(assignment.Precision)
		if !ok || !precisions[assignment.Precision] {
			r.Reason, r.Detail = ReasonUnknownPrecision, assignment.Precision
			return r
		}
		if m.Parameters > ^uint64(0)/bits || r.Budget.WeightedBits > ^uint64(0)-(m.Parameters*bits) {
			r.Reason, r.Detail = ReasonInvalidDescriptor, "budget_overflow"
			return r
		}
		r.Budget.WeightedBits += m.Parameters * bits
		r.Assignments = append(r.Assignments, assignment)
	}
	if r.Budget.ParametersTotal > 0 {
		r.Budget.AverageBits = float64(r.Budget.WeightedBits) / float64(r.Budget.ParametersTotal)
		r.Budget.Coverage = float64(r.Budget.ParametersMatched) / float64(r.Budget.ParametersTotal)
	}
	r.Evidence = append([]EvaluationEvidence(nil), d.Evidence...)
	r.Outcome = combo
	if combo == OutcomeDelegate {
		r.Reason = ReasonRuntimeHandoff
	} else if r.Budget.ModulesFallback > 0 {
		r.Reason = ReasonAcceptedFallback
	} else {
		r.Reason = ReasonSupported
	}
	r.CanonicalID = canonicalID(r)
	return r
}

func ParseAndEvaluate(raw []byte, support Support) (Result, error) {
	var d Descriptor
	dec := json.NewDecoder(strings.NewReader(string(raw)))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&d); err != nil {
		return Result{Outcome: OutcomeRefused, Reason: ReasonInvalidDescriptor, Detail: err.Error()}, err
	}
	if err := dec.Decode(&struct{}{}); err != io.EOF {
		if err == nil {
			err = fmt.Errorf("multiple JSON values")
		}
		return Result{Outcome: OutcomeRefused, Reason: ReasonInvalidDescriptor, Detail: err.Error()}, err
	}
	return Evaluate(d, support), nil
}

func validateRef(ref PinnedRef, supported map[string][]string, unknown, unknownVersion ReasonCode) (ReasonCode, string) {
	if ref.ID == "" || ref.Version == "" || len(ref.SHA256) != 64 {
		return ReasonInvalidDescriptor, ref.ID
	}
	if _, err := hex.DecodeString(ref.SHA256); err != nil {
		return ReasonInvalidDescriptor, ref.ID
	}
	versions, ok := supported[ref.ID]
	if !ok {
		return unknown, ref.ID
	}
	for _, v := range versions {
		if v == ref.Version {
			return "", ""
		}
	}
	return unknownVersion, ref.ID + "@" + ref.Version
}

func combinationOutcome(p Provenance, combos []Combination) Outcome {
	key := func(r PinnedRef) string { return r.ID + "@" + r.Version }
	for _, c := range combos {
		if c.Artifact == key(p.Artifact) && c.Recipe == key(p.Recipe) && c.Runtime == key(p.Runtime) && (c.Outcome == OutcomeSupported || c.Outcome == OutcomeDelegate) {
			return c.Outcome
		}
	}
	return ""
}

func validateEvidence(all []EvaluationEvidence) string {
	for i, e := range all {
		if e.Metric == "" || e.Dataset.ID == "" || e.Dataset.Version == "" || len(e.Dataset.SHA256) != 64 {
			return fmt.Sprintf("evidence[%d]", i)
		}
		if _, err := hex.DecodeString(e.Dataset.SHA256); err != nil {
			return fmt.Sprintf("evidence[%d].dataset", i)
		}
		switch e.Kind {
		case EvidenceModeled:
			if e.ModelBasis == "" || e.Hardware != nil || e.Samples != 0 {
				return fmt.Sprintf("evidence[%d].modeled", i)
			}
		case EvidenceObserved:
			if e.ModelBasis != "" || e.Hardware == nil || e.Hardware.Accelerator == "" || e.Hardware.Runtime == "" || e.Hardware.Driver == "" || e.Hardware.OS == "" || e.Samples == 0 {
				return fmt.Sprintf("evidence[%d].observed", i)
			}
		default:
			return fmt.Sprintf("evidence[%d].kind", i)
		}
	}
	return ""
}

func canonicalName(s string) string      { return strings.ToLower(strings.TrimSpace(s)) }
func canonicalPrecision(s string) string { return strings.ToLower(strings.TrimSpace(s)) }
func match(pattern, name string) bool {
	if pattern == "" {
		return false
	}
	if strings.HasSuffix(pattern, "*") {
		return strings.HasPrefix(name, strings.TrimSuffix(pattern, "*"))
	}
	return pattern == name
}
func stringSet(values []string) map[string]bool {
	m := map[string]bool{}
	for _, v := range values {
		m[canonicalPrecision(v)] = true
	}
	return m
}
func precisionBits(p string) (uint64, bool) {
	switch canonicalPrecision(p) {
	case "fp32":
		return 32, true
	case "fp16", "bf16":
		return 16, true
	case "fp8", "int8":
		return 8, true
	case "int6":
		return 6, true
	case "int4", "nf4", "fp4":
		return 4, true
	case "int3":
		return 3, true
	case "int2":
		return 2, true
	default:
		return 0, false
	}
}
func canonicalID(r Result) string {
	clone := struct {
		Schema      string       `json:"schema"`
		Provenance  Provenance   `json:"provenance"`
		Assignments []Assignment `json:"assignments"`
		Budget      Budget       `json:"budget"`
	}{Schema: SchemaV1, Provenance: r.Provenance, Assignments: r.Assignments, Budget: r.Budget}
	b, _ := json.Marshal(clone)
	sum := sha256.Sum256(b)
	return "sha256:" + hex.EncodeToString(sum[:])
}
