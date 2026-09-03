package nativebench

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"strings"
)

// TreatmentTopology defines the topology mode for a benchmark arm or contract.
// Additive treatments evaluate marginal additions layered on an incumbent stack.
// Replacement treatments evaluate complete component substitutions.
type TreatmentTopology string

const (
	AdditiveTreatment    TreatmentTopology = "additive"
	ReplacementTreatment TreatmentTopology = "replacement"
)

// ErrTopologyMismatch is returned when an additive treatment is compared against a
// replacement-only baseline (or vice versa).
var ErrTopologyMismatch = errors.New("TOPOLOGY_MISMATCH: cannot evaluate additive treatment against replacement-only baseline")

// ValidateTreatmentTopology checks that candidate and baseline topologies are compatible.
// Matched pairs (additive vs additive, replacement vs replacement) are valid.
// Mismatched pairs return ErrTopologyMismatch.
func ValidateTreatmentTopology(candidate, baseline TreatmentTopology) error {
	switch candidate {
	case AdditiveTreatment, ReplacementTreatment:
	default:
		return fmt.Errorf("invalid candidate treatment topology %q", candidate)
	}
	switch baseline {
	case AdditiveTreatment, ReplacementTreatment:
	default:
		return fmt.Errorf("invalid baseline treatment topology %q", baseline)
	}
	if candidate != baseline {
		return ErrTopologyMismatch
	}
	return nil
}

// CanonicalIdentityDigest derives a reproducible content address for a treatment receipt,
// binding the treatment topology, baseline composition, and native implementation path.
// It changes whenever topology or baseline composition changes so different treatments
// cannot have colliding receipt IDs.
func CanonicalIdentityDigest(topology TreatmentTopology, composition string, nativePath string) string {
	h := sha256.New()
	h.Write([]byte(topology))
	h.Write([]byte{0})
	h.Write([]byte(composition))
	h.Write([]byte{0})
	h.Write([]byte(nativePath))
	sum := h.Sum(nil)
	return "sha256:" + hex.EncodeToString(sum)
}

// CandidateArm specifies the candidate implementation arm under benchmark evaluation.
type CandidateArm struct {
	Name              string            `json:"name"`
	CandidateTopology TreatmentTopology `json:"candidate_topology"`
	NativePath        string            `json:"native_path,omitempty"`
}

// BaselineArm specifies the incumbent baseline stack arm under comparison.
type BaselineArm struct {
	Name                string            `json:"name"`
	BaselineTopology    TreatmentTopology `json:"baseline_topology"`
	BaselineComposition string            `json:"baseline_composition"`
}

// Comparison represents an evaluation pair between a candidate arm and a baseline arm.
type Comparison struct {
	Capability        string            `json:"capability,omitempty"`
	Candidate         CandidateArm      `json:"candidate"`
	Baseline          BaselineArm       `json:"baseline"`
	TreatmentTopology TreatmentTopology `json:"treatment_topology,omitempty"`
}

// ComparisonReport captures the verification result, canonical receipt digest, and topology details.
type ComparisonReport struct {
	Capability          string            `json:"capability,omitempty"`
	CandidateArm        string            `json:"candidate_arm"`
	CandidateTopology   TreatmentTopology `json:"candidate_topology"`
	BaselineArm         string            `json:"baseline_arm"`
	BaselineTopology    TreatmentTopology `json:"baseline_topology"`
	BaselineComposition string            `json:"baseline_composition"`
	TreatmentTopology   TreatmentTopology `json:"treatment_topology"`
	CanonicalDigest     string            `json:"canonical_digest,omitempty"`
	Valid               bool              `json:"valid"`
	Reason              string            `json:"reason,omitempty"`
}

// Compare validates the candidate and baseline arms, returning a ComparisonReport and any validation error.
func Compare(candidate CandidateArm, baseline BaselineArm) (ComparisonReport, error) {
	err := ValidateTreatmentTopology(candidate.CandidateTopology, baseline.BaselineTopology)
	report := ComparisonReport{
		CandidateArm:        candidate.Name,
		CandidateTopology:   candidate.CandidateTopology,
		BaselineArm:         baseline.Name,
		BaselineTopology:    baseline.BaselineTopology,
		BaselineComposition: baseline.BaselineComposition,
		TreatmentTopology:   candidate.CandidateTopology,
		Valid:               err == nil,
	}
	if err != nil {
		report.Reason = err.Error()
		return report, err
	}
	report.CanonicalDigest = CanonicalIdentityDigest(candidate.CandidateTopology, baseline.BaselineComposition, candidate.NativePath)
	return report, nil
}

// Human returns a human-readable rendering of the comparison report.
func (r ComparisonReport) Human() string {
	status := "VALID"
	if !r.Valid {
		status = "INVALID"
	}
	var b strings.Builder
	fmt.Fprintf(&b, "%s comparison: treatment_topology=%s\n", status, r.TreatmentTopology)
	fmt.Fprintf(&b, "  candidate: %s (topology=%s)\n", r.CandidateArm, r.CandidateTopology)
	fmt.Fprintf(&b, "  baseline:  %s (topology=%s, composition=%s)\n", r.BaselineArm, r.BaselineTopology, r.BaselineComposition)
	if r.CanonicalDigest != "" {
		fmt.Fprintf(&b, "  digest:    %s\n", r.CanonicalDigest)
	}
	if r.Reason != "" {
		fmt.Fprintf(&b, "  reason:    %s\n", r.Reason)
	}
	return strings.TrimSpace(b.String())
}

// String implements fmt.Stringer for ComparisonReport.
func (r ComparisonReport) String() string {
	return r.Human()
}
