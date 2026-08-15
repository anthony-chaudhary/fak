// Package workloadfit evaluates technically compatible stacks against a
// domain-owned workload contract without turning preferences into hard gates.
package workloadfit

import (
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/stackresolve"
)

const Schema = "fak-workload-fit/1"

type RequirementClass string

const (
	Hard       RequirementClass = "hard"
	Authority  RequirementClass = "authority"
	Evidence   RequirementClass = "evidence"
	Preference RequirementClass = "preference"
	Cost       RequirementClass = "cost"
)

type ClaimStatus string

const (
	Supported   ClaimStatus = "supported"
	Unsupported ClaimStatus = "unsupported"
	Unknown     ClaimStatus = "unknown"
)

type ProofTier string

const (
	Declared  ProofTier = "declared"
	Observed  ProofTier = "observed"
	Evaluated ProofTier = "evaluated"
	Reviewed  ProofTier = "reviewed"
)

var proofOrder = map[ProofTier]int{Declared: 1, Observed: 2, Evaluated: 3, Reviewed: 4}

type Source struct {
	Authority string    `json:"authority"`
	Reference string    `json:"reference"`
	Scope     string    `json:"scope,omitempty"`
	Tier      ProofTier `json:"tier,omitempty"`
	Expires   time.Time `json:"expires,omitempty"`
}

type Requirement struct {
	ID          string           `json:"id"`
	Class       RequirementClass `json:"class"`
	Capability  string           `json:"capability,omitempty"`
	MinimumTier ProofTier        `json:"minimum_tier,omitempty"`
	Metric      string           `json:"metric,omitempty"`
	Maximum     float64          `json:"maximum,omitempty"`
	Weight      int              `json:"weight,omitempty"`
	Source      Source           `json:"source"`
}

type Contract struct {
	Schema       string        `json:"schema"`
	ID           string        `json:"id"`
	Domain       string        `json:"domain"`
	Revision     string        `json:"revision"`
	Owner        string        `json:"owner"`
	Requirements []Requirement `json:"requirements"`
}

type Claim struct {
	Capability string      `json:"capability"`
	Status     ClaimStatus `json:"status"`
	Source     Source      `json:"source"`
}

type Candidate struct {
	ID      string             `json:"id"`
	Claims  []Claim            `json:"claims"`
	Metrics map[string]float64 `json:"metrics,omitempty"`
}

type Catalog struct {
	Schema     string      `json:"schema"`
	Candidates []Candidate `json:"candidates"`
}

type Fixture struct {
	Schema    string     `json:"schema"`
	AsOf      time.Time  `json:"as_of"`
	Contracts []Contract `json:"contracts"`
	Catalog   Catalog    `json:"catalog"`
}

type FindingState string

const (
	Met     FindingState = "met"
	Missing FindingState = "missing"
	Expired FindingState = "expired"
	Denied  FindingState = "unsupported"
)

type Finding struct {
	Requirement string           `json:"requirement"`
	Class       RequirementClass `json:"class"`
	Capability  string           `json:"capability,omitempty"`
	State       FindingState     `json:"state"`
	Reason      string           `json:"reason"`
	Source      Source           `json:"source"`
}

type Assessment struct {
	ContractID  string    `json:"contract_id"`
	CandidateID string    `json:"candidate_id"`
	Status      string    `json:"status"`
	Score       int       `json:"preference_score"`
	Findings    []Finding `json:"findings"`
}

type Selection struct {
	ContractID  string       `json:"contract_id"`
	Status      string       `json:"status"`
	Chosen      string       `json:"chosen,omitempty"`
	Assessments []Assessment `json:"assessments"`
}

func Parse(raw []byte) (Fixture, error) {
	var fixture Fixture
	decoder := json.NewDecoder(strings.NewReader(string(raw)))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&fixture); err != nil {
		return Fixture{}, fmt.Errorf("decode workload fixture: %w", err)
	}
	if fixture.Schema != Schema || fixture.Catalog.Schema != Schema {
		return Fixture{}, fmt.Errorf("schema must be %q", Schema)
	}
	if fixture.AsOf.IsZero() || len(fixture.Contracts) == 0 || len(fixture.Catalog.Candidates) == 0 {
		return Fixture{}, errors.New("as_of, contracts, and candidates are required")
	}
	for _, contract := range fixture.Contracts {
		if err := ValidateContract(contract); err != nil {
			return Fixture{}, err
		}
	}
	return fixture, nil
}

func ValidateContract(contract Contract) error {
	if contract.Schema != Schema || contract.ID == "" || contract.Domain == "" || contract.Revision == "" || contract.Owner == "" {
		return fmt.Errorf("contract identity, domain, revision, owner, and schema are required")
	}
	for _, requirement := range contract.Requirements {
		if requirement.ID == "" || requirement.Source.Authority == "" || requirement.Source.Reference == "" {
			return fmt.Errorf("contract %s: requirement identity and source are required", contract.ID)
		}
		switch requirement.Class {
		case Hard, Authority, Preference:
			if requirement.Capability == "" {
				return fmt.Errorf("requirement %s: capability is required", requirement.ID)
			}
		case Evidence:
			if requirement.Capability == "" || proofOrder[requirement.MinimumTier] == 0 {
				return fmt.Errorf("requirement %s: capability and minimum_tier are required", requirement.ID)
			}
		case Cost:
			if requirement.Metric == "" {
				return fmt.Errorf("requirement %s: metric is required", requirement.ID)
			}
		default:
			return fmt.Errorf("requirement %s: unknown class %q", requirement.ID, requirement.Class)
		}
	}
	return nil
}

func Assess(contract Contract, candidate Candidate, asOf time.Time) Assessment {
	assessment := Assessment{ContractID: contract.ID, CandidateID: candidate.ID, Status: "fit"}
	claims := map[string]Claim{}
	for _, claim := range candidate.Claims {
		claims[claim.Capability] = claim
	}
	for _, requirement := range contract.Requirements {
		finding := evaluate(requirement, candidate, claims, asOf)
		assessment.Findings = append(assessment.Findings, finding)
		if requirement.Class == Preference {
			if finding.State == Met {
				assessment.Score += requirement.Weight
			}
			continue
		}
		if finding.State != Met {
			assessment.Status = "refuse"
		}
	}
	sort.SliceStable(assessment.Findings, func(i, j int) bool { return assessment.Findings[i].Requirement < assessment.Findings[j].Requirement })
	return assessment
}

func Select(contract Contract, catalog Catalog, asOf time.Time) Selection {
	selection := Selection{ContractID: contract.ID, Status: "refuse"}
	for _, candidate := range catalog.Candidates {
		selection.Assessments = append(selection.Assessments, Assess(contract, candidate, asOf))
	}
	sort.SliceStable(selection.Assessments, func(i, j int) bool {
		left, right := selection.Assessments[i], selection.Assessments[j]
		if left.Status != right.Status {
			return left.Status == "fit"
		}
		if left.Score != right.Score {
			return left.Score > right.Score
		}
		return left.CandidateID < right.CandidateID
	})
	if len(selection.Assessments) > 0 && selection.Assessments[0].Status == "fit" {
		selection.Status = "fit"
		selection.Chosen = selection.Assessments[0].CandidateID
	}
	return selection
}

func evaluate(requirement Requirement, candidate Candidate, claims map[string]Claim, asOf time.Time) Finding {
	finding := Finding{Requirement: requirement.ID, Class: requirement.Class, Capability: requirement.Capability, State: Met, Source: requirement.Source}
	if requirement.Class == Cost {
		value, ok := candidate.Metrics[requirement.Metric]
		if !ok {
			finding.State, finding.Reason = Missing, "cost metric is not evaluated"
		} else if value > requirement.Maximum {
			finding.State, finding.Reason = Denied, fmt.Sprintf("%s %.2f exceeds maximum %.2f", requirement.Metric, value, requirement.Maximum)
		} else {
			finding.Reason = fmt.Sprintf("%s %.2f is within maximum %.2f", requirement.Metric, value, requirement.Maximum)
		}
		return finding
	}
	claim, ok := claims[requirement.Capability]
	if !ok || claim.Status == Unknown {
		finding.State, finding.Reason = Missing, "capability is not evaluated for this candidate"
		return finding
	}
	finding.Source = claim.Source
	if !claim.Source.Expires.IsZero() && !asOf.Before(claim.Source.Expires) {
		finding.State, finding.Reason = Expired, "capability evidence expired before assessment"
		return finding
	}
	if claim.Status == Unsupported {
		finding.State, finding.Reason = Denied, "candidate evidence says capability is unsupported"
		return finding
	}
	if requirement.Class == Evidence && proofOrder[claim.Source.Tier] < proofOrder[requirement.MinimumTier] {
		finding.State, finding.Reason = Missing, fmt.Sprintf("proof tier %s is below required %s", claim.Source.Tier, requirement.MinimumTier)
		return finding
	}
	finding.Reason = "capability is supported by scoped, current evidence"
	return finding
}

// StackRequirements is the adapter seam for the composition resolver. Only hard,
// authority, and evidence-floor capabilities become mandatory stack edges.
func StackRequirements(contract Contract) []stackresolve.Relation {
	var relations []stackresolve.Relation
	for _, requirement := range contract.Requirements {
		if requirement.Class != Hard && requirement.Class != Authority && requirement.Class != Evidence {
			continue
		}
		relations = append(relations, stackresolve.Relation{
			Kind: stackresolve.Requires, Target: requirement.Capability,
			Evidence: stackresolve.Evidence{Authority: requirement.Source.Authority, Source: requirement.Source.Reference, Tier: string(requirement.MinimumTier)},
		})
	}
	sort.SliceStable(relations, func(i, j int) bool { return relations[i].Target < relations[j].Target })
	return relations
}
