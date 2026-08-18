package disambiguation

import (
	"errors"
	"fmt"
	"strings"
)

const ClaimsSourceSelfTestSchemaVersion = "fak-disambiguation-claims-source-self-test/1"

var ErrClaimScopeMissing = errors.New("claim scope missing")
var ErrClaimBaselineMissing = errors.New("claim baseline missing")
var ErrClaimProvenanceMissing = errors.New("claim provenance missing")

type ClaimTermFixture struct {
	Term       string `json:"term"`
	Baseline   string `json:"baseline"`
	Provenance string `json:"provenance"`
	Scope      string `json:"scope"`
}

type ClaimsSourceSelfTestReport struct {
	Schema                    string   `json:"schema"`
	IndexVersion              string   `json:"index_version"`
	CanonicalTerms            []string `json:"canonical_terms"`
	MissingBaselineRejected   bool     `json:"missing_baseline_rejected"`
	MissingProvenanceRejected bool     `json:"missing_provenance_rejected"`
	MissingScopeRejected      bool     `json:"missing_scope_rejected"`
}

func ValidateClaimTermFixture(f ClaimTermFixture) error {
	if strings.TrimSpace(f.Baseline) == "" {
		return ErrClaimBaselineMissing
	}
	if strings.TrimSpace(f.Provenance) == "" {
		return ErrClaimProvenanceMissing
	}
	if strings.TrimSpace(f.Scope) == "" {
		return ErrClaimScopeMissing
	}
	if _, err := Resolve(f.Term); err != nil {
		return fmt.Errorf("resolve claim term: %w", err)
	}
	return nil
}

func RunClaimsSourceSelfTest() (ClaimsSourceSelfTestReport, error) {
	terms := []string{"naive baseline", "tuned baseline", "fak measurement arm", "witness provenance", "simulated evidence", "net-true claim"}
	report := ClaimsSourceSelfTestReport{Schema: ClaimsSourceSelfTestSchemaVersion, IndexVersion: PublicIndexVersion}
	for _, term := range terms {
		resolved, err := Resolve(term)
		if err != nil {
			return report, err
		}
		report.CanonicalTerms = append(report.CanonicalTerms, resolved.Entry.Identity.CanonicalTerm)
	}
	base := ClaimTermFixture{Term: "net-true claim", Baseline: "tuned baseline", Provenance: "WITNESSED", Scope: "same model, device, and concurrency"}
	missingBaseline := base
	missingBaseline.Baseline = ""
	missingProvenance := base
	missingProvenance.Provenance = ""
	missingScope := base
	missingScope.Scope = ""
	report.MissingBaselineRejected = errors.Is(ValidateClaimTermFixture(missingBaseline), ErrClaimBaselineMissing)
	report.MissingProvenanceRejected = errors.Is(ValidateClaimTermFixture(missingProvenance), ErrClaimProvenanceMissing)
	report.MissingScopeRejected = errors.Is(ValidateClaimTermFixture(missingScope), ErrClaimScopeMissing)
	if !report.MissingBaselineRejected || !report.MissingProvenanceRejected || !report.MissingScopeRejected {
		return report, errors.New("claim fixture admission failed")
	}
	return report, nil
}
