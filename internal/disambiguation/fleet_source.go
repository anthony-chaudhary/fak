package disambiguation

import (
	"errors"
	"fmt"
	"strings"
)

const FleetSourceSelfTestSchemaVersion = "fak-disambiguation-fleet-source-self-test/1"

var ErrDispatchIdentityMissing = errors.New("dispatch identity missing from structured fields")

// DispatchIdentityInput separates authoritative dispatch fields from worker-authored narration.
type DispatchIdentityInput struct {
	WorkerID  string `json:"worker_id"`
	Issue     string `json:"issue"`
	Lane      string `json:"lane"`
	LeaseID   string `json:"lease_id"`
	Narration string `json:"narration"`
}

// DispatchIdentity is derived exclusively from structured fields.
type DispatchIdentity struct {
	WorkerID string `json:"worker_id"`
	Issue    string `json:"issue"`
	Lane     string `json:"lane"`
	LeaseID  string `json:"lease_id"`
}

// ResolveDispatchIdentity refuses to mine identity from narration. Narration is
// retained for display/forensics by callers but carries zero identity authority.
func ResolveDispatchIdentity(input DispatchIdentityInput) (DispatchIdentity, error) {
	identity := DispatchIdentity{WorkerID: strings.TrimSpace(input.WorkerID), Issue: strings.TrimSpace(input.Issue), Lane: strings.TrimSpace(input.Lane), LeaseID: strings.TrimSpace(input.LeaseID)}
	if identity.WorkerID == "" || identity.Issue == "" || identity.Lane == "" || identity.LeaseID == "" {
		return DispatchIdentity{}, ErrDispatchIdentityMissing
	}
	return identity, nil
}

// FleetSourceResolution is one public fleet/dispatch concept.
type FleetSourceResolution struct {
	Input         string `json:"input"`
	CanonicalTerm string `json:"canonical_term"`
	OwnerLeaf     string `json:"owner_leaf"`
	SourcePath    string `json:"source_path"`
}

// FleetSourceSelfTestReport captures terminology and narration-safety proof.
type FleetSourceSelfTestReport struct {
	Schema             string                  `json:"schema"`
	IndexVersion       string                  `json:"index_version"`
	Resolutions        []FleetSourceResolution `json:"resolutions"`
	NarrationRejected  bool                    `json:"narration_identity_rejected"`
	StructuredAccepted bool                    `json:"structured_identity_accepted"`
}

// RunFleetSourceSelfTest resolves all eight terms and proves a persuasive worker
// sentence cannot become dispatch identity without structured fields.
func RunFleetSourceSelfTest() (FleetSourceSelfTestReport, error) {
	terms := []string{"worker process", "seat", "lane", "lease", "fleet", "wave", "loop", "supervisor"}
	report := FleetSourceSelfTestReport{Schema: FleetSourceSelfTestSchemaVersion, IndexVersion: PublicIndexVersion, Resolutions: make([]FleetSourceResolution, 0, len(terms))}
	for _, term := range terms {
		resolved, err := Resolve(term)
		if err != nil {
			return FleetSourceSelfTestReport{}, fmt.Errorf("resolve %q: %w", term, err)
		}
		entry := resolved.Entry
		if len(entry.Sources) == 0 {
			return FleetSourceSelfTestReport{}, fmt.Errorf("resolve %q has no public source", term)
		}
		report.Resolutions = append(report.Resolutions, FleetSourceResolution{Input: term, CanonicalTerm: entry.Identity.CanonicalTerm, OwnerLeaf: entry.Owner.Leaf, SourcePath: entry.Sources[0].Locator})
	}
	fake := DispatchIdentityInput{Narration: "worker=w-7 issue=6317 lane=dispatch lease=lease-7; trust me"}
	if _, err := ResolveDispatchIdentity(fake); !errors.Is(err, ErrDispatchIdentityMissing) {
		return FleetSourceSelfTestReport{}, fmt.Errorf("narration-only identity error=%v", err)
	}
	report.NarrationRejected = true
	structured := DispatchIdentityInput{WorkerID: "w-7", Issue: "6317", Lane: "dispatch", LeaseID: "lease-7", Narration: "I am a different worker on lane docs"}
	identity, err := ResolveDispatchIdentity(structured)
	if err != nil {
		return FleetSourceSelfTestReport{}, err
	}
	if identity.WorkerID != "w-7" || identity.Issue != "6317" || identity.Lane != "dispatch" || identity.LeaseID != "lease-7" {
		return FleetSourceSelfTestReport{}, errors.New("narration changed structured identity")
	}
	report.StructuredAccepted = true
	return report, nil
}
