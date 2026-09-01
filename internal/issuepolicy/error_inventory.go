package issuepolicy

import (
	"fmt"
	"sort"
	"strings"
	"time"
)

const ErrorInventorySchema = "fak.issue-error-inventory/1"

type ErrorDisposition string

const (
	DispositionActionable        ErrorDisposition = "ACTIONABLE"
	DispositionPossibleDuplicate ErrorDisposition = "POSSIBLE_DUPLICATE"
	DispositionDuplicateProven   ErrorDisposition = "DUPLICATE_PROVEN"
	DispositionFixPresentTrunk   ErrorDisposition = "FIX_PRESENT_TRUNK"
	DispositionFixReleased       ErrorDisposition = "FIX_RELEASED"
	DispositionReproRequired     ErrorDisposition = "REPRO_REQUIRED"
	DispositionStaleEvidence     ErrorDisposition = "STALE_EVIDENCE"
)

type ErrorEvidence struct {
	Status        string `json:"status,omitempty"`
	Fingerprint   string `json:"fingerprint,omitempty"`
	Module        string `json:"module,omitempty"`
	FailureClass  string `json:"failure_class,omitempty"`
	Ref           string `json:"ref,omitempty"`
	Commit        string `json:"commit,omitempty"`
	ModuleVersion string `json:"module_version,omitempty"`
	Witness       string `json:"witness,omitempty"`
	ObservedAt    string `json:"observed_at,omitempty"`
	Stale         bool   `json:"stale,omitempty"`
}

type ReleaseEvidence struct {
	Tag            string `json:"tag,omitempty"`
	FixAncestor    bool   `json:"fix_ancestor,omitempty"`
	PassingReceipt bool   `json:"passing_receipt,omitempty"`
	ModuleVersion  string `json:"module_version,omitempty"`
	Witness        string `json:"witness,omitempty"`
}

type ErrorObservation struct {
	Issue             int               `json:"issue"`
	Observed          ErrorEvidence     `json:"observed_failure"`
	Fix               ErrorEvidence     `json:"proven_fix,omitempty"`
	Current           ErrorEvidence     `json:"current_tested_state"`
	Releases          []ReleaseEvidence `json:"releases,omitempty"`
	PossibleDuplicate int               `json:"possible_duplicate,omitempty"`
	ScopeException    string            `json:"scope_exception,omitempty"`
}

type DuplicateEvidence struct {
	Issue      int     `json:"issue"`
	Kind       string  `json:"kind"`
	Similarity float64 `json:"similarity,omitempty"`
}

type ErrorInventoryIssue struct {
	Issue                  int                 `json:"issue"`
	CanonicalIssue         int                 `json:"canonical_issue"`
	Disposition            ErrorDisposition    `json:"disposition"`
	ReasonCodes            []string            `json:"reason_codes"`
	DuplicateEvidence      []DuplicateEvidence `json:"duplicate_evidence,omitempty"`
	ObservedFailure        ErrorEvidence       `json:"observed_failure"`
	ProvenFix              ErrorEvidence       `json:"proven_fix,omitempty"`
	CurrentTestedState     ErrorEvidence       `json:"current_tested_state"`
	FirstContainingRelease *ReleaseEvidence    `json:"first_containing_release,omitempty"`
	ScopeException         string              `json:"scope_exception,omitempty"`
}

type ErrorInventoryInput struct {
	GeneratedAt    time.Time
	SnapshotDigest string
	Observations   []ErrorObservation
}

type ErrorInventory struct {
	Schema         string                `json:"schema"`
	GeneratedAt    string                `json:"generated_at"`
	SnapshotDigest string                `json:"snapshot_digest"`
	Issues         []ErrorInventoryIssue `json:"issues"`
}

func BuildErrorInventory(in ErrorInventoryInput) (ErrorInventory, error) {
	if in.GeneratedAt.IsZero() {
		return ErrorInventory{}, fmt.Errorf("generated time is required")
	}
	if strings.TrimSpace(in.SnapshotDigest) == "" {
		return ErrorInventory{}, fmt.Errorf("snapshot digest is required")
	}
	byIdentity := make(map[string][]int)
	seen := make(map[int]bool)
	for _, obs := range in.Observations {
		if obs.Issue <= 0 || seen[obs.Issue] {
			return ErrorInventory{}, fmt.Errorf("observation issue numbers must be positive and unique")
		}
		seen[obs.Issue] = true
		if key := failureIdentity(obs.Observed); key != "" {
			byIdentity[key] = append(byIdentity[key], obs.Issue)
		}
	}
	for key := range byIdentity {
		sort.Ints(byIdentity[key])
	}
	out := ErrorInventory{Schema: ErrorInventorySchema, GeneratedAt: in.GeneratedAt.UTC().Format(time.RFC3339), SnapshotDigest: in.SnapshotDigest}
	for _, obs := range in.Observations {
		out.Issues = append(out.Issues, classifyErrorObservation(obs, byIdentity))
	}
	sort.Slice(out.Issues, func(i, j int) bool { return out.Issues[i].Issue < out.Issues[j].Issue })
	return out, nil
}

func classifyErrorObservation(obs ErrorObservation, identities map[string][]int) ErrorInventoryIssue {
	row := ErrorInventoryIssue{Issue: obs.Issue, CanonicalIssue: obs.Issue, ObservedFailure: obs.Observed, ProvenFix: obs.Fix, CurrentTestedState: obs.Current, ScopeException: obs.ScopeException}
	if evidenceStale(obs.Observed) || evidenceStale(obs.Fix) || evidenceStale(obs.Current) {
		row.Disposition = DispositionStaleEvidence
		row.ReasonCodes = []string{"STALE_RECEIPT"}
		return row
	}
	if !isFailure(obs.Observed.Status) || strings.TrimSpace(obs.Observed.Witness) == "" {
		row.Disposition = DispositionReproRequired
		row.ReasonCodes = []string{"OBSERVED_FAILURE_REQUIRED"}
		return row
	}
	if key := failureIdentity(obs.Observed); key != "" && len(identities[key]) > 1 && identities[key][0] != obs.Issue {
		row.CanonicalIssue = identities[key][0]
		row.Disposition = DispositionDuplicateProven
		row.ReasonCodes = []string{"STABLE_FAILURE_IDENTITY_MATCH"}
		row.DuplicateEvidence = []DuplicateEvidence{{Issue: row.CanonicalIssue, Kind: "stable_identity"}}
		return row
	}
	if isFailure(obs.Current.Status) && strings.TrimSpace(obs.Current.Witness) != "" && sameFailure(obs.Observed, obs.Current) {
		row.Disposition = DispositionActionable
		row.ReasonCodes = []string{"CURRENT_REGRESSION_REPRODUCED"}
		return row
	}
	if isPassing(obs.Current.Status) && strings.TrimSpace(obs.Current.Witness) != "" && strings.TrimSpace(obs.Fix.Commit) != "" && strings.TrimSpace(obs.Fix.Witness) != "" {
		for i := range obs.Releases {
			rel := obs.Releases[i]
			if strings.TrimSpace(rel.Witness) != "" && (rel.FixAncestor || rel.PassingReceipt) {
				row.Disposition = DispositionFixReleased
				if rel.FixAncestor {
					row.ReasonCodes = []string{"RELEASE_TAG_CONTAINS_FIX"}
				} else {
					row.ReasonCodes = []string{"RELEASE_PASSING_RECEIPT"}
				}
				row.FirstContainingRelease = &rel
				return row
			}
		}
		row.Disposition = DispositionFixPresentTrunk
		row.ReasonCodes = []string{"TRUNK_PASS_AFTER_OBSERVED_FAILURE"}
		return row
	}
	if obs.PossibleDuplicate > 0 {
		row.Disposition = DispositionPossibleDuplicate
		row.ReasonCodes = []string{"SIMILARITY_REVIEW_REQUIRED"}
		row.DuplicateEvidence = []DuplicateEvidence{{Issue: obs.PossibleDuplicate, Kind: "similarity"}}
		return row
	}
	row.Disposition = DispositionReproRequired
	row.ReasonCodes = []string{"CURRENT_TESTED_STATE_REQUIRED"}
	return row
}

func ActionabilityExit(d ErrorDisposition) int {
	switch d {
	case DispositionActionable:
		return 0
	case DispositionDuplicateProven, DispositionFixPresentTrunk, DispositionFixReleased:
		return 3
	default:
		return 4
	}
}

func failureIdentity(e ErrorEvidence) string {
	parts := []string{strings.TrimSpace(e.Fingerprint), strings.TrimSpace(e.Module), strings.TrimSpace(e.FailureClass)}
	if parts[0] == "" || parts[1] == "" || parts[2] == "" {
		return ""
	}
	return strings.Join(parts, "\x00")
}
func evidenceStale(e ErrorEvidence) bool { return e.Stale }
func isFailure(s string) bool {
	s = strings.ToLower(strings.TrimSpace(s))
	return s == "fail" || s == "failed"
}
func isPassing(s string) bool {
	s = strings.ToLower(strings.TrimSpace(s))
	return s == "pass" || s == "passed"
}
func sameFailure(a, b ErrorEvidence) bool {
	return failureIdentity(a) != "" && failureIdentity(a) == failureIdentity(b)
}
