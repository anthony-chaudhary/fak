package bench

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"sort"
	"strings"
)

// ReasonUnwitnessedDenominator is emitted when benchmark evaluation results
// fail census domain reconciliation — dropped tasks, phantom tasks, duplicates,
// unpermitted simulated runs, or empty domain sets.
const ReasonUnwitnessedDenominator = "UNWITNESSED_DENOMINATOR"

// TaskOutcome represents the terminal disposition of a single benchmark task.
type TaskOutcome string

const (
	OutcomePass    TaskOutcome = "pass"
	OutcomeFail    TaskOutcome = "fail"
	OutcomeTimeout TaskOutcome = "timeout"
	OutcomeCrash   TaskOutcome = "crash"
	OutcomeRefused TaskOutcome = "refused"
)

// CensusManifest defines the expected canonical evaluation domain and execution policy.
type CensusManifest struct {
	TaskDomain          []string `json:"task_domain"`
	ExpectedDomainProof string   `json:"expected_domain_proof,omitempty"`
	AllowSimulation     bool     `json:"allow_simulation,omitempty"`
	RequirePassWitness  bool     `json:"require_pass_witness,omitempty"`
}

// TaskResult records the attested outcome of a single task execution.
type TaskResult struct {
	TaskID       string      `json:"task_id"`
	Outcome      TaskOutcome `json:"outcome"`
	WitnessProof string      `json:"witness_proof,omitempty"`
	Simulated    bool        `json:"simulated,omitempty"`
}

// CensusReport is the fail-closed adjudication summary over a frozen denominator domain.
type CensusReport struct {
	NTotal      int     `json:"n_total"`
	NPass       int     `json:"n_pass"`
	NFail       int     `json:"n_fail"`
	NTimeout    int     `json:"n_timeout"`
	NCrash      int     `json:"n_crash"`
	NRefused    int     `json:"n_refused"`
	NSimulated  int     `json:"n_simulated"`
	PassRate    float64 `json:"pass_rate"` // NPass / float64(NTotal) strictly over NTotal
	Verified    bool    `json:"verified"`
	DomainProof string  `json:"domain_proof"` // SHA-256 of sorted task IDs
}

// ComputeDomainProof returns the deterministic SHA-256 hex digest of canonical
// sorted task IDs defining the evaluation domain.
func ComputeDomainProof(taskDomain []string) string {
	sorted := make([]string, len(taskDomain))
	copy(sorted, taskDomain)
	sort.Strings(sorted)

	h := sha256.New()
	for _, id := range sorted {
		h.Write([]byte(id))
		h.Write([]byte("\n"))
	}
	return hex.EncodeToString(h.Sum(nil))
}

// VerifyCensusDomain validates that results exactly cover the frozen task domain
// specified by manifest without omissions, duplications, unexpected task IDs, or
// unpermitted simulated runs.
func VerifyCensusDomain(manifest CensusManifest, results []TaskResult) (CensusReport, error) {
	if len(manifest.TaskDomain) == 0 {
		return CensusReport{}, fmt.Errorf("manifest task domain is empty: %s", ReasonUnwitnessedDenominator)
	}

	nTotal := len(manifest.TaskDomain)
	expectedTasks := make(map[string]bool, nTotal)
	for _, taskID := range manifest.TaskDomain {
		if taskID == "" {
			return CensusReport{}, fmt.Errorf("manifest domain contains empty task ID: %s", ReasonUnwitnessedDenominator)
		}
		if expectedTasks[taskID] {
			return CensusReport{}, fmt.Errorf("duplicate task in manifest domain %q: %s", taskID, ReasonUnwitnessedDenominator)
		}
		expectedTasks[taskID] = true
	}

	seenResults := make(map[string]bool, len(results))
	var nPass, nFail, nTimeout, nCrash, nRefused, nSimulated int

	for _, res := range results {
		if res.TaskID == "" {
			return CensusReport{}, fmt.Errorf("result contains empty task ID: %s", ReasonUnwitnessedDenominator)
		}
		if seenResults[res.TaskID] {
			return CensusReport{}, fmt.Errorf("duplicate task ID in results %q: %s", res.TaskID, ReasonUnwitnessedDenominator)
		}
		seenResults[res.TaskID] = true

		if !expectedTasks[res.TaskID] {
			return CensusReport{}, fmt.Errorf("unexpected task ID %q not in manifest domain: %s", res.TaskID, ReasonUnwitnessedDenominator)
		}

		if res.Simulated {
			nSimulated++
			if !manifest.AllowSimulation {
				return CensusReport{}, fmt.Errorf("simulated result not permitted in hardware census: %s", ReasonUnwitnessedDenominator)
			}
		}

		switch res.Outcome {
		case OutcomePass:
			if manifest.RequirePassWitness && strings.TrimSpace(res.WitnessProof) == "" {
				return CensusReport{}, fmt.Errorf("task %q marked pass has empty witness proof: %s", res.TaskID, ReasonUnwitnessedDenominator)
			}
			nPass++
		case OutcomeFail:
			nFail++
		case OutcomeTimeout:
			nTimeout++
		case OutcomeCrash:
			nCrash++
		case OutcomeRefused:
			nRefused++
		default:
			return CensusReport{}, fmt.Errorf("unrecognized task outcome %q for task %q: %s", res.Outcome, res.TaskID, ReasonUnwitnessedDenominator)
		}
	}

	for taskID := range expectedTasks {
		if !seenResults[taskID] {
			return CensusReport{}, fmt.Errorf("missing task %q from results: %s", taskID, ReasonUnwitnessedDenominator)
		}
	}

	sumCategorized := nPass + nFail + nTimeout + nCrash + nRefused
	if sumCategorized != nTotal || len(results) != nTotal {
		return CensusReport{}, fmt.Errorf("outcome sum %d != domain total %d: %s", sumCategorized, nTotal, ReasonUnwitnessedDenominator)
	}

	passRate := float64(nPass) / float64(nTotal)
	domainProof := ComputeDomainProof(manifest.TaskDomain)
	if manifest.ExpectedDomainProof != "" && !strings.EqualFold(manifest.ExpectedDomainProof, domainProof) {
		return CensusReport{}, fmt.Errorf("domain proof mismatch (got %s, want %s): %s", domainProof, manifest.ExpectedDomainProof, ReasonUnwitnessedDenominator)
	}

	return CensusReport{
		NTotal:      nTotal,
		NPass:       nPass,
		NFail:       nFail,
		NTimeout:    nTimeout,
		NCrash:      nCrash,
		NRefused:    nRefused,
		NSimulated:  nSimulated,
		PassRate:    passRate,
		Verified:    true,
		DomainProof: domainProof,
	}, nil
}
