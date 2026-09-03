package agentopt

import (
	"encoding/json"
	"io"
	"strings"
)

// Family 3: Self-improvement & feedback loops.
//
// STaR (Self-Taught Reasoner) reasoning trace filtering:
// Filters reasoning trajectories to retain only those with verifiable ground-truth
// external proof (such as green test exit code == 0 or verified oracle output)
// for prompt tuning or model fine-tuning. Trajectories with self-reported success
// lacking external verification are strictly rejected.

// ExternalProof captures ground-truth external execution or verification results.
type ExternalProof struct {
	Verified    bool   `json:"verified"`
	Command     string `json:"command,omitempty"`
	ExitCode    int    `json:"exit_code,omitempty"`
	ProofOutput string `json:"proof_output,omitempty"`
}

// IsSuccessful reports whether the external proof succeeded with a zero exit code.
func (p ExternalProof) IsSuccessful() bool {
	return p.Verified && p.ExitCode == 0
}

// ReasoningTrajectory represents an agent reasoning trace paired with self-reported
// outcome and ground-truth external verification.
type ReasoningTrajectory struct {
	ID                   string           `json:"id"`
	Prompt               string           `json:"prompt"`
	ThoughtTrace         string           `json:"thought_trace,omitempty"`
	Steps                []TrajectoryStep `json:"steps,omitempty"`
	SelfReportedSuccess  bool             `json:"self_reported_success"`
	ExternalVerification ExternalProof    `json:"external_verification"`
	FinalAnswer          string           `json:"final_answer,omitempty"`
	Metadata             map[string]any   `json:"metadata,omitempty"`
}

// VerifiedSuccess reports whether the trajectory satisfies ground-truth external verification.
func (t ReasoningTrajectory) VerifiedSuccess() bool {
	if !t.SelfReportedSuccess {
		return false
	}
	return t.ExternalVerification.IsSuccessful()
}

// STaRTrajectoryFilter filters reasoning trajectories to retain only those with verified
// ground-truth external proof for training or fine-tuning datasets.
type STaRTrajectoryFilter struct {
	// AllowUnreportedSuccess permits trajectories without self-reported success
	// if external proof is verified. Default false (requires self-reported success).
	AllowUnreportedSuccess bool `json:"allow_unreported_success,omitempty"`

	// AllowNonZeroExitCode permits non-zero exit codes. Default false (requires exit code 0).
	AllowNonZeroExitCode bool `json:"allow_non_zero_exit_code,omitempty"`

	// RequireProofOutput requires ProofOutput to be non-empty.
	RequireProofOutput bool `json:"require_proof_output,omitempty"`

	// RequireThoughtTrace requires ThoughtTrace to be non-empty.
	RequireThoughtTrace bool `json:"require_thought_trace,omitempty"`

	// ProofOutputValidator provides custom verification over proof output text.
	ProofOutputValidator func(output string) bool `json:"-"`
}

// NewSTaRTrajectoryFilter creates a default STaR reasoning trajectory filter.
func NewSTaRTrajectoryFilter() *STaRTrajectoryFilter {
	return &STaRTrajectoryFilter{}
}

// FilteredSTaRDataset represents the partitioned output of trajectory filtering.
type FilteredSTaRDataset struct {
	Trajectories  []ReasoningTrajectory `json:"trajectories"`
	Retained      []ReasoningTrajectory `json:"retained,omitempty"`
	Rejected      []ReasoningTrajectory `json:"rejected,omitempty"`
	TotalCount    int                   `json:"total_count"`
	RetainedCount int                   `json:"retained_count"`
	RejectedCount int                   `json:"rejected_count"`
}

// Len returns the count of retained trajectories.
func (d FilteredSTaRDataset) Len() int {
	return len(d.Trajectories)
}

// IsEmpty reports whether no trajectories were retained.
func (d FilteredSTaRDataset) IsEmpty() bool {
	return len(d.Trajectories) == 0
}

// ToTuningPrompts returns the prompt strings of all retained trajectories.
func (d FilteredSTaRDataset) ToTuningPrompts() []string {
	prompts := make([]string, 0, len(d.Trajectories))
	for _, t := range d.Trajectories {
		prompts = append(prompts, t.Prompt)
	}
	return prompts
}

// ExportJSONL writes retained trajectories as JSONL records to w.
func (d FilteredSTaRDataset) ExportJSONL(w io.Writer) error {
	enc := json.NewEncoder(w)
	for _, t := range d.Trajectories {
		if err := enc.Encode(t); err != nil {
			return err
		}
	}
	return nil
}

// Admit evaluates whether an individual trajectory qualifies for retention.
func (f STaRTrajectoryFilter) Admit(traj ReasoningTrajectory) bool {
	if strings.TrimSpace(traj.Prompt) == "" {
		return false
	}
	if !f.AllowUnreportedSuccess && !traj.SelfReportedSuccess {
		return false
	}
	if !traj.ExternalVerification.Verified {
		return false
	}
	if !f.AllowNonZeroExitCode && traj.ExternalVerification.ExitCode != 0 {
		return false
	}
	if f.RequireThoughtTrace && strings.TrimSpace(traj.ThoughtTrace) == "" {
		return false
	}
	if f.RequireProofOutput && strings.TrimSpace(traj.ExternalVerification.ProofOutput) == "" {
		return false
	}
	if f.ProofOutputValidator != nil && !f.ProofOutputValidator(traj.ExternalVerification.ProofOutput) {
		return false
	}
	return true
}

// FilterTrajectories filters a slice of reasoning trajectories and returns a FilteredSTaRDataset.
func (f STaRTrajectoryFilter) FilterTrajectories(trajectories []ReasoningTrajectory) FilteredSTaRDataset {
	var retained []ReasoningTrajectory
	var rejected []ReasoningTrajectory

	for _, traj := range trajectories {
		if f.Admit(traj) {
			retained = append(retained, traj)
		} else {
			rejected = append(rejected, traj)
		}
	}

	return FilteredSTaRDataset{
		Trajectories:  retained,
		Retained:      retained,
		Rejected:      rejected,
		TotalCount:    len(trajectories),
		RetainedCount: len(retained),
		RejectedCount: len(rejected),
	}
}
