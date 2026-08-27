// Package studyprio builds and validates the bounded priority queue derived from a study join ledger.
package studyprio

import "errors"

const (
	Schema        = "fak.study-priority-ledger/1"
	SummarySchema = "fak.study-priority-summary/1"
	RubricVersion = "studyprio-rubric/1"
	sourceSchema  = "fak.study-join-ledger/1"
)

var ErrInvalid = errors.New("studyprio: invalid priority ledger")

type BuildOptions struct{ SourceLedgerPath string }
type ValidateOptions struct{ SourceLedgerPath, LedgerPath, SummaryPath string }

type Rubric struct {
	Version       string            `json:"version"`
	Minimum       int               `json:"minimum"`
	Maximum       int               `json:"maximum"`
	Dimensions    []RubricDimension `json:"dimensions"`
	RequiredGates []string          `json:"required_gates"`
	TieBreaks     []string          `json:"tie_breaks"`
}
type RubricDimension struct {
	Name       string `json:"name"`
	Weight     int    `json:"weight"`
	Definition string `json:"definition"`
}
type SourceReceipt struct {
	Path           string `json:"path"`
	SHA256         string `json:"sha256"`
	Schema         string `json:"schema"`
	SourceRevision string `json:"source_revision"`
	Cutoff         string `json:"cutoff"`
	UncoveredCount int    `json:"uncovered_actionable_count"`
}
type SourceMapping struct {
	ClusterID      string `json:"cluster_id"`
	Mechanism      string `json:"mechanism"`
	Signal         string `json:"signal"`
	Rule           string `json:"rule"`
	MembersSHA256  string `json:"members_sha256"`
	EvidenceSHA256 string `json:"evidence_sha256"`
}
type HardGate struct {
	Name     string `json:"name"`
	Pass     bool   `json:"pass"`
	Evidence string `json:"evidence"`
}
type Dimensions struct {
	ProductCentrality          int `json:"product_centrality"`
	FakNativeQwen38Impact      int `json:"fak_native_qwen3_8_impact"`
	EndToEndValue              int `json:"end_to_end_value"`
	EvidenceStrength           int `json:"evidence_strength"`
	Recurrence                 int `json:"recurrence"`
	DependencyUnlock           int `json:"dependency_unlock"`
	ImplementationCost         int `json:"implementation_cost"`
	HardwareWitnessCost        int `json:"hardware_witness_cost"`
	CompatibilityRisk          int `json:"compatibility_risk"`
	DuplicationConflictPenalty int `json:"duplication_conflict_penalty"`
}
type ValueFrame struct {
	For           string `json:"for"`
	Problem       string `json:"problem"`
	Today         string `json:"today"`
	BetterBecause string `json:"better_because"`
	Witness       string `json:"witness"`
}
type P1P4Check struct {
	Status string `json:"status"`
	Reason string `json:"reason"`
}
type P1P4 struct {
	P1Context    P1P4Check `json:"p1_context"`
	P2NetValue   P1P4Check `json:"p2_net_value"`
	P3Adaptation P1P4Check `json:"p3_adaptation"`
	P4Operations P1P4Check `json:"p4_operations"`
}
type Witness struct {
	Artifact      string `json:"artifact"`
	Command       string `json:"command"`
	PassCondition string `json:"pass_condition"`
	Engine        string `json:"engine"`
	Model         string `json:"model"`
}
type ExecutionContract struct {
	Engine          string `json:"engine"`
	DefaultModel    string `json:"default_model"`
	LlamaCPPUse     string `json:"llama_cpp_use"`
	FallbackAllowed bool   `json:"fallback_allowed"`
}
type Candidate struct {
	ID                 string            `json:"id"`
	Title              string            `json:"title"`
	Category           string            `json:"category"`
	Horizon            string            `json:"horizon"`
	Centrality         string            `json:"centrality"`
	SourceMappings     []SourceMapping   `json:"source_mappings"`
	MergeJustification string            `json:"merge_justification,omitempty"`
	HardGates          []HardGate        `json:"hard_gates"`
	Dimensions         Dimensions        `json:"dimensions"`
	Score              int               `json:"score"`
	Dependencies       []string          `json:"dependencies"`
	Frame              ValueFrame        `json:"frame"`
	P1P4               P1P4              `json:"p1_p4"`
	Witness            Witness           `json:"witness"`
	Execution          ExecutionContract `json:"execution"`
}
type QueueEntry struct {
	Rank         int      `json:"rank"`
	CandidateID  string   `json:"candidate_id"`
	Score        int      `json:"score"`
	Category     string   `json:"category"`
	Horizon      string   `json:"horizon"`
	Dependencies []string `json:"dependencies"`
}
type SensitivityExample struct {
	CandidateID   string            `json:"candidate_id"`
	BaselineScore int               `json:"baseline_score"`
	Steps         []SensitivityStep `json:"steps"`
}
type SensitivityStep struct {
	Dimension     string `json:"dimension"`
	From          int    `json:"from"`
	To            int    `json:"to"`
	AdjustedScore int    `json:"adjusted_score"`
	QueueFirst    string `json:"queue_first"`
	Explanation   string `json:"explanation"`
}
type Ledger struct {
	Schema      string             `json:"schema"`
	Rubric      Rubric             `json:"rubric"`
	Source      SourceReceipt      `json:"source"`
	Candidates  []Candidate        `json:"candidates"`
	Queue       []QueueEntry       `json:"queue"`
	Sensitivity SensitivityExample `json:"sensitivity"`
}
type Summary struct {
	Schema               string   `json:"schema"`
	RubricVersion        string   `json:"rubric_version"`
	SourceLedgerSHA256   string   `json:"source_ledger_sha256"`
	PriorityLedgerSHA256 string   `json:"priority_ledger_sha256"`
	SourceClusterCount   int      `json:"source_cluster_count"`
	CandidateCount       int      `json:"candidate_count"`
	QueueCount           int      `json:"queue_count"`
	QueueCandidateIDs    []string `json:"queue_candidate_ids"`
	Detail               Ledger   `json:"-"`
}

type sourceLedger struct {
	Schema         string       `json:"schema"`
	Cutoff         string       `json:"cutoff"`
	SourceRevision string       `json:"source_revision"`
	Joins          []sourceJoin `json:"joins"`
}
type sourceJoin struct {
	ClusterID     string         `json:"cluster_id"`
	Mechanism     string         `json:"mechanism"`
	Signal        string         `json:"signal"`
	Rule          string         `json:"rule"`
	Actionable    bool           `json:"actionable"`
	Disposition   string         `json:"disposition"`
	Evidence      sourceEvidence `json:"evidence"`
	MembersSHA256 string         `json:"members_checksum"`
}
type sourceEvidence struct {
	Digest string `json:"digest"`
}
