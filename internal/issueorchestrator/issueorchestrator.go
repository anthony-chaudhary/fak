package issueorchestrator

import (
	"github.com/anthony-chaudhary/fak/internal/issuepolicy"
)

// WavePlanSchema is the canonical schema identifier for concurrent-safe issue wave plans.
const WavePlanSchema = "fak.issue-orchestrator-wave-plan.v1"

// CompareSchema is the schema identifier for campaign burndown comparisons.
const CompareSchema = "fak.issue-orchestrator-compare.v1"

// WaveSafety describes concurrency safety classification of a wave.
type WaveSafety string

const (
	// WaveSafetyDisjointLeaf indicates all issues in this wave are pairwise tree-disjoint
	// with verified zero cross-package import contention.
	WaveSafetyDisjointLeaf WaveSafety = "pairwise_tree_disjoint"

	// WaveSafetySerialSingleton indicates a critical/core leaf with high blast radius
	// that MUST execute alone as a dedicated single-worker wave without concurrent siblings.
	WaveSafetySerialSingleton WaveSafety = "serial_singleton"
)

// Issue represents one analyzed, routable issue item.
type Issue struct {
	Number          int                      `json:"number"`
	Key             string                   `json:"key"`
	Title           string                   `json:"title"`
	Lane            string                   `json:"lane"`
	Paths           []string                 `json:"paths"`
	ExpectedSteps   int                      `json:"expected_steps"`
	State           string                   `json:"state,omitempty"`
	Labels          []string                 `json:"labels,omitempty"`
	URL             string                   `json:"url,omitempty"`
	Centrality      string                   `json:"centrality,omitempty"`
	ProblemFrame    issuepolicy.ProblemFrame `json:"problem_frame,omitempty"`
	Dispatchability string                   `json:"dispatchability,omitempty"`
	OpencodeCommand []string                 `json:"opencode_command,omitempty"`
}

// Wave represents one execution wave of concurrent-safe issues.
type Wave struct {
	Index         int            `json:"index"`                    // 1-based wave sequence number
	ID            string         `json:"id"`                       // e.g. "wave-1"
	Safety        WaveSafety     `json:"safety"`                   // pairwise_tree_disjoint or serial_singleton
	WaveSize      int            `json:"wave_size"`                // Number of concurrent workers in this wave
	StepBudget    int            `json:"step_budget"`              // Sum of expected steps across issues in this wave
	Issues        []Issue        `json:"issues"`                   // Issues allocated to this wave
	IssueNumbers  []int          `json:"issue_numbers"`            // Slice of issue numbers
	Lanes         []string       `json:"lanes"`                    // Slice of lane identifiers
	Paths         []string       `json:"paths"`                    // Normalized tree paths touched
	LeaseRegion   []string       `json:"lease_region,omitempty"`   // Minimal tree roots for `dos arbitrate --tree`
	LeaseLanes    []string       `json:"lease_lanes,omitempty"`    // Lanes taking their whole lane
	OpencodeChats []OpencodeChat `json:"opencode_chats,omitempty"` // Ready-to-run OpenCode chat sessions
}

// SubdivideRow records an epic or oversized issue that must be decomposed before dispatch.
type SubdivideRow struct {
	Key              string   `json:"key"`
	IssueNumber      int      `json:"issue_number,omitempty"`
	Title            string   `json:"title"`
	Reasons          []string `json:"reasons"`
	ExpectedSteps    int      `json:"expected_steps"`
	ChildIssueBudget int      `json:"child_issue_budget"`
	Lane             string   `json:"lane,omitempty"`
	Paths            []string `json:"paths,omitempty"`
}

// TriageRow records an issue requiring scope clarification or triage before dispatch.
type TriageRow struct {
	Key             string   `json:"key"`
	IssueNumber     int      `json:"issue_number,omitempty"`
	Title           string   `json:"title"`
	Dispatchability string   `json:"dispatchability"`
	Reasons         []string `json:"reasons,omitempty"`
	MissingFields   []string `json:"missing_fields,omitempty"`
}

// DuplicateGroup records identical marker keys appearing multiple times.
type DuplicateGroup struct {
	Key          string `json:"key"`
	Count        int    `json:"count"`
	IssueNumbers []int  `json:"issue_numbers,omitempty"`
}

// Plan is the full multi-wave campaign plan for issue resolution.
type Plan struct {
	Schema         string           `json:"schema"`
	Workspace      string           `json:"workspace"`
	TotalIssues    int              `json:"total_issues"`
	Dispatchable   int              `json:"dispatchable"`
	Subdividable   int              `json:"subdividable"`
	TriageOnly     int              `json:"triage_only"`
	HeldIssues     []int            `json:"held_issues,omitempty"`
	HeldLanes      []string         `json:"held_lanes,omitempty"`
	ExcludedIssues []int            `json:"excluded_issues,omitempty"`
	ExcludedLanes  []string         `json:"excluded_lanes,omitempty"`
	WaveSizeCap    int              `json:"wave_size_cap"`
	TotalWaves     int              `json:"total_waves"`
	PlannedIssues  int              `json:"planned_issues"`
	PlannedSteps   int              `json:"planned_steps"`
	TargetIssues   int              `json:"target_issues,omitempty"`
	TargetPoints   int              `json:"target_points,omitempty"`
	Waves          []Wave           `json:"waves"`
	Subdivide      []SubdivideRow   `json:"subdivide,omitempty"`
	Triage         []TriageRow      `json:"triage,omitempty"`
	Duplicates     []DuplicateGroup `json:"duplicates,omitempty"`
}

// WavePlanOptions configures campaign wave generation.
type WavePlanOptions struct {
	WaveSize                int
	MaxWaves                int
	TargetIssues            int
	TargetPoints            int
	Limit                   int
	LaneFilter              string
	ExcludedIssues          []int
	ExcludedLanes           []string
	AutoDetectHeld          bool
	StrictProjectWork       bool
	WorkspaceRoot           string
	Graph                   map[string]map[string]struct{}
	IncludeOpencodeCommands bool
	OpencodeOptions         OpencodeChatOptions
}

// OpencodeChatOptions configures fresh OpenCode chat generation.
type OpencodeChatOptions struct {
	Model       string   `json:"model,omitempty"`
	Agent       string   `json:"agent,omitempty"`
	Interactive bool     `json:"interactive,omitempty"`
	WorktreeDir string   `json:"worktree_dir,omitempty"`
	AutoApprove bool     `json:"auto_approve,omitempty"`
	PrintLogs   bool     `json:"print_logs,omitempty"`
	ExtraArgs   []string `json:"extra_args,omitempty"`
}

// OpencodeChat describes a fresh OpenCode chat session for an issue.
type OpencodeChat struct {
	IssueNumber  int      `json:"issue_number"`
	Key          string   `json:"key"`
	Title        string   `json:"title"`
	Lane         string   `json:"lane"`
	SessionTitle string   `json:"session_title"`
	Worktree     string   `json:"worktree,omitempty"`
	Command      []string `json:"command"`
	Prompt       string   `json:"prompt"`
}
