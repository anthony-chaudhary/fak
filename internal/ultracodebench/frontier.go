package ultracodebench

import (
	"fmt"
	"sort"
)

const AccessFrontierSchema = "fak-ultracode-access-frontier/1"

type AccessCell struct {
	Mode                     string `json:"mode"`
	Width                    int    `json:"width"`
	AcceptedOutcome          bool   `json:"accepted_outcome"`
	OutcomeDigest            string `json:"outcome_digest"`
	FullContextInputTokens   int64  `json:"full_context_input_tokens"`
	ScopedContextInputTokens int64  `json:"scoped_context_input_tokens"`
	SharedPrefixReadTokens   int64  `json:"shared_prefix_read_tokens"`
}

type AccessFrontier struct {
	Schema            string       `json:"schema"`
	EvidenceKind      string       `json:"evidence_kind"`
	SourceArtifact    string       `json:"source_artifact,omitempty"`
	TaskDigest        string       `json:"task_digest"`
	ModelClass        string       `json:"model_class"`
	BaselineDigest    string       `json:"baseline_digest"`
	Counterfactual    string       `json:"counterfactual"`
	ExcludedBaselines []string     `json:"excluded_baselines"`
	Cells             []AccessCell `json:"cells"`
}

type AccessCellResult struct {
	Mode                     string `json:"mode"`
	Width                    int    `json:"width"`
	Verdict                  string `json:"verdict"`
	Reason                   string `json:"reason"`
	FullContextInputTokens   int64  `json:"full_context_input_tokens"`
	ScopedContextInputTokens int64  `json:"scoped_context_input_tokens"`
	SharedPrefixReadTokens   int64  `json:"shared_prefix_read_tokens"`
	ScopeAvoidedTokens       int64  `json:"scope_avoided_tokens"`
	CacheAvoidedTokens       int64  `json:"cache_avoided_tokens"`
	TotalAvoidedTokens       int64  `json:"total_avoided_tokens"`
	AvoidedPercent           int64  `json:"avoided_percent"`
}

type HillClimb struct {
	Mode        string `json:"mode"`
	ChosenWidth int    `json:"chosen_width"`
	StopWidth   int    `json:"stop_width,omitempty"`
	Reason      string `json:"reason"`
}

type AccessFrontierReport struct {
	Schema            string             `json:"schema"`
	EvidenceKind      string             `json:"evidence_kind"`
	SourceArtifact    string             `json:"source_artifact,omitempty"`
	TaskDigest        string             `json:"task_digest"`
	ModelClass        string             `json:"model_class"`
	ClaimScope        string             `json:"claim_scope"`
	ExcludedBaselines []string           `json:"excluded_baselines"`
	Cells             []AccessCellResult `json:"cells"`
	HillClimb         []HillClimb        `json:"hill_climb"`
}

func EvaluateAccessFrontier(f AccessFrontier, widths []int) (AccessFrontierReport, error) {
	if f.Schema != AccessFrontierSchema || f.TaskDigest == "" || f.BaselineDigest == "" {
		return AccessFrontierReport{}, fmt.Errorf("invalid access-frontier identity")
	}
	if f.EvidenceKind != "synthetic_fixture" && f.EvidenceKind != "observed_run" {
		return AccessFrontierReport{}, fmt.Errorf("evidence_kind must be synthetic_fixture or observed_run")
	}
	if f.EvidenceKind == "observed_run" && f.SourceArtifact == "" {
		return AccessFrontierReport{}, fmt.Errorf("observed_run requires source_artifact")
	}
	wanted := map[int]bool{}
	for _, width := range widths {
		if width < 1 {
			return AccessFrontierReport{}, fmt.Errorf("width must be positive")
		}
		wanted[width] = true
	}
	cells := append([]AccessCell(nil), f.Cells...)
	sort.Slice(cells, func(i, j int) bool {
		if cells[i].Mode == cells[j].Mode {
			return cells[i].Width < cells[j].Width
		}
		return cells[i].Mode < cells[j].Mode
	})
	report := AccessFrontierReport{
		Schema: AccessFrontierSchema, EvidenceKind: f.EvidenceKind, SourceArtifact: f.SourceArtifact, TaskDigest: f.TaskDigest, ModelClass: f.ModelClass,
		ClaimScope: "same-task agentic context work only", ExcludedBaselines: f.ExcludedBaselines,
	}
	byMode := map[string][]AccessCellResult{}
	for _, cell := range cells {
		if !wanted[cell.Width] {
			continue
		}
		r := evaluateAccessCell(cell, f.BaselineDigest)
		report.Cells = append(report.Cells, r)
		byMode[cell.Mode] = append(byMode[cell.Mode], r)
	}
	if len(report.Cells) == 0 {
		return AccessFrontierReport{}, fmt.Errorf("no cells match requested widths")
	}
	modes := make([]string, 0, len(byMode))
	for mode := range byMode {
		modes = append(modes, mode)
	}
	sort.Strings(modes)
	for _, mode := range modes {
		report.HillClimb = append(report.HillClimb, climb(mode, byMode[mode]))
	}
	return report, nil
}

func evaluateAccessCell(c AccessCell, baseline string) AccessCellResult {
	r := AccessCellResult{Mode: c.Mode, Width: c.Width, FullContextInputTokens: c.FullContextInputTokens, ScopedContextInputTokens: c.ScopedContextInputTokens, SharedPrefixReadTokens: c.SharedPrefixReadTokens}
	if !c.AcceptedOutcome || c.OutcomeDigest != baseline {
		r.Verdict, r.Reason = "ABSTAIN", "accepted outcome does not match the frozen baseline"
		return r
	}
	if c.FullContextInputTokens <= 0 || c.ScopedContextInputTokens <= 0 || c.ScopedContextInputTokens > c.FullContextInputTokens || c.SharedPrefixReadTokens < 0 || c.SharedPrefixReadTokens > c.ScopedContextInputTokens {
		r.Verdict, r.Reason = "ABSTAIN", "context or cache telemetry is missing or inconsistent"
		return r
	}
	r.ScopeAvoidedTokens = c.FullContextInputTokens - c.ScopedContextInputTokens
	r.CacheAvoidedTokens = c.SharedPrefixReadTokens
	r.TotalAvoidedTokens = r.ScopeAvoidedTokens + r.CacheAvoidedTokens
	r.AvoidedPercent = 100 * r.TotalAvoidedTokens / c.FullContextInputTokens
	if r.TotalAvoidedTokens == 0 {
		r.Verdict, r.Reason = "NO_GAIN", "equal outcome but no agentic context work was avoided"
	} else {
		r.Verdict, r.Reason = "GAIN", "equal outcome with independently decomposed scope and prefix-cache savings"
	}
	return r
}

func climb(mode string, cells []AccessCellResult) HillClimb {
	sort.Slice(cells, func(i, j int) bool { return cells[i].Width < cells[j].Width })
	result := HillClimb{Mode: mode, Reason: "all requested widths retained equal outcomes and positive agentic savings"}
	for _, cell := range cells {
		if cell.Verdict != "GAIN" {
			result.StopWidth = cell.Width
			result.Reason = "stopped before the first non-GAIN cell: " + cell.Reason
			return result
		}
		result.ChosenWidth = cell.Width
	}
	return result
}

func AccessFrontierFixture() AccessFrontier {
	const outcome = "sha256:accepted-edit-v1"
	return AccessFrontier{
		Schema: AccessFrontierSchema, EvidenceKind: "synthetic_fixture", SourceArtifact: "internal/ultracodebench.AccessFrontierFixture", TaskDigest: "sha256:frozen-small-model-agentic-task-v1", ModelClass: "small-model-independent", BaselineDigest: outcome,
		Counterfactual:    "each child receives the full parent context and recomputes the shared system+tools prefix",
		ExcludedBaselines: []string{"single-request throughput", "traditional batching throughput", "provider billed tokens", "provider spend"},
		Cells: []AccessCell{
			{Mode: "single", Width: 1, AcceptedOutcome: true, OutcomeDigest: outcome, FullContextInputTokens: 12000, ScopedContextInputTokens: 12000},
			{Mode: "scout_writer", Width: 1, AcceptedOutcome: true, OutcomeDigest: outcome, FullContextInputTokens: 12000, ScopedContextInputTokens: 9000, SharedPrefixReadTokens: 2000},
			{Mode: "scout_writer", Width: 2, AcceptedOutcome: true, OutcomeDigest: outcome, FullContextInputTokens: 24000, ScopedContextInputTokens: 13000, SharedPrefixReadTokens: 5000},
			{Mode: "scout_writer", Width: 4, AcceptedOutcome: true, OutcomeDigest: outcome, FullContextInputTokens: 48000, ScopedContextInputTokens: 21000, SharedPrefixReadTokens: 11000},
			{Mode: "scout_writer", Width: 8, AcceptedOutcome: true, OutcomeDigest: outcome, FullContextInputTokens: 96000, ScopedContextInputTokens: 37000, SharedPrefixReadTokens: 23000},
			{Mode: "multi_writer", Width: 1, AcceptedOutcome: true, OutcomeDigest: outcome, FullContextInputTokens: 12000, ScopedContextInputTokens: 9000, SharedPrefixReadTokens: 2000},
			{Mode: "multi_writer", Width: 2, AcceptedOutcome: true, OutcomeDigest: outcome, FullContextInputTokens: 24000, ScopedContextInputTokens: 15000, SharedPrefixReadTokens: 5000},
			{Mode: "multi_writer", Width: 4, AcceptedOutcome: false, OutcomeDigest: "sha256:conflicting-edit", FullContextInputTokens: 48000, ScopedContextInputTokens: 29000, SharedPrefixReadTokens: 11000},
			{Mode: "multi_writer", Width: 8, AcceptedOutcome: false, OutcomeDigest: "sha256:conflicting-edit", FullContextInputTokens: 96000, ScopedContextInputTokens: 53000, SharedPrefixReadTokens: 23000},
		},
	}
}
