package ultracodebench

import (
	"crypto/sha256"
	"errors"
	"fmt"
	"sort"
	"strings"
)

const ToolResultShapeCampaignSchema = "fak-ultracode-tool-result-shape-campaign/1"
const ToolResultShapeReportSchema = "fak-ultracode-tool-result-shape-report/1"

type ToolResultShapeCampaign struct {
	Schema                 string                `json:"schema"`
	EvidenceKind           string                `json:"evidence_kind"`
	SourceArtifact         string                `json:"source_artifact"`
	Model                  string                `json:"model"`
	Runtime                string                `json:"runtime"`
	Tokenizer              string                `json:"tokenizer"`
	TaskDigest             string                `json:"task_digest"`
	CachePosture           string                `json:"cache_posture"`
	CampaignVersion        string                `json:"campaign_version"`
	AcceptedEffectDigest   string                `json:"accepted_effect_digest"`
	RequiredFactsDigest    string                `json:"required_facts_digest"`
	RoleTokens             int64                 `json:"role_tokens"`
	RepositoryTokens       int64                 `json:"repository_tokens"`
	HistoryTokens          int64                 `json:"history_tokens"`
	PromotionEvidence      string                `json:"promotion_evidence"`
	DemotionEvidence       string                `json:"demotion_evidence"`
	InvalidatingAssumption string                `json:"invalidating_assumption"`
	Cells                  []ToolResultShapeCell `json:"cells"`
}

type ToolResultShapeCell struct {
	Size                  string             `json:"size"`
	Relevance             string             `json:"relevance"`
	ToolResult            string             `json:"tool_result"`
	ToolResultTokens      int64              `json:"tool_result_tokens"`
	Accepted              bool               `json:"accepted"`
	EffectDigest          string             `json:"effect_digest"`
	RetainedFactsDigest   string             `json:"retained_facts_digest"`
	Omitted               ToolResultOmission `json:"omitted_tokens"`
	ScopingOverheadTokens int64              `json:"scoping_overhead_tokens"`
	SourceReceipt         string             `json:"source_receipt"`
}

type ToolResultOmission struct {
	Role        int64 `json:"role_setup"`
	Repository  int64 `json:"repository_context"`
	ToolResults int64 `json:"tool_results"`
	History     int64 `json:"history"`
}

type ToolResultShapeReport struct {
	Schema                 string                  `json:"schema"`
	SourceArtifact         string                  `json:"source_artifact"`
	Verdict                string                  `json:"verdict"`
	Reason                 string                  `json:"reason,omitempty"`
	Cells                  []ToolResultShapeResult `json:"cells"`
	Crossovers             map[string]string       `json:"first_net_positive_size_by_relevance,omitempty"`
	ReplayCommand          string                  `json:"replay_command"`
	PromotionEvidence      string                  `json:"promotion_evidence"`
	DemotionEvidence       string                  `json:"demotion_evidence"`
	InvalidatingAssumption string                  `json:"invalidating_assumption"`
}

type ToolResultShapeResult struct {
	Size      string             `json:"size"`
	Relevance string             `json:"relevance"`
	Verdict   string             `json:"verdict"`
	Reason    string             `json:"reason,omitempty"`
	Omitted   ToolResultOmission `json:"omitted_tokens"`
	NetTokens int64              `json:"net_tokens"`
}

// EvaluateToolResultShape reports only provenance-safe token savings. It does not
// choose runtime behavior: this is the predeclared decision-boundary experiment.
func EvaluateToolResultShape(c ToolResultShapeCampaign) (ToolResultShapeReport, error) {
	if c.Schema != ToolResultShapeCampaignSchema {
		return ToolResultShapeReport{}, fmt.Errorf("schema %q: %w", c.Schema, errors.New("unsupported tool-result shape campaign"))
	}
	r := ToolResultShapeReport{Schema: ToolResultShapeReportSchema, SourceArtifact: c.SourceArtifact, Verdict: "GAIN", Crossovers: map[string]string{}, ReplayCommand: "go test ./internal/ultracodebench -run TestIssue8677ToolResultShapeArtifact -count=1", PromotionEvidence: c.PromotionEvidence, DemotionEvidence: c.DemotionEvidence, InvalidatingAssumption: c.InvalidatingAssumption}
	if c.Model == "" || c.Runtime == "" || c.Tokenizer != "whitespace-v1" || c.TaskDigest == "" || c.CachePosture == "" || c.CampaignVersion == "" || len(c.Cells) != 4 {
		r.Verdict, r.Reason = "ABSTAIN", "incomplete bounded envelope or factorial cells"
		return r, nil
	}
	seen := map[string]bool{}
	for _, cell := range c.Cells {
		key := cell.Size + "/" + cell.Relevance
		if (cell.Size != "small" && cell.Size != "large") || (cell.Relevance != "relevant" && cell.Relevance != "irrelevant") || seen[key] {
			r.Verdict, r.Reason = "ABSTAIN", "cells must be the unique small/large by relevant/irrelevant factorial"
			return r, nil
		}
		seen[key] = true
		receipt := fmt.Sprintf("sha256:%x", sha256.Sum256([]byte(cell.ToolResult)))
		result := ToolResultShapeResult{Size: cell.Size, Relevance: cell.Relevance, Verdict: "GAIN", Omitted: cell.Omitted}
		if !cell.Accepted || cell.EffectDigest != c.AcceptedEffectDigest || cell.RetainedFactsDigest != c.RequiredFactsDigest || cell.SourceReceipt != receipt || cell.ToolResultTokens != int64(len(strings.Fields(cell.ToolResult))) {
			result.Verdict, result.Reason = "ABSTAIN", "unequal accepted effect, required-fact loss, or invalid source telemetry"
			r.Verdict, r.Reason = "ABSTAIN", "at least one cell is not claim-eligible"
		}
		if cell.Omitted.Role != 0 || cell.Omitted.Repository != 0 || cell.Omitted.History != 0 {
			result.Verdict, result.Reason = "ABSTAIN", "role, repository, and history omission must stay at zero"
			r.Verdict, r.Reason = "ABSTAIN", "non-tool-result context changed"
		}
		if cell.Omitted.Role != 0 || cell.Omitted.Repository != 0 || cell.Omitted.History != 0 {
			result.Verdict, result.Reason = "ABSTAIN", "role, repository, and history omission must stay at zero"
			r.Verdict, r.Reason = "ABSTAIN", "non-tool-result context changed"
		}
		if cell.Omitted.Role > c.RoleTokens || cell.Omitted.Repository > c.RepositoryTokens || cell.Omitted.History > c.HistoryTokens || cell.Omitted.ToolResults > cell.ToolResultTokens {
			result.Verdict, result.Reason = "ABSTAIN", "omission exceeds authoritative section tokens"
			r.Verdict, r.Reason = "ABSTAIN", "at least one cell has invalid token accounting"
		}
		result.NetTokens = cell.Omitted.Role + cell.Omitted.Repository + cell.Omitted.ToolResults + cell.Omitted.History - cell.ScopingOverheadTokens
		if result.Verdict == "GAIN" && result.NetTokens <= 0 {
			result.Verdict = "NO_GAIN"
			result.Reason = "scoping overhead is not repaid"
		}
		r.Cells = append(r.Cells, result)
	}
	if r.Verdict == "ABSTAIN" {
		r.Crossovers = nil
		return r, nil
	}
	sort.Slice(r.Cells, func(i, j int) bool {
		if r.Cells[i].Relevance != r.Cells[j].Relevance {
			return r.Cells[i].Relevance < r.Cells[j].Relevance
		}
		return r.Cells[i].Size == "small"
	})
	for _, cell := range r.Cells {
		if cell.NetTokens > 0 && r.Crossovers[cell.Relevance] == "" {
			r.Crossovers[cell.Relevance] = cell.Size
		}
	}
	return r, nil
}
