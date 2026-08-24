package ultracodebench

import (
	"errors"
	"fmt"
	"math"
	"sort"
)

const FactorialCampaignSchema = "fak-ultracode-factorial-campaign/1"
const FactorialReportSchema = "fak-ultracode-factorial-report/1"

type FactorialCampaign struct {
	Schema                 string          `json:"schema"`
	EvidenceKind           string          `json:"evidence_kind"`
	SourceArtifact         string          `json:"source_artifact"`
	CapturedAt             string          `json:"captured_at"`
	Node                   string          `json:"node"`
	Runtime                string          `json:"runtime"`
	Model                  string          `json:"model"`
	Tokenizer              string          `json:"tokenizer"`
	TaskDigest             string          `json:"task_digest"`
	OutcomeDigest          string          `json:"accepted_outcome_digest"`
	Metric                 string          `json:"metric"`
	OrderPolicy            string          `json:"order_policy"`
	PromotionEvidence      string          `json:"promotion_evidence"`
	DemotionEvidence       string          `json:"demotion_evidence"`
	InvalidatingAssumption string          `json:"invalidating_assumption"`
	Cells                  []FactorialCell `json:"cells"`
}

type FactorialCell struct {
	Width      int                  `json:"width"`
	Treatment  string               `json:"treatment"`
	Context    string               `json:"context"`
	Cache      string               `json:"cache"`
	Order      int                  `json:"order"`
	Replicates []FactorialReplicate `json:"replicates"`
}

type FactorialReplicate struct {
	Accepted      bool    `json:"accepted"`
	OutcomeDigest string  `json:"outcome_digest"`
	Work          float64 `json:"work"`
	InputTokens   int64   `json:"input_tokens"`
	CachedTokens  int64   `json:"cached_tokens"`
	ResetReceipt  string  `json:"cache_reset_receipt"`
	WarmupReceipt string  `json:"warmup_receipt,omitempty"`
}

type FactorialReport struct {
	Schema                 string                 `json:"schema"`
	SourceArtifact         string                 `json:"source_artifact"`
	Metric                 string                 `json:"metric"`
	Widths                 []FactorialWidthResult `json:"widths"`
	HillClimb              FactorialHillClimb     `json:"hill_climb"`
	ReplayCommand          string                 `json:"replay_command"`
	PromotionEvidence      string                 `json:"promotion_evidence"`
	DemotionEvidence       string                 `json:"demotion_evidence"`
	InvalidatingAssumption string                 `json:"invalidating_assumption"`
}

type FactorialWidthResult struct {
	Width         int             `json:"width"`
	Verdict       string          `json:"verdict"`
	Reason        string          `json:"reason,omitempty"`
	GenericPrefix FactorialEffect `json:"generic_prefix_b_minus_a"`
	MicroScope    FactorialEffect `json:"micro_scope_c_minus_a"`
	Combined      FactorialEffect `json:"combined_d_minus_a"`
	Interaction   FactorialEffect `json:"interaction_d_minus_b_minus_c_plus_a"`
}

type FactorialEffect struct {
	Estimate float64 `json:"estimate"`
	StdError float64 `json:"std_error"`
}

type FactorialHillClimb struct {
	ChosenWidth int `json:"chosen_width"`
	StopWidth   int `json:"stop_width,omitempty"`
}

func EvaluateFactorialCampaign(c FactorialCampaign, widths []int) (FactorialReport, error) {
	r := FactorialReport{Schema: FactorialReportSchema, SourceArtifact: c.SourceArtifact, Metric: c.Metric, ReplayCommand: "go test ./internal/ultracodebench -run TestIssue8648FactorialArtifact -count=1", PromotionEvidence: c.PromotionEvidence, DemotionEvidence: c.DemotionEvidence, InvalidatingAssumption: c.InvalidatingAssumption}
	if c.Schema != FactorialCampaignSchema {
		return r, fmt.Errorf("schema %q, want %q", c.Schema, FactorialCampaignSchema)
	}
	if c.EvidenceKind == "observed_run" && c.SourceArtifact == "" {
		return r, errors.New("observed campaign requires source_artifact")
	}
	if c.TaskDigest == "" || c.OutcomeDigest == "" || c.Model == "" || c.Runtime == "" || c.Tokenizer == "" || c.Metric == "" {
		return r, errors.New("campaign identity and telemetry metric are required")
	}
	if c.OrderPolicy != "alternating" && c.OrderPolicy != "randomized" {
		return r, fmt.Errorf("order_policy %q is not randomized or alternating", c.OrderPolicy)
	}
	for _, width := range widths {
		wr := evaluateFactorialWidth(c, width)
		r.Widths = append(r.Widths, wr)
		if r.HillClimb.StopWidth == 0 {
			if wr.Verdict == "GAIN" {
				r.HillClimb.ChosenWidth = width
			} else {
				r.HillClimb.StopWidth = width
			}
		}
	}
	return r, nil
}

func evaluateFactorialWidth(c FactorialCampaign, width int) FactorialWidthResult {
	wr := FactorialWidthResult{Width: width, Verdict: "ABSTAIN"}
	cells := map[string]FactorialCell{}
	orders := map[int]bool{}
	for _, cell := range c.Cells {
		if cell.Width != width {
			continue
		}
		if _, exists := cells[cell.Treatment]; exists || orders[cell.Order] {
			wr.Reason = "duplicate treatment or order"
			return wr
		}
		cells[cell.Treatment], orders[cell.Order] = cell, true
	}
	want := map[string][2]string{"A": {"full", "cold"}, "B": {"full", "warm"}, "C": {"scoped", "cold"}, "D": {"scoped", "warm"}}
	for name, factors := range want {
		cell, ok := cells[name]
		if !ok || cell.Context != factors[0] || cell.Cache != factors[1] || cell.Order < 1 || cell.Order > 4 || len(cell.Replicates) < 2 {
			wr.Reason = "incomplete factorial cell or uncertainty sample"
			return wr
		}
		for _, rep := range cell.Replicates {
			if !rep.Accepted || rep.OutcomeDigest != c.OutcomeDigest || rep.Work <= 0 || rep.InputTokens <= 0 || rep.ResetReceipt == "" || (cell.Cache == "warm" && (rep.WarmupReceipt == "" || rep.CachedTokens <= 0)) {
				wr.Reason = "unequal outcome or missing telemetry/receipt"
				return wr
			}
		}
	}
	a, av := sample(cells["A"])
	b, bv := sample(cells["B"])
	cc, cv := sample(cells["C"])
	d, dv := sample(cells["D"])
	wr.GenericPrefix = effect(b-a, bv+av)
	wr.MicroScope = effect(cc-a, cv+av)
	wr.Combined = effect(d-a, dv+av)
	wr.Interaction = effect(d-b-cc+a, dv+bv+cv+av)
	if wr.Combined.Estimate < 0 {
		wr.Verdict = "GAIN"
	} else {
		wr.Verdict = "NO_GAIN"
		wr.Reason = "combined treatment did not reduce measured work"
	}
	return wr
}

func sample(c FactorialCell) (mean, meanVariance float64) {
	for _, r := range c.Replicates {
		mean += r.Work
	}
	mean /= float64(len(c.Replicates))
	if len(c.Replicates) == 1 {
		return mean, 0
	}
	for _, r := range c.Replicates {
		d := r.Work - mean
		meanVariance += d * d
	}
	meanVariance /= float64(len(c.Replicates) * (len(c.Replicates) - 1))
	return mean, meanVariance
}

func effect(estimate, variance float64) FactorialEffect {
	return FactorialEffect{Estimate: estimate, StdError: math.Sqrt(variance)}
}

func FactorialWidths(c FactorialCampaign) []int {
	seen := map[int]bool{}
	for _, cell := range c.Cells {
		seen[cell.Width] = true
	}
	widths := make([]int, 0, len(seen))
	for width := range seen {
		widths = append(widths, width)
	}
	sort.Ints(widths)
	return widths
}
