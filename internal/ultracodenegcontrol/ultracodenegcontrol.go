package ultracodenegcontrol

import (
	"fmt"
	"sort"
)

const (
	CampaignSchema = "fak-ultracode-negative-control-campaign/1"
	ReportSchema   = "fak-ultracode-negative-control-report/1"
)

type Verdict string

const (
	NoGain        Verdict = "NO_GAIN"
	Abstain       Verdict = "ABSTAIN"
	Contradictory Verdict = "CONTRADICTORY"
)

type Envelope struct {
	Model           string `json:"model"`
	Runtime         string `json:"runtime"`
	Tokenizer       string `json:"tokenizer"`
	Task            string `json:"task"`
	CachePosture    string `json:"cache_posture"`
	CampaignVersion string `json:"campaign_version"`
}

type Observation struct {
	AcceptedOutcome    string `json:"accepted_outcome"`
	InputTokens        int64  `json:"input_tokens"`
	AuthoritativeUsage bool   `json:"authoritative_usage"`
	SourceReceipt      string `json:"source_receipt"`
}

type Control struct {
	Name                 string      `json:"name"`
	ExpectedVerdict      Verdict     `json:"expected_verdict"`
	ExpectedAttribution  string      `json:"expected_attribution"`
	PredeclaredBeforeRun bool        `json:"predeclared_before_run"`
	Baseline             Observation `json:"baseline"`
	Observed             Observation `json:"observed"`
}

type Campaign struct {
	Schema   string    `json:"schema"`
	Envelope Envelope  `json:"envelope"`
	Controls []Control `json:"controls"`
}

type Result struct {
	Name                 string   `json:"name"`
	ExpectedVerdict      Verdict  `json:"expected_verdict"`
	Verdict              Verdict  `json:"verdict"`
	CreditedSavings      int64    `json:"credited_savings"`
	Reason               string   `json:"reason"`
	SourceReceipts       []string `json:"source_receipts"`
	PublishContradiction bool     `json:"publish_contradiction"`
}

type Report struct {
	Schema   string   `json:"schema"`
	Envelope Envelope `json:"envelope"`
	Results  []Result `json:"results"`
}

var requiredControls = map[string]Verdict{
	"shuffled-role":            NoGain,
	"duplicated-context":       NoGain,
	"omitted-required-context": Abstain,
	"random-truncation":        Abstain,
}

// Evaluate replays the four frozen controls. Savings are always zero: a
// negative control either refutes a gain, forces abstention, or is published as
// contradictory evidence that invalidates the causal interpretation.
func Evaluate(c Campaign) (Report, error) {
	if c.Schema != CampaignSchema {
		return Report{}, fmt.Errorf("schema: got %q, want %q", c.Schema, CampaignSchema)
	}
	if c.Envelope.Model == "" || c.Envelope.Runtime == "" || c.Envelope.Tokenizer == "" || c.Envelope.Task == "" || c.Envelope.CachePosture == "" || c.Envelope.CampaignVersion == "" {
		return Report{}, fmt.Errorf("production envelope is incomplete")
	}
	if len(c.Controls) != len(requiredControls) {
		return Report{}, fmt.Errorf("controls: got %d, want %d", len(c.Controls), len(requiredControls))
	}
	seen := make(map[string]bool, len(c.Controls))
	report := Report{Schema: ReportSchema, Envelope: c.Envelope}
	for _, control := range c.Controls {
		expected, ok := requiredControls[control.Name]
		if !ok || seen[control.Name] {
			return Report{}, fmt.Errorf("control %q is unknown or duplicated", control.Name)
		}
		seen[control.Name] = true
		if !control.PredeclaredBeforeRun || control.ExpectedVerdict != expected || control.ExpectedAttribution != "none" {
			return Report{}, fmt.Errorf("control %q was not predeclared with verdict %s and attribution none", control.Name, expected)
		}
		result := evaluate(control)
		report.Results = append(report.Results, result)
	}
	sort.Slice(report.Results, func(i, j int) bool { return report.Results[i].Name < report.Results[j].Name })
	return report, nil
}

func evaluate(c Control) Result {
	r := Result{
		Name: c.Name, ExpectedVerdict: c.ExpectedVerdict,
		SourceReceipts: []string{c.Baseline.SourceReceipt, c.Observed.SourceReceipt},
	}
	if !c.Baseline.AuthoritativeUsage || !c.Observed.AuthoritativeUsage || c.Baseline.SourceReceipt == "" || c.Observed.SourceReceipt == "" {
		r.Verdict, r.Reason = Abstain, "authoritative telemetry or replay receipt missing"
		return r
	}
	if c.Baseline.AcceptedOutcome != c.Observed.AcceptedOutcome {
		r.Verdict, r.Reason = Abstain, "accepted outcomes diverged; harmful omission is not savings"
		return r
	}
	if c.ExpectedVerdict == Abstain {
		r.Verdict, r.Reason, r.PublishContradiction = Contradictory, "adversarial omission unexpectedly preserved the accepted outcome", true
		return r
	}
	if c.Observed.InputTokens < c.Baseline.InputTokens {
		r.Verdict, r.Reason, r.PublishContradiction = Contradictory, "placebo unexpectedly produced a shorter accepted prompt", true
		return r
	}
	r.Verdict, r.Reason = NoGain, "placebo produced no token reduction under an equal accepted outcome"
	return r
}
