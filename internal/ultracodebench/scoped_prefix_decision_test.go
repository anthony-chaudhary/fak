package ultracodebench

import (
	"encoding/json"
	"os"
	"strings"
	"testing"
	"time"
)

func TestEvaluateScopedPrefixEvidenceVerdicts(t *testing.T) {
	now := time.Date(2026, 8, 26, 12, 0, 0, 0, time.UTC)
	positive := scopedPrefixFixture(now)
	tests := []struct {
		name string
		edit func(*ScopedPrefixEvidence)
		want ScopedPrefixVerdict
	}{
		{"positive", func(*ScopedPrefixEvidence) {}, ScopedPrefixEnable},
		{"negative control failure", func(e *ScopedPrefixEvidence) { e.Rows[0].NegativeControlPass = false }, ScopedPrefixDisable},
		{"missing telemetry", func(e *ScopedPrefixEvidence) { e.Rows[0].AuthoritativeMetric = false }, ScopedPrefixAbstain},
		{"outcome mismatch", func(e *ScopedPrefixEvidence) { e.Rows[0].AcceptedOutcomeEqual = false }, ScopedPrefixAbstain},
		{"expired evidence", func(e *ScopedPrefixEvidence) { e.ExpiresAt = now }, ScopedPrefixAbstain},
		{"uncertain net gain", func(e *ScopedPrefixEvidence) { e.Rows[0].Uncertainty = 30 }, ScopedPrefixHold},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			e := positive
			e.Rows = append([]ScopedPrefixRow(nil), positive.Rows...)
			tt.edit(&e)
			r := EvaluateScopedPrefixEvidence(e, now)
			if r.Verdict != tt.want {
				t.Fatalf("verdict = %s, want %s; report = %+v", r.Verdict, tt.want, r)
			}
			if r.SmallestNextExperiment == "" || r.Rollback == "" || r.PromotionEvidence == "" || r.DemotionEvidence == "" || r.InvalidatingAssumption == "" {
				t.Fatalf("incomplete report: %+v", r)
			}
			if r.RuntimeDefaultChanged {
				t.Fatal("decision surface changed runtime default")
			}
		})
	}
}

func TestEvaluateScopedPrefixEvidenceRejectsBareSummary(t *testing.T) {
	now := time.Now().UTC()
	r := EvaluateScopedPrefixEvidence(ScopedPrefixEvidence{IndependentWitness: "62.7% scoped / 37.3% prefix"}, now)
	if r.Verdict != ScopedPrefixAbstain || !strings.Contains(r.Reason, "summary is insufficient") {
		t.Fatalf("report = %+v", r)
	}
}

func TestFormatScopedPrefixReportNamesEnvelopeAndContradictions(t *testing.T) {
	now := time.Now().UTC()
	e := scopedPrefixFixture(now)
	e.Rows[0].Uncertainty = 30
	rendered := FormatScopedPrefixReport(EvaluateScopedPrefixEvidence(e, now))
	for _, want := range []string{"decision: HOLD", "model=qwen", "contradictory rows: width-1", "smallest next experiment:", "runtime default changed: false", "rollback:", "promotion evidence:", "demotion evidence:", "invalidating assumption:"} {
		if !strings.Contains(rendered, want) {
			t.Fatalf("render missing %q:\n%s", want, rendered)
		}
	}
}

func scopedPrefixFixture(now time.Time) ScopedPrefixEvidence {
	return ScopedPrefixEvidence{
		Envelope:           ScopedPrefixEnvelope{Model: "qwen", Runtime: "ollama/llama.cpp", Tokenizer: "qwen", Task: "frozen-agent-task", WarmthCondition: "cold/warm factorial", CampaignVersion: "v1"},
		IndependentWitness: "sha256:witness", ObservedAt: now.Add(-time.Hour), ExpiresAt: now.Add(time.Hour),
		Rows: []ScopedPrefixRow{{Name: "width-1", AcceptedOutcomeEqual: true, ScopeAvoided: 30, PrefixReadAvoided: 18, Uncertainty: 2, NetOverhead: 4, AuthoritativeMetric: true, NegativeControlPass: true, SourcesSeparated: true, SourceReceipt: "sha256:row"}},
	}
}

type scopedPrefixCorpus struct {
	Schema          string                  `json:"schema"`
	CampaignVersion string                  `json:"campaign_version"`
	Rows            []scopedPrefixCorpusRow `json:"rows"`
}
type scopedPrefixCorpusRow struct {
	ID                  string  `json:"id"`
	Runtime             string  `json:"runtime"`
	Model               string  `json:"model"`
	Tokenizer           string  `json:"tokenizer"`
	Task                string  `json:"task"`
	CachePosture        string  `json:"cache_posture"`
	OutcomeParity       bool    `json:"outcome_parity"`
	ScopeAvoided        float64 `json:"scope_avoided"`
	PrefixReadAvoided   float64 `json:"cache_avoided"`
	Uncertainty         float64 `json:"uncertainty"`
	NetOverhead         float64 `json:"net_overhead"`
	AuthoritativeMetric bool    `json:"authoritative_metric"`
	NegativeControlPass bool    `json:"negative_control_pass"`
	SourcesSeparated    bool    `json:"attribution_disjoint"`
	ResetOccurred       bool    `json:"cache_reset_detected"`
	SourceReceipt       string  `json:"source_receipt"`
	Expected            struct {
		Decision      ScopedPrefixVerdict `json:"decision"`
		ScopeShareMin float64             `json:"scope_share_min"`
		ScopeShareMax float64             `json:"scope_share_max"`
		NetAvoidedMin float64             `json:"net_avoided_min"`
		NetAvoidedMax float64             `json:"net_avoided_max"`
	} `json:"expected"`
}

func TestScopedPrefixRegressionCorpus(t *testing.T) {
	data, err := os.ReadFile("testdata/scoped-prefix-regression-v1.json")
	if err != nil {
		t.Fatal(err)
	}
	var corpus scopedPrefixCorpus
	if err := json.Unmarshal(data, &corpus); err != nil {
		t.Fatal(err)
	}
	if corpus.Schema != "fak.ultracode.scoped-prefix-regression-corpus.v1" || corpus.CampaignVersion == "" {
		t.Fatalf("unversioned corpus: %+v", corpus)
	}
	if len(corpus.Rows) != 6 { //boundarylint:ignore CHANGE_DETECTOR_TEST the scoped-prefix fixture defines exactly six corpus rows used by the decision
		t.Fatalf("rows = %d, want six predeclared cases", len(corpus.Rows))
	}
	wantRows := map[string]bool{"observed-positive-qwen25-05b": true, "no-gain-control": true, "unequal-outcome": true, "missing-telemetry": true, "double-counted-savings": true, "cache-reset": true}
	now := time.Date(2026, 8, 26, 12, 0, 0, 0, time.UTC)
	seen := map[string]bool{}
	for _, row := range corpus.Rows {
		t.Run(row.ID, func(t *testing.T) {
			if !wantRows[row.ID] {
				t.Fatalf("unexpected corpus row %q", row.ID)
			}
			if row.ID == "" || seen[row.ID] {
				t.Fatalf("missing or duplicate row id %q", row.ID)
			}
			seen[row.ID] = true
			if row.Runtime == "" || row.Model == "" || row.Tokenizer == "" || row.Task == "" || row.CachePosture == "" || row.SourceReceipt == "" {
				t.Fatalf("row is not replayable/versioned: %+v", row)
			}
			e := ScopedPrefixEvidence{Envelope: ScopedPrefixEnvelope{Model: row.Model, Runtime: row.Runtime, Tokenizer: row.Tokenizer, Task: row.Task, WarmthCondition: row.CachePosture, CampaignVersion: corpus.CampaignVersion}, IndependentWitness: row.SourceReceipt, ObservedAt: now.Add(-time.Hour), ExpiresAt: now.Add(time.Hour), Rows: []ScopedPrefixRow{{Name: row.ID, AcceptedOutcomeEqual: row.OutcomeParity, ScopeAvoided: row.ScopeAvoided, PrefixReadAvoided: row.PrefixReadAvoided, Uncertainty: row.Uncertainty, NetOverhead: row.NetOverhead, AuthoritativeMetric: row.AuthoritativeMetric, NegativeControlPass: row.NegativeControlPass, SourcesSeparated: row.SourcesSeparated, ResetOccurred: row.ResetOccurred, SourceReceipt: row.SourceReceipt}}}
			report := EvaluateScopedPrefixEvidence(e, now)
			if report.Verdict != row.Expected.Decision {
				t.Fatalf("verdict = %s, want %s; reason=%s", report.Verdict, row.Expected.Decision, report.Reason)
			}
			if row.Expected.ScopeShareMax > 0 && (report.ScopeShare < row.Expected.ScopeShareMin || report.ScopeShare > row.Expected.ScopeShareMax) {
				t.Fatalf("scope share %.2f outside [%.2f, %.2f]", report.ScopeShare, row.Expected.ScopeShareMin, row.Expected.ScopeShareMax)
			}
			if row.Expected.NetAvoidedMax != 0 && (report.NetAvoided < row.Expected.NetAvoidedMin || report.NetAvoided > row.Expected.NetAvoidedMax) {
				t.Fatalf("net avoided %.0f outside [%.0f, %.0f]", report.NetAvoided, row.Expected.NetAvoidedMin, row.Expected.NetAvoidedMax)
			}
		})
	}
	claimDoc, err := os.ReadFile("../../docs/_witnesses/issue-8624-ultracode-smallmodel/README.md")
	if err != nil {
		t.Fatal(err)
	}
	const claimRow = "corpus-row: observed-positive-qwen25-05b"
	if !strings.Contains(string(claimDoc), claimRow) || !seen["observed-positive-qwen25-05b"] {
		t.Fatalf("claim doc must reference extant %q", claimRow)
	}
}
