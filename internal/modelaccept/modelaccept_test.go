package modelaccept

import (
	"strings"
	"testing"
)

const (
	opus   = "claude-opus-4-8"
	sonnet = "claude-sonnet-4-6"
	haiku  = "claude-haiku-4-5-20251001"
)

func validInput() Input {
	tasks := []Task{{ID: "exact", Tier: 2, Repetitions: 3, Expected: "ACCEPT_OK"}, {ID: "json", Tier: 1, Repetitions: 3, Expected: `{"ok":true}`}, {ID: "tool", Tier: 0, Repetitions: 3, Expected: "TOOL_OK", ToolRequired: true}}
	in := Input{Schema: Schema, Corpus: Corpus{ID: "top3-v1", DeclaredAt: "2026-07-14T23:00:00-07:00", Tasks: tasks, Thresholds: Thresholds{MinSuccessRate: 1, MaxP95LatencyMS: 10000, MaxProviderErrorRate: 0, MaxInvalidToolRate: 0, MaxAverageInputTokens: 30000, MaxAverageCostUSD: 1}}, Models: []ModelRequest{{Model: opus, RequestedTier: 0}, {Model: sonnet, RequestedTier: 1}, {Model: haiku, RequestedTier: 2}}}
	for _, m := range in.Models {
		for _, task := range tasks {
			if task.Tier >= m.RequestedTier {
				for rep := 1; rep <= task.Repetitions; rep++ {
					in.Runs = append(in.Runs, Run{Model: m.Model, ActualModel: m.Model, Task: task.ID, Repetition: rep, Result: task.Expected, ToolValid: true, LatencyMS: 5000, InputTokens: 10000, CostUSD: .1, ObservedAt: "2026-07-14T23:01:00-07:00"})
				}
			}
		}
	}
	return in
}

func decision(in Input, model string) *ModelDecision {
	d := Evaluate(in)
	for i := range d.Models {
		if d.Models[i].Model == model {
			return &d.Models[i]
		}
	}
	return nil
}

func TestEvaluatePassesExactIDsAtDeclaredTiers(t *testing.T) {
	got := Evaluate(validInput())
	if got.Verdict != Pass || len(got.Models) != 3 {
		t.Fatalf("decision=%+v", got)
	}
	for _, d := range got.Models {
		if d.Verdict != Pass {
			t.Fatalf("model=%+v", d)
		}
	}
}

func TestEvaluateRetainsEveryFailureClass(t *testing.T) {
	tests := []struct {
		name, want string
		edit       func(*Input)
	}{
		{"missing", "missing run", func(in *Input) { in.Runs = in.Runs[1:] }},
		{"wrong model", "wrong actual model", func(in *Input) { in.Runs[0].ActualModel = sonnet }},
		{"provider", "provider_error_rate", func(in *Input) { in.Runs[0].ProviderError = true }},
		{"result", "success_rate", func(in *Input) { in.Runs[0].Result = "wrong" }},
		{"tool", "invalid_tool_rate", func(in *Input) {
			for i := range in.Runs {
				if in.Runs[i].Task == "tool" {
					in.Runs[i].ToolValid = false
					break
				}
			}
		}},
		{"latency", "p95_latency_ms", func(in *Input) {
			for i := range in.Runs {
				if in.Runs[i].Model == opus {
					in.Runs[i].LatencyMS = 20000
				}
			}
		}},
		{"tokens", "average_input_tokens", func(in *Input) {
			for i := range in.Runs {
				if in.Runs[i].Model == opus {
					in.Runs[i].InputTokens = 40000
				}
			}
		}},
		{"cost", "average_cost_usd", func(in *Input) {
			for i := range in.Runs {
				if in.Runs[i].Model == opus {
					in.Runs[i].CostUSD = 2
				}
			}
		}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			in := validInput()
			tt.edit(&in)
			got := Evaluate(in)
			if got.Verdict != Hold || !strings.Contains(strings.Join(decision(in, opus).Reasons, " "), tt.want) {
				t.Fatalf("decision=%+v want %q", got, tt.want)
			}
		})
	}
}

func TestEvaluateRefusesTierOverclaim(t *testing.T) {
	in := validInput()
	in.Models[2].RequestedTier = 0
	got := Evaluate(in)
	d := decision(in, haiku)
	if got.Verdict != Hold || d.Verdict != Hold || !strings.Contains(strings.Join(d.Reasons, " "), "missing run: tool/1") {
		t.Fatalf("decision=%+v", got)
	}
}

func TestEvaluateHoldsInvalidContract(t *testing.T) {
	in := validInput()
	in.Schema = "wrong"
	got := Evaluate(in)
	if got.Verdict != Hold || !strings.Contains(strings.Join(got.Reasons, " "), "schema") {
		t.Fatalf("decision=%+v", got)
	}
}

func TestEvaluateRejectsPostHocCorpusDeclaration(t *testing.T) {
	in := validInput()
	in.Corpus.DeclaredAt = "2026-07-14T23:02:00-07:00"
	got := Evaluate(in)
	if got.Verdict != Hold || !strings.Contains(strings.Join(got.Reasons, " "), "predates corpus declaration") {
		t.Fatalf("decision=%+v, want post-hoc declaration HOLD", got)
	}
}
