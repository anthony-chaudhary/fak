package modelaccept

import (
	"strings"
	"testing"
	"time"
)

func TestLatestGenerationPolicySkipsTombstonesUnlessReasoned(t *testing.T) {
	latest := ModelRequest{Model: "vendor-pro-6", Family: "vendor/pro", Generation: "6", Lifecycle: LifecycleLatest}
	old := ModelRequest{Model: "vendor-pro-5", Family: "vendor/pro", Generation: "5", Lifecycle: LifecycleTombstoned}
	exception := old
	exception.EvalException = &EvalException{Reason: "regression bisect", Ticket: "#7001"}
	if !ShouldEvaluate(latest) {
		t.Fatal("latest generation was skipped")
	}
	if ShouldEvaluate(old) {
		t.Fatal("tombstoned generation consumed eval capacity without a reason")
	}
	if !ShouldEvaluate(exception) {
		t.Fatal("named older-generation exception was skipped")
	}
}

func TestEvaluateRecordsTombstoneSkipWithoutDemandingRuns(t *testing.T) {
	in := generationPolicyFixture()
	got := Evaluate(in)
	if got.Verdict != Pass || len(got.Models) != 2 {
		t.Fatalf("decision=%+v", got)
	}
	if got.Models[1].Verdict != Skip || got.Models[1].Samples != 0 || !strings.Contains(strings.Join(got.Models[1].Reasons, " "), "skipped") {
		t.Fatalf("older decision=%+v", got.Models[1])
	}
}

func TestInventoryKeepsTombstoneVisibleWithoutHoldingLatest(t *testing.T) {
	in := generationPolicyFixture()
	got := BuildInventory(in, InventoryOptions{Artifact: "report.json", ArtifactRevision: "rev", ExpectedCorpusID: "generation-policy", AsOf: time.Date(2026, 8, 14, 1, 0, 0, 0, time.UTC)})
	if got.Verdict != Pass || len(got.Rows) != 2 {
		t.Fatalf("inventory=%+v", got)
	}
	var tombstone *InventoryRow
	for i := range got.Rows {
		if got.Rows[i].Lifecycle == LifecycleTombstoned {
			tombstone = &got.Rows[i]
		}
	}
	if tombstone == nil || tombstone.CapabilityGate != Skip {
		t.Fatalf("rows=%+v", got.Rows)
	}
}

func TestOlderGenerationExceptionRequiresNamedReason(t *testing.T) {
	in := generationPolicyFixture()
	in.Models[1].EvalException = &EvalException{Ticket: "#7001"}
	got := Evaluate(in)
	if got.Verdict != Hold || !strings.Contains(strings.Join(got.Reasons, " "), "named reason") {
		t.Fatalf("decision=%+v", got)
	}
}

func generationPolicyFixture() Input {
	return Input{
		Schema: Schema,
		Corpus: Corpus{ID: "generation-policy", DeclaredAt: "2026-08-14T00:00:00Z", Tasks: []Task{{ID: "exact", Tier: 0, Repetitions: 1, Expected: "OK"}}, Thresholds: Thresholds{MinSuccessRate: 1, MaxP95LatencyMS: 1000, MaxAverageInputTokens: 100, MaxAverageCostUSD: 1}},
		Models: []ModelRequest{
			{Model: "vendor-pro-6", Family: "vendor/pro", Generation: "6", Lifecycle: LifecycleLatest},
			{Model: "vendor-pro-5", Family: "vendor/pro", Generation: "5", Lifecycle: LifecycleTombstoned},
		},
		Runs: []Run{{Model: "vendor-pro-6", ActualModel: "vendor-pro-6", Task: "exact", Repetition: 1, Result: "OK", ToolValid: true, LatencyMS: 10, InputTokens: 10, CostUSD: .01, ObservedAt: "2026-08-14T00:01:00Z"}},
	}
}
