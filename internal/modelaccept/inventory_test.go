package modelaccept

import (
	"testing"
	"time"
)

func inventoryFixture() Input {
	return Input{Schema: Schema, Corpus: Corpus{ID: "c1", DeclaredAt: "2026-07-14T20:00:00Z", Tasks: []Task{{ID: "exact", Tier: 2, Repetitions: 1, Expected: "OK"}}, Thresholds: Thresholds{MinSuccessRate: 1, MaxP95LatencyMS: 1000, MaxAverageInputTokens: 100, MaxAverageCostUSD: 1}}, Models: []ModelRequest{{Model: "exact-a", Family: "exact-a", Generation: "current", Lifecycle: LifecycleLatest, RequestedTier: 2}}, Runs: []Run{{Model: "exact-a", ActualModel: "exact-a", Task: "exact", Repetition: 1, Result: "OK", ToolValid: true, LatencyMS: 10, InputTokens: 10, CostUSD: .01, ObservedAt: "2026-07-14T21:00:00Z"}}}
}

func TestBuildInventoryPassCarriesExactProvenance(t *testing.T) {
	got := BuildInventory(inventoryFixture(), InventoryOptions{Artifact: "examples/a.json", ArtifactRevision: "r1+gabc", ExpectedCorpusID: "c1", AsOf: time.Date(2026, 7, 15, 0, 0, 0, 0, time.UTC)})
	if got.Verdict != Pass || len(got.Rows) != 1 {
		t.Fatalf("inventory=%+v", got)
	}
	r := got.Rows[0]
	if r.CapabilityGate != Pass || r.WitnessedTier == nil || *r.WitnessedTier != 2 || r.Samples != 1 || r.ObservedFirst == "" || r.ObservedLast == "" || r.ArtifactRevision != "r1+gabc" {
		t.Fatalf("row=%+v", r)
	}
}

func TestBuildInventoryNeverBorrowsAcrossExactIDs(t *testing.T) {
	in := inventoryFixture()
	in.Models = append(in.Models, ModelRequest{Model: "exact-b", Family: "exact-b", Generation: "current", Lifecycle: LifecycleLatest, RequestedTier: 2})
	got := BuildInventory(in, InventoryOptions{Artifact: "a", ArtifactRevision: "rev", ExpectedCorpusID: "c1", AsOf: time.Date(2026, 7, 15, 0, 0, 0, 0, time.UTC)})
	if got.Verdict != Hold || len(got.Rows) != 2 {
		t.Fatalf("inventory=%+v", got)
	}
	for _, r := range got.Rows {
		if r.Model == "exact-b" && r.CapabilityGate != Hold {
			t.Fatalf("missing exact-b evidence borrowed exact-a PASS: %+v", r)
		}
	}
}

func TestBuildInventoryMismatchStaleAndMissingProvenanceHold(t *testing.T) {
	for _, tc := range []struct {
		name string
		opts InventoryOptions
	}{
		{"corpus mismatch", InventoryOptions{Artifact: "a", ArtifactRevision: "rev", ExpectedCorpusID: "wrong", AsOf: time.Date(2026, 7, 15, 0, 0, 0, 0, time.UTC)}},
		{"stale", InventoryOptions{Artifact: "a", ArtifactRevision: "rev", ExpectedCorpusID: "c1", AsOf: time.Date(2027, 7, 15, 0, 0, 0, 0, time.UTC)}},
		{"missing revision", InventoryOptions{Artifact: "a", ExpectedCorpusID: "c1", AsOf: time.Date(2026, 7, 15, 0, 0, 0, 0, time.UTC)}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := BuildInventory(inventoryFixture(), tc.opts)
			if got.Rows[0].CapabilityGate != Hold {
				t.Fatalf("row=%+v", got.Rows[0])
			}
		})
	}
}

func TestBuildInventoryIsolatesPerModelFailure(t *testing.T) {
	in := inventoryFixture()
	in.Models = append(in.Models, ModelRequest{Model: "exact-b", Family: "exact-b", Generation: "current", Lifecycle: LifecycleLatest, RequestedTier: 2})
	in.Runs = append(in.Runs, Run{Model: "exact-b", ActualModel: "wrong-b", Task: "exact", Repetition: 1, Result: "OK", ToolValid: true, LatencyMS: 10, InputTokens: 10, CostUSD: .01, ObservedAt: "2026-07-14T21:00:00Z"})
	got := BuildInventory(in, InventoryOptions{Artifact: "a", ArtifactRevision: "rev", ExpectedCorpusID: "c1", AsOf: time.Date(2026, 7, 15, 0, 0, 0, 0, time.UTC)})
	if got.Verdict != Hold || len(got.Rows) != 2 {
		t.Fatalf("inventory=%+v", got)
	}
	for _, row := range got.Rows {
		switch row.Model {
		case "exact-a":
			if row.CapabilityGate != Pass {
				t.Fatalf("exact-a inherited exact-b failure: %+v", row)
			}
		case "exact-b":
			if row.CapabilityGate != Hold || row.WitnessedTier != nil {
				t.Fatalf("wrong actual ID did not HOLD exact-b: %+v", row)
			}
		}
	}
}

func TestBuildInventoryGlobalMalformedInputHoldsEveryRow(t *testing.T) {
	in := inventoryFixture()
	in.Models = append(in.Models, ModelRequest{Model: "exact-a", Family: "exact-a", Generation: "current", Lifecycle: LifecycleLatest, RequestedTier: 2})
	got := BuildInventory(in, InventoryOptions{Artifact: "a", ArtifactRevision: "rev", AsOf: time.Date(2026, 7, 15, 0, 0, 0, 0, time.UTC)})
	if got.Verdict != Hold || len(got.Rows) != 2 {
		t.Fatalf("inventory=%+v", got)
	}
	for _, row := range got.Rows {
		if row.CapabilityGate != Hold || row.WitnessedTier != nil {
			t.Fatalf("malformed aggregate did not fail closed: %+v", row)
		}
	}
}

func TestBuildInventoryRejectsFutureDatedEvidence(t *testing.T) {
	got := BuildInventory(inventoryFixture(), InventoryOptions{Artifact: "a", ArtifactRevision: "rev", AsOf: time.Date(2026, 7, 14, 20, 30, 0, 0, time.UTC)})
	if got.Verdict != Hold || got.Rows[0].CapabilityGate != Hold {
		t.Fatalf("future evidence passed: %+v", got)
	}
}

func TestBuildInventoryUndeclaredRunFailsClosed(t *testing.T) {
	in := inventoryFixture()
	in.Runs = append(in.Runs, Run{Model: "undeclared", ActualModel: "undeclared", Task: "exact", Repetition: 1, Result: "OK", ToolValid: true, LatencyMS: 10, InputTokens: 10, CostUSD: .01, ObservedAt: "2026-07-14T21:00:00Z"})
	got := BuildInventory(in, InventoryOptions{Artifact: "a", ArtifactRevision: "rev", AsOf: time.Date(2026, 7, 15, 0, 0, 0, 0, time.UTC)})
	if got.Verdict != Hold || len(got.Rows) != 1 || got.Rows[0].Model != "exact-a" || got.Rows[0].CapabilityGate != Hold {
		t.Fatalf("undeclared run was silently accepted: %+v", got)
	}
}
