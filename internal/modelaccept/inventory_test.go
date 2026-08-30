package modelaccept

import (
	"encoding/json"
	"path/filepath"
	"strings"
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

func TestReadinessInventoryExactLadderJoinIsScopedAndReadbackStable(t *testing.T) {
	dir := filepath.Join("..", "..", "docs", "_witnesses", "issue-8623-qwen38-27b")
	got, admission := BuildQwen38LadderReadinessInventory(InventoryOptions{
		ArtifactRevision: "docs@8cd6a82af97f",
		ExpectedCorpusID: qwen38ExactCorpusID,
	}, LadderEvidenceOptions{Directory: dir, Manifest: filepath.Join(dir, "checksums.json")})
	if admission.Verdict != Pass || admission.Reason.Code != "" {
		t.Fatalf("admission=%+v", admission)
	}
	if got.Schema != Qwen38LadderInventorySchema || got.Verdict != Hold || got.CorpusID != qwen38ExactCorpusID || len(got.Rows) != 1 {
		t.Fatalf("inventory=%+v", got)
	}
	row := got.Rows[0]
	if row.Model != qwen38ExactModel || row.Generation != qwen38ExactRevision || row.CapabilityGate != Hold || row.Samples != 3 || row.LadderEvidence == nil {
		t.Fatalf("row=%+v", row)
	}
	evidence := row.LadderEvidence
	if evidence.Precision != "BF16" || evidence.Topology != "TP2" || evidence.Runtime != "vLLM 0.27.1" || evidence.CapturedAt != "2026-08-22T14:12:32.298303-07:00" || row.DeclaredAt != evidence.CapturedAt || evidence.RuntimePair.BaselineSHA != qwen38ExactBaselineRuntime || evidence.RuntimePair.CandidateSHA != qwen38ExactCandidateRuntime || evidence.CorpusSHA256 != qwen38ExactCorpusSHA || evidence.EnvironmentSHA256 != qwen38ExactEnvironmentSHA || evidence.Correctness != (CorrectnessPair{BaselinePassed: 3, CandidatePassed: 3, Trials: 3}) || evidence.P95.BaselineMetric != qwen38ExactBaselineP95MS || evidence.P95.CandidateMetric != qwen38ExactCandidateP95MS || evidence.P95.Improvement != qwen38ExactImprovementPct {
		t.Fatalf("evidence=%+v", evidence)
	}
	if len(evidence.ArtifactHashes) != len(qwen38ExactArtifactHashes) {
		t.Fatalf("artifact hashes = %+v, want one per exact artifact", evidence.ArtifactHashes)
	}
	for _, artifact := range evidence.ArtifactHashes {
		if want, ok := qwen38ExactArtifactHashes[artifact.Path]; !ok || !strings.EqualFold(artifact.SHA256, want) {
			t.Fatalf("artifact hash is not bound to exact profile: %+v", artifact)
		}
	}
	wantCells := qwen38ReadinessCells(ReadinessCellPass)
	passCells := 0
	if len(row.ReadinessCells) != len(wantCells) {
		t.Fatalf("readiness cells=%+v, want canonical cells=%+v", row.ReadinessCells, wantCells)
	}
	for i, cell := range row.ReadinessCells {
		if cell != wantCells[i] {
			t.Fatalf("readiness cell[%d]=%+v, want %+v", i, cell, wantCells[i])
		}
		if cell.Owner == "" || !strings.HasPrefix(cell.Owner, "https://github.com/anthony-chaudhary/fak/issues/") {
			t.Fatalf("cell has no owning issue: %+v", cell)
		}
		if cell.Status == ReadinessCellPass {
			passCells++
			if cell.ID != "arithmetic_latency" {
				t.Fatalf("unsupported cell passed: %+v", cell)
			}
		}
		if cell.ID != "arithmetic_latency" && cell.Status != ReadinessCellHold && cell.Status != ReadinessCellUnwitnessed {
			t.Fatalf("unsupported cell status=%+v", cell)
		}
	}
	if passCells != 1 {
		t.Fatalf("readiness cells=%+v", row.ReadinessCells)
	}
	if got.Semantics == nil || !strings.Contains(got.Semantics.Default, "explicitly selected") || !strings.Contains(got.Semantics.Replacement, "code-pinned identity") || !strings.Contains(got.Semantics.Rollback, "no serving") {
		t.Fatalf("semantics=%+v", got.Semantics)
	}

	captured, err := json.Marshal(got)
	if err != nil {
		t.Fatal(err)
	}
	var readback Inventory
	if err := json.Unmarshal(captured, &readback); err != nil {
		t.Fatal(err)
	}
	if readback.Verdict != Hold || len(readback.Rows) != 1 || readback.Rows[0].LadderEvidence == nil || readback.Rows[0].LadderEvidence.P95.Improvement != qwen38ExactImprovementPct {
		t.Fatalf("captured readback=%+v", readback)
	}
}

func TestReadinessInventoryExactLadderJoinCorpusMismatchFailsClosed(t *testing.T) {
	dir := filepath.Join("..", "..", "docs", "_witnesses", "issue-8623-qwen38-27b")
	got, admission := BuildQwen38LadderReadinessInventory(InventoryOptions{
		ArtifactRevision: "docs@8cd6a82af97f",
		ExpectedCorpusID: "proxy-corpus",
	}, LadderEvidenceOptions{Directory: dir, Manifest: filepath.Join(dir, "checksums.json")})
	if admission.Verdict != Hold || admission.Reason.Code != LadderEvidenceIdentityMismatch || len(got.Rows) != 1 || got.Rows[0].LadderEvidence != nil || got.Rows[0].CapabilityGate != Hold {
		t.Fatalf("inventory=%+v admission=%+v", got, admission)
	}
	for _, cell := range got.Rows[0].ReadinessCells {
		if cell.Status == ReadinessCellPass {
			t.Fatalf("identity mismatch published PASS: %+v", cell)
		}
	}
}
