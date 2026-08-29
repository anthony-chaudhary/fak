package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/modelaccept"
)

func writeReadinessFixture(t *testing.T) string {
	t.Helper()
	in := modelaccept.Input{Schema: modelaccept.Schema, Corpus: modelaccept.Corpus{ID: "fixture", DeclaredAt: "2026-07-14T20:00:00Z", Tasks: []modelaccept.Task{{ID: "exact", Tier: 2, Repetitions: 1, Expected: "OK"}}, Thresholds: modelaccept.Thresholds{MinSuccessRate: 1, MaxP95LatencyMS: 1000, MaxAverageInputTokens: 100, MaxAverageCostUSD: 1}}, Models: []modelaccept.ModelRequest{{Model: "exact-a", Family: "exact-a", Generation: "current", Lifecycle: modelaccept.LifecycleLatest, RequestedTier: 2}}, Runs: []modelaccept.Run{{Model: "exact-a", ActualModel: "exact-a", Task: "exact", Repetition: 1, Result: "OK", ToolValid: true, LatencyMS: 10, InputTokens: 10, CostUSD: .01, ObservedAt: "2026-07-14T21:00:00Z"}}}
	path := filepath.Join(t.TempDir(), "acceptance.json")
	b, err := json.Marshal(in)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, b, 0600); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestRunModelReadinessInventoryPassCarriesProvenance(t *testing.T) {
	path := writeReadinessFixture(t)
	var out, errout bytes.Buffer
	code := runModelReadinessInventory(&out, &errout, []string{"--input", path, "--artifact", "examples/acceptance.json", "--artifact-revision", "internal/modelaccept@r1+gabc", "--expected-corpus", "fixture", "--as-of", "2026-07-15T00:00:00Z"})
	if code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, errout.String())
	}
	var got modelaccept.Inventory
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Verdict != modelaccept.Pass || len(got.Rows) != 1 || got.Rows[0].Artifact != "examples/acceptance.json" || got.Rows[0].ArtifactRevision != "internal/modelaccept@r1+gabc" || got.Semantics != nil {
		t.Fatalf("inventory=%+v", got)
	}
}

func TestRunModelReadinessInventoryExactLadderScopedHoldReadback(t *testing.T) {
	dir := filepath.Join("..", "..", "docs", "_witnesses", "issue-8623-qwen38-27b")
	var out, errout bytes.Buffer
	code := runModelReadinessInventory(&out, &errout, []string{
		"--ladder-evidence-dir", dir,
		"--artifact-revision", "docs@8cd6a82af97f",
		"--expected-corpus", "qwen38-27b-semantic-answer-quality-v2",
	})
	if code != 4 {
		t.Fatalf("exit=%d stderr=%s output=%s", code, errout.String(), out.String())
	}
	var got modelaccept.Inventory
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Schema != modelaccept.Qwen38LadderInventorySchema || got.Verdict != modelaccept.Hold || len(got.Rows) != 1 || got.Rows[0].CapabilityGate != modelaccept.Hold || got.Rows[0].LadderEvidence == nil || got.Semantics == nil {
		t.Fatalf("inventory=%+v", got)
	}
	evidence := got.Rows[0].LadderEvidence
	if evidence.Model != "Qwen/Qwen3.8-27B" || evidence.Precision != "BF16" || evidence.Topology != "TP2" || evidence.Correctness.Trials != 3 || evidence.Correctness.BaselinePassed != 3 || evidence.Correctness.CandidatePassed != 3 || evidence.P95.BaselineMetric != 3378.019733 || evidence.P95.CandidateMetric != 376.181809 || evidence.P95.Improvement != 88.86383624923602 {
		t.Fatalf("evidence=%+v", evidence)
	}
	manifestBody, err := os.ReadFile(filepath.Join(dir, "checksums.json"))
	if err != nil {
		t.Fatal(err)
	}
	var manifest []struct {
		File   string `json:"file"`
		SHA256 string `json:"sha256"`
	}
	if err := json.Unmarshal(manifestBody, &manifest); err != nil {
		t.Fatal(err)
	}
	wantArtifacts := make(map[string]string, len(manifest))
	for _, artifact := range manifest {
		wantArtifacts[artifact.File] = artifact.SHA256
	}
	if len(evidence.ArtifactHashes) != len(wantArtifacts) {
		t.Fatalf("artifact hashes=%+v want manifest=%+v", evidence.ArtifactHashes, wantArtifacts)
	}
	for _, artifact := range evidence.ArtifactHashes {
		if want, ok := wantArtifacts[artifact.Path]; !ok || artifact.SHA256 != want {
			t.Fatalf("artifact hash %+v not bound to manifest", artifact)
		}
		delete(wantArtifacts, artifact.Path)
	}
	if len(wantArtifacts) != 0 {
		t.Fatalf("manifest artifacts omitted from evidence: %+v", wantArtifacts)
	}
	passCells := 0
	for _, cell := range got.Rows[0].ReadinessCells {
		if cell.Status == modelaccept.ReadinessCellPass {
			passCells++
			if cell.ID != "arithmetic_latency" {
				t.Fatalf("unsupported PASS cell=%+v", cell)
			}
		}
	}
	if passCells != 1 {
		t.Fatalf("readiness cells=%+v", got.Rows[0].ReadinessCells)
	}
	if got.Semantics.Default == "" || got.Semantics.Replacement == "" || got.Semantics.Rollback == "" {
		t.Fatalf("semantics=%+v", got.Semantics)
	}
}

func TestRunModelReadinessInventoryHoldExit(t *testing.T) {
	path := writeReadinessFixture(t)
	var out, errout bytes.Buffer
	code := runModelReadinessInventory(&out, &errout, []string{"--input", path, "--artifact-revision", "rev", "--expected-corpus", "wrong", "--as-of", "2026-07-15T00:00:00Z"})
	if code != 4 {
		t.Fatalf("exit=%d stderr=%s", code, errout.String())
	}
}

func TestRunModelReadinessInventoryUsageAndStrictDecode(t *testing.T) {
	for _, tc := range []struct {
		name string
		args []string
	}{
		{"missing revision", []string{"--input", writeReadinessFixture(t)}},
		{"bad as-of", []string{"--artifact-revision", "rev", "--as-of", "tomorrow"}},
		{"manifest without ladder directory", []string{"--artifact-revision", "rev", "--ladder-manifest", "checksums.json"}},
		{"ambiguous ladder and generic input", []string{"--artifact-revision", "rev", "--input", writeReadinessFixture(t), "--ladder-evidence-dir", "evidence"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var out, errout bytes.Buffer
			if code := runModelReadinessInventory(&out, &errout, tc.args); code != 2 {
				t.Fatalf("exit=%d", code)
			}
		})
	}
	path := filepath.Join(t.TempDir(), "bad.json")
	if err := os.WriteFile(path, []byte(`{"unknown":true}`), 0600); err != nil {
		t.Fatal(err)
	}
	var out, errout bytes.Buffer
	if code := runModelReadinessInventory(&out, &errout, []string{"--input", path, "--artifact-revision", "rev"}); code != 2 {
		t.Fatalf("strict decode exit=%d", code)
	}
}
