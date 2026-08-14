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
	if got.Verdict != modelaccept.Pass || len(got.Rows) != 1 || got.Rows[0].Artifact != "examples/acceptance.json" || got.Rows[0].ArtifactRevision != "internal/modelaccept@r1+gabc" {
		t.Fatalf("inventory=%+v", got)
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
