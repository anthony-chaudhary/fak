package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/modelaccept"
)

func TestRunModelAcceptanceGatePassAndHold(t *testing.T) {
	in := modelaccept.Input{Schema: modelaccept.Schema, Corpus: modelaccept.Corpus{ID: "fixture", DeclaredAt: "2026-07-14T23:00:00-07:00", Tasks: []modelaccept.Task{{ID: "exact", Tier: 2, Repetitions: 1, Expected: "OK"}}, Thresholds: modelaccept.Thresholds{MinSuccessRate: 1, MaxP95LatencyMS: 1000, MaxAverageInputTokens: 100, MaxAverageCostUSD: 1}}, Models: []modelaccept.ModelRequest{{Model: "claude-haiku-4-5-20251001", Family: "claude-haiku-4-5-20251001", Generation: "current", Lifecycle: modelaccept.LifecycleLatest, RequestedTier: 2}}, Runs: []modelaccept.Run{{Model: "claude-haiku-4-5-20251001", ActualModel: "claude-haiku-4-5-20251001", Task: "exact", Repetition: 1, Result: "OK", ToolValid: true, LatencyMS: 100, InputTokens: 10, CostUSD: .01, ObservedAt: "2026-07-14T23:01:00-07:00"}}}
	path := filepath.Join(t.TempDir(), "in.json")
	b, _ := json.Marshal(in)
	if err := os.WriteFile(path, b, 0600); err != nil {
		t.Fatal(err)
	}
	var out, errout bytes.Buffer
	if code := runModelAcceptanceGate(&out, &errout, []string{"--input", path}); code != 0 {
		t.Fatalf("pass exit=%d stderr=%s", code, errout.String())
	}
	in.Runs[0].ActualModel = "wrong"
	b, _ = json.Marshal(in)
	_ = os.WriteFile(path, b, 0600)
	out.Reset()
	errout.Reset()
	if code := runModelAcceptanceGate(&out, &errout, []string{"--input", path}); code != 4 {
		t.Fatalf("hold exit=%d stderr=%s", code, errout.String())
	}
}

func TestRunModelAcceptanceGateRejectsUnknownJSON(t *testing.T) {
	path := filepath.Join(t.TempDir(), "in.json")
	_ = os.WriteFile(path, []byte(`{"unknown":true}`), 0600)
	var out, errout bytes.Buffer
	if code := runModelAcceptanceGate(&out, &errout, []string{"--input", path}); code != 2 {
		t.Fatalf("exit=%d", code)
	}
}
