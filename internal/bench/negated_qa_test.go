package bench

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestNegatedQAProxyValidation(t *testing.T) {
	items, err := LoadNegatedQAFixture(filepath.Join("testdata", "negated_qa.json"))
	if err != nil {
		t.Fatal(err)
	}
	report, err := AnalyzeNegatedQA(items, "")
	if err != nil {
		t.Fatal(err)
	}
	encoded, _ := json.Marshal(report)
	t.Logf("NEGATED_QA %s correlation=%.4f sign_test_p=%.6f improved=%d/%d verdict=%s", report.Provenance, report.Correlation, report.SignTestP, report.ImprovedPairs, report.Pairs, report.Verdict)
	if report.Provenance != NegatedQAOfflineProvenance {
		t.Fatalf("report=%s", encoded)
	}
	if report.Correlation <= 0 || !report.DirectionalPass || !report.Significant || !strings.Contains(report.Verdict, "load-bearing") {
		t.Fatalf("thesis direction failed: %s", encoded)
	}
}

func TestNegatedQALiveWitnessUpgrade(t *testing.T) {
	items, err := LoadNegatedQAFixture(filepath.Join("testdata", "negated_qa.json"))
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "witness.jsonl")
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	for _, item := range items {
		fmt.Fprintf(f, `{"id":%q,"arm":"low","output":"safe answer"}`+"\n", item.ID)
		fmt.Fprintf(f, `{"id":%q,"arm":"high","output":%q}`+"\n", item.ID, "answer includes "+item.Forbidden[0])
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	report, err := AnalyzeNegatedQA(items, path)
	if err != nil {
		t.Fatal(err)
	}
	if report.Provenance != NegatedQALiveProvenance || report.ImprovedPairs != len(items) {
		t.Fatalf("report=%+v", report)
	}
}

func TestNegatedQALiveWitnessMustBeComplete(t *testing.T) {
	items, err := LoadNegatedQAFixture(filepath.Join("testdata", "negated_qa.json"))
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "incomplete.jsonl")
	if err := os.WriteFile(path, []byte(`{"id":"fruit-except-banana","arm":"low","output":"safe"}`+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := AnalyzeNegatedQA(items, path); err == nil {
		t.Fatal("incomplete live witness accepted")
	}
}
