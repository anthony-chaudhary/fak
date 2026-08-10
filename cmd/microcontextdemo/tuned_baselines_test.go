package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestTunedBaselinesExactFrontier(t *testing.T) {
	root := filepath.Join("..", "..", "experiments", "microcontext")
	pub := filepath.Join(root, "s8f-github-issues-public-2026-08-09.json")
	ans := filepath.Join(root, "s8f-github-issues-answers-2026-08-09.json")
	out := filepath.Join(t.TempDir(), "baselines.json")
	if err := runTunedBaselines(pub, ans, out); err != nil {
		t.Fatal(err)
	}
	if err := verifyTunedBaselines(out); err != nil {
		t.Fatal(err)
	}
	b, _ := os.ReadFile(out)
	var r tunedBaselineReport
	if err := json.Unmarshal(b, &r); err != nil {
		t.Fatal(err)
	}
	if r.SemanticResidualRecords != 0 || r.TuneRecords != 108 || r.HeldOutRecords != 680 {
		t.Fatalf("unexpected boundary: %+v", r)
	}
	for _, x := range r.HeldOutResults {
		if x.ModelCalls != 0 || !x.Grade.QualityPass {
			t.Fatalf("pipeline %s: %+v", x.Pipeline, x)
		}
	}
}

func TestVerifyTunedBaselinesRejectsModeledSemanticCall(t *testing.T) {
	r := tunedBaselineReport{Schema: tunedBaselinesSchema, HeldOutRecords: 1, Decision: "falsifies", Configurations: make([]baselineConfiguration, 5), HeldOutResults: make([]baselineDryResult, 5)}
	for i := range r.HeldOutResults {
		r.HeldOutResults[i] = baselineDryResult{Pipeline: "p", Grade: gradeReport{QualityPass: true, TestRecords: 1}}
	}
	r.HeldOutResults[0].ModelCalls = 1
	b, _ := json.Marshal(r)
	p := filepath.Join(t.TempDir(), "bad.json")
	_ = os.WriteFile(p, b, 0644)
	if verifyTunedBaselines(p) == nil {
		t.Fatal("accepted unsupported semantic call")
	}
}
