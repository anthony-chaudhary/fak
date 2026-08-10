package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestVerifyTailPolicyMatrix(t *testing.T) {
	if err := verifyTailPolicyMatrix(filepath.Join("..", "..", "experiments", "microcontext", "s8l-tail-policy-matrix-2026-08-10.json")); err != nil {
		t.Fatal(err)
	}
}
func TestTailVerifierRejectsNoCancellation(t *testing.T) {
	r := tailPolicyReport{Schema: tailPolicySchema, Trials: 2, WindowDeadlineMS: 1, TaskDeadlineMS: 2}
	for _, n := range []string{"wait-all", "deadline-abstain", "sufficiency-stop", "bounded-hedge"} {
		p := tailPolicyResult{Policy: n, Grade: semanticGrade{Records: 16}}
		for i := 0; i < 2; i++ {
			p.Trials = append(p.Trials, tailTrial{Receipts: make([]windowReceipt, 16), HedgesOpened: func() int {
				if n == "bounded-hedge" {
					return 1
				}
				return 0
			}()})
		}
		r.Policies = append(r.Policies, p)
	}
	b, _ := json.Marshal(r)
	p := filepath.Join(t.TempDir(), "bad.json")
	_ = os.WriteFile(p, b, 0644)
	if verifyTailPolicyMatrix(p) == nil {
		t.Fatal("accepted no-cancellation matrix")
	}
}
