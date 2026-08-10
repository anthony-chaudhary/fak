package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestBuildAuditRecordsDoesNotOverattributeNonUnanimousGold(t *testing.T) {
	fold := semanticTripleFold{Judgments: []semanticTripleJudgment{{ID: "issue-1", ToolNeed: "current_state", Votes: map[string]string{"a": "current_state", "b": "current_state", "c": "read_only"}, Unanimous: false}}}
	live := liveFilterToolReport{Policies: []string{"adaptive"}, Receipts: []liveToolReceipt{
		{Record: "issue-1", Policy: "adaptive", Phase: "cold", Gold: "current_state", Predicted: "read_only", Status: "completed"},
		{Record: "issue-1", Policy: "adaptive", Phase: "warm", Gold: "current_state", Predicted: "current_state", Status: "completed"},
	}}
	records, counts, err := buildAuditRecords(fold, live)
	if err != nil {
		t.Fatal(err)
	}
	if got := records[0].PrimaryClass; got != "questionable_gold" {
		t.Fatalf("primary class = %q", got)
	}
	if records[0].ColdWarmFlips != 1 || counts["records_with_cold_warm_flip"] != 1 {
		t.Fatal("cold/warm variance not preserved")
	}
}

func TestVerifyDisagreementAuditArtifact(t *testing.T) {
	p := filepath.Join("..", "..", "experiments", "microcontext", "s8p-live-disagreement-audit-2026-08-10.json")
	if err := verifyDisagreementAudit(p); err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile(p)
	if err != nil {
		t.Fatal(err)
	}
	var r disagreementAudit
	if err := json.Unmarshal(b, &r); err != nil {
		t.Fatal(err)
	}
	if r.Verdict != "no_pre_answer_signal" {
		t.Fatalf("verdict = %q", r.Verdict)
	}
	if r.Counts["records_with_stable_receipt_effect"] != 0 {
		t.Fatal("post-admission receipt effect was promoted to routing signal")
	}
}
