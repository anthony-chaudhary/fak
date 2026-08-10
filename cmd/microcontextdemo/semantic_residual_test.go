package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestSemanticPacketFoldAndBlindGrade(t *testing.T) {
	root := filepath.Join("..", "..", "experiments", "microcontext")
	packet := filepath.Join(root, "s8i-semantic-packet-2026-08-10.json")
	var p semanticPacket
	pb, err := os.ReadFile(packet)
	if err != nil {
		t.Fatal(err)
	}
	if err = json.Unmarshal(pb, &p); err != nil {
		t.Fatal(err)
	}
	if len(p.Records) != 32 {
		t.Fatalf("records=%d", len(p.Records))
	}
	for _, x := range p.Records {
		if x.Split != "tune" && x.Split != "test" {
			t.Fatalf("unexpected split %q", x.Split)
		}
	}
	gold := filepath.Join(t.TempDir(), "gold.json")
	if err := foldSemanticAdjudicators(packet, filepath.Join(root, "s8i-adjudicator-a-2026-08-10.json"), filepath.Join(root, "s8i-adjudicator-openai-2026-08-10.json"), gold); err != nil {
		t.Fatal(err)
	}
	if err := verifySemanticGold(gold); err != nil {
		t.Fatal(err)
	}
	grade := filepath.Join(t.TempDir(), "grade.json")
	if err := gradeSemanticFiles(gold, filepath.Join(root, "s8i-semantic-oracle-submission-2026-08-10.json"), grade, "test"); err != nil {
		t.Fatal(err)
	}
	if err := verifySemanticGrade(grade); err != nil {
		t.Fatal(err)
	}
	b, _ := os.ReadFile(grade)
	var g semanticGrade
	if err := json.Unmarshal(b, &g); err != nil {
		t.Fatal(err)
	}
	if g.SemanticResidual != 3 || g.Abstentions != 16 || g.Exact != 16 {
		t.Fatalf("unexpected grade: %+v", g)
	}
}

func TestSemanticVerifierRejectsSelfAgreement(t *testing.T) {
	g := semanticGold{Schema: semanticGoldSchema, Adjudicators: []string{"same", "same"}, Models: []string{"same", "same"}, SemanticResidual: 1, AbstentionRecords: 1, ContaminationChecks: map[string]int{}, Answers: make([]semanticConsensus, 16)}
	b, _ := json.Marshal(g)
	p := filepath.Join(t.TempDir(), "bad.json")
	_ = os.WriteFile(p, b, 0644)
	if verifySemanticGold(p) == nil {
		t.Fatal("accepted non-independent adjudicators")
	}
}
