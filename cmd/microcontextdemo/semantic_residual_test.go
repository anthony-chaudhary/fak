package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
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

func TestToolNeedV2PromptSeparatesAnswerAndActionEvidence(t *testing.T) {
	x := semanticRecord{ID: "x", Title: "Implement parser", Body: "Add the parser described here."}
	prompt := adjudicationPrompt(x, semanticToolPromptV2)
	for _, want := range []string{"answer_evidence", "action_evidence", "packet->none", "repository->read_only", "live->current_state"} {
		if !strings.Contains(prompt, want) {
			t.Fatalf("prompt missing %q", want)
		}
	}
}

func TestDecodeSemanticJudgmentPreservesEvidenceAxes(t *testing.T) {
	j, err := decodeSemanticJudgment(`{"id":"x","semantic_need":"semantic","tool_need":"read_only","actionability":"actionable","answer_evidence":"packet","action_evidence":"repository","confidence":0.8,"evidence":["Add parser"],"rationale":"repo state matters"}`)
	if err != nil {
		t.Fatal(err)
	}
	if j.AnswerEvidence != "packet" || j.ActionEvidence != "repository" || j.ToolNeed != "read_only" {
		t.Fatalf("unexpected judgment: %+v", j)
	}
}

func TestSemanticTripleFoldIsPacketBoundAndReproducible(t *testing.T) {
	root := filepath.Join("..", "..", "experiments", "microcontext")
	out1 := filepath.Join(t.TempDir(), "fold-a.json")
	out2 := filepath.Join(t.TempDir(), "fold-b.json")
	args := []string{
		filepath.Join(root, "s8i-semantic-packet-2026-08-10.json"),
		filepath.Join(root, "s8i-adjudicator-a-2026-08-10.json"),
		filepath.Join(root, "s8i-adjudicator-openai-2026-08-10.json"),
		filepath.Join(root, "s8m-adjudicator-openai-2026-08-10.json"),
		filepath.Join(root, "s8m-adjudicator-groq-2026-08-10.json"),
	}
	if err := foldSemanticToolTriple(args[0], args[1], args[2], args[3], args[4], out1); err != nil {
		t.Fatal(err)
	}
	if err := foldSemanticToolTriple(args[0], args[1], args[2], args[3], args[4], out2); err != nil {
		t.Fatal(err)
	}
	a, _ := os.ReadFile(out1)
	b, _ := os.ReadFile(out2)
	if string(a) != string(b) {
		t.Fatal("triple fold is not deterministic")
	}
	var f semanticTripleFold
	if err := json.Unmarshal(a, &f); err != nil {
		t.Fatal(err)
	}
	if f.Changes.OldExactAgreement != 0.21875 || f.Changes.NewResolved != 32 || f.Counts["read_only"] == 0 || f.Counts["current_state"] == 0 {
		t.Fatalf("unexpected fold: %+v", f)
	}
	if len(f.SourceSHA256) != 4 || f.GoldSHA256 == "" {
		t.Fatalf("missing provenance: %+v", f)
	}
}
