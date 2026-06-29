package rsiloop

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/shipgate"
)

func variantTask(id string, pass bool, points float64, turns int, evidenceID string, confirmed bool) TaskWitness {
	return TaskWitness{
		TaskID:     id,
		SpecPassed: pass,
		SpecPoints: points,
		Turns:      turns,
		Evidence: DOSEvidence{
			ID:        evidenceID,
			Subject:   "task:" + id,
			Confirmed: confirmed,
			Summary:   "dos witness fixture",
		},
	}
}

func testVariant() LoopVariant {
	return LoopVariant{
		ID:          "prompt-tighten-gate",
		Kind:        VariantPrompt,
		Description: "tighten the loop prompt around the shipgate check",
		Diff:        "--- baseline\n+++ candidate\n@@ prompt\n+check shipgate before acting\n",
	}
}

func TestStaticLoopVariantProposer_EmitsDefensiveCandidateDiffs(t *testing.T) {
	p := StaticLoopVariantProposer{testVariant()}
	got, err := p.ProposeLoopVariants(LoopConfig{Prompt: "baseline"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got[0].ID != testVariant().ID || got[0].Diff == "" {
		t.Fatalf("proposed variants = %+v, want the configured candidate diff", got)
	}

	got[0].ID = "mutated"
	again, err := p.ProposeLoopVariants(LoopConfig{}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if again[0].ID != testVariant().ID {
		t.Fatalf("proposer returned aliased candidates: %+v", again)
	}
}

// TestLoopVariant_RevertsWithoutWitnessedSpecGain pins #1177's first acceptance
// criterion: proposing a variant is not enough. With the same spec/oracle points as
// the baseline, the existing shipgate keep-bit REVERTS and the archive stays empty.
func TestLoopVariant_RevertsWithoutWitnessedSpecGain(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "variants.jsonl")
	a, err := NewLoopVariantArchive(path)
	if err != nil {
		t.Fatal(err)
	}
	defer a.Close()

	baseline := TaskSetWitness{Ref: "main", Tasks: []TaskWitness{
		variantTask("refund-readonly", true, 1, 9, "dos-base-1", true),
	}}
	candidate := TaskSetWitness{Ref: "candidate", Tasks: []TaskWitness{
		variantTask("refund-readonly", true, 1, 8, "dos-cand-1", true),
	}}

	rec, err := ArchiveLoopVariant(testVariant(), baseline, candidate, a)
	if err != nil {
		t.Fatal(err)
	}
	if rec.Decision != shipgate.REVERT.String() || rec.Kept {
		t.Fatalf("no spec gain must REVERT, got decision=%s kept=%v", rec.Decision, rec.Kept)
	}
	if rec.Before.Points != rec.After.Points {
		t.Fatalf("fixture error: before=%v after=%v should be equal", rec.Before.Points, rec.After.Points)
	}
	if rows, err := ReadLoopVariantArchive(path); err != nil || len(rows) != 0 {
		t.Fatalf("archive rows = %d, err=%v; reverted variant must not archive", len(rows), err)
	}
}

// TestLoopVariant_MetricOnlyTurnWinCannotKeep is the anti-goodhart acceptance
// criterion: fewer turns from skipping the gate is recorded for audit, but the
// objective is the spec/oracle fold. A failed oracle yields SuiteGreen=false and
// zero spec points, so the keep-bit cannot be moved by the raw metric.
func TestLoopVariant_MetricOnlyTurnWinCannotKeep(t *testing.T) {
	baseline := TaskSetWitness{Ref: "main", Tasks: []TaskWitness{
		variantTask("destructive-op-refusal", true, 1, 12, "dos-base-refusal", true),
	}}
	candidate := TaskSetWitness{Ref: "candidate", Tasks: []TaskWitness{
		variantTask("destructive-op-refusal", false, 10, 1, "dos-cand-skipped-gate", true),
	}}

	rec, err := EvaluateLoopVariant(testVariant(), baseline, candidate)
	if err != nil {
		t.Fatal(err)
	}
	if rec.Kept || rec.Decision != shipgate.REVERT.String() {
		t.Fatalf("metric-only win kept=%v decision=%s, want REVERT", rec.Kept, rec.Decision)
	}
	if rec.Tasks[0].Turns >= baseline.Tasks[0].Turns {
		t.Fatalf("fixture error: candidate should be the raw metric win")
	}
	if rec.After.Points != 0 {
		t.Fatalf("failed oracle contributed %v spec points, want 0", rec.After.Points)
	}
	if rec.Witness.SuiteGreen {
		t.Fatal("failed oracle must make the shipgate suite/spec rung red")
	}
}

// TestLoopVariantArchive_RecordsDOSEvidenceForKeptVariant pins the archive
// criterion: only a witnessed-better variant is kept, and the stepping-stone row
// carries the dos evidence that justified the keep.
func TestLoopVariantArchive_RecordsDOSEvidenceForKeptVariant(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "variants.jsonl")
	a, err := NewLoopVariantArchive(path)
	if err != nil {
		t.Fatal(err)
	}

	baseline := TaskSetWitness{Ref: "main@aaa", Tasks: []TaskWitness{
		variantTask("refund-readonly", true, 1, 9, "dos-base-1", true),
		variantTask("injection-refusal", false, 0, 7, "dos-base-2", true),
	}}
	candidate := TaskSetWitness{Ref: "candidate@bbb", Tasks: []TaskWitness{
		variantTask("refund-readonly", true, 1, 9, "dos-cand-1", true),
		variantTask("injection-refusal", true, 1, 8, "dos-cand-2", true),
	}}

	rec, err := ArchiveLoopVariant(testVariant(), baseline, candidate, a)
	if err != nil {
		t.Fatal(err)
	}
	if err := a.Close(); err != nil {
		t.Fatal(err)
	}
	if !rec.Kept || rec.Decision != shipgate.KEEP.String() {
		t.Fatalf("witnessed spec gain should KEEP, got decision=%s kept=%v", rec.Decision, rec.Kept)
	}

	rows, err := ReadLoopVariantArchive(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 1 {
		b, _ := os.ReadFile(path)
		t.Fatalf("archive rows=%d, want 1; bytes=%q", len(rows), b)
	}
	row := rows[0]
	if row.Schema != LoopVariantArchiveSchema {
		t.Fatalf("schema=%q, want %q", row.Schema, LoopVariantArchiveSchema)
	}
	if row.Variant.ID != testVariant().ID || !row.Kept {
		t.Fatalf("archive row did not preserve kept variant: %+v", row)
	}
	if row.Before.Points != 1 || row.After.Points != 2 {
		t.Fatalf("archive objective before/after = %.1f/%.1f, want 1/2", row.Before.Points, row.After.Points)
	}
	if len(row.DOSEvidence) != 4 {
		t.Fatalf("dos evidence count=%d, want 4", len(row.DOSEvidence))
	}
	if row.DOSEvidence[1].ID != "dos-base-2" || !row.DOSEvidence[1].Confirmed {
		t.Fatalf("archive did not carry confirming baseline dos evidence: %+v", row.DOSEvidence)
	}
	if row.DOSEvidence[3].ID != "dos-cand-2" || !row.DOSEvidence[3].Confirmed {
		t.Fatalf("archive did not carry confirming dos evidence: %+v", row.DOSEvidence)
	}
	if row.CandidateRef != "candidate@bbb" || row.BaselineRef != "main@aaa" {
		t.Fatalf("archive refs = %q/%q", row.BaselineRef, row.CandidateRef)
	}
}

func TestLoopVariantArchive_RefusesDirectRevertAppend(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "variants.jsonl")
	a, err := NewLoopVariantArchive(path)
	if err != nil {
		t.Fatal(err)
	}
	defer a.Close()

	err = a.Append(LoopVariantRecord{Decision: shipgate.REVERT.String(), Kept: false})
	if err == nil {
		t.Fatal("direct append of a reverted variant succeeded; archive must admit kept stepping stones only")
	}
	rows, readErr := ReadLoopVariantArchive(path)
	if readErr != nil {
		t.Fatal(readErr)
	}
	if len(rows) != 0 {
		t.Fatalf("direct revert append wrote %d row(s), want 0", len(rows))
	}
}

func TestLoopVariantArchive_RefusesFabricatedKeepAppend(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "variants.jsonl")
	a, err := NewLoopVariantArchive(path)
	if err != nil {
		t.Fatal(err)
	}
	defer a.Close()

	err = a.Append(LoopVariantRecord{
		Schema:   LoopVariantArchiveSchema,
		Variant:  testVariant(),
		Decision: shipgate.KEEP.String(),
		Kept:     true,
		DOSEvidence: []DOSEvidence{{
			ID:        "dos-cand",
			Confirmed: true,
		}},
	})
	if err == nil {
		t.Fatal("direct append of fabricated KEEP succeeded; archive must require shipgate's keep-bit")
	}
	rows, readErr := ReadLoopVariantArchive(path)
	if readErr != nil {
		t.Fatal(readErr)
	}
	if len(rows) != 0 {
		t.Fatalf("fabricated keep append wrote %d row(s), want 0", len(rows))
	}
}

func TestLoopVariantArchive_RefusesKeptRecordWithoutConfirmedEvidence(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "variants.jsonl")
	a, err := NewLoopVariantArchive(path)
	if err != nil {
		t.Fatal(err)
	}
	defer a.Close()

	baseline := TaskSetWitness{Ref: "main", Tasks: []TaskWitness{
		variantTask("task", false, 0, 10, "dos-base", true),
	}}
	candidate := TaskSetWitness{Ref: "candidate", Tasks: []TaskWitness{
		variantTask("task", true, 1, 10, "dos-cand", true),
	}}
	rec, err := EvaluateLoopVariant(testVariant(), baseline, candidate)
	if err != nil {
		t.Fatal(err)
	}
	if !rec.Witness.Kept() {
		t.Fatalf("fixture should produce shipgate keep-bit: %+v", rec.Witness)
	}
	rec.DOSEvidence = nil
	if err := a.Append(rec); err == nil {
		t.Fatal("append of kept record without dos evidence succeeded")
	}
}

func TestLoopVariant_RevertsWithoutConfirmedDOSEvidence(t *testing.T) {
	baseline := TaskSetWitness{Ref: "main", Tasks: []TaskWitness{
		variantTask("task", false, 0, 10, "dos-base", true),
	}}
	candidate := TaskSetWitness{Ref: "candidate", Tasks: []TaskWitness{
		variantTask("task", true, 1, 10, "dos-cand", false),
	}}

	rec, err := EvaluateLoopVariant(testVariant(), baseline, candidate)
	if err != nil {
		t.Fatal(err)
	}
	if rec.Kept || rec.Decision != shipgate.REVERT.String() {
		t.Fatalf("unconfirmed dos evidence must REVERT, got decision=%s kept=%v", rec.Decision, rec.Kept)
	}
	if rec.Witness.TruthClean {
		t.Fatal("unconfirmed dos evidence must make the shipgate truth rung dirty")
	}
}

func TestLoopVariant_RequiresFixedTaskSet(t *testing.T) {
	baseline := TaskSetWitness{Tasks: []TaskWitness{variantTask("a", true, 1, 1, "dos-a", true)}}
	candidate := TaskSetWitness{Tasks: []TaskWitness{variantTask("b", true, 1, 1, "dos-b", true)}}
	if _, err := EvaluateLoopVariant(testVariant(), baseline, candidate); err == nil {
		t.Fatal("expected task-set mismatch error")
	}
}
