package guardrsi

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/pkg/scorecard"
)

func TestCleanJournalScores100(t *testing.T) {
	p := writeJournal(t, []map[string]any{
		{"verdict": "ALLOW", "kind": "DECIDE", "tool": "Read"},
		{"verdict": "DENY", "kind": "DENY", "reason": "OUT_OF_TREE_WRITE", "witness": map[string]any{"policy": "workspace"}},
		{"verdict": "QUARANTINE", "kind": "QUARANTINE", "reason": "TAINTED_RESULT", "witness": map[string]any{"screen": "prompt-injection"}},
	})
	fold := FoldRows([]string{p})
	if fold.TotalRows != 3 || fold.BlankReasonOnDeny != 0 || fold.UnknownVerdict != 0 || fold.WitnesslessBlock != 0 {
		t.Fatalf("fold = %+v", fold)
	}
	if got := VerdictQuality(fold); got != 100 {
		t.Fatalf("quality = %v, want 100", got)
	}
}

func TestWitnesslessBlockLowersQuality(t *testing.T) {
	p := writeJournal(t, []map[string]any{
		{"verdict": "ALLOW", "kind": "DECIDE"},
		{"verdict": "DENY", "kind": "DENY", "reason": "OUT_OF_TREE_WRITE", "witness": map[string]any{"policy": "workspace"}},
		{"verdict": "QUARANTINE", "kind": "QUARANTINE", "reason": "TAINTED_RESULT"},
	})
	fold := FoldRows([]string{p})
	if fold.WitnesslessBlock != 1 || fold.BlankReasonOnDeny != 0 || fold.UnknownVerdict != 0 {
		t.Fatalf("fold = %+v, want one witnessless block only", fold)
	}
	if got, want := VerdictQuality(fold), 83.333; got != want {
		t.Fatalf("quality = %v, want %v", got, want)
	}
	worst := WorstBucket(fold)
	if worst.Bucket != "witnessless_block" || worst.Count != 1 || !strings.Contains(worst.Lever, "#1958") {
		t.Fatalf("worst = %+v, want witnessless #1958 bucket", worst)
	}
}

func TestUnexplainedBlockLowersQuality(t *testing.T) {
	p := writeJournal(t, []map[string]any{
		{"verdict": "ALLOW", "kind": "DECIDE"},
		{"verdict": "DENY", "kind": "DENY"},
		{"verdict": "ZALGO", "kind": "ZALGO"},
	})
	fold := FoldRows([]string{p})
	if fold.BlankReasonOnDeny != 1 || fold.UnknownVerdict != 1 {
		t.Fatalf("fold = %+v", fold)
	}
	if got, want := VerdictQuality(fold), 33.333; got != want {
		t.Fatalf("quality = %v, want %v", got, want)
	}
}

func TestRunKeepsOnlyOnGainWithWitness(t *testing.T) {
	root := t.TempDir()
	p := writeJournal(t, []map[string]any{
		{"verdict": "ALLOW", "kind": "DECIDE"},
		{"verdict": "DENY", "kind": "DENY"},
	})
	it := RunIteration(root, p, map[string]any{"ok": true, "suite": "go test ./... PASS"})
	if !it.Kept || it.MeasuredDelta <= 0 {
		t.Fatalf("iteration = %+v, want kept with strict gain", it)
	}
	if v := CheckIteration(it); len(v) != 0 {
		t.Fatalf("check violations = %v", v)
	}
}

func TestRunRevertsWithoutWitness(t *testing.T) {
	root := t.TempDir()
	p := writeJournal(t, []map[string]any{
		{"verdict": "ALLOW", "kind": "DECIDE"},
		{"verdict": "DENY", "kind": "DENY"},
	})
	it := RunIteration(root, p, nil)
	if it.Kept || it.MeasuredDelta <= 0 || !strings.Contains(it.Reason, "witness") {
		t.Fatalf("iteration = %+v, want revert without witness", it)
	}
}

func TestRunRefusesEmptyJournal(t *testing.T) {
	root := t.TempDir()
	p := writeJournal(t, nil)
	it := RunIteration(root, p, map[string]any{"ok": true})
	if it.Kept || it.Fold.TotalRows != 0 || !strings.Contains(it.Reason, "empty journal") {
		t.Fatalf("iteration = %+v, want empty-journal refusal", it)
	}
}

func TestCheckRejectsFabricatedKeptIteration(t *testing.T) {
	it := Iteration{
		Schema:        VerdictSchema,
		Kept:          true,
		MeasuredDelta: 0,
		Witness:       nil,
		Fold:          Fold{TotalRows: 0},
	}
	violations := CheckIteration(it)
	if len(violations) < 3 {
		t.Fatalf("violations = %v, want rows/delta/witness failures", violations)
	}
}

func TestScorecardPayloadShapeAndGrade(t *testing.T) {
	root := t.TempDir()
	writeGuardRSITree(t, root, true)

	payload := BuildScorecard(root)
	if payload.Schema != ScorecardSchema || payload.Corpus["guard_rsi_debt"] != 0 {
		t.Fatalf("payload = %+v", payload)
	}
	if scorecard.GradeStd(90) != "A" || scorecard.GradeStd(80) != "B" || scorecard.GradeStd(70) != "C" || scorecard.GradeStd(60) != "D" || scorecard.GradeStd(59) != "F" {
		t.Fatalf("grade boundaries changed")
	}
}

// TestScorecardGradeRidesSharedKernelTable pins guard-rsi's grade to the shared
// pkg/scorecard.GradeStd table (#1511): the emitted corpus grade must be exactly the
// shared-table letter, so a re-copied local grade table can no longer drift this card off
// the family curve. Both fixtures sit far from a grade boundary so the letter is
// unambiguous, and the gap is forced by a root-only source omission (the audit journal
// stays present) rather than the audit count, which loadContext also reads from the host.
func TestScorecardGradeRidesSharedKernelTable(t *testing.T) {
	t.Run("clean_tree_grades_off_shared_table", func(t *testing.T) {
		// Every hard KPI passes -> composite 100 -> A, debt 0.
		root := t.TempDir()
		writeGuardRSITree(t, root, true)
		assertGradeRidesTable(t, BuildScorecard(root), 100, 0)
	})
	t.Run("hard_gap_grades_off_shared_table", func(t *testing.T) {
		// Dropping the control-pane registration fails one hard realized KPI -> composite
		// ~89.1 -> B; only that gap moves the letter, so the grade is host-independent.
		root := t.TempDir()
		writeGuardRSITree(t, root, true)
		mustWrite(t, filepath.Join(root, "tools", "scorecard_control_pane.py"), []byte(""))
		assertGradeRidesTable(t, BuildScorecard(root), 89.1, 1)
	})
}

func assertGradeRidesTable(t *testing.T, p scorecard.Payload, composite float64, wantDebt int) {
	t.Helper()
	if got, want := p.Corpus["grade"], scorecard.GradeStd(composite); got != want {
		t.Fatalf("grade = %v, want shared-table GradeStd(%.1f)=%v", got, composite, want)
	}
	if got, ok := p.Corpus[DebtKey].(int); !ok || got != wantDebt {
		t.Fatalf("%s = %v, want %d", DebtKey, p.Corpus[DebtKey], wantDebt)
	}
}

func writeJournal(t *testing.T, rows []map[string]any) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "guard-audit.jsonl")
	var b strings.Builder
	for _, row := range rows {
		raw, err := json.Marshal(row)
		if err != nil {
			t.Fatalf("marshal row: %v", err)
		}
		b.Write(raw)
		b.WriteByte('\n')
	}
	if err := os.WriteFile(path, []byte(b.String()), 0o600); err != nil {
		t.Fatalf("write journal: %v", err)
	}
	return path
}

// writeGuardRSITree lays down the minimal source/skill/doc tree BuildScorecard reads. With
// withJournal it also drops one real adjudicated row so every hard KPI can pass (a clean,
// grade-A build); without it the missing journal leaves one hard realized gap (debt 1).
func writeGuardRSITree(t *testing.T, root string, withJournal bool) {
	t.Helper()
	mustWrite(t, filepath.Join(root, "cmd", "fak", "main.go"), []byte("guard-verdict-rsi guard-rsi-scorecard"))
	mustWrite(t, filepath.Join(root, "cmd", "fak", "guardrsi.go"), []byte("package main"))
	mustWrite(t, filepath.Join(root, "cmd", "fak", "guard.go"), []byte("package main"))
	mustWrite(t, filepath.Join(root, "internal", "guardrsi", "guardrsi_test.go"), []byte("package guardrsi"))
	mustWrite(t, filepath.Join(root, "tools", "guard_hop_rsi.py"), []byte("PENDING_MEASUREMENT check_plan"))
	mustWrite(t, filepath.Join(root, "tools", "scorecard_control_pane.py"), []byte("guard-rsi-scorecard guard_rsi_debt"))
	mustWrite(t, filepath.Join(root, "tools", "scorecard_baseline.json"), []byte(`{"guard_rsi":1}`))
	mustWrite(t, filepath.Join(root, ".claude", "skills", "guard-rsi-score", "SKILL.md"), []byte("skill"))
	mustWrite(t, filepath.Join(root, "docs", "fak", "guard-verdict-rsi-loop.md"), []byte("doc"))
	if withJournal {
		mustWrite(t, filepath.Join(root, ".dispatch-runs", "guard-audit", "one.jsonl"), []byte(`{"verdict":"DENY","reason":"POLICY_BLOCK","witness":{"policy":"fixture"}}`+"\n"))
	}
}

func mustWrite(t *testing.T, path string, b []byte) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatalf("mkdir %s: %v", path, err)
	}
	if err := os.WriteFile(path, b, 0o600); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
}
