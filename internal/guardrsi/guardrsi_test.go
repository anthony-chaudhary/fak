package guardrsi

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
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

// TestAdvisoryIsNotAnUnknownVerdict pins the vocabulary fix. An ADVISORY row is a
// rule on logged trial (arg_rules[].advisory) or a tool-definition prune: the call was
// ADMITTED and the would-deny was recorded alongside it. That is the most fully
// explained row the journal carries, so counting it as an unknown-verdict honesty hole
// inverted the metric — and because ADVISORY is the dominant verdict in real journals,
// the miscount was big enough to pin WorstBucket on a defect that does not exist.
//
// The nonsense verdict is kept in the same fixture so this asserts a VOCABULARY fix,
// not a blanket "stop counting unknowns": ZALGO must still score as one.
func TestAdvisoryIsNotAnUnknownVerdict(t *testing.T) {
	p := writeJournal(t, []map[string]any{
		{"verdict": "ALLOW", "kind": "DECIDE", "tool": "Read"},
		{"verdict": "ADVISORY", "kind": "DECIDE", "tool": "Bash", "reason": "SHELL_DIALECT"},
		{"verdict": "ADVISORY", "kind": "TOOL_DEFINITION_PRUNED", "tool": "Bash", "reason": "DEFAULT_DENY"},
		{"verdict": "ZALGO", "kind": "ZALGO"},
	})
	fold := FoldRows([]string{p})
	if fold.UnknownVerdict != 1 {
		t.Errorf("UnknownVerdict = %d, want 1 (only ZALGO) — an admitted-and-recorded advisory is not an honesty hole", fold.UnknownVerdict)
	}
	if fold.ByVerdict["ADVISORY"] != 2 {
		t.Errorf("ByVerdict[ADVISORY] = %d, want 2 — advisories must still be COUNTED, just not as unknowns", fold.ByVerdict["ADVISORY"])
	}
	// An advisory carries no DENY, so it must not be charged as a blank-reason or
	// witnessless block either.
	if fold.BlankReasonOnDeny != 0 || fold.WitnesslessBlock != 0 {
		t.Errorf("fold = %+v, want no block-quality charges from advisory rows", fold)
	}
}

func TestOperationalRowsAreClassifiedAndQualityNeutral(t *testing.T) {
	p := writeJournal(t, []map[string]any{
		{"verdict": "ALLOW", "kind": "DECIDE", "tool": "Read"},
		{"kind": "CONFIG_SWAP"},
		{"kind": "RESTART_HOP"},
		{"kind": "CAPABILITY_GRANT"},
		{"kind": "CHILD_EXIT", "reason": "clean_exit"},
		{"kind": "NOVEL_CONTROL"},
	})
	fold := FoldRows([]string{p})
	if fold.OperationalRows != 4 {
		t.Fatalf("OperationalRows = %d, want 4", fold.OperationalRows)
	}
	for _, kind := range []string{"CONFIG_SWAP", "RESTART_HOP", "CAPABILITY_GRANT", "CHILD_EXIT"} {
		if got := fold.ByOperationalKind[kind]; got != 1 {
			t.Errorf("ByOperationalKind[%s] = %d, want 1", kind, got)
		}
	}
	if fold.UnknownVerdict != 1 || fold.ByVerdict["NOVEL_CONTROL"] != 1 {
		t.Fatalf("fold = %+v, want only NOVEL_CONTROL classified as unknown", fold)
	}
	if got, want := VerdictQuality(fold), 50.0; got != want {
		t.Fatalf("quality = %v, want %v; operational rows must not dilute the verdict denominator", got, want)
	}
}

func TestUpstreamBadRequestIsProviderOutcomeNotUnknownVerdict(t *testing.T) {
	p := writeJournal(t, []map[string]any{
		{"verdict": "ALLOW", "kind": "DECIDE", "tool": "Read"},
		{"kind": "UPSTREAM_BAD_REQUEST", "reason": "scrubbed provider detail"},
		{"kind": "NOVEL_CONTROL"},
	})
	fold := FoldRows([]string{p})
	if fold.ProviderOutcomeRows != 1 {
		t.Fatalf("ProviderOutcomeRows = %d, want 1", fold.ProviderOutcomeRows)
	}
	if got := fold.ByProviderOutcomeKind["UPSTREAM_BAD_REQUEST"]; got != 1 {
		t.Fatalf("ByProviderOutcomeKind[UPSTREAM_BAD_REQUEST] = %d, want 1", got)
	}
	if got := fold.ByVerdict["UPSTREAM_BAD_REQUEST"]; got != 0 {
		t.Fatalf("ByVerdict[UPSTREAM_BAD_REQUEST] = %d, want 0; provider outcomes are not decision verdicts", got)
	}
	if fold.UnknownVerdict != 1 || fold.ByVerdict["NOVEL_CONTROL"] != 1 {
		t.Fatalf("fold = %+v, want only NOVEL_CONTROL classified as unknown", fold)
	}
	if got, want := VerdictQuality(fold), 50.0; got != want {
		t.Fatalf("quality = %v, want %v; provider outcomes must not dilute the verdict denominator", got, want)
	}
}

func TestCleanChildExitTwoRowWitnessDoesNotEmitUnknownVerdict(t *testing.T) {
	p := writeJournal(t, []map[string]any{
		{"verdict": "ALLOW", "kind": "DECIDE", "tool": "Read"},
		{"kind": "CHILD_EXIT", "verdict": "CHILD_EXIT", "reason": "clean_exit", "exit_code": 0},
	})
	fold := FoldRows([]string{p})
	if fold.UnknownVerdict != 0 {
		t.Fatalf("UnknownVerdict = %d, want 0", fold.UnknownVerdict)
	}
	if fold.OperationalRows != 1 {
		t.Fatalf("OperationalRows = %d, want 1", fold.OperationalRows)
	}
	if got := VerdictQuality(fold); got != 100.0 {
		t.Fatalf("VerdictQuality = %v, want 100.0", got)
	}
}

func TestProviderOutcomeClassificationDeterminism(t *testing.T) {
	p := writeJournal(t, []map[string]any{
		{"verdict": "ALLOW", "kind": "DECIDE", "tool": "Read"},
		{"kind": "UPSTREAM_BAD_REQUEST", "reason": "scrubbed provider detail one"},
		{"kind": "UPSTREAM_BAD_REQUEST", "reason": "scrubbed provider detail two"},
		{"kind": "NOVEL_CONTROL"},
	})

	first := FoldRows([]string{p})
	second := FoldRows([]string{p})
	if !reflect.DeepEqual(first, second) {
		t.Fatalf("same provider-outcome journal produced different folds:\nfirst:  %+v\nsecond: %+v", first, second)
	}
	if first.ProviderOutcomeRows != 2 {
		t.Fatalf("ProviderOutcomeRows = %d, want 2", first.ProviderOutcomeRows)
	}
	if got := first.ByProviderOutcomeKind["UPSTREAM_BAD_REQUEST"]; got != 2 {
		t.Fatalf("ByProviderOutcomeKind[UPSTREAM_BAD_REQUEST] = %d, want 2", got)
	}
	if first.UnknownVerdict != 1 || first.ByVerdict["NOVEL_CONTROL"] != 1 {
		t.Fatalf("fold = %+v, want only NOVEL_CONTROL classified as unknown", first)
	}
	if got := first.ByVerdict["UPSTREAM_BAD_REQUEST"]; got != 0 {
		t.Fatalf("ByVerdict[UPSTREAM_BAD_REQUEST] = %d, want 0; provider outcomes are not decision verdicts", got)
	}
}

func TestProviderOutcomeClassificationEdges(t *testing.T) {
	tests := []struct {
		name               string
		rows               []map[string]any
		wantProviderRows   int
		wantUnknownVerdict int
		wantByVerdict      map[string]int
		wantQuality        float64
	}{
		{
			name: "normalized lowercase and whitespace kind",
			rows: []map[string]any{
				{"verdict": "ALLOW", "kind": "DECIDE"},
				{"kind": " \tupstream_bad_request\n", "reason": "scrubbed provider detail"},
			},
			wantProviderRows: 1,
			wantByVerdict:    map[string]int{"ALLOW": 1},
			wantQuality:      100,
		},
		{
			name: "two provider rows count additively",
			rows: []map[string]any{
				{"verdict": "ALLOW", "kind": "DECIDE"},
				{"kind": "UPSTREAM_BAD_REQUEST"},
				{"kind": "UPSTREAM_BAD_REQUEST"},
			},
			wantProviderRows: 2,
			wantByVerdict:    map[string]int{"ALLOW": 1},
			wantQuality:      100,
		},
		{
			name: "provider kind wins over conflicting explicit verdict",
			rows: []map[string]any{
				{"verdict": "ALLOW", "kind": "DECIDE"},
				{"kind": "UPSTREAM_BAD_REQUEST", "verdict": "DENY"},
			},
			wantProviderRows: 1,
			wantByVerdict:    map[string]int{"ALLOW": 1},
			wantQuality:      100,
		},
		{
			name: "near match remains unknown",
			rows: []map[string]any{
				{"verdict": "ALLOW", "kind": "DECIDE"},
				{"kind": "UPSTREAM_BAD_REQUEST_EXTRA"},
			},
			wantUnknownVerdict: 1,
			wantByVerdict: map[string]int{
				"ALLOW":                      1,
				"UPSTREAM_BAD_REQUEST_EXTRA": 1,
			},
			wantQuality: 50,
		},
		{
			name: "provider token in verdict without kind remains unknown",
			rows: []map[string]any{
				{"verdict": "ALLOW", "kind": "DECIDE"},
				{"verdict": "UPSTREAM_BAD_REQUEST"},
			},
			wantUnknownVerdict: 1,
			wantByVerdict: map[string]int{
				"ALLOW":                1,
				"UPSTREAM_BAD_REQUEST": 1,
			},
			wantQuality: 50,
		},
		{
			name: "provider only fold has zero verdict denominator",
			rows: []map[string]any{
				{"kind": "UPSTREAM_BAD_REQUEST"},
			},
			wantProviderRows: 1,
			wantByVerdict:    map[string]int{},
			wantQuality:      0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := writeJournal(t, tt.rows)
			fold := FoldRows([]string{p})
			if fold.TotalRows != len(tt.rows) {
				t.Fatalf("TotalRows = %d, want %d", fold.TotalRows, len(tt.rows))
			}
			if fold.ProviderOutcomeRows != tt.wantProviderRows {
				t.Errorf("ProviderOutcomeRows = %d, want %d", fold.ProviderOutcomeRows, tt.wantProviderRows)
			}
			if got := fold.ByProviderOutcomeKind["UPSTREAM_BAD_REQUEST"]; got != tt.wantProviderRows {
				t.Errorf("ByProviderOutcomeKind[UPSTREAM_BAD_REQUEST] = %d, want %d", got, tt.wantProviderRows)
			}
			if fold.UnknownVerdict != tt.wantUnknownVerdict {
				t.Errorf("UnknownVerdict = %d, want %d", fold.UnknownVerdict, tt.wantUnknownVerdict)
			}
			if len(fold.ByVerdict) != len(tt.wantByVerdict) {
				t.Errorf("ByVerdict = %v, want %v", fold.ByVerdict, tt.wantByVerdict)
			}
			for verdict, want := range tt.wantByVerdict {
				if got := fold.ByVerdict[verdict]; got != want {
					t.Errorf("ByVerdict[%s] = %d, want %d", verdict, got, want)
				}
			}
			if fold.BlankReasonOnDeny != 0 {
				t.Errorf("BlankReasonOnDeny = %d, want 0", fold.BlankReasonOnDeny)
			}
			if got := VerdictQuality(fold); got != tt.wantQuality {
				t.Errorf("quality = %v, want %v", got, tt.wantQuality)
			}
		})
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

func TestScorecardRecognizesBothGuardRSICommandSpellings(t *testing.T) {
	for _, command := range []string{
		"go run ./cmd/fak guard-rsi-scorecard --json",
		"go run ./cmd/fak score guard-rsi --json",
	} {
		t.Run(command, func(t *testing.T) {
			root := t.TempDir()
			writeGuardRSITree(t, root, true)
			mustWrite(t, filepath.Join(root, "tools", "scorecard_control_pane.py"), []byte(command))
			if got := BuildScorecard(root).Corpus[DebtKey]; got != 0 {
				t.Fatalf("%q registration debt = %v, want 0", command, got)
			}
		})
	}

	root := t.TempDir()
	writeGuardRSITree(t, root, true)
	mustWrite(t, filepath.Join(root, "tools", "scorecard_control_pane.py"), []byte("unrelated-scorecard"))
	if got := BuildScorecard(root).Corpus[DebtKey]; got != 1 {
		t.Fatalf("missing registration debt = %v, want 1", got)
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
