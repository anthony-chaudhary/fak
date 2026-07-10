package trajctl

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// The committed ledger fixture is rewritten from the canonical task with the
// package's shared -update flag (see updateGolden in curve_test.go). Run:
// go test ./internal/trajctl -run TestBenchLedgerFixtureGolden -update

// fixtureMillis is a fixed stamp for the committed fixture. Pure code never reads
// the wall clock, so the stamp is injected and the golden bytes stay stable.
const fixtureMillis int64 = 1751414400000 // 2025-07-02T00:00:00Z

// canonicalObjective declares the swebench task's phases the fixture scores
// against; its phase ids match the task's sub-goal ids.
func canonicalObjective() Objective {
	return Objective{
		ID:        "swebench/astropy__astropy-12907",
		Statement: "SWE-bench task astropy__astropy-12907: make the failing tests pass without regressing the passing ones",
		Plan: []PlanPhase{
			{ID: "patch_applies", Title: "generated patch applies to the repo"},
			{ID: "fail_to_pass", Title: "the target failing tests now pass"},
			{ID: "pass_to_pass", Title: "the previously passing tests still pass"},
		},
		Scorers: []string{BenchScorerMethod},
		Status:  StatusActive,
	}
}

// canonicalResolvedTask is a resolved swebench instance: every harness sub-goal
// verified, so the progress curve rises to 1.0 and agrees with resolve=true.
func canonicalResolvedTask() BenchTask {
	return BenchTask{
		ObjectiveID: "swebench/astropy__astropy-12907",
		Benchmark:   "swebench",
		TaskID:      "astropy__astropy-12907",
		SubGoals: []BenchSubGoal{
			{ID: "patch_applies", Title: "generated patch applies to the repo", Verified: true},
			{ID: "fail_to_pass", Title: "the target failing tests now pass", Verified: true},
			{ID: "pass_to_pass", Title: "the previously passing tests still pass", Verified: true},
		},
		Resolved: true,
	}
}

func TestBenchTaskProgressRowsResolved(t *testing.T) {
	rows, err := canonicalResolvedTask().ProgressRows(fixtureMillis)
	if err != nil {
		t.Fatal(err)
	}
	wantValues := []float64{1.0 / 3.0, 2.0 / 3.0, 1.0}
	if len(rows) != len(wantValues) {
		t.Fatalf("got %d rows, want %d", len(rows), len(wantValues))
	}
	for i, row := range rows {
		if row.Value != wantValues[i] {
			t.Errorf("row %d value = %v, want %v", i, row.Value, wantValues[i])
		}
		if row.Witness != W3 {
			t.Errorf("row %d witness = %q, want W3 (harness-verified)", i, row.Witness)
		}
		if row.Method != BenchScorerMethod || row.Version != BenchScorerVersion {
			t.Errorf("row %d method/version = %q/%q", i, row.Method, row.Version)
		}
		if len(row.Evidence) != i+1 {
			t.Errorf("row %d carries %d evidence refs, want %d (one per credited sub-goal)", i, len(row.Evidence), i+1)
		}
		if err := validateScore(row); err != nil {
			t.Errorf("row %d does not validate: %v", i, err)
		}
	}
	// The final row agrees with the harness resolve bit.
	if final := rows[len(rows)-1]; final.Value != 1.0 {
		t.Fatalf("resolved task must end at 1.0, got %v", final.Value)
	}
}

func TestBenchTaskProgressRowsUnresolved(t *testing.T) {
	task := canonicalResolvedTask()
	task.SubGoals[2].Verified = false // pass_to_pass regressed
	task.Resolved = false
	rows, err := task.ProgressRows(fixtureMillis)
	if err != nil {
		t.Fatal(err)
	}
	wantValues := []float64{1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0}
	for i, row := range rows {
		if row.Value != wantValues[i] {
			t.Errorf("row %d value = %v, want %v", i, row.Value, wantValues[i])
		}
	}
	// The final row agrees with resolve=false: it must be strictly below 1.0.
	if final := rows[len(rows)-1]; final.Value >= 1.0 {
		t.Fatalf("unresolved task must end below 1.0, got %v", final.Value)
	}
	// The final (unverified) sub-goal earns no evidence pointer.
	if got := len(rows[2].Evidence); got != 2 {
		t.Fatalf("unresolved final row carries %d evidence refs, want 2", got)
	}
}

func TestBenchTaskRefusesResolveContradiction(t *testing.T) {
	// Resolved=true but a sub-goal unverified — the harness cannot claim resolve
	// while a sub-goal is unwitnessed.
	over := canonicalResolvedTask()
	over.SubGoals[1].Verified = false
	if _, err := over.ProgressRows(fixtureMillis); err == nil {
		t.Fatal("resolve=true with an unverified sub-goal must be refused")
	}

	// Resolved=false but every sub-goal verified — the curve would reach 1.0 while
	// the harness denies resolution, a contradiction the adapter refuses.
	under := canonicalResolvedTask()
	under.Resolved = false
	if _, err := under.ProgressRows(fixtureMillis); err == nil {
		t.Fatal("resolve=false with all sub-goals verified must be refused")
	}
}

func TestBenchTaskProgressRowsErrors(t *testing.T) {
	cases := []struct {
		name   string
		mutate func(*BenchTask)
	}{
		{"missing objective id", func(t *BenchTask) { t.ObjectiveID = "" }},
		{"missing task id", func(t *BenchTask) { t.TaskID = "" }},
		{"no sub-goals", func(t *BenchTask) { t.SubGoals = nil }},
		{"missing sub-goal id", func(t *BenchTask) { t.SubGoals[0].ID = "" }},
		{"duplicate sub-goal", func(t *BenchTask) { t.SubGoals[1].ID = t.SubGoals[0].ID }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			task := canonicalResolvedTask()
			tc.mutate(&task)
			if _, err := task.ProgressRows(fixtureMillis); err == nil {
				t.Fatalf("%s must be refused", tc.name)
			}
		})
	}
}

func TestBenchProgressScorerViaRegistry(t *testing.T) {
	reg := NewRegistry()
	if err := reg.Register(BenchProgressScorer{}); err != nil {
		t.Fatal(err)
	}
	scorer, ok := reg.Get(BenchScorerMethod)
	if !ok {
		t.Fatalf("scorer %q not registered", BenchScorerMethod)
	}
	obj := canonicalObjective()
	// A fixture resolver that witnesses two of the three phases.
	verified := map[string]bool{"patch_applies": true, "fail_to_pass": true}
	win := EvidenceWindow{
		Resolve: func(ref EvidenceRef) EvidenceStatus {
			if ref.Kind == evidenceKindHarness && verified[ref.Detail] {
				return EvidenceVerified
			}
			return EvidenceUnknown
		},
		UnixMillis: fixtureMillis,
	}
	rows := scorer.Score(obj, win)
	if len(rows) != 1 {
		t.Fatalf("got %d rows, want 1 cumulative row", len(rows))
	}
	if rows[0].Value != 2.0/3.0 {
		t.Fatalf("value = %v, want 2/3", rows[0].Value)
	}
	if rows[0].Witness != W3 || len(rows[0].Evidence) != 2 {
		t.Fatalf("row = %+v, want W3 with 2 evidence refs", rows[0])
	}
	if err := validateScore(rows[0]); err != nil {
		t.Fatal(err)
	}
	// A nil resolver credits nothing (fail-closed).
	if got := (BenchProgressScorer{}).Score(obj, EvidenceWindow{}); got[0].Value != 0 {
		t.Fatalf("nil resolver must score 0, got %v", got[0].Value)
	}
}

// buildCanonicalLedger renders the fixture ledger: the objective row followed by
// the resolved task's progress-rate rows, each schema-validated.
func buildCanonicalLedger(t *testing.T) []byte {
	t.Helper()
	rows, err := canonicalResolvedTask().ProgressRows(fixtureMillis)
	if err != nil {
		t.Fatal(err)
	}
	var buf bytes.Buffer
	write := func(row Row) {
		if err := Validate(row); err != nil {
			t.Fatalf("fixture row does not validate: %v", err)
		}
		b, err := json.Marshal(row)
		if err != nil {
			t.Fatal(err)
		}
		buf.Write(b)
		buf.WriteByte('\n')
	}
	write(ObjectiveRecord(canonicalObjective()))
	for _, s := range rows {
		write(ScoreRecord(s))
	}
	return buf.Bytes()
}

// TestBenchLedgerFixtureGolden is #2547's witness: one instrumented benchmark
// run's ledger, committed as a fixture. It pins the exact bytes the adapter
// emits for a resolved swebench task and re-checks that the fixture folds back to
// an objective plus a rising progress curve whose final row agrees with resolve.
func TestBenchLedgerFixtureGolden(t *testing.T) {
	got := buildCanonicalLedger(t)
	path := filepath.Join("testdata", "bench_progress_swebench.jsonl")
	if *updateGolden {
		if err := os.WriteFile(path, got, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	want, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read golden fixture (run with -update-golden to create): %v", err)
	}
	if !bytes.Equal(got, want) {
		t.Fatalf("golden mismatch:\n got:\n%s\nwant:\n%s", got, want)
	}

	// The committed fixture must fold back to the same curve.
	st := Fold(ParseLedger(string(want)))
	scores := st.ScoresFor("swebench/astropy__astropy-12907")
	if len(scores) != 3 {
		t.Fatalf("folded %d score rows, want 3", len(scores))
	}
	if final := scores[len(scores)-1]; final.Value != 1.0 || final.Witness != W3 {
		t.Fatalf("final folded row = value %v witness %q, want 1.0/W3 (agrees with resolve)", final.Value, final.Witness)
	}
}
