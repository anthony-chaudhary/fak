package nightrun

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// --- cross-box novelty isolation (the per-THIS-box promise) -----------------

func TestLastCollectedCrossBoxIsolation(t *testing.T) {
	// A datum collected on a DIFFERENT box must not count as collected here, so it
	// stays novel on this box. This guards the load-bearing box filter in
	// lastCollected — dropping it would silently suppress novelty fleet-wide.
	rows := []CollectRow{
		{Schema: CollectSchema, TaskID: "T", Box: "a100", Outcome: string(OutcomeCollected),
			Date: "2026-06-25", GeneratedAt: "2026-06-25T00:00:00Z"},
	}
	if _, ok := lastCollected(rows, "T", "mac"); ok {
		t.Error("a row collected on a100 must NOT count as collected on mac")
	}
	if _, ok := lastCollected(rows, "T", "a100"); !ok {
		t.Error("a row collected on a100 MUST count when querying a100")
	}
	// An empty-box row must not wildcard-suppress novelty on a real box.
	boxless := []CollectRow{
		{Schema: CollectSchema, TaskID: "T", Box: "", Outcome: string(OutcomeCollected),
			Date: "2026-06-25", GeneratedAt: "2026-06-25T00:00:00Z"},
	}
	if _, ok := lastCollected(boxless, "T", "mac"); ok {
		t.Error("a box-less row must not count as collected on a named box")
	}
}

// --- apply-path ledger row provenance (the honesty contract) ----------------

func TestRunLoopApplyRecordsObservedFields(t *testing.T) {
	caps := Capabilities{Box: "ci", GPU: "none", Net: true, Creds: map[string]bool{}}
	now := mustTime(t, "2026-06-27T00:00:00Z")
	tasks := []Task{
		{ID: "off-a", Value: ValueCoverage, Run: "echo a"},
		{ID: "off-b", Value: ValueWitness, Run: "echo b"},
	}
	// off-a "collects" with an observed number; off-b "times out" with no number.
	outcomes := map[string]struct {
		o Outcome
		n string
	}{
		"off-a": {OutcomeCollected, "143 GB/s"},
		"off-b": {OutcomeTimeout, ""},
	}
	var rows []CollectRow
	_, err := RunLoop(context.Background(), RunOptions{
		Root: "/repo", Caps: caps, Tasks: tasks, Now: now,
		Apply: true, Loop: true, Max: 0,
		ReadLedger: func() []CollectRow { return rows },
		AppendRow:  func(r CollectRow) error { rows = append(rows, r); return nil },
		Executor: func(_ context.Context, tk Task, _ string) (Outcome, string, time.Duration, error) {
			oc := outcomes[tk.ID]
			return oc.o, oc.n, time.Second, nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 2 {
		t.Fatalf("want 2 ledger rows, got %d", len(rows))
	}
	byID := map[string]CollectRow{}
	for _, r := range rows {
		byID[r.TaskID] = r
	}
	a := byID["off-a"]
	if a.Outcome != string(OutcomeCollected) || a.Number != "143 GB/s" || a.Command != "echo a" || a.Value != string(ValueCoverage) || a.Box != "ci" {
		t.Errorf("collected row provenance wrong: %+v", a)
	}
	b := byID["off-b"]
	if b.Outcome != string(OutcomeTimeout) {
		t.Errorf("timeout row should record outcome=timeout, got %q", b.Outcome)
	}
	if b.Number != "" {
		t.Errorf("a timeout must NEVER fabricate a number, got %q", b.Number)
	}
}

// --- overlay loader branches (the operator/agent extension surface) ----------

func TestBacklogOverlay(t *testing.T) {
	write := func(name, body string) string {
		p := filepath.Join(t.TempDir(), name)
		if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
		return p
	}

	// (a) a valid additive task appears with Source==overlay and a defaulted Value.
	ok := write("ok.json", `[{"id":"overlay-new","run":"echo hi","acceptance":"a line"}]`)
	tasks, err := Backlog(ok)
	if err != nil {
		t.Fatalf("valid overlay: %v", err)
	}
	var found *Task
	for i := range tasks {
		if tasks[i].ID == "overlay-new" {
			found = &tasks[i]
		}
	}
	if found == nil {
		t.Fatal("overlay task not in backlog")
	}
	if found.Source != SourceOverlay {
		t.Errorf("overlay task Source = %q, want overlay", found.Source)
	}
	if found.Value != ValueCoverage {
		t.Errorf("overlay task with no value should default to coverage, got %q", found.Value)
	}

	// (b) an id colliding with a built-in fails LOUD.
	dup := write("dup.json", `[{"id":"bench-ablate","run":"echo x"}]`)
	if _, err := Backlog(dup); err == nil {
		t.Error("an overlay id colliding with a built-in must fail loud")
	}

	// (c) malformed JSON fails.
	bad := write("bad.json", `{ not json`)
	if _, err := Backlog(bad); err == nil {
		t.Error("malformed overlay JSON must error")
	}

	// (d) a task with no id fails.
	noid := write("noid.json", `[{"run":"echo x"}]`)
	if _, err := Backlog(noid); err == nil {
		t.Error("an overlay task with no id must error")
	}

	// (e) a task with no run command fails.
	norun := write("norun.json", `[{"id":"x"}]`)
	if _, err := Backlog(norun); err == nil {
		t.Error("an overlay task with no run must error")
	}

	// (f) a missing overlay path is fine — built-ins still load.
	tasks2, err := Backlog(filepath.Join(t.TempDir(), "absent.json"))
	if err != nil || len(tasks2) == 0 {
		t.Errorf("a missing overlay path must be fine with built-ins; err=%v n=%d", err, len(tasks2))
	}
}

// --- parseNumber units + ageDays fallback -----------------------------------

func TestParseNumberUnits(t *testing.T) {
	cases := map[string]string{
		"mul_mv_q8_0: 143 GB/s sustained": "143 GB/s",
		"decode 12.4 ms/tok":              "12.4 ms/tok",
		"throughput 8.0 it/s":             "8.0 it/s",
		"no unit-bearing token here":      "",
	}
	for in, want := range cases {
		if got := parseNumber(in); got != want {
			t.Errorf("parseNumber(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestAgeDaysDateFallback(t *testing.T) {
	now := mustTime(t, "2026-06-27T00:00:00Z")
	// date-only (no/garbled generated_at) still yields a valid age.
	if age, ok := ageDays(now, "", "2026-06-20"); !ok || age < 6.9 || age > 7.1 {
		t.Errorf("date-only ageDays = %.2f ok=%v, want ~7 days valid", age, ok)
	}
	// neither parseable => invalid.
	if _, ok := ageDays(now, "", ""); ok {
		t.Error("ageDays with no timestamps must be invalid")
	}
}

// --- env-prefix splitting (shell-agnostic env application) -------------------

func TestSplitEnvPrefix(t *testing.T) {
	env, rest := splitEnvPrefix("FAK_METAL_DECODE=1 FAK_X=2 go run ./cmd/x -flag v")
	if len(env) != 2 || env[0] != "FAK_METAL_DECODE=1" || env[1] != "FAK_X=2" {
		t.Errorf("env prefix = %v, want the two assignments", env)
	}
	if rest != "go run ./cmd/x -flag v" {
		t.Errorf("residual = %q, want the command without the env prefix", rest)
	}
	env2, rest2 := splitEnvPrefix("go run ./cmd/y")
	if env2 != nil || rest2 != "go run ./cmd/y" {
		t.Errorf("a command with no env prefix must pass through; got env=%v rest=%q", env2, rest2)
	}
}
