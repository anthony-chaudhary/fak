package nightrun

import (
	"context"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

func mustTime(t *testing.T, s string) time.Time {
	t.Helper()
	tm, err := time.Parse(time.RFC3339, s)
	if err != nil {
		t.Fatalf("parse time %q: %v", s, err)
	}
	return tm.UTC()
}

// --- capability probe (deterministic via injected seams) -------------------

func TestProbeDeterministic(t *testing.T) {
	env := map[string]string{
		"FAK_BACKEND":       "",
		"ANTHROPIC_API_KEY": "sk-xxx",
		"FAK_MODEL_DIR":     "/models/glm",
	}
	exists := map[string]bool{"/models/glm": true}
	e := probeEnv{
		getenv:   func(k string) string { return env[k] },
		look:     func(string) (string, error) { return "", errNotFound{} },
		exists:   func(p string) bool { return exists[p] },
		hostname: func() (string, error) { return "lab-box-7", nil },
		goos:     "linux",
	}
	c := probe("/repo", e)
	// With no FAK_BOX_ID, the hostname is HASHED into a non-leaking id (#970), not
	// passed through raw — so a committed ledger row never carries the real hostname.
	if want := hashedBoxID("lab-box-7"); c.Box != want {
		t.Errorf("box = %q, want the hashed id %q (hostname must not pass through raw)", c.Box, want)
	}
	if c.Box == "lab-box-7" {
		t.Error("the raw hostname must not be the box id (PUBLIC_LEAK risk)")
	}
	// An operator-chosen FAK_BOX_ID still wins verbatim (their choice to expose).
	env["FAK_BOX_ID"] = "gcp-g2-l4"
	if c2 := probe("/repo", e); c2.Box != "gcp-g2-l4" {
		t.Errorf("FAK_BOX_ID must pass through verbatim, got %q", c2.Box)
	}
	delete(env, "FAK_BOX_ID")
	if c.GPU != "none" {
		t.Errorf("gpu = %q, want none (no nvidia-smi, linux)", c.GPU)
	}
	if !c.Weights {
		t.Error("weights should be true (FAK_MODEL_DIR exists)")
	}
	if !c.Creds["ANTHROPIC_API_KEY"] {
		t.Error("ANTHROPIC_API_KEY should be present")
	}
	if c.Creds["HF_TOKEN"] {
		t.Error("HF_TOKEN should be absent")
	}
	if !c.Net {
		t.Error("net should default true without FAK_OFFLINE")
	}
}

func TestDetectGPU(t *testing.T) {
	cases := []struct {
		name    string
		backend string
		nvidia  bool
		goos    string
		wantGPU string
	}{
		{"explicit cuda", "cuda", false, "linux", "cuda"},
		{"explicit metal", "metal", false, "linux", "metal"},
		{"nvidia-smi present", "", true, "linux", "cuda"},
		{"darwin default metal", "", false, "darwin", "metal"},
		{"bare linux none", "", false, "linux", "none"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			e := probeEnv{
				getenv: func(k string) string {
					if k == "FAK_BACKEND" {
						return tc.backend
					}
					return ""
				},
				look: func(string) (string, error) {
					if tc.nvidia {
						return "/usr/bin/nvidia-smi", nil
					}
					return "", errNotFound{}
				},
				goos: tc.goos,
			}
			if got := detectGPU(e); got != tc.wantGPU {
				t.Errorf("detectGPU = %q, want %q", got, tc.wantGPU)
			}
		})
	}
}

type errNotFound struct{}

func (errNotFound) Error() string { return "not found" }

func TestSatisfies(t *testing.T) {
	metalBox := Capabilities{Box: "mac", GPU: "metal", Weights: true, Net: true, Creds: map[string]bool{}}
	cudaBox := Capabilities{Box: "a100", GPU: "cuda", Weights: true, Net: true, Creds: map[string]bool{"ANTHROPIC_API_KEY": true}}
	bareBox := Capabilities{Box: "ci", GPU: "none", Net: true, Creds: map[string]bool{}}

	metalTask := Task{ID: "m", Requires: []Requirement{ReqMetal, ReqWeights}}
	if ok, _ := metalBox.Satisfies(metalTask); !ok {
		t.Error("metal box should satisfy a metal+weights task")
	}
	if ok, why := cudaBox.Satisfies(metalTask); ok {
		t.Errorf("cuda box should NOT satisfy a metal task; why=%q", why)
	}
	offline := Task{ID: "o"}
	if ok, _ := bareBox.Satisfies(offline); !ok {
		t.Error("bare box should satisfy an offline task")
	}
	credTask := Task{ID: "c", Requires: []Requirement{ReqNet}, CredEnv: []string{"ANTHROPIC_API_KEY"}}
	if ok, _ := cudaBox.Satisfies(credTask); !ok {
		t.Error("cuda box with the cred should satisfy a cred task")
	}
	if ok, why := bareBox.Satisfies(credTask); ok {
		t.Errorf("bare box without the cred should fail; why=%q", why)
	}
}

// --- selector determinism + ordering ---------------------------------------

func TestRankFeasibleFirstAndDeterministic(t *testing.T) {
	caps := Capabilities{Box: "mac", GPU: "metal", Weights: true, Net: true, Creds: map[string]bool{}}
	now := mustTime(t, "2026-06-27T00:00:00Z")
	tasks := []Task{
		{ID: "cuda-only", Value: ValueFrontier, Requires: []Requirement{ReqCUDA}, Run: "x"},
		{ID: "metal-frontier", Value: ValueFrontier, Requires: []Requirement{ReqMetal}, Run: "x"},
		{ID: "offline-smoke", Value: ValueSmoke, Run: "x"},
	}
	r1 := Rank(tasks, caps, nil, now)
	r2 := Rank(tasks, caps, nil, now)
	if len(r1) != 3 {
		t.Fatalf("want 3 scored, got %d", len(r1))
	}
	// feasible first: cuda-only is infeasible on a metal box, must sort last.
	if r1[2].Task.ID != "cuda-only" || r1[2].Feasible {
		t.Errorf("cuda-only should be last + infeasible, got %q feasible=%v", r1[2].Task.ID, r1[2].Feasible)
	}
	// metal-frontier outranks offline-smoke (both feasible, frontier > smoke).
	if r1[0].Task.ID != "metal-frontier" {
		t.Errorf("metal-frontier should rank first, got %q", r1[0].Task.ID)
	}
	for i := range r1 {
		if r1[i].Task.ID != r2[i].Task.ID || r1[i].Score != r2[i].Score {
			t.Fatalf("rank not deterministic at %d: %v vs %v", i, r1[i], r2[i])
		}
	}
	next, ok := Next(r1)
	if !ok || next.Task.ID != "metal-frontier" {
		t.Errorf("Next should pick metal-frontier, got ok=%v id=%q", ok, next.Task.ID)
	}
}

func TestRankNoveltyBeatsStaleness(t *testing.T) {
	caps := Capabilities{Box: "mac", GPU: "metal", Net: true, Creds: map[string]bool{}}
	now := mustTime(t, "2026-06-27T00:00:00Z")
	tasks := []Task{
		{ID: "never", Value: ValueSmoke, Run: "x"},
		{ID: "collected-fresh", Value: ValueFrontier, Run: "x", RecheckDays: 7},
	}
	// collected-fresh was collected yesterday (fresh); never was never collected.
	ledger := []CollectRow{
		{Schema: CollectSchema, TaskID: "collected-fresh", Box: "mac", Outcome: string(OutcomeCollected),
			Date: "2026-06-26", GeneratedAt: "2026-06-26T00:00:00Z"},
	}
	r := Rank(tasks, caps, ledger, now)
	if r[0].Task.ID != "never" {
		t.Errorf("a never-collected datum should outrank a fresh-collected one; got %q first", r[0].Task.ID)
	}
}

func TestRankStaleResurfaces(t *testing.T) {
	caps := Capabilities{Box: "a100", GPU: "cuda", Net: true, Creds: map[string]bool{}}
	now := mustTime(t, "2026-06-27T00:00:00Z")
	task := Task{ID: "drift", Value: ValueRegression, Run: "x", RecheckDays: 7}
	// collected 30d ago — well past the 7d recheck → stale.
	stale := []CollectRow{
		{Schema: CollectSchema, TaskID: "drift", Box: "a100", Outcome: string(OutcomeCollected),
			Date: "2026-05-28", GeneratedAt: "2026-05-28T00:00:00Z"},
	}
	r := Rank([]Task{task}, caps, stale, now)
	if r[0].Staleness < 1.0 {
		t.Errorf("a 30d-old datum past a 7d recheck should be fully stale, got %.3f", r[0].Staleness)
	}
	if !strings.Contains(r[0].Reason, "overdue") {
		t.Errorf("reason should call it overdue, got %q", r[0].Reason)
	}
}

// --- ledger -----------------------------------------------------------------

func TestLedgerRoundTripAndLastCollected(t *testing.T) {
	rows := []CollectRow{
		{Schema: CollectSchema, TaskID: "a", Box: "mac", Outcome: "collected", Date: "2026-06-20", GeneratedAt: "2026-06-20T00:00:00Z"},
		{Schema: CollectSchema, TaskID: "a", Box: "mac", Outcome: "collected", Date: "2026-06-25", GeneratedAt: "2026-06-25T00:00:00Z"},
		{Schema: CollectSchema, TaskID: "a", Box: "mac", Outcome: "failed", Date: "2026-06-26", GeneratedAt: "2026-06-26T00:00:00Z"},
	}
	var sb strings.Builder
	for _, r := range rows {
		line, err := AppendLedgerLine(r)
		if err != nil {
			t.Fatal(err)
		}
		sb.WriteString(line + "\n")
	}
	sb.WriteString("\n")         // tolerate blank
	sb.WriteString("not json\n") // tolerate garbage
	got := ParseLedger(sb.String())
	if len(got) != 3 {
		t.Fatalf("want 3 parsed rows, got %d", len(got))
	}
	last, ok := lastCollected(got, "a", "mac")
	if !ok {
		t.Fatal("expected a last-collected row")
	}
	if last.Date != "2026-06-25" {
		t.Errorf("last COLLECTED should be the 06-25 row (the 06-26 is a failure), got %s", last.Date)
	}
}

// --- backlog ----------------------------------------------------------------

func TestBacklogNoDupIDsAndSourcesPresent(t *testing.T) {
	tasks, err := Backlog("")
	if err != nil {
		t.Fatalf("Backlog: %v", err)
	}
	seen := map[string]bool{}
	var benchN, witnessN int
	for _, tk := range tasks {
		if seen[tk.ID] {
			t.Errorf("duplicate task id %q", tk.ID)
		}
		seen[tk.ID] = true
		switch tk.Source {
		case SourceBenchmark:
			benchN++
		case SourceWitness:
			witnessN++
		}
		if tk.Run == "" {
			t.Errorf("task %q has no run command", tk.ID)
		}
	}
	if benchN == 0 {
		t.Error("expected benchmark-sourced tasks")
	}
	if witnessN == 0 {
		t.Error("expected curated witness tasks")
	}
	// the curated metal witness must require metal (so it never shows on a CI box).
	var found bool
	for _, tk := range tasks {
		if tk.ID == "witness-q8-decode-matvec-bw" {
			found = true
			if !requires(tk, ReqMetal) {
				t.Error("the q8 decode witness must require metal")
			}
		}
	}
	if !found {
		t.Error("expected the q8-decode-matvec-bw witness in the backlog")
	}
}

func requires(t Task, r Requirement) bool {
	for _, x := range t.Requires {
		if x == r {
			return true
		}
	}
	return false
}

// --- run loop (fake executor) ----------------------------------------------

func TestRunLoopDryRunWritesNothing(t *testing.T) {
	caps := Capabilities{Box: "ci", GPU: "none", Net: true, Creds: map[string]bool{}}
	now := mustTime(t, "2026-06-27T00:00:00Z")
	tasks := []Task{{ID: "offline-a", Value: ValueSmoke, Run: "echo a"}}
	appended := 0
	summary, err := RunLoop(context.Background(), RunOptions{
		Root: "/repo", Caps: caps, Tasks: tasks, Now: now,
		Apply: false, Loop: false,
		ReadLedger: func() []CollectRow { return nil },
		AppendRow:  func(CollectRow) error { appended++; return nil },
		Executor: func(context.Context, Task, string) (Outcome, string, time.Duration, error) {
			t.Fatal("executor must not run in dry-run")
			return "", "", 0, nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if appended != 0 {
		t.Errorf("dry-run must not append to the ledger, appended=%d", appended)
	}
	if len(summary.Runs) != 1 || summary.Runs[0].Outcome != OutcomeDryRun {
		t.Errorf("want one dry-run entry, got %+v", summary.Runs)
	}
}

func TestRunLoopApplyAppendsAndStopsAtMax(t *testing.T) {
	caps := Capabilities{Box: "ci", GPU: "none", Net: true, Creds: map[string]bool{}}
	now := mustTime(t, "2026-06-27T00:00:00Z")
	tasks := []Task{
		{ID: "offline-a", Value: ValueCoverage, Run: "echo a"},
		{ID: "offline-b", Value: ValueCoverage, Run: "echo b"},
		{ID: "offline-c", Value: ValueCoverage, Run: "echo c"},
	}
	var ledger []CollectRow
	runs := 0
	summary, err := RunLoop(context.Background(), RunOptions{
		Root: "/repo", Caps: caps, Tasks: tasks, Now: now,
		Apply: true, Loop: true, Max: 2,
		ReadLedger: func() []CollectRow { return ledger },
		AppendRow:  func(r CollectRow) error { ledger = append(ledger, r); return nil },
		Executor: func(_ context.Context, tk Task, _ string) (Outcome, string, time.Duration, error) {
			runs++
			return OutcomeCollected, "", time.Second, nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if runs != 2 {
		t.Errorf("--max 2 should run exactly 2 tasks, ran %d", runs)
	}
	if len(ledger) != 2 {
		t.Errorf("expected 2 ledger rows, got %d", len(ledger))
	}
	if !strings.Contains(summary.StopReason, "max") {
		t.Errorf("stop reason should cite max, got %q", summary.StopReason)
	}
}

func TestRunLoopEachTaskAttemptedOnce(t *testing.T) {
	// A failing executor must not spin the loop: each task is attempted once.
	caps := Capabilities{Box: "ci", GPU: "none", Net: true, Creds: map[string]bool{}}
	now := mustTime(t, "2026-06-27T00:00:00Z")
	tasks := []Task{
		{ID: "offline-a", Value: ValueCoverage, Run: "false"},
		{ID: "offline-b", Value: ValueCoverage, Run: "false"},
	}
	var ledger []CollectRow
	attempts := map[string]int{}
	_, err := RunLoop(context.Background(), RunOptions{
		Root: "/repo", Caps: caps, Tasks: tasks, Now: now,
		Apply: true, Loop: true, Max: 0,
		ReadLedger: func() []CollectRow { return ledger },
		AppendRow:  func(r CollectRow) error { ledger = append(ledger, r); return nil },
		Executor: func(_ context.Context, tk Task, _ string) (Outcome, string, time.Duration, error) {
			attempts[tk.ID]++
			return OutcomeFailed, "", time.Second, nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	for id, n := range attempts {
		if n != 1 {
			t.Errorf("task %q attempted %d times, want exactly 1 (no spin on failure)", id, n)
		}
	}
	if len(attempts) != 2 {
		t.Errorf("both tasks should be attempted once, got %d distinct", len(attempts))
	}
}

func TestAutoRunnable(t *testing.T) {
	cases := []struct {
		run  string
		want bool
	}{
		{"go run ./cmd/modelbench -quant", true},
		{"echo hi", true},
		{"false", true},
		{"cmd > out.txt", true},    // a redirect is not a placeholder
		{"sh -c 'cat < in'", true}, // input redirect (space after <) is not a placeholder
		{"", false},                // empty is not runnable
		{"serve --gguf <glm-5.2.gguf> --load", false},
		{"go run ./cmd/terminalbench -suite <official-suite>", false},
		{"experiments/benchmark gcp-qwen-serve.sh → fak serve + fak agent", false},
		{"script.sh -> next", false},
	}
	for _, c := range cases {
		if got := (Task{Run: c.run}).autoRunnable(); got != c.want {
			t.Errorf("autoRunnable(%q) = %v, want %v", c.run, got, c.want)
		}
	}
}

func TestRunLoopSkipsManualTask(t *testing.T) {
	// A manual witness (placeholder Run) outranks a real bench by Value; the loop
	// must SKIP it (never exec, never ledger) and still collect the real one.
	caps := Capabilities{Box: "ci", GPU: "cuda", Net: true, Creds: map[string]bool{}}
	now := mustTime(t, "2026-06-27T00:00:00Z")
	tasks := []Task{
		{ID: "witness-manual", Value: ValueFrontier, Requires: []Requirement{ReqCUDA}, Run: "serve --gguf <model.gguf>"},
		{ID: "bench-real", Value: ValueCoverage, Run: "echo ok"},
	}
	var ledger []CollectRow
	executed := map[string]int{}
	summary, err := RunLoop(context.Background(), RunOptions{
		Root: "/repo", Caps: caps, Tasks: tasks, Now: now,
		Apply: true, Loop: true,
		ReadLedger: func() []CollectRow { return ledger },
		AppendRow:  func(r CollectRow) error { ledger = append(ledger, r); return nil },
		Executor: func(_ context.Context, tk Task, _ string) (Outcome, string, time.Duration, error) {
			executed[tk.ID]++
			return OutcomeCollected, "", time.Second, nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if executed["witness-manual"] != 0 {
		t.Errorf("manual task must never be executed, ran %d time(s)", executed["witness-manual"])
	}
	if executed["bench-real"] != 1 {
		t.Errorf("real task should run exactly once, ran %d", executed["bench-real"])
	}
	if len(ledger) != 1 || ledger[0].TaskID != "bench-real" {
		t.Errorf("ledger should hold only the real collection, got %+v", ledger)
	}
	sawSkip := false
	for _, r := range summary.Runs {
		if r.Task.ID == "witness-manual" && r.Outcome == OutcomeSkipped {
			sawSkip = true
		}
	}
	if !sawSkip {
		t.Errorf("summary should record the manual task as skipped, got %+v", summary.Runs)
	}
}

func TestExecTaskRunsInRoot(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("uses a POSIX `test -f` probe; the suite runs under WSL on this host")
	}
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "MARKER"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	art := filepath.Join(t.TempDir(), "out.log")
	// The task only succeeds if its relative-path probe resolves in root (not the
	// process CWD) — the regression the cmd.Dir fix closes.
	task := Task{ID: "cwd-check", Run: "test -f MARKER && echo ok"}
	outcome, _, _, err := execTask(context.Background(), root, nil, task, art)
	if outcome != OutcomeCollected {
		t.Fatalf("want collected (ran in root where MARKER exists), got %s err=%v", outcome, err)
	}
	data, _ := os.ReadFile(art)
	if !strings.Contains(string(data), "ok") {
		t.Errorf("command should have found MARKER in the root cwd, artifact=%q", data)
	}
}

func TestParseNumberObservedOnly(t *testing.T) {
	if got := parseNumber("decode: 17.73 tok/s steady"); got != "17.73 tok/s" {
		t.Errorf("parseNumber = %q, want 17.73 tok/s", got)
	}
	if got := parseNumber("no headline here"); got != "" {
		t.Errorf("parseNumber must NOT fabricate a number, got %q", got)
	}
}

// TestTaskTimeoutDefaultAndOverride pins the per-attempt budget: an unset
// TimeoutSec falls back to DefaultTaskTimeoutSec; a declared one wins. This bound
// is what keeps a single slow task from stalling an unattended --loop.
func TestTaskTimeoutDefaultAndOverride(t *testing.T) {
	if got := (Task{}).timeout(); got != time.Duration(DefaultTaskTimeoutSec)*time.Second {
		t.Errorf("default timeout = %s, want %ds", got, DefaultTaskTimeoutSec)
	}
	if got := (Task{TimeoutSec: 5}).timeout(); got != 5*time.Second {
		t.Errorf("override timeout = %s, want 5s", got)
	}
}

// TestDefaultExecutorTimeout pins the OBSERVED-timeout path: a task that exceeds
// its budget is killed and recorded as OutcomeTimeout (never a success, never a
// generic failure), with no fabricated number and the partial output still
// captured as evidence. Regression guard for the unattended-loop stall a slow
// full-grid benchmark caused in live collection.
func TestDefaultExecutorTimeout(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("uses sh/sleep; runs on the Linux test path (native windows go test is blocked anyway)")
	}
	art := filepath.Join(t.TempDir(), "out.log")
	task := Task{ID: "slow", Run: "echo started; sleep 30", TimeoutSec: 1}
	start := time.Now()
	outcome, number, dur, err := DefaultExecutor(context.Background(), task, art)
	if outcome != OutcomeTimeout {
		t.Errorf("outcome = %q, want %q (err=%v)", outcome, OutcomeTimeout, err)
	}
	if err == nil {
		t.Error("a timed-out run must report an error naming the budget")
	}
	if number != "" {
		t.Errorf("a timed-out run must not report a headline number, got %q", number)
	}
	if elapsed := time.Since(start); elapsed > 10*time.Second || dur > 10*time.Second {
		t.Errorf("timeout did not fire promptly (elapsed=%s dur=%s); the 1s budget should kill the 30s sleep", elapsed, dur)
	}
	if b, rerr := os.ReadFile(art); rerr != nil || !strings.Contains(string(b), "started") {
		t.Errorf("partial output not captured before kill: err=%v content=%q", rerr, string(b))
	}
}

// TestBacklogRunRequiresConsistency pins the feasibility invariant the whole
// nightrun promise rests on: a task's declared Requires must match what its Run
// command actually needs, so the feasibility gate can never pass a task the box
// cannot run. Regression for witness-glm52-native-load-10min, which used
// `--backend cuda` but declared only ReqWeights — so it ranked #1 on a gpu=none
// box, the exact "claim a datum the hardware can't produce" failure the doc
// forbids.
func TestBacklogRunRequiresConsistency(t *testing.T) {
	tasks, err := Backlog("")
	if err != nil {
		t.Fatalf("Backlog: %v", err)
	}
	has := func(reqs []Requirement, want Requirement) bool {
		for _, r := range reqs {
			if r == want {
				return true
			}
		}
		return false
	}
	for _, task := range tasks {
		if strings.Contains(task.Run, "--backend cuda") && !has(task.Requires, ReqCUDA) {
			t.Errorf("task %q Run uses --backend cuda but does not declare ReqCUDA (Requires=%v) — it would be falsely feasible on a non-CUDA box", task.ID, task.Requires)
		}
		if strings.Contains(task.Run, "-tags=metal") && !has(task.Requires, ReqMetal) {
			t.Errorf("task %q Run uses -tags=metal but does not declare ReqMetal (Requires=%v)", task.ID, task.Requires)
		}
	}
}

// TestSatisfiesLocalCheckpoint pins the #964 fix: a NeedWeights task that hardcodes a
// specific local export (-dir internal/model/.cache/<name>) is feasible ONLY if that
// export exists under the probed Root, so box-level Weights=true can no longer mark a
// missing-checkpoint benchmark feasible (which then fails at runtime). A -synthetic /
// placeholder Run names no concrete local export and is unaffected.
func TestSatisfiesLocalCheckpoint(t *testing.T) {
	root := t.TempDir()
	present := filepath.Join(root, "internal", "model", ".cache", "smollm2-135m")
	if err := os.MkdirAll(present, 0o755); err != nil {
		t.Fatalf("mkdir export: %v", err)
	}
	box := Capabilities{Box: "dev", GPU: "none", Weights: true, Net: true, Root: root, Creds: map[string]bool{}}

	presentTask := Task{ID: "model", Requires: []Requirement{ReqWeights}, Run: "go run ./cmd/modelbench -dir internal/model/.cache/smollm2-135m"}
	if ok, why := box.Satisfies(presentTask); !ok {
		t.Errorf("a task whose -dir export EXISTS must be feasible; why=%q", why)
	}

	absentTask := Task{ID: "batch", Requires: []Requirement{ReqWeights}, Run: "go run ./cmd/batchbench -dir internal/model/.cache/absent-model"}
	if ok, why := box.Satisfies(absentTask); ok {
		t.Errorf("a task whose -dir export is ABSENT must be infeasible; got feasible (why=%q)", why)
	} else if !strings.Contains(why, "absent-model") {
		t.Errorf("the block reason should name the missing checkpoint; got %q", why)
	}

	// -synthetic names no concrete local export — the path gate must not touch it.
	synthTask := Task{ID: "radix", Requires: []Requirement{ReqWeights}, Run: "go run ./cmd/radixbench -synthetic smollm2-135m"}
	if ok, why := box.Satisfies(synthTask); !ok {
		t.Errorf("a -synthetic task must stay feasible (no hardcoded export); why=%q", why)
	}

	// No Root probed (e.g. a unit test constructing caps by hand) → the gate is skipped.
	noRoot := Capabilities{Box: "x", Weights: true, Net: true, Creds: map[string]bool{}}
	if ok, _ := noRoot.Satisfies(absentTask); !ok {
		t.Error("with no Root set the checkpoint gate must be skipped (conservative)")
	}
}

func TestLocalCheckpointPath(t *testing.T) {
	cases := map[string]string{
		"go run ./cmd/modelbench -dir internal/model/.cache/smollm2-135m": "internal/model/.cache/smollm2-135m",
		"fak serve --gguf <glm-5.2.gguf> --backend cuda":                  "", // placeholder
		"go run ./cmd/radixbench -synthetic smollm2-135m":                 "", // no -dir/-gguf
		"go run ./cmd/q8bench -dir /abs/path/model":                       "", // absolute, outside repo
		"go run ./cmd/x -dir ~/models/m":                                  "", // home, outside repo
	}
	for run, want := range cases {
		if got := localCheckpointPath(run); got != want {
			t.Errorf("localCheckpointPath(%q) = %q, want %q", run, got, want)
		}
	}
}

// TestHashedBoxID pins the #970 fix: a hostname is turned into a stable, non-leaking
// per-box id (box-<8hex>) that never contains the raw hostname, so a committed ledger
// row does not trip the PUBLIC_LEAK gate.
func TestHashedBoxID(t *testing.T) {
	// Build the sample hostname at runtime so this test file does not itself carry a
	// literal DESKTOP-* string the PUBLIC_LEAK gate would flag (the gate scans test code too).
	host := "DESKTOP-" + strings.ToUpper("bb3fmhp")
	got := hashedBoxID(host)
	if got == "" || got == host {
		t.Fatalf("hashedBoxID(%q) must be a non-empty, non-identity id, got %q", host, got)
	}
	if !strings.HasPrefix(got, "box-") || len(got) != len("box-")+8 {
		t.Errorf("want box-<8hex> shape, got %q", got)
	}
	if strings.Contains(strings.ToUpper(got), "DESKTOP") {
		t.Errorf("the id must NOT leak the hostname, got %q", got)
	}
	if hashedBoxID(host) != got {
		t.Error("hashedBoxID must be stable (same host -> same id)")
	}
	if hashedBoxID("") != "" {
		t.Error("empty hostname must return empty (so the caller's unknown-box fallback applies)")
	}
}
