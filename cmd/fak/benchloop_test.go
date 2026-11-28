package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"
)

func TestStopConditions(t *testing.T) {
	limit := func(n int) *int { return &n }
	deadline := func(d time.Duration) *time.Duration { return &d }

	t.Run("closed ordered set", func(t *testing.T) {
		want := []string{"lifecycle", "dag_lifecycle", "request_count", "session_count", "duration"}
		var got []string
		for _, cond := range benchStopConditions {
			got = append(got, cond.Name)
		}
		if !slices.Equal(got, want) {
			t.Fatalf("stop conditions = %v, want %v", got, want)
		}
		all := newBenchStopChecker(benchStopLimits{
			MaxRequests: limit(1), MaxSessions: limit(1), Duration: deadline(time.Second),
		})
		if !slices.Equal(all.Active(), want) {
			t.Fatalf("active with every limit = %v, want %v", all.Active(), want)
		}
	})

	t.Run("unset limits are not checked", func(t *testing.T) {
		bare := newBenchStopChecker(benchStopLimits{})
		if got, want := bare.Active(), []string{"lifecycle", "dag_lifecycle"}; !slices.Equal(got, want) {
			t.Fatalf("active with no limits = %v, want %v", got, want)
		}
		huge := benchStopState{RequestsSent: 1 << 30, SessionsSent: 1 << 30, Elapsed: 1000 * time.Hour}
		if !bare.CanSendAnyTurn(huge) || !bare.CanStartNewSession(huge) || !bare.CanSendDAGChildTurn(huge) {
			t.Fatal("a checker with no limits refused a turn")
		}
		onlyDuration := newBenchStopChecker(benchStopLimits{Duration: deadline(time.Minute)})
		if got, want := onlyDuration.Active(), []string{"lifecycle", "dag_lifecycle", "duration"}; !slices.Equal(got, want) {
			t.Fatalf("active with a duration = %v, want %v", got, want)
		}
	})

	t.Run("first refusing condition wins", func(t *testing.T) {
		c := newBenchStopChecker(benchStopLimits{
			MaxRequests: limit(5), MaxSessions: limit(1), Duration: deadline(time.Second),
		})
		everything := benchStopState{
			Cancelled: true, RequestsSent: 5, SessionsSent: 1, Elapsed: time.Minute,
		}
		if got := c.FirstStop(everything, false); got != "lifecycle" {
			t.Fatalf("loop first stop = %q, want lifecycle", got)
		}
		if got := c.FirstStop(everything, true); got != "dag_lifecycle" {
			t.Fatalf("child first stop = %q, want dag_lifecycle", got)
		}
		everything.Cancelled = false
		if got := c.FirstStop(everything, false); got != "request_count" {
			t.Fatalf("loop first stop without cancel = %q, want request_count", got)
		}
		if got := c.FirstStop(benchStopState{}, false); got != "" {
			t.Fatalf("fresh loop first stop = %q, want none", got)
		}
	})

	// The DAG-child policy matrix: one stopping fact per row, and whether it
	// stops the send loop and child dispatch.
	t.Run("dag child policy", func(t *testing.T) {
		c := newBenchStopChecker(benchStopLimits{
			MaxRequests: limit(10), MaxSessions: limit(2), Duration: deadline(time.Minute),
		})
		for _, row := range []struct {
			name      string
			state     benchStopState
			loopSends bool
			childSend bool
		}{
			{"nothing reached", benchStopState{}, true, true},
			{"cancelled", benchStopState{Cancelled: true}, false, false},
			{"sending complete", benchStopState{SendingComplete: true}, false, true},
			{"request count reached", benchStopState{RequestsSent: 10}, false, false},
			{"duration elapsed", benchStopState{Elapsed: time.Minute}, false, false},
			{"session count reached", benchStopState{SessionsSent: 2, RootRequestsSent: 6, TotalSessionTurns: 6}, false, true},
		} {
			if got := c.CanSendAnyTurn(row.state); got != row.loopSends {
				t.Errorf("%s: CanSendAnyTurn = %v, want %v", row.name, got, row.loopSends)
			}
			if got := c.CanSendDAGChildTurn(row.state); got != row.childSend {
				t.Errorf("%s: CanSendDAGChildTurn = %v, want %v", row.name, got, row.childSend)
			}
		}
	})

	t.Run("session target stops new sessions not open ones", func(t *testing.T) {
		c := newBenchStopChecker(benchStopLimits{MaxSessions: limit(2)})
		open := benchStopState{SessionsSent: 2, RootRequestsSent: 3, TotalSessionTurns: 6}
		if !c.CanSendAnyTurn(open) {
			t.Fatal("continuation turns of open sessions were refused")
		}
		if c.CanStartNewSession(open) {
			t.Fatal("a new session started past the session target")
		}
		// Child wires inflate RequestsSent; only root wires finish a session.
		open.RequestsSent = 40
		if !c.CanSendAnyTurn(open) {
			t.Fatal("child requests were counted against the root session plan")
		}
		below := benchStopState{SessionsSent: 1, RootRequestsSent: 3, TotalSessionTurns: 3}
		if !c.CanStartNewSession(below) {
			t.Fatal("a new session was refused below the session target")
		}
	})

	t.Run("new session implies any turn", func(t *testing.T) {
		c := newBenchStopChecker(benchStopLimits{MaxRequests: limit(1), MaxSessions: limit(5)})
		if c.CanStartNewSession(benchStopState{RequestsSent: 1}) {
			t.Fatal("a new session started with the request cap reached")
		}
		if c.CanStartNewSession(benchStopState{Cancelled: true}) {
			t.Fatal("a new session started after cancellation")
		}
	})

	t.Run("zero limits stop immediately", func(t *testing.T) {
		c := newBenchStopChecker(benchStopLimits{MaxRequests: limit(0)})
		if c.CanSendAnyTurn(benchStopState{}) || c.CanSendDAGChildTurn(benchStopState{}) {
			t.Fatal("a zero request cap allowed a turn")
		}
	})
}

func TestBenchLoopStatusAndNext(t *testing.T) {
	root := t.TempDir()
	writeBenchLoopFile(t, filepath.Join(root, "BENCHMARK-AUTHORITY.md"), "**Last updated:** 2026-06-29\n")
	benchDir := filepath.Join(root, "experiments", "benchmark")
	if err := os.MkdirAll(benchDir, 0o755); err != nil {
		t.Fatal(err)
	}
	writeBenchLoopJSON(t, filepath.Join(benchDir, "catalog.json"), map[string]any{"runs": []map[string]any{{
		"run_id": "run-a", "machine_id": "box-a", "model": "smollm2",
		"precision": "q8", "timestamp": "2026-06-29T00:00:00Z",
	}}})
	ledgerDir := filepath.Join(root, "docs", "nightrun")
	if err := os.MkdirAll(ledgerDir, 0o755); err != nil {
		t.Fatal(err)
	}
	writeBenchLoopFile(t, filepath.Join(ledgerDir, "collected.jsonl"),
		`{"schema":"fak-nightrun-collect/1","date":"2026-06-30","box":"ci-box","task_id":"bench-ablate","value":"smoke","command":"fak ablate --sweep vdso","outcome":"collected","generated_at":"2026-06-30T00:00:00Z"}`+"\n")

	t.Setenv("FAK_BOX_ID", "ci-box")
	t.Setenv("FAK_OFFLINE", "1")

	var out, errb bytes.Buffer
	if code := runBenchLoop(&out, &errb, []string{"status", "--workspace", root, "--now", "2026-06-30T00:00:00Z"}); code != 0 {
		t.Fatalf("status code=%d stderr=%s", code, errb.String())
	}
	for _, want := range []string{"benchmark super-loop status", "registry:", "catalog:", "next action:"} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("status missing %q:\n%s", want, out.String())
		}
	}

	out.Reset()
	errb.Reset()
	if code := runBenchLoop(&out, &errb, []string{"next", "--workspace", root, "--now", "2026-06-30T00:00:00Z", "--json"}); code != 0 {
		t.Fatalf("next code=%d stderr=%s", code, errb.String())
	}
	var action struct {
		Kind    string `json:"kind"`
		Command string `json:"command"`
	}
	if err := json.Unmarshal(out.Bytes(), &action); err != nil {
		t.Fatalf("next json: %v\n%s", err, out.String())
	}
	if action.Kind == "" || action.Command == "" {
		t.Fatalf("next action incomplete: %+v", action)
	}
}

func TestBenchLoopWalk(t *testing.T) {
	var out, errb bytes.Buffer
	if code := runBenchLoop(&out, &errb, []string{"walk"}); code != 0 {
		t.Fatalf("walk code=%d stderr=%s", code, errb.String())
	}
	for _, want := range []string{"benchmark super-loop walk", "fak bench-loop status", "fak bench-loop run --apply --loop"} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("walk missing %q:\n%s", want, out.String())
		}
	}
}

func TestWorkspaceOrRepoRoot(t *testing.T) {
	if got, want := workspaceOrRepoRoot(""), repoRoot(); got != want {
		t.Fatalf("empty workspace: got %q, want repo root %q", got, want)
	}
	abs := t.TempDir()
	if got, want := filepath.Clean(workspaceOrRepoRoot(abs)), filepath.Clean(abs); got != want {
		t.Fatalf("absolute workspace: got %q, want %q", got, want)
	}
	rel := filepath.Join("some", "rel")
	got := workspaceOrRepoRoot(rel)
	if !filepath.IsAbs(got) {
		t.Fatalf("relative workspace not absolutized: %q", got)
	}
	if !strings.HasSuffix(got, rel) {
		t.Fatalf("absolutized workspace %q lost the relative tail %q", got, rel)
	}
}

func TestParsePinnedTime(t *testing.T) {
	for _, tc := range []struct {
		in   string
		want string
	}{
		{"2026-06-30T05:00:00+02:00", "2026-06-30T03:00:00Z"}, // RFC3339, normalized to UTC
		{"20260630T030000Z", "2026-06-30T03:00:00Z"},          // compact
		{"2026-06-30", "2026-06-30T00:00:00Z"},                // date-only
	} {
		got, err := parsePinnedTimeFlag("--now", tc.in)
		if err != nil {
			t.Fatalf("parsePinnedTimeFlag(%q): %v", tc.in, err)
		}
		if got.Format("2006-01-02T15:04:05Z07:00") != tc.want {
			t.Fatalf("parsePinnedTimeFlag(%q) = %s, want %s", tc.in, got, tc.want)
		}
	}
	if _, err := parsePinnedTimeFlag("--now", "yesterday"); err == nil {
		t.Fatal("bad stamp: want error, got nil")
	} else if msg := err.Error(); !strings.Contains(msg, "--now") || !strings.Contains(msg, "YYYY-MM-DD") {
		t.Fatalf("error message lost flag name or format hint: %s", msg)
	}
}

func writeBenchLoopJSON(t *testing.T, path string, v any) {
	t.Helper()
	raw, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	writeBenchLoopFile(t, path, string(raw)+"\n")
}

func writeBenchLoopFile(t *testing.T, path, body string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestProgressStall(t *testing.T) {
	t0 := time.Unix(1000, 0)
	w := newPhaseProgressWatchdog("prefill", t0, 5*time.Second)
	if w.Update("prefill", 0, t0.Add(3*time.Second)) {
		t.Fatal("should not stall before timeout")
	}
	if !w.Update("prefill", 0, t0.Add(6*time.Second)) {
		t.Fatal("should stall after timeout without progress")
	}
	if w.Update("decode", 0, t0.Add(6*time.Second)) {
		t.Fatal("phase transition must reset stall timer")
	}
}

func TestGrader(t *testing.T) {
	cot, ans := splitThinkingAndAnswer("<think>expected is 99</think>\n 42 ")
	if cot != "expected is 99" || ans != "42" {
		t.Fatalf("split = (%q, %q)", cot, ans)
	}
	if !gradeExactMatch("<think>99</think>42", "42") || gradeExactMatch("<think>42</think>99", "42") {
		t.Fatal("grader must evaluate final answer, not CoT")
	}
}
