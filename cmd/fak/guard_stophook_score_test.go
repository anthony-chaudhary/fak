package main

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/trajctl"
)

// guard_stophook_score_test.go — issue #2539: the hook integration witnesses.
// The pure folds (Sample, CompactionBoundary, the scorer panic shield) are
// covered in internal/trajctl/turnend_test.go; these tests drive the REAL hook
// entry points (runGuardStopHook, runGuardPreCompact) and the bounded fail-open
// wrapper the hooks run them under.

func seedTrajctlObjective(t *testing.T, ledger, id string, planPhases ...string) {
	t.Helper()
	obj := trajctl.Objective{ID: id, Statement: "test objective for " + id, Status: trajctl.StatusActive}
	for _, p := range planPhases {
		obj.Plan = append(obj.Plan, trajctl.PlanPhase{ID: p})
	}
	if err := trajctl.Append(ledger, trajctl.ObjectiveRecord(obj)); err != nil {
		t.Fatalf("seed objective: %v", err)
	}
}

func trajctlScores(t *testing.T, ledger string) []trajctl.ScoreRow {
	t.Helper()
	return trajctl.Fold(trajctl.ReadLedgerFile(ledger)).Scores
}

// TestRunGuardStopHookScoresTurnEnd is the Stop-hook integration witness: a session
// with a declared (planned) objective gains at least one ScoreRow per turn end,
// stamped with the session id from the hook's stdin payload — and the sampling is
// independent of the deny-all mode (here: off), never changing the exit code.
func TestRunGuardStopHookScoresTurnEnd(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "trajctl.jsonl")
	seedTrajctlObjective(t, ledger, "obj-2539", "phase-1", "phase-2")
	t.Setenv(guardTrajctlEnvLedger, ledger)
	t.Setenv(guardTrajctlEnvMode, "")

	var stderr strings.Builder
	for turn := 1; turn <= 2; turn++ {
		stdin := strings.NewReader(`{"session_id":"sess-2539","stop_hook_active":false}`)
		if code := runGuardStopHook(&stderr, stdin, []string{"--mode", "off"}); code != 0 {
			t.Fatalf("turn %d: exit = %d, want 0 (scoring must never change the hook's exit)", turn, code)
		}
		var commitRows int
		for _, row := range trajctlScores(t, ledger) {
			if row.Method != trajctl.CommitScorerMethod {
				continue
			}
			commitRows++
			if row.SessionID != "sess-2539" {
				t.Fatalf("row not stamped with the hook session id: %+v", row)
			}
			if row.Witness != trajctl.W3 {
				t.Fatalf("commit scorer row witness = %q, want W3", row.Witness)
			}
		}
		if commitRows < turn {
			t.Fatalf("after %d turn end(s): %d commit-progress rows, want >= %d (one point per turn)", turn, commitRows, turn)
		}
	}
	if !strings.Contains(stderr.String(), "turn-end scored") {
		t.Fatalf("stderr missing the advisory score line: %q", stderr.String())
	}
}

// TestRunGuardStopHookScoringNoopWithoutLedger: without the guard-wired ledger env the
// sampling is a total no-op — no rows, no files, no stderr noise.
func TestRunGuardStopHookScoringNoopWithoutLedger(t *testing.T) {
	t.Setenv(guardTrajctlEnvLedger, "")
	var stderr strings.Builder
	stdin := strings.NewReader(`{"session_id":"sess-none"}`)
	if code := runGuardStopHook(&stderr, stdin, []string{"--mode", "off"}); code != 0 {
		t.Fatalf("exit = %d, want 0", code)
	}
	if strings.Contains(stderr.String(), "trajctl") {
		t.Fatalf("expected silence without a wired ledger, got: %q", stderr.String())
	}
}

// TestRunGuardStopHookScoringModeOff: the escape hatch (mode=off) disables sampling even
// with a ledger wired.
func TestRunGuardStopHookScoringModeOff(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "trajctl.jsonl")
	seedTrajctlObjective(t, ledger, "obj-off", "p1")
	t.Setenv(guardTrajctlEnvLedger, ledger)
	t.Setenv(guardTrajctlEnvMode, "off")

	if code := runGuardStopHook(&strings.Builder{}, strings.NewReader(`{}`), []string{"--mode", "off"}); code != 0 {
		t.Fatalf("exit = %d, want 0", code)
	}
	if rows := trajctlScores(t, ledger); len(rows) != 0 {
		t.Fatalf("mode=off must not score, got %d rows", len(rows))
	}
}

// TestRunGuardPreCompactAppendsBoundaryRow is the PreCompact twin witness: an ALLOWED
// compaction (here: hook mode off, exit 0) appends one compaction-boundary row per open
// objective, stamped with the session id, so curve readers can see the context reset.
func TestRunGuardPreCompactAppendsBoundaryRow(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "trajctl.jsonl")
	seedTrajctlObjective(t, ledger, "obj-compact") // no plan needed: boundary rows are per open objective
	t.Setenv(guardTrajctlEnvLedger, ledger)
	t.Setenv(guardTrajctlEnvMode, "")

	var stderr strings.Builder
	stdin := strings.NewReader(`{"session_id":"sess-2539"}`)
	if code := runGuardPreCompact(nil, &stderr, stdin, []string{"--mode", "off"}); code != 0 {
		t.Fatalf("exit = %d, want 0", code)
	}
	rows := trajctlScores(t, ledger)
	if len(rows) != 1 {
		t.Fatalf("boundary rows = %d, want exactly 1: %+v", len(rows), rows)
	}
	row := rows[0]
	if row.Method != trajctl.CompactionBoundaryMethod || row.ObjectiveID != "obj-compact" || row.SessionID != "sess-2539" {
		t.Fatalf("unexpected boundary row: %+v", row)
	}
}

// TestRunGuardPreCompactBlockedCompactionMarksNoBoundary: a BLOCKED compaction (exit 2)
// resets no context, so no boundary row may be written.
func TestRunGuardPreCompactBlockedCompactionMarksNoBoundary(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "trajctl.jsonl")
	seedTrajctlObjective(t, ledger, "obj-blocked")
	t.Setenv(guardTrajctlEnvLedger, ledger)
	t.Setenv(guardTrajctlEnvMode, "")

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte("fak_harness_coherence_posture 1\n"))
	}))
	defer srv.Close()

	code := runGuardPreCompact(nil, &strings.Builder{}, strings.NewReader(`{"session_id":"s"}`), []string{
		"--mode", guardPreCompactModeEnforce,
		"--metrics-url", srv.URL + "/metrics",
	})
	if code != 2 {
		t.Fatalf("exit = %d, want 2 (blocked compaction)", code)
	}
	if rows := trajctlScores(t, ledger); len(rows) != 0 {
		t.Fatalf("blocked compaction must mark no boundary, got %d rows", len(rows))
	}
}

// TestGuardTrajctlSampleBoundedSwallowsPanic: an injected panic inside the sampling pass
// is swallowed fail-open — the wrapper returns normally and reports it, costing at most
// the pass's own rows, never the hook.
func TestGuardTrajctlSampleBoundedSwallowsPanic(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "trajctl.jsonl")
	var stderr strings.Builder
	guardTrajctlSampleBounded(&stderr, "turn-end", ledger, 1, func(trajctl.State, trajctl.EvidenceWindow) trajctl.TurnSample {
		panic("injected scorer panic")
	})
	if !strings.Contains(stderr.String(), "fail-open") || !strings.Contains(stderr.String(), "injected scorer panic") {
		t.Fatalf("stderr = %q, want a fail-open panic report", stderr.String())
	}
	if rows := trajctlScores(t, ledger); len(rows) != 0 {
		t.Fatalf("panicked pass wrote %d rows, want 0", len(rows))
	}
}

// TestGuardTrajctlSampleBoundedDeadline: a wedged sampling pass is abandoned at the
// wall-clock deadline so hook latency stays bounded.
func TestGuardTrajctlSampleBoundedDeadline(t *testing.T) {
	var stderr strings.Builder
	start := time.Now()
	guardTrajctlSampleBounded(&stderr, "turn-end", filepath.Join(t.TempDir(), "trajctl.jsonl"), 1,
		func(trajctl.State, trajctl.EvidenceWindow) trajctl.TurnSample {
			time.Sleep(5 * time.Second)
			return trajctl.TurnSample{}
		})
	if elapsed := time.Since(start); elapsed > 2*time.Second {
		t.Fatalf("bounded pass took %s, want <= deadline (+margin)", elapsed)
	}
	if !strings.Contains(stderr.String(), "timed out") {
		t.Fatalf("stderr = %q, want a timeout note", stderr.String())
	}
}

// detourFixtureEvents is a live tool stream that folds to exactly one CLOSED detour:
// a {internal/trajctl, go} baseline, a 3-error go-test burst, a sustained ops/net
// topic shift, then a return to the parent shape.
func detourFixtureEvents() []trajctl.ToolEvent {
	ev := func(tool, target string, isErr bool) trajctl.ToolEvent {
		return trajctl.ToolEvent{Tool: tool, Target: target, IsError: isErr}
	}
	return []trajctl.ToolEvent{
		ev("Read", "internal/trajctl/a.go", false),
		ev("Read", "internal/trajctl/b.go", false),
		ev("Bash", "go test ./...", false),
		ev("Read", "internal/trajctl/a.go", false),
		ev("Bash", "go build ./...", false),
		ev("Bash", "go test ./...", true),
		ev("Bash", "go test ./...", true),
		ev("Bash", "go test ./...", true),
		ev("Read", "ops/net/x.conf", false),
		ev("Read", "ops/net/x.conf", false),
		ev("Read", "ops/net/x.conf", false),
		ev("Read", "internal/trajctl/a.go", false),
		ev("Bash", "go test ./...", false),
	}
}

// writeDetourTranscript emits a minimal Claude Code JSONL transcript for events: one
// assistant tool_use record and one user tool_result record per call, in order — the
// shape ParseToolStream folds. A spaced target rides on `command`, a path-like target
// on `file_path`, matching streamTarget's key preference.
func writeDetourTranscript(t *testing.T, path string, events []trajctl.ToolEvent) {
	t.Helper()
	var b strings.Builder
	for i, e := range events {
		id := fmt.Sprintf("tu-%02d", i)
		key := "file_path"
		if strings.Contains(e.Target, " ") {
			key = "command"
		}
		fmt.Fprintf(&b, `{"type":"assistant","message":{"content":[{"type":"tool_use","id":%q,"name":%q,"input":{%q:%q}}]}}`+"\n",
			id, e.Tool, key, e.Target)
		fmt.Fprintf(&b, `{"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":%q,"is_error":%v,"content":"x"}]}}`+"\n",
			id, e.IsError)
	}
	if err := os.WriteFile(path, []byte(b.String()), 0o644); err != nil {
		t.Fatalf("write transcript: %v", err)
	}
}

// TestDetectDetoursFailOpenWiresLiveDetector is the #3669 hook-integration witness:
// the live turn-end pass folds the finished turn's transcript into a MET detour child
// under the session's root objective (root resumed ACTIVE), and a second identical
// pass double-opens nothing.
func TestDetectDetoursFailOpenWiresLiveDetector(t *testing.T) {
	dir := t.TempDir()
	ledger := filepath.Join(dir, "trajctl.jsonl")
	seedTrajctlObjective(t, ledger, "obj-3669") // a root objective (no parent)
	transcript := filepath.Join(dir, "session.jsonl")
	writeDetourTranscript(t, transcript, detourFixtureEvents())
	t.Setenv(guardTrajctlEnvLedger, ledger)
	t.Setenv(guardTrajctlEnvMode, "")

	var stderr strings.Builder
	detectDetoursFailOpen(&stderr, transcript, trajctl.Stamp{SessionID: "sess-3669"}, 4242)

	st := trajctl.Fold(trajctl.ReadLedgerFile(ledger))
	child, ok := st.Objectives["obj-3669-detour-1"]
	if !ok {
		t.Fatalf("no detour child opened; objectives = %v", st.ObjectiveIDs())
	}
	if child.Status != trajctl.StatusMet {
		t.Fatalf("detour child status = %q, want %q", child.Status, trajctl.StatusMet)
	}
	if got := st.Objectives["obj-3669"].Status; got != trajctl.StatusActive {
		t.Fatalf("root status = %q, want %q (resumed)", got, trajctl.StatusActive)
	}
	if !strings.Contains(stderr.String(), "detour detection appended") {
		t.Fatalf("stderr missing the advisory detour line: %q", stderr.String())
	}

	// Replay: a second identical pass opens no second child.
	detectDetoursFailOpen(&stderr, transcript, trajctl.Stamp{SessionID: "sess-3669"}, 5555)
	if _, dup := trajctl.Fold(trajctl.ReadLedgerFile(ledger)).Objectives["obj-3669-detour-2"]; dup {
		t.Fatalf("replay double-opened a second detour child")
	}
}

// TestDetectDetoursFailOpenNoopWithoutLedgerOrTranscript: the pass is a total no-op
// without a guard-wired ledger, and a no-op (no rows, no noise) when the stop event
// carried no transcript path.
func TestDetectDetoursFailOpenNoopWithoutLedgerOrTranscript(t *testing.T) {
	t.Setenv(guardTrajctlEnvLedger, "")
	var s1 strings.Builder
	detectDetoursFailOpen(&s1, "whatever.jsonl", trajctl.Stamp{}, 1)
	if strings.Contains(s1.String(), "trajctl") {
		t.Fatalf("expected silence without a wired ledger, got %q", s1.String())
	}

	ledger := filepath.Join(t.TempDir(), "trajctl.jsonl")
	seedTrajctlObjective(t, ledger, "obj-x")
	t.Setenv(guardTrajctlEnvLedger, ledger)
	var s2 strings.Builder
	detectDetoursFailOpen(&s2, "   ", trajctl.Stamp{}, 1)
	if _, opened := trajctl.Fold(trajctl.ReadLedgerFile(ledger)).Objectives["obj-x-detour-1"]; opened {
		t.Fatalf("no-transcript pass opened a detour child")
	}
	if strings.Contains(s2.String(), "detour") {
		t.Fatalf("expected silence with no transcript path, got %q", s2.String())
	}
}

// TestParseHookSessionID covers the advisory stdin parse: nil, non-JSON, and a real
// payload.
func TestParseHookSessionID(t *testing.T) {
	if got := parseHookSessionID(nil); got != "" {
		t.Fatalf("nil payload -> %q, want empty", got)
	}
	if got := parseHookSessionID([]byte("not json")); got != "" {
		t.Fatalf("bad payload -> %q, want empty", got)
	}
	if got := parseHookSessionID([]byte(`{"session_id":"abc","stop_hook_active":true}`)); got != "abc" {
		t.Fatalf("session id = %q, want abc", got)
	}
}

// TestInstallGuardStopHookInjectsTrajctlLedger: the installer wires the default ledger
// path into the hook children so a real guarded session scores at turn cadence without
// any operator setup.
func TestInstallGuardStopHookInjectsTrajctlLedger(t *testing.T) {
	dir := t.TempDir()
	_, env, install, err := installGuardStopHookAt(
		[]string{"claude", "-p", "hi"}, guardPreCompactModeEnforce, "http://127.0.0.1:4567",
		filepath.Join(dir, "fak.exe"), dir, "", 3, 7, 9, 6, guardPreCompactModeOff)
	if err != nil || !install.Applied {
		t.Fatalf("install: applied=%v err=%v", install.Applied, err)
	}
	want := guardTrajctlLedgerDefault()
	if want == "" {
		t.Skip("repo root not resolvable from test cwd")
	}
	for _, kv := range env {
		if kv[0] == guardTrajctlEnvLedger {
			if kv[1] != want {
				t.Fatalf("ledger env = %q, want %q", kv[1], want)
			}
			return
		}
	}
	t.Fatalf("env missing %s: %v", guardTrajctlEnvLedger, env)
}
