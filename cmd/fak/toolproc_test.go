package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/toolproc"
)

// TestToolprocHookJournalRoundTrip drives the seam-4 adapter end to end at
// the file level: pre → post → stop firings append journal lines that the
// same fold `fak toolproc ps` uses reads back as a clean table with the
// orphaned survivor flagged.
func TestToolprocHookJournalRoundTrip(t *testing.T) {
	journal := filepath.Join(t.TempDir(), "journal.jsonl")

	pre := `{"session_id":"s1","tool_name":"Bash","tool_use_id":"toolu_a","tool_input":{"command":"make test"}}`
	post := `{"session_id":"s1","tool_name":"Bash","tool_use_id":"toolu_a","tool_input":{"command":"make test"},"tool_response":{"is_error":false}}`
	preOrphan := `{"session_id":"s1","tool_name":"Bash","tool_use_id":"toolu_b","tool_input":{"command":"tail -f x"}}`
	stop := `{"session_id":"s1"}`

	steps := []struct {
		kind, payload string
		atMS          int64
	}{
		{"pre", pre, 1_000},
		{"post", post, 3_000},
		{"pre", preOrphan, 4_000},
		{"stop", stop, 9_000},
	}
	for _, s := range steps {
		if err := toolprocHookOnce(strings.NewReader(s.payload), s.kind, journal, toolproc.HookEnvelope{}, s.atMS); err != nil {
			t.Fatalf("hook %s: %v", s.kind, err)
		}
	}

	f, err := os.Open(journal)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	events, err := toolproc.ParseEvents(f)
	if err != nil {
		t.Fatalf("journal must be fold-clean: %v", err)
	}
	tab, err := toolproc.Fold(events, 10_000, toolproc.Config{})
	if err != nil {
		t.Fatal(err)
	}
	if tab.Counts.Done != 1 || tab.Counts.Running != 1 || tab.Counts.Orphaned != 1 {
		t.Fatalf("want 1 done + 1 running orphan, got %+v", tab.Counts)
	}
	if !tab.AttentionNeeded {
		t.Error("the orphaned survivor must need attention")
	}
}

// TestToolprocHookFailOpen: garbage stdin, an unknown kind, and a missing
// journal directory all report an error from the inner helper but the hook
// entry point still exits 0 — observation never wedges the harness.
func TestToolprocHookFailOpen(t *testing.T) {
	journal := filepath.Join(t.TempDir(), "j.jsonl")
	if err := toolprocHookOnce(strings.NewReader(`{nope`), "pre", journal, toolproc.HookEnvelope{}, 1_000); err == nil {
		t.Error("garbage payload must error internally")
	}
	var errOut strings.Builder
	if rc := runToolprocHook(strings.NewReader(`{nope`), &errOut, []string{"pre", "--journal", journal}); rc != 0 {
		t.Errorf("hook must exit 0 on failure (fail-open), got %d", rc)
	}
	if !strings.Contains(errOut.String(), "fail-open") {
		t.Errorf("failure must be reported to stderr, got %q", errOut.String())
	}
	if rc := runToolprocHook(strings.NewReader(""), &errOut, nil); rc != 0 {
		t.Errorf("missing kind must still exit 0, got %d", rc)
	}
}

// TestToolprocHookGrantsPolicyEnvelope is the seam-5 wire at the hook: the
// same manifest that admits a tool declares its runtime envelope, and a pre
// firing stamps the resolved grant on the spawn event — exact row first,
// flag pair only when no row matches, fail-open to flags on a bad manifest.
func TestToolprocHookGrantsPolicyEnvelope(t *testing.T) {
	dir := t.TempDir()
	manifest := filepath.Join(dir, "policy.json")
	if err := os.WriteFile(manifest, []byte(`{
		"version": "fak-policy/v1",
		"allow": ["Bash", "Fetch"],
		"tool_runtime": [{"tool": "Bash", "deadline_ms": 600000, "heartbeat_every_ms": 30000}]
	}`), 0o644); err != nil {
		t.Fatal(err)
	}

	spawn := func(t *testing.T, journal string, argv []string, payload string) toolproc.Event {
		t.Helper()
		var errOut strings.Builder
		if rc := runToolprocHook(strings.NewReader(payload), &errOut, argv); rc != 0 {
			t.Fatalf("hook exit %d, stderr %q", rc, errOut.String())
		}
		f, err := os.Open(journal)
		if err != nil {
			t.Fatal(err)
		}
		defer f.Close()
		evs, err := toolproc.ParseEvents(f)
		if err != nil {
			t.Fatal(err)
		}
		if len(evs) == 0 {
			t.Fatal("no journal event appended")
		}
		return evs[len(evs)-1]
	}

	// The manifest row wins over the flag pair for a tool it names.
	j1 := filepath.Join(dir, "j1.jsonl")
	ev := spawn(t, j1, []string{"pre", "--journal", j1, "--policy", manifest, "--deadline-ms", "5"},
		`{"session_id":"s1","tool_name":"Bash","tool_use_id":"toolu_p1"}`)
	if ev.DeadlineMS != 600000 || ev.HeartbeatEveryMS != 30000 {
		t.Fatalf("spawn envelope = %d/%d, want the manifest row 600000/30000", ev.DeadlineMS, ev.HeartbeatEveryMS)
	}

	// No row and no catch-all: the flag pair fills.
	j2 := filepath.Join(dir, "j2.jsonl")
	ev = spawn(t, j2, []string{"pre", "--journal", j2, "--policy", manifest, "--deadline-ms", "5"},
		`{"session_id":"s1","tool_name":"Fetch","tool_use_id":"toolu_p2"}`)
	if ev.DeadlineMS != 5 || ev.HeartbeatEveryMS != 0 {
		t.Fatalf("spawn envelope = %d/%d, want the flag fallback 5/0", ev.DeadlineMS, ev.HeartbeatEveryMS)
	}

	// An unreadable manifest falls open to the flags — never wedges the hook.
	j3 := filepath.Join(dir, "j3.jsonl")
	var errOut strings.Builder
	if rc := runToolprocHook(strings.NewReader(`{"session_id":"s1","tool_name":"Bash","tool_use_id":"toolu_p3"}`),
		&errOut, []string{"pre", "--journal", j3, "--policy", filepath.Join(dir, "absent.json"), "--deadline-ms", "7"}); rc != 0 {
		t.Fatalf("hook exit %d on unreadable policy, want 0 (fail-open)", rc)
	}
	if !strings.Contains(errOut.String(), "fail-open") {
		t.Fatalf("unreadable policy must be reported, got %q", errOut.String())
	}
	ev = spawn(t, j3, []string{"post", "--journal", j3},
		`{"session_id":"s1","tool_name":"Bash","tool_use_id":"toolu_p3","tool_response":{"is_error":false}}`)
	if ev.Kind != toolproc.EvExit {
		t.Fatalf("post after fail-open spawn = %s, want exit (spawn landed with flag envelope)", ev.Kind)
	}
}

// TestToolprocHookBridgesBackgroundJob is the streamed-output pulse source at
// the file level: a background launch post spawns the job proc alongside the
// launch call's exit, each output poll pulses it, and the completion poll
// exits it — one journal, fold-clean, the job visible the whole way.
func TestToolprocHookBridgesBackgroundJob(t *testing.T) {
	journal := filepath.Join(t.TempDir(), "journal.jsonl")
	steps := []struct {
		kind, payload string
		atMS          int64
	}{
		{"pre", `{"session_id":"s1","tool_name":"Bash","tool_use_id":"toolu_l","tool_input":{"command":"make bench","run_in_background":true}}`, 1_000},
		{"post", `{"session_id":"s1","tool_name":"Bash","tool_use_id":"toolu_l","tool_input":{"command":"make bench","run_in_background":true},"tool_response":"Command running in background with ID: j1"}`, 2_000},
		{"pre", `{"session_id":"s1","tool_name":"BashOutput","tool_use_id":"toolu_p1","tool_input":{"bash_id":"j1"}}`, 8_000},
		{"post", `{"session_id":"s1","tool_name":"BashOutput","tool_use_id":"toolu_p1","tool_input":{"bash_id":"j1"},"tool_response":{"status":"running","stdout":"chunk"}}`, 9_000},
		{"pre", `{"session_id":"s1","tool_name":"BashOutput","tool_use_id":"toolu_p2","tool_input":{"bash_id":"j1"}}`, 20_000},
		{"post", `{"session_id":"s1","tool_name":"BashOutput","tool_use_id":"toolu_p2","tool_input":{"bash_id":"j1"},"tool_response":{"status":"completed"}}`, 21_000},
	}
	for _, s := range steps {
		if err := toolprocHookOnce(strings.NewReader(s.payload), s.kind, journal, toolproc.HookEnvelope{}, s.atMS); err != nil {
			t.Fatalf("hook %s @%d: %v", s.kind, s.atMS, err)
		}
	}
	f, err := os.Open(journal)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	events, err := toolproc.ParseEvents(f)
	if err != nil {
		t.Fatalf("journal must be fold-clean: %v", err)
	}
	tab, err := toolproc.Fold(events, 30_000, toolproc.Config{})
	if err != nil {
		t.Fatal(err)
	}
	var job *toolproc.Proc
	for i := range tab.Procs {
		if tab.Procs[i].CallID == "bg:j1" {
			job = &tab.Procs[i]
		}
	}
	if job == nil {
		t.Fatalf("bridged job missing from table: %+v", tab.Procs)
	}
	if job.Tool != "Bash[bg]" || job.State != toolproc.StateDone || job.ExitStatus != "ok" || job.Pulses != 1 {
		t.Fatalf("job = tool=%s state=%s exit=%s pulses=%d, want Bash[bg]/DONE/ok/1",
			job.Tool, job.State, job.ExitStatus, job.Pulses)
	}
}
