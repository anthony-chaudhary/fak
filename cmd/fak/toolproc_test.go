package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strconv"
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

// TestToolprocHookStopCompactsOversizedJournal proves the SessionEnd wiring end
// to end: a shared journal that has grown past the tail-read window is bounded
// by the stop firing, which reclaims fully-terminal history while preserving a
// long-lived spawn that sat far outside the window — the #3488 leak fix.
func TestToolprocHookStopCompactsOversizedJournal(t *testing.T) {
	journal := filepath.Join(t.TempDir(), "journal.jsonl")
	beforeSize := writeOversizedJournal(t, journal)

	// The stop firing appends the session_end and then compacts.
	if err := toolprocHookOnce(strings.NewReader(`{"session_id":"s1"}`), "stop", journal, toolproc.HookEnvelope{}, 9_000); err != nil {
		t.Fatalf("stop hook: %v", err)
	}

	after, err := os.Stat(journal)
	if err != nil {
		t.Fatal(err)
	}
	if after.Size() >= beforeSize || after.Size() > toolproc.JournalCompactThresholdBytes {
		t.Fatalf("stop hook did not bound the journal: before=%d after=%d threshold=%d",
			beforeSize, after.Size(), int64(toolproc.JournalCompactThresholdBytes))
	}

	// The long-lived spawn survived compaction and folds to a running (now
	// orphaned, since its session ended) proc — the whole point of preserving it.
	tab := foldJournalFile(t, journal, 10_000)
	var live *toolproc.Proc
	for i := range tab.Procs {
		if tab.Procs[i].CallID == "orphan-live" {
			live = &tab.Procs[i]
		}
	}
	if live == nil {
		t.Fatalf("long-lived spawn dropped by stop-hook compaction: %+v", tab.Counts)
	}
	// It must survive as the RUNNING, now-orphaned proc — not merely as a row.
	// A compaction that kept the spawn but lost its session_end would fold it as
	// a healthy RUNNING proc, and this assertion is what catches that.
	if live.State != toolproc.StateRunning || !live.Orphaned {
		t.Fatalf("preserved spawn not running+orphaned: state=%s orphaned=%v", live.State, live.Orphaned)
	}
}

// TestToolprocHookGrowthCompactsWithoutAnyStop is #3557's guard: the journal bound
// must not depend on a clean SessionEnd ever arriving. A session killed mid-tool —
// OOM, host reboot, a harness that skips Stop — fires only pre/post, and on a box
// where that is the norm a stop-GATED compaction never runs at all, so the shared
// file grows until some sibling session happens to exit cleanly. Here an oversized
// journal is bounded by an ordinary PRE firing with no stop fired at any point, and
// the far-older live spawn plus the firing's own fresh spawn both survive.
func TestToolprocHookGrowthCompactsWithoutAnyStop(t *testing.T) {
	journal := filepath.Join(t.TempDir(), "journal.jsonl")
	beforeSize := writeOversizedJournal(t, journal)

	pre := `{"session_id":"s1","tool_name":"Bash","tool_use_id":"toolu_a","tool_input":{"command":"make test"}}`
	if err := toolprocHookOnce(strings.NewReader(pre), "pre", journal, toolproc.HookEnvelope{}, 9_000); err != nil {
		t.Fatalf("pre hook: %v", err)
	}

	after, err := os.Stat(journal)
	if err != nil {
		t.Fatal(err)
	}
	if after.Size() >= beforeSize || after.Size() > toolproc.JournalCompactThresholdBytes {
		t.Fatalf("an ordinary pre firing did not bound the journal: before=%d after=%d threshold=%d",
			beforeSize, after.Size(), int64(toolproc.JournalCompactThresholdBytes))
	}

	// Growth-triggered compaction keeps CompactJournal's invariant exactly as the
	// stop-triggered one does: the pre-existing long-lived spawn is still there, and
	// so is the spawn this very firing appended — a bound that swallowed the current
	// call's own event would break the pairing the next post firing needs.
	tab := foldJournalFile(t, journal, 10_000)
	seen := map[string]toolproc.State{}
	for _, p := range tab.Procs {
		seen[p.CallID] = p.State
	}
	if st, ok := seen["orphan-live"]; !ok || st != toolproc.StateRunning {
		t.Fatalf("long-lived spawn lost to growth-triggered compaction: state=%q present=%v", st, ok)
	}
	if st, ok := seen["toolu_a"]; !ok || st != toolproc.StateRunning {
		t.Fatalf("this firing's own spawn lost to growth-triggered compaction: state=%q present=%v", st, ok)
	}
}

// TestToolprocHookGrowthCompactsPastUnreadableTail is #3557's crash-heavy case at
// its sharpest. The SAME kill that skips Stop also tears the journal: the append
// path holds no fsync, so an OOM-kill, a host reboot, or a power loss mid-write
// leaves a truncated final row. The strict fold reader is fail-closed by design,
// so that one row makes every subsequent firing's ParseTailFile refuse — and a
// refusal that returns before the compaction leaves the file pinned above the
// window with NO firing, pre/post/stop alike, able to reclaim it: growth-triggered
// compaction is exactly as dead as the stop-gated one it replaced.
//
// It is the hazard CompactJournalFile already solved one level down (#3556's
// lenient read exists so a bad row is EXPELLED rather than making the file
// un-boundable forever) — reachable only if the firing actually gets there. So an
// ordinary pre firing must still bound a torn journal, and the fault must still be
// reported rather than swallowed.
func TestToolprocHookGrowthCompactsPastUnreadableTail(t *testing.T) {
	journal := filepath.Join(t.TempDir(), "journal.jsonl")
	beforeSize := writeOversizedJournal(t, journal)

	// What a session killed mid-append leaves behind: a half-written final row.
	f, err := os.OpenFile(journal, os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.WriteString(`{"kind":"spawn","call_id":"torn","at_uni` + "\n"); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	beforeSize, err = func() (int64, error) { fi, err := os.Stat(journal); return fi.Size(), err }()
	if err != nil {
		t.Fatal(err)
	}

	// The torn row must genuinely defeat the strict reader, or this test proves nothing.
	if _, err := toolproc.ParseTailFile(journal); err == nil {
		t.Fatal("fixture is not actually unreadable: ParseTailFile accepted the torn row")
	}

	pre := `{"session_id":"s1","tool_name":"Bash","tool_use_id":"toolu_a","tool_input":{"command":"make test"}}`
	hookErr := toolprocHookOnce(strings.NewReader(pre), "pre", journal, toolproc.HookEnvelope{}, 9_000)

	after, err := os.Stat(journal)
	if err != nil {
		t.Fatal(err)
	}
	if after.Size() >= beforeSize || after.Size() > toolproc.JournalCompactThresholdBytes {
		t.Fatalf("a torn tail row blocked the bound: before=%d after=%d threshold=%d",
			beforeSize, after.Size(), int64(toolproc.JournalCompactThresholdBytes))
	}

	// The unreadable-journal fault is still surfaced (runToolprocHook renders it
	// fail-open); bounding the file must not silently swallow it.
	if hookErr == nil {
		t.Error("the unreadable-tail fault must still be reported, not swallowed by the compaction")
	}

	// The compaction expelled the torn row, so the journal comes back fold-clean and
	// the next firing parses normally — the wedge is self-healing, not permanent.
	tab := foldJournalFile(t, journal, 10_000)
	var found bool
	for _, p := range tab.Procs {
		if p.CallID == "orphan-live" {
			found = p.State == toolproc.StateRunning
		}
		if p.CallID == "torn" {
			t.Error("the torn row survived compaction")
		}
	}
	if !found {
		t.Fatalf("long-lived spawn lost while bounding a torn journal: %+v", tab.Counts)
	}
}

// writeOversizedJournal builds a shared journal past JournalCompactThresholdBytes:
// one live spawn at the very front — older than the 4 MiB tail window, so only
// CompactJournal's keep-every-un-exited-spawn invariant can save it — followed by
// enough fully-terminal spawn/exit pairs to push the file over the threshold. It
// returns the pre-compaction size.
func writeOversizedJournal(t *testing.T, path string) int64 {
	t.Helper()
	var buf []byte
	appendEvent := func(ev toolproc.Event) {
		line, err := json.Marshal(ev)
		if err != nil {
			t.Fatal(err)
		}
		buf = append(buf, append(line, '\n')...)
	}
	appendEvent(toolproc.Event{Kind: toolproc.EvSpawn, CallID: "orphan-live", Tool: "Bash", Session: "s1", AtMS: 1})
	for i := 0; len(buf) <= toolproc.JournalCompactThresholdBytes+(64<<10); i++ {
		id := "done-" + strconv.Itoa(i)
		appendEvent(toolproc.Event{Kind: toolproc.EvSpawn, CallID: id, Tool: "Bash", Session: "s1", AtMS: int64(2 + 2*i)})
		appendEvent(toolproc.Event{Kind: toolproc.EvExit, CallID: id, AtMS: int64(3 + 2*i), Status: "ok"})
	}
	if err := os.WriteFile(path, buf, 0o644); err != nil {
		t.Fatal(err)
	}
	fi, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if fi.Size() <= toolproc.JournalCompactThresholdBytes {
		t.Fatalf("test fixture must exceed the threshold, got %d", fi.Size())
	}
	return fi.Size()
}

// foldJournalFile reads a journal through the STRICT fold reader (ParseEvents) and
// folds it, so any post-compaction journal that is not fold-clean fails here rather
// than silently yielding a thinner table.
func foldJournalFile(t *testing.T, path string, nowMS int64) toolproc.Table {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	events, err := toolproc.ParseEvents(f)
	if err != nil {
		t.Fatalf("journal must stay fold-clean: %v", err)
	}
	tab, err := toolproc.Fold(events, nowMS, toolproc.Config{})
	if err != nil {
		t.Fatal(err)
	}
	return tab
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
		// Session-qualified: the harness's background id is per-session and this
		// journal is workspace-shared (#5880).
		if tab.Procs[i].CallID == "bg:s1:j1" {
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

// TestAppendJournalLinesLeavesNoHandleBlockingSwap pins the #3555/#3671
// contract of the appender call-site directly: appendJournalLines appends
// (never truncates) and closes its handle before returning, so the very next
// step the stop-hook takes — swapping the journal via a rename during
// CompactJournalFile — is never blocked by a lingering open handle. On Windows
// a plain non-share-delete open left dangling would deny this rename; that is
// exactly the third open-site #3555 named and this test is its regression guard.
func TestAppendJournalLinesLeavesNoHandleBlockingSwap(t *testing.T) {
	journal := filepath.Join(t.TempDir(), "journal.jsonl")

	// Two successive appends: the second must extend the first, proving the
	// open is append-mode (O_APPEND) and each call closed cleanly rather than
	// reopening a truncating handle.
	if err := appendJournalLines(journal, []byte("first\n")); err != nil {
		t.Fatalf("first append: %v", err)
	}
	if err := appendJournalLines(journal, []byte("second\n")); err != nil {
		t.Fatalf("second append: %v", err)
	}
	got, err := os.ReadFile(journal)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != "first\nsecond\n" {
		t.Fatalf("append did not extend the journal: got %q, want %q", got, "first\nsecond\n")
	}

	// The swap the stop-hook compaction performs. If appendJournalLines held an
	// open handle across return, this rename is denied on Windows (the #3555
	// contention) — the failure this guard exists to catch.
	swapped := journal + ".swapped"
	if err := os.Rename(journal, swapped); err != nil {
		t.Fatalf("journal swap blocked by a lingering append handle: %v", err)
	}
	// And the rename moved the content intact, not an empty stand-in.
	after, err := os.ReadFile(swapped)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(after), "second") {
		t.Fatalf("swapped journal lost its content: %q", after)
	}
}
