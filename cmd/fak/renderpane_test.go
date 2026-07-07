package main

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/toolproc"
	"github.com/anthony-chaudhary/fak/internal/toolprocgate"
)

// TestRenderPaneCrashSparesSupervisorAndSiblings is the #2170 acceptance
// row-2 regression witness (split to #3139): a TUI render pane CRASHES
// mid-paint (a real panic, the render-surface analogue of the 2026-07-01
// pwsh FailFast) and
//
//   - the supervisor survives: the panic stops at the pane boundary, the
//     fold still answers, and NEW work still admits afterwards,
//   - the sibling session's live proc is untouched (no shared fate),
//   - the crash is recorded as the typed ConsoleRendererExit class via
//     Supervisor.ExitConsoleFault — a structured child failure, not prose,
//   - the journaled row is searchable from the operator surface:
//     `fak toolproc console-faults` folds the same JSONL (#2170 row 4).
func TestRenderPaneCrashSparesSupervisorAndSiblings(t *testing.T) {
	sup := toolprocgate.NewSupervisor(toolproc.Config{})
	// The sibling: a live long-runner in another session that must survive
	// the pane crash untouched.
	if err := sup.Spawn("sibling-call", "bg_tail", "s-sibling", 0, 0, 1_000, nil); err != nil {
		t.Fatalf("spawn sibling: %v", err)
	}
	// The render pane itself: a supervised child like any tool call.
	if err := sup.Spawn("render-pane", "tui[pane]", "s-operator", 0, 0, 1_000, nil); err != nil {
		t.Fatalf("spawn render pane: %v", err)
	}

	journalPath := filepath.Join(t.TempDir(), "console-faults.jsonl")
	sink, err := os.OpenFile(journalPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		t.Fatalf("open fault sink: %v", err)
	}

	ev, faulted, err := runRenderPane(sup, "render-pane", func() int64 { return 2_000 }, sink, func() error {
		panic("render pane blew up mid-paint")
	})
	// Reaching this line at all is half the witness: the pane's panic did NOT
	// unwind into the test (the supervisor's frame).
	if err != nil {
		t.Fatalf("runRenderPane seam error: %v", err)
	}
	if closeErr := sink.Close(); closeErr != nil {
		t.Fatalf("close fault sink: %v", closeErr)
	}
	if !faulted {
		t.Fatal("renderer crash not recorded as a fault")
	}
	if ev.Class != toolprocgate.ConsoleRendererExit {
		t.Fatalf("fault class = %q, want %q", ev.Class, toolprocgate.ConsoleRendererExit)
	}
	if ev.Surface != string(toolprocgate.ConsoleSurfaceRenderer) {
		t.Fatalf("fault surface = %q, want renderer", ev.Surface)
	}
	if ev.Tool != "tui[pane]" || ev.Session != "s-operator" {
		t.Fatalf("fault row lost child identity: tool=%q session=%q", ev.Tool, ev.Session)
	}
	if !strings.Contains(ev.Detail, "render pane blew up mid-paint") {
		t.Fatalf("fault detail lost the crash signature: %q", ev.Detail)
	}

	// The supervisor's table stays consistent: the pane folded to a normal
	// exit(error), the sibling is still RUNNING and not orphaned.
	tab, err := sup.Table(3_000)
	if err != nil {
		t.Fatalf("supervisor table after pane crash: %v", err)
	}
	var sawPane, sawSibling bool
	for _, p := range tab.Procs {
		switch p.CallID {
		case "render-pane":
			sawPane = true
			if p.State != toolproc.StateDone || p.ExitStatus != "error" {
				t.Fatalf("render pane state=%s exit=%q, want DONE/error", p.State, p.ExitStatus)
			}
		case "sibling-call":
			sawSibling = true
			if p.State != toolproc.StateRunning {
				t.Fatalf("sibling state=%s, want RUNNING (shared fate!)", p.State)
			}
			if p.Orphaned {
				t.Fatal("sibling orphaned by the pane crash")
			}
		}
	}
	if !sawPane || !sawSibling {
		t.Fatalf("table lost procs: pane=%v sibling=%v", sawPane, sawSibling)
	}

	// The supervisor still ADMITS new work after the crash — alive, not wedged.
	if err := sup.Spawn("post-crash", "bg_tail", "s-sibling", 0, 0, 4_000, nil); err != nil {
		t.Fatalf("supervisor refused new work after pane crash: %v", err)
	}

	// Row 4 binding: the journaled row round-trips the fail-closed parser and
	// the operator CLI folds it — the crash class is searchable from the fak
	// surface, not Event Viewer.
	f, err := os.Open(journalPath)
	if err != nil {
		t.Fatalf("open journal: %v", err)
	}
	rows, err := toolprocgate.ParseConsoleFaultEvents(f)
	f.Close()
	if err != nil {
		t.Fatalf("journaled fault row failed the fail-closed parser: %v", err)
	}
	if len(rows) != 1 || rows[0].Class != toolprocgate.ConsoleRendererExit {
		t.Fatalf("journal rows = %+v, want one CONSOLE_RENDERER_EXIT", rows)
	}
	var stdout, stderr strings.Builder
	if rc := runToolproc(&stdout, &stderr, []string{"console-faults", "--events", journalPath}); rc != 0 {
		t.Fatalf("toolproc console-faults rc=%d stderr=%s", rc, stderr.String())
	}
	for _, want := range []string{"console faults: 1 row(s)", "CONSOLE_RENDERER_EXIT", "render-pane"} {
		if !strings.Contains(stdout.String(), want) {
			t.Fatalf("operator report missing %q:\n%s", want, stdout.String())
		}
	}
}

// TestRenderPaneErrorExitRecordsRendererFault: a pane run loop that returns an
// error died unexpectedly too — same typed class, same containment, no panic
// required.
func TestRenderPaneErrorExitRecordsRendererFault(t *testing.T) {
	sup := toolprocgate.NewSupervisor(toolproc.Config{})
	if err := sup.Spawn("render-pane", "tui[pane]", "s-operator", 0, 0, 1_000, nil); err != nil {
		t.Fatalf("spawn render pane: %v", err)
	}
	var journal strings.Builder
	ev, faulted, err := runRenderPane(sup, "render-pane", func() int64 { return 2_000 }, &journal, func() error {
		return errors.New("pty write: broken pipe")
	})
	if err != nil || !faulted {
		t.Fatalf("faulted=%v err=%v, want fault recorded cleanly", faulted, err)
	}
	if ev.Class != toolprocgate.ConsoleRendererExit || !strings.Contains(ev.Detail, "broken pipe") {
		t.Fatalf("fault row = %+v, want CONSOLE_RENDERER_EXIT carrying the exit error", ev)
	}
	if !strings.Contains(journal.String(), string(toolprocgate.ConsoleRendererExit)) {
		t.Fatalf("journal missing typed class: %q", journal.String())
	}
}

// TestRenderPaneCleanDetachRecordsNoFault: a nil return is an intentional
// detach — normal exit(ok), no fault row, nothing journaled.
func TestRenderPaneCleanDetachRecordsNoFault(t *testing.T) {
	sup := toolprocgate.NewSupervisor(toolproc.Config{})
	if err := sup.Spawn("render-pane", "tui[pane]", "s-operator", 0, 0, 1_000, nil); err != nil {
		t.Fatalf("spawn render pane: %v", err)
	}
	var journal strings.Builder
	_, faulted, err := runRenderPane(sup, "render-pane", func() int64 { return 2_000 }, &journal, func() error {
		return nil
	})
	if err != nil || faulted {
		t.Fatalf("faulted=%v err=%v, want clean detach", faulted, err)
	}
	if journal.Len() != 0 {
		t.Fatalf("clean detach journaled %q, want nothing", journal.String())
	}
	tab, err := sup.Table(3_000)
	if err != nil {
		t.Fatalf("table: %v", err)
	}
	if len(tab.Procs) != 1 || tab.Procs[0].State != toolproc.StateDone || tab.Procs[0].ExitStatus != "ok" {
		t.Fatalf("procs = %+v, want one DONE/ok", tab.Procs)
	}
}
