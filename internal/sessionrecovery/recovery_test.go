package sessionrecovery

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"time"

	"github.com/anthony-chaudhary/fak/internal/sessionjournal"
)

func candidate(cwd string) Session {
	return Session{Thread: &Thread{ID: "t1", Source: "interactive_tui", CWD: cwd}, LatestTurn: &Turn{Status: "inProgress", StartedAt: "2026-08-18T01:00:00Z"}}
}

func TestSelectIgnoresInventoryRowsWithoutThreads(t *testing.T) {
	report := InventoryReport{Sessions: []Session{
		{},
		candidate(`C:\work\fak`),
		{LatestTurn: &Turn{Status: "inProgress"}},
	}}
	got := Select(report, Options{Limit: 1, ReceiptDir: t.TempDir()})
	if len(got) != 1 || got[0].ThreadID != "t1" {
		t.Fatalf("requests=%+v", got)
	}
}

func TestSelectPreservesPromptAsOneArg(t *testing.T) {
	got := Select(InventoryReport{Sessions: []Session{candidate(`C:\work\fak`)}}, Options{Limit: 1, Prompt: "continue the exact task", ReceiptDir: t.TempDir()})
	want := []string{"fak", "guard", "--", "codex", "resume", "t1", "continue the exact task"}
	if len(got) != 1 || !reflect.DeepEqual(got[0].Argv, want) {
		t.Fatalf("argv=%q want %q", got[0].Argv, want)
	}
}
func TestSelectCWDHandling(t *testing.T) {
	report := InventoryReport{Sessions: []Session{candidate("")}}
	got := Select(report, Options{Limit: 1, ReceiptDir: t.TempDir()})
	if got[0].Status != "refused" || got[0].Reason != "cwd_unknown" {
		t.Fatalf("got %+v", got[0])
	}
	got = Select(report, Options{Limit: 1, CWDOverride: `D:\authoritative`, ReceiptDir: t.TempDir()})
	if got[0].CWD != `D:\authoritative` || got[0].Status != "candidate" {
		t.Fatalf("override=%+v", got[0])
	}
}
func TestReceiptIsLedgerFirstAndIdempotent(t *testing.T) {
	dir := t.TempDir()
	req := Select(InventoryReport{Sessions: []Session{candidate(dir)}}, Options{Limit: 1, ReceiptDir: dir})[0]
	wrote, err := WriteReceipt(req, time.Unix(1, 0))
	if err != nil || !wrote {
		t.Fatalf("first %v %v", wrote, err)
	}
	wrote, err = WriteReceipt(req, time.Unix(2, 0))
	if err != nil || wrote {
		t.Fatalf("second %v %v", wrote, err)
	}
	var r Receipt
	b, _ := os.ReadFile(filepath.Clean(req.ReceiptPath))
	if json.Unmarshal(b, &r) != nil || r.State != "launch_intent" {
		t.Fatalf("receipt=%s", b)
	}
}
func TestWitnessRequiresProcessGuardAndNewTurn(t *testing.T) {
	before := InventoryReport{Sessions: []Session{candidate(`C:\x`)}}
	after := InventoryReport{Sessions: []Session{candidate(`C:\x`)}}
	after.Sessions[0].ProcessTrees = []ProcessTree{{RootPID: 42}}
	if got := Witness(before, after, "t1"); got != "launched_unproven" {
		t.Fatal(got)
	}
	after.Sessions[0].GuardReceipt = &GuardReceipt{RecordedAt: "2026-08-18T02:00:00Z"}
	after.Sessions[0].LatestTurn = &Turn{Status: "inProgress", StartedAt: "2026-08-18T03:00:00Z"}
	if got := Witness(before, after, "t1"); got != "productive" {
		t.Fatal(got)
	}
}

func TestSelectTwentySessionCohort(t *testing.T) {
	report := InventoryReport{}
	for i := 0; i < 25; i++ {
		row := candidate(`C:\work\fak`)
		row.Thread.ID = fmt.Sprintf("t%02d", i)
		report.Sessions = append(report.Sessions, row)
	}
	got := Select(report, Options{Limit: 20, ReceiptDir: t.TempDir()})
	if len(got) != 20 {
		t.Fatalf("got %d candidates", len(got))
	}
	for _, req := range got {
		if req.Status != "candidate" {
			t.Fatalf("request=%+v", req)
		}
	}
}

func TestMergeJournalCrashesUsesRecordedCWDAndDeduplicates(t *testing.T) {
	classified := []sessionjournal.Classified{
		{Session: sessionjournal.Session{ID: "already", CWD: `C:\authoritative`}, Status: sessionjournal.StatusCrashed, Reason: "MACHINE_REBOOT"},
		{Session: sessionjournal.Session{ID: "journal", CWD: `D:\repos\real tree`}, Status: sessionjournal.StatusCrashed, Reason: "MACHINE_REBOOT"},
		{Session: sessionjournal.Session{ID: "live", CWD: `D:\repos\live`}, Status: sessionjournal.StatusLive},
	}
	got := MergeJournalCrashes([]Request{{ThreadID: "already", CWD: `C:\slug-recovered`}}, classified, Options{Limit: 3, Prompt: "resume safely", ReceiptDir: t.TempDir()})
	if len(got) != 2 {
		t.Fatalf("requests=%+v", got)
	}
	if got[0].ThreadID != "already" || got[0].CWD != `C:\authoritative` || got[0].Source != "session_journal" {
		t.Fatalf("journal did not replace reconstructed cwd: %+v", got[0])
	}
	wantArgv := []string{"fak", "guard", "--", "codex", "resume", "journal", "resume safely"}
	if got[1].CWD != `D:\repos\real tree` || got[1].Source != "session_journal" || !reflect.DeepEqual(got[1].Argv, wantArgv) {
		t.Fatalf("journal request=%+v", got[1])
	}
}

func TestVisibleLauncherRefusesMissingCommandBeforeTerminalSpawn(t *testing.T) {
	launcher := VisibleLauncher{TerminalBin: filepath.Join(t.TempDir(), "terminal-that-must-not-run.exe")}
	err := launcher.Launch(Request{CWD: t.TempDir(), Argv: []string{"definitely-missing-session-recovery-command"}})
	if err == nil || !strings.Contains(err.Error(), "resolve") {
		t.Fatalf("err=%v want command-resolution refusal", err)
	}
}
