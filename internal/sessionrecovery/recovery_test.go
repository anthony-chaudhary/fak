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

func TestSelectExplicitDeadExecIsCandidate(t *testing.T) {
	report := InventoryReport{Sessions: []Session{{Thread: &Thread{ID: "exec-1", Source: "exec", CWD: `C:\work\fak`}, LatestTurn: &Turn{Status: "inProgress"}}}}
	got := Select(report, Options{ManagerBin: `C:\bin\fak.exe`, Threads: map[string]bool{"exec-1": true}, Limit: 1, ReceiptDir: t.TempDir()})
	if len(got) != 1 || got[0].Status != "candidate" || got[0].Source != "exec" || got[0].Reason != "explicit_dead_exec" {
		t.Fatalf("requests=%+v", got)
	}
}

func TestSelectExplicitDeadExecWithoutLatestTurnIsCandidate(t *testing.T) {
	report := InventoryReport{Sessions: []Session{{Thread: &Thread{ID: "exec-crashed", Source: "exec", CWD: `\\?\C:\work\fak\_scratch\session-recovery-live-drill`}}}}
	got := Select(report, Options{ManagerBin: `C:\bin\fak.exe`, Threads: map[string]bool{"exec-crashed": true}, Limit: 1, ReceiptDir: t.TempDir()})
	if len(got) != 1 || got[0].Status != "candidate" || got[0].Reason != "explicit_dead_exec_no_turn" {
		t.Fatalf("explicit dead exec without latest turn = %#v, want one explained candidate", got)
	}
}

func TestSelectBroadPreviewStillExcludesDeadExec(t *testing.T) {
	report := InventoryReport{Sessions: []Session{{Thread: &Thread{ID: "exec-1", Source: "exec", CWD: `C:\work\fak`}, LatestTurn: &Turn{Status: "inProgress"}}}}
	if got := Select(report, Options{Limit: 1, ReceiptDir: t.TempDir()}); len(got) != 0 {
		t.Fatalf("requests=%+v", got)
	}
}

func TestSelectAdmitsOnlyExactDeadCodexCLI(t *testing.T) {
	const deadID = "93380001-0000-4000-8000-000000000001"
	const liveID = "93380002-0000-4000-8000-000000000002"
	const completedID = "93380003-0000-4000-8000-000000000003"
	const unknownID = "93380004-0000-4000-8000-000000000004"
	const claudeID = "93380005-0000-4000-8000-000000000005"
	const unidentifiedID = "93380006-0000-4000-8000-000000000006"
	report := InventoryReport{Sessions: []Session{
		{Thread: &Thread{ID: deadID, Source: " CLI ", CWD: `C:\work\fak`}, Harness: ProviderCodex, HarnessSource: "legacy_codex_inventory", LatestTurn: &Turn{Status: "inProgress"}},
		{Thread: &Thread{ID: liveID, Source: "cli", CWD: `C:\work\fak`}, Harness: ProviderCodex, HarnessSource: "legacy_codex_inventory", LatestTurn: &Turn{Status: "inProgress"}, ProcessTrees: []ProcessTree{{RootPID: 42, HasCodex: true}}},
		{Thread: &Thread{ID: completedID, Source: "cli", CWD: `C:\work\fak`}, Harness: ProviderCodex, HarnessSource: "legacy_codex_inventory", LatestTurn: &Turn{Status: "completed"}},
		{Thread: &Thread{ID: unknownID, Source: "cli", CWD: `C:\work\fak`}, Harness: ProviderCodex, HarnessSource: "legacy_codex_inventory"},
		{Thread: &Thread{ID: claudeID, Source: "cli", CWD: `C:\work\fak`}, Harness: ProviderClaude, HarnessSource: "session_registration", LatestTurn: &Turn{Status: "inProgress"}},
		{Thread: &Thread{ID: unidentifiedID, Source: "cli", CWD: `C:\work\fak`}, LatestTurn: &Turn{Status: "inProgress"}},
	}}
	threads := map[string]bool{deadID: true, liveID: true, completedID: true, unknownID: true, claudeID: true, unidentifiedID: true}
	got := Select(report, Options{ManagerBin: "fak", CodexBin: "codex", Threads: threads, Limit: len(threads), ReceiptDir: t.TempDir()})
	if len(got) != len(threads) {
		t.Fatalf("requests=%+v", got)
	}
	byID := make(map[string]Request, len(got))
	for _, req := range got {
		byID[req.ThreadID] = req
	}
	wantArgv := []string{"fak", "guard", "--", "codex", "exec", "--cd", `C:\work\fak`, "resume", deadID}
	if dead := byID[deadID]; dead.Status != "candidate" || dead.Provider != ProviderCodex || !reflect.DeepEqual(dead.Argv, wantArgv) {
		t.Fatalf("dead Codex cli request=%+v want argv=%q", dead, wantArgv)
	}
	for _, id := range []string{liveID, completedID, unknownID, claudeID, unidentifiedID} {
		if req := byID[id]; req.Status == "candidate" || len(req.Argv) != 0 {
			t.Errorf("nonlaunchable cli row %s became actionable: %+v", id, req)
		}
	}
}

func TestSelectExplicitIneligibleAndMissingThreadsReturnReasons(t *testing.T) {
	report := InventoryReport{Sessions: []Session{{Thread: &Thread{ID: "done", Source: "interactive_tui", CWD: `C:\work\fak`}, LatestTurn: &Turn{Status: "completed"}}}}
	got := Select(report, Options{Threads: map[string]bool{"done": true, "missing": true}, Limit: 2})
	if len(got) != 2 || got[0].ThreadID != "done" || got[0].Reason != "latest_turn_completed" || got[1].ThreadID != "missing" || got[1].Reason != "thread_not_found" {
		t.Fatalf("requests=%+v", got)
	}
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
	want := []string{"fak", "guard", "--", "codex", "exec", "--cd", `C:\work\fak`, "resume", "t1", "continue the exact task"}
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
func TestFinalizeReceiptPersistsTerminalState(t *testing.T) {
	dir := t.TempDir()
	req := Select(InventoryReport{Sessions: []Session{candidate(dir)}}, Options{Limit: 1, ReceiptDir: dir})[0]
	if wrote, err := WriteReceipt(req, time.Unix(1, 0)); err != nil || !wrote {
		t.Fatalf("write receipt: wrote=%v err=%v", wrote, err)
	}
	if err := FinalizeReceipt(req, "launch_failed", "terminal missing", time.Unix(2, 0)); err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile(req.ReceiptPath)
	if err != nil {
		t.Fatal(err)
	}
	var got Receipt
	if err := json.Unmarshal(b, &got); err != nil {
		t.Fatal(err)
	}
	if got.State != "launch_failed" || got.Reason != "terminal missing" || got.UpdatedAt == "" {
		t.Fatalf("receipt=%+v", got)
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
		{Session: sessionjournal.Session{ID: "journal", CWD: `D:\repos\real tree`, Agent: "claude"}, Status: sessionjournal.StatusCrashed, Reason: "MACHINE_REBOOT"},
		{Session: sessionjournal.Session{ID: "live", CWD: `D:\repos\live`}, Status: sessionjournal.StatusLive},
	}
	got := MergeJournalCrashes([]Request{{ThreadID: "already", CWD: `C:\slug-recovered`}}, classified, Options{Limit: 3, Prompt: "resume safely", ReceiptDir: t.TempDir()})
	if len(got) != 2 {
		t.Fatalf("requests=%+v", got)
	}
	if got[0].ThreadID != "already" || got[0].CWD != `C:\authoritative` || got[0].Source != "session_journal" {
		t.Fatalf("journal did not replace reconstructed cwd: %+v", got[0])
	}
	wantArgv := []string{"fak", "guard", "--", "claude", "--resume", "journal", "resume safely"}
	if got[1].CWD != `D:\repos\real tree` || got[1].Source != "session_journal" || got[1].Provider != ProviderClaude || !reflect.DeepEqual(got[1].Argv, wantArgv) {
		t.Fatalf("journal request=%+v", got[1])
	}
}

func TestMergeJournalCrashesUsesCodexOnlyWithStateRequest(t *testing.T) {
	const id = "94000001-0000-4000-8000-000000000001"
	requests := []Request{{ThreadID: id, CWD: `C:\old`, Provider: ProviderCodex, Category: CategorySubstantive, Action: ActionRecover, Status: "candidate"}}
	classified := []sessionjournal.Classified{{Session: sessionjournal.Session{ID: id, CWD: `D:\authoritative`, Agent: "codex"}, Status: sessionjournal.StatusCrashed, Reason: "MACHINE_REBOOT"}}
	got := MergeJournalCrashes(requests, classified, Options{Limit: 1, ReceiptDir: t.TempDir()})
	want := []string{"fak", "guard", "--", "codex", "exec", "--cd", `D:\authoritative`, "resume", id}
	if len(got) != 1 || got[0].Provider != ProviderCodex || !reflect.DeepEqual(got[0].Argv, want) {
		t.Fatalf("requests=%+v want=%q", got, want)
	}

	blocked := MergeJournalCrashes(nil, classified, Options{Limit: 1, ReceiptDir: t.TempDir()})
	if len(blocked) != 1 || blocked[0].Status != "identity_blocked" || len(blocked[0].Argv) != 0 {
		t.Fatalf("unverified Codex journal row became actionable: %+v", blocked)
	}
}

func TestMergeJournalCrashesAccountsAllRowsAndLimitsOnlyCandidates(t *testing.T) {
	requests := []Request{
		{ThreadID: "probe", Provider: ProviderClaude, Category: CategoryProbe, Action: ActionExcludeProbe, Status: "probe"},
		{ThreadID: "blocked", Provider: ProviderClaude, Category: CategoryIdentityBlocked, Action: ActionLoginRequired, Status: "identity_blocked"},
		{ThreadID: "candidate-a", CWD: `C:\a`, Provider: ProviderClaude, Category: CategorySubstantive, Action: ActionRecover, Status: "candidate"},
		{ThreadID: "candidate-b", CWD: `C:\b`, Provider: ProviderClaude, Category: CategorySubstantive, Action: ActionRecover, Status: "candidate"},
	}
	classified := []sessionjournal.Classified{{Session: sessionjournal.Session{ID: "journal-c", CWD: `C:\c`, Agent: "claude"}, Status: sessionjournal.StatusCrashed, Reason: "MACHINE_REBOOT"}}
	got := MergeJournalCrashes(requests, classified, Options{Limit: 2, ReceiptDir: t.TempDir()})
	if len(got) != 5 {
		t.Fatalf("cohort truncated: %+v", got)
	}
	candidates, deferred := 0, 0
	for _, req := range got {
		switch req.Status {
		case "candidate":
			candidates++
		case "deferred":
			deferred++
		}
	}
	if candidates != 2 || deferred != 1 {
		t.Fatalf("candidate cap counted non-actions: candidates=%d deferred=%d rows=%+v", candidates, deferred, got)
	}
}

func TestVisibleLauncherRefusesMissingCommandBeforeTerminalSpawn(t *testing.T) {
	launcher := VisibleLauncher{TerminalBin: filepath.Join(t.TempDir(), "terminal-that-must-not-run.exe")}
	err := launcher.Launch(Request{CWD: t.TempDir(), Argv: []string{"definitely-missing-session-recovery-command"}})
	if err == nil || !strings.Contains(err.Error(), "resolve") {
		t.Fatalf("err=%v want command-resolution refusal", err)
	}
}

func TestSelectUsesRecordedHarnessAndLogsProvenance(t *testing.T) {
	report := InventoryReport{Sessions: []Session{{
		Thread:  &Thread{ID: "0198ec3d-6d66-7c82-a700-cedf64660a44", Source: "session_registration", CWD: `C:\work\fak`},
		Harness: ProviderCodex, HarnessSource: "session_registration", Category: CategorySubstantive, Action: ActionRecover,
	}}}
	got := Select(report, Options{Limit: 1, ReceiptDir: t.TempDir(), ManagerBin: "fak", CodexBin: "codex"})
	if len(got) != 1 {
		t.Fatalf("requests=%d", len(got))
	}
	if got[0].Harness != ProviderCodex || got[0].HarnessSource != "session_registration" || got[0].Provider != ProviderCodex {
		t.Fatalf("identity not preserved: %+v", got[0])
	}
	if strings.Contains(strings.Join(got[0].Argv, " "), "claude") || !strings.Contains(strings.Join(got[0].Argv, " "), "codex") {
		t.Fatalf("wrong harness argv: %q", got[0].Argv)
	}
}

func TestSelectFailsClosedWhenHarnessUnknown(t *testing.T) {
	report := InventoryReport{Sessions: []Session{{
		Thread:   &Thread{ID: "0198ec3d-6d66-7c82-a700-cedf64660a44", Source: "session_registration", CWD: `C:\work\fak`},
		Category: CategorySubstantive, Action: ActionRecover,
	}}}
	got := Select(report, Options{Limit: 1, ReceiptDir: t.TempDir()})
	if len(got) != 1 || got[0].Status != "identity_blocked" || got[0].Reason != "harness_identity_unavailable" || len(got[0].Argv) != 0 {
		t.Fatalf("unknown harness must fail closed: %+v", got)
	}
}

func TestMergeJournalCrashesDoesNotDefaultMissingAgentToClaude(t *testing.T) {
	id := "0198ec3d-6d66-7c82-a700-cedf64660a44"
	classified := []sessionjournal.Classified{{Status: sessionjournal.StatusCrashed, Reason: "process_dead", Session: sessionjournal.Session{ID: id, CWD: `C:\work\fak`}}}
	got := MergeJournalCrashes(nil, classified, Options{Limit: 1, ReceiptDir: t.TempDir()})
	if len(got) != 1 || got[0].Status != "identity_blocked" || got[0].Provider != "" || got[0].Harness != "" || len(got[0].Argv) != 0 {
		t.Fatalf("missing journal harness must not cross-launch: %+v", got)
	}
}

func TestWriteReceiptPersistsHarnessIdentity(t *testing.T) {
	req := Request{ThreadID: "0198ec3d-6d66-7c82-a700-cedf64660a44", CWD: `C:\work\fak`, Provider: ProviderCodex, Harness: ProviderCodex, HarnessSource: "session_registration", Category: CategorySubstantive, Status: "candidate", Argv: []string{"fak", "guard", "--", "codex"}, ReceiptPath: filepath.Join(t.TempDir(), "receipt.json")}
	if wrote, err := WriteReceipt(req, time.Now()); err != nil || !wrote {
		t.Fatalf("wrote=%v err=%v", wrote, err)
	}
	b, err := os.ReadFile(req.ReceiptPath)
	if err != nil {
		t.Fatal(err)
	}
	var got Receipt
	if err := json.Unmarshal(b, &got); err != nil {
		t.Fatal(err)
	}
	if got.Harness != ProviderCodex || got.HarnessSource != "session_registration" {
		t.Fatalf("receipt identity=%+v", got)
	}
}
