package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/sessionjournal"
	"github.com/anthony-chaudhary/fak/internal/sessionrecovery"
)

type captureLauncher struct{ got []sessionrecovery.Request }

func (c *captureLauncher) Launch(r sessionrecovery.Request) error {
	c.got = append(c.got, r)
	return nil
}

func TestSessionRecoverPromptAndCWD(t *testing.T) {
	oldInv, oldJournal, oldLaunch, oldSleep := recoveryInventory, recoveryJournalCrashes, recoveryLaunch, recoverySleep
	defer func() {
		recoveryInventory, recoveryJournalCrashes, recoveryLaunch, recoverySleep = oldInv, oldJournal, oldLaunch, oldSleep
	}()
	recoveryJournalCrashes = func(string, time.Time) ([]sessionjournal.Classified, error) { return nil, nil }
	calls := 0
	recoveryInventory = func(time.Duration) (sessionrecovery.InventoryReport, error) {
		calls++
		r := sessionrecovery.Session{Thread: &sessionrecovery.Thread{ID: "t1", Source: "interactive_tui"}, LatestTurn: &sessionrecovery.Turn{Status: "inProgress"}}
		if calls > 1 {
			r.ProcessTrees = []sessionrecovery.ProcessTree{{RootPID: 9}}
			r.GuardReceipt = &sessionrecovery.GuardReceipt{}
		}
		return sessionrecovery.InventoryReport{Sessions: []sessionrecovery.Session{r}}, nil
	}
	cap := &captureLauncher{}
	recoveryLaunch = cap
	recoverySleep = func(time.Duration) {}
	var out, er bytes.Buffer
	code := runSessionRecover(&out, &er, []string{"--apply", "--cwd", `C:\work\fak`, "--prompt", "continue this exact task", "--receipts", t.TempDir(), "--settle", "0"})
	if code != 0 {
		t.Fatalf("code=%d err=%s", code, er.String())
	}
	guardBin, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	want := []string{guardBin, "guard", "--", "codex", "resume", "t1", "continue this exact task"}
	if len(cap.got) != 1 || !reflect.DeepEqual(cap.got[0].Argv, want) || cap.got[0].CWD != `C:\work\fak` {
		t.Fatalf("launch=%+v", cap.got)
	}
}

func TestSessionRecoverUsesJournalRecordedCWD(t *testing.T) {
	oldInv := recoveryInventory
	defer func() { recoveryInventory = oldInv }()
	recoveryInventory = func(time.Duration) (sessionrecovery.InventoryReport, error) {
		return sessionrecovery.InventoryReport{}, nil
	}
	journalPath := filepath.Join(t.TempDir(), "session-journal.jsonl")
	if err := sessionjournal.Append(journalPath, sessionjournal.Event{
		Schema: sessionjournal.Schema, Kind: sessionjournal.KindOpen,
		ID: "journal-thread", CWD: `D:\repos\actual tree`, PID: 0,
		TS: "2000-01-01T00:00:00Z", Boot: "boot-old",
	}); err != nil {
		t.Fatal(err)
	}
	var out, er bytes.Buffer
	code := runSessionRecover(&out, &er, []string{"--limit", "1", "--journal-path", journalPath, "--receipts", t.TempDir(), "--prompt", "continue exactly"})
	if code != 0 {
		t.Fatalf("code=%d err=%s", code, er.String())
	}
	var got []sessionrecovery.Request
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	guardBin, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	wantArgv := []string{guardBin, "guard", "--", "codex", "resume", "journal-thread", "continue exactly"}
	if len(got) != 1 || got[0].CWD != `D:\repos\actual tree` || got[0].Source != "session_journal" || !reflect.DeepEqual(got[0].Argv, wantArgv) {
		t.Fatalf("requests=%+v", got)
	}
}
