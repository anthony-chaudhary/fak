package main

import (
	"bytes"
	"reflect"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/sessionrecovery"
)

type captureLauncher struct{ got []sessionrecovery.Request }

func (c *captureLauncher) Launch(r sessionrecovery.Request) error {
	c.got = append(c.got, r)
	return nil
}

func TestSessionRecoverPromptAndCWD(t *testing.T) {
	oldInv, oldLaunch, oldSleep := recoveryInventory, recoveryLaunch, recoverySleep
	defer func() { recoveryInventory, recoveryLaunch, recoverySleep = oldInv, oldLaunch, oldSleep }()
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
	want := []string{"codex", "resume", "t1", "continue this exact task"}
	if len(cap.got) != 1 || !reflect.DeepEqual(cap.got[0].Argv, want) || cap.got[0].CWD != `C:\work\fak` {
		t.Fatalf("launch=%+v", cap.got)
	}
}
