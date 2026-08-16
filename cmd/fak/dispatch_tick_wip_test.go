package main

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
	"github.com/anthony-chaudhary/fak/internal/loopmgr"
)

func TestDispatchPreflightWIPReadsDurableStartedMinusEnded(t *testing.T) {
	root := t.TempDir()
	ledger := filepath.Join(root, defaultLoopLedger())
	t.Setenv(dispatchtick.WIPLimitEnv, "2")
	now := time.Now()
	events := []loopmgr.Event{
		{LoopID: "a", RunID: "a1", Kind: loopmgr.EventAdmit, Status: loopmgr.StatusAdmitted},
		{LoopID: "a", RunID: "a1", Kind: loopmgr.EventStart, Status: loopmgr.StatusRunning},
		{LoopID: "b", RunID: "b1", Kind: loopmgr.EventAdmit, Status: loopmgr.StatusAdmitted},
		{LoopID: "b", RunID: "b1", Kind: loopmgr.EventStart, Status: loopmgr.StatusRunning},
		{LoopID: "b", RunID: "b1", Kind: loopmgr.EventEnd, Status: loopmgr.StatusWitnessedDone},
		{LoopID: "q", RunID: "q1", Kind: loopmgr.EventAdmit, Status: loopmgr.StatusAdmitted},
	}
	for i, ev := range events {
		_, err := loopmgr.Append(ledger, ev, loopmgr.WithClock(func() time.Time { return now.Add(time.Duration(i) * time.Second) }))
		if err != nil {
			t.Fatal(err)
		}
	}
	got := dispatchPreflightWIP(root)
	if !got.Measured || got.Started != 1 || got.Inventory != 1 || got.Limit != 2 {
		t.Fatalf("census = %+v", got)
	}
}

func TestDispatchPreflightWIPAbstainsWithoutValidLimitOrLedger(t *testing.T) {
	root := t.TempDir()
	for _, raw := range []string{"", "0", "bad"} {
		t.Run("limit_"+raw, func(t *testing.T) {
			t.Setenv(dispatchtick.WIPLimitEnv, raw)
			if got := dispatchPreflightWIP(root); got.Binding() {
				t.Fatalf("binding census: %+v", got)
			}
		})
	}
	t.Setenv(dispatchtick.WIPLimitEnv, "3")
	got := dispatchPreflightWIP(root)
	if got.Measured || got.Limit != 3 {
		t.Fatalf("missing ledger census = %+v", got)
	}
	ledger := filepath.Join(root, defaultLoopLedger())
	if err := os.MkdirAll(filepath.Dir(ledger), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(ledger, []byte("broken\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	got = dispatchPreflightWIP(root)
	if got.Measured {
		t.Fatalf("malformed ledger should abstain: %+v", got)
	}
}
