package gardenbudget

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"
)

type fakeClock struct{ t time.Time }

func (c *fakeClock) now() time.Time      { return c.t }
func (c *fakeClock) add(d time.Duration) { c.t = c.t.Add(d) }

var testPhases = []Phase{"leases", "lock-files", "intents", "growth-logs", "sentinel-fold"}

func TestExecuteUnboundedRunsEveryPhaseAndCheckpoints(t *testing.T) {
	var got, checkpoints []Phase
	res := Execute(testPhases, Cursor{Stage: "act", Ticks: 1}, Options{
		Checkpoint: func(c Cursor) error {
			checkpoints = append(checkpoints, c.Next)
			return nil
		},
	}, func(p Phase) error {
		got = append(got, p)
		return nil
	})
	if len(got) != len(testPhases) {
		t.Fatalf("ran %v, want %v", got, testPhases)
	}
	for i := range testPhases {
		if got[i] != testPhases[i] {
			t.Fatalf("ran %v, want %v", got, testPhases)
		}
	}
	wantCheckpoints := []Phase{"lock-files", "intents", "growth-logs", "sentinel-fold", ""}
	for i := range wantCheckpoints {
		if checkpoints[i] != wantCheckpoints[i] {
			t.Fatalf("checkpoints = %v, want %v", checkpoints, wantCheckpoints)
		}
	}
	if !res.Complete || res.Exhausted || res.Next.Next != "" {
		t.Fatalf("full pass = %+v, want complete with empty next", res)
	}
}

func TestExecuteNeverStartsWorkAfterBudgetSpent(t *testing.T) {
	clk := &fakeClock{t: time.Unix(1_700_000_000, 0)}
	var got []Phase
	res := Execute(testPhases, Cursor{Next: "intents"}, Options{
		Budget: time.Second,
		Start:  clk.now().Add(-time.Hour),
		Now:    clk.now,
	}, func(p Phase) error {
		got = append(got, p)
		return nil
	})
	if len(got) != 0 {
		t.Fatalf("expired global budget started %v", got)
	}
	if !res.Exhausted || res.Next.Next != "intents" {
		t.Fatalf("expired result = %+v, want retry at intents", res)
	}
	want := []Phase{"intents", "growth-logs", "sentinel-fold"}
	if len(res.Deferred) != len(want) {
		t.Fatalf("deferred = %v, want %v", res.Deferred, want)
	}
}

func TestExecuteResumesSuffixWithoutReplayingCompletedPrefix(t *testing.T) {
	var got []Phase
	res := Execute(testPhases, Cursor{Next: "growth-logs"}, Options{}, func(p Phase) error {
		got = append(got, p)
		return nil
	})
	want := []Phase{"growth-logs", "sentinel-fold"}
	if len(got) != len(want) || got[0] != want[0] || got[1] != want[1] {
		t.Fatalf("ran %v, want %v", got, want)
	}
	if !res.Complete {
		t.Fatalf("resumed suffix did not complete: %+v", res)
	}
}

func TestExecuteCheckpointFailureStopsBeforeMoreWork(t *testing.T) {
	var got []Phase
	res := Execute(testPhases, Cursor{}, Options{
		Checkpoint: func(Cursor) error { return errors.New("disk full") },
	}, func(p Phase) error {
		got = append(got, p)
		return nil
	})
	if len(got) != 1 || got[0] != "leases" {
		t.Fatalf("checkpoint failure ran %v, want only leases", got)
	}
	if res.CheckpointError == "" || len(res.Deferred) != len(testPhases)-1 {
		t.Fatalf("result = %+v, want typed checkpoint failure and deferred suffix", res)
	}
}

func TestExecutePhaseErrorAdvancesToAvoidStrandingBestEffortSweeps(t *testing.T) {
	var got []Phase
	res := Execute(testPhases, Cursor{}, Options{}, func(p Phase) error {
		got = append(got, p)
		if p == "lock-files" {
			return errors.New("boom")
		}
		return nil
	})
	if len(got) != len(testPhases) || res.Errors() != 1 || !res.Complete {
		t.Fatalf("error pass = %+v ran=%v", res, got)
	}
}

func TestStampCarriesPayloadAndCountsOneInvocation(t *testing.T) {
	now := time.Unix(1_700_000_000, 0)
	payload := json.RawMessage(`{"members":2}`)
	got := Stamp(Cursor{Payload: payload, Ticks: 7}, "collect", now)
	if got.Schema != CursorSchema || got.Stage != "collect" || got.Ticks != 8 || got.UpdatedUnix != now.Unix() {
		t.Fatalf("Stamp = %+v", got)
	}
	if string(got.Payload) != string(payload) {
		t.Fatalf("payload changed: %s", got.Payload)
	}
}

func TestRemaining(t *testing.T) {
	now := time.Unix(1_700_000_010, 0)
	start := now.Add(-4 * time.Second)
	if got := Remaining(10*time.Second, start, func() time.Time { return now }); got != 6*time.Second {
		t.Fatalf("Remaining = %s, want 6s", got)
	}
	if got := Remaining(time.Second, start, func() time.Time { return now }); got != 0 {
		t.Fatalf("expired Remaining = %s, want 0", got)
	}
	if got := Remaining(0, start, func() time.Time { return now }); got <= 24*time.Hour {
		t.Fatalf("unbounded Remaining = %s", got)
	}
}

func TestCursorRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "nested", "tick-cursor.json")
	want := Cursor{
		Schema: CursorSchema, Stage: "act", Next: "intents", Ticks: 7,
		UpdatedUnix: 1_700_000_000, Payload: json.RawMessage(`{"results":[1]}`),
	}
	if err := SaveCursor(path, want); err != nil {
		t.Fatalf("SaveCursor: %v", err)
	}
	got, err := LoadCursor(path)
	if err != nil {
		t.Fatalf("LoadCursor: %v", err)
	}
	if got.Schema != want.Schema || got.Stage != want.Stage || got.Next != want.Next ||
		got.Ticks != want.Ticks || got.UpdatedUnix != want.UpdatedUnix ||
		string(got.Payload) != string(want.Payload) {
		t.Fatalf("round trip = %+v, want %+v", got, want)
	}
	want.Ticks = 8
	if err := SaveCursor(path, want); err != nil {
		t.Fatalf("SaveCursor overwrite: %v", err)
	}
	if got, _ = LoadCursor(path); got.Ticks != 8 {
		t.Fatalf("overwrite left Ticks=%d", got.Ticks)
	}
	ents, err := os.ReadDir(filepath.Dir(path))
	if err != nil {
		t.Fatal(err)
	}
	if len(ents) != 1 {
		t.Fatalf("checkpoint dir holds %d entries, want 1", len(ents))
	}
}

func TestLoadCursorFailsOpen(t *testing.T) {
	dir := t.TempDir()
	if c, err := LoadCursor(filepath.Join(dir, "absent.json")); err != nil || c.Next != "" {
		t.Fatalf("missing cursor = (%+v, %v)", c, err)
	}
	corrupt := filepath.Join(dir, "corrupt.json")
	if err := os.WriteFile(corrupt, []byte("{not json"), 0o644); err != nil {
		t.Fatal(err)
	}
	if c, err := LoadCursor(corrupt); err == nil || c.Next != "" {
		t.Fatalf("corrupt cursor = (%+v, %v)", c, err)
	}
	old := filepath.Join(dir, "old.json")
	if err := os.WriteFile(old, []byte(`{"schema":"fak.garden-tick-cursor.v1","next":"intents"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if c, err := LoadCursor(old); err != nil || c.Next != "" {
		t.Fatalf("old cursor = (%+v, %v)", c, err)
	}
}
