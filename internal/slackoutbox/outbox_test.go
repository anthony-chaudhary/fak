package slackoutbox

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func testOutbox(t *testing.T) *Outbox {
	t.Helper()
	o, err := Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	return o
}

func TestEnqueueIsDurableAndStampsDefaults(t *testing.T) {
	o := testOutbox(t)
	nonce, err := o.Enqueue(Row{Channel: "C1", Text: "hello", Source: "test"})
	if err != nil {
		t.Fatal(err)
	}
	if nonce == "" {
		t.Fatal("nonce not generated")
	}
	// A fresh Outbox over the same dir sees the row: durability is the whole point.
	o2, err := Open(o.Dir())
	if err != nil {
		t.Fatal(err)
	}
	snap, err := o2.Load()
	if err != nil {
		t.Fatal(err)
	}
	if len(snap.Rows) != 1 || snap.Rows[0].Nonce != nonce || snap.Rows[0].EnqueuedAt == "" {
		t.Fatalf("row not durable: %+v", snap.Rows)
	}
	if snap.state(nonce).terminal() {
		t.Fatal("fresh row must be pending")
	}
}

func TestEnqueueValidates(t *testing.T) {
	o := testOutbox(t)
	if _, err := o.Enqueue(Row{Text: "no channel"}); err == nil {
		t.Fatal("missing channel accepted")
	}
	if _, err := o.Enqueue(Row{Channel: "C1"}); err == nil {
		t.Fatal("missing text accepted")
	}
	if _, err := o.Enqueue(Row{Channel: "C1", Text: "x", UpdateTS: "1.0", ThreadTS: "2.0"}); err == nil {
		t.Fatal("update_ts+thread_ts accepted")
	}
}

func TestEnqueueDefaultsCardKeyForUpdates(t *testing.T) {
	o := testOutbox(t)
	if _, err := o.Enqueue(Row{Channel: "C1", Text: "v2", UpdateTS: "9.9"}); err != nil {
		t.Fatal(err)
	}
	snap, err := o.Load()
	if err != nil {
		t.Fatal(err)
	}
	if snap.Rows[0].CardKey != "C1\x009.9" {
		t.Fatalf("card key not defaulted: %q", snap.Rows[0].CardKey)
	}
}

func TestLoadCountsCorruptLinesAndDuplicateNonces(t *testing.T) {
	o := testOutbox(t)
	n, err := o.Enqueue(Row{Channel: "C1", Text: "ok"})
	if err != nil {
		t.Fatal(err)
	}
	spool := filepath.Join(o.Dir(), "spool.jsonl")
	f, err := os.OpenFile(spool, os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		t.Fatal(err)
	}
	// one malformed line + one duplicate-nonce line
	if _, err := f.WriteString("{not json\n{\"nonce\":\"" + n + "\",\"channel\":\"C1\",\"text\":\"dupe\"}\n"); err != nil {
		t.Fatal(err)
	}
	f.Close()

	snap, err := o.Load()
	if err != nil {
		t.Fatal(err)
	}
	if len(snap.Rows) != 1 || snap.Corrupt != 2 {
		t.Fatalf("rows=%d corrupt=%d, want 1/2", len(snap.Rows), snap.Corrupt)
	}
}

func TestStatusFoldsStatesAndAges(t *testing.T) {
	o := testOutbox(t)
	base := time.Date(2026, 7, 2, 12, 0, 0, 0, time.UTC)
	o.now = func() time.Time { return base }

	nPost, _ := o.Enqueue(Row{Channel: "C1", Text: "posted one"})
	nDead, _ := o.Enqueue(Row{Channel: "C1", Text: "dead one", Source: "feeder"})
	if _, err := o.Enqueue(Row{Channel: "C1", Text: "still pending"}); err != nil {
		t.Fatal(err)
	}
	if err := o.appendState(transition{Nonce: nPost, State: statePosted, TS: "1.1"}); err != nil {
		t.Fatal(err)
	}
	if err := o.appendState(transition{Nonce: nDead, State: stateDead, Reason: "invalid_auth", Attempts: 5}); err != nil {
		t.Fatal(err)
	}
	if err := o.appendState(transition{State: stateDrainPass}); err != nil {
		t.Fatal(err)
	}

	st, err := o.Status(base.Add(90 * time.Second))
	if err != nil {
		t.Fatal(err)
	}
	if st.Pending != 1 || st.Posted != 1 || st.Dead != 1 {
		t.Fatalf("counts wrong: %+v", st)
	}
	if st.OldestPendingAgeS != 90 {
		t.Fatalf("oldest pending age = %d, want 90", st.OldestPendingAgeS)
	}
	if st.LastDrainAgeS != 90 {
		t.Fatalf("last drain age = %d, want 90", st.LastDrainAgeS)
	}
	if len(st.DeadRows) != 1 || st.DeadRows[0].Reason != "invalid_auth" || st.DeadRows[0].Source != "feeder" {
		t.Fatalf("dead rows wrong: %+v", st.DeadRows)
	}
}

func TestRetryReArmsOnlyDeadRows(t *testing.T) {
	o := testOutbox(t)
	nDead, _ := o.Enqueue(Row{Channel: "C1", Text: "was dead"})
	nPosted, _ := o.Enqueue(Row{Channel: "C1", Text: "was posted"})
	if err := o.appendState(transition{Nonce: nDead, State: stateDead, Reason: "x", Attempts: 5}); err != nil {
		t.Fatal(err)
	}
	if err := o.appendState(transition{Nonce: nPosted, State: statePosted, TS: "1.0"}); err != nil {
		t.Fatal(err)
	}

	armed, err := o.Retry("")
	if err != nil {
		t.Fatal(err)
	}
	if len(armed) != 1 || armed[0] != nDead {
		t.Fatalf("armed = %v, want [%s]", armed, nDead)
	}
	snap, _ := o.Load()
	if got := snap.state(nDead); got.State != statePending || got.Attempts != 0 {
		t.Fatalf("dead row not re-armed: %+v", got)
	}
	if snap.state(nPosted).State != statePosted {
		t.Fatal("posted row must stay posted")
	}
	if _, err := o.Retry(nPosted); err == nil || !strings.Contains(err.Error(), "not a dead row") {
		t.Fatalf("retrying a posted row must refuse: %v", err)
	}
}
