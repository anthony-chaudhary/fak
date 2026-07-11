package slackoutbox

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/slackwire"
)

// testOutboxAt returns an outbox whose clock is a mutable variable, so a test can drain at one
// instant and reap at a later one. Mutate *clk to advance time.
func testOutboxAt(t *testing.T, start time.Time) (*Outbox, *time.Time) {
	t.Helper()
	o := testOutbox(t)
	cur := start
	o.now = func() time.Time { return cur }
	return o, &cur
}

// ephemeralSet builds the reaper's channel-default predicate over the given channel ids.
func ephemeralSet(chans ...string) func(string) bool {
	set := map[string]bool{}
	for _, c := range chans {
		set[c] = true
	}
	return func(c string) bool { return set[c] }
}

// drainOnce runs one drain and fails on error — the setup step reap tests share.
func drainOnce(t *testing.T, o *Outbox, w Wire) *DrainReport {
	t.Helper()
	rep, err := o.Drain(context.Background(), w, DrainOpts{})
	if err != nil {
		t.Fatalf("drain: %v", err)
	}
	return rep
}

func stateOf(t *testing.T, o *Outbox, nonce string) rowState {
	t.Helper()
	snap, err := o.Load()
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	return snap.state(nonce)
}

func TestReapDeletesEphemeralChannelMessagePastTTL(t *testing.T) {
	start := time.Date(2026, 7, 10, 12, 0, 0, 0, time.UTC)
	o, clk := testOutboxAt(t, start)
	w := newFakeWire()

	nonce, err := o.Enqueue(Row{Channel: "Cbridge", Text: "run started", Source: "runcard"})
	if err != nil {
		t.Fatal(err)
	}
	drainOnce(t, o, w) // posts at `start`; ts "1.0"

	*clk = start.Add(31 * time.Minute) // idle past the 30m default
	rep, err := o.Reap(context.Background(), w, ReapOpts{Ephemeral: ephemeralSet("Cbridge")})
	if err != nil {
		t.Fatal(err)
	}
	if rep.Eligible != 1 || rep.Deleted != 1 || rep.Gone != 0 || rep.Failed != 0 {
		t.Fatalf("report = %+v, want 1 eligible/deleted", rep)
	}
	if len(w.deletes) != 1 || w.deletes[0] != "Cbridge/1.0" {
		t.Fatalf("deletes = %v, want [Cbridge/1.0]", w.deletes)
	}
	if s := stateOf(t, o, nonce); s.State != stateReaped {
		t.Fatalf("state = %q, want reaped", s.State)
	}
}

func TestReapSkipsMessageWithinTTL(t *testing.T) {
	start := time.Date(2026, 7, 10, 12, 0, 0, 0, time.UTC)
	o, clk := testOutboxAt(t, start)
	w := newFakeWire()
	nonce, _ := o.Enqueue(Row{Channel: "Cbridge", Text: "hi"})
	drainOnce(t, o, w)

	*clk = start.Add(10 * time.Minute) // still fresh
	rep, err := o.Reap(context.Background(), w, ReapOpts{Ephemeral: ephemeralSet("Cbridge")})
	if err != nil {
		t.Fatal(err)
	}
	if rep.Eligible != 0 || len(w.deletes) != 0 {
		t.Fatalf("report=%+v deletes=%v, want nothing reaped", rep, w.deletes)
	}
	if s := stateOf(t, o, nonce); s.State != statePosted {
		t.Fatalf("state = %q, want still posted", s.State)
	}
}

func TestReapSkipsNonEphemeralChannel(t *testing.T) {
	start := time.Date(2026, 7, 10, 12, 0, 0, 0, time.UTC)
	o, clk := testOutboxAt(t, start)
	w := newFakeWire()
	o.Enqueue(Row{Channel: "Cother", Text: "keep me"})
	drainOnce(t, o, w)

	*clk = start.Add(3 * time.Hour) // very old, but the channel never opted in
	rep, err := o.Reap(context.Background(), w, ReapOpts{Ephemeral: ephemeralSet("Cbridge")})
	if err != nil {
		t.Fatal(err)
	}
	if rep.Eligible != 0 || len(w.deletes) != 0 {
		t.Fatalf("report=%+v deletes=%v, want nothing reaped in a non-ephemeral channel", rep, w.deletes)
	}
}

func TestReapHonorsPerRowDeleteAfter(t *testing.T) {
	start := time.Date(2026, 7, 10, 12, 0, 0, 0, time.UTC)
	o, clk := testOutboxAt(t, start)
	w := newFakeWire()
	// No ephemeral channel configured — the row opts itself in with a 60s TTL.
	nonce, _ := o.Enqueue(Row{Channel: "Cother", Text: "self-expiring", DeleteAfterS: 60})
	drainOnce(t, o, w)

	*clk = start.Add(2 * time.Minute)
	rep, err := o.Reap(context.Background(), w, ReapOpts{Ephemeral: nil})
	if err != nil {
		t.Fatal(err)
	}
	if rep.Deleted != 1 || len(w.deletes) != 1 {
		t.Fatalf("report=%+v deletes=%v, want the DeleteAfterS row reaped", rep, w.deletes)
	}
	if s := stateOf(t, o, nonce); s.State != stateReaped {
		t.Fatalf("state = %q, want reaped", s.State)
	}
}

// TestReapMeasuresFromLastActivity proves the idle window is measured from the newest activity
// (an in-place update), not the original post — a card still being edited is never culled.
func TestReapMeasuresFromLastActivity(t *testing.T) {
	start := time.Date(2026, 7, 10, 12, 0, 0, 0, time.UTC)
	o, clk := testOutboxAt(t, start)
	w := newFakeWire()

	postNonce, _ := o.Enqueue(Row{Channel: "Cbridge", Text: "card v1", Source: "runcard"})
	drainOnce(t, o, w) // posts at `start`; ts "1.0"

	*clk = start.Add(25 * time.Minute)
	if _, err := o.Enqueue(Row{Channel: "Cbridge", Text: "card v2", UpdateTS: "1.0", Source: "runcard"}); err != nil {
		t.Fatal(err)
	}
	drainOnce(t, o, w) // update recorded at start+25m — the new "last activity"

	// 40m after the original post, but only 15m since the last update: NOT reapable.
	*clk = start.Add(40 * time.Minute)
	rep, err := o.Reap(context.Background(), w, ReapOpts{Ephemeral: ephemeralSet("Cbridge")})
	if err != nil {
		t.Fatal(err)
	}
	if rep.Eligible != 0 || len(w.deletes) != 0 {
		t.Fatalf("report=%+v deletes=%v, want NOT reaped (last activity 15m ago)", rep, w.deletes)
	}
	if s := stateOf(t, o, postNonce); s.State != statePosted {
		t.Fatalf("post state = %q, want still posted", s.State)
	}

	// 31m after the last update (which landed at start+25m): now idle past the window.
	*clk = start.Add(56 * time.Minute)
	rep, err = o.Reap(context.Background(), w, ReapOpts{Ephemeral: ephemeralSet("Cbridge")})
	if err != nil {
		t.Fatal(err)
	}
	if rep.Deleted != 1 || len(w.deletes) != 1 || w.deletes[0] != "Cbridge/1.0" {
		t.Fatalf("report=%+v deletes=%v, want the card reaped once", rep, w.deletes)
	}
	// Both the post row and the update row share ts 1.0 — both fold to reaped.
	if s := stateOf(t, o, postNonce); s.State != stateReaped {
		t.Fatalf("post state = %q, want reaped", s.State)
	}
}

func TestReapAlreadyGoneIsRecordedReaped(t *testing.T) {
	start := time.Date(2026, 7, 10, 12, 0, 0, 0, time.UTC)
	o, clk := testOutboxAt(t, start)
	w := newFakeWire()
	w.deleteErrs = []error{&slackwire.APIError{Method: "chat.delete", Code: "message_not_found", Status: 200}}
	nonce, _ := o.Enqueue(Row{Channel: "Cbridge", Text: "already gone"})
	drainOnce(t, o, w)

	*clk = start.Add(31 * time.Minute)
	rep, err := o.Reap(context.Background(), w, ReapOpts{Ephemeral: ephemeralSet("Cbridge")})
	if err != nil {
		t.Fatal(err)
	}
	if rep.Gone != 1 || rep.Deleted != 0 || rep.Failed != 0 {
		t.Fatalf("report = %+v, want 1 gone", rep)
	}
	if s := stateOf(t, o, nonce); s.State != stateReaped {
		t.Fatalf("state = %q, want reaped (already-gone is success)", s.State)
	}
}

func TestReapTransientFailureLeavesPosted(t *testing.T) {
	start := time.Date(2026, 7, 10, 12, 0, 0, 0, time.UTC)
	o, clk := testOutboxAt(t, start)
	w := newFakeWire()
	w.deleteErrs = []error{fmt.Errorf("network reset")}
	nonce, _ := o.Enqueue(Row{Channel: "Cbridge", Text: "flaky delete"})
	drainOnce(t, o, w)

	*clk = start.Add(31 * time.Minute)
	rep, err := o.Reap(context.Background(), w, ReapOpts{Ephemeral: ephemeralSet("Cbridge")})
	if err != nil {
		t.Fatal(err)
	}
	if rep.Failed != 1 || rep.Deleted != 0 || rep.Gone != 0 {
		t.Fatalf("report = %+v, want 1 failed", rep)
	}
	if s := stateOf(t, o, nonce); s.State != statePosted {
		t.Fatalf("state = %q, want still posted (retry next pass)", s.State)
	}
}

func TestReapDryRunCountsWithoutDeleting(t *testing.T) {
	start := time.Date(2026, 7, 10, 12, 0, 0, 0, time.UTC)
	o, clk := testOutboxAt(t, start)
	w := newFakeWire()
	nonce, _ := o.Enqueue(Row{Channel: "Cbridge", Text: "dry"})
	drainOnce(t, o, w)

	*clk = start.Add(31 * time.Minute)
	rep, err := o.Reap(context.Background(), w, ReapOpts{Ephemeral: ephemeralSet("Cbridge"), DryRun: true})
	if err != nil {
		t.Fatal(err)
	}
	if !rep.DryRun || rep.Eligible != 1 || rep.Deleted != 0 {
		t.Fatalf("report = %+v, want 1 eligible, 0 deleted in dry run", rep)
	}
	if len(w.deletes) != 0 {
		t.Fatalf("dry run must not touch the wire: %v", w.deletes)
	}
	if s := stateOf(t, o, nonce); s.State != statePosted {
		t.Fatalf("dry run changed state to %q", s.State)
	}
}

// TestDrainAutoReapsAfterDelivery pins the auto path: a drain configured with an ephemeral
// channel reaps expired messages at its tail, no separate scheduler.
func TestDrainAutoReapsAfterDelivery(t *testing.T) {
	start := time.Date(2026, 7, 10, 12, 0, 0, 0, time.UTC)
	o, clk := testOutboxAt(t, start)
	w := newFakeWire()
	nonce, _ := o.Enqueue(Row{Channel: "Cbridge", Text: "old post"})
	drainOnce(t, o, w) // posts at `start`

	*clk = start.Add(31 * time.Minute)
	rep, err := o.Drain(context.Background(), w, DrainOpts{ReapEphemeral: ephemeralSet("Cbridge")})
	if err != nil {
		t.Fatal(err)
	}
	if rep.Reaped != 1 {
		t.Fatalf("drain report reaped = %d, want 1", rep.Reaped)
	}
	if len(w.deletes) != 1 || w.deletes[0] != "Cbridge/1.0" {
		t.Fatalf("deletes = %v, want the expired post deleted by the drain", w.deletes)
	}
	if s := stateOf(t, o, nonce); s.State != stateReaped {
		t.Fatalf("state = %q, want reaped", s.State)
	}
}

// TestStatusCountsReaped folds a reaped row into the reaped bucket, not pending — a reaped
// message is gone from the channel, not owed a delivery.
func TestStatusCountsReaped(t *testing.T) {
	start := time.Date(2026, 7, 10, 12, 0, 0, 0, time.UTC)
	o, clk := testOutboxAt(t, start)
	w := newFakeWire()
	o.Enqueue(Row{Channel: "Cbridge", Text: "x"})
	drainOnce(t, o, w)
	*clk = start.Add(31 * time.Minute)
	if _, err := o.Reap(context.Background(), w, ReapOpts{Ephemeral: ephemeralSet("Cbridge")}); err != nil {
		t.Fatal(err)
	}
	st, err := o.Status(*clk)
	if err != nil {
		t.Fatal(err)
	}
	if st.Reaped != 1 || st.Posted != 0 || st.Pending != 0 {
		t.Fatalf("status = %+v, want reaped 1, posted 0, pending 0", st)
	}
}
