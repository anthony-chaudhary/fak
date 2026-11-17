package slackoutbox

import (
	"context"
	"testing"
	"time"
)

// quietFake is a fakeWire that also reports the global brake, so the HOLD path is
// exercised through the same interface a real *slackwire.Client satisfies.
type quietFake struct {
	*fakeWire
	quiet bool
}

func (q quietFake) Quiet() bool { return q.quiet }

// TestDrainHoldsEveryRowWhileQuiet is the core outbox contract for FAK_SLACK_QUIET: a
// quiet drain must attempt NOTHING. The witness is the fake's send log — a hold that
// still called PostMessageIdem would be a leak, and one that recorded a failure would
// burn the row's attempt budget.
func TestDrainHoldsEveryRowWhileQuiet(t *testing.T) {
	o := testOutbox(t)
	for _, txt := range []string{"one", "two", "three"} {
		if _, err := o.Enqueue(Row{Channel: "C1", Text: txt, Source: "test"}); err != nil {
			t.Fatal(err)
		}
	}
	f := newFakeWire()
	rep, err := o.Drain(context.Background(), quietFake{f, true}, drainOpts(nil))
	if err != nil {
		t.Fatalf("Drain err = %v, want nil (a quiet drain is not a failure)", err)
	}
	if len(f.posts) != 0 {
		t.Fatalf("quiet drain sent %v; the brake must hold every row", f.posts)
	}
	if rep.Held != 3 || rep.Posted != 0 || rep.Failed != 0 || rep.Dead != 0 {
		t.Errorf("report = %+v, want Held=3 Posted=0 Failed=0 Dead=0", rep)
	}
	if rep.Remaining != 3 {
		t.Errorf("Remaining = %d, want 3 (held rows are still owed)", rep.Remaining)
	}
}

// TestQuietDrainSpendsNoAttemptBudget is the reason the hold exists rather than letting
// each row fail into ErrQuiet: five quiet drains must not dead-letter the queue. The
// rows must come out byte-identical — no Attempts, no `sending` intent marker that the
// next drain would treat as an ambiguous post and probe for.
func TestQuietDrainSpendsNoAttemptBudget(t *testing.T) {
	o := testOutbox(t)
	nonce, err := o.Enqueue(Row{Channel: "C1", Text: "survives", Source: "test"})
	if err != nil {
		t.Fatal(err)
	}
	f := newFakeWire()
	// Ten quiet drains — twice the default MaxAttempts of 5.
	for i := 0; i < 10; i++ {
		if _, err := o.Drain(context.Background(), quietFake{f, true}, drainOpts(nil)); err != nil {
			t.Fatalf("quiet drain %d err = %v", i, err)
		}
	}
	snap, err := o.Load()
	if err != nil {
		t.Fatal(err)
	}
	st := snap.state(nonce)
	if st.State != statePending {
		t.Fatalf("state after 10 quiet drains = %q, want %q (a hold must not transition the row)", st.State, statePending)
	}
	if st.Attempts != 0 {
		t.Errorf("Attempts = %d after 10 quiet drains, want 0", st.Attempts)
	}

	// Disengage: the very next drain must deliver everything, untouched.
	rep, err := o.Drain(context.Background(), f, drainOpts(nil))
	if err != nil {
		t.Fatalf("post-quiet Drain err = %v", err)
	}
	if rep.Posted != 1 || rep.Held != 0 {
		t.Errorf("post-quiet report = %+v, want Posted=1 Held=0", rep)
	}
	if len(f.posts) != 1 || f.posts[0] != "C1/survives" {
		t.Errorf("posts = %v, want [C1/survives]", f.posts)
	}
}

// TestQuietDrainHoldsUpdatesToo covers the update half: a run card's in-place edit is a
// write, so it is held exactly like a post.
func TestQuietDrainHoldsUpdatesToo(t *testing.T) {
	o := testOutbox(t)
	if _, err := o.Enqueue(Row{Channel: "C1", Text: "card v1", CardKey: "k", Source: "test"}); err != nil {
		t.Fatal(err)
	}
	f := newFakeWire()
	if _, err := o.Drain(context.Background(), f, drainOpts(nil)); err != nil {
		t.Fatal(err)
	}
	if _, err := o.Enqueue(Row{Channel: "C1", Text: "card v2", CardKey: "k", UpdateTS: "1.0", Source: "test"}); err != nil {
		t.Fatal(err)
	}
	rep, err := o.Drain(context.Background(), quietFake{f, true}, drainOpts(nil))
	if err != nil {
		t.Fatal(err)
	}
	if rep.Held != 1 || rep.Updated != 0 {
		t.Errorf("report = %+v, want Held=1 Updated=0", rep)
	}
	if len(f.posts) != 1 {
		t.Errorf("posts = %v, want only the pre-quiet post", f.posts)
	}
}

// TestReapHoldsWhileQuiet keeps the brake total: chat.delete is a write, so a quiet
// fleet must not keep mutating channels by deleting its own history. The message stays
// posted and is reaped on the first pass after the switch is disengaged.
func TestReapHoldsWhileQuiet(t *testing.T) {
	o := testOutbox(t)
	nonce, err := o.Enqueue(Row{Channel: "C1", Text: "ephemeral", DeleteAfterS: 1, Source: "test"})
	if err != nil {
		t.Fatal(err)
	}
	f := newFakeWire()
	if _, err := o.Drain(context.Background(), f, drainOpts(nil)); err != nil {
		t.Fatal(err)
	}
	// Age the posted transition past its 1s TTL.
	base := time.Now()
	o.now = func() time.Time { return base.Add(time.Hour) }

	rep, err := o.Reap(context.Background(), quietFake{f, true}, ReapOpts{Now: base.Add(time.Hour), Sleep: drainOpts(nil).Sleep})
	if err != nil {
		t.Fatalf("Reap err = %v", err)
	}
	if len(f.deletes) != 0 {
		t.Fatalf("quiet reap deleted %v; the brake covers chat.delete", f.deletes)
	}
	if rep.Held != 1 || rep.Deleted != 0 {
		t.Errorf("report = %+v, want Held=1 Deleted=0", rep)
	}
	snap, err := o.Load()
	if err != nil {
		t.Fatal(err)
	}
	if st := snap.state(nonce); st.State != statePosted {
		t.Errorf("state = %q, want %q (a held reap must not record `reaped`)", st.State, statePosted)
	}

	// Disengage: the next pass reaps it.
	rep, err = o.Reap(context.Background(), f, ReapOpts{Now: base.Add(time.Hour), Sleep: drainOpts(nil).Sleep})
	if err != nil {
		t.Fatal(err)
	}
	if rep.Deleted != 1 || rep.Held != 0 {
		t.Errorf("post-quiet report = %+v, want Deleted=1 Held=0", rep)
	}
}

// TestUnbrakedWireIsNeverHeld pins the fail-safe direction: a transport that cannot
// report the brake (every existing fake, and any future one) drains exactly as before.
func TestUnbrakedWireIsNeverHeld(t *testing.T) {
	o := testOutbox(t)
	if _, err := o.Enqueue(Row{Channel: "C1", Text: "flows", Source: "test"}); err != nil {
		t.Fatal(err)
	}
	f := newFakeWire()
	rep, err := o.Drain(context.Background(), f, drainOpts(nil))
	if err != nil {
		t.Fatal(err)
	}
	if rep.Posted != 1 || rep.Held != 0 {
		t.Errorf("report = %+v, want Posted=1 Held=0 for a wire with no Quiet()", rep)
	}
}
