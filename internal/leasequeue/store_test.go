package leasequeue

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

func testStore(t *testing.T) *Store {
	t.Helper()
	return NewStore(filepath.Join(t.TempDir(), "fak", "leasequeue"))
}

// THE store claim: a re-attempt refreshes ONE ticket and PRESERVES the enqueue clock. That
// preservation is what turns a re-race into a queue -- if the clock reset on every poll, every
// waiter would look like it just arrived and the order would be a lottery again.
func TestMintPreservesTheEnqueueClockAcrossRetries(t *testing.T) {
	s := testStore(t)
	t0 := time.Unix(1_000_000, 0)

	first, err := s.Mint(Ticket{Actor: "session:me", Lane: "gateway"}, t0)
	if err != nil {
		t.Fatalf("first mint: %v", err)
	}
	if first.ID == "" {
		t.Fatal("mint returned a ticket with no id")
	}
	if first.EnqueuedUnix != t0.Unix() || first.Parks != 1 {
		t.Fatalf("first mint enqueued=%d parks=%d, want %d/1", first.EnqueuedUnix, first.Parks, t0.Unix())
	}

	t1 := t0.Add(90 * time.Second)
	second, err := s.Mint(Ticket{Actor: "session:me", Lane: "gateway"}, t1)
	if err != nil {
		t.Fatalf("second mint: %v", err)
	}
	if second.ID != first.ID {
		t.Fatalf("retry minted a NEW ticket %q (was %q); the waiter lost its place", second.ID, first.ID)
	}
	if second.EnqueuedUnix != t0.Unix() {
		t.Errorf("retry reset the enqueue clock to %d, want the original %d", second.EnqueuedUnix, t0.Unix())
	}
	if second.Parks != 2 {
		t.Errorf("parks = %d after two refusals, want 2", second.Parks)
	}
	if second.RenewedUnix != t1.Unix() || second.LastParkUnix != t1.Unix() {
		t.Errorf("renew/park stamps = %d/%d, want %d", second.RenewedUnix, second.LastParkUnix, t1.Unix())
	}

	live, err := s.Live(t1)
	if err != nil {
		t.Fatalf("live: %v", err)
	}
	if len(live) != 1 {
		t.Fatalf("live tickets = %d, want exactly 1 (a retry must not stack tickets)", len(live))
	}
	if live[0].EnqueuedUnix != t0.Unix() {
		t.Errorf("persisted enqueue clock = %d, want %d", live[0].EnqueuedUnix, t0.Unix())
	}
}

// A waiter that stopped polling long enough to lapse rejoins as a FRESH arrival; a stale file
// must never grant unearned seniority.
func TestMintAfterLapseResetsSeniority(t *testing.T) {
	s := testStore(t)
	t0 := time.Unix(1_000_000, 0)
	if _, err := s.Mint(Ticket{Actor: "session:me", Lane: "gateway", TTLSeconds: 60}, t0); err != nil {
		t.Fatalf("mint: %v", err)
	}
	late := t0.Add(10 * time.Minute)
	again, err := s.Mint(Ticket{Actor: "session:me", Lane: "gateway", TTLSeconds: 60}, late)
	if err != nil {
		t.Fatalf("re-mint: %v", err)
	}
	if again.EnqueuedUnix != late.Unix() {
		t.Errorf("lapsed ticket kept its old clock %d, want a fresh %d", again.EnqueuedUnix, late.Unix())
	}
	if again.Parks != 1 {
		t.Errorf("parks = %d after a lapse, want a reset to 1", again.Parks)
	}
}

// A lapsed ticket is dropped on READ, so no reaper daemon has to exist for an abandoned waiter to
// stop reserving a region.
func TestLiveDropsLapsedTicketsWithoutAReaper(t *testing.T) {
	s := testStore(t)
	t0 := time.Unix(1_000_000, 0)
	if _, err := s.Mint(Ticket{Actor: "session:a", Lane: "gateway", TTLSeconds: 60}, t0); err != nil {
		t.Fatalf("mint a: %v", err)
	}
	if _, err := s.Mint(Ticket{Actor: "session:b", Lane: "model", TTLSeconds: 3600}, t0); err != nil {
		t.Fatalf("mint b: %v", err)
	}
	live, err := s.Live(t0.Add(10 * time.Minute))
	if err != nil {
		t.Fatalf("live: %v", err)
	}
	if len(live) != 1 {
		t.Fatalf("live tickets = %d, want 1 (the 60s ticket must have lapsed)", len(live))
	}
	if live[0].Lane != "model" {
		t.Errorf("surviving ticket lane = %q, want model", live[0].Lane)
	}
}

// Two DIFFERENT waiters occupy different files, so minting never needs a lock and a queue can
// hold more than one line at a time.
func TestMintKeepsDistinctWaitersDistinct(t *testing.T) {
	s := testStore(t)
	now := time.Unix(1_000_000, 0)
	a, err := s.Mint(Ticket{Actor: "session:a", Lane: "gateway"}, now)
	if err != nil {
		t.Fatalf("mint a: %v", err)
	}
	b, err := s.Mint(Ticket{Actor: "session:b", Lane: "gateway"}, now)
	if err != nil {
		t.Fatalf("mint b: %v", err)
	}
	if a.ID == b.ID {
		t.Fatalf("two actors collided on one ticket id %q", a.ID)
	}
	live, err := s.Live(now)
	if err != nil {
		t.Fatalf("live: %v", err)
	}
	if len(live) != 2 {
		t.Fatalf("live tickets = %d, want 2", len(live))
	}
}

// Dropping a ticket is what a caller does once it finally holds the region.
func TestDropRemovesTheTicketAndIsIdempotent(t *testing.T) {
	s := testStore(t)
	now := time.Unix(1_000_000, 0)
	tk, err := s.Mint(Ticket{Actor: "session:a", Lane: "gateway"}, now)
	if err != nil {
		t.Fatalf("mint: %v", err)
	}
	if err := s.Drop(tk.ID); err != nil {
		t.Fatalf("drop: %v", err)
	}
	live, err := s.Live(now)
	if err != nil {
		t.Fatalf("live: %v", err)
	}
	if len(live) != 0 {
		t.Fatalf("live tickets = %d after drop, want 0", len(live))
	}
	if err := s.Drop(tk.ID); err != nil {
		t.Errorf("dropping a missing ticket: %v, want nil", err)
	}
}

// The id is stable under glob order and separator style, so the same question always refreshes
// the same ticket rather than minting a second place in line for one waiter.
func TestTicketIDIsStableAcrossTreeOrderAndSeparators(t *testing.T) {
	a := TicketID("session:me", "gateway", []string{"internal/gateway/**", "docs/gateway/**"})
	b := TicketID("session:me", "gateway", []string{"docs\\gateway\\**", "internal\\gateway\\**"})
	if a != b {
		t.Errorf("ticket id changed with glob order/separators: %q vs %q", a, b)
	}
	if c := TicketID("session:other", "gateway", []string{"internal/gateway/**"}); c == a {
		t.Error("two different actors share one ticket id")
	}
	if d := TicketID("session:me", "model", []string{"internal/gateway/**", "docs/gateway/**"}); d == a {
		t.Error("two different lanes share one ticket id")
	}
}

// A ticket id is a hex digest, so it is already a safe basename. Anything else is refused rather
// than sanitized, so no caller can be talked into writing outside the journal.
func TestStoreRefusesATraversingTicketID(t *testing.T) {
	s := testStore(t)
	for _, bad := range []string{"..", "a/b", `a\b`, "a.b", "c:evil"} {
		if _, err := s.Mint(Ticket{ID: bad, Actor: "x"}, time.Unix(1, 0)); err == nil {
			t.Errorf("Mint accepted the unsafe ticket id %q", bad)
		}
		if err := s.Drop(bad); err == nil {
			t.Errorf("Drop accepted the unsafe ticket id %q", bad)
		}
	}
	// An EMPTY id is not unsafe, it is absent: Mint derives the stable id from the waiter's own
	// question, while Drop has nothing to name and refuses.
	if got, err := s.Mint(Ticket{Actor: "x", Lane: "gateway"}, time.Unix(1, 0)); err != nil || got.ID == "" {
		t.Errorf("Mint with no id = (%q, %v), want a derived id and no error", got.ID, err)
	}
	if err := s.Drop(""); err == nil {
		t.Error("Drop accepted an empty ticket id")
	}
}

// A queue that has never been written is empty, not an error: a first-ever refusal must not fail
// because no waiter has ever stood in line.
func TestLiveOnAnAbsentQueueIsEmptyNotAnError(t *testing.T) {
	live, err := testStore(t).Live(time.Unix(1_000_000, 0))
	if err != nil {
		t.Fatalf("live on an absent queue: %v", err)
	}
	if len(live) != 0 {
		t.Fatalf("live = %d, want 0", len(live))
	}
}

// One corrupt ticket must not blind the whole queue.
func TestLiveSkipsAnUnparsableTicket(t *testing.T) {
	s := testStore(t)
	now := time.Unix(1_000_000, 0)
	if _, err := s.Mint(Ticket{Actor: "session:a", Lane: "gateway"}, now); err != nil {
		t.Fatalf("mint: %v", err)
	}
	if err := os.WriteFile(filepath.Join(s.Dir(), "deadbeefdeadbeef.json"), []byte("{not json"), 0o644); err != nil {
		t.Fatalf("write corrupt ticket: %v", err)
	}
	live, err := s.Live(now)
	if err != nil {
		t.Fatalf("live: %v", err)
	}
	if len(live) != 1 {
		t.Fatalf("live = %d, want 1 (the good ticket survives a corrupt neighbour)", len(live))
	}
}

// QueueDir resolves both an ordinary .git directory and a worktree's .git POINTER FILE, so a
// per-worker worktree queues in its own git dir instead of silently nowhere.
func TestQueueDirResolvesBothGitDirShapes(t *testing.T) {
	plain := t.TempDir()
	if err := os.MkdirAll(filepath.Join(plain, ".git"), 0o755); err != nil {
		t.Fatalf("mkdir .git: %v", err)
	}
	got, err := QueueDir(plain)
	if err != nil {
		t.Fatalf("QueueDir(plain): %v", err)
	}
	if want := filepath.Join(plain, ".git", "fak", "leasequeue"); got != want {
		t.Errorf("plain queue dir = %q, want %q", got, want)
	}

	wt := t.TempDir()
	real := filepath.Join(t.TempDir(), "worktrees", "bb")
	if err := os.MkdirAll(real, 0o755); err != nil {
		t.Fatalf("mkdir real gitdir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(wt, ".git"), []byte("gitdir: "+real+"\n"), 0o644); err != nil {
		t.Fatalf("write .git pointer: %v", err)
	}
	got, err = QueueDir(wt)
	if err != nil {
		t.Fatalf("QueueDir(worktree): %v", err)
	}
	if want := filepath.Join(real, "fak", "leasequeue"); got != want {
		t.Errorf("worktree queue dir = %q, want %q", got, want)
	}

	if _, err := QueueDir(filepath.Join(t.TempDir(), "not-a-repo")); err == nil {
		t.Error("QueueDir on a non-repo returned no error")
	}
}

// End to end through the durable store: two refusals by two waiters, then a Plan over what was
// actually persisted. This is the whole fix in one test -- the refusals did not evaporate, and
// the earlier arrival holds first place.
func TestMintedTicketsFeedPlanInArrivalOrder(t *testing.T) {
	s := testStore(t)
	t0 := time.Unix(1_000_000, 0)
	if _, err := s.Mint(Ticket{Actor: "session:early", Lane: "gateway", Class: ClassLoop}, t0); err != nil {
		t.Fatalf("mint early: %v", err)
	}
	late := t0.Add(30 * time.Minute)
	if _, err := s.Mint(Ticket{Actor: "session:late", Lane: "gateway", Class: ClassLoop}, late); err != nil {
		t.Fatalf("mint late: %v", err)
	}
	tickets, err := s.Live(late)
	if err != nil {
		t.Fatalf("live: %v", err)
	}
	if len(tickets) != 2 {
		t.Fatalf("live tickets = %d, want 2", len(tickets))
	}

	res := Plan(tickets, []Holder{{Lease: gatewayHolder("h1")}}, testTax(), Params{NowUnix: late.Unix()})
	if res.Depth != 2 {
		t.Fatalf("depth = %d, want 2", res.Depth)
	}
	var firstPlace string
	for _, e := range res.Entries {
		if e.Place == 1 {
			firstPlace = e.Actor
		}
	}
	if firstPlace != "session:early" {
		t.Errorf("first in line = %q, want session:early", firstPlace)
	}
	if res.OldestWaitSeconds != 1800 {
		t.Errorf("oldest wait = %d, want 1800", res.OldestWaitSeconds)
	}
}
