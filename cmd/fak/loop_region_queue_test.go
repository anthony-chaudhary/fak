package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/leasequeue"
	"github.com/anthony-chaudhary/fak/internal/leaseref"
	"github.com/anthony-chaudhary/fak/internal/regionadmit"
)

// loopRegionQueueRepo is a bare repo-shaped dir: the queue only needs a git dir to live beside.
func loopRegionQueueRepo(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	if err := os.MkdirAll(filepath.Join(root, ".git"), 0o755); err != nil {
		t.Fatalf("mkdir .git: %v", err)
	}
	return root
}

func loopRegionQueueTax() regionadmit.Taxonomy {
	return regionadmit.Taxonomy{
		Exclusive: map[string]bool{},
		Trees:     map[string][]string{"gateway": {"internal/gateway/**"}},
	}
}

func loopRegionQueueHeldLease(now time.Time) []leaseref.Record {
	return []leaseref.Record{{
		ID: "h1", Holder: "peer", TreeGlobs: []string{"internal/gateway/**"},
		AcquiredAt: now.Unix(), TTLSeconds: 600,
	}}
}

// The wiring's whole point: a refusal MINTS a place, a second waiter is told it is second, and a
// returning waiter KEEPS the place it earned instead of re-racing from scratch.
func TestLoopRegionEnqueueMintsAndPreservesAPlaceInLine(t *testing.T) {
	root := loopRegionQueueRepo(t)
	tax := loopRegionQueueTax()
	t0 := time.Unix(1_700_000_000, 0)
	live := loopRegionQueueHeldLease(t0)

	early := regionadmit.Request{Actor: "session:early", Lane: "gateway"}
	late := regionadmit.Request{Actor: "session:late", Lane: "gateway"}

	first := loopRegionEnqueue(root, early, tax, live, leasequeue.ClassLoop, t0)
	if first == nil {
		t.Fatal("enqueue returned no report; the refusal evaporated")
	}
	if first.Entry.Place != 1 || first.Depth != 1 {
		t.Fatalf("first waiter place=%d depth=%d, want 1/1", first.Entry.Place, first.Depth)
	}
	if first.Entry.Blocker == nil || first.Entry.Blocker.ID != "h1" {
		t.Fatalf("blocker = %+v, want the live lease h1", first.Entry.Blocker)
	}
	if first.Entry.BlockerKind != leasequeue.BlockedByLease {
		t.Errorf("blocker kind = %q, want %q", first.Entry.BlockerKind, leasequeue.BlockedByLease)
	}
	// The holder declared a TTL, so the ETA is real evidence rather than a guess.
	if !first.Entry.ETAKnown || first.Entry.ETASeconds != 600 {
		t.Errorf("eta known=%v seconds=%d, want true/600", first.Entry.ETAKnown, first.Entry.ETASeconds)
	}

	half := t0.Add(5 * time.Minute)
	second := loopRegionEnqueue(root, late, tax, live, leasequeue.ClassLoop, half)
	if second == nil {
		t.Fatal("second enqueue returned no report")
	}
	if second.Entry.Place != 2 || second.Depth != 2 {
		t.Fatalf("late waiter place=%d depth=%d, want 2/2", second.Entry.Place, second.Depth)
	}

	// The early waiter polls again. It must keep place 1 -- if the retry reset its clock it
	// would fall behind the waiter that arrived five minutes after it.
	back := loopRegionEnqueue(root, early, tax, live, leasequeue.ClassLoop, half.Add(time.Minute))
	if back == nil {
		t.Fatal("re-enqueue returned no report")
	}
	if back.Entry.ID != first.Entry.ID {
		t.Fatalf("retry minted a new ticket %q (was %q)", back.Entry.ID, first.Entry.ID)
	}
	if back.Entry.Place != 1 {
		t.Errorf("returning waiter place = %d, want 1 (its earned place)", back.Entry.Place)
	}
	if back.Depth != 2 {
		t.Errorf("depth = %d, want 2 (a retry must not stack a second ticket)", back.Depth)
	}
	if back.Entry.WaitSeconds != 360 {
		t.Errorf("wait = %ds, want 360 (measured from the ORIGINAL arrival)", back.Entry.WaitSeconds)
	}
	if back.Entry.Parks != 2 {
		t.Errorf("parks = %d after two refusals, want 2", back.Entry.Parks)
	}
}

// An admitted caller gives up its place, so it stops blocking the waiters behind it.
func TestLoopRegionDequeueGivesUpThePlace(t *testing.T) {
	root := loopRegionQueueRepo(t)
	tax := loopRegionQueueTax()
	now := time.Unix(1_700_000_000, 0)
	req := regionadmit.Request{Actor: "session:me", Lane: "gateway"}

	if r := loopRegionEnqueue(root, req, tax, loopRegionQueueHeldLease(now), "", now); r == nil {
		t.Fatal("enqueue returned no report")
	}
	loopRegionDequeue(root, req, tax)

	store, err := leasequeue.OpenStore(root)
	if err != nil {
		t.Fatalf("open store: %v", err)
	}
	live, err := store.Live(now)
	if err != nil {
		t.Fatalf("live: %v", err)
	}
	if len(live) != 0 {
		t.Fatalf("tickets after dequeue = %d, want 0", len(live))
	}
}

// The queue is best-effort and OFF the decision: a root with no git dir yields no report and no
// panic, which is what keeps an unwritable queue from ever changing a verdict.
func TestLoopRegionEnqueueOnANonRepoIsSilent(t *testing.T) {
	root := filepath.Join(t.TempDir(), "not-a-repo")
	req := regionadmit.Request{Actor: "session:me", Lane: "gateway"}
	if r := loopRegionEnqueue(root, req, loopRegionQueueTax(), nil, "", time.Unix(1, 0)); r != nil {
		t.Fatalf("enqueue on a non-repo returned %+v, want nil", r)
	}
	loopRegionDequeue(root, req, loopRegionQueueTax()) // must not panic
}

// A nil report renders to nothing, so the refusal output is byte-identical to the pre-queue verb
// whenever the queue is unavailable or opted out with --no-queue.
func TestNilQueueReportRendersNothing(t *testing.T) {
	var r *loopRegionQueueReport
	if got := r.line(); got != "" {
		t.Errorf("nil report line = %q, want empty", got)
	}
	if got := r.payload(); got != nil {
		t.Errorf("nil report payload = %v, want nil", got)
	}
}

// The expiry projection is the ETA's only evidence: a lease with no TTL declares no expiry, so no
// ETA is invented for it, and a renewed lease measures from the RENEWAL.
func TestLoopRegionHoldersCarryExpiryOnlyWhenDeclared(t *testing.T) {
	got := loopRegionHolders([]leaseref.Record{
		{ID: "ttl", AcquiredAt: 1000, TTLSeconds: 600},
		{ID: "renewed", AcquiredAt: 1000, RenewedAt: 1500, TTLSeconds: 600},
		{ID: "nottl", AcquiredAt: 1000},
		{ID: "noclock", TTLSeconds: 600},
	})
	if len(got) != 4 {
		t.Fatalf("holders = %d, want 4", len(got))
	}
	want := map[string]int64{"ttl": 1600, "renewed": 2100, "nottl": 0, "noclock": 0}
	for _, h := range got {
		if h.ExpiresUnix != want[h.Lease.ID] {
			t.Errorf("%s expires = %d, want %d", h.Lease.ID, h.ExpiresUnix, want[h.Lease.ID])
		}
	}
}

// The human line names the place, the blocker and when to ask again -- the four facts a bare
// REFUSE line never carried.
func TestQueueReportLineNamesPlaceBlockerAndRetry(t *testing.T) {
	root := loopRegionQueueRepo(t)
	now := time.Unix(1_700_000_000, 0)
	r := loopRegionEnqueue(root, regionadmit.Request{Actor: "session:me", Lane: "gateway"},
		loopRegionQueueTax(), loopRegionQueueHeldLease(now), leasequeue.ClassInteractive, now)
	if r == nil {
		t.Fatal("enqueue returned no report")
	}
	line := r.line()
	for _, want := range []string{"QUEUED ticket ", "place 1 of 1", "blocked by lease h1", "(holder peer)", "expires in 10m0s", "retry after "} {
		if !strings.Contains(line, want) {
			t.Errorf("queue line %q missing %q", line, want)
		}
	}

	p := r.payload()
	if p["schema"] != leasequeue.Schema {
		t.Errorf("payload schema = %v, want %q", p["schema"], leasequeue.Schema)
	}
	if p["place"] != 1 || p["depth"] != 1 {
		t.Errorf("payload place=%v depth=%v, want 1/1", p["place"], p["depth"])
	}
	if p["eta_known"] != true || p["eta_seconds"] != int64(600) {
		t.Errorf("payload eta_known=%v eta_seconds=%v, want true/600", p["eta_known"], p["eta_seconds"])
	}
	if p["class"] != leasequeue.ClassInteractive {
		t.Errorf("payload class = %v, want %q", p["class"], leasequeue.ClassInteractive)
	}
	blocker, ok := p["blocker"].(map[string]any)
	if !ok || blocker["id"] != "h1" || blocker["kind"] != leasequeue.BlockedByLease {
		t.Errorf("payload blocker = %v", p["blocker"])
	}
	if dir, _ := p["dir"].(string); !strings.Contains(dir, filepath.Join(".git", "fak", "leasequeue")) {
		t.Errorf("payload dir = %q, want the queue journal path", dir)
	}
}
