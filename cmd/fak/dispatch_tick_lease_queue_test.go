package main

// #5505 at the dispatch tick's refusal site: `acquireDispatchLaneLease` used to record
// LANE_LEASE_HELD and nothing else, so a refused tick evaporated into a lottery — the next tick
// re-raced from scratch and whoever polled first after a release won. These tests pin the waiter
// plane it now mints instead, and — the load-bearing half — pin that minting it CANNOT move the
// verdict or the field the caller branches on.

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/leasequeue"
	"github.com/anthony-chaudhary/fak/internal/leaseref"
)

// dispatchQueueTickets reads the waiter journal straight off disk — the durable state that
// distinguishes a real place in line from a log line about one.
func dispatchQueueTickets(t *testing.T, dir string) []leasequeue.Ticket {
	t.Helper()
	store, err := leasequeue.OpenStore(dir)
	if err != nil {
		t.Fatalf("open waiter journal: %v", err)
	}
	live, err := store.Live(time.Now())
	if err != nil {
		t.Fatalf("read waiter journal: %v", err)
	}
	return live
}

// dispatchQueueAge back-dates a minted ticket's enqueue clock on disk, so the ORDER under test is
// decided by wait time rather than by a sub-second tie-break between two tickets minted in the
// same instant. It rewrites the file the production path wrote, keyed by the id that path chose,
// so nothing here has to re-derive the ticket identity.
func dispatchQueueAge(t *testing.T, dir, id string, by time.Duration) {
	t.Helper()
	qdir, err := leasequeue.QueueDir(dir)
	if err != nil {
		t.Fatalf("queue dir: %v", err)
	}
	path := filepath.Join(qdir, id+".json")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read ticket %s: %v", id, err)
	}
	var ticket leasequeue.Ticket
	if err := json.Unmarshal(raw, &ticket); err != nil {
		t.Fatalf("decode ticket %s: %v", id, err)
	}
	shift := int64(by / time.Second)
	ticket.EnqueuedUnix -= shift
	ticket.LastParkUnix -= shift
	blob, err := json.Marshal(ticket)
	if err != nil {
		t.Fatalf("encode ticket %s: %v", id, err)
	}
	if err := os.WriteFile(path, blob, 0o644); err != nil {
		t.Fatalf("write ticket %s: %v", id, err)
	}
}

func dispatchQueuePayload(t *testing.T, lease map[string]any) map[string]any {
	t.Helper()
	q, ok := lease["queue"].(map[string]any)
	if !ok {
		t.Fatalf("refusal carries no queue report (the refusal evaporated), got %+v", lease)
	}
	return q
}

// TestDispatchLaneLeaseRefusalTakesAPlaceInLine is the success condition: a refused tick gets a
// DURABLE place — a ticket on disk — its retry keeps the position and the wait clock it earned,
// and a later arrival queues BEHIND it instead of re-racing it.
func TestDispatchLaneLeaseRefusalTakesAPlaceInLine(t *testing.T) {
	dir := initRegionTestRepo(t)
	acquireRegionTestLease(t, dir, "peer-gateway", "peer", []string{"internal/gateway/**"})

	// The early tick is refused and mints its place.
	t.Setenv("FAK_LEASE_OWNER", "owner-early")
	early := acquireDispatchLaneLease(dir, "resolve-gateway", "gateway", []string{"internal/gateway/**"}, 600, "")
	if refused, _ := early["refused"].(bool); !refused {
		t.Fatalf("a live peer lease on the same tree must refuse, got %+v", early)
	}
	q := dispatchQueuePayload(t, early)
	if q["place"] != 1 || q["depth"] != 1 {
		t.Fatalf("first waiter place=%v depth=%v, want 1/1", q["place"], q["depth"])
	}
	if q["schema"] != leasequeue.Schema {
		t.Errorf("queue schema = %v, want %q", q["schema"], leasequeue.Schema)
	}
	if q["class"] != leasequeue.ClassLoop {
		t.Errorf("queue class = %v, want %q (the tick IS the background driver)", q["class"], leasequeue.ClassLoop)
	}
	blocker, ok := q["blocker"].(map[string]any)
	if !ok || blocker["id"] != "peer-gateway" || blocker["kind"] != leasequeue.BlockedByLease {
		t.Errorf("queue blocker = %v, want the live peer lease", q["blocker"])
	}

	// The place is DURABLE: one ticket file exists under the repo's waiter journal, and its id is
	// the one the refusal reported. A log line alone would not survive this assertion.
	ticketID, _ := q["ticket"].(string)
	live := dispatchQueueTickets(t, dir)
	if len(live) != 1 {
		t.Fatalf("tickets on disk = %d, want 1", len(live))
	}
	if live[0].ID != ticketID || ticketID == "" {
		t.Fatalf("on-disk ticket %q does not match the reported ticket %q", live[0].ID, ticketID)
	}
	if live[0].Lane != "gateway" || live[0].Actor != "owner-early" {
		t.Errorf("ticket records lane=%q actor=%q, want gateway/owner-early", live[0].Lane, live[0].Actor)
	}

	// Age the early waiter ten minutes so the order under test is decided by the wait clock.
	dispatchQueueAge(t, dir, ticketID, 10*time.Minute)

	// A later arrival queues BEHIND it — this is the lottery closing.
	t.Setenv("FAK_LEASE_OWNER", "owner-late")
	late := acquireDispatchLaneLease(dir, "resolve-gateway-2", "gateway", []string{"internal/gateway/**"}, 600, "")
	if refused, _ := late["refused"].(bool); !refused {
		t.Fatalf("the later tick must also refuse, got %+v", late)
	}
	lateQ := dispatchQueuePayload(t, late)
	if lateQ["place"] != 2 || lateQ["depth"] != 2 {
		t.Fatalf("late waiter place=%v depth=%v, want 2/2 (it must not overtake the early waiter)", lateQ["place"], lateQ["depth"])
	}

	// The early tick polls again. It must KEEP place 1 and its wait clock: if the retry re-minted
	// from scratch the clock would collapse to ~0 and it would fall in behind the newcomer.
	t.Setenv("FAK_LEASE_OWNER", "owner-early")
	back := acquireDispatchLaneLease(dir, "resolve-gateway", "gateway", []string{"internal/gateway/**"}, 600, "")
	if refused, _ := back["refused"].(bool); !refused {
		t.Fatalf("the returning tick must still refuse, got %+v", back)
	}
	backQ := dispatchQueuePayload(t, back)
	if backQ["ticket"] != ticketID {
		t.Fatalf("retry minted a new ticket %v (was %q); the place was not preserved", backQ["ticket"], ticketID)
	}
	if backQ["place"] != 1 {
		t.Errorf("returning waiter place = %v, want 1 (the place it earned by waiting)", backQ["place"])
	}
	if wait, _ := backQ["wait_seconds"].(int64); wait < 600 {
		t.Errorf("wait = %ds, want >= 600 (measured from the ORIGINAL arrival, not this poll)", wait)
	}
	if backQ["parks"] != 2 {
		t.Errorf("parks = %v after two refusals, want 2", backQ["parks"])
	}
	if got := dispatchQueueTickets(t, dir); len(got) != 2 {
		t.Fatalf("tickets on disk = %d, want 2 (a retry must refresh one ticket, not stack another)", len(got))
	}
}

// TestDispatchLaneLeaseQueueNeverChangesTheVerdict is the CONTROL. The waiter plane may only ADD a
// report key: the admit/refuse verdict and the `refused` field the callers branch on (the
// LANE_LEASE_HELD verdict in dispatchTickLiveSpawn and dispatchTickHostEnroll) must be identical
// whether the queue works or is entirely unavailable. An unwritable queue degrades the payload and
// never the decision.
func TestDispatchLaneLeaseQueueNeverChangesTheVerdict(t *testing.T) {
	decisionKeys := []string{"acquired", "refused", "id", "holder", "reason", "rung", "detail", "lane", "lane_kind", "mode"}

	// Run A: a working queue.
	withQueue := initRegionTestRepo(t)
	acquireRegionTestLease(t, withQueue, "peer-gateway", "peer", []string{"internal/gateway/**"})
	t.Setenv("FAK_LEASE_OWNER", "owner-a")
	queued := acquireDispatchLaneLease(withQueue, "resolve-gateway", "gateway", []string{"internal/gateway/**"}, 600, "")

	// Run B: the SAME question against a repo whose waiter journal cannot be created — a plain
	// file sits where the queue directory would go, so every mint fails.
	broken := initRegionTestRepo(t)
	if err := os.WriteFile(filepath.Join(broken, ".git", "fak"), []byte("not a directory"), 0o644); err != nil {
		t.Fatalf("plant the unwritable queue: %v", err)
	}
	acquireRegionTestLease(t, broken, "peer-gateway", "peer", []string{"internal/gateway/**"})
	bare := acquireDispatchLaneLease(broken, "resolve-gateway", "gateway", []string{"internal/gateway/**"}, 600, "")

	if _, ok := bare["queue"]; ok {
		t.Fatalf("an unavailable queue must report nothing, got %+v", bare["queue"])
	}
	if _, ok := queued["queue"]; !ok {
		t.Fatalf("the working-queue run reported no ticket, so this control proves nothing: %+v", queued)
	}

	// The verdict is byte-identical across the two: refused stays refused, and every field the
	// exit path reads is the same. This is the invariant a queue must never break.
	for _, k := range decisionKeys {
		if queued[k] != bare[k] {
			t.Errorf("queue changed the decision field %q: with-queue=%v without-queue=%v", k, queued[k], bare[k])
		}
	}
	if refused, _ := queued["refused"].(bool); !refused {
		t.Fatal("queueing turned a refusal into something else")
	}
	if acquired, _ := queued["acquired"].(bool); acquired {
		t.Fatal("queueing admitted a caller the region decision refused")
	}
	if queued["reason"] != "COLLISION_RISK" {
		t.Errorf("reason = %v, want COLLISION_RISK (unchanged by the queue)", queued["reason"])
	}
	// "queue" is the ONLY key the plane adds; it never removes or renames a decision key.
	if len(queued) != len(bare)+1 {
		t.Errorf("with-queue keys = %d, without-queue = %d; want exactly one added key", len(queued), len(bare))
	}
	for k := range bare {
		if _, ok := queued[k]; !ok {
			t.Errorf("the queue removed decision key %q", k)
		}
	}

	// And an ADMIT is untouched: a free lane acquires with no ticket and no queue key, so the
	// plane cannot manufacture a wait where the region was available.
	free := acquireDispatchLaneLease(withQueue, "resolve-docs", "docs", []string{"docs/**"}, 600, "")
	if acquired, _ := free["acquired"].(bool); !acquired {
		t.Fatalf("a free lane must still acquire, got %+v", free)
	}
	if _, ok := free["queue"]; ok {
		t.Errorf("an admitted caller must carry no queue report, got %v", free["queue"])
	}
}

// TestDispatchLaneLeaseAdmitGivesUpThePlace pins the mirror half: once the tick finally holds the
// region it drops its ticket, so it stops reserving a place ahead of the peers behind it.
func TestDispatchLaneLeaseAdmitGivesUpThePlace(t *testing.T) {
	dir := initRegionTestRepo(t)
	acquireRegionTestLease(t, dir, "peer-gateway", "peer", []string{"internal/gateway/**"})
	t.Setenv("FAK_LEASE_OWNER", "owner-a")

	refused := acquireDispatchLaneLease(dir, "resolve-gateway", "gateway", []string{"internal/gateway/**"}, 600, "")
	if got, _ := refused["refused"].(bool); !got {
		t.Fatalf("the held region must refuse, got %+v", refused)
	}
	if got := dispatchQueueTickets(t, dir); len(got) != 1 {
		t.Fatalf("tickets after the refusal = %d, want 1", len(got))
	}

	if err := leaseref.NewInDir(dir).Release(context.Background(), "peer-gateway"); err != nil {
		t.Fatalf("release the blocking lease: %v", err)
	}
	admitted := acquireDispatchLaneLease(dir, "resolve-gateway", "gateway", []string{"internal/gateway/**"}, 600, "")
	if got, _ := admitted["acquired"].(bool); !got {
		t.Fatalf("the freed region must acquire, got %+v", admitted)
	}
	if got := dispatchQueueTickets(t, dir); len(got) != 0 {
		t.Fatalf("tickets after the admit = %d, want 0 (an admitted caller gives up its place)", len(got))
	}
}
