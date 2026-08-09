package leasequeue

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/dispatchaging"
	"github.com/anthony-chaudhary/fak/internal/regionadmit"
	"github.com/anthony-chaudhary/fak/internal/seatpark"
)

// nowFixture is the clock every case reads. It is far enough above zero that a multi-hour
// enqueue stamp stays a positive unix time.
const nowFixture int64 = 1_000_000

// testTax is a two-lane workspace with disjoint canonical trees plus one exclusive lane.
func testTax() regionadmit.Taxonomy {
	return regionadmit.Taxonomy{
		Exclusive: map[string]bool{"release": true},
		Trees: map[string][]string{
			"gateway": {"internal/gateway/**"},
			"model":   {"internal/model/**"},
			"release": {"**/*"},
		},
	}
}

func gatewayHolder(id string) regionadmit.Lease {
	return regionadmit.Lease{ID: id, Holder: "peer", Lane: "gateway", Tree: []string{"internal/gateway/**"}}
}

// waiter is a background-loop waiter that arrived at `enqueued` and is STILL POLLING at
// nowFixture. The two clocks are distinct on purpose: EnqueuedUnix is the seniority the order
// reads, RenewedUnix is the liveness the lapse check reads. A long wait plus a fresh renewal is
// exactly the shape Store.Mint produces on a repeat refusal.
func waiter(id, actor, lane string, enqueued int64) Ticket {
	return Ticket{ID: id, Actor: actor, Lane: lane, Class: ClassLoop,
		EnqueuedUnix: enqueued, RenewedUnix: nowFixture, TTLSeconds: 1800}
}

// operator is the same, in the interactive priority class.
func operator(id, actor, lane string, enqueued int64) Ticket {
	t := waiter(id, actor, lane, enqueued)
	t.Class = ClassInteractive
	return t
}

func entryFor(t *testing.T, res Result, id string) Entry {
	t.Helper()
	e, ok := res.Find(id)
	if !ok {
		t.Fatalf("no entry for waiter %q in %+v", id, res.Entries)
	}
	return e
}

// The core claim: two waiters on one held lane are ordered by ARRIVAL, and that order does not
// depend on the order they happen to be presented in. Under the pre-queue behavior there is no
// order at all -- whoever polls first after a release wins -- so a stable arrival order is
// exactly the property the lottery lacks. The younger waiter's id sorts FIRST alphabetically, so
// an implementation that fell back to id order would flunk this.
func TestPlanOrdersBlockedWaitersByArrivalNotByInputOrder(t *testing.T) {
	const now = nowFixture
	older := waiter("zzzold", "session:four-hours", "gateway", now-4*3600)
	newer := waiter("aaanew", "session:200ms", "gateway", now-1)
	holders := []Holder{{Lease: gatewayHolder("h1")}}

	forward := Plan([]Ticket{older, newer}, holders, testTax(), Params{NowUnix: now})
	reversed := Plan([]Ticket{newer, older}, holders, testTax(), Params{NowUnix: now})

	for _, res := range []Result{forward, reversed} {
		if got := entryFor(t, res, "zzzold").Place; got != 1 {
			t.Errorf("four-hour waiter place = %d, want 1", got)
		}
		if got := entryFor(t, res, "aaanew").Place; got != 2 {
			t.Errorf("200ms waiter place = %d, want 2", got)
		}
		if res.Depth != 2 {
			t.Errorf("queue depth = %d, want 2", res.Depth)
		}
		if res.OldestWaitSeconds != 4*3600 {
			t.Errorf("oldest wait = %ds, want %d", res.OldestWaitSeconds, 4*3600)
		}
	}
	if forward.Entries[0].ID != reversed.Entries[0].ID {
		t.Errorf("rank order depends on input order: %q vs %q", forward.Entries[0].ID, reversed.Entries[0].ID)
	}
}

// A refused waiter gets the OBJECT the bare admission refusal never returned: its blocker, the
// closed refusal tokens, and a poll schedule that says when to ask again.
func TestPlanBlockedWaiterCarriesBlockerReasonAndPollSchedule(t *testing.T) {
	const now = nowFixture
	w := waiter("w1", "session:me", "gateway", now-60)
	w.Parks = 1
	w.LastParkUnix = now - 5

	res := Plan([]Ticket{w}, []Holder{{Lease: gatewayHolder("h1")}}, testTax(), Params{NowUnix: now})
	e := entryFor(t, res, "w1")

	if e.Grant {
		t.Fatalf("waiter granted against a live same-lane holder")
	}
	if e.Blocker == nil || e.Blocker.ID != "h1" {
		t.Fatalf("blocker = %+v, want the live lease h1", e.Blocker)
	}
	if e.BlockerKind != BlockedByLease {
		t.Errorf("blocker kind = %q, want %q", e.BlockerKind, BlockedByLease)
	}
	if e.Reason != regionadmit.ReasonCollisionRisk {
		t.Errorf("reason = %q, want %q", e.Reason, regionadmit.ReasonCollisionRisk)
	}
	if e.Rung != regionadmit.RungSameLane {
		t.Errorf("rung = %q, want %q", e.Rung, regionadmit.RungSameLane)
	}
	if e.Detail == "" {
		t.Error("refusal detail is empty; the waiter is told nothing about its blocker")
	}
	if e.WaitSeconds != 60 {
		t.Errorf("wait = %ds, want 60", e.WaitSeconds)
	}
	// Parked one park in: the documented 30s base window anchored on the last refusal.
	if e.Poll.Status != seatpark.StatusParked {
		t.Errorf("poll status = %q, want %q", e.Poll.Status, seatpark.StatusParked)
	}
	if want := (now - 5) + seatpark.DefaultBaseSeconds; e.Poll.NextRetryUnix != want {
		t.Errorf("next retry = %d, want %d", e.Poll.NextRetryUnix, want)
	}
}

// Conservative backfill: a waiter on a DISJOINT region is admitted past a blocked one (the whole
// point of backfill), while a waiter sharing the blocked region is placed BEHIND it rather than
// alongside it.
func TestPlanBackfillsDisjointRegionAndQueuesTheSharedOne(t *testing.T) {
	const now = nowFixture
	blocked := waiter("wold", "session:a", "gateway", now-3600)
	disjoint := waiter("wfree", "session:b", "model", now-60)
	sharer := waiter("wnew", "session:c", "gateway", now-30)

	res := Plan([]Ticket{blocked, disjoint, sharer}, []Holder{{Lease: gatewayHolder("h1")}}, testTax(), Params{NowUnix: now})

	if e := entryFor(t, res, "wfree"); !e.Grant {
		t.Errorf("disjoint-region waiter was not backfilled: %+v", e)
	}
	if e := entryFor(t, res, "wold"); e.Grant || e.Place != 1 {
		t.Errorf("oldest blocked waiter grant=%v place=%d, want grant=false place=1", e.Grant, e.Place)
	}
	// The younger sharer stands behind the older one, not beside it: that is the anti-lottery
	// guarantee. Its place counts the ONE waiter ahead of it that actually conflicts.
	if e := entryFor(t, res, "wnew"); e.Grant || e.Place != 2 {
		t.Errorf("younger same-lane waiter grant=%v place=%d, want grant=false place=2", e.Grant, e.Place)
	}
	if res.Depth != 2 {
		t.Errorf("depth = %d, want 2 (the disjoint waiter is granted, not queued)", res.Depth)
	}
	if len(res.Granted) != 1 || res.Granted[0] != "wfree" {
		t.Errorf("granted = %v, want [wfree]", res.Granted)
	}
}

// A waiter on a FREE region is granted, and the next waiter for that same region is told it is
// blocked BY A WAITER rather than by a lease -- the difference between "someone is working" and
// "someone is ahead of you in line". Without the queue both would simply re-race.
func TestPlanSecondWaiterOnFreeRegionIsBlockedByTheFirst(t *testing.T) {
	const now = nowFixture
	first := waiter("wfirst", "session:a", "gateway", now-3600)
	second := waiter("wsecond", "session:b", "gateway", now-60)

	res := Plan([]Ticket{second, first}, nil, testTax(), Params{NowUnix: now})

	if e := entryFor(t, res, "wfirst"); !e.Grant {
		t.Fatalf("first waiter on a free region was not granted: %+v", e)
	}
	e := entryFor(t, res, "wsecond")
	if e.Grant {
		t.Fatalf("both waiters granted the same lane -- the queue admitted a collision")
	}
	if e.BlockerKind != BlockedByWaiter {
		t.Errorf("blocker kind = %q, want %q", e.BlockerKind, BlockedByWaiter)
	}
	if e.Blocker == nil || e.Blocker.ID != "wfirst" {
		t.Fatalf("blocker = %+v, want the granted waiter wfirst", e.Blocker)
	}
	if e.Place != 1 {
		t.Errorf("place = %d, want 1 (nobody is blocked ahead of it)", e.Place)
	}
}

// The hard starvation deadline crosses the lease hop: a background loop that has waited past the
// deadline is served ahead of a fresh interactive operator, so no class can starve another.
func TestPlanStarvedLoopOvertakesFreshInteractive(t *testing.T) {
	const now = nowFixture
	loop := waiter("wloop", "loop:nightly", "gateway", now-7*3600)
	hot := operator("whot", "session:op", "gateway", now-5)

	p := Params{NowUnix: now, Aging: dispatchaging.DefaultParams(now)}
	res := Plan([]Ticket{hot, loop}, []Holder{{Lease: gatewayHolder("h1")}}, testTax(), p)

	starved := entryFor(t, res, "wloop")
	if starved.Standing != dispatchaging.StandingStarved {
		t.Fatalf("7h-waiting loop standing = %q, want %q", starved.Standing, dispatchaging.StandingStarved)
	}
	if starved.Place != 1 {
		t.Errorf("starved loop place = %d, want 1 (ahead of the fresh operator)", starved.Place)
	}
	if got := entryFor(t, res, "whot").Place; got != 2 {
		t.Errorf("fresh interactive place = %d, want 2 (behind the starved loop)", got)
	}
}

// With no starvation in play the hotter class leads, which is the class policy W3 asks for: an
// operator does not queue behind a background loop that arrived moments earlier.
func TestPlanInteractiveOutranksLoopWhenNeitherIsStarved(t *testing.T) {
	const now = nowFixture
	loop := waiter("wloop", "loop:nightly", "gateway", now-120)
	hot := operator("whot", "session:op", "gateway", now-5)

	res := Plan([]Ticket{loop, hot}, []Holder{{Lease: gatewayHolder("h1")}}, testTax(),
		Params{NowUnix: now, Aging: dispatchaging.DefaultParams(now)})

	if got := entryFor(t, res, "whot").Place; got != 1 {
		t.Errorf("interactive place = %d, want 1", got)
	}
	if got := entryFor(t, res, "wloop").Place; got != 2 {
		t.Errorf("loop place = %d, want 2", got)
	}
	if ClassWeight(ClassInteractive) <= ClassWeight(ClassLoop) {
		t.Errorf("interactive weight %d must exceed loop weight %d",
			ClassWeight(ClassInteractive), ClassWeight(ClassLoop))
	}
	if ClassWeight(ClassUnknown) != ClassWeight(ClassLoop) {
		t.Error("an unclassified waiter must rank as a loop, never as an operator")
	}
}

// An abandoned waiter is dropped from the line: it neither holds a place nor reserves a region.
// A waiter that STOPPED POLLING is exactly what lapses -- a long wait alone never does, which is
// the difference between seniority (EnqueuedUnix) and liveness (RenewedUnix).
func TestPlanAbandonsLapsedTickets(t *testing.T) {
	const now = nowFixture
	gone := Ticket{ID: "wgone", Actor: "session:dead", Lane: "gateway",
		EnqueuedUnix: now - 9000, RenewedUnix: now - 9000, TTLSeconds: 1800}
	livewaiter := waiter("wlive", "session:b", "gateway", now-60)

	res := Plan([]Ticket{gone, livewaiter}, nil, testTax(), Params{NowUnix: now})

	if len(res.Lapsed) != 1 || res.Lapsed[0] != "wgone" {
		t.Fatalf("lapsed = %v, want [wgone]", res.Lapsed)
	}
	if _, ok := res.Find("wgone"); ok {
		t.Error("a lapsed ticket still holds a place in line")
	}
	// The live waiter must be GRANTED: the abandoned ticket reserved nothing.
	if e := entryFor(t, res, "wlive"); !e.Grant {
		t.Errorf("live waiter blocked by an abandoned ticket: %+v", e)
	}
}

// Seniority survives a long wait as long as the waiter keeps polling: a four-hour waiter that
// renewed a second ago is NOT lapsed and still leads. This pins the two clocks apart.
func TestPlanKeepsSeniorityOfALongButStillPollingWaiter(t *testing.T) {
	const now = nowFixture
	old := Ticket{ID: "wold", Actor: "session:patient", Lane: "gateway",
		EnqueuedUnix: now - 4*3600, RenewedUnix: now - 1, TTLSeconds: 1800}

	res := Plan([]Ticket{old}, []Holder{{Lease: gatewayHolder("h1")}}, testTax(), Params{NowUnix: now})
	if len(res.Lapsed) != 0 {
		t.Fatalf("a still-polling waiter lapsed: %v", res.Lapsed)
	}
	if got := entryFor(t, res, "wold").WaitSeconds; got != 4*3600 {
		t.Errorf("wait = %ds, want %d (seniority is the ENQUEUE clock, not the renewal)", got, 4*3600)
	}
}

// An ETA is reported only when the blocking holder's expiry is actually declared. An unknown
// expiry yields NO eta rather than an invented one.
func TestPlanReportsETAOnlyFromADeclaredExpiry(t *testing.T) {
	const now = nowFixture
	w := waiter("w1", "session:me", "gateway", now-60)

	known := Plan([]Ticket{w}, []Holder{{Lease: gatewayHolder("h1"), ExpiresUnix: now + 360}},
		testTax(), Params{NowUnix: now})
	e := entryFor(t, known, "w1")
	if !e.ETAKnown || e.ETASeconds != 360 {
		t.Errorf("eta known=%v seconds=%d, want true/360", e.ETAKnown, e.ETASeconds)
	}

	unknown := Plan([]Ticket{w}, []Holder{{Lease: gatewayHolder("h1")}}, testTax(), Params{NowUnix: now})
	u := entryFor(t, unknown, "w1")
	if u.ETAKnown || u.ETASeconds != 0 {
		t.Errorf("eta known=%v seconds=%d, want false/0 for an undeclared expiry", u.ETAKnown, u.ETASeconds)
	}

	// A blocker that is another WAITER has no lease expiry at all, so it must carry no ETA.
	pair := Plan([]Ticket{w, waiter("w2", "session:other", "gateway", now-30)}, nil, testTax(),
		Params{NowUnix: now})
	if b := entryFor(t, pair, "w2"); b.ETAKnown {
		t.Error("a waiter-blocked entry reported an ETA it cannot know")
	}
}

// An exclusive lane still runs alone: the queue makes the wait legible, it never relaxes exclusion.
func TestPlanDoesNotRelaxExclusiveLanes(t *testing.T) {
	const now = nowFixture
	w := waiter("wrel", "session:me", "release", now-60)

	res := Plan([]Ticket{w}, []Holder{{Lease: gatewayHolder("h1")}}, testTax(), Params{NowUnix: now})
	e := entryFor(t, res, "wrel")
	if e.Grant {
		t.Fatal("an exclusive-lane waiter was admitted while another lease was live")
	}
	if e.Rung != regionadmit.RungExclusiveRequested {
		t.Errorf("rung = %q, want %q", e.Rung, regionadmit.RungExclusiveRequested)
	}
}

// The zero-value Params must change no existing decision: aging off leaves the order at arrival
// semantics, so wiring the queue in is safe before any knob is turned on.
func TestZeroParamsLeavesTheOrderAtArrival(t *testing.T) {
	const now = nowFixture
	// The loop waited far longer; with aging OFF the heavier interactive class still leads
	// (base weight leads), and with the default aging law ON the long wait promotes the loop.
	loop := waiter("wloop", "loop:nightly", "gateway", now-7*3600)
	hot := operator("whot", "session:op", "gateway", now-5)
	holders := []Holder{{Lease: gatewayHolder("h1")}}

	zero := Plan([]Ticket{loop, hot}, holders, testTax(), Params{NowUnix: now})
	if got := entryFor(t, zero, "whot").Place; got != 1 {
		t.Errorf("zero-Params place for the heavier class = %d, want 1 (base weight leads)", got)
	}
	for _, e := range zero.Entries {
		if e.Standing != dispatchaging.StandingFresh {
			t.Errorf("zero Params boosted %q to standing %q; aging must be off", e.ID, e.Standing)
		}
		if e.EffectiveWeight != e.weight() {
			t.Errorf("zero Params changed %q effective weight to %d, want %d",
				e.ID, e.EffectiveWeight, e.weight())
		}
	}

	tuned := Plan([]Ticket{loop, hot}, holders, testTax(),
		Params{NowUnix: now, Aging: dispatchaging.DefaultParams(now)})
	if got := entryFor(t, tuned, "wloop").Place; got != 1 {
		t.Errorf("with aging on, the starved loop place = %d, want 1", got)
	}
}

// Total over degenerate input.
func TestPlanIsTotalOverEmptyInput(t *testing.T) {
	res := Plan(nil, nil, regionadmit.Taxonomy{}, Params{})
	if res.Schema != Schema || len(res.Entries) != 0 || res.Depth != 0 {
		t.Fatalf("empty plan = %+v", res)
	}
}
