package gateway

// admission_reservation_driver_test.go ΓÇö the adversarial witness for pub#13388:
// the served admission path must DRIVE advisory KV cache reservations from
// Table/#811 TurnIntent hints, mint exactly one bounded ReservationObservation per
// scan, and never let a hint promote a reservation or borrow admission capacity.
//
// Written from the SPEC only (no implementation reasoning). It exercises the real
// served `AdmissionController.Schedule` entry point ΓÇö not a private helper ΓÇö with a
// deterministic `SetClock` and the controller's CURRENTLY BOUND table + scheduler.

import (
	"fmt"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/session"
)

// reservationDriverClock is the fixed captured time every Schedule round in this
// test observes. time.Unix(100, 0) is deterministic and exercises the UnixNano
// reservation math without a monotonic reading.
var reservationDriverClock = time.Unix(100, 0)

// newReservationDriverController wires the controller exactly as the spec describes:
// a session.Table holding hints, a strict-priority scheduler ATTACHED to that same
// table (ReserveKnownComing walks the scheduler's OWN table snapshot, not the
// controller's), the controller bound to both, and a frozen clock.
//
// maxNumSeqs is a parameter, not a constant: the served promotion path can only
// admit a waiter while the running count has headroom. A controller whose Schedule
// is asserted non-empty must leave room for the real waiter (maxNumSeqs >= 2 after
// the real blocker is admitted); a capacity-pressure edge instead isolates the axis
// it means to stress (e.g. TokenBudget 20) rather than tripping over the seq cap.
func newReservationDriverController(t *testing.T, budget, maxNumSeqs int) (*AdmissionController, *session.Table, *session.Scheduler) {
	t.Helper()
	tbl := session.NewTable()
	sched := session.NewScheduler(session.StrictPriority)
	sched.Attach(tbl, session.AttachOptions{})

	ctl := NewAdmissionController(AdmissionPolicy{
		TokenBudget: budget,
		MaxNumSeqs:  maxNumSeqs,
		MaxWaiting:  64,
		AgingRounds: 1,
	})
	ctl.SetTable(tbl)
	ctl.SetSequencer(sched)
	ctl.SetClock(func() time.Time { return reservationDriverClock })
	return ctl, tbl, sched
}

// saturateRealQueue fills the running set to the max-num-seqs cap with real work and
// queues one real overflow waiter, then returns the first admitted blocker's trace id.
// A subsequent Complete(first) opens exactly one slot on the per-lane reclaim edge, so
// the next real Schedule() has a genuine waiter to PROMOTE ΓÇö the served promotion path,
// not the empty-queue early return. The overflow waiter really queues because the fast
// path can only admit while the running count has headroom. The hint table is
// independent of these traces.
func saturateRealQueue(t *testing.T, ctl *AdmissionController) string {
	t.Helper()
	capacity := ctl.Policy().MaxNumSeqs
	if capacity < 1 {
		t.Fatalf("saturateRealQueue: maxNumSeqs=%d leaves no running slot to fill", capacity)
	}
	first := ""
	for i := 0; i < capacity; i++ {
		id := "real-blocker"
		if i > 0 {
			id = fmt.Sprintf("real-blocker-%d", i)
		}
		if v := ctl.Offer(SeqRequest{TraceID: id, Tokens: 10}); v != VerdictAdmitted {
			t.Fatalf("%s: verdict = %s, want admitted", id, v)
		}
		if i == 0 {
			first = id
		}
	}
	if v := ctl.Offer(SeqRequest{TraceID: "real-waiter", Tokens: 5}); v != VerdictQueued {
		t.Fatalf("real-waiter: verdict = %s, want queued", v)
	}
	return first
}

// TestServedAdmissionDrivesReservations is the pub#13388 exit gate. It asserts the
// three acceptance criteria and the adversarial edges the ticket names.
func TestServedAdmissionDrivesReservations(t *testing.T) {
	// --- Criterion 2(a): no table attached yields NO observation -----------------
	//
	// Exercises the served Schedule path on a controller with no table; the scan must
	// stay empty rather than fabricate a reservation. The seq cap is irrelevant here
	// (the scan is skipped entirely) ΓÇö what is asserted is the empty readback.
	t.Run("no_table_no_observation", func(t *testing.T) {
		bare := NewAdmissionController(AdmissionPolicy{TokenBudget: 100, MaxNumSeqs: 1, MaxWaiting: 8, AgingRounds: 1})
		bare.SetClock(func() time.Time { return reservationDriverClock })
		bare.Offer(SeqRequest{TraceID: "b-blocker", Tokens: 10})
		bare.Offer(SeqRequest{TraceID: "b-waiter", Tokens: 5})
		bare.Schedule()
		if got := bare.ReservationObservations(); len(got) != 0 {
			t.Fatalf("no table attached: ReservationObservations() = %+v, want empty", got)
		}
	})

	// --- Criterion 1 + 3 + edges: creation, boundedness, shared At ---------------
	const knownPrefix = "sha256:known-prefix"
	ctl, tbl, sched := newReservationDriverController(t, 1000, 2)
	firstBlocker := saturateRealQueue(t, ctl)

	// Seed the controller's BOUND table with a live reservation hint.
	if _, ok := tbl.SetTurnIntent("tool-waiting", session.TurnIntent{
		ArrivingInMillis: 50,
		Prefix:           knownPrefix,
	}); !ok {
		t.Fatal("SetTurnIntent(tool-waiting) rejected")
	}

	// A real probe offered while the hint is present must NOT be granted capacity by
	// it: the running set is saturated and a real waiter is already ahead of it.
	offerBefore := ctl.Offer(SeqRequest{TraceID: "probe-before", Tokens: 5})
	if offerBefore != VerdictQueued && offerBefore != VerdictShed {
		t.Fatalf("probe with hint present = %s, want queued/shed (hint must not grant capacity)", offerBefore)
	}

	// Open exactly one real slot on the per-lane reclaim edge, then drive the REAL
	// served admission path with the deterministic clock. The real waiter must be
	// promoted here, proving this is the genuine promotion path rather than the
	// empty-queue early return.
	ctl.Complete(firstBlocker)
	if admitted := ctl.Schedule(); len(admitted) == 0 {
		t.Fatal("Schedule admitted nothing; test needs the served promotion path (not the empty-queue early return)")
	}

	obs := ctl.ReservationObservations()

	// Criterion 1: exactly one "created" observation with the exact prefix and the
	// captured clock time ΓÇö the SAME time `SetClock` injected.
	var created *ReservationObservation
	for i := range obs {
		if obs[i].Action == "created" {
			created = &obs[i]
			break
		}
	}
	if created == nil {
		t.Fatalf("no 'created' ReservationObservation after Schedule; got %+v", obs)
	}
	if created.TraceID != "tool-waiting" {
		t.Fatalf("created.TraceID = %q, want %q", created.TraceID, "tool-waiting")
	}
	if created.Prefix != knownPrefix {
		t.Fatalf("created.Prefix = %q, want %q", created.Prefix, knownPrefix)
	}
	if !created.At.Equal(reservationDriverClock) {
		t.Fatalf("created.At = %v, want the injected capture time %v", created.At, reservationDriverClock)
	}

	// Criterion 3: every observation is labeled with a closed Action token and a
	// non-empty Reason, and NO promotion happened ΓÇö the scheduler still HOLDS the
	// reservation on the creation round (promotion would have deleted it). All three
	// tokens in the closed vocabulary are now reachable and witnessed on this served
	// path: "created" here, "expired" on the generation-1 in-place reclaim below, and
	// "canceled" on the generation-2 hint withdrawal.
	for _, o := range obs {
		switch o.Action {
		case "created", "expired", "canceled":
		default:
			t.Fatalf("observation Action %q outside {created,expired,canceled}: %+v", o.Action, o)
		}
		if o.Reason == "" {
			t.Fatalf("observation %+v has empty Reason, want a non-empty reason token", o)
		}
	}
	if got := sched.Reservations(reservationDriverClock); len(got) == 0 {
		t.Fatal("scheduler holds no reservation after a 'created' scan; a promotion may have deleted it")
	}

	// Criterion 2(b): the hint did NOT move real-request accounting. The running set
	// and token gauge reflect ONLY the real work admitted this round ΓÇö real-blocker
	// (10 tokens) plus the promoted real-waiter (5) = 2 running / 15 tokens. Any
	// phantom token charged by the advisory reservation would break the exact sum.
	statsAfter := ctl.Stats()
	if statsAfter.Running != 2 || statsAfter.TokensInUse != 15 {
		t.Fatalf("hint perturbed real accounting: running=%d tokens=%d, want 2/15 (real work only)",
			statsAfter.Running, statsAfter.TokensInUse)
	}

	// Edge: re-scanning the SAME unchanged hint must NOT mint a duplicate "created"
	// observation, and the readback must stay bounded across many scans.
	for i := 0; i < 200; i++ {
		ctl.Schedule()
	}
	after := ctl.ReservationObservations()
	if len(after) > 64 {
		t.Fatalf("ReservationObservations() len = %d after 200 scans, want bounded (<= 64)", len(after))
	}
	createdForHint := 0
	for _, o := range after {
		if o.Action == "created" && o.TraceID == "tool-waiting" && o.Prefix == knownPrefix && o.At.Equal(reservationDriverClock) {
			createdForHint++
		}
	}
	if createdForHint != 1 {
		t.Fatalf("unchanged hint minted %d 'created' observations, want exactly 1 (generation must be stable)", createdForHint)
	}

	// Edge: two reservations created in ONE round share the identical captured At.
	ctl2, tbl2, _ := newReservationDriverController(t, 1000, 2)
	saturateRealQueue(t, ctl2)
	tbl2.SetTurnIntent("hint-a", session.TurnIntent{ArrivingInMillis: 40, Prefix: "sha256:prefix-a"})
	tbl2.SetTurnIntent("hint-b", session.TurnIntent{ArrivingInMillis: 60, Prefix: "sha256:prefix-b"})
	ctl2.Schedule()
	var atA, atB time.Time
	for _, o := range ctl2.ReservationObservations() {
		if o.Action != "created" {
			continue
		}
		switch o.TraceID {
		case "hint-a":
			atA = o.At
		case "hint-b":
			atB = o.At
		}
	}
	if atA.IsZero() || atB.IsZero() {
		t.Fatalf("two hints in one round did not both mint 'created' observations: %+v", ctl2.ReservationObservations())
	}
	if !atA.Equal(atB) {
		t.Fatalf("two observations in ONE round have different At: %v != %v (capture must happen once)", atA, atB)
	}

	// Edge (criterion 2b, paired control): an IDENTICAL real workload on a hint-free
	// controller yields identical running/token gauges and identical verdicts. This is
	// the non-vacuous form of "the hint changed nothing": the hinted peer really did
	// mint a reservation, yet its real-request accounting is byte-identical.
	control, _, _ := newReservationDriverController(t, 1000, 2)
	hinted, hintedTbl, _ := newReservationDriverController(t, 1000, 2)
	hintedTbl.SetTurnIntent("tool-waiting", session.TurnIntent{ArrivingInMillis: 50, Prefix: knownPrefix})
	cv1 := control.Offer(SeqRequest{TraceID: "r1", Tokens: 10})
	hv1 := hinted.Offer(SeqRequest{TraceID: "r1", Tokens: 10})
	cv2 := control.Offer(SeqRequest{TraceID: "r2", Tokens: 5})
	hv2 := hinted.Offer(SeqRequest{TraceID: "r2", Tokens: 5})
	control.Schedule()
	hinted.Schedule()
	cs, hs := control.Stats(), hinted.Stats()
	if cv1 != hv1 || cv2 != hv2 {
		t.Fatalf("hint changed real verdicts: hint-free (%s,%s) vs hinted (%s,%s)", cv1, cv2, hv1, hv2)
	}
	if cs.Running != hs.Running || cs.TokensInUse != hs.TokensInUse {
		t.Fatalf("hint changed real capacity: hint-free running=%d tokens=%d, hinted running=%d tokens=%d",
			cs.Running, cs.TokensInUse, hs.Running, hs.TokensInUse)
	}
	if len(hinted.ReservationObservations()) == 0 {
		t.Fatal("hinted control minted no reservation; the equality above would be vacuous")
	}

	// Edge: expiry is terminal for an unchanged intent. A genuinely cleared and
	// re-published intent may then establish a fresh generation for the same trace+prefix.
	ctl3, tbl3, sched3 := newReservationDriverController(t, 1000, 2)
	saturateRealQueue(t, ctl3)
	tbl3.SetTurnIntent("gen-hint", session.TurnIntent{ArrivingInMillis: 30, Prefix: "sha256:gen-prefix"})
	ctl3.Schedule()
	firstCreated := latestCreated(ctl3.ReservationObservations(), "gen-hint", "sha256:gen-prefix")
	if firstCreated == nil {
		t.Fatalf("no first 'created' for gen-hint: %+v", ctl3.ReservationObservations())
	}

	// Advance past the reservation's expiry boundary (arrival + grace).
	expiredAt := reservationDriverClock.Add(time.Duration(session.DefaultReservationGrace) + time.Hour)
	ctl3.SetClock(func() time.Time { return expiredAt })
	ctl3.Schedule()

	// The controller's own readback is the deliverable: the reclaim must surface as an
	// "expired" ReservationObservation and the unchanged hint must not re-mint a hold.
	var expiredSeen bool
	for _, o := range ctl3.ReservationObservations() {
		if o.Action == "expired" && o.TraceID == "gen-hint" && o.Prefix == "sha256:gen-prefix" {
			expiredSeen = true
		}
	}
	if !expiredSeen {
		t.Fatalf("no 'expired' observation for gen-hint after crossing the expiry boundary: %+v", ctl3.ReservationObservations())
	}
	if got := sched3.Reservations(expiredAt); len(got) != 0 {
		t.Fatalf("unchanged expired intent re-minted reservations: %+v", got)
	}
	if got := latestCreated(ctl3.ReservationObservations(), "gen-hint", "sha256:gen-prefix"); got == nil || got.Generation != firstCreated.Generation {
		t.Fatalf("unchanged expired intent minted a new generation: first=%+v latest=%+v", firstCreated, got)
	}

	// Clear is a witnessed identity transition. Re-publishing after that transition
	// may establish a new generation at the newly captured clock.
	clearedAt := expiredAt.Add(time.Hour)
	ctl3.SetClock(func() time.Time { return clearedAt })
	tbl3.SetTurnIntent("gen-hint", session.TurnIntent{})
	ctl3.Schedule()
	republishedAt := clearedAt.Add(time.Hour)
	ctl3.SetClock(func() time.Time { return republishedAt })
	tbl3.SetTurnIntent("gen-hint", session.TurnIntent{ArrivingInMillis: 30, Prefix: "sha256:gen-prefix"})
	ctl3.Schedule()
	secondCreated := latestCreated(ctl3.ReservationObservations(), "gen-hint", "sha256:gen-prefix")
	if secondCreated == nil {
		t.Fatalf("no created observation after clear and re-publish: %+v", ctl3.ReservationObservations())
	}
	if secondCreated.Generation == firstCreated.Generation {
		t.Fatalf("re-created reservation kept Generation %d; want a distinct generation for the same trace+prefix",
			secondCreated.Generation)
	}
	if secondCreated.Generation <= firstCreated.Generation {
		t.Fatalf("re-created Generation %d not strictly after first %d; generation must be monotonic",
			secondCreated.Generation, firstCreated.Generation)
	}
	if !secondCreated.At.Equal(republishedAt) {
		t.Fatalf("re-created At = %v, want the re-publish clock %v", secondCreated.At, republishedAt)
	}

	// Now make the hint GENUINELY vanish from the scheduler's scan (WillDiscard), so
	// `ReserveKnownComing` reports nothing and the scheduler truly drops the hold.
	// ONLY then may the "scheduler holds none" assertion be made.
	vanishAt := republishedAt.Add(time.Hour)
	ctl3.SetClock(func() time.Time { return vanishAt })
	tbl3.SetTurnIntent("gen-hint", session.TurnIntent{
		ArrivingInMillis: 30,
		Prefix:           "sha256:gen-prefix",
		WillDiscard:      true,
	})
	ctl3.Schedule()
	if got := sched3.Reservations(vanishAt); len(got) != 0 {
		t.Fatalf("after the hint vanished the scheduler still holds %+v, want none", got)
	}
	// The controller retires the vanished hold as a "canceled" observation carrying the
	// SECOND generation's identity ΓÇö distinct from the generation-1 in-place "expired"
	// reclaim above. "canceled" is a genuinely reachable third lifecycle token: it marks
	// a hint WITHDRAWN before arrival, whereas "expired" marks an in-place reclaim at the
	// expiry boundary while the hint is still live. A missing observation here would let
	// "scheduler empty" pass for the wrong reason.
	var canceledSeen bool
	for _, o := range ctl3.ReservationObservations() {
		if o.Action == "canceled" && o.TraceID == "gen-hint" && o.Prefix == "sha256:gen-prefix" &&
			o.Generation == secondCreated.Generation && o.At.Equal(vanishAt) {
			if o.Reason == "" {
				t.Fatalf("canceled observation for generation %d has empty Reason: %+v", secondCreated.Generation, o)
			}
			canceledSeen = true
		}
	}
	if !canceledSeen {
		t.Fatalf("vanished hint minted no 'canceled' retirement observation for generation %d: %+v",
			secondCreated.Generation, ctl3.ReservationObservations())
	}

	// Edge: an invalid hint (WillDiscard, empty Prefix, or non-positive
	// ArrivingInMillis) yields NO reservation and NO observation.
	ctl4, tbl4, sched4 := newReservationDriverController(t, 1000, 2)
	saturateRealQueue(t, ctl4)
	tbl4.SetTurnIntent("discard-hint", session.TurnIntent{ArrivingInMillis: 50, Prefix: "sha256:x", WillDiscard: true})
	tbl4.SetTurnIntent("noprefix-hint", session.TurnIntent{ArrivingInMillis: 50})
	tbl4.SetTurnIntent("noarrive-hint", session.TurnIntent{Prefix: "sha256:y"})
	ctl4.Schedule()
	for _, o := range ctl4.ReservationObservations() {
		switch o.TraceID {
		case "discard-hint", "noprefix-hint", "noarrive-hint":
			t.Fatalf("invalid hint %q produced an observation: %+v", o.TraceID, o)
		}
	}
	if got := sched4.Reservations(reservationDriverClock); len(got) != 0 {
		t.Fatalf("invalid hints minted reservations: %+v", got)
	}

	// Edge: a table hint must never let a real request bypass capacity. With a
	// saturated TokenBudget, a real request with a hint present still queues/sheds
	// exactly as it would with no hint, and TokensInUse reflects only the real work.
	ctl5, tbl5, _ := newReservationDriverController(t, 20, 2)
	if v := ctl5.Offer(SeqRequest{TraceID: "sat-blocker", Tokens: 20}); v != VerdictAdmitted {
		t.Fatalf("sat-blocker: verdict = %s, want admitted (budget 20)", v)
	}
	tbl5.SetTurnIntent("sat-hint", session.TurnIntent{ArrivingInMillis: 50, Prefix: "sha256:sat"})
	ctl5.Schedule()
	satVerdict := ctl5.Offer(SeqRequest{TraceID: "sat-real", Tokens: 20})
	if satVerdict != VerdictQueued && satVerdict != VerdictShed {
		t.Fatalf("real request with a hint present and a saturated budget = %s, want queued/shed", satVerdict)
	}
	if st := ctl5.Stats(); st.TokensInUse != 20 {
		t.Fatalf("hint consumed admission budget: TokensInUse = %d, want 20 (reservations must not decrement capacity)", st.TokensInUse)
	}

	// Edge 1: a table revision bump for the SAME (trace, prefix) is INERT. A hold's
	// identity is its (trace, prefix); while its expiry boundary is still in the future
	// an unchanged / re-asserted hint ΓÇö including an unrelated TurnIntent field change
	// or a SetPriority bump that advances Table.Rev ΓÇö mints NO lifecycle observation.
	// The tracked generation is refreshed in place. This is the regression lock for the
	// in-place-refresh contract: BEFORE it landed, a Rev bump for a live hold fabricated
	// an "expired"+"created" pair and advanced the generation, so the count/exactly-one
	// assertions below would have FAILED (2 new observations for the same trace+prefix,
	// and a bumped generation) ΓÇö this edge is therefore a genuine lock, not vacuous.
	bumpCtl, bumpTbl, bumpSched := newReservationDriverController(t, 1000, 2)
	saturateRealQueue(t, bumpCtl)
	const bumpPrefix = "sha256:bump-prefix"
	bumpTbl.SetTurnIntent("bump-hint", session.TurnIntent{ArrivingInMillis: 50, Prefix: bumpPrefix})
	bumpCtl.Schedule()

	firstObs := bumpCtl.ReservationObservations()
	firstGen := latestCreated(firstObs, "bump-hint", bumpPrefix)
	if firstGen == nil {
		t.Fatalf("no initial 'created' for bump-hint: %+v", firstObs)
	}
	if got := bumpSched.Reservations(reservationDriverClock); len(got) != 1 {
		t.Fatalf("bump-hint: scheduler holds %d reservations after first scan, want exactly 1", len(got))
	}

	// Bump the table Rev WITHOUT changing the known-coming fields (Priority is not a
	// reservation input), then re-scan at the same frozen clock.
	if _, ok := bumpTbl.SetPriority("bump-hint", 1); !ok {
		t.Fatal("SetPriority(bump-hint) rejected")
	}
	bumpCtl.Schedule()

	afterObs := bumpCtl.ReservationObservations()
	if len(afterObs) != len(firstObs) {
		t.Fatalf("rev bump on a live hint changed observation count: %d -> %d (want unchanged): %+v",
			len(firstObs), len(afterObs), afterObs)
	}
	for _, o := range afterObs {
		if o.TraceID == "bump-hint" && o.Prefix == bumpPrefix && o.Action != "created" {
			t.Fatalf("rev bump on a live hint minted a %q observation for the same trace+prefix: %+v", o.Action, o)
		}
	}
	if afterGen := latestCreated(afterObs, "bump-hint", bumpPrefix); afterGen == nil {
		t.Fatalf("rev bump lost the bump-hint 'created' observation: %+v", afterObs)
	} else if afterGen.Generation != firstGen.Generation {
		t.Fatalf("rev bump changed the tracked generation: %d -> %d (want in-place refresh, generation unchanged)",
			firstGen.Generation, afterGen.Generation)
	}
	if got := bumpSched.Reservations(reservationDriverClock); len(got) != 1 {
		t.Fatalf("rev bump: scheduler holds %d reservations, want exactly 1 (unchanged hold)", len(got))
	}

	// Edge 2: a controller bound to a real Table but with NO scheduler is inert ΓÇö the
	// scan cannot run, so no observation is minted and real admission is unaffected.
	noSched := NewAdmissionController(AdmissionPolicy{TokenBudget: 1000, MaxNumSeqs: 2, MaxWaiting: 64, AgingRounds: 1})
	noSchedTbl := session.NewTable()
	noSched.SetTable(noSchedTbl) // deliberately NO SetSequencer
	noSched.SetClock(func() time.Time { return reservationDriverClock })
	firstNoSched := saturateRealQueue(t, noSched)
	noSchedTbl.SetTurnIntent("no-sched-hint", session.TurnIntent{ArrivingInMillis: 50, Prefix: "sha256:no-sched"})
	noSched.Complete(firstNoSched)
	if admitted := noSched.Schedule(); len(admitted) == 0 {
		t.Fatal("no-scheduler controller admitted nothing; real queue was not saturated/promoted")
	}
	if got := noSched.ReservationObservations(); len(got) != 0 {
		t.Fatalf("no-scheduler controller minted observations: %+v (scan must stay inert)", got)
	}
	if st := noSched.Stats(); st.Running != 2 || st.TokensInUse != 15 {
		t.Fatalf("no-scheduler controller real accounting perturbed: running=%d tokens=%d, want 2/15", st.Running, st.TokensInUse)
	}

	// Edge 3: a nil controller's readback is safe and empty, not a panic.
	var nilCtl *AdmissionController
	if got := nilCtl.ReservationObservations(); got != nil {
		t.Fatalf("nil controller ReservationObservations() = %+v, want nil", got)
	}
}

// The ReservationObservation.Action vocabulary is CLOSED and each token has exactly one
// meaning on the served path: "created" mints a fresh advisory hold; "expired" is
// expiry-ONLY (an in-place reclaim at the expiry boundary while the hint is still live,
// re-created under a new generation); "canceled" is withdrawal-ONLY (the hint genuinely
// vanished ΓÇö WillDiscard or a cleared/changed intent ΓÇö so the hold is retired, never
// re-created). A table Rev bump for an unchanged, still-live hold belongs to none of
// these and mints nothing (edge 1).

// latestCreated returns the most recently minted "created" observation for the given
// trace+prefix, or nil. It is the readback helper for the generation-identity edges.
func latestCreated(obs []ReservationObservation, trace, prefix string) *ReservationObservation {
	var out *ReservationObservation
	for i := range obs {
		o := obs[i]
		if o.Action == "created" && o.TraceID == trace && o.Prefix == prefix {
			out = &obs[i]
		}
	}
	return out
}
