package ctxplan

import (
	"context"
	"testing"
)

// issue #1587: the context page-fault PROTOCOL — a forecast miss must produce exactly
// one closed-vocabulary decision (page-in / query-user / deny / continue-with-pointer),
// never a silent omission. Each test isolates one outcome (plus the "no fault" resident
// case) so the property under test is the only thing that moves, matching the
// fault_test.go / assumption_test.go style already in this package.

// TestDecidePageFaultChoosesPageInForRequiredReconstructableSpan is the headline "ordinary
// forecast miss" case: a live, reconstructable span the turn actually needs pages in.
func TestDecidePageFaultChoosesPageInForRequiredReconstructableSpan(t *testing.T) {
	ev := PageFaultEvent{
		SpanID: "span:2", Step: 2, State: PageFaultSpanLive,
		Durability: DurabilityTurn, Required: true, SilentlyReconstructable: true,
	}
	d := DecidePageFault(ev, DefaultPageFaultPolicy())
	if d.Outcome != PageFaultPageIn {
		t.Fatalf("a required, reconstructable, live span must page in, got %q (%s)", d.Outcome, d.Reason)
	}
	if d.SpanID != "span:2" || d.Step != 2 {
		t.Errorf("decision must echo the event identity, got %+v", d)
	}
	if d.Reason == "" {
		t.Error("a decision must carry a non-empty, operator-readable reason")
	}
}

// TestDecidePageFaultQueriesUserWhenNotSilentlyReconstructable: a live span the caller
// has flagged as NOT safely reconstructable (e.g. a one-time human input) must ask the
// user rather than assume a silent reload is faithful — even though the store still has
// bytes for it.
func TestDecidePageFaultQueriesUserWhenNotSilentlyReconstructable(t *testing.T) {
	ev := PageFaultEvent{
		SpanID: "span:5", State: PageFaultSpanLive,
		Required: true, SilentlyReconstructable: false,
	}
	d := DecidePageFault(ev, DefaultPageFaultPolicy())
	if d.Outcome != PageFaultQueryUser {
		t.Fatalf("a live-but-not-safely-reconstructable span must query the user, got %q (%s)", d.Outcome, d.Reason)
	}
}

// TestDecidePageFaultDeniesRequiredSealedSpan: the trust gate holds even at the decision
// layer. A required span that is sealed or tombstoned must DENY, never page in (which
// would launder poison/suppression into context) and never silently continue (which
// would drop a stated requirement).
func TestDecidePageFaultDeniesRequiredSealedSpan(t *testing.T) {
	for _, state := range []PageFaultSpanState{PageFaultSpanSealed, PageFaultSpanTombstoned} {
		ev := PageFaultEvent{SpanID: "span:9", State: state, Required: true, SilentlyReconstructable: true}
		d := DecidePageFault(ev, DefaultPageFaultPolicy())
		if d.Outcome != PageFaultDeny {
			t.Errorf("a required %s span must deny, got %q (%s)", state, d.Outcome, d.Reason)
		}
	}
}

// TestDecidePageFaultDeniesRequiredGoneSpan: a span that never entered the durable store
// at all (a hidden-restart drop) and is required this turn must also deny — there is
// nothing to silently reconstruct.
func TestDecidePageFaultDeniesRequiredGoneSpan(t *testing.T) {
	ev := PageFaultEvent{SpanID: "span:gone", State: PageFaultSpanGone, Required: true}
	d := DecidePageFault(ev, DefaultPageFaultPolicy())
	if d.Outcome != PageFaultDeny {
		t.Fatalf("a required, unrecoverable (gone) span must deny, got %q (%s)", d.Outcome, d.Reason)
	}
}

// TestDecidePageFaultContinuesWithPointerForUnrequiredUnrecoverableSpan: an unrecoverable
// span (sealed/tombstoned/gone) the turn is NOT blocked on must not stall the turn, but
// must also not silently vanish — it continues with a pointer.
func TestDecidePageFaultContinuesWithPointerForUnrequiredUnrecoverableSpan(t *testing.T) {
	for _, state := range []PageFaultSpanState{PageFaultSpanSealed, PageFaultSpanTombstoned, PageFaultSpanGone} {
		ev := PageFaultEvent{SpanID: "span:aside", State: state, Required: false}
		d := DecidePageFault(ev, DefaultPageFaultPolicy())
		if d.Outcome != PageFaultContinuePointer {
			t.Errorf("an unrequired %s span must continue with a pointer, got %q (%s)", state, d.Outcome, d.Reason)
		}
	}
}

// TestDecidePageFaultContinuesWithPointerForDurableUnrequiredSpan: a live, reconstructable,
// but NOT required span at or above the policy's durability floor defers to a pointer
// instead of paying an immediate fault — the "defer the reload, the fact is durable
// enough to recover later" branch.
func TestDecidePageFaultContinuesWithPointerForDurableUnrequiredSpan(t *testing.T) {
	ev := PageFaultEvent{
		SpanID: "span:durable", State: PageFaultSpanLive,
		Durability: DurabilityDurable, Required: false, SilentlyReconstructable: true,
	}
	d := DecidePageFault(ev, DefaultPageFaultPolicy())
	if d.Outcome != PageFaultContinuePointer {
		t.Fatalf("a durable, unrequired, live span must continue with a pointer, got %q (%s)", d.Outcome, d.Reason)
	}
}

// TestDecidePageFaultPagesInShortLivedUnrequiredSpan: a live, reconstructable, unrequired
// span BELOW the durability floor (turn-scoped) still pages in immediately — deferring a
// short-lived fact risks losing it, so the cheap default is to fault it in now.
func TestDecidePageFaultPagesInShortLivedUnrequiredSpan(t *testing.T) {
	ev := PageFaultEvent{
		SpanID: "span:ephemeral", State: PageFaultSpanLive,
		Durability: DurabilityTurn, Required: false, SilentlyReconstructable: true,
	}
	d := DecidePageFault(ev, DefaultPageFaultPolicy())
	if d.Outcome != PageFaultPageIn {
		t.Fatalf("a short-lived, unrequired, live span must still page in, got %q (%s)", d.Outcome, d.Reason)
	}
}

// TestDecidePageFaultIsClosedVocabulary: every decision DecidePageFault produces, across
// a broad sweep of the input space, must be a member of the closed PageFaultOutcome
// vocabulary — there is no reachable "no decision" / empty-outcome state.
func TestDecidePageFaultIsClosedVocabulary(t *testing.T) {
	states := []PageFaultSpanState{PageFaultSpanLive, PageFaultSpanSealed, PageFaultSpanTombstoned, PageFaultSpanGone, "", "garbage"}
	durabilities := []string{DurabilityTurn, DurabilitySession, DurabilityBounded, DurabilityDurable, "", "unknown"}
	bools := []bool{true, false}
	n := 0
	for _, s := range states {
		for _, dur := range durabilities {
			for _, req := range bools {
				for _, recon := range bools {
					ev := PageFaultEvent{SpanID: "span:x", State: s, Durability: dur, Required: req, SilentlyReconstructable: recon}
					d := DecidePageFault(ev, DefaultPageFaultPolicy())
					if !ValidPageFaultOutcome(d.Outcome) {
						t.Fatalf("outcome %q is not in the closed vocabulary for event %+v", d.Outcome, ev)
					}
					n++
				}
			}
		}
	}
	if n == 0 {
		t.Fatal("sweep produced no cases")
	}
}

// TestDecidePageFaultDecisionIsReplayable is the DETERMINISM witness: the same event and
// policy, decided many times (including via a fresh PageFaultLog), always produce the
// byte-identical outcome and reason — no wall-clock/random nondeterminism in the
// decision logic.
func TestDecidePageFaultDecisionIsReplayable(t *testing.T) {
	ev := PageFaultEvent{
		SpanID: "span:3", Step: 3, State: PageFaultSpanLive,
		Durability: DurabilityBounded, Required: false, SilentlyReconstructable: true,
	}
	policy := DefaultPageFaultPolicy()
	first := DecidePageFault(ev, policy)
	for i := 0; i < 25; i++ {
		got := DecidePageFault(ev, policy)
		if got != first {
			t.Fatalf("replay %d diverged: first=%+v got=%+v", i, first, got)
		}
	}
}

// TestPageFaultLogAppendAndReplay exercises the PERSISTED-STATE half: a log of several
// typed transitions must replay to the SAME decisions from its own stored
// (event, policy) pairs, with zero divergence — the log is a faithful, replayable record
// of what was decided, not just a cache of a claim.
func TestPageFaultLogAppendAndReplay(t *testing.T) {
	var log PageFaultLog
	policy := DefaultPageFaultPolicy()

	events := []PageFaultEvent{
		{SpanID: "span:a", Step: 0, State: PageFaultSpanLive, Durability: DurabilityTurn, Required: true, SilentlyReconstructable: true},
		{SpanID: "span:b", Step: 1, State: PageFaultSpanSealed, Required: true},
		{SpanID: "span:c", Step: 2, State: PageFaultSpanSealed, Required: false},
		{SpanID: "span:d", Step: 3, State: PageFaultSpanLive, Durability: DurabilityDurable, Required: false, SilentlyReconstructable: true},
		{SpanID: "span:e", Step: 4, State: PageFaultSpanLive, Required: true, SilentlyReconstructable: false},
	}
	wantOutcomes := []PageFaultOutcome{
		PageFaultPageIn, PageFaultDeny, PageFaultContinuePointer, PageFaultContinuePointer, PageFaultQueryUser,
	}

	for i, ev := range events {
		d := log.Append(ev, policy)
		if d.Outcome != wantOutcomes[i] {
			t.Fatalf("event %d (%s): got outcome %q, want %q", i, ev.SpanID, d.Outcome, wantOutcomes[i])
		}
	}

	entries := log.Entries()
	if len(entries) != len(events) {
		t.Fatalf("log must record every appended entry: got %d, want %d", len(entries), len(events))
	}

	verdicts, allMatch := log.Replay()
	if !allMatch {
		t.Fatalf("a freshly-built log must replay with zero divergence: %+v", verdicts)
	}
	if len(verdicts) != len(events) {
		t.Fatalf("replay must produce one verdict per entry: got %d, want %d", len(verdicts), len(events))
	}
	for i, v := range verdicts {
		if v.Diverged {
			t.Errorf("entry %d (%s) diverged on replay: stored=%q recomputed=%q", i, v.SpanID, v.Stored, v.Recomputed)
		}
	}

	sum := log.Summary()
	if sum.PageIn != 1 || sum.Deny != 1 || sum.ContinuePointer != 2 || sum.QueryUser != 1 {
		t.Errorf("summary must match the per-event outcomes: %+v", sum)
	}

	if explain := log.Explain(); explain == "" {
		t.Error("Explain must render a non-empty operator report")
	}
}

// TestPageFaultLogReplayDetectsDivergence proves Replay is a REAL check, not a rubber
// stamp: an entry manually corrupted to disagree with what DecidePageFault would compute
// from its own stored event+policy must be flagged Diverged, and allMatch must go false.
func TestPageFaultLogReplayDetectsDivergence(t *testing.T) {
	var log PageFaultLog
	policy := DefaultPageFaultPolicy()
	ev := PageFaultEvent{SpanID: "span:tamper", State: PageFaultSpanLive, Required: true, SilentlyReconstructable: true}
	log.Append(ev, policy) // decides PageFaultPageIn

	// Tamper with the stored decision directly (simulating a corrupted/foreign log entry).
	log.entries[0].Decision.Outcome = PageFaultDeny

	verdicts, allMatch := log.Replay()
	if allMatch {
		t.Fatal("Replay must detect a tampered entry, not report allMatch=true")
	}
	if len(verdicts) != 1 || !verdicts[0].Diverged {
		t.Fatalf("the tampered entry must be reported Diverged: %+v", verdicts)
	}
	if verdicts[0].Stored != PageFaultDeny || verdicts[0].Recomputed != PageFaultPageIn {
		t.Errorf("verdict must name both the stored and recomputed outcome: %+v", verdicts[0])
	}
}

// TestEventFromMissBuildsEventFromSpan exercises the ctxplan-vocabulary adapter: a Span
// the planner already elided lowers into a PageFaultEvent with the right State.
func TestEventFromMissBuildsEventFromSpan(t *testing.T) {
	live := Span{ID: "span:1", Step: 1, Durability: DurabilityBounded}
	sealed := Span{ID: "span:2", Step: 2, Sealed: true}
	tomb := Span{ID: "span:3", Step: 3, Tombstoned: true}

	if ev := EventFromMiss(live, true, true); ev.State != PageFaultSpanLive {
		t.Errorf("a live span must map to PageFaultSpanLive, got %q", ev.State)
	}
	if ev := EventFromMiss(sealed, true, true); ev.State != PageFaultSpanSealed {
		t.Errorf("a sealed span must map to PageFaultSpanSealed, got %q", ev.State)
	}
	if ev := EventFromMiss(tomb, true, true); ev.State != PageFaultSpanTombstoned {
		t.Errorf("a tombstoned span must map to PageFaultSpanTombstoned, got %q", ev.State)
	}
}

// TestMaterializeFaultDecisionExecutesPageIn: a PageFaultPageIn decision, when executed
// against a real Store+View, must actually splice the span in via DemandPage — the
// decision is not just advisory metadata, it drives the real mechanism.
func TestMaterializeFaultDecisionExecutesPageIn(t *testing.T) {
	v, store := faultView(t)
	if residentIDs(v)["span:2"] {
		t.Fatalf("setup: span:2 must be elided for the test")
	}
	ev := PageFaultEvent{SpanID: "span:2", Step: 2, State: PageFaultSpanLive, Durability: DurabilityTurn, Required: true, SilentlyReconstructable: true}
	decision := DecidePageFault(ev, DefaultPageFaultPolicy())
	if decision.Outcome != PageFaultPageIn {
		t.Fatalf("setup: expected a page_in decision, got %q", decision.Outcome)
	}

	out, fault, err := MaterializeFaultDecision(context.Background(), store, v, decision)
	if err != nil {
		t.Fatal(err)
	}
	if fault.Status != FaultServed {
		t.Fatalf("executing a page_in decision must serve the fault, got status=%q", fault.Status)
	}
	if !residentIDs(out)["span:2"] {
		t.Errorf("after executing a page_in decision span:2 must be resident, got %v", residentIDs(out))
	}
}

// TestMaterializeFaultDecisionSkipsNonPageIn: the other three outcomes must NOT touch
// the store or the view — MaterializeFaultDecision only executes page_in.
func TestMaterializeFaultDecisionSkipsNonPageIn(t *testing.T) {
	v, store := faultView(t)
	for _, outcome := range []PageFaultOutcome{PageFaultQueryUser, PageFaultDeny, PageFaultContinuePointer} {
		decision := PageFaultDecision{SpanID: "span:2", Outcome: outcome, Reason: "test"}
		out, fault, err := MaterializeFaultDecision(context.Background(), store, v, decision)
		if err != nil {
			t.Fatalf("outcome %q: unexpected error %v", outcome, err)
		}
		if fault.Status != "" {
			t.Errorf("outcome %q must not attempt a page-in, got fault status=%q", outcome, fault.Status)
		}
		if residentIDs(out)["span:2"] {
			t.Errorf("outcome %q must not make span:2 resident", outcome)
		}
		if len(out.Rendered) != len(v.Rendered) {
			t.Errorf("outcome %q must leave the view unchanged", outcome)
		}
	}
}

// TestPageFaultNoFaultCaseIsNotInThisVocabulary documents the boundary: when a span is
// already resident, the FORECAST never emits a miss/PageFaultEvent for it at all — the
// "no fault" case is the ABSENCE of a PageFaultEvent, not a fifth PageFaultOutcome
// member. This test pins that contract against fault.go's own idempotent-resident
// behavior so the two layers cannot silently drift: DemandPage on an already-resident
// span reports FaultResident (not a page-fault-protocol outcome), and the resident set
// is unchanged, matching a caller that correctly never constructed an event for it.
func TestPageFaultNoFaultCaseIsNotInThisVocabulary(t *testing.T) {
	v, store := faultView(t)
	if !residentIDs(v)["span:0"] {
		t.Fatalf("setup: span:0 must be pinned-resident")
	}
	// No PageFaultEvent is built for a resident span; demand-paging it directly (the
	// defensive/idempotent path) must stay a fault.go-level no-op, never surface as one
	// of the four PageFaultOutcome values.
	out, fault, err := DemandPage(context.Background(), store, v, "span:0")
	if err != nil {
		t.Fatal(err)
	}
	if fault.Status != FaultResident {
		t.Fatalf("an already-resident span must stay FaultResident, got %q", fault.Status)
	}
	if ValidPageFaultOutcome(PageFaultOutcome(fault.Status)) {
		t.Errorf("fault.go's FaultResident must not collide with the PageFaultOutcome vocabulary")
	}
	if len(out.Rendered) != len(v.Rendered) {
		t.Errorf("a resident span must not change the view")
	}
}
