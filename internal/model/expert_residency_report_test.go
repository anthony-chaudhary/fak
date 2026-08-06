package model

import (
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// expert_residency_report_test.go — the R6 witnesses for #5617 (epic #5606).
//
// The rung's claim is that activated-expert residency is now an object an operator can read, and
// that the reading is CHECKABLE rather than merely printed. The issue names three properties and
// each gets a test:
//
//	the report is the ring's own accounting — the reconciliation identity holds over a real decode
//	  window, is derived from an independently computed expectation, and (the part that makes it
//	  worth reporting) actually FAILS when a counter is made to drift;
//	the stranded placement gauge is live — a deliberately stale pin-set scores maximum drift, and
//	  repinning against what the window really routed moves the number in the expected direction;
//	observing does not change what is observed — polling the report through a window leaves both the
//	  outputs and the ring's counters byte-identical to a window that was never polled.
//
// Plus the cross-agent fold: under a shared ring (R7) the report carries the coalescing ledger and
// reconciles it against the ring's own counters, which are separate increments in separate files.

// moeReportWindow drives `steps` routed-expert activations through one session and returns the
// per-step outputs, so a second arm can be compared byte-for-byte against it.
func moeReportWindow(m *Model, s *Session, x []float32, window []int) [][]float32 {
	out := make([][]float32, len(window))
	for i, e := range window {
		out[i] = expertSwiGLU(m, 0, e, x, sessionQ4KKernel{s: s})
	}
	return out
}

func moeReportCheck(t *testing.T, rep MoEResidencyReport, name string) MoEResidencyCheck {
	t.Helper()
	for _, c := range rep.Reconciliation.Checks {
		if c.Name == name {
			return c
		}
	}
	t.Fatalf("reconciliation has no check %q; it reported %+v", name, rep.Reconciliation.Checks)
	return MoEResidencyCheck{}
}

// TestMoEResidencyReportReconcilesWithTheRingsOwnAccounting is the rung's central witness. It drives
// a window that exercises hits, page-ins and evictions through a ring too small to hold it, then
// checks the report three ways: the identity the ring counts on both sides, an expectation derived
// from the window itself (not from the report), and — the part that proves the check is live — a
// deliberately corrupted counter, which must turn the verdict red.
func TestMoEResidencyReportReconcilesWithTheRingsOwnAccounting(t *testing.T) {
	const H, E = 256, 6
	m := expertRingTestModel(t, H, E)
	m.Cfg.NumExpertsPerTok = 2
	x := expertRingTestInput(H)
	weight := expertRingWeightBytes(t, m)
	budget := weight * 6 // two whole experts; a six-expert window must evict

	s := expertRingSession(m, budget)
	window := []int{0, 1, 2, 3, 0, 1, 4, 5, 0}
	moeReportWindow(m, s, x, window)

	rep := s.MoEResidency(MoEResidencyOptions{Tokens: int64(len(window))})
	if !rep.Ring.Enabled {
		t.Fatal("report says the ring is disabled after a window that staged routed experts")
	}
	if !rep.Reconciliation.OK {
		t.Fatalf("reconciliation failed on a healthy ring: %+v", rep.Reconciliation.Checks)
	}

	// Derived from the WINDOW, not from the report: one activation stages gate/up/down, so the ring
	// saw exactly three stagings per step. If the report agreed with itself but not with this, the
	// identity would be a tautology.
	wantLookups := len(window) * 3
	if rep.Ring.Lookups != wantLookups {
		t.Fatalf("ring booked %d lookups over %d activations, want %d (3 projections each)",
			rep.Ring.Lookups, len(window), wantLookups)
	}
	if got := rep.Ring.Hits + rep.Ring.PageIns + rep.Ring.Refusals; got != rep.Ring.Lookups {
		t.Fatalf("hits(%d)+page_ins(%d)+refusals(%d)=%d != lookups(%d)",
			rep.Ring.Hits, rep.Ring.PageIns, rep.Ring.Refusals, got, rep.Ring.Lookups)
	}
	if rep.Ring.Evictions == 0 {
		t.Fatal("no eviction over a six-expert window through a two-expert budget; the bound was never exercised")
	}
	if rep.Ring.PeakBytes > budget {
		t.Fatalf("peak resident %d exceeds budget %d", rep.Ring.PeakBytes, budget)
	}
	// Every projection in the test model is the same size, so the byte meter is exactly checkable.
	if want := int64(rep.Ring.PageIns) * weight; rep.Ring.PageInBytes != want {
		t.Fatalf("page_in_bytes=%d, want %d (%d page-ins x %d B)",
			rep.Ring.PageInBytes, want, rep.Ring.PageIns, weight)
	}

	// The shape framing: the report is unreadable without k/E, which is the ladder's whole premise.
	if rep.Shape.Experts != E || rep.Shape.ExpertsPerToken != 2 {
		t.Fatalf("shape %+v does not carry the routing shape the rates are read against", rep.Shape)
	}
	if want := 2.0 / float64(E); rep.Shape.ActivatedFraction != want {
		t.Fatalf("activated fraction %.4f, want %.4f", rep.Shape.ActivatedFraction, want)
	}

	// Rates are ratios of fields in this same report, so each is re-derivable by hand.
	wantHit := float64(rep.Ring.Hits) / float64(rep.Ring.Hits+rep.Ring.PageIns)
	if rep.Rates.HitRate != wantHit {
		t.Fatalf("hit rate %.6f, want %.6f", rep.Rates.HitRate, wantHit)
	}
	if want := float64(rep.Ring.PageInBytes) / float64(len(window)); rep.Rates.ExpertBytesPerToken != want {
		t.Fatalf("expert bytes/token %.3f, want %.3f", rep.Rates.ExpertBytesPerToken, want)
	}
	if rep.Rates.RefusalRate != 0 {
		t.Fatalf("refusal rate %.4f on a budget that fits an expert; a refusal means the ring gave up "+
			"the bound and promoted a weight to permanent halW residency", rep.Rates.RefusalRate)
	}

	// A check that cannot fail is decoration. Make the two sides disagree by exactly one hit and the
	// verdict must go red, naming the identity rather than some generic error.
	s.expertRing.hit++
	bad := s.MoEResidency(MoEResidencyOptions{Tokens: int64(len(window))})
	if bad.Reconciliation.OK {
		t.Fatal("reconciliation still reports OK after a counter was made to drift by one; the check " +
			"is not reading the ring's accounting")
	}
	if c := moeReportCheck(t, bad, "lookups-identity"); c.OK {
		t.Fatalf("lookups-identity passed on a drifted ring: %s", c.Detail)
	}
	s.expertRing.hit-- // leave the ring honest for anything that follows
}

// TestMoEResidencyPlacementDriftFallsWhenThePinSetIsRepinned proves #3902's coverage/drift gauge is
// LIVE rather than a constant: it is pointed at a pin-set deliberately warm-started from a prior that
// names experts this window never routes, and the number must move the right way once the between-
// turns actuator repins against what actually happened.
func TestMoEResidencyPlacementDriftFallsWhenThePinSetIsRepinned(t *testing.T) {
	const H, E = 256, 6
	m := expertRingTestModel(t, H, E)
	x := expertRingTestInput(H)
	budget := expertRingWeightBytes(t, m) * 9

	// A stale prior: experts 4 and 5 were hot in some earlier life. This window routes 0 and 1.
	stale := NewExpertUsageHistogram()
	stale.Observe(0, 4, 100)
	stale.Observe(0, 5, 90)
	path := filepath.Join(t.TempDir(), "expert-usage.json")
	if err := stale.Persist(path); err != nil {
		t.Fatalf("persist prior: %v", err)
	}

	s := expertRingSession(m, budget)
	s.ExpertPinBudget = 2
	s.ExpertUsagePath = path
	window := make([]int, 0, 32)
	for i := 0; i < 16; i++ {
		window = append(window, 0, 1)
	}
	moeReportWindow(m, s, x, window)

	before := s.MoEResidency(MoEResidencyOptions{})
	if before.Placement.Basis != "pin-set" {
		t.Fatalf("placement basis %q, want pin-set: a declared plan must be scored in preference to "+
			"whatever recency happens to have left resident", before.Placement.Basis)
	}
	if before.Placement.BasisWidth != 2 {
		t.Fatalf("basis width %d, want the declared pin budget 2", before.Placement.BasisWidth)
	}
	if before.Placement.Drift != 1 {
		t.Fatalf("drift %.3f against a pin-set disjoint from every expert the window routed, want 1.0",
			before.Placement.Drift)
	}
	if before.Placement.Coverage != 0 {
		t.Fatalf("coverage %.3f: no touch was served by a pinned expert, so it must be 0",
			before.Placement.Coverage)
	}
	if before.Placement.ObservedTouches == 0 {
		t.Fatal("gauge scored a plan against zero observed touches; that is not evidence")
	}

	// The between-turns actuator (R2) repins against what this turn actually routed. The decay has to
	// be strong enough that a turn's real routing outweighs the prior's accumulated mass — at 1.0 the
	// stale pins are immortal, which is itself a tuning fact the drift number is how you'd notice.
	swaps, err := s.ExpertRingEndTurn(0.1, 2)
	if err != nil {
		t.Fatalf("ExpertRingEndTurn: %v", err)
	}
	if len(swaps) != 2 {
		t.Fatalf("repin made %d swaps over a turn that routed only unpinned experts, want 2: %+v",
			len(swaps), swaps)
	}
	after := s.MoEResidency(MoEResidencyOptions{})
	if after.Placement.Drift >= before.Placement.Drift {
		t.Fatalf("drift %.3f did not fall after repinning against the observed routing (was %.3f); "+
			"the gauge is not reading the live plan", after.Placement.Drift, before.Placement.Drift)
	}
	if after.Placement.Coverage <= before.Placement.Coverage {
		t.Fatalf("coverage %.3f did not rise after repinning (was %.3f)",
			after.Placement.Coverage, before.Placement.Coverage)
	}
	if after.Placement.Drift != 0 {
		t.Fatalf("drift %.3f after repinning onto exactly the two experts the window routed, want 0",
			after.Placement.Drift)
	}
}

// TestMoEResidencyReportDoesNotPerturbTheRingItObserves is the zero-overhead-when-off witness in its
// strongest honest form: the report is PULL-based, so polling it after every activation must leave
// both the outputs and the ring's own counters byte-identical to a window that was never polled.
// A meter that touched recency, booked a lookup or settled a fence would show up here.
//
// It also covers the ring-disabled default: a session with no budget reports Enabled=false and a
// reconciliation that says why there is nothing to reconcile, rather than a green check it did not earn.
func TestMoEResidencyReportDoesNotPerturbTheRingItObserves(t *testing.T) {
	const H, E = 256, 6
	m := expertRingTestModel(t, H, E)
	x := expertRingTestInput(H)
	budget := expertRingWeightBytes(t, m) * 6
	window := []int{0, 1, 2, 0, 3, 1, 4, 0}

	quiet := expertRingSession(m, budget)
	quietOut := moeReportWindow(m, quiet, x, window)

	polled := expertRingSession(m, budget)
	polledOut := make([][]float32, len(window))
	for i, e := range window {
		polledOut[i] = expertSwiGLU(m, 0, e, x, sessionQ4KKernel{s: polled})
		polled.MoEResidency(MoEResidencyOptions{Tokens: int64(i + 1), Regret: true})
	}

	for i := range window {
		if len(quietOut[i]) != len(polledOut[i]) {
			t.Fatalf("step %d: polled len=%d, quiet len=%d", i, len(polledOut[i]), len(quietOut[i]))
		}
		for j := range quietOut[i] {
			if quietOut[i][j] != polledOut[i][j] {
				t.Fatalf("step %d out[%d]=%v, unpolled=%v — reading a meter must not change the answer",
					i, j, polledOut[i][j], quietOut[i][j])
			}
		}
	}
	q, p := quiet.MoEResidency(MoEResidencyOptions{}), polled.MoEResidency(MoEResidencyOptions{})
	if q.Ring != p.Ring {
		t.Fatalf("polling changed the ring's ledger:\n polled %+v\n quiet  %+v", p.Ring, q.Ring)
	}

	// The default session: no budget, no ring, and an honest report rather than an invented one.
	off := expertRingSession(m, 0)
	moeReportWindow(m, off, x, window)
	if off.expertRing != nil {
		t.Fatal("a session with no declared budget built a ring; the ladder must be inert by default")
	}
	rep := off.MoEResidency(MoEResidencyOptions{Tokens: int64(len(window))})
	if rep.Ring.Enabled {
		t.Fatalf("report claims a ring on a session that has none: %+v", rep.Ring)
	}
	if !rep.Reconciliation.OK || len(rep.Reconciliation.Checks) != 1 {
		t.Fatalf("ringless reconciliation %+v, want exactly one check saying there is nothing to reconcile",
			rep.Reconciliation)
	}
	if c := moeReportCheck(t, rep, "ring-absent"); !c.OK {
		t.Fatalf("ring-absent check failed: %s", c.Detail)
	}
	if rep.Placement.Basis != "none" || rep.Placement.Reason == "" {
		t.Fatalf("placement %+v: with no ring there is no plan to score, and the report must say so",
			rep.Placement)
	}
}

// TestMoEResidencyReportsCoalescingAndReconcilesTheSharedLedger folds R7 into the operator view. The
// two shared-ring checks are the strongest in the report: the ledger's refusal count and page-in
// bytes are booked by the shared ring's hooks and the ring's by the ring, in different files, so
// agreement is evidence rather than a restatement.
func TestMoEResidencyReportsCoalescingAndReconcilesTheSharedLedger(t *testing.T) {
	const H, E, B = 256, 6, 3
	m := expertRingTestModel(t, H, E)
	x := expertRingTestInput(H)
	budget := expertRingWeightBytes(t, m) * 12

	be := &sharedRingBackend{Backend: compute.Default()}
	sh, err := NewSharedExpertRing(SharedExpertRingConfig{Model: m, Backend: be, BudgetBytes: budget})
	if err != nil {
		t.Fatalf("NewSharedExpertRing: %v", err)
	}
	agents := make([]*Session, B)
	for a := range agents {
		agents[a] = sharedRingAgent(m, be)
		if err := sh.Attach(agents[a], agentName(a)); err != nil {
			t.Fatalf("attach %d: %v", a, err)
		}
	}
	// A shared hot core plus one private expert each — the overlap #5243's lever assumes.
	windows := [B][]int{{0, 1, 2, 3, 0, 1}, {0, 1, 2, 4, 0, 1}, {0, 1, 2, 5, 0, 1}}
	for step := 0; step < len(windows[0]); step++ {
		for a := 0; a < B; a++ {
			expertSwiGLU(m, 0, windows[a][step], x, sessionQ4KKernel{s: agents[a]})
			sh.NoteAgentToken()
		}
	}

	rep := agents[0].MoEResidency(MoEResidencyOptions{Tokens: int64(B * len(windows[0]))})
	if rep.Shared == nil {
		t.Fatal("report from a session on a shared ring carries no coalescing ledger")
	}
	if !rep.Reconciliation.OK {
		t.Fatalf("shared reconciliation failed: %+v", rep.Reconciliation.Checks)
	}
	for _, name := range []string{"shared-refusals-agree", "shared-page-in-bytes-agree", "shared-demands-bounded", "shared-serves-bounded"} {
		if c := moeReportCheck(t, rep, name); !c.OK {
			t.Fatalf("%s failed: %s", name, c.Detail)
		}
	}
	if rep.Shared.Agents != B {
		t.Fatalf("report says %d agents attached, want %d", rep.Shared.Agents, B)
	}
	// The per-session view under a shared ring is the AGGREGATE bound, which is what actually bounds
	// this session's experts — so the two must be the same object's numbers.
	if rep.Ring != rep.Shared.Ring {
		t.Fatalf("report's ring %+v differs from the shared ring's %+v", rep.Ring, rep.Shared.Ring)
	}
	if rep.Rates.AgentsPerPageIn <= 1 {
		t.Fatalf("agents per page-in %.3f, want > 1: at 1.0 every page-in served only the agent that "+
			"paid for it, which is B private rings wearing one name", rep.Rates.AgentsPerPageIn)
	}
	if rep.Rates.CrossAgentHitRate <= 0 {
		t.Fatalf("cross-agent hit rate %.3f, want > 0", rep.Rates.CrossAgentHitRate)
	}
	for _, a := range agents {
		sh.Detach(a)
	}
	if err := sh.Close(); err != nil {
		t.Fatalf("Close after every agent detached: %v", err)
	}
}

// TestMoEResidencyRegretIsReportedOnlyWhenAsked pins the last stranded gauge (#4233) to the surface,
// and pins its cost model: the replay is O(trace) work an operator polling a live serve every second
// must not pay by default, so it is opt-in and absent otherwise.
func TestMoEResidencyRegretIsReportedOnlyWhenAsked(t *testing.T) {
	const H, E = 256, 6
	m := expertRingTestModel(t, H, E)
	x := expertRingTestInput(H)
	s := expertRingSession(m, expertRingWeightBytes(t, m)*6)
	moeReportWindow(m, s, x, []int{0, 1, 2, 3, 0, 1, 2, 4, 0, 1})

	if rep := s.MoEResidency(MoEResidencyOptions{}); rep.Regret != nil {
		t.Fatalf("regret replay ran without being asked for: %+v", rep.Regret)
	}
	rep := s.MoEResidency(MoEResidencyOptions{Regret: true})
	if rep.Regret == nil {
		t.Fatal("regret was asked for over a window with a replayable trace and none came back")
	}
	if rep.Regret.Reason == "" {
		t.Fatal("regret verdict carries no reason; a gate that only explains its promotions turns " +
			"every demotion into an unexplained default")
	}
	if rep.Regret.LRUGoodDecisionRatio < 0 || rep.Regret.LRUGoodDecisionRatio > 1 {
		t.Fatalf("LRU good-decision ratio %.3f is outside [0,1]", rep.Regret.LRUGoodDecisionRatio)
	}
}
