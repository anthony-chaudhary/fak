package rehydrate

import (
	"context"
	"sort"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/dormancy"
)

// fullLadder builds a Gate of all five canonical rungs. Each rung records that it ran (by
// appending its reason to *ran) and clears, except a reason listed in refuse, which refuses.
// It is the test double for the four children that have not landed yet (#1182-#1186): the
// orchestrator composes rungs, so its acceptance is proven with deterministic rung doubles.
func fullLadder(ran *[]Reason, refuse map[Reason]bool) *Gate {
	mk := func(reason Reason) Rung {
		return NewRung(reason, func(context.Context) Verdict {
			*ran = append(*ran, reason)
			if refuse[reason] {
				return Refuse(reason, "test refusal")
			}
			return Clear()
		})
	}
	return NewGate(
		mk(ColdCache), mk(StaleCred), mk(StaleRecall), mk(StaleLease), mk(StalePlan),
	)
}

func sortedReasons(rs []Reason) []Reason {
	out := append([]Reason(nil), rs...)
	sort.Slice(out, func(i, j int) bool { return out[i] < out[j] })
	return out
}

func eqReasons(a, b []Reason) bool {
	if len(a) != len(b) {
		return false
	}
	a, b = sortedReasons(a), sortedReasons(b)
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// TestWarmRestoreRunsZeroRungs is half the #1181 acceptance: a warm restore runs no extra
// rungs and is admitted unconditionally — the resume-verbatim path is untouched.
func TestWarmRestoreRunsZeroRungs(t *testing.T) {
	var ran []Reason
	g := fullLadder(&ran, nil)
	adm := g.Admit(context.Background(), dormancy.Warm)
	if !adm.Admitted {
		t.Fatalf("warm restore not admitted: %+v", adm)
	}
	if len(adm.Ran) != 0 || len(ran) != 0 {
		t.Fatalf("warm restore ran rungs: Ran=%v sideEffects=%v", adm.RanReasons(), ran)
	}
}

// TestFrozenRestoreRunsAllRungs is the other half: a frozen restore runs the
// cred/lease/recall/cache/plan rungs and, when each clears, admits.
func TestFrozenRestoreRunsAllRungs(t *testing.T) {
	var ran []Reason
	g := fullLadder(&ran, nil)
	adm := g.Admit(context.Background(), dormancy.Frozen)
	if !adm.Admitted {
		t.Fatalf("frozen restore with all-clearing rungs not admitted: %+v", adm)
	}
	want := []Reason{ColdCache, StaleCred, StaleRecall, StaleLease, StalePlan}
	if !eqReasons(adm.RanReasons(), want) {
		t.Fatalf("frozen restore ran %v, want all five %v", adm.RanReasons(), want)
	}
}

// TestRefusesAdmissionUntilEachClears proves the gate refuses while any applicable rung is
// unclear and admits once it clears — "refuses admission until each clears."
func TestRefusesAdmissionUntilEachClears(t *testing.T) {
	// A frozen restore where STALE_LEASE refuses: admission is refused, named by that reason.
	var ran []Reason
	g := fullLadder(&ran, map[Reason]bool{StaleLease: true})
	adm := g.Admit(context.Background(), dormancy.Frozen)
	if adm.Admitted {
		t.Fatalf("admitted despite a refusing rung: %+v", adm)
	}
	if adm.RefusedBy != StaleLease {
		t.Fatalf("RefusedBy = %q, want %q", adm.RefusedBy, StaleLease)
	}
	if adm.Detail == "" {
		t.Fatal("refusal carried no detail")
	}
	// Short-circuit: a rung after the refuser (STALE_PLAN fires at the same band but sorts
	// after STALE_LEASE) must NOT have run.
	for _, r := range adm.RanReasons() {
		if r == StalePlan {
			t.Fatalf("rung after the refuser still ran: %v", adm.RanReasons())
		}
	}

	// Once the rung clears, the same frozen restore is admitted.
	var ran2 []Reason
	g2 := fullLadder(&ran2, nil)
	adm2 := g2.Admit(context.Background(), dormancy.Frozen)
	if !adm2.Admitted {
		t.Fatalf("not admitted once the rung clears: %+v", adm2)
	}
}

// TestStagingIsMonotonic proves the load-bearing property: a longer gap runs strictly more
// rungs (warm 0 < cool 1 < cold 3 < frozen 5), never fewer.
func TestStagingIsMonotonic(t *testing.T) {
	cases := []struct {
		h    dormancy.Horizon
		want int
	}{
		{dormancy.Warm, 0},
		{dormancy.Cool, 1},
		{dormancy.Cold, 3},
		{dormancy.Frozen, 5},
		{dormancy.Ancient, 5}, // ancient currently runs the same five (full revalidation)
	}
	prev := -1
	for _, c := range cases {
		var ran []Reason
		g := fullLadder(&ran, nil)
		adm := g.Admit(context.Background(), c.h)
		if len(adm.Ran) != c.want {
			t.Fatalf("band %s ran %d rungs (%v), want %d", c.h, len(adm.Ran), adm.RanReasons(), c.want)
		}
		if len(adm.Ran) < prev {
			t.Fatalf("band %s ran FEWER rungs than a shorter gap (%d < %d)", c.h, len(adm.Ran), prev)
		}
		prev = len(adm.Ran)
	}
}

// TestCanonicalLadderBands pins the staging policy: each reason fires at the band where the
// thing it guards first decays (the dormancy band semantics).
func TestCanonicalLadderBands(t *testing.T) {
	want := map[Reason]dormancy.Horizon{
		ColdCache:   dormancy.Cool,
		StaleCred:   dormancy.Cold,
		StaleRecall: dormancy.Cold,
		StaleLease:  dormancy.Frozen,
		StalePlan:   dormancy.Frozen,
	}
	for r, h := range want {
		if got := CanonicalFiresAt(r); got != h {
			t.Errorf("CanonicalFiresAt(%q) = %s, want %s", r, got, h)
		}
		if !r.Known() {
			t.Errorf("%q should be a known reason", r)
		}
	}
	// An unknown reason fails closed to Ancient and is dropped at gate construction.
	bogus := Reason("STALE_BOGUS")
	if bogus.Known() {
		t.Fatal("bogus reason reported Known")
	}
	if got := CanonicalFiresAt(bogus); got != dormancy.Ancient {
		t.Fatalf("unknown reason fired at %s, want Ancient (fail-closed)", got)
	}
	var ran []Reason
	g := NewGate(NewRung(bogus, func(context.Context) Verdict { ran = append(ran, bogus); return Clear() }))
	adm := g.Admit(context.Background(), dormancy.Ancient)
	if len(adm.Ran) != 0 || len(ran) != 0 {
		t.Fatalf("gate ran an unknown-reason rung: %v", adm.RanReasons())
	}
}

// TestNilGateAdmits proves a nil Gate is the unconditional-resume default (the no-staging
// case the sessionimage wire-in relies on).
func TestNilGateAdmits(t *testing.T) {
	var g *Gate
	adm := g.Admit(context.Background(), dormancy.Ancient)
	if !adm.Admitted || len(adm.Ran) != 0 {
		t.Fatalf("nil gate not an unconditional admit: %+v", adm)
	}
}
