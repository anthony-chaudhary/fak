package superloop

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/scoreboard"
)

// reportFeederRoster is the reporting family this intent must walk: the Slack feeders
// that fold onto the ONE CI/CD report channel (scoreboard.CICDReportChannel), keyed by
// registry ref and carrying the revival command each row's Enter hint must print.
//
// It is a hand-kept mirror (TODO(#4865): swap for the canonical roster source once it
// lands). Keeping the expectation HERE, spelled out, is the point: the test fails loudly
// when the registry drifts from the family rather than re-deriving the same list from the
// code under test and passing vacuously.
var reportFeederRoster = map[string]string{
	"report:product":    "fak slack refresh --surface product",
	"report:blockers":   "fak slack refresh --surface blockers",
	"report:cachevalue": "fak slack refresh --surface cachevalue",
	"report:bench":      "fak slack refresh --surface bench",
	"report:dojo":       "fak slack refresh --surface dojo",
	"report:backlog":    "fak slack refresh --surface backlog --backlog-channel " + scoreboard.CICDReportChannel,
	"report:node-usage": "fak slack refresh --surface node-usage",
	"report:steering":   "fak slack refresh --surface steering",
}

// TestTendReportingMirrorsTheReportChannelFamily is the roster no-drift witness (#4863):
// tend-reporting walks EVERY reporting-family feeder and nothing else, every member is a
// liveness-bearing loop with a runnable revival hint, and the intent is rooted so the
// `tend` walk can never miss it.
func TestTendReportingMirrorsTheReportChannelFamily(t *testing.T) {
	s, ok := Lookup("tend-reporting")
	if !ok {
		t.Fatal("tend-reporting not registered")
	}

	got := map[string]Member{}
	for _, m := range s.Members {
		if m.Kind != KindLoop {
			t.Errorf("feeder %q has kind %q, want %q — a feeder is a ledgered post cadence, and only a loop carries liveness debt", m.Ref, m.Kind, KindLoop)
		}
		if _, dup := got[m.Ref]; dup {
			t.Errorf("feeder %q is listed twice — its debt would fold twice inside one walk", m.Ref)
		}
		got[m.Ref] = m

		wantEnter, known := reportFeederRoster[m.Ref]
		if !known {
			t.Errorf("feeder %q is not in the reporting-family roster — either it does not fold onto the CI/CD report channel, or the roster is stale", m.Ref)
			continue
		}
		if m.Enter != wantEnter {
			t.Errorf("feeder %q Enter = %q, want %q — the worklist action must be runnable exactly as printed", m.Ref, m.Enter, wantEnter)
		}
		if strings.TrimSpace(m.Why) == "" {
			t.Errorf("feeder %q has no Why — a worklist row must say what going dark costs", m.Ref)
		}
		// A report cadence is neither gardening nor backlog-drain: a benign idle feed must
		// not be scored by the throughput mix, so it stays neutral.
		if c := classifyWork(m); c != WorkNeutral {
			t.Errorf("feeder %q classifies %q, want %q — a report cadence is not issue-drain work", m.Ref, c, WorkNeutral)
		}
	}
	for ref := range reportFeederRoster {
		if _, ok := got[ref]; !ok {
			t.Errorf("tend-reporting is missing reporting-family feeder %q — a feeder nothing walks can go dark unnoticed, which is the whole point of the intent", ref)
		}
	}

	// The feeder refs are the FEED, deliberately distinct from any same-named underlying
	// loop ("dojo" the gym loop vs "report:dojo" the rollup feed). No other intent may walk
	// one, or the root fold would count its debt twice.
	for _, other := range Registry() {
		if other.Name == "tend-reporting" {
			continue
		}
		for _, m := range other.Members {
			if _, isFeeder := reportFeederRoster[m.Ref]; isFeeder {
				t.Errorf("feeder %q is also walked by %q — the root fold would count its liveness debt twice", m.Ref, other.Name)
			}
		}
	}

	// Rooted THROUGH the fleet meta-walker (#4958): the reporting family is
	// re-parented as ONE KindSuperloop child of tend-fleet — a loop family the
	// generic meta-walk supervises through the identical SubwalkStatus fold — and
	// tend descends tend-fleet, so a dark feeder still surfaces at
	// `fak superloop walk tend` (now via the fleet meta-walk, counted once).
	fleet, ok := Lookup("tend-fleet")
	if !ok {
		t.Fatal("tend-fleet not registered")
	}
	var fleetDescends bool
	for _, m := range fleet.Members {
		if m.Kind == KindSuperloop && m.Ref == "tend-reporting" {
			fleetDescends = true
		}
	}
	if !fleetDescends {
		t.Error("tend-fleet must descend tend-reporting as its ONE reporting-family child (#4862 re-parent), else the feeder family escapes the fleet meta-walk")
	}
	tend, ok := Lookup("tend")
	if !ok {
		t.Fatal("tend not registered")
	}
	var rootDescends, direct bool
	for _, m := range tend.Members {
		if m.Kind == KindSuperloop && m.Ref == "tend-fleet" {
			rootDescends = true
		}
		if m.Kind == KindSuperloop && m.Ref == "tend-reporting" {
			direct = true
		}
	}
	if !rootDescends {
		t.Error("tend must descend tend-fleet, else the fleet meta-walk (and the reporting family beneath it) can go dark without the root walk noticing")
	}
	if direct {
		t.Error("tend still descends tend-reporting DIRECTLY — the #4862 re-parent moved it beneath tend-fleet; a second parent under the same root would fold its liveness debt twice")
	}
}

// TestTendReportingHoldsWhileFeederDark is the failure-class proof (#4863): the intent
// folds ONLY when every feeder is confirmed up. A dark feeder, or one whose liveness could
// not be read, must keep the walk unsatisfied and rank worst-first with its revival hint —
// an unread feed is never silently reported as clean.
func TestTendReportingHoldsWhileFeederDark(t *testing.T) {
	s, ok := Lookup("tend-reporting")
	if !ok {
		t.Fatal("tend-reporting not registered")
	}

	// live builds an all-clear status set, then applies one mutation to a single feeder.
	live := func(mutate func(ref string, st *MemberStatus)) []MemberStatus {
		out := make([]MemberStatus, 0, len(s.Members))
		for _, m := range s.Members {
			st := MemberStatus{Member: m, Measured: true, Debt: 0}
			if mutate != nil {
				mutate(m.Ref, &st)
			}
			out = append(out, st)
		}
		return out
	}

	t.Run("every feeder up folds", func(t *testing.T) {
		rep := Walk(s, live(nil))
		if !rep.Satisfied {
			t.Errorf("every feeder measured, live and at floor must fold: %s", rep.Reason)
		}
		if len(rep.Worklist) != 0 {
			t.Errorf("a fully-live family has nothing to revive, got %d worklist row(s)", len(rep.Worklist))
		}
	})

	t.Run("a dark feeder holds the fold and ranks first", func(t *testing.T) {
		rep := Walk(s, live(func(ref string, st *MemberStatus) {
			if ref == "report:dojo" {
				st.Dark = true
			}
		}))
		if rep.Satisfied {
			t.Error("a DARK feeder must hold the fold — a family with a silent feed is not tended")
		}
		if rep.Dark != 1 {
			t.Errorf("dark count = %d, want 1", rep.Dark)
		}
		if len(rep.Worklist) == 0 {
			t.Fatal("a dark feeder must be surfaced to revive")
		}
		head := rep.Worklist[0]
		if head.Member.Ref != "report:dojo" {
			t.Errorf("worst-first must enter the dark feeder, got %q", head.Member.Ref)
		}
		if !strings.Contains(head.Action, reportFeederRoster["report:dojo"]) {
			t.Errorf("the dark feeder's action must carry its runnable revival hint, got %q", head.Action)
		}
		if rep.Finding != "superloop_dark" {
			t.Errorf("finding = %q, want superloop_dark", rep.Finding)
		}
	})

	t.Run("an unread feeder holds the fold as UNMEASURED, never clean", func(t *testing.T) {
		rep := Walk(s, live(func(ref string, st *MemberStatus) {
			if ref == "report:blockers" {
				st.Measured = false
			}
		}))
		if rep.Satisfied {
			t.Error("an UNMEASURED feeder must hold the fold — liveness we could not read is not liveness we have")
		}
		if rep.Unmeasured != 1 {
			t.Errorf("unmeasured count = %d, want 1", rep.Unmeasured)
		}
		if len(rep.Worklist) == 0 || rep.Worklist[0].Member.Ref != "report:blockers" {
			t.Fatalf("worst-first must enter the unread feeder, got %+v", rep.Worklist)
		}
		if !strings.Contains(rep.Worklist[0].Detail, "UNMEASURED") {
			t.Errorf("an unread feeder must read UNMEASURED, got %q", rep.Worklist[0].Detail)
		}
	})
}
