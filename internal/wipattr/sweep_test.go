package wipattr

import "testing"

func TestSweepGuardVocabulary(t *testing.T) {
	self := "me"
	live := map[string]bool{"peerLive": true} // peerLive is live; peerDead is not
	attrs := []Attribution{
		{File: "mine.go", State: AttrOwned, Owner: "me"},
		{File: "live.go", State: AttrOwned, Owner: "peerLive"},
		{File: "dead.go", State: AttrOwned, Owner: "peerDead"},
		{File: "shared.go", State: AttrShared, Owners: []string{"me", "peerLive"}},
		{File: "orphan.go", State: AttrOrphan},
	}
	got := SweepGuard(attrs, self, live)
	if len(got) != len(attrs) {
		t.Fatalf("totality broken: %d verdicts for %d attrs", len(got), len(attrs))
	}
	want := map[string]SweepRisk{
		"mine.go":   SweepSafe,
		"live.go":   SweepHazard,
		"dead.go":   SweepHazard,
		"shared.go": SweepHazard,
		"orphan.go": SweepHazard,
	}
	for _, v := range got {
		if v.Risk != want[v.File] {
			t.Errorf("%s: want %s, got %s (%s)", v.File, want[v.File], v.Risk, v.Reason)
		}
	}
}

// TestSweepSafeOnlyForSelf is the load-bearing invariant: across the whole state space,
// SAFE is emitted ONLY for a hunk OWNED solely by self. Nothing peer-owned, shared, or
// orphaned is ever cleared to sweep.
func TestSweepSafeOnlyForSelf(t *testing.T) {
	self := "me"
	states := []Attribution{
		{State: AttrOwned, Owner: "me"},
		{State: AttrOwned, Owner: "other"},
		{State: AttrShared, Owners: []string{"me", "other"}},
		{State: AttrShared, Owners: []string{"me"}}, // degenerate shared
		{State: AttrOrphan},
	}
	for _, liveOther := range []bool{false, true} {
		live := map[string]bool{}
		if liveOther {
			live["other"] = true
		}
		for _, a := range states {
			a.File = "f"
			v := SweepGuard([]Attribution{a}, self, live)[0]
			if v.Risk == SweepSafe && !(a.State == AttrOwned && a.Owner == self) {
				t.Fatalf("SAFETY VIOLATION: cleared a non-self hunk to sweep: state=%s owner=%q owners=%v", a.State, a.Owner, a.Owners)
			}
		}
	}
}

// TestSweepEmptySelfIsNeverSafe: when the guard cannot resolve a self id, nothing is
// SAFE — an unknown "me" must not silently clear a peer's OWNED hunk.
func TestSweepEmptySelfIsNeverSafe(t *testing.T) {
	attrs := []Attribution{{File: "x.go", State: AttrOwned, Owner: ""}}
	if v := SweepGuard(attrs, "", nil)[0]; v.Risk == SweepSafe {
		t.Fatalf("empty self cleared an empty-owner hunk to SAFE: %+v", v)
	}
}

func TestSweepHazardsFilter(t *testing.T) {
	vs := []SweepVerdict{
		{File: "a", Risk: SweepSafe},
		{File: "b", Risk: SweepHazard},
		{File: "c", Risk: SweepHazard},
	}
	h := SweepHazards(vs)
	if len(h) != 2 || h[0].File != "b" || h[1].File != "c" {
		t.Fatalf("hazard filter wrong: %+v", h)
	}
	if got := SweepHazards(nil); got == nil {
		t.Errorf("want non-nil empty slice for nil input")
	}
}

// TestSweepLivenessSharpensReason: a live peer and a crashed peer are both HAZARD but
// carry distinct reasons, so the operator can tell "reconcile" from "hands off".
func TestSweepLivenessSharpensReason(t *testing.T) {
	attrs := []Attribution{
		{File: "live.go", State: AttrOwned, Owner: "p"},
		{File: "dead.go", State: AttrOwned, Owner: "q"},
	}
	got := SweepGuard(attrs, "me", map[string]bool{"p": true})
	var liveReason, deadReason string
	for _, v := range got {
		switch v.File {
		case "live.go":
			liveReason = v.Reason
		case "dead.go":
			deadReason = v.Reason
		}
	}
	if liveReason == deadReason {
		t.Fatalf("live and crashed peer reasons should differ: %q", liveReason)
	}
	if want := "LIVE"; !contains(liveReason, want) {
		t.Errorf("live-peer reason %q should mention %q", liveReason, want)
	}
}

func contains(s, sub string) bool {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}
