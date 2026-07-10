package dispatchaging

import (
	"encoding/json"
	"sort"
	"strings"
	"testing"
)

// base is a fixed "now" (unix seconds) the tests reason against; ready times are offsets below it.
const base = int64(1_000_000)

// P-tier weights mirror dispatchtick.PriorityWeight so the fixtures read like the real taxonomy.
const (
	wP0      = 1000
	wP1      = 400
	wP2      = 150
	wDefault = 60
)

// readyAgo returns the ready-since stamp for a unit that has been waiting `secs` seconds at base.
func readyAgo(secs int64) int64 { return base - secs }

// --- standing verdicts and boost math -------------------------------------------------

func TestStandingAndBoost(t *testing.T) {
	p := DefaultParams(base) // interval 600s, +60/interval, uncapped, starve at 6h
	cases := []struct {
		name      string
		waitSecs  int64
		wantStand Standing
		wantBoost int
	}{
		{"just ready, under one interval", 300, StandingFresh, 0},
		{"exactly one interval", 600, StandingAging, 60},
		{"two and a half intervals floors to two", 1500, StandingAging, 120},
		{"one hour", 3600, StandingAging, 360},
		{"just under the hard deadline", 6*3600 - 1, StandingAging, 2100},
		{"at the hard deadline is starved", 6 * 3600, StandingStarved, 2160},
		{"well past the deadline stays starved", 24 * 3600, StandingStarved, 8640},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			r := Fold([]Candidate{{ID: "x", BaseWeight: wDefault, ReadySince: readyAgo(tc.waitSecs)}}, p)
			got := r.Order[0]
			if got.Standing != tc.wantStand {
				t.Errorf("standing = %q, want %q", got.Standing, tc.wantStand)
			}
			if got.AgingBoost != tc.wantBoost {
				t.Errorf("boost = %d, want %d", got.AgingBoost, tc.wantBoost)
			}
			if got.EffectiveWeight != wDefault+tc.wantBoost {
				t.Errorf("effective = %d, want %d", got.EffectiveWeight, wDefault+tc.wantBoost)
			}
		})
	}
}

// --- the core anti-starvation property: soft aging lets a light unit overtake a fresh heavy one ---

func TestSoftAgingOvertakesFreshHigherPriority(t *testing.T) {
	p := DefaultParams(base)
	// A default-weight (60) unit that has waited 2h has boost +720 => effective 780, which beats a
	// FRESH P1 (400) but not a fresh P0 (1000). This is the whole point: priority no longer leads
	// absolutely; long waiting overtakes a fixed lighter tier.
	cands := []Candidate{
		{ID: "fresh-p0", BaseWeight: wP0, ReadySince: readyAgo(10)},
		{ID: "fresh-p1", BaseWeight: wP1, ReadySince: readyAgo(10)},
		{ID: "aged-default", BaseWeight: wDefault, ReadySince: readyAgo(2 * 3600)},
	}
	got := Fold(cands, p).PickOrder
	want := []string{"fresh-p0", "aged-default", "fresh-p1"}
	assertOrder(t, got, want)
}

// --- the hard guarantee: past the deadline, a light unit is served before even a fresh P0 -------

func TestStarvedBeatsEverything(t *testing.T) {
	p := DefaultParams(base)
	cands := []Candidate{
		{ID: "fresh-p0-a", BaseWeight: wP0, ReadySince: readyAgo(5)},
		{ID: "starved-default", BaseWeight: wDefault, ReadySince: readyAgo(7 * 3600)}, // past 6h deadline
		{ID: "fresh-p0-b", BaseWeight: wP0, ReadySince: readyAgo(5)},
	}
	r := Fold(cands, p)
	if r.Pick() != "starved-default" {
		t.Fatalf("Pick = %q, want the starved unit to be served first", r.Pick())
	}
	if r.StarvedCount != 1 {
		t.Errorf("StarvedCount = %d, want 1", r.StarvedCount)
	}
}

// Multiple starved units serve worst-starved (longest wait) first, ahead of all non-starved.
func TestStarvedOrderedByWorstFirst(t *testing.T) {
	p := DefaultParams(base)
	cands := []Candidate{
		{ID: "fresh-p0", BaseWeight: wP0, ReadySince: readyAgo(5)},
		{ID: "starved-8h", BaseWeight: wDefault, ReadySince: readyAgo(8 * 3600)},
		{ID: "starved-10h", BaseWeight: wP2, ReadySince: readyAgo(10 * 3600)},
	}
	got := Fold(cands, p).PickOrder
	// 10h starves worse than 8h even though 8h's unit could carry a different weight; both precede P0.
	assertOrder(t, got, []string{"starved-10h", "starved-8h", "fresh-p0"})
}

// --- additive no-regression: aging OFF reproduces the priority-then-oldest-first order ----------

func TestAgingDisabledIsPreAgingOrder(t *testing.T) {
	// Zero-value aging knobs (only a real clock) disable both soft aging and the hard deadline.
	p := Params{NowUnix: base}
	cands := []Candidate{
		{ID: "d-old", BaseWeight: wDefault, ReadySince: readyAgo(9 * 3600)}, // would starve if aging were on
		{ID: "p1-new", BaseWeight: wP1, ReadySince: readyAgo(30)},
		{ID: "p1-old", BaseWeight: wP1, ReadySince: readyAgo(3600)},
		{ID: "p0", BaseWeight: wP0, ReadySince: readyAgo(120)},
		{ID: "d-new", BaseWeight: wDefault, ReadySince: readyAgo(15)},
	}
	r := Fold(cands, p)

	// No unit is boosted or starved when aging is off — that is the no-regression contract.
	for _, u := range r.Order {
		if u.AgingBoost != 0 || u.EffectiveWeight != u.BaseWeight || u.Standing != StandingFresh {
			t.Fatalf("aging-off leaked a boost: %+v", u)
		}
	}

	// The order must equal the pre-aging rule: base weight desc, then longer wait (oldest) first,
	// then ID — computed here independently as the reference.
	ref := append([]Candidate(nil), cands...)
	sort.SliceStable(ref, func(i, j int) bool {
		if ref[i].BaseWeight != ref[j].BaseWeight {
			return ref[i].BaseWeight > ref[j].BaseWeight
		}
		wi, wj := base-ref[i].ReadySince, base-ref[j].ReadySince
		if wi != wj {
			return wi > wj
		}
		return ref[i].ID < ref[j].ID
	})
	want := make([]string, len(ref))
	for i, c := range ref {
		want[i] = c.ID
	}
	assertOrder(t, r.PickOrder, want)
}

// --- edge cases: empty, unknown/absent ready time, clock skew, ranks, census --------------------

func TestEmptyInput(t *testing.T) {
	r := Fold(nil, DefaultParams(base))
	if len(r.Order) != 0 || len(r.PickOrder) != 0 || r.Pick() != "" {
		t.Fatalf("empty input should yield an empty result, got %+v", r)
	}
	if r.Schema != Schema {
		t.Errorf("schema = %q, want %q", r.Schema, Schema)
	}
}

func TestUnknownReadySinceWaitsZeroNeverStarves(t *testing.T) {
	p := DefaultParams(base)
	cands := []Candidate{
		{ID: "unknown", BaseWeight: wDefault, ReadySince: 0},          // unknown -> wait 0
		{ID: "skewed", BaseWeight: wDefault, ReadySince: base + 5000}, // ready in the future -> wait 0
	}
	r := Fold(cands, p)
	for _, u := range r.Order {
		if u.WaitSeconds != 0 || u.Standing != StandingFresh {
			t.Errorf("%s: want wait 0 / fresh, got wait=%d standing=%s", u.ID, u.WaitSeconds, u.Standing)
		}
	}
	if r.StarvedCount != 0 {
		t.Errorf("StarvedCount = %d, want 0 (unknown/future ready time never starves)", r.StarvedCount)
	}
}

func TestRanksAndCensus(t *testing.T) {
	p := DefaultParams(base)
	cands := []Candidate{
		{ID: "starved", BaseWeight: wDefault, ReadySince: readyAgo(7 * 3600)},
		{ID: "aging", BaseWeight: wP2, ReadySince: readyAgo(1200)},
		{ID: "fresh", BaseWeight: wP0, ReadySince: readyAgo(10)},
	}
	r := Fold(cands, p)
	if r.StarvedCount != 1 || r.AgingCount != 1 || r.FreshCount != 1 {
		t.Errorf("census = starved %d / aging %d / fresh %d, want 1/1/1",
			r.StarvedCount, r.AgingCount, r.FreshCount)
	}
	for i, u := range r.Order {
		if u.Rank != i {
			t.Errorf("Order[%d] has Rank %d, want %d", i, u.Rank, i)
		}
	}
	if r.OldestWaitSeconds != 7*3600 {
		t.Errorf("OldestWaitSeconds = %d, want %d", r.OldestWaitSeconds, int64(7*3600))
	}
}

// --- MaxBoostPoints caps the soft boost -------------------------------------------------------

func TestMaxBoostCap(t *testing.T) {
	p := DefaultParams(base)
	p.MaxBoostPoints = 100 // cap soft aging at +100 regardless of wait
	r := Fold([]Candidate{{ID: "x", BaseWeight: wDefault, ReadySince: readyAgo(5 * 3600)}}, p)
	if r.Order[0].AgingBoost != 100 {
		t.Errorf("capped boost = %d, want 100", r.Order[0].AgingBoost)
	}
	// The hard deadline is independent of the soft cap: a unit past the deadline still starves.
	r2 := Fold([]Candidate{{ID: "y", BaseWeight: wDefault, ReadySince: readyAgo(7 * 3600)}}, p)
	if r2.Order[0].Standing != StandingStarved {
		t.Errorf("standing = %s, want starved even with a soft cap", r2.Order[0].Standing)
	}
}

// --- cooldown pauses the wait clock (pause, not reset) -----------------------------------------

// While a unit is cooling, its eligible wait must read the SAME at every instant of the window:
// the clock is paused, so cooling time can never grow the boost or the standing.
func TestCoolingPausesWaitClock(t *testing.T) {
	ready := readyAgo(3600) // became ready 1h before base
	coolStart := base - 600 // attempted 10 min before base...
	coolEnd := base + 600   // ...cooling for 20 min total
	c := Candidate{ID: "cooling", BaseWeight: wDefault, ReadySince: ready,
		CoolingSince: coolStart, CoolingUntil: coolEnd}
	const wantPaused = int64(3000) // the 50 eligible minutes accrued before the cooldown began
	for _, now := range []int64{coolStart, base, coolEnd - 1, coolEnd} {
		u := Fold([]Candidate{c}, DefaultParams(now)).Order[0]
		if u.WaitSeconds != wantPaused {
			t.Errorf("now=%d: WaitSeconds = %d, want %d (clock paused while cooling)",
				now, u.WaitSeconds, wantPaused)
		}
		if u.Standing != StandingAging || u.AgingBoost != 300 { // 3000s = 5 intervals * 60
			t.Errorf("now=%d: standing/boost = %s/+%d, want aging/+300 (frozen while cooling)",
				now, u.Standing, u.AgingBoost)
		}
	}
}

// Cooling time ALONE can never starve a unit: attempted the moment it became ready, then cooled
// for wall time far past the hard deadline — zero eligible seconds, so it stays fresh.
func TestCoolingTimeAloneNeverStarves(t *testing.T) {
	ready := readyAgo(10 * 3600) // 10h of wall wait, all of it inside the cooldown
	c := Candidate{ID: "flaky", BaseWeight: wDefault, ReadySince: ready,
		CoolingSince: ready, CoolingUntil: base}
	r := Fold([]Candidate{c}, DefaultParams(base))
	u := r.Order[0]
	if u.WaitSeconds != 0 || u.Standing != StandingFresh || u.AgingBoost != 0 {
		t.Errorf("wait/standing/boost = %d/%s/+%d, want 0/fresh/+0 (cooling accrues no pressure)",
			u.WaitSeconds, u.Standing, u.AgingBoost)
	}
	if r.StarvedCount != 0 {
		t.Errorf("StarvedCount = %d, want 0 (cooling time alone never starves)", r.StarvedCount)
	}
}

// Once cooled, the unit resumes accruing wait from where it paused — the cooled span is excluded
// forever after, so there is no phantom jump the moment the window ends.
func TestCooledResumesFromPausedWait(t *testing.T) {
	ready := readyAgo(3000) // 30 eligible minutes, then a 20-min cooldown ending at base
	c := Candidate{ID: "cooled", BaseWeight: wDefault, ReadySince: ready,
		CoolingSince: base - 1200, CoolingUntil: base}
	if got := Fold([]Candidate{c}, DefaultParams(base)).Order[0].WaitSeconds; got != 1800 {
		t.Errorf("at cooldown end: WaitSeconds = %d, want 1800 (exactly where it paused)", got)
	}
	if got := Fold([]Candidate{c}, DefaultParams(base+600)).Order[0].WaitSeconds; got != 2400 {
		t.Errorf("10 min after cooling: WaitSeconds = %d, want 2400 (paused 1800 + 600 eligible)", got)
	}
}

// The zero/legacy input — no cooling info — is byte-identical to the pre-cooling fold: the wait
// clock is the plain now-ReadySince, and the encoded artifact carries no cooling keys (omitempty).
func TestNoCoolingInfoIsLegacyFold(t *testing.T) {
	cands := []Candidate{
		{ID: "a", BaseWeight: wP1, ReadySince: readyAgo(3600)},
		{ID: "b", BaseWeight: wDefault, ReadySince: readyAgo(7 * 3600)},
		{ID: "c", BaseWeight: wP0, ReadySince: readyAgo(10)},
	}
	r := Fold(cands, DefaultParams(base))
	for _, u := range r.Order {
		if u.WaitSeconds != base-u.ReadySince {
			t.Errorf("%s: WaitSeconds = %d, want legacy now-ReadySince = %d",
				u.ID, u.WaitSeconds, base-u.ReadySince)
		}
	}
	b, err := json.Marshal(r)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if strings.Contains(string(b), "cooling") {
		t.Errorf("legacy fold leaked cooling keys into the encoded Result: %s", b)
	}
}

// Window edges: irrelevant windows are no-ops, half-declared windows fail closed, and the
// subtraction clips to the waited span.
func TestCoolingWindowEdges(t *testing.T) {
	cases := []struct {
		name string
		c    Candidate
		want int64
	}{
		{"window entirely before ready is a no-op",
			Candidate{ReadySince: readyAgo(600), CoolingSince: readyAgo(900), CoolingUntil: readyAgo(700)}, 600},
		{"window entirely in the future is a no-op",
			Candidate{ReadySince: readyAgo(600), CoolingSince: base + 100, CoolingUntil: base + 700}, 600},
		{"window straddling ready clips at ready",
			Candidate{ReadySince: readyAgo(600), CoolingSince: readyAgo(900), CoolingUntil: readyAgo(300)}, 300},
		{"end without a start cools the whole wait so far (fail-closed)",
			Candidate{ReadySince: readyAgo(600), CoolingUntil: base + 100}, 0},
		{"start without a declared end is no window",
			Candidate{ReadySince: readyAgo(600), CoolingSince: readyAgo(300)}, 600},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			tc.c.ID = "x"
			tc.c.BaseWeight = wDefault
			got := Fold([]Candidate{tc.c}, DefaultParams(base)).Order[0].WaitSeconds
			if got != tc.want {
				t.Errorf("WaitSeconds = %d, want %d", got, tc.want)
			}
		})
	}
}

// determinism: equal-standing, equal-weight, equal-wait ties break by ID, and Fold is stable.
func TestDeterministicIDTiebreak(t *testing.T) {
	p := Params{NowUnix: base} // aging off, so all fresh/equal-wait when ready times match
	cands := []Candidate{
		{ID: "c", BaseWeight: wP1, ReadySince: readyAgo(100)},
		{ID: "a", BaseWeight: wP1, ReadySince: readyAgo(100)},
		{ID: "b", BaseWeight: wP1, ReadySince: readyAgo(100)},
	}
	assertOrder(t, Fold(cands, p).PickOrder, []string{"a", "b", "c"})
}

func assertOrder(t *testing.T, got, want []string) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("order length = %d, want %d (got %v)", len(got), len(want), got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("order = %v, want %v", got, want)
		}
	}
}
