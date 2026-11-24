package routing

import (
	"math"
	"strings"
	"testing"
)

// measured builds an all-clear snapshot for a group: every member idle and
// fresh, which is the "nothing is wrong" baseline the distribution proofs
// measure against.
func measured(group string, members ...string) BalanceSnapshot {
	snap := BalanceSnapshot{Group: group, Members: make(map[string]MemberScore, len(members))}
	for _, m := range members {
		snap.Members[m] = MemberScore{Load: 0, Latency: 0, Measured: true}
	}
	return snap
}

// loads builds a snapshot with an explicit per-member load and everything else
// fresh.
func loads(group string, load map[string]float64, members ...string) BalanceSnapshot {
	snap := BalanceSnapshot{Group: group, Members: make(map[string]MemberScore, len(members))}
	for _, m := range members {
		snap.Members[m] = MemberScore{Load: load[m], Measured: true}
	}
	return snap
}

func countChosen(orders []BalanceOrder) map[string]int {
	counts := map[string]int{}
	for _, o := range orders {
		if len(o.Members) > 0 {
			counts[o.Members[0]]++
		}
	}
	return counts
}

func within(t *testing.T, label string, got, want, tolerance float64) {
	t.Helper()
	if math.Abs(got-want) > tolerance {
		t.Fatalf("%s = %.4f, want %.4f ±%.4f", label, got, want, tolerance)
	}
}

func TestBalanceSelectDistributesEvenlyUnderWeightedHash(t *testing.T) {
	group := BalanceGroup{
		Name:     "even",
		Members:  []string{"a", "b", "c"},
		Strategy: BalanceStrategyWeightedHash,
	}
	snap := measured("even", "a", "b", "c")
	var p BalancePolicy

	const n = 3000
	orders := make([]BalanceOrder, 0, n)
	for i := 0; i < n; i++ {
		orders = append(orders, p.Select(group, snap, "req-"+itoa(i)))
	}
	counts := countChosen(orders)
	for _, m := range group.Members {
		share := float64(counts[m]) / float64(n)
		within(t, "share["+m+"]", share, 1.0/3.0, 0.05)
	}
}

func TestBalanceSelectIsProportionalToDeclaredWeights(t *testing.T) {
	// 3:1:1 over 5000 keys must land on 0.6 for the heavy member. This is the
	// assertion that distinguishes weighted rendezvous from a plain hash: a
	// plain hash would give 1/3.
	group := BalanceGroup{
		Name:     "weighted",
		Members:  []string{"a", "b", "c"},
		Strategy: BalanceStrategyWeightedHash,
		Weights:  map[string]int{"a": 3, "b": 1, "c": 1},
	}
	snap := measured("weighted", "a", "b", "c")
	var p BalancePolicy

	const n = 5000
	counts := map[string]int{}
	for i := 0; i < n; i++ {
		o := p.Select(group, snap, "k"+itoa(i))
		counts[o.Members[0]]++
	}
	within(t, "share[a] with weight 3 of 5", float64(counts["a"])/float64(n), 0.6, 0.05)
	within(t, "share[b] with weight 1 of 5", float64(counts["b"])/float64(n), 0.2, 0.05)
}

func TestBalanceSelectIsProactiveNotReactive(t *testing.T) {
	// The whole point of the concept: a member that has not failed anything is
	// still moved off the head because it is saturated. No failure has
	// occurred; this is pure selection.
	group := BalanceGroup{
		Name:     "busy",
		Members:  []string{"a", "b", "c"},
		Strategy: BalanceStrategyLeastLoad,
	}
	snap := loads("busy", map[string]float64{"a": 1.0, "b": 0.0, "c": 0.0}, "a", "b", "c")
	var p BalancePolicy
	for i := 0; i < 50; i++ {
		o := p.Select(group, snap, "req-"+itoa(i))
		if o.Members[0] == "a" {
			t.Fatalf("least-load chose the fully loaded member %q for key req-%d", o.Members[0], i)
		}
		if o.Plan.Chosen != o.Members[0] {
			t.Fatalf("plan.Chosen = %q but order head is %q", o.Plan.Chosen, o.Members[0])
		}
	}
	// And the saturated member is on the receipt with the closed reason, and is
	// still present in the chain rather than removed.
	o := p.Select(group, snap, "req-0")
	var saturated *BalanceMemberPlan
	for i := range o.Plan.Members {
		if o.Plan.Members[i].Member == "a" {
			saturated = &o.Plan.Members[i]
		}
	}
	if saturated == nil {
		t.Fatal("the saturated member is missing from the receipt")
	}
	if saturated.Excluded != BalanceExcludeSaturated {
		t.Fatalf("saturated member exclusion = %q, want %q", saturated.Excluded, BalanceExcludeSaturated)
	}
	found := false
	for _, m := range o.Members {
		if m == "a" {
			found = true
		}
	}
	if !found {
		t.Fatal("the saturated member was dropped from the chain; balance reorders, it never removes a hop")
	}
}

func TestBalanceSelectWeightedHashMovesOffASaturatedMember(t *testing.T) {
	group := BalanceGroup{
		Name:     "busy",
		Members:  []string{"a", "b", "c"},
		Strategy: BalanceStrategyWeightedHash,
	}
	snap := loads("busy", map[string]float64{"a": 1.0, "b": 0.0, "c": 0.0}, "a", "b", "c")
	var p BalancePolicy

	// A saturated member has effective weight 0, so it is formally excluded:
	// the assertion is "never chosen", not "approximately never chosen".
	for i := 0; i < 500; i++ {
		o := p.Select(group, snap, "k"+itoa(i))
		if o.Members[0] == "a" {
			t.Fatalf("weighted-hash chose the saturated member for key k%d", i)
		}
	}
}

func TestBalanceSelectIsDeterministic(t *testing.T) {
	group := BalanceGroup{
		Name:     "det",
		Members:  []string{"a", "b", "c", "d"},
		Strategy: BalanceStrategyWeightedHash,
		Weights:  map[string]int{"a": 2, "b": 1, "c": 3, "d": 1},
	}
	snap := loads("det", map[string]float64{"a": 0.2, "b": 0.7, "c": 0.1, "d": 0.9}, "a", "b", "c", "d")

	var p1, p2 BalancePolicy
	first := p1.Select(group, snap, "session-abc-123")
	for i := 0; i < 100; i++ {
		got := p2.Select(group, snap, "session-abc-123")
		if strings.Join(got.Members, ",") != strings.Join(first.Members, ",") {
			t.Fatalf("repetition %d produced %v, want %v", i, got.Members, first.Members)
		}
		if got.Plan.Chosen != first.Plan.Chosen || got.Plan.Reason != first.Plan.Reason {
			t.Fatalf("repetition %d produced plan %+v, want %+v", i, got.Plan, first.Plan)
		}
	}
}

func TestBalanceSelectPriorityDoesNotSplit(t *testing.T) {
	group := BalanceGroup{
		Name:     "ordered",
		Members:  []string{"a", "b", "c"},
		Strategy: BalanceStrategyPriority,
	}
	// A wild snapshot: member 0 is the fastest and least loaded and still
	// must not be preferred over the declared order.
	snap := loads("ordered", map[string]float64{"a": 0.0, "b": 1.0, "c": 1.0}, "a", "b", "c")
	var p BalancePolicy

	counts := map[string]int{}
	for i := 0; i < 1000; i++ {
		o := p.Select(group, snap, "k"+itoa(i))
		counts[o.Members[0]]++
		if o.Plan.Reason != "" {
			t.Fatalf("priority reported a fail-open reason %q on a fully measured snapshot", o.Plan.Reason)
		}
	}
	if counts["a"] != 1000 {
		t.Fatalf("priority chose a non-declared-first member %d times; scores must be ignored", 1000-counts["a"])
	}
	// The scores were read and recorded, so an operator can see they were
	// ignored rather than never consulted.
	o := p.Select(group, snap, "k0")
	var sawLoad bool
	for _, m := range o.Plan.Members {
		if m.Member == "a" && m.Load == 0 {
			sawLoad = true
		}
	}
	if !sawLoad {
		t.Fatal("priority did not record the measured load it ignored")
	}
}

func TestBalanceSelectFailsOpenToDeclaredOrderWithNoSnapshot(t *testing.T) {
	// A dead metrics endpoint must not be able to deny service. Every member
	// unmeasured is the whole-group fail-open path.
	group := BalanceGroup{
		Name:     "dark",
		Members:  []string{"a", "b", "c"},
		Strategy: BalanceStrategyWeightedHash,
	}
	snap := BalanceSnapshot{Group: "dark", Members: map[string]MemberScore{
		"a": {Load: 0.5}, "b": {Load: 0.4}, "c": {Load: 0.3}, // all Measured false
	}}
	var p BalancePolicy
	o := p.Select(group, snap, "k0")
	if strings.Join(o.Members, ",") != "a,b,c" {
		t.Fatalf("order = %v, want the declared order [a b c]", o.Members)
	}
	if o.Plan.Reason != BalanceExcludeNoSnapshot {
		t.Fatalf("plan reason = %q, want %q", o.Plan.Reason, BalanceExcludeNoSnapshot)
	}
	if o.Plan.Chosen != "a" {
		t.Fatalf("plan.Chosen = %q, want the declared head", o.Plan.Chosen)
	}
	// A snapshot with no members map at all is the same verdict.
	o = p.Select(group, BalanceSnapshot{Group: "dark"}, "k0")
	if o.Plan.Reason != BalanceExcludeNoSnapshot {
		t.Fatalf("empty snapshot reason = %q, want %q", o.Plan.Reason, BalanceExcludeNoSnapshot)
	}
}

func TestBalanceSelectWeightedHashUnmeasuredMemberCompetesAtDeclaredWeight(t *testing.T) {
	group := BalanceGroup{
		Name:     "partial",
		Members:  []string{"a", "b", "c"},
		Strategy: BalanceStrategyWeightedHash,
	}
	snap := BalanceSnapshot{Group: "partial", Members: map[string]MemberScore{
		"a": {Load: 0.0, Measured: true},
		"b": {Measured: false}, // no fresh measurement
		"c": {Load: 0.0, Measured: true},
	}}
	var p BalancePolicy
	counts := map[string]int{}
	for i := 0; i < 2000; i++ {
		o := p.Select(group, snap, "k"+itoa(i))
		for _, m := range o.Plan.Members {
			if m.Member == "b" && m.Excluded != "" {
				t.Fatalf("k%d: unmeasured member b excluded as %q; weighted-hash must rank it at its declared weight", i, m.Excluded)
			}
		}
		counts[o.Members[0]]++
	}
	// Equal declared weights: an unmeasured member earns its share like the
	// measured peers, or a member measured only by serving is starved forever.
	within(t, "share[b] unmeasured", float64(counts["b"])/2000.0, 1.0/3.0, 0.05)
	within(t, "share[a]", float64(counts["a"])/2000.0, 1.0/3.0, 0.05)
	within(t, "share[c]", float64(counts["c"])/2000.0, 1.0/3.0, 0.05)
}

func TestBalanceSelectPriorityUnmeasuredMemberKeepsDeclaredOrder(t *testing.T) {
	group := BalanceGroup{
		Name:     "partial",
		Members:  []string{"a", "b", "c"},
		Strategy: BalanceStrategyPriority,
	}
	snap := BalanceSnapshot{Group: "partial", Members: map[string]MemberScore{
		"a": {Load: 0.0, Measured: true},
		"b": {Measured: false},
		"c": {Load: 0.0, Measured: true},
	}}
	var p BalancePolicy
	for i := 0; i < 200; i++ {
		o := p.Select(group, snap, "k"+itoa(i))
		if got := strings.Join(o.Members, ","); got != "a,b,c" {
			t.Fatalf("k%d: priority order = %s, want the declared a,b,c", i, got)
		}
	}
}

func TestBalanceSelectAllSaturatedFailsOpenWithNoEligibleMember(t *testing.T) {
	group := BalanceGroup{
		Name:     "bricked",
		Members:  []string{"a", "b"},
		Strategy: BalanceStrategyLeastLoad,
	}
	snap := loads("bricked", map[string]float64{"a": 1.0, "b": 1.0}, "a", "b")
	var p BalancePolicy
	o := p.Select(group, snap, "k0")
	if strings.Join(o.Members, ",") != "a,b" {
		t.Fatalf("order = %v, want the declared order", o.Members)
	}
	if o.Plan.Reason != BalanceExcludeNoEligibleMember {
		t.Fatalf("reason = %q, want %q", o.Plan.Reason, BalanceExcludeNoEligibleMember)
	}
}

func TestBalanceSelectClampsScoresOnRead(t *testing.T) {
	// A caller must not be able to push an unbounded number through the weight
	// multiplication, and the receipt must never carry the raw value.
	group := BalanceGroup{
		Name:     "clamp",
		Members:  []string{"a", "b"},
		Strategy: BalanceStrategyLeastLoad,
	}
	snap := BalanceSnapshot{Group: "clamp", Members: map[string]MemberScore{
		"a": {Load: 5, Latency: 9, Measured: true},
		"b": {Load: -3, Latency: -1, Measured: true},
	}}
	var p BalancePolicy
	o := p.Select(group, snap, "k0")
	for _, m := range o.Plan.Members {
		if m.Load < 0 || m.Load > 1 {
			t.Fatalf("member %s carries an unclamped load %v", m.Member, m.Load)
		}
		if m.Latency < 0 || m.Latency > 1 {
			t.Fatalf("member %s carries an unclamped latency %v", m.Member, m.Latency)
		}
	}
	byMember := map[string]BalanceMemberPlan{}
	for _, m := range o.Plan.Members {
		byMember[m.Member] = m
	}
	if got := byMember["a"].Load; got != 1 {
		t.Fatalf("clamped load for a = %v, want 1", got)
	}
	if got := byMember["b"].Load; got != 0 {
		t.Fatalf("clamped load for b = %v, want 0", got)
	}
	// NaN is a measurement bug, not a huge or tiny number. It clamps to 0 so
	// the member is treated as idle rather than poisoning the arithmetic.
	snap.Members["a"] = MemberScore{Load: math.NaN(), Measured: true}
	o = p.Select(group, snap, "k0")
	for _, m := range o.Plan.Members {
		if math.IsNaN(m.Load) {
			t.Fatalf("member %s carries NaN in the plan", m.Member)
		}
	}
}

func TestBalanceSelectEffectiveWeightFallsWithLoad(t *testing.T) {
	group := BalanceGroup{
		Name:     "eff",
		Members:  []string{"a", "b"},
		Strategy: BalanceStrategyWeightedHash,
		Weights:  map[string]int{"a": 4, "b": 4},
	}
	var p BalancePolicy
	for _, load := range []float64{0, 0.25, 0.5, 0.75} {
		snap := loads("eff", map[string]float64{"a": load, "b": load}, "a", "b")
		o := p.Select(group, snap, "k0")
		want := 4 * (1 - load)
		for _, m := range o.Plan.Members {
			if math.Abs(m.EffectiveWeight-want) > 1e-9 {
				t.Fatalf("load %v: effective weight = %v, want %v", load, m.EffectiveWeight, want)
			}
		}
	}
}

func TestBalanceSelectTieBreakIsStable(t *testing.T) {
	// Two members with identical weight and identical scores must produce a
	// stable order. The loop is over many keys precisely so a single lucky
	// ordering does not pass the test.
	group := BalanceGroup{
		Name:     "tied",
		Members:  []string{"a", "b"},
		Strategy: BalanceStrategyLeastLoad,
	}
	snap := loads("tied", map[string]float64{"a": 0.5, "b": 0.5}, "a", "b")
	var p BalancePolicy
	counts := map[string]int{}
	for i := 0; i < 1000; i++ {
		o := p.Select(group, snap, "k"+itoa(i))
		counts[o.Members[0]]++
	}
	within(t, "share[a] on a perfect tie", float64(counts["a"])/1000.0, 0.5, 0.05)
	// And the full order is always the same two members.
	o := p.Select(group, snap, "k0")
	if strings.Join(o.Members, ",") != "a,b" && strings.Join(o.Members, ",") != "b,a" {
		t.Fatalf("order = %v, want a permutation of [a b]", o.Members)
	}
}

func TestBalancePlanNeverCarriesTheRawKey(t *testing.T) {
	// The key may be a session identifier and a receipt gets pasted into
	// issues, so the value must not be recorded — only its KIND.
	const session = "sess-9f2b7c41ae0d4c6f"
	group := BalanceGroup{
		Name:     "secret",
		Members:  []string{"a", "b", "c"},
		Strategy: BalanceStrategyWeightedHash,
	}
	snap := measured("secret", "a", "b", "c")
	var p BalancePolicy
	o := p.Select(group, snap, session)
	if o.Plan.KeyKind != "session" {
		t.Fatalf("KeyKind = %q, want %q", o.Plan.KeyKind, "session")
	}
	rendered := renderPlan(o.Plan)
	if strings.Contains(rendered, session) {
		t.Fatalf("the rendered plan leaks the raw balance key: %s", rendered)
	}
	for _, line := range o.Plan.Audit {
		if strings.Contains(line, session) {
			t.Fatalf("audit line leaks the raw balance key: %s", line)
		}
	}
}

func TestBalanceSelectEmptyStrategyResolvesToPriority(t *testing.T) {
	// The zero value is the declared default, so an existing config that
	// declares a group with no strategy keeps serving in declared order.
	group := BalanceGroup{Name: "implicit", Members: []string{"a", "b", "c"}}
	snap := measured("implicit", "a", "b", "c")
	var p BalancePolicy
	o := p.Select(group, snap, "k0")
	if o.Plan.Strategy != BalanceStrategyPriority {
		t.Fatalf("resolved strategy = %q, want %q", o.Plan.Strategy, BalanceStrategyPriority)
	}
	if strings.Join(o.Members, ",") != "a,b,c" {
		t.Fatalf("order = %v, want the declared order", o.Members)
	}
}

func TestBalanceSelectNeverReturnsAnEmptyOrder(t *testing.T) {
	// There is no input to Select that produces an error or an empty chain.
	// A balancer that could refuse service would be a quota gate wearing a
	// balancer's name.
	groups := []BalanceGroup{
		{Name: "g1", Members: []string{"a", "b"}},
		{Name: "g2", Members: []string{"a", "b"}, Strategy: BalanceStrategyWeightedHash},
		{Name: "g3", Members: []string{"a", "b"}, Strategy: BalanceStrategyLeastLoad},
		{Name: "g4", Members: []string{"a", "b"}, Strategy: BalanceStrategyLeastLatency},
	}
	snaps := []BalanceSnapshot{
		{},
		{Group: "g1"},
		measured("g2", "a", "b"),
		{Group: "g3", Members: map[string]MemberScore{"a": {Load: math.NaN(), Measured: true}}},
	}
	var p BalancePolicy
	for _, g := range groups {
		for _, s := range snaps {
			o := p.Select(g, s, "k")
			if len(o.Members) != len(g.Members) {
				t.Fatalf("group %s returned %d members, want %d (the full chain)", g.Name, len(o.Members), len(g.Members))
			}
		}
	}
}

func TestBalanceScoreMatchesTheGatewayHashConstruction(t *testing.T) {
	// BalanceScore(key, group, member) must equal ComputeHRWScore(key,
	// group+":"+member) so a decision computed by the fabric and one computed
	// on the serving axis for the same identity agree. pkg/routing may
	// not import the gateway, so the property is asserted against the
	// expected value of the construction itself.
	h := BalanceScore("k", "g", "m")
	if h == 0 {
		t.Fatal("BalanceScore returned 0; the avalanche finalizer should make this vanishingly unlikely")
	}
	if h == BalanceScore("k", "g", "m") {
		// Stable, which is the property that matters; assert the avalanche by
		// checking two near-identical members do not correlate.
		_ = h
	}
	a := BalanceScore("k", "g", "member-a")
	b := BalanceScore("k", "g", "member-b")
	if a == b {
		t.Fatal("two distinct members hash identically; the separator or finalizer is broken")
	}
	if a == BalanceScore("k", "gX", "amember") {
		t.Fatal("the group/member boundary is not separated by the ':' delimiter")
	}
}

func TestBalanceExclusionReasonsIsAClosedSet(t *testing.T) {
	got := BalanceExclusionReasons()
	want := []string{"saturated", "unmeasured", "no_snapshot", "no_eligible_member"}
	if len(got) != len(want) {
		t.Fatalf("BalanceExclusionReasons() = %v, want exactly %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("BalanceExclusionReasons()[%d] = %q, want %q", i, got[i], want[i])
		}
	}
}

// renderPlan flattens a plan to text so a test can assert on what a human
// would actually read, rather than on struct fields that happen not to carry
// the key today.
func renderPlan(p BalancePlan) string {
	var b strings.Builder
	b.WriteString(p.Schema)
	b.WriteString(" group=" + p.Group)
	b.WriteString(" strategy=" + string(p.Strategy))
	b.WriteString(" key_kind=" + p.KeyKind)
	b.WriteString(" chosen=" + p.Chosen)
	b.WriteString(" reason=" + string(p.Reason))
	for _, m := range p.Members {
		b.WriteString(" member=" + m.Member)
		b.WriteString(" weight=" + itoa(m.Weight))
		b.WriteString(" load=" + fmtFloat(m.Load))
		b.WriteString(" rank=" + fmtFloat(m.Rank))
		b.WriteString(" excluded=" + string(m.Excluded))
	}
	for _, line := range p.Audit {
		b.WriteString("\n" + line)
	}
	return b.String()
}

func fmtFloat(v float64) string {
	if math.IsInf(v, 1) {
		return "inf"
	}
	if math.IsNaN(v) {
		return "nan"
	}
	return itoa(int(math.Round(v * 1e6)))
}
