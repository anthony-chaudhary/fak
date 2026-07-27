package modelroute

import (
	"reflect"
	"strings"
	"testing"
)

// Tests for spawn.go — a delegated turn getting its own rung (epic #5416,
// track E). The two claims that matter are that the parent CANNOT reach the
// decision, and that inheriting the parent's model would have bypassed the
// child's floor.

// placeOr is a Place that fails the test rather than returning an error, for
// building the parent placements the spawn cases hang off.
func placeOr(t *testing.T, r Roster, class WorkClass, cands []Candidate) Placement {
	t.Helper()
	p, err := r.Place(class, cands)
	if err != nil {
		t.Fatalf("Place(%q) = error %v, want a placement to spawn from", class, err)
	}
	return p
}

func spawnOr(t *testing.T, r Roster, parent Placement, class WorkClass, cands []Candidate) SpawnPlacement {
	t.Helper()
	s, err := r.PlaceSpawn(parent, class, cands)
	if err != nil {
		t.Fatalf("PlaceSpawn(parent=%s/%s, %q) = error %v, want a placement",
			parent.Zone, parent.Model, class, err)
	}
	return s
}

func allClasses() []WorkClass {
	return []WorkClass{ClassRoutine, ClassNormalImpl, ClassSecurityRelease, ClassUltraHard}
}

// TestSpawnParentCannotChangeTheChildsPlacement is the load-bearing invariant:
// the parent's zone is an INPUT, never a floor. If a parent could move a child's
// rung at all, delegation would ratchet — one frontier turn near the root would
// pin its whole subtree to the vendor rung, which is the behavior this track
// exists to remove.
//
// It is asserted as an EXACT equality against Place over every (parent, class)
// pair, not as a spot check on zones: an equality is the only form of this claim
// that a later edit cannot satisfy while quietly special-casing one rung.
func TestSpawnParentCannotChangeTheChildsPlacement(t *testing.T) {
	r := threeZoneRoster()
	pool := measured()

	var parents []Placement
	parents = append(parents, Placement{}) // the root turn
	for _, pc := range allClasses() {
		parents = append(parents, placeOr(t, r, pc, pool))
	}
	// A parent that is not in the roster's own ladder at all, to be sure the
	// equality does not hold merely because every parent happened to be one of
	// the three placements above.
	parents = append(parents, Placement{Model: "someone-elses-model", Zone: ZoneVendor, Measured: true})

	for _, class := range allClasses() {
		want, err := r.Place(class, pool)
		if err != nil {
			t.Fatalf("Place(%q) = error %v", class, err)
		}
		for _, parent := range parents {
			got := spawnOr(t, r, parent, class, pool)
			if !reflect.DeepEqual(got.Placement, want) {
				t.Fatalf("PlaceSpawn(parent=%s/%s, %q).Placement = %+v, want %+v exactly.\n"+
					"The parent must not be able to move the child's rung; if it can, delegation ratchets.",
					parent.Zone, parent.Model, class, got.Placement, want)
			}
		}
	}
}

// TestSpawnFromAVendorTurnCanStillLandOnTheLaptop is the concrete form of the
// same rule, and the epic's headline event: a sub-agent spawned by a frontier
// turn runs on the engineer's own machine.
func TestSpawnFromAVendorTurnCanStillLandOnTheLaptop(t *testing.T) {
	r := threeZoneRoster()
	pool := measured()

	parent := placeOr(t, r, ClassUltraHard, pool)
	if parent.Zone != ZoneVendor {
		t.Fatalf("fixture: ultra-hard parent landed in %q, want %q", parent.Zone, ZoneVendor)
	}

	child := spawnOr(t, r, parent, ClassRoutine, pool)
	if child.Placement.Zone != ZoneDevice {
		t.Fatalf("routine sub-agent of a vendor turn landed in %q, want %q.\n"+
			"A parent's rung is not a floor: %+v", child.Placement.Zone, ZoneDevice, child)
	}
	if !child.Descended {
		t.Fatalf("Descended = false for a %s child of a %s parent, want true", child.Placement.Zone, parent.Zone)
	}
	if !child.SelfHostedDescent() {
		t.Fatalf("SelfHostedDescent() = false, want true: the child landed on %q having come from %q",
			child.Placement.Zone, parent.Zone)
	}
	if !hasReason(child.Reasons, ReasonSpawnDescended) {
		t.Fatalf("reasons %v missing %q", child.Reasons, ReasonSpawnDescended)
	}
	if child.ParentZone != ZoneVendor || child.ParentModel != parent.Model {
		t.Fatalf("provenance = %s/%s, want %s/%s", child.ParentZone, child.ParentModel, ZoneVendor, parent.Model)
	}
}

// TestSpawnInheritanceWouldHaveBypassedTheSecurityFloor is the safety half of
// this track, and the reason it is not merely a cost optimization.
//
// A routine turn is legitimately placed on a small T2 model on the laptop. It
// spawns a security/release/destructive child. Under inheritance that child runs
// on the T2 model — and no gate in this package is violated, because the child
// was never handed to one. Re-placing it puts it back above its own floor.
func TestSpawnInheritanceWouldHaveBypassedTheSecurityFloor(t *testing.T) {
	r := threeZoneRoster()
	pool := measured()

	parent := placeOr(t, r, ClassRoutine, pool)
	if parent.Zone != ZoneDevice || parent.Choice.Capability != TierT2 {
		t.Fatalf("fixture: routine parent = %s/%s at %s, want a device T2 model",
			parent.Zone, parent.Model, parent.Choice.Capability)
	}

	child := spawnOr(t, r, parent, ClassSecurityRelease, pool)

	if !child.InheritWouldUnderTier {
		t.Fatalf("InheritWouldUnderTier = false, but inheriting %q (%s) for %q work would sit below its T1 floor.\n"+
			"This is the bypass the track exists to close: %+v", parent.Model, parent.Choice.Capability, ClassSecurityRelease, child)
	}
	if !hasReason(child.Reasons, ReasonSpawnInheritUnderTier) {
		t.Fatalf("reasons %v missing %q", child.Reasons, ReasonSpawnInheritUnderTier)
	}
	// And the child's OWN placement clears the floor it would have bypassed.
	if !child.Placement.Choice.Admitted || !child.Placement.Choice.Capability.MeetsRequirement(TierT1) {
		t.Fatalf("the re-placed child landed on %q (%s), which does not clear the %q floor: %+v",
			child.Placement.Model, child.Placement.Choice.Capability, ClassSecurityRelease, child.Placement)
	}
	if child.Placement.Zone == ZoneDevice {
		t.Fatalf("security-release child stayed on the device rung (%q) — the floor did not bind", child.Placement.Model)
	}
}

// TestSpawnSaysSoWhenInheritingWouldHaveBeenSafe keeps the counterfactual from
// degenerating into a field that is always true. An expensive parent is
// over-capable for a routine child: inheriting it would have wasted money, not
// bypassed a floor, and an operator triaging the two needs them distinguished.
func TestSpawnSaysSoWhenInheritingWouldHaveBeenSafe(t *testing.T) {
	r := threeZoneRoster()
	pool := measured()

	parent := placeOr(t, r, ClassUltraHard, pool) // frontier, T0
	child := spawnOr(t, r, parent, ClassRoutine, pool)

	if child.InheritWouldUnderTier {
		t.Fatalf("InheritWouldUnderTier = true for a T0 parent against a routine floor: %+v", child)
	}
	if !hasReason(child.Reasons, ReasonSpawnInheritAdmitted) {
		t.Fatalf("reasons %v missing %q", child.Reasons, ReasonSpawnInheritAdmitted)
	}
}

// TestSpawnWillNotJudgeInheritanceAgainstAnUnmeasuredParent guards the trap
// Placement.Measured documents: an unmeasured candidate enters Admit at the
// zero-value tier, which is the MOST demanding one, so an ungraded parent
// appears to clear every floor in the system. Reporting that as "inheriting
// would have been fine" would put a number nobody computed on an operator's
// screen — and would do it in the exact direction that hides a bypass.
func TestSpawnWillNotJudgeInheritanceAgainstAnUnmeasuredParent(t *testing.T) {
	r := threeZoneRoster()
	pool := measured()

	cases := []struct {
		name   string
		parent Placement
	}{
		{
			// The literal artifact: a hand-built parent whose Choice was never
			// filled, so Capability reads T0 by zero value alone.
			name:   "zero-valued choice",
			parent: Placement{Model: "tiny", Zone: ZoneDevice},
		},
		{
			// The asserted one: somebody claimed T0 and nobody measured it.
			name:   "asserted T0",
			parent: Placement{Model: "tiny", Zone: ZoneDevice, Choice: TierChoice{Capability: TierT0, Admitted: true}},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			child := spawnOr(t, r, tc.parent, ClassUltraHard, pool)
			if child.InheritWouldUnderTier {
				t.Fatalf("InheritWouldUnderTier = true from an unmeasured parent, want false (nobody knows): %+v", child)
			}
			if !hasReason(child.Reasons, ReasonSpawnInheritUnmeasured) {
				t.Fatalf("reasons %v missing %q", child.Reasons, ReasonSpawnInheritUnmeasured)
			}
			if hasReason(child.Reasons, ReasonSpawnInheritAdmitted) {
				t.Fatalf("reasons %v claim %q on the strength of an ungraded capability", child.Reasons, ReasonSpawnInheritAdmitted)
			}
			if hasReason(child.Reasons, ReasonSpawnInheritUnderTier) {
				t.Fatalf("reasons %v claim %q on the strength of an ungraded capability", child.Reasons, ReasonSpawnInheritUnderTier)
			}
		})
	}
}

// TestSpawnRelationTracksTheRungsExactly pins all three directions, including
// the one that costs money. A routine turn on the laptop that spawns genuinely
// hard work SHOULD rise to a vendor, and an operator reading the bill has to be
// able to find it.
func TestSpawnRelationTracksTheRungsExactly(t *testing.T) {
	r := threeZoneRoster()
	pool := measured()

	cases := []struct {
		name        string
		parentClass WorkClass
		childClass  WorkClass
		wantReason  string
		wantDescend bool
	}{
		{"vendor parent, routine child", ClassUltraHard, ClassRoutine, ReasonSpawnDescended, true},
		{"fleet parent, fleet child", ClassNormalImpl, ClassNormalImpl, ReasonSpawnHeld, false},
		{"device parent, ultra-hard child", ClassRoutine, ClassUltraHard, ReasonSpawnRose, false},
		{"vendor parent, fleet child", ClassUltraHard, ClassNormalImpl, ReasonSpawnDescended, true},
		{"device parent, routine child", ClassRoutine, ClassRoutine, ReasonSpawnHeld, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			parent := placeOr(t, r, tc.parentClass, pool)
			child := spawnOr(t, r, parent, tc.childClass, pool)

			if !hasReason(child.Reasons, tc.wantReason) {
				t.Fatalf("%s -> %s (%s -> %s): reasons %v missing %q",
					tc.parentClass, tc.childClass, parent.Zone, child.Placement.Zone, child.Reasons, tc.wantReason)
			}
			if child.Descended != tc.wantDescend {
				t.Fatalf("Descended = %v, want %v (%s -> %s)",
					child.Descended, tc.wantDescend, parent.Zone, child.Placement.Zone)
			}
			// Descended must mean STRICTLY cheaper, never merely different.
			if child.Descended != (child.Placement.Zone.Rank() < parent.Zone.Rank()) {
				t.Fatalf("Descended = %v but ranks are child=%d parent=%d",
					child.Descended, child.Placement.Zone.Rank(), parent.Zone.Rank())
			}
		})
	}
}

// TestSpawnWithNoParentReportsARootTurn: a turn nobody spawned has no
// relationship to report, and must not have one invented for it.
func TestSpawnWithNoParentReportsARootTurn(t *testing.T) {
	r := threeZoneRoster()
	root := spawnOr(t, r, Placement{}, ClassNormalImpl, measured())

	if !hasReason(root.Reasons, ReasonSpawnRootTurn) {
		t.Fatalf("reasons %v missing %q", root.Reasons, ReasonSpawnRootTurn)
	}
	if len(root.Reasons) != 1 {
		t.Fatalf("reasons = %v, want exactly [%s]: there is no parent to relate to", root.Reasons, ReasonSpawnRootTurn)
	}
	if root.ParentZone != "" || root.ParentModel != "" {
		t.Fatalf("root turn carries provenance %s/%s, want none", root.ParentZone, root.ParentModel)
	}
	if root.Descended || root.InheritWouldUnderTier {
		t.Fatalf("root turn claims a relationship: %+v", root)
	}
	// A root spawn is still a real placement.
	want, err := r.Place(ClassNormalImpl, measured())
	if err != nil {
		t.Fatalf("Place = error %v", err)
	}
	if !reflect.DeepEqual(root.Placement, want) {
		t.Fatalf("root spawn placement = %+v, want %+v", root.Placement, want)
	}
}

// TestSpawnRefusesAnUnclassifiedChild. An unclassified spawn is a missing
// classification, not a routine one. Place tolerates an empty class because
// PolicyFor answers conservatively for it; this path refuses, because
// "unclassified" arriving at a spawn is the exact shape a floor bypass takes,
// and a conservative default would hide it behind a correct-looking placement.
func TestSpawnRefusesAnUnclassifiedChild(t *testing.T) {
	r := threeZoneRoster()
	parent := placeOr(t, r, ClassRoutine, measured())

	if _, err := r.PlaceSpawn(parent, "", measured()); err == nil {
		t.Fatal("PlaceSpawn with an empty class = nil error, want a refusal")
	}
	// The looser contract on Place is deliberate and must stay: changing it
	// would be a behavior change on a live path, not a tightening of a new one.
	if _, err := r.Place("", measured()); err != nil {
		t.Fatalf("Place with an empty class = error %v, want the historical conservative placement", err)
	}
}

// TestSpawnRefusesAHalfKnownParent. A parent is either absent or complete. A
// caller holding half of one has lost track of where the spawning turn landed,
// and a guessed relationship is a provenance claim nobody can stand behind.
func TestSpawnRefusesAHalfKnownParent(t *testing.T) {
	r := threeZoneRoster()
	pool := measured()

	cases := []struct {
		name   string
		parent Placement
	}{
		{"zone with no model", Placement{Zone: ZoneFleet}},
		{"model in an unknown zone", Placement{Model: "corp-mid", Zone: PlacementZone("datacenter")}},
		{"model with no zone", Placement{Model: "corp-mid"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := r.PlaceSpawn(tc.parent, ClassRoutine, pool); err == nil {
				t.Fatalf("PlaceSpawn(%+v) = nil error, want a refusal", tc.parent)
			}
		})
	}
}

// TestSpawnPropagatesPlacementFailuresUnchanged: the spawn wrapper must not
// convert a placement error into a placement. Every fail-loud path Place owns
// stays fail-loud through this seam.
func TestSpawnPropagatesPlacementFailuresUnchanged(t *testing.T) {
	r := threeZoneRoster()
	parent := placeOr(t, r, ClassRoutine, measured())

	// A roster with NO default account: an id it does not bind is genuinely
	// unresolvable. threeZoneRoster names laptop as its default, so an unknown id
	// there resolves through that account rather than failing — which is the
	// roster's documented behavior, not a hole in this seam.
	noDefault := threeZoneRoster()
	noDefault.Default = ""

	cases := []struct {
		name   string
		roster Roster
		cands  []Candidate
	}{
		{"no candidates", r, nil},
		{"unresolvable model, no default account", noDefault, []Candidate{{Model: "typo-model", Capability: TierT0, Measured: true}}},
		{"empty model id", r, []Candidate{{Model: "", Capability: TierT0, Measured: true}}},
		{"nothing clears the floor", r, []Candidate{{Model: "tiny", Capability: TierT2, Measured: true}}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			class := ClassUltraHard
			_, wantErr := tc.roster.Place(class, tc.cands)
			if wantErr == nil {
				t.Fatalf("fixture: Place(%q) unexpectedly succeeded", class)
			}
			got, err := tc.roster.PlaceSpawn(parent, class, tc.cands)
			if err == nil {
				t.Fatalf("PlaceSpawn = %+v, nil error; Place refused with %v", got, wantErr)
			}
			if err.Error() != wantErr.Error() {
				t.Fatalf("PlaceSpawn error = %q, want Place's own %q (the wrapper must not reword a refusal)",
					err, wantErr)
			}
		})
	}
}

// TestSpawnComposesWithTheServingGate: a delegated turn must not be the one path
// that keeps dispatching to a host that stopped answering (track H).
func TestSpawnComposesWithTheServingGate(t *testing.T) {
	r := threeZoneRoster()
	pool := measured()
	parent := placeOr(t, r, ClassUltraHard, pool)

	// The whole fleet rung is down; a normal-impl child must leave it.
	report := ServingReport{
		Schema: ServingReportSchema,
		Covers: []PlacementZone{ZoneDevice, ZoneFleet},
		Models: map[string]ServingObservation{
			"corp-mid":     {State: ServingDown},
			"corp-agentic": {State: ServingDown},
			"frontier":     {State: ServingUp},
		},
	}
	child, err := r.PlaceSpawnWithServing(parent, ClassNormalImpl, pool, report)
	if err != nil {
		t.Fatalf("PlaceSpawnWithServing = error %v", err)
	}
	if child.Placement.Zone == ZoneFleet {
		t.Fatalf("child landed on the down fleet rung: %+v", child.Placement)
	}
	if !child.Placement.FailedOver {
		t.Fatalf("FailedOver = false after routing around a down rung: %+v", child.Placement)
	}
	// Same equality as the no-serving case: the parent still cannot move it.
	want, err := r.PlaceWithServing(ClassNormalImpl, pool, report)
	if err != nil {
		t.Fatalf("PlaceWithServing = error %v", err)
	}
	if !reflect.DeepEqual(child.Placement, want) {
		t.Fatalf("spawn placement = %+v, want %+v", child.Placement, want)
	}
	// And an invalid snapshot is refused here too, not ignored.
	bad := ServingReport{Models: map[string]ServingObservation{"corp-mid": {State: ServingUp}}}
	if _, err := r.PlaceSpawnWithServing(parent, ClassNormalImpl, pool, bad); err == nil {
		t.Fatal("PlaceSpawnWithServing accepted an unversioned report carrying observations")
	}
}

// TestSelfHostedDescentIsNarrowerThanSelfHosted. The epic's number counts tokens
// that MOVED off a vendor. A child that held an already-self-hosted rung moved
// nothing, and counting it would inflate the only measurement that says whether
// any of this works.
func TestSelfHostedDescentIsNarrowerThanSelfHosted(t *testing.T) {
	r := threeZoneRoster()
	pool := measured()

	held := spawnOr(t, r, placeOr(t, r, ClassNormalImpl, pool), ClassNormalImpl, pool)
	if !held.Placement.SelfHosted() {
		t.Fatalf("fixture: normal-impl child landed in %q, want a self-hosted rung", held.Placement.Zone)
	}
	if held.SelfHostedDescent() {
		t.Fatalf("SelfHostedDescent() = true for a child that held its parent's rung: %+v", held)
	}

	// On the three-rung ladder every descent lands on device or fleet, so
	// Descended IMPLIES SelfHosted and the two halves of the conjunction cannot be
	// told apart by any placement Place can produce. They still have to be told
	// apart: SpawnPlacement is exported, carries json tags, and is meant to be
	// published and counted, so a DECODED report can present a combination the
	// ladder never produces. A descent that did not land on hardware the
	// organization operates is not a self-hosted descent, whatever produced it.
	decoded := SpawnPlacement{Descended: true, Placement: Placement{Zone: ZoneVendor}}
	if decoded.SelfHostedDescent() {
		t.Fatal("SelfHostedDescent() = true for a descent that landed on the vendor rung")
	}

	// And the conjunction holds across everything the ladder itself produces.
	for _, pc := range allClasses() {
		for _, cc := range allClasses() {
			s := spawnOr(t, r, placeOr(t, r, pc, pool), cc, pool)
			if s.SelfHostedDescent() != (s.Descended && s.Placement.SelfHosted()) {
				t.Fatalf("SelfHostedDescent() = %v but Descended=%v SelfHosted=%v (%s -> %s)",
					s.SelfHostedDescent(), s.Descended, s.Placement.SelfHosted(), pc, cc)
			}
		}
	}
}

// TestSpawnReasonsStayInTheClosedVocabulary. A status surface renders these
// verbatim; a token invented at a call site would render as noise.
func TestSpawnReasonsStayInTheClosedVocabulary(t *testing.T) {
	known := map[string]bool{
		ReasonSpawnDescended:         true,
		ReasonSpawnHeld:              true,
		ReasonSpawnRose:              true,
		ReasonSpawnRootTurn:          true,
		ReasonSpawnInheritUnderTier:  true,
		ReasonSpawnInheritAdmitted:   true,
		ReasonSpawnInheritUnmeasured: true,
	}
	r := threeZoneRoster()
	pool := measured()

	parents := []Placement{{}, {Model: "tiny", Zone: ZoneDevice}}
	for _, pc := range allClasses() {
		parents = append(parents, placeOr(t, r, pc, pool))
	}
	for _, parent := range parents {
		for _, cc := range allClasses() {
			s := spawnOr(t, r, parent, cc, pool)
			if len(s.Reasons) == 0 {
				t.Fatalf("spawn (parent=%s, class=%s) recorded no reason at all", parent.Zone, cc)
			}
			for _, got := range s.Reasons {
				if !known[got] {
					t.Fatalf("reason %q is outside the closed vocabulary (parent=%s, class=%s)", got, parent.Zone, cc)
				}
				if !strings.HasPrefix(got, "spawn-") {
					t.Fatalf("reason %q does not carry the spawn- prefix its vocabulary is namespaced by", got)
				}
			}
		}
	}
}

// TestSpawnIsDeterministic. Same roster, same parent, same class, same
// candidates => the same answer, every time. There is no clock, no map
// iteration, and no scoring in the decision.
func TestSpawnIsDeterministic(t *testing.T) {
	r := threeZoneRoster()
	pool := measured()
	parent := placeOr(t, r, ClassUltraHard, pool)

	first := spawnOr(t, r, parent, ClassNormalImpl, pool)
	for i := 0; i < 50; i++ {
		got := spawnOr(t, r, parent, ClassNormalImpl, pool)
		if !reflect.DeepEqual(got, first) {
			t.Fatalf("iteration %d = %+v, want %+v", i, got, first)
		}
	}
}
