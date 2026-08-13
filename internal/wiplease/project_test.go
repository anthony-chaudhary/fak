package wiplease

import (
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/laneadmit"
	"github.com/anthony-chaudhary/fak/internal/wipattr"
)

func liveSet(ids ...string) map[string]bool {
	m := map[string]bool{}
	for _, id := range ids {
		m[id] = true
	}
	return m
}

func owned(file, owner string) wipattr.Attribution {
	return wipattr.Attribution{File: file, State: wipattr.AttrOwned, Owner: owner}
}

// The load-bearing property: a live session's dirty footprint becomes lease geometry the
// EXISTING admission decision can refuse on, without that session ever declaring anything.
// This is the ~88% of live sessions the lease plane currently knows nothing about.
func TestProjectMakesLiveOwnerFootprintVisible(t *testing.T) {
	got := Project([]wipattr.Attribution{
		owned("internal/a/a.go", "sess-live"),
		owned("internal/a/b.go", "sess-live"),
	}, liveSet("sess-live"), Options{})

	want := []laneadmit.Lease{{
		ID:     "wip:sess-live",
		Holder: "sess-live",
		Tree:   []string{"internal/a/a.go", "internal/a/b.go"},
	}}
	if !reflect.DeepEqual(got.Active, want) {
		t.Fatalf("Active = %+v, want %+v", got.Active, want)
	}
	if len(got.Reclaimable) != 0 {
		t.Errorf("Reclaimable = %+v, want empty", got.Reclaimable)
	}
	if got.Undeclared != 1 {
		t.Errorf("Undeclared = %d, want 1 (the session declared no lease)", got.Undeclared)
	}
}

// The projection must actually compose with the decision it feeds — a peer requesting a
// path a live session is dirtying is refused by laneadmit on geometry alone. This is the
// whole point of emitting laneadmit.Lease rather than inventing a second gate.
func TestProjectedLeaseIsRefusedByLaneAdmit(t *testing.T) {
	occ := Project([]wipattr.Attribution{owned("internal/a/a.go", "sess-live")},
		liveSet("sess-live"), Options{})

	v := laneadmit.Decide(laneadmit.Request{
		Surface: laneadmit.SurfaceManual,
		Tree:    []string{"internal/a/a.go"},
		Holder:  "peer",
		LeaseID: "peer-lease",
	}, occ.Active, laneadmit.Taxonomy{})

	if v.Admit {
		t.Fatalf("peer was admitted onto a live session's dirty path; verdict = %+v", v)
	}
	if len(v.Conflicts) == 0 || v.Conflicts[0].LeaseID != "wip:sess-live" {
		t.Errorf("conflict does not name the projected lease: %+v", v.Conflicts)
	}
}

// A path whose owner is gone is ADOPTABLE, not blocking. Projecting it as a lease would
// wall off abandoned work permanently, since no live holder remains to release it.
func TestProjectTreatsDeadOwnerAsReclaimable(t *testing.T) {
	got := Project([]wipattr.Attribution{owned("internal/a/a.go", "sess-dead")},
		liveSet("sess-other"), Options{})

	if len(got.Active) != 0 {
		t.Fatalf("Active = %+v, want empty (owner is dead)", got.Active)
	}
	want := []Reclaimable{{Path: "internal/a/a.go", Owner: "sess-dead", Reason: ReclaimOwnerDead}}
	if !reflect.DeepEqual(got.Reclaimable, want) {
		t.Fatalf("Reclaimable = %+v, want %+v", got.Reclaimable, want)
	}
}

// An unattributed path is reported with its own reason, not merged into OWNER_DEAD: the
// remedies differ (adopt a known delta vs. investigate an unknown one).
func TestProjectReportsOrphanSeparately(t *testing.T) {
	got := Project([]wipattr.Attribution{
		{File: "internal/a/a.go", State: wipattr.AttrOrphan},
	}, liveSet(), Options{})

	want := []Reclaimable{{Path: "internal/a/a.go", Reason: ReclaimUnattributed}}
	if !reflect.DeepEqual(got.Reclaimable, want) {
		t.Fatalf("Reclaimable = %+v, want %+v", got.Reclaimable, want)
	}
}

// A SHARED path cannot be assigned to one session without guessing, and guessing wrong
// means telling a peer a path is free while a live session edits it. Every live claimant
// carries it.
func TestProjectSharedPathGoesToEveryLiveClaimant(t *testing.T) {
	got := Project([]wipattr.Attribution{{
		File: "internal/a/a.go", State: wipattr.AttrShared, Owners: []string{"s1", "s2", "s3"},
	}}, liveSet("s1", "s3"), Options{})

	if len(got.Active) != 2 {
		t.Fatalf("Active = %+v, want one lease each for s1 and s3", got.Active)
	}
	for _, l := range got.Active {
		if !reflect.DeepEqual(l.Tree, []string{"internal/a/a.go"}) {
			t.Errorf("lease %s tree = %v, want the shared path", l.ID, l.Tree)
		}
	}
	if len(got.Reclaimable) != 0 {
		t.Errorf("a shared path with live claimants is not reclaimable: %+v", got.Reclaimable)
	}
}

// A shared path whose claimants are ALL dead falls to reclaimable with every claimant
// named, so an adopter knows the full provenance of what it is picking up.
func TestProjectSharedPathAllDeadIsReclaimable(t *testing.T) {
	got := Project([]wipattr.Attribution{{
		File: "internal/a/a.go", State: wipattr.AttrShared, Owners: []string{"s2", "s1"},
	}}, liveSet(), Options{})

	want := []Reclaimable{{Path: "internal/a/a.go", Owners: []string{"s1", "s2"}, Reason: ReclaimOwnerDead}}
	if !reflect.DeepEqual(got.Reclaimable, want) {
		t.Fatalf("Reclaimable = %+v, want %+v", got.Reclaimable, want)
	}
}

// Undeclared is the adoption gap in one number: a session that already declared a lease
// is still projected (its dirt may exceed what it declared) but is not counted as a gap.
func TestProjectUndeclaredCountsOnlyUndeclaredOwners(t *testing.T) {
	attrs := []wipattr.Attribution{owned("a.go", "declared"), owned("b.go", "silent")}
	got := Project(attrs, liveSet("declared", "silent"),
		Options{Declared: map[string]bool{"declared": true}})

	if len(got.Active) != 2 {
		t.Fatalf("Active = %+v, want both owners projected", got.Active)
	}
	if got.Undeclared != 1 {
		t.Errorf("Undeclared = %d, want 1 (only the silent session)", got.Undeclared)
	}
}

// Fail-safe direction: an unknown session is NOT live, so its paths become reclaimable
// rather than a lease nobody can release. Over-reporting a lease would deadlock a peer
// behind a session that no longer exists.
func TestProjectUnknownSessionFailsToReclaimable(t *testing.T) {
	got := Project([]wipattr.Attribution{owned("a.go", "ghost")}, nil, Options{})
	if len(got.Active) != 0 {
		t.Fatalf("Active = %+v, want empty for an unknown session", got.Active)
	}
	if len(got.Reclaimable) != 1 || got.Reclaimable[0].Reason != ReclaimOwnerDead {
		t.Fatalf("Reclaimable = %+v, want one OWNER_DEAD row", got.Reclaimable)
	}
}

// Determinism: same input, same bytes out — leases ordered by id, trees and reclaimables
// sorted. A projection feeding a refusal must not reorder its evidence between reads.
func TestProjectIsDeterministic(t *testing.T) {
	attrs := []wipattr.Attribution{
		owned("z.go", "s2"), owned("a.go", "s1"), owned("m.go", "s2"),
		{File: "o2.go", State: wipattr.AttrOrphan},
		{File: "o1.go", State: wipattr.AttrOrphan},
	}
	first := Project(attrs, liveSet("s1", "s2"), Options{})
	second := Project(attrs, liveSet("s1", "s2"), Options{})
	if !reflect.DeepEqual(first, second) {
		t.Fatalf("nondeterministic:\n%+v\n%+v", first, second)
	}
	if first.Active[0].ID != "wip:s1" || first.Active[1].ID != "wip:s2" {
		t.Errorf("leases not ordered by id: %+v", first.Active)
	}
	if !reflect.DeepEqual(first.Active[1].Tree, []string{"m.go", "z.go"}) {
		t.Errorf("tree not sorted: %v", first.Active[1].Tree)
	}
	if first.Reclaimable[0].Path != "o1.go" {
		t.Errorf("reclaimable not sorted: %+v", first.Reclaimable)
	}
}

// Totality: every dirty path lands in exactly one bucket. A path that is neither active
// nor reclaimable would be a path the tree forgot, which is the failure this leaf exists
// to prevent.
func TestProjectIsTotalOverEveryPath(t *testing.T) {
	attrs := []wipattr.Attribution{
		owned("live.go", "s1"),
		owned("dead.go", "gone"),
		{File: "orphan.go", State: wipattr.AttrOrphan},
		{File: "shared.go", State: wipattr.AttrShared, Owners: []string{"s1", "gone"}},
	}
	got := Project(attrs, liveSet("s1"), Options{})

	seen := map[string]int{}
	for _, l := range got.Active {
		for _, p := range l.Tree {
			seen[p]++
		}
	}
	for _, r := range got.Reclaimable {
		seen[r.Path]++
	}
	for _, a := range attrs {
		if seen[a.File] == 0 {
			t.Errorf("path %q landed in no bucket", a.File)
		}
	}
	if len(seen) != 4 {
		t.Errorf("saw %d distinct paths, want 4: %+v", len(seen), seen)
	}
}

// An empty tree projects to an empty occupancy, not to a nil-deref or a phantom lease.
func TestProjectEmptyInput(t *testing.T) {
	got := Project(nil, nil, Options{})
	if len(got.Active) != 0 || len(got.Reclaimable) != 0 || got.Undeclared != 0 {
		t.Fatalf("empty input produced %+v", got)
	}
}
