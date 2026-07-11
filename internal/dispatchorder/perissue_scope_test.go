package dispatchorder

// Contract test for #3592: price per-issue file scope so same-lane disjoint
// issues run concurrently.
//
// The live wave (cmd/fak/dispatch_wave.go scoped loop) already narrows the tree
// it hands the pricer: for each ready issue that declares a `Paths:` contract it
// emits a Candidate whose Tree is the per-issue path set and whose Lane is a
// DISTINCT per-issue lease id (dispatchIssueLeaseID(lane, num)) — not the coarse
// dispatch lane. That distinct lease id is load-bearing: fileCollision() collides
// two candidates unconditionally when they share a non-empty Lane string, so the
// only way same-dispatch-lane issues get priced on their file geometry (rather
// than serialized by lane identity) is to give each a unique lease Lane and let
// TreesOverlap decide. These tests pin that contract at the pricer boundary so a
// regression that widens the fed tree — or collapses the per-issue lease back to
// the coarse lane — is caught here, in the package that owns the geometry.
//
// Promotion evidence (gen/next -> now): a live `fak dispatch wave` price row
// showing SafeConcurrency > lane-count on a real trunk with two disjoint-tree
// same-lane issues ready is what moves this from dogfood to default-on.
// Demotion/retirement: if per-issue `Paths:` provenance proves too sparse or
// noisy to trust, the wave falls back to the whole-lane tree (the undeclared
// path below) and this concurrency gain retires to lane-count — unchanged safety.
// Invalidating assumption: that a declared `Paths:` faithfully bounds the diff's
// real write footprint. If a worker writes outside its declared scope, two
// "disjoint" issues can still collide on disk; the tree lease is only as honest
// as the contract that produced it.

import "testing"

// TestPerIssueScopeSameLaneDisjointRunConcurrently is #3592 acceptance #1: two
// issues in the SAME dispatch lane whose declared Paths are disjoint are BOTH
// kept (SafeConcurrency=2), where a coarse whole-lane exclusive tree serialized
// them to width 1.
func TestPerIssueScopeSameLaneDisjointRunConcurrently(t *testing.T) {
	// Modeled exactly as the wave's scoped loop emits them: one dispatch lane
	// ("dispatchorder"), a distinct per-issue lease Lane, per-issue disjoint Tree.
	r := Plan(Input{NowUnix: base, Candidates: []Candidate{
		{ID: "iss-101", Key: "iss-101", Lane: "dispatchorder#101", Tree: []string{"internal/dispatchorder/**"}, Mode: "exclusive", UpdatedUnix: base - 100},
		{ID: "iss-202", Key: "iss-202", Lane: "dispatchorder#202", Tree: []string{"internal/dispatchtick/**"}, Mode: "exclusive", UpdatedUnix: base - 50},
	}})
	if r.KeepCount != 2 || r.CollisionCount != 0 || len(r.Collisions) != 0 {
		t.Fatalf("disjoint same-lane = keep %d collision %d edges %d, want 2/0/0",
			r.KeepCount, r.CollisionCount, len(r.Collisions))
	}
	if r.SafeConcurrency != 2 {
		t.Fatalf("SafeConcurrency = %d, want 2 (both disjoint-tree issues run at once)", r.SafeConcurrency)
	}
	if dispoOf(r, "iss-101") != DispKeep || dispoOf(r, "iss-202") != DispKeep {
		t.Fatalf("dispositions iss-101=%q iss-202=%q, want keep/keep",
			dispoOf(r, "iss-101"), dispoOf(r, "iss-202"))
	}
}

// TestPerIssueScopeSameLaneOverlapSerializes is #3592 acceptance #2 (part 1):
// same modeling, but overlapping declared Paths still serialize (COLLISION_RISK)
// — narrowing the tree never admits a genuine overlap.
func TestPerIssueScopeSameLaneOverlapSerializes(t *testing.T) {
	r := Plan(Input{NowUnix: base, Candidates: []Candidate{
		{ID: "iss-303", Key: "iss-303", Lane: "dispatchorder#303", Tree: []string{"internal/dispatchorder/**"}, Mode: "exclusive", UpdatedUnix: base - 100},
		{ID: "iss-404", Key: "iss-404", Lane: "dispatchorder#404", Tree: []string{"internal/dispatchorder/plan.go"}, Mode: "exclusive", UpdatedUnix: base - 50},
	}})
	if r.KeepCount != 1 || r.CollisionCount != 1 {
		t.Fatalf("overlapping same-lane = keep %d collision %d, want 1/1", r.KeepCount, r.CollisionCount)
	}
	if r.SafeConcurrency != 1 {
		t.Fatalf("SafeConcurrency = %d, want 1 (overlapping trees serialize)", r.SafeConcurrency)
	}
	if len(r.Collisions) != 1 || r.Collisions[0].Reason != ReasonCollisionRisk {
		t.Fatalf("collisions = %+v, want one %s edge", r.Collisions, ReasonCollisionRisk)
	}
}

// TestPerIssueScopeUndeclaredFallsBackToLaneTree is #3592 acceptance #2 (part 2):
// an issue that declares NO Paths falls back to the whole lane tree (grp.Tree),
// which conservatively overlaps a scoped sibling in that lane — unchanged
// serialize behavior. This is the exact disjoint pair from acceptance #1 with one
// issue's declared narrow tree replaced by the lane-tree fallback, so it shows
// declaration is precisely what unlocks the concurrency.
func TestPerIssueScopeUndeclaredFallsBackToLaneTree(t *testing.T) {
	r := Plan(Input{NowUnix: base, Candidates: []Candidate{
		// declared, narrow
		{ID: "iss-501", Key: "iss-501", Lane: "dispatchorder#501", Tree: []string{"internal/dispatchorder/plan.go"}, Mode: "exclusive", UpdatedUnix: base - 50},
		// undeclared -> lane-tree fallback (whole dispatchorder tree)
		{ID: "iss-502", Key: "iss-502", Lane: "dispatchorder#502", Tree: []string{"internal/dispatchorder/**"}, Mode: "exclusive", UpdatedUnix: base - 100},
	}})
	if r.KeepCount != 1 || r.CollisionCount != 1 {
		t.Fatalf("undeclared fallback = keep %d collision %d, want 1/1 (whole-lane tree collides conservatively)",
			r.KeepCount, r.CollisionCount)
	}
	if r.SafeConcurrency != 1 {
		t.Fatalf("SafeConcurrency = %d, want 1 (undeclared issue claims the whole lane)", r.SafeConcurrency)
	}
}
