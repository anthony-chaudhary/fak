package main

// Cmd-layer closure witness for #3592: price per-issue file scope so same-lane
// disjoint issues run concurrently.
//
// internal/dispatchorder/perissue_scope_test.go already pins the geometry at the
// pricer boundary (given per-issue Candidate.Tree, disjoint same-lane trees keep
// concurrently). But that test hand-builds the Candidates. It does NOT witness the
// wave seam that #3592 actually changed: that a routed issue's declared `Paths:`
// survives dispatchRouteIssues -> priceDispatchWavePayload into the priced rows,
// that the live pricer then returns SafeConcurrency=2 for two same-dispatch-lane
// disjoint issues, and that the operator-facing price rows (incl. JSON) expose the
// per-issue tree rather than the coarse lane tree. These tests drive the real
// priceDispatchWavePayload over a synthetic RouterPayload so a regression that
// widens the fed tree back to grp.Tree -- or drops Paths on the way through the
// wave -- is caught in the command that owns the seam.
//
// Promotion evidence (gen/next -> now): a live `fak dispatch wave` price row on a
// real trunk showing safe_concurrency > lane count with two disjoint-tree same-lane
// issues ready. Demotion/retirement: if per-issue `Paths:` provenance is too sparse
// to trust, the wave keeps falling back to grp.Tree (the undeclared case below) and
// the gain retires to lane-count with unchanged safety. Invalidating assumption:
// that a declared `Paths:` faithfully bounds the diff's real write footprint -- a
// worker that writes outside its declared scope can still collide two "disjoint"
// issues on disk; the tree lease is only as honest as the contract that produced it.

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/dispatchorder"
	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
)

// dispatchWavePriceCandByIssue finds the priced candidate row for an issue number.
func dispatchWavePriceCandByIssue(price dispatchWavePrice, issue int) (dispatchWaveCandidate, bool) {
	for _, c := range price.Candidates {
		if c.Issue == issue {
			return c, true
		}
	}
	return dispatchWaveCandidate{}, false
}

// TestDispatchWavePricePerIssueScopeDisjointConcurrent is #3592 acceptance #1 at the
// wave seam: two issues in the SAME dispatch lane whose declared `Paths:` are disjoint
// are BOTH kept (SafeConcurrency=2) and both selected to run, where the coarse
// whole-lane exclusive tree serialized the lane to width 1. It also witnesses that the
// per-issue tree (not the lane tree) reaches the priced rows and the JSON readout.
func TestDispatchWavePricePerIssueScopeDisjointConcurrent(t *testing.T) {
	router := dispatchtick.RouterPayload{
		Lanes: map[string]dispatchtick.RouterLaneGroup{
			"dispatchorder": {
				Count:      2,
				StepBudget: 8,
				Issues:     []int{101, 202},
				Tree:       []string{"internal/dispatchorder/**"},
				Priority: map[int]int{
					101: dispatchtick.PriorityWeightDefault,
					202: dispatchtick.PriorityWeightDefault,
				},
			},
		},
		Issues: []dispatchtick.IssueRoute{
			{Number: 101, Lane: "dispatchorder", Paths: []string{"internal/dispatchorder/plan.go"}, ExpectedSteps: 4},
			{Number: 202, Lane: "dispatchorder", Paths: []string{"internal/dispatchtick/router.go"}, ExpectedSteps: 4},
		},
	}
	price, err := priceDispatchWavePayload(t.TempDir(), router, 2, 2, "", nil, 0, dispatchGoalProfileThroughput)
	if err != nil {
		t.Fatal(err)
	}
	if price.SafeConcurrency != 2 {
		t.Fatalf("SafeConcurrency = %d, want 2 (two disjoint-tree same-lane issues run at once)", price.SafeConcurrency)
	}
	if len(price.RunTargets) != 2 {
		t.Fatalf("run targets = %d (%+v), want 2 selected", len(price.RunTargets), price.RunTargets)
	}
	// Both selected candidates share the dispatch lane, so the extra beyond the first
	// is same-lane parallelism the coarse-tree wave could never produce.
	if price.SameLaneParallelism != 1 {
		t.Fatalf("SameLaneParallelism = %d, want 1 (both run in dispatch lane 'dispatchorder')", price.SameLaneParallelism)
	}
	if price.ScopedCount != 2 || price.UnscopedCount != 0 {
		t.Fatalf("scoped/unscoped = %d/%d, want 2/0", price.ScopedCount, price.UnscopedCount)
	}
	// The priced rows must carry the per-issue tree, not the coarse lane tree.
	for _, issue := range []int{101, 202} {
		cand, ok := dispatchWavePriceCandByIssue(price, issue)
		if !ok {
			t.Fatalf("no priced candidate row for issue %d", issue)
		}
		if !cand.Scoped {
			t.Fatalf("issue %d row Scoped=false, want true (declared Paths)", issue)
		}
		if len(cand.Tree) != 1 {
			t.Fatalf("issue %d row Tree = %v, want the single per-issue path", issue, cand.Tree)
		}
		if strings.Contains(cand.Tree[0], "**") {
			t.Fatalf("issue %d row Tree = %v carries the whole-lane glob, want per-issue path", issue, cand.Tree)
		}
	}
	// JSON readout exposes the per-issue tree (issue 202's distinct dispatchtick path,
	// which the whole-lane dispatchorder tree does not contain).
	blob, err := json.Marshal(price)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(blob), "internal/dispatchtick/router.go") {
		t.Fatalf("price JSON does not expose the per-issue tree; got %s", blob)
	}
}

// TestDispatchWavePricePerIssueScopeOverlapSerializes is #3592 acceptance #2 (part 1)
// at the wave seam: two same-dispatch-lane issues whose declared `Paths:` overlap still
// serialize (one COLLISION_RISK edge, SafeConcurrency=1) -- narrowing the fed tree never
// admits a genuine overlap. The collision row still carries the per-issue tree.
func TestDispatchWavePricePerIssueScopeOverlapSerializes(t *testing.T) {
	router := dispatchtick.RouterPayload{
		Lanes: map[string]dispatchtick.RouterLaneGroup{
			"dispatchorder": {
				Count:      2,
				StepBudget: 8,
				Issues:     []int{303, 404},
				Tree:       []string{"internal/dispatchorder/**"},
				Priority: map[int]int{
					303: dispatchtick.PriorityWeightDefault,
					404: dispatchtick.PriorityWeightDefault,
				},
			},
		},
		Issues: []dispatchtick.IssueRoute{
			{Number: 303, Lane: "dispatchorder", Paths: []string{"internal/dispatchorder/**"}, ExpectedSteps: 4},
			{Number: 404, Lane: "dispatchorder", Paths: []string{"internal/dispatchorder/plan.go"}, ExpectedSteps: 4},
		},
	}
	price, err := priceDispatchWavePayload(t.TempDir(), router, 2, 2, "", nil, 0, dispatchGoalProfileThroughput)
	if err != nil {
		t.Fatal(err)
	}
	if price.SafeConcurrency != 1 {
		t.Fatalf("SafeConcurrency = %d, want 1 (overlapping declared trees serialize)", price.SafeConcurrency)
	}
	if len(price.RunTargets) != 1 {
		t.Fatalf("run targets = %d (%+v), want 1 (overlap serializes to width 1)", len(price.RunTargets), price.RunTargets)
	}
	if price.SameLaneParallelism != 0 {
		t.Fatalf("SameLaneParallelism = %d, want 0 (overlap keeps only one)", price.SameLaneParallelism)
	}
	if len(price.Collisions) != 1 || price.Collisions[0].Reason != dispatchorder.ReasonCollisionRisk {
		t.Fatalf("collisions = %+v, want one %s edge", price.Collisions, dispatchorder.ReasonCollisionRisk)
	}
	// Both candidates are still scoped to their declared per-issue tree even though they
	// collide -- the overlap is decided on that tree, not the coarse lane tree.
	for _, issue := range []int{303, 404} {
		cand, ok := dispatchWavePriceCandByIssue(price, issue)
		if !ok {
			t.Fatalf("no priced candidate row for issue %d", issue)
		}
		if !cand.Scoped {
			t.Fatalf("issue %d row Scoped=false, want true (declared Paths)", issue)
		}
	}
}

// TestDispatchWavePriceUndeclaredFallsBackToLaneTree is #3592 acceptance #2 (part 2) at
// the wave seam: an issue that declares NO `Paths:` falls back to the whole lane tree
// (grp.Tree) and is priced unscoped -- the conservative, unchanged behavior. This is the
// safety floor the concurrency gain retires to if per-issue provenance is absent.
func TestDispatchWavePriceUndeclaredFallsBackToLaneTree(t *testing.T) {
	laneTree := []string{"internal/dispatchorder/**"}
	router := dispatchtick.RouterPayload{
		Lanes: map[string]dispatchtick.RouterLaneGroup{
			"dispatchorder": {
				Count:      1,
				StepBudget: 4,
				Issues:     []int{505},
				Tree:       laneTree,
				Priority:   map[int]int{505: dispatchtick.PriorityWeightDefault},
			},
		},
		Issues: []dispatchtick.IssueRoute{
			// No Paths declared -> lane-tree fallback.
			{Number: 505, Lane: "dispatchorder", ExpectedSteps: 4},
		},
	}
	price, err := priceDispatchWavePayload(t.TempDir(), router, 2, 2, "", nil, 0, dispatchGoalProfileThroughput)
	if err != nil {
		t.Fatal(err)
	}
	if price.ScopedCount != 0 || price.UnscopedCount != 1 {
		t.Fatalf("scoped/unscoped = %d/%d, want 0/1 (undeclared issue is unscoped)", price.ScopedCount, price.UnscopedCount)
	}
	cand, ok := dispatchWavePriceCandByIssue(price, 505)
	if !ok {
		t.Fatalf("no priced candidate row for issue 505")
	}
	if cand.Scoped {
		t.Fatalf("issue 505 row Scoped=true, want false (no declared Paths)")
	}
	if len(cand.Tree) != len(laneTree) || cand.Tree[0] != laneTree[0] {
		t.Fatalf("issue 505 row Tree = %v, want the lane tree %v", cand.Tree, laneTree)
	}
}
