package main

// Wave-seam guard: the collision PRICE and the worker's --lease-tree must keep being derived
// from the SAME path set until the lease record can carry two trees (#5854).
//
// The temptation this file exists to refute: dispatchtick.ExtractIssueRepoPaths
// (internal/dispatchtick/router.go:625, regex :49) scrapes issue prose and routinely yields a
// bare DIRECTORY token ("cmd/fak") next to a real file ("cmd/fak/foo.go"). dispatchorder's
// overlap test is literal prefix containment (internal/dispatchorder/dispatchorder.go:1154
// treeOverlap, after normalizeTree :1165 strips "/**"), so the bare directory contains EVERY
// sibling cmd/fak/*.go and the issue becomes a universal collider that serializes its lane.
// dispatchWaveOrderStamp's +1e6 core-source boost then sorts that collider FIRST, so the greedy
// safe set admits it ahead of everyone it blocks.
//
// The obvious fix is to narrow ONLY the pricer's input and let --lease-tree stay coarse, so a
// worker never loses the lease coverage it needs to edit an unnamed sibling. That prototype
// produces PHANTOM CONCURRENCY and must not ship. The wave prices narrow but still launches each
// worker with the coarse --lease-tree, and two runtime gates re-impose the coarse geometry:
//
//	cmd/fak/dispatch_tick.go:946   treeCollisionFromScopes(liveScopes, pick.Tree) -> COLLISION_RISK
//	cmd/fak/dispatch_tick.go:1119  acquireDispatchLaneLease(..., pick.Tree, ...)
//	                               -> internal/regionadmit/regionadmit.go:287 RungTreeOverlap,
//	                                  via the SAME dispatchorder.TreesOverlap
//
// So two issues price as disjoint, both spawn, and the second is refused at lease acquire --
// burning an account seat and a cooldown attempt for zero throughput. Measured offline over the
// cached backlog: priced safe set 14 -> 30 and collision edges 741 -> 276, but EFFECTIVE
// concurrency (the priced set walked under the coarse lease gate) is 14 -> 14 whole-backlog,
// cmd 7 -> 7, docs 7 -> 7, and gateway REGRESSES 3 -> 1 because narrowing reorders the greedy
// pick so an early admit blocks more successors on its coarse tree.
//
// The real fix is to carry BOTH geometries end to end -- an admission tree (narrow, for
// regionadmit.Decide and treeCollisionFromScopes) distinct from an authorization tree (coarse,
// for internal/gitgate cleanLeaseTree and dispatchPathInLeaseTree). That is a schema change on
// the live lease path, tracked in #5854, not a drive-by.
//
// These tests pin the CURRENT honest behavior (price and lease are one value, so the price does
// not over-promise) and keep the refutation executable, so the next attempt at pricer-only
// narrowing fails here rather than in a live wave.

import (
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dispatchorder"
	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
)

// collisionPaths is the pricer-only narrowing UNDER EVALUATION -- deliberately test-local, never
// wired into priceDispatchWavePayloadFiltered. It drops a directory token when the same issue
// already named something strictly inside it, because that token then tells the pricer nothing
// the narrower path does not. Kept here so the phantom-concurrency refutation below stays
// executable and re-measurable when #5854 lands the two-tree lease record.
func collisionPaths(paths []string) []string {
	if len(paths) < 2 {
		return paths
	}
	norm := make([]string, len(paths))
	for i, p := range paths {
		norm[i] = normalizeCollisionPath(p)
	}
	out := make([]string, 0, len(paths))
	for i, p := range paths {
		if !collisionPathWitnessed(norm, i) {
			out = append(out, p)
		}
	}
	if len(out) == 0 {
		return paths
	}
	return out
}

// collisionPathWitnessed reports whether some OTHER path in norm lies strictly inside norm[i]. A
// universal token ("", "**", "**/*") is never witnessed: it is an explicit whole-repo claim, not
// an incidental scrape artifact, and must keep colliding with everything.
func collisionPathWitnessed(norm []string, i int) bool {
	dir := norm[i]
	if dir == "" || dir == "**" || dir == "**/*" {
		return false
	}
	for j, other := range norm {
		if j == i || other == "" {
			continue
		}
		if strings.HasPrefix(other, dir+"/") {
			return true
		}
	}
	return false
}

// normalizeCollisionPath mirrors dispatchorder's unexported normalizeTree so the containment test
// here agrees with the containment test the pricer applies to the same tokens.
func normalizeCollisionPath(t string) string {
	t = strings.TrimSpace(strings.ReplaceAll(t, "\\", "/"))
	t = strings.TrimPrefix(t, "./")
	t = strings.TrimSuffix(t, "/")
	t = strings.TrimSuffix(t, "/**")
	t = strings.TrimSuffix(t, "/*")
	return strings.TrimSuffix(t, "/")
}

// dispatchWaveNarrowRouter builds a synthetic single-lane RouterPayload over cmd/fak so the tests
// below differ only in the declared per-issue Paths.
func dispatchWaveNarrowRouter(routes ...dispatchtick.IssueRoute) dispatchtick.RouterPayload {
	nums := make([]int, 0, len(routes))
	prio := map[int]int{}
	for _, r := range routes {
		nums = append(nums, r.Number)
		prio[r.Number] = dispatchtick.PriorityWeightDefault
	}
	return dispatchtick.RouterPayload{
		Lanes: map[string]dispatchtick.RouterLaneGroup{
			"cmdfak": {
				Count:      len(routes),
				StepBudget: 8,
				Issues:     nums,
				Tree:       []string{"cmd/fak/**"},
				Priority:   prio,
			},
		},
		Issues: routes,
	}
}

// TestDispatchWavePricesTheTreeItLeases is the invariant that keeps the wave's price HONEST: the
// tree handed to the collision pricer is the same tree handed to the worker as --lease-tree. A
// change that narrows one without the other makes SafeConcurrency claim parallelism the runtime
// lease gate will refuse, so it must fail here.
func TestDispatchWavePricesTheTreeItLeases(t *testing.T) {
	router := dispatchWaveNarrowRouter(
		// The universal-collider shape: a directory token plus a file inside it.
		dispatchtick.IssueRoute{Number: 111, Lane: "cmdfak", Paths: []string{"cmd/fak", "cmd/fak/foo.go"}, ExpectedSteps: 4},
		dispatchtick.IssueRoute{Number: 222, Lane: "cmdfak", Paths: []string{"cmd/fak/bar.go"}, ExpectedSteps: 4},
	)
	price, err := priceDispatchWavePayload(t.TempDir(), router, 2, 2, "", nil, 0, dispatchGoalProfileThroughput)
	if err != nil {
		t.Fatal(err)
	}
	// Coarse pricing: "cmd/fak" contains "cmd/fak/bar.go", so these two serialize -- and they
	// really do serialize at runtime, which is why a safe set of 1 is the honest answer.
	if price.SafeConcurrency != 1 {
		t.Fatalf("SafeConcurrency = %d, want 1: the pricer must see the same coarse tree the lease gate enforces (#5854)", price.SafeConcurrency)
	}
	cand, ok := dispatchWavePriceCandByIssue(price, 111)
	if !ok {
		t.Fatalf("no priced candidate row for issue 111")
	}
	want := []string{"cmd/fak", "cmd/fak/foo.go"}
	if strings.Join(cand.Tree, ",") != strings.Join(want, ",") {
		t.Fatalf("issue 111 row Tree = %v, want the coarse %v", cand.Tree, want)
	}
	args := dispatchTickArgsForLaunchTarget(cand)
	if joined := strings.Join(args, " "); !strings.Contains(joined, "--lease-tree cmd/fak,cmd/fak/foo.go") {
		t.Fatalf("launch args = %q, want --lease-tree to carry the coarse set cmd/fak,cmd/fak/foo.go", joined)
	}
}

// TestCollisionNarrowingWouldPricePhantomConcurrency is the executable refutation of the
// pricer-only fix: narrowing DOES double the priced safe set, and the runtime lease gate then
// refuses the extra pick, so the added concurrency is phantom. Both halves are asserted, so this
// test also documents exactly what #5854 has to change to make the win real.
func TestCollisionNarrowingWouldPricePhantomConcurrency(t *testing.T) {
	// The coarse trees are what the worker is launched with and what the lease gate adjudicates.
	coarse := map[string][]string{
		"cmdfak#111": {"cmd/fak", "cmd/fak/foo.go"},
		"cmdfak#222": {"cmd/fak/bar.go"},
	}
	ids := []string{"cmdfak#111", "cmdfak#222"}
	priced := func(narrow bool) []string {
		cands := make([]dispatchorder.Candidate, 0, len(ids))
		for _, id := range ids {
			tree := coarse[id]
			if narrow {
				tree = collisionPaths(tree)
			}
			cands = append(cands, dispatchorder.Candidate{
				ID: id, Key: id, Lane: id, Tree: tree, Mode: "exclusive",
			})
		}
		return dispatchorder.Plan(dispatchorder.Input{
			Candidates: cands, NowUnix: time.Now().Unix(), CooldownSeconds: -1,
		}).Keep
	}
	// admitted replays the runtime gate: both treeCollisionFromScopes (dispatch_tick.go:946) and
	// regionadmit.Decide (regionadmit.go:287, RungTreeOverlap) test the COARSE tree through the
	// very same dispatchorder.TreesOverlap, so a priced pick whose coarse tree hits a live lease
	// never actually runs.
	admitted := func(keep []string) int {
		var live [][]string
		n := 0
		for _, id := range keep {
			clash := false
			for _, l := range live {
				if dispatchorder.TreesOverlap(coarse[id], l) {
					clash = true
					break
				}
			}
			if clash {
				continue
			}
			live = append(live, coarse[id])
			n++
		}
		return n
	}
	if got, want := len(priced(false)), 1; got != want {
		t.Fatalf("coarse priced safe set = %d, want %d", got, want)
	}
	if got, want := admitted(priced(false)), 1; got != want {
		t.Fatalf("coarse effective concurrency = %d, want %d (coarse pricing does not over-promise)", got, want)
	}
	// Narrowing the pricer buys a second slot on paper...
	if got, want := len(priced(true)), 2; got != want {
		t.Fatalf("narrowed priced safe set = %d, want %d", got, want)
	}
	// ...that the lease gate immediately takes back. This equality IS the phantom: while it
	// holds, narrowing only the pricer spawns a worker that is refused at acquire.
	if got := admitted(priced(true)); got != 1 {
		t.Fatalf("narrowed effective concurrency = %d, want 1 while the lease still carries one tree; "+
			"if this now exceeds 1, the two-tree lease record from #5854 has landed and the pricer "+
			"may finally be narrowed", got)
	}
}

// TestCollisionPathsNarrowing pins the candidate narrowing itself, including the cases where it
// must NOT fire: a directory the issue named nothing inside of keeps its token even when an
// unrelated file-level path sits elsewhere in the same set.
func TestCollisionPathsNarrowing(t *testing.T) {
	for _, tc := range []struct {
		name string
		in   []string
		want []string
	}{
		{"witnessed directory drops", []string{"cmd/fak", "cmd/fak/foo.go"}, []string{"cmd/fak/foo.go"}},
		{"directory only survives", []string{"cmd/fak"}, []string{"cmd/fak"}},
		{"unwitnessed sibling directory survives", []string{"cmd/fak", "cmd/fak/foo.go", "internal/recall"}, []string{"cmd/fak/foo.go", "internal/recall"}},
		{"glob-suffixed directory narrows like the pricer sees it", []string{"cmd/fak/**", "cmd/fak/foo.go"}, []string{"cmd/fak/foo.go"}},
		{"disjoint files untouched", []string{"cmd/fak/foo.go", "internal/recall/x.go"}, []string{"cmd/fak/foo.go", "internal/recall/x.go"}},
		{"nested chain keeps only the leaf", []string{"cmd", "cmd/fak", "cmd/fak/foo.go"}, []string{"cmd/fak/foo.go"}},
		{"universal token never narrows", []string{"**", "cmd/fak/foo.go"}, []string{"**", "cmd/fak/foo.go"}},
		{"empty stays empty", nil, nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := collisionPaths(tc.in)
			if strings.Join(got, ",") != strings.Join(tc.want, ",") {
				t.Fatalf("collisionPaths(%v) = %v, want %v", tc.in, got, tc.want)
			}
		})
	}
}
