package dispatchtick

import (
	"reflect"
	"testing"
)

// TestDuplicateRiskCache witnesses the #4171 memo: two RouteIssues calls over an
// unchanged routable backlog run the O(n^2) DuplicateRiskIssueNumbers scan
// exactly once (the second is a hash hit), the flagged set matches the plain
// scan, the full RouterPayload is identical to the uncached path, and mutating
// one issue's body at the same set size invalidates the memo (counter -> 2)
// with the recomputed set still correct.
func TestDuplicateRiskCache(t *testing.T) {
	issues := []Issue{
		routerIssue(101, "gateway: retry budget leak", nil, duplicateRiskIssueBody("internal/gateway/http.go", "The retry budget leak fixture is deduped before worker launch.")),
		routerIssue(102, "gateway: retry budget accounting", nil, duplicateRiskIssueBody("internal/gateway/http.go", "The retry budget accounting fixture is deduped before worker launch.")),
		routerIssue(103, "gateway: cache owner report", nil, duplicateRiskIssueBody("internal/gateway/cache.go", "The cache owner report remains a safe candidate.")),
	}
	for _, issue := range issues {
		if !IsDispatchable(issue, BlockedByHumanLabel) {
			t.Fatalf("fixture issue #%d is not dispatchable; routable slice would not equal the fixture slice", issue.Number)
		}
	}
	cache := &DuplicateRiskCache{}
	route := func(c *DuplicateRiskCache) RouterPayload {
		return RouteIssues(RouterInput{
			Workspace:          "C:/work/fak",
			Taxonomy:           routerTestTaxonomy,
			IssueLimit:         1000,
			Issues:             issues,
			DuplicateRiskCache: c,
		})
	}

	first := route(cache)
	second := route(cache)
	if got := cache.Recomputes(); got != 1 {
		t.Fatalf("recomputes after two unchanged routes = %d, want 1 (second call must be a memo hit)", got)
	}
	if got := cache.Hits(); got != 1 {
		t.Fatalf("hits after two unchanged routes = %d, want 1", got)
	}
	want := DuplicateRiskIssueNumbers(issues)
	if got := flaggedDuplicateNumbers(first); !reflect.DeepEqual(got, want) {
		t.Fatalf("first cached flagged set = %v, want %v (plain DuplicateRiskIssueNumbers)", got, want)
	}
	if got := flaggedDuplicateNumbers(second); !reflect.DeepEqual(got, want) {
		t.Fatalf("second cached flagged set = %v, want %v (memo hit changed the verdict)", got, want)
	}
	uncached := route(nil)
	if !reflect.DeepEqual(first, uncached) || !reflect.DeepEqual(second, uncached) {
		t.Fatalf("cached RouterPayload diverged from the uncached path over identical input")
	}

	// Same set size, edited body: the content hash must move and force one recompute.
	issues[2].Body += "\n\nEdited: the cache owner report grew a follow-up paragraph."
	if !IsDispatchable(issues[2], BlockedByHumanLabel) {
		t.Fatalf("mutated fixture issue #%d fell out of the routable slice; body edit broke the contract", issues[2].Number)
	}
	third := route(cache)
	if got := cache.Recomputes(); got != 2 {
		t.Fatalf("recomputes after body edit = %d, want 2 (edit at same set size must invalidate)", got)
	}
	want = DuplicateRiskIssueNumbers(issues)
	if got := flaggedDuplicateNumbers(third); !reflect.DeepEqual(got, want) {
		t.Fatalf("post-edit flagged set = %v, want %v", got, want)
	}
	if !reflect.DeepEqual(third, route(nil)) {
		t.Fatalf("post-edit cached RouterPayload diverged from the uncached path")
	}
}

// flaggedDuplicateNumbers projects a RouterPayload back onto the
// DuplicateRiskIssueNumbers map shape: the issues skipped for duplicate risk.
func flaggedDuplicateNumbers(p RouterPayload) map[int]bool {
	out := map[int]bool{}
	for _, skipped := range p.SkippedHumanBlocked {
		if skipped.Reason == ReasonDuplicateRisk {
			out[skipped.Number] = true
		}
	}
	return out
}
