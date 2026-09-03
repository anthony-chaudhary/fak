package orgdebt

import (
	"path/filepath"
	"runtime"
	"testing"
)

func TestEvaluateClean(t *testing.T) {
	in := Input{
		Issues: []Issue{
			{
				Number: 101,
				Title:  "feat(cache): add bounded kv cache lease #101",
				Body: `## Current state
The cache is unbounded.

## Scope
Bound KV cache storage to 500MB.

## Done condition
Cache evicts LRU items above 500MB.

## Witness
go test ./internal/cache -run TestEviction

## Likely files
internal/cache/cache.go`,
				Labels: []string{"class:dev", "priority/P1"},
			},
		},
		Commits: []Commit{
			{
				SHA:          "abcdef123456",
				Subject:      "feat(cache): add bounded kv cache lease #101 (fak cache)",
				FilesTouched: []string{"internal/cache/cache.go", "internal/cache/cache_test.go"},
				LinesAdded:   120,
			},
		},
		InternalPackages: []string{"cache"},
		DeclaredLanes:    []string{"cache"},
	}

	res := Evaluate(in)
	if !res.OK {
		t.Fatalf("expected OK payload, got finding: %s, defects: %v", res.Finding, res.Reason)
	}
	debt, ok := res.Corpus[DebtKey].(int)
	if !ok || debt != 0 {
		t.Fatalf("expected org_debt 0, got %v", res.Corpus[DebtKey])
	}
	score, ok := res.Corpus["score"].(float64)
	if !ok || score != 100.0 {
		t.Fatalf("expected score 100, got %v", res.Corpus["score"])
	}
	grade, ok := res.Corpus["grade"].(string)
	if !ok || grade != "A" {
		t.Fatalf("expected grade A, got %v", res.Corpus["grade"])
	}
}

func TestEvaluateBacklogUnready(t *testing.T) {
	in := Input{
		Issues: []Issue{
			{
				Number: 202,
				Title:  "vague task without contract",
				Body:   "please fix the bug soon",
				Labels: []string{"bug"}, // missing class:* and priority/P*
			},
		},
	}
	res := Evaluate(in)
	if res.OK {
		t.Fatal("expected action payload for unready backlog issue")
	}
	debt := res.Corpus[DebtKey].(int)
	if debt == 0 {
		t.Fatalf("expected nonzero org_debt, got %d", debt)
	}
}

func TestEvaluateScopeOversize(t *testing.T) {
	in := Input{
		Issues: []Issue{
			{
				Number: 303,
				Title:  "monolithic mega feature",
				Body: `## Current state
None.

## Scope
- add primary engine runner
- implement parser rewrite
- rewrite gateway proxy wire
- migrate database schema

## Done condition
Everything done.

## Witness
make test

## Likely files
internal/engine/engine.go`,
				Labels: []string{"class:dev", "priority/P1"},
			},
		},
	}
	res := Evaluate(in)
	if res.OK {
		t.Fatal("expected oversize issue to fail gate")
	}
	found := false
	for _, kpi := range res.KPIs {
		if kpi.Key == "scope_oversize" && len(kpi.Defects) > 0 {
			found = true
			break
		}
	}
	if !found {
		t.Fatal("expected scope_oversize KPI to flag defect")
	}
}

func TestEvaluateMergeKnots(t *testing.T) {
	in := Input{
		Commits: []Commit{
			{
				SHA:     "150be815d",
				Subject: "Merge remote-tracking branch 'origin/main'",
				Parents: []string{"4c75de760", "32ed8b0f5"},
			},
		},
	}
	res := Evaluate(in)
	if res.OK {
		t.Fatal("expected merge knot to trigger action")
	}
	found := false
	for _, kpi := range res.KPIs {
		if kpi.Key == "merge_knots" && len(kpi.Defects) > 0 {
			found = true
			break
		}
	}
	if !found {
		t.Fatal("expected merge_knots KPI to flag defect")
	}
}

func TestEvaluateUnfannedSpines(t *testing.T) {
	in := Input{
		Commits: []Commit{
			{
				SHA:          "aaaa1111",
				Subject:      "feat(engine): add overlap runner without issue link (fak engine)",
				LinesAdded:   350,
				FilesTouched: []string{"internal/engine/overlap.go"},
			},
		},
	}
	res := Evaluate(in)
	if res.OK {
		t.Fatal("expected unfanned spine without issue reference to flag defect")
	}
	found := false
	for _, kpi := range res.KPIs {
		if kpi.Key == "unfanned_spines" && len(kpi.Defects) > 0 {
			found = true
			break
		}
	}
	if !found {
		t.Fatal("expected unfanned_spines KPI to flag defect")
	}
}

func TestEvaluateUnwitnessedCommits(t *testing.T) {
	in := Input{
		Commits: []Commit{
			{
				SHA:          "bbbb2222",
				Subject:      "random unformatted commit subject",
				FilesTouched: []string{"internal/gateway/wire.go"},
			},
			{
				SHA:          "cccc3333",
				Subject:      "feat(gateway): missing lane trailer",
				FilesTouched: []string{"internal/gateway/wire.go"},
			},
			{
				SHA:          "dddd4444",
				Subject:      "feat(gateway): claiming code but touching only markdown (fak gateway)",
				FilesTouched: []string{"README.md"},
			},
		},
	}
	res := Evaluate(in)
	if res.OK {
		t.Fatal("expected unwitnessed commits to trigger action")
	}
	found := false
	for _, kpi := range res.KPIs {
		if kpi.Key == "unwitnessed_commits" && len(kpi.Defects) >= 3 {
			found = true
			break
		}
	}
	if !found {
		t.Fatal("expected unwitnessed_commits KPI to flag at least 3 defects")
	}
}

func TestEvaluateLaneContention(t *testing.T) {
	in := Input{
		Commits: []Commit{
			{
				SHA:          "111111",
				Subject:      "feat(gateway): touch server state (fak gateway)",
				FilesTouched: []string{"internal/gateway/server_state.go"},
			},
			{
				SHA:          "222222",
				Subject:      "feat(top): also touch server state (fak gateway)",
				FilesTouched: []string{"internal/gateway/server_state.go"},
			},
		},
	}
	res := Evaluate(in)
	if res.OK {
		t.Fatal("expected concurrent collision on server_state.go to trigger defect")
	}
	found := false
	for _, kpi := range res.KPIs {
		if kpi.Key == "lane_contention" && len(kpi.Defects) > 0 {
			found = true
			break
		}
	}
	if !found {
		t.Fatal("expected lane_contention KPI to flag defect")
	}
}

func TestEvaluateUnindexedLeaves(t *testing.T) {
	in := Input{
		InternalPackages: []string{"gateway", "phantom_package"},
		DeclaredLanes:    []string{"gateway"},
	}
	res := Evaluate(in)
	if res.OK {
		t.Fatal("expected unindexed phantom_package to flag defect")
	}
	found := false
	for _, kpi := range res.KPIs {
		if kpi.Key == "unindexed_leaves" && len(kpi.Defects) > 0 {
			found = true
			break
		}
	}
	if !found {
		t.Fatal("expected unindexed_leaves KPI to flag defect")
	}
}

func TestScanWorkspace(t *testing.T) {
	_, filename, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("cannot determine test location")
	}
	root := filepath.Dir(filepath.Dir(filepath.Dir(filename)))
	in, err := ScanWorkspace(root)
	if err != nil {
		t.Fatalf("ScanWorkspace error: %v", err)
	}
	if len(in.InternalPackages) == 0 {
		t.Fatal("expected internal packages to be found")
	}
	if len(in.DeclaredLanes) == 0 {
		t.Fatal("expected declared lanes in dos.toml to be found")
	}
}
