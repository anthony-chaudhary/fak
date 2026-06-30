package dispatchtick

import (
	"strings"
	"testing"
)

var routerTestTaxonomy = LaneTaxonomy{
	Concurrent: []string{
		"gateway", "compute", "docs", "tools", "experiments", "model", "abi",
		"bench", "ci", "sessionimage", "promptmmu", "devindex", "metrics", "examples",
	},
	Trees: map[string][]string{
		"gateway":      {"internal/gateway/**"},
		"compute":      {"internal/compute/**"},
		"docs":         {"docs/**"},
		"tools":        {"tools/**"},
		"experiments":  {"experiments/**"},
		"model":        {"internal/model/**"},
		"abi":          {"internal/abi/**"},
		"bench":        {"internal/bench/**"},
		"ci":           {".github/**"},
		"sessionimage": {"internal/sessionimage/**"},
		"promptmmu":    {"internal/promptmmu/**"},
		"devindex":     {"internal/devindex/**"},
		"metrics":      {"internal/metrics/**"},
		"examples":     {"examples/**"},
	},
}

func routerIssue(number int, title string, labels []string, body string) Issue {
	labs := make([]IssueLabel, 0, len(labels))
	for _, label := range labels {
		labs = append(labs, IssueLabel{Name: label})
	}
	return Issue{Number: number, Title: title, Body: body, Labels: labs}
}

func routeTestIssue(issue Issue) IssueRoute {
	return RouteIssue(issue, routerTestTaxonomy, RouteOptions{})
}

func TestRouterPathNormalization(t *testing.T) {
	if got := PathMatchesLane("fak/internal/gateway/x.go", routerTestTaxonomy.Trees); len(got) != 1 || got[0] != "gateway" {
		t.Fatalf("doc-link path lanes = %#v, want gateway", got)
	}
	if got := PathMatchesLane("internal/gateway/x.go", routerTestTaxonomy.Trees); len(got) != 1 || got[0] != "gateway" {
		t.Fatalf("real-layout path lanes = %#v, want gateway", got)
	}
	if got := ExtractRepoPaths("scheduled by `.github/workflows/security-audit.yml`"); len(got) != 1 || got[0] != ".github/workflows/security-audit.yml" {
		t.Fatalf("dot-root paths = %#v, want .github workflow", got)
	}
	if got := ExtractRepoPaths("see x.github/y and mytools/x"); len(got) != 0 {
		t.Fatalf("embedded roots matched unexpectedly: %#v", got)
	}
}

func TestRouterRungs(t *testing.T) {
	tests := []struct {
		name       string
		issue      Issue
		lane       string
		confidence string
	}{
		{"exact scope", routerIssue(1, "fix(gateway): silent fallback", nil, ""), "gateway", "exact-scope"},
		{"alias scope", routerIssue(2, "gpu(cuda): residency budget", nil, ""), "compute", "alias"},
		{"label fallback", routerIssue(3, "GPU server benchmark", []string{"gpu"}, ""), "compute", "label"},
		{"keyword fallback", routerIssue(4, "promptmmu rung 6", nil, ""), "promptmmu", "keyword"},
		{"path override", routerIssue(5, "docs(readme): wrong scope", nil, "bug is in tools/issue_triage.py"), "tools", "path-confirmed"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := routeTestIssue(tt.issue)
			if got.Lane != tt.lane || got.Confidence != tt.confidence {
				t.Fatalf("route = lane %q confidence %q, want %q/%q (%+v)", got.Lane, got.Confidence, tt.lane, tt.confidence, got)
			}
		})
	}
}

func TestRouterCarriesPathScope(t *testing.T) {
	got := routeTestIssue(routerIssue(6, "fix(gateway): split handlers", nil,
		"touches fak/internal/gateway/http.go and fak/internal/gateway/mcp.go"))
	if got.Lane != "gateway" || got.Confidence != "path-confirmed" {
		t.Fatalf("route = %+v, want path-confirmed gateway", got)
	}
	want := []string{"internal/gateway/http.go", "internal/gateway/mcp.go"}
	if len(got.Paths) != len(want) {
		t.Fatalf("paths = %#v, want %#v", got.Paths, want)
	}
	for i := range want {
		if got.Paths[i] != want[i] {
			t.Fatalf("paths = %#v, want %#v", got.Paths, want)
		}
	}
}

func TestRouterCarriesAgentSchedulingMetadata(t *testing.T) {
	body := strings.Join([]string{
		"## Work unit",
		"leaf",
		"## Expected steps",
		"5",
		"## Trigger",
		"Gateway report crossed the retry-error threshold.",
		"## Batch policy",
		"One issue per gateway retry class; reruns update by marker.",
		"## Path hints",
		"- `internal/gateway/http.go`",
	}, "\n\n")
	issue := routerIssue(12, "gateway: retry class", nil, body)
	got := routeTestIssue(issue)
	if got.WorkUnit != "leaf" || got.ExpectedSteps != 5 {
		t.Fatalf("route metadata = work_unit %q steps %d, want leaf/5 (%+v)", got.WorkUnit, got.ExpectedSteps, got)
	}
	if got.Trigger != "Gateway report crossed the retry-error threshold." {
		t.Fatalf("trigger = %q", got.Trigger)
	}
	if got.BatchPolicy != "One issue per gateway retry class; reruns update by marker." {
		t.Fatalf("batch policy = %q", got.BatchPolicy)
	}

	p := RouteIssues(RouterInput{
		Workspace:  "C:/work/fak",
		Taxonomy:   routerTestTaxonomy,
		IssueLimit: 1000,
		Issues:     []Issue{issue},
	})
	grp := p.Lanes["gateway"]
	if p.Counts.RoutedStepBudget != 5 || grp.StepBudget != 5 {
		t.Fatalf("step budget = counts %d lane %d, want 5/5", p.Counts.RoutedStepBudget, grp.StepBudget)
	}
	if grp.WorkUnits[12] != "leaf" || grp.IssueSteps[12] != 5 {
		t.Fatalf("lane metadata = work_units=%+v issue_steps=%+v, want issue 12 leaf/5", grp.WorkUnits, grp.IssueSteps)
	}
}

func TestRouterExclusiveAndAmbiguity(t *testing.T) {
	excl := routeTestIssue(routerIssue(7, "abi: hoist the public ABI surface", nil, ""))
	if excl.Lane != "" || excl.Confidence != "none" || excl.UnroutedReason == "" {
		t.Fatalf("exclusive route = %+v, want unrouted operator-gated", excl)
	}

	body := "touches fak/internal/gateway/a.go and fak/internal/compute/b.go"
	got := routeTestIssue(routerIssue(8, "fix(compute): shared", nil, body))
	if got.Lane != "compute" || !got.SignalConflict {
		t.Fatalf("ambiguous scoped route = %+v, want compute conflict", got)
	}
}

func TestRouterPayloadCountsAndVerdicts(t *testing.T) {
	routes := []IssueRoute{
		routeTestIssue(routerIssue(1, "fix(gateway): a", nil, "")),
		routeTestIssue(routerIssue(2, "fix(gateway): b", nil, "")),
		routeTestIssue(routerIssue(3, "Merge branches", nil, "")),
	}
	p := BuildRouterPayload(RouterPayloadInput{
		Workspace:       "C:/work/fak",
		Routes:          routes,
		Trees:           routerTestTaxonomy.Trees,
		MaxUnroutedFrac: 0.25,
		Coverage:        RouterCoverage{Complete: true, Notes: []string{}},
	})
	if p.OK || p.Verdict != "ACTION" || p.Finding != "high_unrouted" {
		t.Fatalf("payload verdict = %s/%s ok=%v, want ACTION/high_unrouted/false", p.Verdict, p.Finding, p.OK)
	}
	if p.Counts.Routed != 2 || p.Counts.Unrouted != 1 || p.Lanes["gateway"].Count != 2 {
		t.Fatalf("payload counts = %+v lanes=%+v", p.Counts, p.Lanes)
	}

	p = BuildRouterPayload(RouterPayloadInput{
		Workspace: "C:/work/fak",
		Routes:    routes[:2],
		Trees:     routerTestTaxonomy.Trees,
		Coverage:  RouterCoverage{Complete: false, Notes: []string{"cap"}},
	})
	if p.OK || p.Verdict != "ACTION" || p.Finding != "incomplete_coverage" {
		t.Fatalf("truncated payload = %s/%s ok=%v, want incomplete coverage ACTION", p.Verdict, p.Finding, p.OK)
	}
}

func TestRouterRouteIssuesSkipsNonDispatchable(t *testing.T) {
	p := RouteIssues(RouterInput{
		Workspace:  "C:/work/fak",
		Taxonomy:   routerTestTaxonomy,
		IssueLimit: 1000,
		Issues: []Issue{
			routerIssue(1, "fix(gateway): a", nil, ""),
			routerIssue(2, "epic(gateway): umbrella", []string{"epic"}, ""),
			routerIssue(3, "needs a filing", []string{BlockedByHumanLabel}, ""),
			routerIssue(4, "idea: maybe do a thing", []string{"idea-scout"}, ""),
			routerIssue(5, "guard complaint [false-positive]", []string{"guard-complaint"}, ""),
			routerIssue(6, "needs scope", []string{"triage-only"}, ""),
			routerIssue(7, "dispatch-log-audit: auth wall", []string{"dispatch"}, "- dispatchability: `triage_only`"),
			routerIssue(8, "gateway: decompose serving follow-ups", nil, "## Work unit\n\nepic\n\n## Working spine\n\nBreak the serving program into leaves."),
			routerIssue(9, "research: study cache prior art", []string{"research"}, ""),
			routerIssue(10, "gateway: oversized leaf", nil, "## Work unit\n\nleaf\n\n## Expected steps\n\n12\n\n## Path hints\n\n- `internal/gateway/http.go`"),
		},
	})
	if p.Counts.Routed != 1 || p.Counts.SkippedHumanBlocked != 9 {
		t.Fatalf("route issues counts = %+v skipped=%+v, want routed=1 skipped=9", p.Counts, p.SkippedHumanBlocked)
	}
	wantReasons := map[string]int{
		"BLOCKED_BY_HUMAN":               1,
		"ISSUE_NOT_DISPATCH_LEAF":        2,
		"ISSUE_OVERSIZED_EXPECTED_STEPS": 1,
		"ISSUE_TRIAGE_ONLY":              5,
	}
	if !sameStringIntMap(p.Counts.SkippedByReason, wantReasons) {
		t.Fatalf("skipped reasons = %#v, want %#v", p.Counts.SkippedByReason, wantReasons)
	}
	if strings.Contains(p.Reason, "human-blocked skipped") {
		t.Fatalf("router reason kept legacy human-blocked wording: %q", p.Reason)
	}
	if skipped := skippedIssueByNumber(p.SkippedHumanBlocked, 10); skipped.Reason != "ISSUE_OVERSIZED_EXPECTED_STEPS" || skipped.ExpectedSteps != 12 {
		t.Fatalf("oversized skipped issue = %+v, want reason ISSUE_OVERSIZED_EXPECTED_STEPS steps=12", skipped)
	}
	if skipped := skippedIssueByNumber(p.SkippedHumanBlocked, 8); skipped.Reason != "ISSUE_NOT_DISPATCH_LEAF" || skipped.WorkUnit != "epic" {
		t.Fatalf("non-leaf skipped issue = %+v, want non-dispatch leaf epic", skipped)
	}
	if p.Lanes["gateway"].Issues[0] != 1 {
		t.Fatalf("gateway issues = %#v, want #1", p.Lanes["gateway"].Issues)
	}
}

func TestRouterKeepsSmallExpectedStepLeafDispatchable(t *testing.T) {
	issue := routerIssue(11, "gateway: scoped leaf", nil, "## Work unit\n\nleaf\n\n## Expected steps\n\n4\n\n## Path hints\n\n- `internal/gateway/http.go`")
	if !IsDispatchable(issue, BlockedByHumanLabel) {
		t.Fatalf("small expected-step leaf was not dispatchable")
	}
}

func skippedIssueByNumber(skipped []SkippedIssue, number int) SkippedIssue {
	for _, issue := range skipped {
		if issue.Number == number {
			return issue
		}
	}
	return SkippedIssue{}
}

func sameStringIntMap(got, want map[string]int) bool {
	if len(got) != len(want) {
		return false
	}
	for key, wantValue := range want {
		if got[key] != wantValue {
			return false
		}
	}
	return true
}
