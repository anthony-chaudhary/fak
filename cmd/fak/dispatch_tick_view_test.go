package main

// Witness for #1411: the unattended Go dispatch tick scopes its issue routing
// to the operator-marked `current` issue-view by default, `--view ""` disables
// the filter, and an unresolvable or empty view fail-softs to the full open
// backlog (parity with tools/issue_lane_router.py --view, 54b18e78).

import (
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
)

func stubDispatchIssueFetches(t *testing.T,
	view func(root, slug string, limit int) ([]dispatchtick.Issue, error),
	backlog func(root string, limit int) ([]dispatchtick.Issue, error)) {
	t.Helper()
	oldView, oldBacklog := dispatchFetchViewIssues, dispatchFetchBacklogIssues
	dispatchFetchViewIssues = view
	dispatchFetchBacklogIssues = backlog
	t.Cleanup(func() {
		dispatchFetchViewIssues = oldView
		dispatchFetchBacklogIssues = oldBacklog
	})
}

// dispatchViewTestIssue carries the full worker-ready contract body so it
// passes dispatchtick.IsDispatchable (mirrors the router_test fixture).
func dispatchViewTestIssue(number int) dispatchtick.Issue {
	body := strings.Join([]string{
		"## Parent context",
		"tools dispatch fixture",
		"## Current state",
		"Tools routing already recognizes the target lane.",
		"## Why this is next",
		"The tick must default to the operator-marked view.",
		"## Working spine",
		"Scoped tools issues enter the worker queue with a witness.",
		"## Work unit",
		"leaf",
		"## Expected steps",
		"3",
		"## Trigger",
		"The named-view drive marked this leaf current.",
		"## Batch policy",
		"One issue per view seam; reruns update by marker.",
		"## In scope",
		"Route this tools leaf and preserve its worker metadata.",
		"## Out of scope",
		"Do not alter unrelated lanes or dispatch policy.",
		"## Done condition",
		"The dispatch payload admits the scoped tools issue.",
		"## Witness",
		"go test ./cmd/fak",
		"## Acceptance gate",
		"go test ./cmd/fak",
		"## Lane",
		"tools",
		"## Path hints",
		"- `tools/issue_views.py`",
		"## Boundary notes",
		"Public issue only.",
		"## Closure binding",
		"Resolving commit cites #N and carries `(fak tools)`.",
	}, "\n\n")
	return dispatchtick.Issue{Number: number, Title: "fix(tools): scoped view leaf", Body: body}
}

func TestDispatchTickViewFlagDefaultsToCurrent(t *testing.T) {
	opts, _, code := parseDispatchTickFlags(io.Discard, []string{"--workspace", t.TempDir()})
	if code != 0 {
		t.Fatalf("parse exit = %d, want 0", code)
	}
	if opts.View != dispatchDefaultView || dispatchDefaultView != "current" {
		t.Fatalf("default view = %q (const %q), want current", opts.View, dispatchDefaultView)
	}
	opts, _, code = parseDispatchTickFlags(io.Discard, []string{"--workspace", t.TempDir(), "--view", ""})
	if code != 0 {
		t.Fatalf("parse exit = %d, want 0", code)
	}
	if opts.View != "" {
		t.Fatalf("explicit empty --view = %q, want disabled (empty)", opts.View)
	}
}

func TestDispatchScopedFetchUsesNamedViewSlice(t *testing.T) {
	stubDispatchIssueFetches(t,
		func(root, slug string, limit int) ([]dispatchtick.Issue, error) {
			if slug != "current" {
				t.Fatalf("view slug = %q, want current", slug)
			}
			return []dispatchtick.Issue{dispatchViewTestIssue(41)}, nil
		},
		func(root string, limit int) ([]dispatchtick.Issue, error) {
			t.Fatal("full-backlog fetch must not run when the view resolves")
			return nil, nil
		})
	issues, injected, err := dispatchFetchScopedIssues(t.TempDir(), io.Discard, "current", 1000)
	if err != nil {
		t.Fatalf("scoped fetch: %v", err)
	}
	if !injected || len(issues) != 1 || issues[0].Number != 41 {
		t.Fatalf("scoped fetch = injected %v issues %+v, want the injected view slice #41", injected, issues)
	}
}

func TestDispatchScopedFetchEmptyViewDisables(t *testing.T) {
	stubDispatchIssueFetches(t,
		func(root, slug string, limit int) ([]dispatchtick.Issue, error) {
			t.Fatal("view fetch must not run when --view is empty")
			return nil, nil
		},
		func(root string, limit int) ([]dispatchtick.Issue, error) {
			return []dispatchtick.Issue{{Number: 52, Title: "fix(tools): backlog leaf"}}, nil
		})
	issues, injected, err := dispatchFetchScopedIssues(t.TempDir(), io.Discard, "", 1000)
	if err != nil {
		t.Fatalf("scoped fetch: %v", err)
	}
	if injected || len(issues) != 1 || issues[0].Number != 52 {
		t.Fatalf("disabled view = injected %v issues %+v, want the full backlog #52", injected, issues)
	}
}

func TestDispatchScopedFetchFailSoftOnViewError(t *testing.T) {
	stubDispatchIssueFetches(t,
		func(root, slug string, limit int) ([]dispatchtick.Issue, error) {
			return nil, errors.New("unknown view \"current\" in issue-views.json")
		},
		func(root string, limit int) ([]dispatchtick.Issue, error) {
			return []dispatchtick.Issue{{Number: 63, Title: "fix(tools): backlog leaf"}}, nil
		})
	var warn strings.Builder
	issues, injected, err := dispatchFetchScopedIssues(t.TempDir(), &warn, "current", 1000)
	if err != nil {
		t.Fatalf("scoped fetch: %v", err)
	}
	if injected || len(issues) != 1 || issues[0].Number != 63 {
		t.Fatalf("fail-soft = injected %v issues %+v, want the full backlog #63", injected, issues)
	}
	if !strings.Contains(warn.String(), "using full open backlog") {
		t.Fatalf("fail-soft warn = %q, want the full-open-backlog notice", warn.String())
	}
}

func TestDispatchScopedFetchFailSoftWhenViewHasNoDispatchableIssue(t *testing.T) {
	epic := dispatchtick.Issue{
		Number: 7,
		Title:  "epic(tools): umbrella",
		Labels: []dispatchtick.IssueLabel{{Name: "epic"}},
	}
	stubDispatchIssueFetches(t,
		func(root, slug string, limit int) ([]dispatchtick.Issue, error) {
			return []dispatchtick.Issue{epic}, nil
		},
		func(root string, limit int) ([]dispatchtick.Issue, error) {
			return []dispatchtick.Issue{{Number: 74, Title: "fix(tools): backlog leaf"}}, nil
		})
	var warn strings.Builder
	issues, injected, err := dispatchFetchScopedIssues(t.TempDir(), &warn, "current", 1000)
	if err != nil {
		t.Fatalf("scoped fetch: %v", err)
	}
	if injected || len(issues) != 1 || issues[0].Number != 74 {
		t.Fatalf("no-dispatchable fail-soft = injected %v issues %+v, want the full backlog #74", injected, issues)
	}
	if !strings.Contains(warn.String(), "no dispatchable issues") {
		t.Fatalf("fail-soft warn = %q, want the no-dispatchable notice", warn.String())
	}
}

func TestDispatchViewQueryResolvesConfigSlug(t *testing.T) {
	root := t.TempDir()
	cfg := `{"repo": "anthony-chaudhary/fak", "views": [{"slug": "current", "query": "is:open label:current no:assignee"}]}`
	if err := os.MkdirAll(filepath.Join(root, ".github"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, ".github", "issue-views.json"), []byte(cfg), 0o644); err != nil {
		t.Fatal(err)
	}
	repo, query, err := dispatchViewQuery(root, "current")
	if err != nil {
		t.Fatalf("resolve current: %v", err)
	}
	if repo != "anthony-chaudhary/fak" || query != "is:open label:current no:assignee" {
		t.Fatalf("resolved = repo %q query %q", repo, query)
	}
	if _, _, err := dispatchViewQuery(root, "no-such-view"); err == nil {
		t.Fatal("unknown slug must error so the caller fail-softs")
	}
	if _, _, err := dispatchViewQuery(t.TempDir(), "current"); err == nil {
		t.Fatal("missing config must error so the caller fail-softs")
	}
}

// TestDispatchTickThreadsViewFlagToRouter proves the live wiring: a tick run
// through the CLI surface hands its --view (default current) to the native
// router seam before the lane pick.
func TestDispatchTickThreadsViewFlagToRouter(t *testing.T) {
	withDispatchJSONHelper(t, dispatchHappyHelper(t))
	oldRoute := dispatchRouteIssues
	oldView := dispatchTickView
	seen := "unset"
	dispatchRouteIssues = func(root string, _ io.Writer) (dispatchtick.RouterPayload, error) {
		seen = dispatchTickView
		return dispatchtick.RouterPayload{
			Schema: dispatchtick.RouterSchema,
			OK:     true,
			Lanes: map[string]dispatchtick.RouterLaneGroup{
				"docs": {Tree: []string{"docs/**"}, Issues: []int{12}, Count: 1},
			},
		}, nil
	}
	t.Cleanup(func() {
		dispatchRouteIssues = oldRoute
		dispatchTickView = oldView
	})

	root := t.TempDir()
	runDispatchAt("tick", "--workspace", root, "--lane", "docs", "--no-refresh", "--no-loop-ledger", "--json")
	if seen != dispatchDefaultView {
		t.Fatalf("router saw view %q, want the default %q", seen, dispatchDefaultView)
	}

	seen = "unset"
	runDispatchAt("tick", "--workspace", root, "--lane", "docs", "--view", "", "--no-refresh", "--no-loop-ledger", "--json")
	if seen != "" {
		t.Fatalf("router saw view %q after --view \"\", want disabled (empty)", seen)
	}
}
