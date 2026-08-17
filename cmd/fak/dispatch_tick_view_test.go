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
	"time"

	"github.com/anthony-chaudhary/fak/internal/dispatchcache"
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

// TestDispatchViewFailsoftSignalClassifiesReason witnesses #4172: the two fail-soft
// branches now record a package-level structured signal carrying the view slug, a
// closed reason class (view_unreadable vs view_empty), and a process-lifetime count --
// distinct from the free-text WARN the sibling tests assert. A resolved, dispatchable
// view records nothing, so an operator can tell a still-scoped tick from one that has
// silently dropped to the full backlog. Assertions are delta-based over the process
// global so ordering with other tests never matters.
func TestDispatchViewFailsoftSignalClassifiesReason(t *testing.T) {
	dispatchable := func(root string, limit int) ([]dispatchtick.Issue, error) {
		return []dispatchtick.Issue{{Number: 90, Title: "fix(tools): backlog leaf"}}, nil
	}

	t.Run("view_unreadable on fetch error", func(t *testing.T) {
		before := dispatchViewFailsoftSignal().Count
		stubDispatchIssueFetches(t,
			func(root, slug string, limit int) ([]dispatchtick.Issue, error) {
				return nil, errors.New("unknown view \"current\" in issue-views.json")
			}, dispatchable)
		if _, _, err := dispatchFetchScopedIssues(t.TempDir(), io.Discard, "current", 1000); err != nil {
			t.Fatalf("scoped fetch: %v", err)
		}
		got := dispatchViewFailsoftSignal()
		if got.Count != before+1 {
			t.Fatalf("count=%d, want %d (one fail-soft recorded)", got.Count, before+1)
		}
		if got.Reason != dispatchViewFailsoftUnreadable {
			t.Fatalf("reason=%q, want %q", got.Reason, dispatchViewFailsoftUnreadable)
		}
		if got.View != "current" {
			t.Fatalf("view=%q, want current", got.View)
		}
	})

	t.Run("view_empty when no dispatchable issue", func(t *testing.T) {
		epic := dispatchtick.Issue{
			Number: 7,
			Title:  "epic(tools): umbrella",
			Labels: []dispatchtick.IssueLabel{{Name: "epic"}},
		}
		before := dispatchViewFailsoftSignal().Count
		stubDispatchIssueFetches(t,
			func(root, slug string, limit int) ([]dispatchtick.Issue, error) {
				return []dispatchtick.Issue{epic}, nil
			}, dispatchable)
		if _, _, err := dispatchFetchScopedIssues(t.TempDir(), io.Discard, "current", 1000); err != nil {
			t.Fatalf("scoped fetch: %v", err)
		}
		got := dispatchViewFailsoftSignal()
		if got.Count != before+1 {
			t.Fatalf("count=%d, want %d (one fail-soft recorded)", got.Count, before+1)
		}
		if got.Reason != dispatchViewFailsoftEmpty {
			t.Fatalf("reason=%q, want %q", got.Reason, dispatchViewFailsoftEmpty)
		}
	})

	t.Run("no signal when the view resolves", func(t *testing.T) {
		before := dispatchViewFailsoftSignal()
		stubDispatchIssueFetches(t,
			func(root, slug string, limit int) ([]dispatchtick.Issue, error) {
				return []dispatchtick.Issue{dispatchViewTestIssue(41)}, nil
			},
			func(root string, limit int) ([]dispatchtick.Issue, error) {
				t.Fatal("full-backlog fetch must not run when the view resolves")
				return nil, nil
			})
		issues, injected, err := dispatchFetchScopedIssues(t.TempDir(), io.Discard, "current", 1000)
		if err != nil || !injected || len(issues) != 1 {
			t.Fatalf("scoped fetch injected=%v issues=%+v err=%v, want the resolved view slice", injected, issues, err)
		}
		if got := dispatchViewFailsoftSignal(); got.Count != before.Count {
			t.Fatalf("a dispatchable view recorded a fail-soft: before=%+v after=%+v", before, got)
		}
	})
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

func TestDispatchViewDigestBindsExactConfigBytes(t *testing.T) {
	root := t.TempDir()
	if err := os.MkdirAll(filepath.Join(root, ".github"), 0o755); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(root, ".github", "issue-views.json")
	first := []byte(`{"repo":"o/r","views":[{"slug":"current","query":"is:open label:a"}]}`)
	if err := os.WriteFile(path, first, 0o644); err != nil {
		t.Fatal(err)
	}
	_, query, d1, err := dispatchViewQueryWithDigest(root, "current")
	if err != nil {
		t.Fatal(err)
	}
	if query != "is:open label:a" || d1 == "" {
		t.Fatalf("query=%q digest=%q", query, d1)
	}
	_, _, d1b, err := dispatchViewQueryWithDigest(root, "current")
	if err != nil || d1b != d1 {
		t.Fatalf("stable digest=%q err=%v, want %q", d1b, err, d1)
	}
	second := []byte(`{"repo":"o/r","views":[{"slug":"current","query":"is:open label:b"}]}`)
	if err := os.WriteFile(path, second, 0o644); err != nil {
		t.Fatal(err)
	}
	_, _, d2, err := dispatchViewQueryWithDigest(root, "current")
	if err != nil {
		t.Fatal(err)
	}
	if d2 == d1 {
		t.Fatalf("digest did not change after config edit: %q", d1)
	}
}

func TestSeedDispatchTickPayloadCarriesViewProvenance(t *testing.T) {
	pr := dispatchTickPick{pick: dispatchLanePick{
		Lane: "docs", View: "current", ViewQuery: "is:open label:focus",
		ViewDigest: "sha256:abc", ViewFallback: true, ViewFallbackReason: "view_empty",
		ByLaneStepBudget: map[string]int{},
	}}
	got := seedDispatchTickPayload(t.TempDir(), dispatchTickOptions{}, map[string]any{}, map[string]any{}, dispatchtick.Account{}, pr)
	for key, want := range map[string]any{"view": "current", "view_query": "is:open label:focus", "view_digest": "sha256:abc", "view_fallback": true, "view_fallback_reason": "view_empty"} {
		if got[key] != want {
			t.Fatalf("%s=%v, want %v", key, got[key], want)
		}
	}
	if route := dispatchLastRouteDecision(4026, "docs", "current"); route != "view=current lane=docs target=#4026" {
		t.Fatalf("route=%q", route)
	}
}

func TestSeedDispatchTickPayloadOmitsDisabledViewProvenance(t *testing.T) {
	pr := dispatchTickPick{pick: dispatchLanePick{ByLaneStepBudget: map[string]int{}}}
	got := seedDispatchTickPayload(t.TempDir(), dispatchTickOptions{}, map[string]any{}, map[string]any{}, dispatchtick.Account{}, pr)
	for _, key := range []string{"view", "view_query", "view_digest", "view_fallback", "view_fallback_reason"} {
		if value, ok := got[key]; ok && value != "" && value != false {
			t.Fatalf("disabled view %s=%v", key, value)
		}
	}
}

func TestDispatchRouteIssuesCompleteUsesFullCoverageLimit(t *testing.T) {
	oldView, oldBacklog := dispatchTickView, dispatchFetchBacklogIssues
	dispatchTickView = ""
	t.Cleanup(func() { dispatchTickView, dispatchFetchBacklogIssues = oldView, oldBacklog })
	var gotLimit int
	dispatchFetchBacklogIssues = func(_ string, limit int) ([]dispatchtick.Issue, error) {
		gotLimit = limit
		return nil, nil
	}
	dispatchRoutedBacklogCache = dispatchcache.New[dispatchtick.RouterPayload](time.Now)

	payload, err := dispatchRouteIssuesComplete(t.TempDir(), io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	if gotLimit != 100000 {
		t.Fatalf("issue limit = %d, want complete-coverage limit 100000", gotLimit)
	}
	if payload.Coverage.IssueLimit != 100000 || !payload.Coverage.Complete {
		t.Fatalf("coverage = %+v", payload.Coverage)
	}
}
