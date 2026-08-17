package main

import (
	"errors"
	"strings"
	"testing"
)

func TestDispatchProgressLineSurfacesDeliveryAndStallStates(t *testing.T) {
	for _, tc := range []struct {
		name string
		p    *dispatchProgressSummary
		want []string
	}{
		{"progress", &dispatchProgressSummary{Available: true, Verdict: "PROGRESS", Commits: 2, IssuesClosed: 1, WIPFiles: 4, OpenIssues: 9, GitHubAvailable: true, NextAction: "keep shipping"}, []string{"progress PROGRESS", "commits=2", "closed=1 open=9"}},
		{"wip", &dispatchProgressSummary{Available: true, Verdict: "WIP_ONLY", WIPFiles: 7, GitHubAvailable: false, NextAction: "finish leaf"}, []string{"progress WIP_ONLY", "wip=7", "GitHub=unavailable"}},
		{"stalled", &dispatchProgressSummary{Available: true, Verdict: "STALLED", WIPFiles: 7, GitHubAvailable: true, NextAction: "replan"}, []string{"progress STALLED", "next=replan"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := dispatchProgressLine(tc.p)
			for _, want := range tc.want {
				if !strings.Contains(got, want) {
					t.Fatalf("missing %q in %q", want, got)
				}
			}
		})
	}
}

func TestDispatchProgressFoldFailsOpen(t *testing.T) {
	old := progressCommand
	t.Cleanup(func() { progressCommand = old })
	progressCommand = func(_, _ string, _ ...string) ([]byte, error) { return nil, errors.New("offline") }
	got := dispatchProgressFold(t.TempDir())
	if got.Available || got.Error == "" {
		t.Fatalf("expected unavailable summary: %+v", got)
	}
	snap := dispatchStatusSnapshot{Schema: "fleet-dispatch-status/1", Progress: got}
	if line := renderDispatchStatus(snap); !strings.Contains(line, "progress unavailable") {
		t.Fatalf("missing fail-open status: %s", line)
	}
}

func TestDispatchStatusRenderIncludesProgressInHumanAndMarkdown(t *testing.T) {
	snap := dispatchStatusSnapshot{Progress: &dispatchProgressSummary{Available: true, Verdict: "STALLED", GitHubAvailable: true, NextAction: "replan"}}
	if got := renderDispatchStatus(snap); !strings.Contains(got, "progress STALLED") {
		t.Fatalf("human: %s", got)
	}
	if got := renderDispatchStatusMarkdown(snap); !strings.Contains(got, "progress STALLED") {
		t.Fatalf("markdown: %s", got)
	}
}
