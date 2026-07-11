package main

import (
	"strings"
	"testing"
)

func TestFoldReleaseContentsGroupsAndRollsUp(t *testing.T) {
	commits := []prPlanCommit{
		{SHA: "a1", Subject: "feat(gateway): x (fak gateway)", Leaf: "gateway", Type: "feat", Resolves: []string{"#10"}},
		{SHA: "b2", Subject: "fix(gateway): y (fak gateway)", Leaf: "gateway", Type: "fix", Resolves: []string{"#11"}},
		{SHA: "c3", Subject: "feat(release): z (fak release)", Leaf: "release", Type: "feat", Resolves: []string{"#10"}},
		{SHA: "d4", Subject: "chore: untended", Type: "chore"},
	}
	got := foldReleaseContents(commits)

	if got["schema"] != releaseContentsSchema {
		t.Fatalf("schema = %v", got["schema"])
	}
	if got["commit_count"] != 4 {
		t.Fatalf("commit_count = %v, want 4", got["commit_count"])
	}
	if got["lane_count"] != 2 {
		t.Fatalf("lane_count = %v, want 2", got["lane_count"])
	}
	if got["unstamped_count"] != 1 {
		t.Fatalf("unstamped_count = %v, want 1", got["unstamped_count"])
	}
	if got["resolves_count"] != 2 {
		t.Fatalf("resolves_count = %v, want 2 (deduped across lanes)", got["resolves_count"])
	}
	resolves := releaseStatusStringSlice(got["resolves"])
	if strings.Join(resolves, ",") != "#10,#11" {
		t.Fatalf("resolves = %v, want deduped+sorted [#10 #11]", resolves)
	}
	types := got["types"].(map[string]int)
	if types["feat"] != 2 || types["fix"] != 1 || types["chore"] != 1 {
		t.Fatalf("types = %#v, want feat=2 fix=1 chore=1", types)
	}
	lanes := releaseStatusContentsLanes(got)
	if len(lanes) != 2 {
		t.Fatalf("lanes = %#v, want 2", lanes)
	}
	// biggest-first: gateway (2 commits) then release (1).
	if releaseStatusString(lanes[0]["leaf"]) != "gateway" || releaseStatusInt(lanes[0]["commits"]) != 2 {
		t.Fatalf("lane[0] = %#v, want gateway with 2 commits", lanes[0])
	}
	if releaseStatusString(lanes[1]["leaf"]) != "release" || releaseStatusInt(lanes[1]["commits"]) != 1 {
		t.Fatalf("lane[1] = %#v, want release with 1 commit", lanes[1])
	}
}

func TestFoldReleaseContentsEmpty(t *testing.T) {
	got := foldReleaseContents(nil)
	if got["commit_count"] != 0 || got["lane_count"] != 0 {
		t.Fatalf("empty fold = %#v, want zero counts", got)
	}
	// resolves/lanes must be non-nil so the JSON envelope stays typed.
	if _, ok := got["resolves"].([]string); !ok {
		t.Fatalf("resolves = %#v, want []string{}", got["resolves"])
	}
	if got["range"] != nil {
		t.Fatalf("bare fold should not set range: %#v", got["range"])
	}
}

func TestReleaseStatusContentsNoTag(t *testing.T) {
	got := releaseStatusContents(t.TempDir(), "")
	if got["commit_count"] != 0 {
		t.Fatalf("no-tag contents commit_count = %v, want 0", got["commit_count"])
	}
	if got["range"] != nil {
		t.Fatalf("no-tag contents range = %v, want nil", got["range"])
	}
}

func TestReleaseStatusRenderContents(t *testing.T) {
	contents := map[string]any{
		"commit_count":    4,
		"lane_count":      2,
		"unstamped_count": 1,
		"resolves":        []string{"#10", "#11"},
		"lanes": []map[string]any{
			{"leaf": "gateway", "commits": 2},
			{"leaf": "release", "commits": 1},
		},
	}
	got := releaseStatusRenderContents(contents)
	want := "  contents: 4 commit(s) across 2 lane(s); top: gateway 2, release 1; closes #10, #11; 1 unstamped"
	if got != want {
		t.Fatalf("render =\n%q\nwant\n%q", got, want)
	}

	// Empty range renders nothing (the commits-since-tag line already says 0).
	if line := releaseStatusRenderContents(map[string]any{"commit_count": 0}); line != "" {
		t.Fatalf("empty render = %q, want \"\"", line)
	}
	// A nil/absent contents map must not panic and must render nothing.
	if line := releaseStatusRenderContents(releaseStatusMap(nil)); line != "" {
		t.Fatalf("nil render = %q, want \"\"", line)
	}
}

func TestReleaseStatusRenderContentsTruncatesResolves(t *testing.T) {
	contents := map[string]any{
		"commit_count": 6,
		"lane_count":   1,
		"resolves":     []string{"#1", "#2", "#3", "#4", "#5", "#6"},
		"lanes":        []map[string]any{{"leaf": "core", "commits": 6}},
	}
	got := releaseStatusRenderContents(contents)
	if !strings.Contains(got, "closes #1, #2, #3, #4 (+2 more)") {
		t.Fatalf("render missing truncated resolves:\n%s", got)
	}
}
