package main

import (
	"bytes"
	"context"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/safecommit"
)

// prefixResolver is a test laneResolver: the first path segment is the lane, but a root-level
// file (no "/") has no lane, exercising the classifier's unplaceable-path bucket.
func prefixResolver(path string) string {
	p := strings.ReplaceAll(path, "\\", "/")
	if i := strings.IndexByte(p, '/'); i >= 0 {
		return p[:i]
	}
	return ""
}

func TestClassifyDirtyGroupsByLane(t *testing.T) {
	entries := []dirtyEntry{
		{Path: "docs/b.md", Status: "M"},
		{Path: "docs/a.md", Status: "M"},
		{Path: "gateway/http.go", Status: "M"},
		{Path: "MISC.txt", Status: "M"},                                        // root-level -> no-lane
		{Path: "experiments/x/.run.err", Status: "??", Untracked: true},        // junk
		{Path: "stray-scratchpad-in-Temp.json", Status: "??", Untracked: true}, // junk (flattened temp)
	}
	plan := classifyDirty(entries, prefixResolver)

	if plan.TotalDirty != 6 {
		t.Fatalf("TotalDirty = %d, want 6", plan.TotalDirty)
	}
	if len(plan.Groups) != 2 {
		t.Fatalf("len(Groups) = %d, want 2 (docs, gateway)", len(plan.Groups))
	}
	// Lanes are sorted: docs before gateway.
	if plan.Groups[0].Lane != "docs" || plan.Groups[1].Lane != "gateway" {
		t.Fatalf("lanes = %q,%q want docs,gateway", plan.Groups[0].Lane, plan.Groups[1].Lane)
	}
	// Paths within a group are sorted.
	if got := plan.Groups[0].Paths; len(got) != 2 || got[0] != "docs/a.md" || got[1] != "docs/b.md" {
		t.Fatalf("docs paths = %v, want [docs/a.md docs/b.md]", got)
	}
	if plan.Groups[0].Trailer != "(fak docs)" {
		t.Fatalf("docs trailer = %q, want (fak docs)", plan.Groups[0].Trailer)
	}
	if plan.Groups[0].Score != 100 {
		t.Fatalf("docs score = %d, want 100", plan.Groups[0].Score)
	}
	if len(plan.NoLane) != 1 || plan.NoLane[0].Path != "MISC.txt" {
		t.Fatalf("NoLane = %v, want [MISC.txt]", plan.NoLane)
	}
	if len(plan.Junk) != 2 {
		t.Fatalf("len(Junk) = %d, want 2", len(plan.Junk))
	}
	if n := stampableCount(plan); n != 3 {
		t.Fatalf("stampableCount = %d, want 3", n)
	}
}

func TestScoreSweepGroupSurfacesRiskSignals(t *testing.T) {
	score, reasons := scoreSweepGroup([]dirtyEntry{
		{Path: "docs/a.md", Status: "M"},
		{Path: "docs/new.md", Status: "??", Untracked: true},
		{Path: "docs/old.md", Status: "D"},
	})
	if score != 74 {
		t.Fatalf("score = %d, want 74", score)
	}
	for _, want := range []string{"mixed git statuses", "includes untracked source", "includes deletions"} {
		if !containsString(reasons, want) {
			t.Fatalf("reasons = %v, missing %q", reasons, want)
		}
	}
}

func TestRenderSweepPlanIncludesScore(t *testing.T) {
	plan := sweepPlan{TotalDirty: 1, Groups: []sweepGroup{{
		Lane:         "docs",
		Trailer:      "(fak docs)",
		Paths:        []string{"docs/a.md"},
		Score:        92,
		ScoreReasons: []string{"includes untracked source"},
	}}}
	var out bytes.Buffer
	renderSweepPlan(&out, plan)
	got := out.String()
	for _, want := range []string{"score  92", "score notes: includes untracked source"} {
		if !strings.Contains(got, want) {
			t.Fatalf("rendered sweep plan missing %q:\n%s", want, got)
		}
	}
}

func containsString(xs []string, want string) bool {
	for _, x := range xs {
		if x == want {
			return true
		}
	}
	return false
}

func TestIsSweepJunk(t *testing.T) {
	cases := []struct {
		name string
		e    dirtyEntry
		want bool
	}{
		{"flattened temp scratchpad", dirtyEntry{Path: "CUsersUSERAppDataLocalTempclaudeFOOscratchpadaudit_output.json", Untracked: true}, true},
		{"run err log", dirtyEntry{Path: "experiments/x/.run.err", Untracked: true}, true},
		{"run out log", dirtyEntry{Path: "experiments/x/y.run.out", Untracked: true}, true},
		{"root coverage file", dirtyEntry{Path: "coverage", Untracked: true}, true},
		{"root coverprofile", dirtyEntry{Path: "unit.coverprofile", Untracked: true}, true},
		{"private-use glyph root dir", dirtyEntry{Path: "\uf05c/", Untracked: true}, true},
		{"tracked file with run.err suffix is not junk", dirtyEntry{Path: "experiments/x/.run.err", Untracked: false}, false},
		{"ordinary untracked source", dirtyEntry{Path: "internal/foo/bar.go", Untracked: true}, false},
		{"scratchpad but inside a real tree (has slash) is not junk", dirtyEntry{Path: "tools/scratchpad/temp.json", Untracked: true}, false},
		{"ordinary root doc is not junk", dirtyEntry{Path: "README.md", Untracked: true}, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := isSweepJunk(tc.e); got != tc.want {
				t.Fatalf("isSweepJunk(%+v) = %v, want %v", tc.e, got, tc.want)
			}
		})
	}
}

func TestParsePorcelainZ(t *testing.T) {
	// NUL-terminated "XY PATH" records, with a trailing empty field after the final NUL.
	out := " M cmd/fak/sweep.go\x00?? newpkg/foo.go\x00 D internal/old/gone.go\x00"
	got := parsePorcelainZ(out)
	if len(got) != 3 {
		t.Fatalf("len = %d, want 3 (%+v)", len(got), got)
	}
	if got[0].Path != "cmd/fak/sweep.go" || got[0].Status != "M" || got[0].Untracked {
		t.Fatalf("entry0 = %+v", got[0])
	}
	if got[1].Path != "newpkg/foo.go" || got[1].Status != "??" || !got[1].Untracked {
		t.Fatalf("entry1 = %+v", got[1])
	}
	if got[2].Path != "internal/old/gone.go" || got[2].Status != "D" {
		t.Fatalf("entry2 = %+v", got[2])
	}
}

func TestEnsureTrailer(t *testing.T) {
	if got := ensureTrailer("docs: update the guide", "docs"); got != "docs: update the guide (fak docs)" {
		t.Fatalf("append: got %q", got)
	}
	// Already stamped -> untouched (the lint then catches a mismatch).
	already := "docs: update the guide (fak gateway)"
	if got := ensureTrailer(already, "docs"); got != already {
		t.Fatalf("preserve: got %q", got)
	}
	// A multi-line message keeps its body; only the subject line gets the stamp.
	multi := "feat: add thing\n\nbody line"
	if got := ensureTrailer(multi, "cmd"); got != "feat: add thing (fak cmd)\n\nbody line" {
		t.Fatalf("multiline: got %q", got)
	}
}

func TestIntersectPaths(t *testing.T) {
	have := []string{"docs/a.md", "docs/b.md", "docs/c.md"}
	// want uses backslashes and a leading ./ to prove normalization.
	want := []string{"docs\\b.md", "./docs/c.md", "docs/missing.md"}
	got := intersectPaths(have, want)
	if len(got) != 2 || got[0] != "docs/b.md" || got[1] != "docs/c.md" {
		t.Fatalf("intersect = %v, want [docs/b.md docs/c.md]", got)
	}
}

func TestRunSweepApplyHappyPath(t *testing.T) {
	root := t.TempDir() // no dos.toml -> lane recognition degrades gracefully (convention)
	plan := sweepPlan{Groups: []sweepGroup{{Lane: "docs", Trailer: "(fak docs)", Paths: []string{"docs/x.md"}}}}

	var got safecommit.Options
	called := false
	orig := commitFn
	commitFn = func(_ context.Context, opts safecommit.Options) (safecommit.Result, error) {
		called = true
		got = opts
		return safecommit.Result{SHA: "deadbeefcafe", Paths: opts.Paths}, nil
	}
	defer func() { commitFn = orig }()

	var stdout, stderr bytes.Buffer
	code := runSweepApply(&stdout, &stderr, root, plan, "docs", "docs: update the guide", nil, false)
	if code != 0 {
		t.Fatalf("exit = %d, want 0; stderr=%s", code, stderr.String())
	}
	if !called {
		t.Fatal("commitFn was not called")
	}
	if !strings.Contains(got.Message, "(fak docs)") {
		t.Fatalf("message missing stamp: %q", got.Message)
	}
	if len(got.Paths) != 1 || got.Paths[0] != "docs/x.md" {
		t.Fatalf("paths = %v, want [docs/x.md]", got.Paths)
	}
	if !got.SignOff {
		t.Fatal("SignOff should default to true")
	}
}

func TestRunSweepApplyRefusesOffLaneStamp(t *testing.T) {
	root := t.TempDir()
	plan := sweepPlan{Groups: []sweepGroup{{Lane: "docs", Trailer: "(fak docs)", Paths: []string{"docs/x.md"}}}}

	called := false
	orig := commitFn
	commitFn = func(_ context.Context, opts safecommit.Options) (safecommit.Result, error) {
		called = true
		return safecommit.Result{}, nil
	}
	defer func() { commitFn = orig }()

	var stdout, stderr bytes.Buffer
	// Subject already carries a (fak gateway) stamp on a docs path -> mismatch -> refuse.
	code := runSweepApply(&stdout, &stderr, root, plan, "docs", "docs: update x (fak gateway)", nil, false)
	if code != 3 {
		t.Fatalf("exit = %d, want 3 (pre-commit refusal)", code)
	}
	if called {
		t.Fatal("commitFn must NOT be called when the pre-lint refuses")
	}
}

func TestRunSweepApplyValidation(t *testing.T) {
	plan := sweepPlan{Groups: []sweepGroup{{Lane: "docs", Paths: []string{"docs/x.md"}}}}
	var out, errb bytes.Buffer

	// Missing -m -> usage error (2).
	if code := runSweepApply(&out, &errb, t.TempDir(), plan, "docs", "", nil, false); code != 2 {
		t.Fatalf("missing -m: exit = %d, want 2", code)
	}
	// Unknown lane -> pre-commit refusal (3).
	if code := runSweepApply(&out, &errb, t.TempDir(), plan, "gateway", "feat: x", nil, false); code != 3 {
		t.Fatalf("unknown lane: exit = %d, want 3", code)
	}
}
