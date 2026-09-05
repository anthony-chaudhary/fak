package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

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
	// A nil origin probe means origin-awareness is off: the plan must be exactly what it was before
	// the origin field existed — no Origin relation, no AlreadyShipped rollup.
	plan := classifyDirty(entries, prefixResolver, nil)

	if plan.TotalDirty != 6 {
		t.Fatalf("TotalDirty = %d, want 6", plan.TotalDirty)
	}
	for _, g := range plan.Groups {
		if len(g.AlreadyShipped) != 0 || g.AllAlready {
			t.Fatalf("nil probe should leave AlreadyShipped empty and AllAlready false, got %v/%v for lane %s", g.AlreadyShipped, g.AllAlready, g.Lane)
		}
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
	if !strings.Contains(plan.NextAction, "remove 2 junk path(s)") {
		t.Fatalf("NextAction = %q, want junk cleanup first", plan.NextAction)
	}
	if n := stampableCount(plan); n != 3 {
		t.Fatalf("stampableCount = %d, want 3", n)
	}
}

func TestSweepNextActionPriority(t *testing.T) {
	cases := []struct {
		name string
		plan sweepPlan
		want string
	}{
		{"clean", sweepPlan{}, "working tree is clean"},
		{"junk first", sweepPlan{TotalDirty: 2, Junk: []sweepEntry{{dirtyEntry: dirtyEntry{Path: "wave.err"}}}, Groups: []sweepGroup{{Lane: "docs", Paths: []string{"docs/a.md"}}}}, "`fak sweep --clean-junk`"},
		{"no lane before stampable", sweepPlan{TotalDirty: 2, NoLane: []sweepEntry{{dirtyEntry: dirtyEntry{Path: "TODO"}}}, Groups: []sweepGroup{{Lane: "docs", Paths: []string{"docs/a.md"}}}}, "inspect 1 no-lane path(s)"},
		{"unit first", sweepPlan{TotalDirty: 11, Groups: []sweepGroup{{Lane: "docs", Paths: []string{"docs/a.md"}, Units: []sweepSubUnit{{Index: 1, Paths: []string{"docs/a.md"}}}}}}, "--lane docs --unit 1"},
		{"lane group", sweepPlan{TotalDirty: 1, Groups: []sweepGroup{{Lane: "docs", Paths: []string{"docs/a.md"}}}}, "--apply --lane docs -m"},
		{"all already", sweepPlan{TotalDirty: 1, Groups: []sweepGroup{{Lane: "docs", Paths: []string{"docs/a.md"}, AllAlready: true}}}, "discard already-shipped"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := sweepNextAction(tc.plan); !strings.Contains(got, tc.want) {
				t.Fatalf("sweepNextAction = %q, want substring %q", got, tc.want)
			}
		})
	}
}

func TestRunSweepCleanJunkJSONRemovesClassifiedJunk(t *testing.T) {
	root := t.TempDir()
	sweepGit(t, root, "init")
	junk := filepath.Join(root, "wave.err")
	if err := os.WriteFile(junk, []byte("stderr\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	var out, errb bytes.Buffer
	code := runSweep(&out, &errb, []string{"--dir", root, "--clean-junk", "--json", "--no-origin"})
	if code != 0 {
		t.Fatalf("exit = %d, want 0; stderr=%s stdout=%s", code, errb.String(), out.String())
	}
	var got sweepCleanJunkResult
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatalf("decode clean-junk JSON: %v\n%s", err, out.String())
	}
	if !got.OK || len(got.Removed) != 1 || got.Removed[0] != "wave.err" {
		t.Fatalf("result = %+v, want wave.err removed", got)
	}
	if _, err := os.Stat(junk); !os.IsNotExist(err) {
		t.Fatalf("wave.err still exists or stat failed unexpectedly: %v", err)
	}
	if !strings.Contains(got.NextAction, "fak sweep --json") {
		t.Fatalf("next action = %q, want rerun sweep", got.NextAction)
	}
}

func TestCleanSweepJunkRefusesDirectoriesAndEscapes(t *testing.T) {
	root := t.TempDir()
	if err := os.Mkdir(filepath.Join(root, "capture.log"), 0o755); err != nil {
		t.Fatal(err)
	}
	outside := filepath.Join(root, "..", "outside.err")
	plan := sweepPlan{Junk: []sweepEntry{
		{dirtyEntry: dirtyEntry{Path: "capture.log"}},
		{dirtyEntry: dirtyEntry{Path: "../outside.err"}},
		{dirtyEntry: dirtyEntry{Path: filepath.ToSlash(outside)}},
	}}
	got := cleanSweepJunk(root, plan)
	if got.OK {
		t.Fatalf("result unexpectedly OK: %+v", got)
	}
	if len(got.Refused) != 3 {
		t.Fatalf("refused = %+v, want directory plus two unsafe paths", got.Refused)
	}
	if got.Refused[0].Reason != "directory_refused" {
		t.Fatalf("first refusal = %+v, want directory_refused", got.Refused[0])
	}
	if got.Refused[1].Reason != "unsafe_path" || got.Refused[2].Reason != "unsafe_path" {
		t.Fatalf("escape refusals = %+v, want unsafe_path", got.Refused)
	}
	if _, err := os.Stat(filepath.Join(root, "capture.log")); err != nil {
		t.Fatalf("directory should remain for manual inspection: %v", err)
	}
}

func TestCleanSweepJunkAutoArchive(t *testing.T) {
	root := t.TempDir()
	cmdInit := exec.Command("git", "init", "-q", "-b", "main")
	cmdInit.Dir = root
	if out, err := cmdInit.CombinedOutput(); err != nil {
		t.Fatalf("git init: %v\n%s", err, out)
	}
	junkPath := filepath.Join(root, "junk.log")
	junkBytes := []byte("temporary junk log data\n")
	if err := os.WriteFile(junkPath, junkBytes, 0o644); err != nil {
		t.Fatal(err)
	}
	plan := sweepPlan{Junk: []sweepEntry{
		{dirtyEntry: dirtyEntry{Path: "junk.log"}},
	}}
	got := cleanSweepJunk(root, plan, true)
	if !got.OK {
		t.Fatalf("cleanSweepJunk failed: %+v", got)
	}
	if got.QuarantineRef == "" || got.QuarantineSHA == "" {
		t.Fatalf("expected quarantine ref and sha, got: %+v", got)
	}
	if got.ArchivedCount != 1 || got.ArchivedBytes != int64(len(junkBytes)) {
		t.Fatalf("unexpected archived stats: count=%d, bytes=%d", got.ArchivedCount, got.ArchivedBytes)
	}
	if _, err := os.Stat(junkPath); !os.IsNotExist(err) {
		t.Fatalf("junk.log should have been removed from working tree: %v", err)
	}
}

func TestRenderSweepCleanJunkText(t *testing.T) {
	var out bytes.Buffer
	renderSweepCleanJunk(&out, sweepCleanJunkResult{
		OK:      false,
		Removed: []string{"wave.err"},
		Skipped: []string{"route.err"},
		Refused: []sweepCleanJunkRefusal{{
			Path:   "capture.log",
			Reason: "directory_refused",
			Detail: "manual inspection required",
		}},
		NextAction: "inspect refused junk paths, then rerun `fak sweep --json`",
	})
	got := out.String()
	for _, want := range []string{
		"removed 1 junk path(s):",
		"  wave.err",
		"skipped 1 already-gone junk path(s):",
		"  route.err",
		"refused 1 junk path(s):",
		"  capture.log  directory_refused - manual inspection required",
		"next: inspect refused junk paths, then rerun `fak sweep --json`",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("rendered text missing %q:\n%s", want, got)
		}
	}
}

func TestClassifyDirtyRollsUpOldestDirtyAge(t *testing.T) {
	entries := []dirtyEntry{
		{Path: "docs/newer.md", Status: "M", MTime: 2000, AgeSeconds: 60},
		{Path: "docs/older.md", Status: "M", MTime: 1000, AgeSeconds: 1060},
		{Path: "gateway/http.go", Status: "M", MTime: 1500, AgeSeconds: 560},
	}
	plan := classifyDirty(entries, prefixResolver, nil)
	if plan.OldestDirtyPath != "docs/older.md" || plan.OldestDirtyMTime != 1000 || plan.OldestDirtyAgeSeconds != 1060 {
		t.Fatalf("plan oldest = %q/%d/%d, want docs/older.md/1000/1060",
			plan.OldestDirtyPath, plan.OldestDirtyMTime, plan.OldestDirtyAgeSeconds)
	}
	byLane := map[string]sweepGroup{}
	for _, g := range plan.Groups {
		byLane[g.Lane] = g
	}
	docs := byLane["docs"]
	if docs.OldestDirtyPath != "docs/older.md" || docs.OldestDirtyMTime != 1000 || docs.OldestDirtyAgeSeconds != 1060 {
		t.Fatalf("docs oldest = %q/%d/%d, want docs/older.md/1000/1060",
			docs.OldestDirtyPath, docs.OldestDirtyMTime, docs.OldestDirtyAgeSeconds)
	}
	gateway := byLane["gateway"]
	if gateway.OldestDirtyPath != "gateway/http.go" || gateway.OldestDirtyMTime != 1500 || gateway.OldestDirtyAgeSeconds != 560 {
		t.Fatalf("gateway oldest = %q/%d/%d, want gateway/http.go/1500/560",
			gateway.OldestDirtyPath, gateway.OldestDirtyMTime, gateway.OldestDirtyAgeSeconds)
	}
}

// TestClassifyDirtyOriginRelation proves the injected origin probe annotates every stampable entry
// and rolls up per lane: a mixed lane surfaces its ALREADY subset without being AllAlready, while a
// lane whose every path is byte-identical to the trunk is flagged AllAlready (nothing to ship). The
// probe is a pure closure keyed on path — no git tree, exactly like the laneResolver fakes.
func TestClassifyDirtyOriginRelation(t *testing.T) {
	entries := []dirtyEntry{
		{Path: "resume/new.go", Status: "??", Untracked: true}, // genuinely new work
		{Path: "resume/old.go", Status: "M"},                   // already on origin (stale dup)
		{Path: "resume/edit.go", Status: "M"},                  // ahead: real local change
		{Path: "docs/shipped-a.md", Status: "M"},               // whole docs lane already shipped
		{Path: "docs/shipped-b.md", Status: "M"},               //
		{Path: "junk/.run.err", Status: "??", Untracked: true}, // junk: never probed
	}
	origin := func(path string) originRelation {
		switch path {
		case "resume/new.go":
			return originNew
		case "resume/old.go", "docs/shipped-a.md", "docs/shipped-b.md":
			return originAlready
		case "resume/edit.go":
			return originAhead
		default:
			t.Fatalf("origin probe called for unexpected path %q (junk must not be probed)", path)
			return originUnknown
		}
	}

	plan := classifyDirty(entries, prefixResolver, origin)

	byLane := map[string]sweepGroup{}
	for _, g := range plan.Groups {
		byLane[g.Lane] = g
	}

	// resume: mixed — one ALREADY of three, so AlreadyShipped names it but AllAlready is false.
	resume := byLane["resume"]
	if len(resume.AlreadyShipped) != 1 || resume.AlreadyShipped[0] != "resume/old.go" {
		t.Fatalf("resume AlreadyShipped = %v, want [resume/old.go]", resume.AlreadyShipped)
	}
	if resume.AllAlready {
		t.Fatalf("resume AllAlready = true, want false (only 1 of 3 paths already shipped)")
	}

	// docs: every path is ALREADY on origin, so the whole lane is a no-op.
	docs := byLane["docs"]
	if !docs.AllAlready {
		t.Fatalf("docs AllAlready = false, want true (both paths already on origin)")
	}
	if len(docs.AlreadyShipped) != 2 {
		t.Fatalf("docs AlreadyShipped = %v, want both paths", docs.AlreadyShipped)
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
		if !containsSweepString(reasons, want) {
			t.Fatalf("reasons = %v, missing %q", reasons, want)
		}
	}
}

// TestSplitLaneUnitsDirectoryCoherent proves a too-large lane splits into one sub-unit per
// immediate-parent directory, ordered by the sorted directory key, 1-based Index, each unit's Paths
// sorted and holding exactly that directory's files, and each unit scored on its own slice.
func TestSplitLaneUnitsDirectoryCoherent(t *testing.T) {
	var entries []dirtyEntry
	// Deliberately interleave the input order to prove the split sorts, not preserves input order.
	for _, p := range []string{
		"a/y/4", "b/z/3", "a/x/1", "a/x/3", "a/y/1", "b/z/1",
		"a/x/2", "a/y/2", "b/z/2", "a/x/5", "a/y/3", "a/x/4",
	} {
		entries = append(entries, dirtyEntry{Path: p, Status: "M"})
	}
	units := splitLaneUnits(entries, atomicUnitTarget)
	if len(units) != 3 {
		t.Fatalf("len(units) = %d, want 3 (a/x, a/y, b/z)", len(units))
	}
	wantDirs := []string{"a/x", "a/y", "b/z"}
	wantCounts := []int{5, 4, 3}
	for i, u := range units {
		if u.Index != i+1 {
			t.Fatalf("unit[%d].Index = %d, want %d", i, u.Index, i+1)
		}
		if u.Dir != wantDirs[i] {
			t.Fatalf("unit[%d].Dir = %q, want %q", i, u.Dir, wantDirs[i])
		}
		if len(u.Paths) != wantCounts[i] {
			t.Fatalf("unit[%d] (%s) has %d paths, want %d", i, u.Dir, len(u.Paths), wantCounts[i])
		}
		for _, p := range u.Paths {
			if sweepDirKey(p) != u.Dir {
				t.Fatalf("unit[%d] (%s) contains foreign path %q", i, u.Dir, p)
			}
		}
		if !isSortedStrings(u.Paths) {
			t.Fatalf("unit[%d] (%s) Paths not sorted: %v", i, u.Dir, u.Paths)
		}
	}
}

// TestSplitLaneUnitsConservation is the load-bearing invariant: the union of every sub-unit's Paths
// equals the input path set EXACTLY, each path appearing in exactly one unit. This is the proof the
// split never drops or duplicates an in-progress change — the "don't falsely destroy WIP" guarantee
// applied to the re-grouping.
func TestSplitLaneUnitsConservation(t *testing.T) {
	var entries []dirtyEntry
	for _, p := range []string{
		"internal/a/1.go", "internal/a/2.go", "internal/b/1.go", "internal/b/2.go",
		"internal/c/1.go", "cmd/x/1.go", "cmd/x/2.go", "cmd/y/1.go",
		"docs/z/a.md", "docs/z/b.md", "root-file.txt", "another-root.txt",
	} {
		entries = append(entries, dirtyEntry{Path: p, Status: "M"})
	}
	units := splitLaneUnits(entries, atomicUnitTarget)
	if len(units) == 0 {
		t.Fatal("expected a split for a 12-path lane")
	}
	seen := map[string]int{}
	for _, u := range units {
		for _, p := range u.Paths {
			seen[p]++
		}
	}
	if len(seen) != len(entries) {
		t.Fatalf("union has %d distinct paths, want %d (a path was dropped or duplicated)", len(seen), len(entries))
	}
	for _, e := range entries {
		if seen[e.Path] != 1 {
			t.Fatalf("path %q appears %d times across units, want exactly 1", e.Path, seen[e.Path])
		}
	}
}

// TestSplitLaneUnitsFastPath pins the ceiling boundary: a lane of exactly atomicUnitTarget stays
// whole (nil), and one path over the target splits.
func TestSplitLaneUnitsFastPath(t *testing.T) {
	mk := func(n int) []dirtyEntry {
		var es []dirtyEntry
		for i := 0; i < n; i++ {
			es = append(es, dirtyEntry{Path: fmt.Sprintf("d%d/f.go", i), Status: "M"})
		}
		return es
	}
	if u := splitLaneUnits(mk(atomicUnitTarget), atomicUnitTarget); u != nil {
		t.Fatalf("exactly atomicUnitTarget (%d) paths should NOT split, got %d units", atomicUnitTarget, len(u))
	}
	if u := splitLaneUnits(mk(atomicUnitTarget+1), atomicUnitTarget); len(u) == 0 {
		t.Fatalf("atomicUnitTarget+1 (%d) paths should split, got nil", atomicUnitTarget+1)
	}
}

// TestClassifyDirtyPopulatesUnitsForLargeLane proves the wiring: a lane over the target gets an
// ordered, conserving Units slice, while a small lane leaves Units nil (unchanged JSON shape).
func TestClassifyDirtyPopulatesUnitsForLargeLane(t *testing.T) {
	var entries []dirtyEntry
	for i := 0; i < 12; i++ {
		entries = append(entries, dirtyEntry{Path: fmt.Sprintf("big/sub%d/f.go", i%3), Status: "M"})
	}
	entries = append(entries, dirtyEntry{Path: "small/a.go", Status: "M"})
	plan := classifyDirty(entries, prefixResolver, nil)

	byLane := map[string]sweepGroup{}
	for _, g := range plan.Groups {
		byLane[g.Lane] = g
	}
	big := byLane["big"]
	if len(big.Units) == 0 {
		t.Fatalf("large lane 'big' (%d paths) should have Units", len(big.Paths))
	}
	total := 0
	for _, u := range big.Units {
		total += len(u.Paths)
	}
	if total != len(big.Paths) {
		t.Fatalf("big Units cover %d paths, want %d (conservation)", total, len(big.Paths))
	}
	if small := byLane["small"]; small.Units != nil {
		t.Fatalf("small lane should leave Units nil, got %v", small.Units)
	}
}

// isSortedStrings reports whether xs is in ascending order. Named a predicate — attest.go's
// sortedStrings (same package at test-build time) returns a sorted COPY, and the clash bricked
// the cmd/fak test build.
func isSortedStrings(xs []string) bool {
	for i := 1; i < len(xs); i++ {
		if xs[i-1] > xs[i] {
			return false
		}
	}
	return true
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

func TestRenderSweepPlanIncludesOldestAge(t *testing.T) {
	plan := sweepPlan{
		TotalDirty:            1,
		OldestDirtyPath:       "docs/a.md",
		OldestDirtyAgeSeconds: 2 * 3600,
		NextAction:            "run `fak sweep --apply --lane docs -m \"docs(docs): update note\"`",
		Groups: []sweepGroup{{
			Lane:                  "docs",
			Trailer:               "(fak docs)",
			Paths:                 []string{"docs/a.md"},
			Score:                 100,
			OldestDirtyPath:       "docs/a.md",
			OldestDirtyAgeSeconds: 2 * 3600,
		}},
	}
	var out bytes.Buffer
	renderSweepPlan(&out, plan)
	got := out.String()
	for _, want := range []string{"oldest dirty: 2h at docs/a.md", "next: run `fak sweep --apply --lane docs", "oldest: 2h at docs/a.md"} {
		if !strings.Contains(got, want) {
			t.Fatalf("rendered sweep plan missing %q:\n%s", want, got)
		}
	}
}

// TestRenderSweepPlanShowsSubUnits proves a too-large lane renders the directory-coherent sub-unit
// block with a per-unit `--apply --unit N` hint, while an unsplit lane still renders exactly the
// single whole-lane hint (the sub-unit block must not leak into the small-lane path).
func TestRenderSweepPlanShowsSubUnits(t *testing.T) {
	plan := sweepPlan{
		TotalDirty: 12,
		Groups: []sweepGroup{
			{ // large lane with two sub-units
				Lane:    "internal",
				Trailer: "(fak internal)",
				Paths:   []string{"internal/hooks/a.go", "internal/safecommit/b.go"},
				Score:   75,
				Units: []sweepSubUnit{
					{Index: 1, Dir: "internal/hooks", Paths: []string{"internal/hooks/a.go"}, Score: 100},
					{Index: 2, Dir: "internal/safecommit", Paths: []string{"internal/safecommit/b.go"}, Score: 100},
				},
			},
			{ // small lane: no units
				Lane:    "docs",
				Trailer: "(fak docs)",
				Paths:   []string{"docs/a.md"},
				Score:   100,
			},
		},
	}
	var out bytes.Buffer
	renderSweepPlan(&out, plan)
	got := out.String()

	for _, want := range []string{
		"LARGE lane (2 paths) — commit in 2 directory-coherent sub-unit(s):",
		"unit 1  dir internal/hooks",
		"--apply --lane internal --unit 1",
		"unit 2  dir internal/safecommit",
		"--apply --lane internal --unit 2",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("rendered plan missing %q:\n%s", want, got)
		}
	}
	// The small lane keeps its single whole-lane hint and shows NO sub-unit block.
	if !strings.Contains(got, "--apply --lane docs -m") {
		t.Fatalf("small lane missing its whole-lane hint:\n%s", got)
	}
	if strings.Contains(got, "--apply --lane docs --unit") {
		t.Fatalf("small lane must not render a sub-unit hint:\n%s", got)
	}
}

// TestRenderSweepPlanFlagsAlreadyShipped captures the render bytes and proves the two origin
// call-outs reach the surface: a per-path [ALREADY on origin] tag on the stale path, and the
// whole-lane no-op line when every path in a lane already matches the trunk. This is the exact
// output that turns a multi-probe investigation into one glance.
func TestRenderSweepPlanFlagsAlreadyShipped(t *testing.T) {
	plan := sweepPlan{
		TotalDirty: 3,
		Groups: []sweepGroup{
			{ // mixed lane: one path already shipped, one still to ship
				Lane:           "resume",
				Trailer:        "(fak resume)",
				Paths:          []string{"resume/edit.go", "resume/old.go"},
				Score:          100,
				AlreadyShipped: []string{"resume/old.go"},
			},
			{ // whole lane already on origin — nothing to commit
				Lane:           "docs",
				Trailer:        "(fak docs)",
				Paths:          []string{"docs/a.md"},
				Score:          100,
				AlreadyShipped: []string{"docs/a.md"},
				AllAlready:     true,
			},
		},
	}
	var out bytes.Buffer
	renderSweepPlan(&out, plan)
	got := out.String()

	if !strings.Contains(got, "resume/old.go  [ALREADY on origin]") {
		t.Fatalf("expected the stale path tagged [ALREADY on origin]:\n%s", got)
	}
	if !strings.Contains(got, "ALREADY on origin — all 1 path(s) match the trunk") {
		t.Fatalf("expected the all-already lane no-op call-out:\n%s", got)
	}
	// An all-already lane must NOT print the "commit this lane" hint — there is nothing to ship.
	if strings.Contains(got, "--apply --lane docs") {
		t.Fatalf("all-already lane should not suggest a commit:\n%s", got)
	}
	// The mixed lane still gets its commit hint (it has real work).
	if !strings.Contains(got, "--apply --lane resume") {
		t.Fatalf("mixed lane should still suggest a commit:\n%s", got)
	}
}

func containsSweepString(xs []string, want string) bool {
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
		{"root err spill", dirtyEntry{Path: "wave.err", Untracked: true}, true},
		{"root out spill", dirtyEntry{Path: "route.out", Untracked: true}, true},
		{"root json output spill", dirtyEntry{Path: "boundary_out.json", Untracked: true}, true},
		{"root coverage file", dirtyEntry{Path: "coverage", Untracked: true}, true},
		{"root coverprofile", dirtyEntry{Path: "unit.coverprofile", Untracked: true}, true},
		{"private-use glyph root dir", dirtyEntry{Path: "\uf05c/", Untracked: true}, true},
		{"tracked file with run.err suffix is not junk", dirtyEntry{Path: "experiments/x/.run.err", Untracked: false}, false},
		{"experiment err evidence stays lane-owned", dirtyEntry{Path: "experiments/x/driver.err", Untracked: true}, false},
		{"experiment json evidence stays lane-owned", dirtyEntry{Path: "experiments/x/boundary_out.json", Untracked: true}, false},
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

func TestFilterContentDirtyDropsStatCacheGhosts(t *testing.T) {
	entries := parsePorcelainZ(" M real.go\x00 M ghost.go\x00M  staged.go\x00MM both.go\x00?? new.go\x00")
	got := filterContentDirty(entries, map[string]struct{}{
		"real.go": {},
		"both.go": {},
	})

	var paths []string
	for _, entry := range got {
		paths = append(paths, entry.Path)
	}
	want := []string{"real.go", "staged.go", "both.go", "new.go"}
	if fmt.Sprint(paths) != fmt.Sprint(want) {
		t.Fatalf("paths = %v, want %v", paths, want)
	}
}

func TestFilterContentDirtyKeepsStagedSideWhenWorktreeIsGhost(t *testing.T) {
	entries := parsePorcelainZ("MM staged-and-ghost.go\x00")
	got := filterContentDirty(entries, nil)
	if len(got) != 1 || got[0].Path != "staged-and-ghost.go" {
		t.Fatalf("got = %#v, want staged path retained", got)
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

func TestAnnotateDirtyAgesStatsExistingPaths(t *testing.T) {
	root := t.TempDir()
	if err := os.MkdirAll(filepath.Join(root, "docs"), 0o755); err != nil {
		t.Fatal(err)
	}
	p := filepath.Join(root, "docs", "a.md")
	if err := os.WriteFile(p, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	mtime := time.Unix(1_700, 0)
	if err := os.Chtimes(p, mtime, mtime); err != nil {
		t.Fatal(err)
	}
	got := annotateDirtyAges(root, []dirtyEntry{
		{Path: "docs/a.md", Status: "M"},
		{Path: "docs/missing.md", Status: "D"},
	}, time.Unix(2_000, 0))
	if got[0].MTime != 1_700 || got[0].AgeSeconds != 300 {
		t.Fatalf("existing path age = %d/%d, want 1700/300", got[0].MTime, got[0].AgeSeconds)
	}
	if got[1].MTime != 0 || got[1].AgeSeconds != 0 {
		t.Fatalf("missing path should omit age, got %d/%d", got[1].MTime, got[1].AgeSeconds)
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
	code := runSweepApply(&stdout, &stderr, root, plan, "docs", "docs: update the guide", nil, 0, false)
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

// TestRunSweepApplySelectsSubUnit proves --unit N narrows the commit to exactly that sub-unit's
// paths through the same safe-commit executor, stamping the lane trailer as usual.
func TestRunSweepApplySelectsSubUnit(t *testing.T) {
	// A dos.toml declares `svc` as a lane over svc/**, so the pre-commit lint resolves both
	// sub-unit directories (svc/a, svc/b) to the SAME leaf `svc` — the real-world shape where a
	// lane spans several directories and a sub-unit is one of them. Without it the no-config lint
	// would infer the immediate parent dir as the lane and refuse the whole-lane stamp.
	root := t.TempDir()
	writeDosTomlLane(t, root, "svc", "svc/**")
	plan := sweepPlan{Groups: []sweepGroup{{
		Lane:    "svc",
		Trailer: "(fak svc)",
		Paths:   []string{"svc/a/x.go", "svc/b/y.go"},
		Units: []sweepSubUnit{
			{Index: 1, Dir: "svc/a", Paths: []string{"svc/a/x.go"}},
			{Index: 2, Dir: "svc/b", Paths: []string{"svc/b/y.go"}},
		},
	}}}

	var got safecommit.Options
	orig := commitFn
	commitFn = func(_ context.Context, opts safecommit.Options) (safecommit.Result, error) {
		got = opts
		return safecommit.Result{SHA: "deadbeefcafe", Paths: opts.Paths}, nil
	}
	defer func() { commitFn = orig }()

	var stdout, stderr bytes.Buffer
	code := runSweepApply(&stdout, &stderr, root, plan, "svc", "refactor(svc): split b", nil, 2, false)
	if code != 0 {
		t.Fatalf("exit = %d, want 0; stderr=%s", code, stderr.String())
	}
	if len(got.Paths) != 1 || got.Paths[0] != "svc/b/y.go" {
		t.Fatalf("committed paths = %v, want [svc/b/y.go] (only sub-unit 2)", got.Paths)
	}
	if !strings.Contains(got.Message, "(fak svc)") {
		t.Fatalf("message missing stamp: %q", got.Message)
	}
}

// writeDosTomlLane writes a minimal dos.toml into root declaring one lane over one glob tree, so the
// pre-commit lint (hooks.LintCommitMessage) resolves paths under that tree to the lane leaf.
func writeDosTomlLane(t *testing.T, root, lane, glob string) {
	t.Helper()
	body := fmt.Sprintf("[lanes.trees]\n%s = [%q]\n", lane, glob)
	if err := os.WriteFile(filepath.Join(root, "dos.toml"), []byte(body), 0o644); err != nil {
		t.Fatalf("write dos.toml: %v", err)
	}
}

// TestRunSweepApplyUnitOutOfRange proves a malformed --unit is a usage refusal (2) that never
// reaches the committer: an out-of-range index on a split lane, and any --unit on an unsplit lane.
func TestRunSweepApplyUnitOutOfRange(t *testing.T) {
	split := sweepPlan{Groups: []sweepGroup{{
		Lane:    "internal",
		Trailer: "(fak internal)",
		Paths:   []string{"internal/a/x.go", "internal/b/y.go"},
		Units: []sweepSubUnit{
			{Index: 1, Dir: "internal/a", Paths: []string{"internal/a/x.go"}},
			{Index: 2, Dir: "internal/b", Paths: []string{"internal/b/y.go"}},
		},
	}}}
	unsplit := sweepPlan{Groups: []sweepGroup{{
		Lane: "docs", Trailer: "(fak docs)", Paths: []string{"docs/a.md"},
	}}}

	called := false
	orig := commitFn
	commitFn = func(_ context.Context, _ safecommit.Options) (safecommit.Result, error) {
		called = true
		return safecommit.Result{}, nil
	}
	defer func() { commitFn = orig }()

	var out, errb bytes.Buffer
	if code := runSweepApply(&out, &errb, t.TempDir(), split, "internal", "feat: x", nil, 99, false); code != 2 {
		t.Fatalf("out-of-range unit: exit = %d, want 2", code)
	}
	if code := runSweepApply(&out, &errb, t.TempDir(), unsplit, "docs", "docs: x", nil, 1, false); code != 2 {
		t.Fatalf("--unit on unsplit lane: exit = %d, want 2", code)
	}
	if called {
		t.Fatal("commitFn must NOT be called for a malformed --unit")
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
	code := runSweepApply(&stdout, &stderr, root, plan, "docs", "docs: update x (fak gateway)", nil, 0, false)
	if code != safecommit.ExitRefused {
		t.Fatalf("exit = %d, want %d (a refusal on the merits, not retryable contention)", code, safecommit.ExitRefused)
	}
	if called {
		t.Fatal("commitFn must NOT be called when the pre-lint refuses")
	}
}

func TestRunSweepApplyValidation(t *testing.T) {
	plan := sweepPlan{Groups: []sweepGroup{{Lane: "docs", Paths: []string{"docs/x.md"}}}}
	var out, errb bytes.Buffer

	// Missing -m -> usage error (2).
	if code := runSweepApply(&out, &errb, t.TempDir(), plan, "docs", "", nil, 0, false); code != 2 {
		t.Fatalf("missing -m: exit = %d, want 2", code)
	}
	// Unknown lane -> a refusal on the merits (4); retrying never invents the lane.
	if code := runSweepApply(&out, &errb, t.TempDir(), plan, "gateway", "feat: x", nil, 0, false); code != safecommit.ExitRefused {
		t.Fatalf("unknown lane: exit = %d, want %d", code, safecommit.ExitRefused)
	}
}

func TestRunSweepApplyRefusesCommittedRed(t *testing.T) {
	root := t.TempDir()
	writeDosTomlLane(t, root, "svc", "svc/**")
	plan := sweepPlan{Groups: []sweepGroup{{
		Lane:    "svc",
		Trailer: "(fak svc)",
		Paths:   []string{"svc/a/broken.go"},
	}}}

	oldBuild := commitBuildCheckGate
	t.Cleanup(func() { commitBuildCheckGate = oldBuild })
	commitBuildCheckGate = func(_ io.Writer, _ string, _ []string) (safecommit.BuildCheckOutcome, string) {
		return safecommit.BuildCheckFailed, "prospective build failed: syntax error in svc/a/broken.go"
	}

	commitCalled := false
	oldCommit := commitFn
	t.Cleanup(func() { commitFn = oldCommit })
	commitFn = func(_ context.Context, _ safecommit.Options) (safecommit.Result, error) {
		commitCalled = true
		return safecommit.Result{}, nil
	}

	var stdout, stderr bytes.Buffer
	code := runSweepApply(&stdout, &stderr, root, plan, "svc", "refactor(svc): split code (fak svc)", nil, 0, false)
	if code != safecommit.ExitRefused {
		t.Fatalf("runSweepApply exit = %d, want %d (ExitRefused); stderr=%s", code, safecommit.ExitRefused, stderr.String())
	}
	if commitCalled {
		t.Fatal("commitFn must NOT be called when prospective build fails")
	}
	if !strings.Contains(stderr.String(), "COMMITTED_RED") {
		t.Fatalf("stderr missing COMMITTED_RED: %s", stderr.String())
	}
	if !strings.Contains(stderr.String(), "syntax error in svc/a/broken.go") {
		t.Fatalf("stderr missing error detail: %s", stderr.String())
	}
}

func TestRunSweepApplyBypassesBuildCheckWhenOff(t *testing.T) {
	root := t.TempDir()
	writeDosTomlLane(t, root, "svc", "svc/**")
	plan := sweepPlan{Groups: []sweepGroup{{
		Lane:    "svc",
		Trailer: "(fak svc)",
		Paths:   []string{"svc/a/broken.go"},
	}}}

	t.Setenv("FAK_COMMIT_BUILD_CHECK", "off")

	oldBuild := commitBuildCheckGate
	t.Cleanup(func() { commitBuildCheckGate = oldBuild })
	buildCheckCalled := false
	commitBuildCheckGate = func(_ io.Writer, _ string, _ []string) (safecommit.BuildCheckOutcome, string) {
		buildCheckCalled = true
		return safecommit.BuildCheckFailed, "broken"
	}

	commitCalled := false
	oldCommit := commitFn
	t.Cleanup(func() { commitFn = oldCommit })
	commitFn = func(_ context.Context, _ safecommit.Options) (safecommit.Result, error) {
		commitCalled = true
		return safecommit.Result{SHA: "1234567890ab", Paths: []string{"svc/a/broken.go"}}, nil
	}

	var stdout, stderr bytes.Buffer
	code := runSweepApply(&stdout, &stderr, root, plan, "svc", "refactor(svc): split code (fak svc)", nil, 0, false)
	if code != 0 {
		t.Fatalf("runSweepApply with build check off exit = %d, want 0; stderr=%s", code, stderr.String())
	}
	if buildCheckCalled {
		t.Fatal("commitBuildCheckGate should NOT be called when FAK_COMMIT_BUILD_CHECK=off")
	}
	if !commitCalled {
		t.Fatal("commitFn should be called when build check is off")
	}
}

func sweepGit(t *testing.T, cwd string, args ...string) {
	t.Helper()
	cmd := exec.Command("git", args...)
	cmd.Dir = cwd
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("git %v in %s: %v\n%s", args, cwd, err, out)
	}
}
