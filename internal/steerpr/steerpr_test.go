package steerpr

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"
)

// logRecord builds one record in the `git log --no-merges --name-only
// --format=%x1e%H%x1f%s%x1f%b%x1f` wire format the real fold consumes.
func logRecord(sha, subject, body string, files ...string) string {
	return "\x1e" + sha + "\x1f" + subject + "\x1f" + body + "\x1f" + strings.Join(files, "\n")
}

func TestBandForMapsEveryVerdict(t *testing.T) {
	cases := []struct {
		verdict Verdict
		want    Band
	}{
		{VerdictWitnessed, BandCleared},
		{VerdictUnwitnessed, BandResidual},
		{VerdictAbstain, BandUnverifiable},
		{VerdictNoCommit, BandUnverifiable},
		// An ungraded commit must never read as CLEARED: "not yet graded" is
		// not "confirmed".
		{VerdictUnknown, BandUnverifiable},
		{Verdict("something-unrecognized"), BandUnverifiable},
	}
	for _, tc := range cases {
		if got := BandFor(tc.verdict); got != tc.want {
			t.Errorf("BandFor(%q) = %q, want %q", tc.verdict, got, tc.want)
		}
	}
}

// TestFoldBandTakesWorstMember is the load-bearing property: an operator who
// reads CLEARED must be able to conclude EVERY member was witnessed.
func TestFoldBandTakesWorstMember(t *testing.T) {
	c := func(bands ...Band) []Commit {
		out := make([]Commit, 0, len(bands))
		for i, b := range bands {
			out = append(out, Commit{SHA: fmt.Sprintf("sha%d", i), Band: b})
		}
		return out
	}
	cases := []struct {
		name string
		in   []Commit
		want Band
	}{
		{"empty is not vacuously cleared", nil, BandUnverifiable},
		{"all cleared", c(BandCleared, BandCleared), BandCleared},
		{"one residual reds the unit", c(BandCleared, BandResidual), BandResidual},
		{"cleared+unverifiable", c(BandCleared, BandUnverifiable), BandUnverifiable},
		{"unverifiable+residual", c(BandUnverifiable, BandResidual), BandResidual},
		{"residual dominates all", c(BandCleared, BandUnverifiable, BandResidual), BandResidual},
		{"single residual", c(BandResidual), BandResidual},
		{"majority cleared still reds", c(BandCleared, BandCleared, BandCleared, BandResidual), BandResidual},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := FoldBand(tc.in); got != tc.want {
				t.Errorf("FoldBand() = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestFoldBandDerivesFromVerdictWhenBandUnset(t *testing.T) {
	got := FoldBand([]Commit{
		{SHA: "a", Verdict: VerdictWitnessed},
		{SHA: "b", Verdict: VerdictUnwitnessed},
	})
	if got != BandResidual {
		t.Fatalf("FoldBand() = %q, want %q (verdict must drive the band when Band is unset)", got, BandResidual)
	}
}

func TestParseLogExtractsStampTypeAndIssueRefs(t *testing.T) {
	raw := logRecord("abc123", "fix(gateway): treat same-tick ready as positive (#42) (fak gateway)",
		"Body mentions #99 and #42.", "internal/gateway/a.go", "internal/gateway/b.go")
	commits := ParseLog(raw)
	if len(commits) != 1 {
		t.Fatalf("ParseLog() returned %d commits, want 1", len(commits))
	}
	c := commits[0]
	if c.SHA != "abc123" {
		t.Errorf("SHA = %q, want abc123", c.SHA)
	}
	if c.Leaf != "gateway" {
		t.Errorf("Leaf = %q, want gateway", c.Leaf)
	}
	if c.Type != "fix" {
		t.Errorf("Type = %q, want fix", c.Type)
	}
	// #42 is subject-bound => closure-grade, and must NOT also appear as a
	// body mention.
	if len(c.Resolves) != 1 || c.Resolves[0] != "#42" {
		t.Errorf("Resolves = %v, want [#42]", c.Resolves)
	}
	if len(c.Mentions) != 1 || c.Mentions[0] != "#99" {
		t.Errorf("Mentions = %v, want [#99] (a subject-bound ref must not double as a mention)", c.Mentions)
	}
	if len(c.Files) != 2 {
		t.Errorf("Files = %v, want 2 entries", c.Files)
	}
}

// TestFoldUnitsPartitionIsTotalAndDisjoint proves no landed commit is ever
// invisible to an operator: every commit lands in exactly one of
// (a unit, the unstamped set) — none in zero, none in two.
func TestFoldUnitsPartitionIsTotalAndDisjoint(t *testing.T) {
	raw := strings.Join([]string{
		logRecord("a1", "feat(steerpr): add the band (fak steerpr)", "", "internal/steerpr/band.go"),
		logRecord("a2", "fix(steerpr): tighten the fold (fak steerpr)", "", "internal/steerpr/steerpr.go"),
		logRecord("b1", "docs(cache): explain the ladder (fak cache)", "", "docs/cache.md"),
		logRecord("c1", "chore: no stamp at all", "", "misc.txt"),
		logRecord("c2", "fix(x): stamp is (fak mid) not at the end", "", "x.go"),
	}, "")

	commits := ParseLog(raw)
	if len(commits) != 5 { //boundarylint:ignore CHANGE_DETECTOR_TEST — closed fixture/contract cardinality
		t.Fatalf("ParseLog() = %d commits, want 5", len(commits))
	}

	units, unstamped := FoldUnits(commits)

	// Totality: members + orphans == everything seen. This is the arithmetic
	// invariant the overlay's credibility rests on.
	members := 0
	seen := map[string]int{}
	for _, u := range units {
		members += len(u.Commits)
		for _, c := range u.Commits {
			seen[c.SHA]++
		}
	}
	for _, c := range unstamped {
		seen[c.SHA]++
	}
	if members+len(unstamped) != len(commits) {
		t.Errorf("partition not total: members(%d) + unstamped(%d) != commits(%d)",
			members, len(unstamped), len(commits))
	}
	// Disjointness: nothing counted twice, nothing dropped.
	for _, c := range commits {
		if seen[c.SHA] != 1 {
			t.Errorf("commit %s appears %d times across the partition, want exactly 1", c.SHA, seen[c.SHA])
		}
	}

	// A mid-subject stamp is not a stamp: the regex anchors to end-of-subject.
	// Pinned deliberately so a grammar change is a visible decision, not a drift.
	if len(unstamped) != 2 {
		t.Errorf("unstamped = %d, want 2 (an unstamped commit and a mid-subject stamp)", len(unstamped))
	}
}

func TestFoldUnitsGroupsByLeafBiggestFirstOldestWithin(t *testing.T) {
	raw := strings.Join([]string{
		// git log yields newest-first.
		logRecord("a2", "fix(steerpr): second (fak steerpr)", "", "b.go"),
		logRecord("a1", "feat(steerpr): first (fak steerpr)", "", "a.go"),
		logRecord("b1", "docs(cache): only (fak cache)", "", "docs/c.md"),
	}, "")
	units, _ := FoldUnits(ParseLog(raw))
	if len(units) != 2 {
		t.Fatalf("units = %d, want 2", len(units))
	}
	if units[0].Leaf != "steerpr" {
		t.Errorf("units[0].Leaf = %q, want steerpr (biggest-first)", units[0].Leaf)
	}
	// A PR body reads oldest-first.
	if units[0].Commits[0].SHA != "a1" {
		t.Errorf("units[0].Commits[0] = %q, want a1 (oldest-first within a unit)", units[0].Commits[0].SHA)
	}
	// A single-commit unit titles as its subject.
	if units[1].Title != "docs(cache): only (fak cache)" {
		t.Errorf("single-commit title = %q, want the bare subject", units[1].Title)
	}
	// A multi-commit unit summarizes its type histogram.
	if !strings.Contains(units[0].Title, "2 commits") {
		t.Errorf("multi-commit title = %q, want a commit-count summary", units[0].Title)
	}
}

func TestFoldUnitsBandsEachUnitByWorstMember(t *testing.T) {
	commits := []Commit{
		{SHA: "a1", Leaf: "steerpr", Type: "feat", Verdict: VerdictWitnessed},
		{SHA: "a2", Leaf: "steerpr", Type: "fix", Verdict: VerdictUnwitnessed},
		{SHA: "b1", Leaf: "cache", Type: "docs", Verdict: VerdictWitnessed},
		{SHA: "c1", Leaf: "gateway", Type: "chore", Verdict: VerdictAbstain},
	}
	units, _ := FoldUnits(commits)
	got := map[string]Band{}
	for _, u := range units {
		got[u.Leaf] = u.Band
	}
	want := map[string]Band{
		"steerpr": BandResidual,     // one unwitnessed member reds the unit
		"cache":   BandCleared,      // all witnessed
		"gateway": BandUnverifiable, // no checkable claim
	}
	for leaf, w := range want {
		if got[leaf] != w {
			t.Errorf("unit %q band = %q, want %q", leaf, got[leaf], w)
		}
	}
}

// TestSortWorstFirstSurfacesAttentionFirst proves the operator view leads with
// what owes attention, not with what is biggest.
func TestSortWorstFirstSurfacesAttentionFirst(t *testing.T) {
	units := []Unit{
		{Leaf: "big-cleared", Band: BandCleared, Commits: make([]Commit, 10)},
		{Leaf: "small-residual", Band: BandResidual, Commits: make([]Commit, 1)},
		{Leaf: "mid-unverifiable", Band: BandUnverifiable, Commits: make([]Commit, 5)},
	}
	SortWorstFirst(units)
	want := []string{"small-residual", "mid-unverifiable", "big-cleared"}
	for i, w := range want {
		if units[i].Leaf != w {
			t.Errorf("units[%d].Leaf = %q, want %q (worst band must outrank size)", i, units[i].Leaf, w)
		}
	}
}

// TestSortWorstFirstIsATotalOrder pins the leaf tiebreak, so two units with the
// same band and size can never swap between runs on map-iteration randomness.
func TestSortWorstFirstIsATotalOrder(t *testing.T) {
	mk := func() []Unit {
		return []Unit{
			{Leaf: "zeta", Band: BandResidual, Commits: make([]Commit, 2)},
			{Leaf: "alpha", Band: BandResidual, Commits: make([]Commit, 2)},
		}
	}
	for i := 0; i < 50; i++ {
		u := mk()
		SortWorstFirst(u)
		if u[0].Leaf != "alpha" {
			t.Fatalf("iteration %d: units[0].Leaf = %q, want alpha (leaf tiebreak must be total)", i, u[0].Leaf)
		}
	}
}

// TestFoldIsDeterministic proves the same input yields a byte-identical
// payload. The osp_residual trend is meaningless if the fold is unstable.
func TestFoldIsDeterministic(t *testing.T) {
	raw := strings.Join([]string{
		logRecord("a1", "feat(steerpr): one (fak steerpr)", "", "a.go"),
		logRecord("a2", "fix(steerpr): two (fak steerpr)", "", "b.go"),
		logRecord("b1", "docs(cache): three (fak cache)", "#7", "docs/c.md"),
		logRecord("d1", "chore: orphan", "", "x.txt"),
	}, "")

	var first string
	for i := 0; i < 25; i++ {
		units, unstamped := FoldUnits(ParseLog(raw))
		SortWorstFirst(units)
		payload, err := json.Marshal(map[string]any{
			"schema": Schema, "units": units, "unstamped": unstamped,
		})
		if err != nil {
			t.Fatalf("marshal: %v", err)
		}
		if i == 0 {
			first = string(payload)
			continue
		}
		if string(payload) != first {
			t.Fatalf("fold is non-deterministic on iteration %d:\n first=%s\n  got=%s", i, first, payload)
		}
	}
}

// TestResidualIsIndependentOfMemberCount pins the headline number's meaning:
// it counts UNITS owing attention, not commits.
func TestResidualCountsUnitsOwingAttention(t *testing.T) {
	units := []Unit{
		{Leaf: "a", Band: BandResidual, Commits: make([]Commit, 9)},
		{Leaf: "b", Band: BandResidual, Commits: make([]Commit, 1)},
		{Leaf: "c", Band: BandCleared, Commits: make([]Commit, 4)},
		{Leaf: "d", Band: BandUnverifiable, Commits: make([]Commit, 2)},
	}
	if got := Residual(units); got != 2 {
		t.Errorf("Residual() = %d, want 2 (residual units, not commits)", got)
	}
	if got := Residual(nil); got != 0 {
		t.Errorf("Residual(nil) = %d, want 0", got)
	}
}

func TestIssuesSubjectBindingOutranksBodyMention(t *testing.T) {
	resolves := Issues("fix(x): thing (#10) (fak x)", nil)
	mentions := Issues("relates to #10 and #20", resolves)
	if len(resolves) != 1 || resolves[0] != "#10" {
		t.Fatalf("resolves = %v, want [#10]", resolves)
	}
	if len(mentions) != 1 || mentions[0] != "#20" {
		t.Fatalf("mentions = %v, want [#20] (#10 is already a closure)", mentions)
	}
}

func TestParseLogSkipsMalformedRecords(t *testing.T) {
	// Empty input, a record with too few fields, and an empty-subject record
	// must all be skipped without panicking.
	for _, raw := range []string{"", "\x1e", "\x1eonlysha", "\x1esha\x1f\x1fbody\x1f"} {
		if got := ParseLog(raw); len(got) != 0 {
			t.Errorf("ParseLog(%q) = %d commits, want 0", raw, len(got))
		}
	}
}
