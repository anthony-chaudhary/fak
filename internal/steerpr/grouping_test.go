package steerpr

// grouping_test.go — the #5040 acceptance gate: the MIXED case (some commits
// wave-bound, some leaf-only) groups correctly, the basis is explicit on every
// unit, and `release prplan`'s leaf fold is unchanged.

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"
)

// groupingCommits is the mixed corpus: two commits bound to wave 0 across TWO
// different leaves (the whole point — a wave spans lanes), one bound to wave 1,
// two with no wave binding at all, and one with no ship-stamp.
func groupingCommits() []Commit {
	return []Commit{
		{SHA: "a1", Subject: "feat(gateway): arm the cold-tool lever (#101) (fak gateway)", Leaf: "gateway", Type: "feat", Resolves: []string{"#101"}, Files: []string{"internal/gateway/a.go"}, Verdict: VerdictWitnessed},
		{SHA: "b2", Subject: "fix(model): unbreak the decode path (#102) (fak model)", Leaf: "model", Type: "fix", Resolves: []string{"#102"}, Files: []string{"internal/model/b.go"}, Verdict: VerdictUnwitnessed},
		{SHA: "c3", Subject: "test(ablate): pin the savings fixture (#103) (fak ablate)", Leaf: "ablate", Type: "test", Resolves: []string{"#103"}, Files: []string{"internal/ablate/c.go"}, Verdict: VerdictWitnessed},
		{SHA: "d4", Subject: "docs(gateway): record the lane split (#900) (fak gateway)", Leaf: "gateway", Type: "docs", Resolves: []string{"#900"}, Files: []string{"docs/d.md"}, Verdict: VerdictWitnessed},
		{SHA: "e5", Subject: "feat(bench): add the inventory row (#901) (fak bench)", Leaf: "bench", Type: "feat", Resolves: []string{"#901"}, Files: []string{"internal/bench/e.go"}, Verdict: VerdictAbstain},
		{SHA: "f6", Subject: "chore: sweep the tree", Files: []string{"x.txt"}},
	}
}

// groupingWaves binds #101 and #102 to wave 0 and #103 to wave 1 — the shape
// `fak issue cohort --from-issues` produces.
func groupingWaves() []WaveBinding {
	return []WaveBinding{
		{Index: 0, Issues: []string{"#101", "102"}},
		{Index: 1, Issues: []string{"#103"}},
	}
}

func unitByKey(t *testing.T, units []Unit, key string) Unit {
	t.Helper()
	for _, u := range units {
		if u.Leaf == key {
			return u
		}
	}
	t.Fatalf("no unit keyed %q in %v", key, unitKeys(units))
	return Unit{}
}

func unitKeys(units []Unit) []string {
	out := make([]string, 0, len(units))
	for _, u := range units {
		out = append(out, u.Leaf+"/"+u.GroupedBy)
	}
	return out
}

// TestGroupingMixedWaveAndLeaf is the acceptance gate's first half: in a corpus
// where some commits are wave-bound and some are not, the wave-bound ones fold
// into one unit PER WAVE (across leaves) and the rest fall back to leaf.
func TestGroupingMixedWaveAndLeaf(t *testing.T) {
	units, unstamped := FoldUnitsByWave(groupingCommits(), WaveIndex(groupingWaves()))

	if len(units) != 4 {
		t.Fatalf("units = %v, want 4 (wave:0, wave:1, gateway, bench)", unitKeys(units))
	}
	w0 := unitByKey(t, units, "wave:0")
	if len(w0.Commits) != 2 {
		t.Fatalf("wave:0 commits = %d, want 2 (a1 gateway + b2 model)", len(w0.Commits))
	}
	// The load-bearing property: one wave unit spans TWO leaves. Leaf grouping
	// could never produce this unit, which is why the regrouping exists.
	if !reflect.DeepEqual(w0.Leaves, []string{"gateway", "model"}) {
		t.Fatalf("wave:0 leaves = %v, want [gateway model] — a wave unit must span its lanes", w0.Leaves)
	}
	w1 := unitByKey(t, units, "wave:1")
	if len(w1.Commits) != 1 || !reflect.DeepEqual(w1.Leaves, []string{"ablate"}) {
		t.Fatalf("wave:1 = %d commit(s) leaves %v, want 1 commit in [ablate]", len(w1.Commits), w1.Leaves)
	}

	// The fallback half: an unbound commit keeps its leaf unit, and the gateway
	// leaf unit holds ONLY the unbound gateway commit — a1 left for wave:0.
	gw := unitByKey(t, units, "gateway")
	if len(gw.Commits) != 1 || gw.Commits[0].SHA != "d4" {
		t.Fatalf("gateway unit = %+v, want only the unbound d4", gw.Commits)
	}
	if bench := unitByKey(t, units, "bench"); len(bench.Commits) != 1 {
		t.Fatalf("bench unit = %d commit(s), want 1", len(bench.Commits))
	}

	// The partition stays total and disjoint: no commit is duplicated or dropped.
	seen := map[string]int{}
	for _, u := range units {
		for _, c := range u.Commits {
			seen[c.SHA]++
		}
	}
	for _, c := range unstamped {
		seen[c.SHA]++
	}
	if len(seen) != len(groupingCommits()) {
		t.Fatalf("membership = %v, want every commit exactly once", seen)
	}
	for sha, n := range seen {
		if n != 1 {
			t.Fatalf("commit %s appears %d times, want exactly 1", sha, n)
		}
	}
	if len(unstamped) != 1 || unstamped[0].SHA != "f6" {
		t.Fatalf("unstamped = %+v, want only the unstamped f6 — a wave binding must not absorb legibility debt", unstamped)
	}
}

// TestGroupingBasisIsExplicitOnEveryUnit is the issue's stated fail condition
// inverted: two bases coexisting WITHOUT an explicit grouped_by is a fail, so
// every unit — both bases — must carry one, and it must survive JSON.
func TestGroupingBasisIsExplicitOnEveryUnit(t *testing.T) {
	units, _ := FoldUnitsByWave(groupingCommits(), WaveIndex(groupingWaves()))
	for _, u := range units {
		want := GroupedByLeaf
		if IsWaveKey(u.Leaf) {
			want = GroupedByWave
		}
		if u.GroupedBy != want {
			t.Fatalf("unit %q grouped_by = %q, want %q", u.Leaf, u.GroupedBy, want)
		}
		if GroupingBasis(u) != want {
			t.Fatalf("unit %q renders basis %q, want %q", u.Leaf, GroupingBasis(u), want)
		}
		raw, err := json.Marshal(u)
		if err != nil {
			t.Fatalf("marshal unit %q: %v", u.Leaf, err)
		}
		if !strings.Contains(string(raw), `"grouped_by":"`+want+`"`) {
			t.Fatalf("unit %q json = %s, want an explicit grouped_by", u.Leaf, raw)
		}
	}
	// An unset basis is reported as unknown, never defaulted to the common case:
	// silently reading leaf would recreate exactly the guessing this removes.
	if got := GroupingBasis(Unit{Leaf: "gateway"}); got != "unknown" {
		t.Fatalf("GroupingBasis(unset) = %q, want %q", got, "unknown")
	}
}

// TestGroupingLeafFallbackWhenNoWaves proves the fallback is the DEFAULT: with
// no bindings at all (the common case), the fold is pure leaf grouping.
func TestGroupingLeafFallbackWhenNoWaves(t *testing.T) {
	for name, waveOf := range map[string]map[string]string{
		"nil":            nil,
		"empty":          {},
		"no-match":       WaveIndex([]WaveBinding{{Index: 0, Issues: []string{"#77777"}}}),
		"non-numeric":    WaveIndex([]WaveBinding{{Index: 0, Issues: []string{"steerpr-int-cohort", ""}}}),
		"mention-only":   WaveIndex([]WaveBinding{{Index: 3, Issues: []string{"#5015"}}}),
		"unstamped-only": WaveIndex([]WaveBinding{{Index: 4, Issues: []string{"#404"}}}),
	} {
		units, unstamped := FoldUnitsByWave(groupingCommits(), waveOf)
		base, baseUnstamped := FoldUnits(groupingCommits())
		if !reflect.DeepEqual(units, base) || !reflect.DeepEqual(unstamped, baseUnstamped) {
			t.Fatalf("%s: fold = %v, want the leaf fold %v", name, unitKeys(units), unitKeys(base))
		}
		for _, u := range units {
			if u.GroupedBy != GroupedByLeaf {
				t.Fatalf("%s: unit %q grouped_by = %q, want leaf", name, u.Leaf, u.GroupedBy)
			}
		}
	}
}

// TestGroupingWaveBandFoldsWorstMember: a wave unit is banded by the SAME
// pessimistic worst-member rule as a leaf unit. Regrouping must never clear a
// band — wave:0 holds one witnessed and one unwitnessed commit, so it is
// RESIDUAL, not CLEARED.
func TestGroupingWaveBandFoldsWorstMember(t *testing.T) {
	units, _ := FoldUnitsByWave(groupingCommits(), WaveIndex(groupingWaves()))
	if got := unitByKey(t, units, "wave:0").Band; got != BandResidual {
		t.Fatalf("wave:0 band = %q, want %q — the worst member (b2, unwitnessed) must win", got, BandResidual)
	}
	if got := unitByKey(t, units, "wave:1").Band; got != BandCleared {
		t.Fatalf("wave:1 band = %q, want %q (its only member was witnessed)", got, BandCleared)
	}
	// The fence in one assertion: a wave whose members are all witnessed EXCEPT
	// one cannot be cleared by regrouping the witnessed ones around it.
	mixed := []Commit{
		{SHA: "z1", Subject: "feat(a): x (#101) (fak a)", Leaf: "a", Resolves: []string{"#101"}, Verdict: VerdictWitnessed},
		{SHA: "z2", Subject: "feat(b): y (#102) (fak b)", Leaf: "b", Resolves: []string{"#102"}, Verdict: VerdictWitnessed},
		{SHA: "z3", Subject: "feat(c): z (#103) (fak c)", Leaf: "c", Resolves: []string{"#103"}, Verdict: VerdictUnwitnessed},
	}
	all := WaveIndex([]WaveBinding{{Index: 7, Issues: []string{"#101", "#102", "#103"}}})
	units, _ = FoldUnitsByWave(mixed, all)
	if len(units) != 1 || units[0].Band != BandResidual {
		t.Fatalf("single wave unit = %+v, want one RESIDUAL unit", units)
	}
}

// TestGroupingPRPlanFoldUnchanged is the acceptance gate's second half:
// `release prplan` calls FoldUnits, whose grouping must be untouched by this
// change — same units, same membership, same order, all grouped_by leaf, even
// when wave bindings exist for the very same commits.
func TestGroupingPRPlanFoldUnchanged(t *testing.T) {
	commits := groupingCommits()
	units, unstamped := FoldUnits(commits)

	// prplan's fold ignores wave bindings by construction: it takes none.
	if len(units) != 4 {
		t.Fatalf("prplan units = %v, want one per leaf (gateway, model, ablate, bench)", unitKeys(units))
	}
	for _, u := range units {
		if u.GroupedBy != GroupedByLeaf {
			t.Fatalf("prplan unit %q grouped_by = %q, want leaf — prplan stays leaf-grouped", u.Leaf, u.GroupedBy)
		}
		if len(u.Leaves) != 0 {
			t.Fatalf("prplan unit %q carries leaves %v, want none on a leaf unit", u.Leaf, u.Leaves)
		}
	}
	// Biggest-first, then by leaf: gateway holds 2, the rest 1 each.
	if units[0].Leaf != "gateway" || len(units[0].Commits) != 2 {
		t.Fatalf("prplan order = %v, want gateway (2 commits) first", unitKeys(units))
	}
	wantOrder := []string{"gateway", "ablate", "bench", "model"}
	for i, want := range wantOrder {
		if units[i].Leaf != want {
			t.Fatalf("prplan order = %v, want %v", unitKeys(units), wantOrder)
		}
	}
	if len(unstamped) != 1 || unstamped[0].SHA != "f6" {
		t.Fatalf("prplan unstamped = %+v, want the one unstamped commit", unstamped)
	}

	// And the invariance is structural, not incidental: folding the same commits
	// with bindings present changes the WAVE view only — prplan re-folds
	// byte-identically afterwards, so the two surfaces cannot drift.
	_, _ = FoldUnitsByWave(commits, WaveIndex(groupingWaves()))
	again, againUnstamped := FoldUnits(commits)
	if !reflect.DeepEqual(units, again) || !reflect.DeepEqual(unstamped, againUnstamped) {
		t.Fatal("prplan's fold changed after a wave fold ran over the same commits")
	}
	a, err := json.Marshal(units)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	b, err := json.Marshal(again)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if string(a) != string(b) {
		t.Fatalf("prplan units not byte-identical across folds:\n%s\n%s", a, b)
	}
}

// TestGroupingWaveIndexFirstBindingWins pins the ambiguity rule: an issue
// claimed by two waves stays with the first, so a later wave cannot silently
// steal a commit out of an earlier wave's unit.
func TestGroupingWaveIndexFirstBindingWins(t *testing.T) {
	idx := WaveIndex([]WaveBinding{
		{Index: 0, Issues: []string{"#101"}},
		{Index: 1, Issues: []string{"#101", "#102"}},
	})
	if idx["#101"] != WaveKey(0) {
		t.Fatalf("#101 -> %q, want %q (first binding wins)", idx["#101"], WaveKey(0))
	}
	if idx["#102"] != WaveKey(1) {
		t.Fatalf("#102 -> %q, want %q", idx["#102"], WaveKey(1))
	}
	// Deterministic across runs: the map is built from binding order, never from
	// map iteration.
	for i := 0; i < 8; i++ {
		if again := WaveIndex([]WaveBinding{{Index: 0, Issues: []string{"#101"}}, {Index: 1, Issues: []string{"#101"}}}); again["#101"] != WaveKey(0) {
			t.Fatalf("run %d: #101 -> %q, want stable %q", i, again["#101"], WaveKey(0))
		}
	}
}

// TestGroupingWaveKeyCannotCollideWithALeaf: the wave key space and the leaf key
// space are disjoint by construction. A lane literally named `wave-0` must not
// absorb wave 0's commits — the ship-stamp grammar cannot produce the colon.
func TestGroupingWaveKeyCannotCollideWithALeaf(t *testing.T) {
	commits := []Commit{
		{SHA: "p1", Subject: "feat(x): a (#101) (fak wave-0)", Leaf: "wave-0", Resolves: []string{"#101"}},
		{SHA: "p2", Subject: "feat(x): b (#900) (fak wave-0)", Leaf: "wave-0", Resolves: []string{"#900"}},
	}
	units, _ := FoldUnitsByWave(commits, WaveIndex([]WaveBinding{{Index: 0, Issues: []string{"#101"}}}))
	if len(units) != 2 {
		t.Fatalf("units = %v, want the wave unit and the wave-0 LEAF unit kept apart", unitKeys(units))
	}
	if got := unitByKey(t, units, "wave:0"); len(got.Commits) != 1 || got.Commits[0].SHA != "p1" {
		t.Fatalf("wave:0 = %+v, want only the bound p1", got.Commits)
	}
	if got := unitByKey(t, units, "wave-0"); len(got.Commits) != 1 || got.Commits[0].SHA != "p2" {
		t.Fatalf("leaf wave-0 = %+v, want only the unbound p2", got.Commits)
	}
	// The parser cannot mint a wave key from a stamp, in either direction.
	if IsWaveKey("wave-0") || !IsWaveKey(WaveKey(12)) {
		t.Fatalf("IsWaveKey misclassifies: wave-0=%v wave:12=%v", IsWaveKey("wave-0"), IsWaveKey(WaveKey(12)))
	}
	for _, c := range ParseLog("\x1eabc\x1ffeat(x): y (fak wave:0)\x1f\x1ff.go\n") {
		if c.Leaf != "" {
			t.Fatalf("a colon-bearing stamp parsed to leaf %q, want no leaf", c.Leaf)
		}
	}
}

// TestGroupingWaveBindsOnClosureNotMention: grouping follows the closure-grade
// subject binding only. A body mention of a wave member's issue is not a
// statement of membership, and grouping on it would pull unrelated work into an
// operator's decision unit.
func TestGroupingWaveBindsOnClosureNotMention(t *testing.T) {
	commits := []Commit{
		{SHA: "m1", Subject: "feat(gateway): x (#101) (fak gateway)", Leaf: "gateway", Resolves: []string{"#101"}},
		{SHA: "m2", Subject: "feat(model): y (fak model)", Leaf: "model", Mentions: []string{"#101"}},
	}
	units, _ := FoldUnitsByWave(commits, WaveIndex([]WaveBinding{{Index: 0, Issues: []string{"#101"}}}))
	if len(units) != 2 {
		t.Fatalf("units = %v, want the mention-only commit left in its leaf unit", unitKeys(units))
	}
	if got := unitByKey(t, units, "model"); got.GroupedBy != GroupedByLeaf {
		t.Fatalf("mention-only unit grouped_by = %q, want leaf", got.GroupedBy)
	}
}

// TestGroupingWaveUnitsIsCountable backs the header count an operator reads
// before opening any unit.
func TestGroupingWaveUnitsIsCountable(t *testing.T) {
	units, _ := FoldUnitsByWave(groupingCommits(), WaveIndex(groupingWaves()))
	if got := len(WaveUnits(units)); got != 2 {
		t.Fatalf("wave unit count = %d, want 2", got)
	}
	leafOnly, _ := FoldUnits(groupingCommits())
	if got := len(WaveUnits(leafOnly)); got != 0 {
		t.Fatalf("wave unit count on a leaf fold = %d, want 0", got)
	}
}
