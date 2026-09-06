package skillfootprint

import (
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/capindex"
)

// skillCard builds one at-rest skill card with a chosen name and resident
// description, so a fold assertion can be written against exact byte lengths.
func skillCard(name, desc string) capindex.CapCard {
	return capindex.CapCard{
		Ref:       capindex.CapRef{Kind: capindex.CapKindSkill, Name: name},
		Trigger:   desc,
		CardBytes: []byte(`{"name":"` + name + `"}`),
	}
}

// repoRootForTest locates the module root from THIS test's own source path, so the
// real-tree probes below are independent of the working directory a runner picks.
func repoRootForTest(tb testing.TB) string {
	tb.Helper()
	_, thisFile, _, ok := runtime.Caller(0)
	if !ok {
		tb.Fatal("runtime.Caller(0) failed — cannot locate the test source path")
	}
	return filepath.Clean(filepath.Join(filepath.Dir(thisFile), "..", ".."))
}

// TestFoldPartitionsEveryFloorExactly pins the property the gate depends on: each
// of the three floors is the EXACT sum of its per-entry column, with no rounding
// and nothing dropped. If the fold ever stopped being a faithful partition, the
// gated number would silently stop describing the rows the trimmer reads.
func TestFoldPartitionsEveryFloorExactly(t *testing.T) {
	cards := []capindex.CapCard{
		skillCard("release", "Use when you cut a versioned release"),
		skillCard("verify", "Use when you bind a done-claim to a green test run"),
		skillCard("dos-dispatch", "Use when you take a lane through the arbiter"),
	}
	fp := Fold(cards)

	if fp.SkillCount != len(cards) {
		t.Fatalf("SkillCount = %d, want %d", fp.SkillCount, len(cards))
	}
	if len(fp.Entries) != len(cards) {
		t.Fatalf("Entries = %d, want %d", len(fp.Entries), len(cards))
	}
	var desc, name, card int
	for _, e := range fp.Entries {
		if e.DescBytes < 0 || e.NameBytes != len(e.Name) {
			t.Errorf("entry %q: NameBytes=%d, want %d", e.Name, e.NameBytes, len(e.Name))
		}
		desc += e.DescBytes
		name += e.NameBytes
		card += e.CardBytes
	}
	if desc != fp.DescFloor {
		t.Errorf("DescFloor = %d, want sum of entry DescBytes = %d", fp.DescFloor, desc)
	}
	if name != fp.NameFloor {
		t.Errorf("NameFloor = %d, want sum of entry NameBytes = %d", fp.NameFloor, name)
	}
	if card != fp.CardFloor {
		t.Errorf("CardFloor = %d, want sum of entry CardBytes = %d", fp.CardFloor, card)
	}
	// The name-only floor (#3612 headless profile) must be strictly smaller than the
	// name+description floor (#3234 interactive profile) whenever descriptions exist.
	if fp.NameFloor >= fp.DescFloor {
		t.Errorf("NameFloor (%d) must be < DescFloor (%d)", fp.NameFloor, fp.DescFloor)
	}
}

// TestFoldOrdersHeaviestFirstDeterministically pins the ranking contract: heaviest
// resident description leads, and equal-weight rows break by name then kind — so
// two runs over the same catalog render identically and a trimmer always sees the
// biggest target at the top.
func TestFoldOrdersHeaviestFirstDeterministically(t *testing.T) {
	cards := []capindex.CapCard{
		skillCard("bbb", "same"),
		skillCard("heavy", strings.Repeat("x", 100)),
		skillCard("aaa", "same"),
	}
	fp := Fold(cards)
	got := []string{fp.Entries[0].Name, fp.Entries[1].Name, fp.Entries[2].Name}
	want := []string{"heavy", "aaa", "bbb"}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("order = %v, want %v (heaviest first, ties by name)", got, want)
		}
	}
	// Determinism: the same input folds to the same order every time.
	again := Fold(cards)
	for i := range again.Entries {
		if again.Entries[i].Name != fp.Entries[i].Name {
			t.Fatalf("fold is not deterministic: %v vs %v", again.Entries, fp.Entries)
		}
	}
}

// TestSkillsDirIsTheClaudeSkillsPath pins the one place the tree layout is spelled.
// The verb, the gate and the real-tree test all route through it, so a layout change
// can never leave one of them pricing a different directory.
func TestSkillsDirIsTheClaudeSkillsPath(t *testing.T) {
	want := filepath.Join("root", ".claude", "skills")
	if got := SkillsDir("root"); got != want {
		t.Fatalf("SkillsDir = %q, want %q", got, want)
	}
}

// TestMeasureReadsTheRealSkillsTree is the userland analog of
// mcpfootprint.TestRealFakMCPFloor: it prices fak's ACTUAL .claude/skills tree
// through the shipped SkillResolver and proves the number is non-trivial and a
// faithful partition — so the baseline doc's figure is reproducible, not hand-typed.
func TestMeasureReadsTheRealSkillsTree(t *testing.T) {
	fp := Measure(repoRootForTest(t))
	if fp.SkillCount == 0 {
		t.Fatal("real .claude/skills tree priced as 0 skills — the resolver is not seeing the tree")
	}
	sum := 0
	for _, e := range fp.Entries {
		sum += e.DescBytes
	}
	if sum != fp.DescFloor {
		t.Fatalf("real tree is not a faithful partition: sum=%d != DescFloor=%d", sum, fp.DescFloor)
	}
	if fp.DescFloor <= 0 {
		t.Fatalf("real resident description floor priced as %d bytes", fp.DescFloor)
	}
	t.Logf("resident .claude/skills description floor: %d skills, %d bytes (~%d est. tokens); name-only floor %d B; card floor %d B",
		fp.SkillCount, fp.DescFloor, ApproxTokens(fp.DescFloor), fp.NameFloor, fp.CardFloor)
	for i := 0; i < min(8, len(fp.Entries)); i++ {
		t.Logf("  top %d: %5d B  %s", i+1, fp.Entries[i].DescBytes, fp.Entries[i].Name)
	}
}

// TestMeasureOnAnAbsentTreeFoldsToZeroAndRefuses proves the fail-closed posture end
// to end: a root with no skills directory does not error out into a green pass, it
// folds to a zero floor and the gate refuses it as STALE. A broken probe must never
// read as a satisfied budget.
func TestMeasureOnAnAbsentTreeFoldsToZeroAndRefuses(t *testing.T) {
	fp := Measure(t.TempDir())
	if fp.SkillCount != 0 || fp.DescFloor != 0 {
		t.Fatalf("absent skills tree folded to %+v, want a zero Floor", fp)
	}
	if err := CheckDescriptions(fp); err == nil {
		t.Fatal("the gate admitted a tree it could not read — it greens on measuring nothing")
	}
}

// TestApproxTokensUsesTheHouseDivisor pins the estimator seam: the printed token
// figure is the byte floor over the single named divisor, never a second estimator.
func TestApproxTokensUsesTheHouseDivisor(t *testing.T) {
	if got := ApproxTokens(4 * 1234); got != 1234 {
		t.Fatalf("ApproxTokens(4936) = %d, want 1234", got)
	}
	if BytesPerTokenEstimate != 4 {
		t.Fatalf("BytesPerTokenEstimate = %d, want the house ~4 bytes/token divisor", BytesPerTokenEstimate)
	}
}
