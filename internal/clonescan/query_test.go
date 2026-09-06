package clonescan

import "testing"

// A block long enough to exceed WindowTokens (34) with real logic tokens.
const sampleBlock = `
func process(items []int) int {
	total := 0
	for i := 0; i < len(items); i++ {
		if items[i] > 0 {
			total += items[i] * 2
		} else {
			total -= items[i]
		}
	}
	return total
}
`

func TestQueryFindsIdenticalBlock(t *testing.T) {
	tree := map[string]string{
		"existing.go": "package a\n" + sampleBlock,
		"unrelated.go": `package a
func hello() string { return "world" }
`,
	}
	got := Query(sampleBlock, tree, "", 0)
	if len(got) == 0 {
		t.Fatalf("expected the identical block in existing.go to be found, got none")
	}
	if got[0].File != "existing.go" {
		t.Fatalf("top match = %q, want existing.go", got[0].File)
	}
	if got[0].Windows < 1 {
		t.Fatalf("match reported %d windows, want >=1", got[0].Windows)
	}
}

func TestQueryExcludesSelfPath(t *testing.T) {
	tree := map[string]string{
		"mine.go": "package a\n" + sampleBlock,
	}
	// Querying the block that lives at mine.go, telling Query it is mine.go, must
	// not report mine.go as its own duplicate.
	got := Query(sampleBlock, tree, "mine.go", 0)
	if len(got) != 0 {
		t.Fatalf("self-path should be excluded, got %d matches: %+v", len(got), got)
	}
}

func TestQueryRenameChangesIdentityByDesign(t *testing.T) {
	// Identifiers are kept verbatim (normalizeIdents=false) to match the scorecard's
	// precision choice: distinct code with distinct names must not false-match. So a
	// block with every identifier renamed is NOT reported as a duplicate. This test
	// pins that deliberate contract — if it flips, the query and the scorecard have
	// drifted on what a clone is.
	renamed := `
func compute(values []int) int {
	sum := 0
	for k := 0; k < len(values); k++ {
		if values[k] > 0 {
			sum += values[k] * 2
		} else {
			sum -= values[k]
		}
	}
	return sum
}
`
	tree := map[string]string{"existing.go": "package a\n" + sampleBlock}
	got := Query(renamed, tree, "", 0)
	if len(got) != 0 {
		t.Fatalf("renamed block should not match (identifiers kept), got %+v", got)
	}
}

func TestQueryIgnoresDataBlocks(t *testing.T) {
	// A pure declaration / composite-literal block carries no non-assignment logic,
	// so no window qualifies and it is never a clone — even duplicated verbatim.
	data := `
var table = []struct {
	name  string
	value int
	label string
	extra string
	more  string
	again string
}{
	{"a", 1, "x", "p", "q", "r"},
	{"b", 2, "y", "s", "t", "u"},
	{"c", 3, "z", "v", "w", "xx"},
}
`
	tree := map[string]string{"existing.go": "package a\n" + data}
	got := Query(data, tree, "", 0)
	if len(got) != 0 {
		t.Fatalf("a data/declaration block should never be a clone, got %+v", got)
	}
}

func TestQueryEmptyCandidate(t *testing.T) {
	tree := map[string]string{"existing.go": "package a\n" + sampleBlock}
	if got := Query("", tree, "", 0); got != nil {
		t.Fatalf("empty candidate should yield no matches, got %+v", got)
	}
	if got := Query("func f() {}", tree, "", 0); got != nil {
		t.Fatalf("sub-window candidate should yield no matches, got %+v", got)
	}
}

func TestQueryRanksByOverlap(t *testing.T) {
	// A file containing TWO copies of the block shares more windows than a file with
	// one copy, so it ranks first.
	tree := map[string]string{
		"one.go": "package a\n" + sampleBlock,
		"two.go": "package a\n" + sampleBlock + "\n" + sampleBlock,
	}
	got := Query(sampleBlock, tree, "", 0)
	if len(got) != 2 {
		t.Fatalf("expected both files matched, got %d: %+v", len(got), got)
	}
	if got[0].File != "two.go" {
		t.Fatalf("file with two copies should rank first, got %q", got[0].File)
	}
	if got[0].Windows <= got[1].Windows {
		t.Fatalf("expected two.go (%d) > one.go (%d) windows", got[0].Windows, got[1].Windows)
	}
}

func TestQueryMaxResultsCap(t *testing.T) {
	tree := map[string]string{
		"a.go": "package a\n" + sampleBlock,
		"b.go": "package a\n" + sampleBlock,
		"c.go": "package a\n" + sampleBlock,
	}
	got := Query(sampleBlock, tree, "", 2)
	if len(got) != 2 {
		t.Fatalf("maxResults=2 should cap to 2, got %d", len(got))
	}
}

func TestGoTokensDropsCommentsAndWhitespace(t *testing.T) {
	a := goTokens("func f(){ x := 1 // comment\n }", false)
	b := goTokens("func   f ( ) {\n\tx := 1\n}", false)
	if len(a) != len(b) {
		t.Fatalf("comment/whitespace should not change token count: %d vs %d", len(a), len(b))
	}
	for i := range a {
		if a[i].sym != b[i].sym {
			t.Fatalf("token %d differs: %q vs %q", i, a[i].sym, b[i].sym)
		}
	}
}

func TestGoTokensLiteralsCollapse(t *testing.T) {
	toks := goTokens(`x := "hello"`, false)
	// x := L  -> the string literal collapses to L
	if len(toks) == 0 || toks[len(toks)-1].sym != "L" {
		t.Fatalf("string literal should collapse to L, got %+v", toks)
	}
}

func TestCoalesceCloneWindowsOverlappingAndDisjoint(t *testing.T) {
	// Case 1: Overlapping clone windows sharing matching site cardinality and files
	// coalesce into a single representative match.
	w1 := CloneWindow{
		Key: "k1",
		Sites: []CloneSite{
			{File: "pkg/a.go", StartLine: 10, EndLine: 20},
		},
	}
	w2 := CloneWindow{
		Key: "k2",
		Sites: []CloneSite{
			{File: "pkg/a.go", StartLine: 15, EndLine: 25},
		},
	}

	gotCoalesced := CoalesceCloneWindows([]CloneWindow{w1, w2})
	if len(gotCoalesced) != 1 {
		t.Fatalf("overlapping clone windows should coalesce into 1 match, got %d: %+v", len(gotCoalesced), gotCoalesced)
	}
	if gotCoalesced[0].File != "pkg/a.go" {
		t.Errorf("file = %q, want pkg/a.go", gotCoalesced[0].File)
	}
	if gotCoalesced[0].StartLine != 10 || gotCoalesced[0].EndLine != 25 {
		t.Errorf("span = [%d, %d], want [10, 25]", gotCoalesced[0].StartLine, gotCoalesced[0].EndLine)
	}
	if gotCoalesced[0].Windows != 2 {
		t.Errorf("windows = %d, want 2", gotCoalesced[0].Windows)
	}

	// Case 2: Disjoint clone windows must remain separately reported.
	wDisjoint := CloneWindow{
		Key: "k3",
		Sites: []CloneSite{
			{File: "pkg/a.go", StartLine: 100, EndLine: 110},
		},
	}
	gotDisjoint := CoalesceCloneWindows([]CloneWindow{w1, wDisjoint})
	if len(gotDisjoint) != 2 {
		t.Fatalf("disjoint clone windows must remain separately reported (got %d, want 2)", len(gotDisjoint))
	}
	// Verify both disjoint spans are preserved
	spans := map[[2]int]bool{}
	for _, m := range gotDisjoint {
		spans[[2]int{m.StartLine, m.EndLine}] = true
	}
	if !spans[[2]int{10, 20}] || !spans[[2]int{100, 110}] {
		t.Errorf("expected separate spans [10, 20] and [100, 110], got %+v", gotDisjoint)
	}

	// Case 3: Multi-site candidate clone windows overlapping at EVERY site coalesce.
	wMulti1 := CloneWindow{
		Key: "km1",
		Sites: []CloneSite{
			{File: "pkg/a.go", StartLine: 10, EndLine: 20},
			{File: "pkg/b.go", StartLine: 30, EndLine: 40},
		},
	}
	wMulti2 := CloneWindow{
		Key: "km2",
		Sites: []CloneSite{
			{File: "pkg/a.go", StartLine: 12, EndLine: 22},
			{File: "pkg/b.go", StartLine: 32, EndLine: 42},
		},
	}
	gotMulti := CoalesceCloneWindows([]CloneWindow{wMulti1, wMulti2})
	if len(gotMulti) != 2 {
		t.Fatalf("multi-site coalesced group with 2 sites should emit 2 matches (1 per site), got %d: %+v", len(gotMulti), gotMulti)
	}
	for _, m := range gotMulti {
		if m.Windows != 2 {
			t.Errorf("match %s:%d-%d has windows=%d, want 2", m.File, m.StartLine, m.EndLine, m.Windows)
		}
		if m.File == "pkg/a.go" && (m.StartLine != 10 || m.EndLine != 22) {
			t.Errorf("pkg/a.go span = [%d, %d], want [10, 22]", m.StartLine, m.EndLine)
		}
		if m.File == "pkg/b.go" && (m.StartLine != 30 || m.EndLine != 42) {
			t.Errorf("pkg/b.go span = [%d, %d], want [30, 42]", m.StartLine, m.EndLine)
		}
	}

	// Case 4: Partial overlap (overlap at site 0, but disjoint at site 1) must NOT coalesce.
	wPartialDisjoint := CloneWindow{
		Key: "km3",
		Sites: []CloneSite{
			{File: "pkg/a.go", StartLine: 12, EndLine: 22},
			{File: "pkg/b.go", StartLine: 300, EndLine: 310},
		},
	}
	gotPartial := CoalesceCloneWindows([]CloneWindow{wMulti1, wPartialDisjoint})
	if len(gotPartial) != 4 {
		t.Fatalf("partially disjoint multi-site windows must NOT coalesce (got %d matches, want 4): %+v", len(gotPartial), gotPartial)
	}

	// Case 5: Distinct occurrence sets (mismatched site cardinality) must NOT coalesce.
	wDiffCard := CloneWindow{
		Key: "kd1",
		Sites: []CloneSite{
			{File: "pkg/a.go", StartLine: 12, EndLine: 22},
		},
	}
	gotDiffCard := CoalesceCloneWindows([]CloneWindow{wMulti1, wDiffCard})
	if len(gotDiffCard) != 3 {
		t.Fatalf("mismatched site cardinality (2 vs 1) must NOT coalesce (got %d matches, want 3): %+v", len(gotDiffCard), gotDiffCard)
	}
}

func TestCoalesceMatches(t *testing.T) {
	// Overlapping matches in the same file coalesce; disjoint matches in the same file
	// or different files remain separately reported.
	matches := []Match{
		{File: "a.go", StartLine: 10, EndLine: 20, Windows: 2},
		{File: "a.go", StartLine: 15, EndLine: 25, Windows: 3},
		{File: "a.go", StartLine: 100, EndLine: 110, Windows: 1},
		{File: "b.go", StartLine: 5, EndLine: 15, Windows: 1},
	}
	got := CoalesceMatches(matches)
	if len(got) != 3 {
		t.Fatalf("expected 3 matches after coalescing (2 for a.go, 1 for b.go), got %d: %+v", len(got), got)
	}
	var aSpans [][2]int
	for _, m := range got {
		if m.File == "a.go" {
			aSpans = append(aSpans, [2]int{m.StartLine, m.EndLine})
		}
	}
	if len(aSpans) != 2 {
		t.Fatalf("expected 2 distinct spans for a.go, got %d: %+v", len(aSpans), aSpans)
	}
	if aSpans[0] != [2]int{10, 25} || aSpans[1] != [2]int{100, 110} {
		t.Errorf("a.go spans = %+v, want [10, 25] and [100, 110]", aSpans)
	}
}

func TestQueryCoalescedEndToEnd(t *testing.T) {
	computeBlock := `
func compute(items []int) int {
	sum := 0
	for j := 0; j < len(items); j++ {
		if items[j] > 0 {
			sum += items[j] * 2
		} else {
			sum -= items[j]
		}
	}
	return sum
}
`
	unrelatedBlock := `
func unrelatedHelper(values []string) int {
	count := 0
	for idx := 0; idx < len(values); idx++ {
		if len(values[idx]) > 3 {
			count += len(values[idx])
		} else {
			count--
		}
	}
	return count
}
`
	// In target.go, process is at top, separated by unrelated code from compute.
	targetSrc := "package p\n" + sampleBlock + "\n" + unrelatedBlock + "\n" + computeBlock
	tree := map[string]string{
		"target.go": targetSrc,
	}

	// 1. Querying only sampleBlock (which has overlapping windows internally)
	// coalesces into a single representative match for target.go.
	gotSingle := QueryCoalesced(sampleBlock, tree, "", 0)
	if len(gotSingle) != 1 {
		t.Fatalf("overlapping candidate windows should yield 1 representative match, got %d: %+v", len(gotSingle), gotSingle)
	}
	if gotSingle[0].File != "target.go" {
		t.Errorf("file = %q, want target.go", gotSingle[0].File)
	}
	if gotSingle[0].Windows < 1 {
		t.Errorf("windows = %d, want >= 1", gotSingle[0].Windows)
	}

	novelSeparator := `
func novelSeparator(tags map[string]int) string {
	for k, v := range tags {
		if v > 100 {
			return k
		}
	}
	return "none"
}
`
	// 2. Querying candidate with both sampleBlock and computeBlock (two disjoint clone blocks
	// separated by non-matching logic code) yields 2 separately reported matches in target.go.
	combinedCand := sampleBlock + "\n" + novelSeparator + "\n" + computeBlock
	gotDisjoint := QueryCoalesced(combinedCand, tree, "", 0)
	if len(gotDisjoint) != 2 {
		t.Fatalf("candidate with two disjoint clone regions must yield 2 separately reported matches, got %d: %+v", len(gotDisjoint), gotDisjoint)
	}
	if gotDisjoint[0].File != "target.go" || gotDisjoint[1].File != "target.go" {
		t.Errorf("both matches should be in target.go, got %+v", gotDisjoint)
	}
	// Verify the two matches have disjoint line spans
	m1, m2 := gotDisjoint[0], gotDisjoint[1]
	if m1.StartLine > m2.StartLine {
		m1, m2 = m2, m1
	}
	if m1.EndLine >= m2.StartLine {
		t.Errorf("matches should be disjoint, but m1=[%d, %d] and m2=[%d, %d] overlap", m1.StartLine, m1.EndLine, m2.StartLine, m2.EndLine)
	}
}
