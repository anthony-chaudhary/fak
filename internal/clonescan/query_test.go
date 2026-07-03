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
