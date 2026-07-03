package main

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/clonescan"
)

func TestParseAddedGoExtractsAddedLinesPerFile(t *testing.T) {
	diff := strings.Join([]string{
		"diff --git a/foo.go b/foo.go",
		"--- a/foo.go",
		"+++ b/foo.go",
		"@@ -0,0 +1,2 @@",
		"+func added() {}",
		"+var x = 1",
		"diff --git a/bar.go b/bar.go",
		"--- a/bar.go",
		"+++ b/bar.go",
		"@@ -1 +1 @@",
		"-old line",
		"+func changed() {}",
		" context line",
	}, "\n")
	got := parseAddedGo(diff)
	if got["foo.go"] != "func added() {}\nvar x = 1\n" {
		t.Fatalf("foo.go added = %q", got["foo.go"])
	}
	if got["bar.go"] != "func changed() {}\n" {
		t.Fatalf("bar.go added = %q (removed/context lines must be excluded)", got["bar.go"])
	}
	if _, ok := got["+++ b/"]; ok {
		t.Fatalf("the +++ header must never be treated as a file or a line")
	}
}

func TestParseAddedGoIgnoresRemovalsAndContext(t *testing.T) {
	// A diff that only removes and shows context adds nothing.
	diff := strings.Join([]string{
		"+++ b/only_removals.go",
		"@@ -1,2 +0,0 @@",
		"-gone one",
		"-gone two",
		" still here",
	}, "\n")
	got := parseAddedGo(diff)
	if len(got) != 0 {
		t.Fatalf("a removal-only diff should add nothing, got %+v", got)
	}
}

// TestDupGuardLogicWarnsOnClone exercises the guard's decision core: an added block
// identical to a tracked site produces a match; a novel added block does not. This
// is the same clonescan.Query the guard runs, over the same shape of input, without
// shelling to git — proving the advisory verdict independent of the diff plumbing.
func TestDupGuardLogicWarnsOnClone(t *testing.T) {
	const block = `
func tallyItems(items []int) int {
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
	tree := map[string]string{"existing.go": "package a\n" + block}

	// An added block cloning existing.go warns (>=1 match).
	added := parseAddedGo("+++ b/new.go\n" + plusPrefix(block))
	if m := clonescan.Query(added["new.go"], tree, "new.go", 5); len(m) == 0 {
		t.Fatalf("guard should flag an added block cloning a tracked site")
	}

	// A novel added block is silent (0 matches).
	novel := "func whollyUnique() { println(\"nothing like the tree 987 zxcv\") }\n"
	added2 := parseAddedGo("+++ b/novel.go\n" + plusPrefix(novel))
	if m := clonescan.Query(added2["novel.go"], tree, "novel.go", 5); len(m) != 0 {
		t.Fatalf("guard should be silent on a novel added block, got %+v", m)
	}
}

// plusPrefix turns a source block into diff '+'-prefixed added lines.
func plusPrefix(src string) string {
	var sb strings.Builder
	for _, line := range strings.Split(strings.TrimPrefix(src, "\n"), "\n") {
		sb.WriteString("+")
		sb.WriteString(line)
		sb.WriteString("\n")
	}
	return sb.String()
}
