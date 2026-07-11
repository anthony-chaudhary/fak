package clonescan

import (
	"fmt"
	"reflect"
	"sort"
	"strings"
	"testing"
)

// legacyQuery is a byte-for-byte copy of the pre-#4326 per-call Query algorithm:
// it re-tokenizes the WHOLE tree on every call, collects hit spans in file-position
// order, folds them to (min start, max end, count), then sorts (Windows desc, File
// asc) and caps. It exists ONLY here, as the differential oracle the new
// BuildTreeIndex + TreeIndex.Query path must match exactly — same set, same fields,
// same order. If this and the index path ever disagree, the refactor changed
// behavior, which #4326 forbids.
func legacyQuery(candidate string, tree map[string]string, selfPath string, maxResults int) []Match {
	want := CandidateKeys(candidate)
	if len(want) == 0 {
		return nil
	}
	var matches []Match
	files := make([]string, 0, len(tree))
	for rel := range tree {
		if rel == selfPath {
			continue
		}
		files = append(files, rel)
	}
	sort.Strings(files)
	for _, rel := range files {
		keys, spans := qualifyingWindows(goTokens(tree[rel], false))
		var hitSpans []span
		for i, k := range keys {
			if want[k] {
				hitSpans = append(hitSpans, spans[i])
			}
		}
		if len(hitSpans) == 0 {
			continue
		}
		lo, hi := hitSpans[0][0], hitSpans[0][1]
		for _, s := range hitSpans[1:] {
			if s[0] < lo {
				lo = s[0]
			}
			if s[1] > hi {
				hi = s[1]
			}
		}
		matches = append(matches, Match{File: rel, StartLine: lo, EndLine: hi, Windows: len(hitSpans)})
	}
	sort.SliceStable(matches, func(i, j int) bool {
		if matches[i].Windows != matches[j].Windows {
			return matches[i].Windows > matches[j].Windows
		}
		return matches[i].File < matches[j].File
	})
	if maxResults > 0 && len(matches) > maxResults {
		matches = matches[:maxResults]
	}
	return matches
}

// diffBlock is a real multi-line logic block (>34 normalized tokens, control flow +
// computation) parameterized by a name so distinct copies get distinct identifier
// keys — the same shape the drift fixture uses.
const diffBlock = `
func %s(items []int) int {
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

func namedBlock(name string) string { return strings.Replace(diffBlock, "%s", name, 1) }

// TestTreeIndexQueryMatchesLegacy is the #4326 differential-equivalence lock:
// BuildTreeIndex + TreeIndex.Query must return Match slices identical (File,
// StartLine, EndLine, Windows AND order) to the legacy per-call Query, over a
// MULTI-file corpus that exercises the three cases the acceptance names:
//
//	(a) a window key repeated at several positions within ONE file (triple.go holds
//	    the same block three times) — locks the len(hitSpans)/Windows aggregation.
//	(b) MULTIPLE candidates queried against ONE shared prebuilt index.
//	(c) selfPath applied at QUERY time, not baked into the index — the same index
//	    answers queries with different selfPaths.
func TestTreeIndexQueryMatchesLegacy(t *testing.T) {
	proc := namedBlock("process")   // candidate: the "process" block
	comp := namedBlock("compute")   // candidate: a distinct block (different idents)
	novel := namedBlock("brandnew") // candidate: matches nothing in the corpus

	// (a) triple.go repeats the process block three times → the process window keys
	// occur at several positions in one file, so it must out-window a single copy.
	tripled := "package c\n" + namedBlock("process") + "\n" + namedBlock("process") + "\n" + namedBlock("process")

	tree := map[string]string{
		"alpha.go":   "package a\n" + proc,
		"beta.go":    "package b\n" + proc,
		"triple.go":  tripled,
		"compute.go": "package d\n" + comp,
		"mixed.go":   "package e\n" + proc + "\n" + comp,
		// A file with no qualifying window — must never appear in results.
		"trivial.go": "package f\nfunc g() {}\n",
	}

	// (b) ONE shared index, reused for every candidate/selfPath/cap combination.
	index := BuildTreeIndex(tree)

	candidates := map[string]string{
		"process": proc,
		"compute": comp,
		"novel":   novel,
		"empty":   "",              // no keys → nil in both paths
		"tiny":    "func h() {}\n", // sub-window → nil in both paths
	}
	// (c) selfPath varied at query time against the single prebuilt index.
	selfPaths := []string{"", "beta.go", "triple.go", "mixed.go"}
	caps := []int{0, 1, 2, 100}

	for cname, cand := range candidates {
		for _, self := range selfPaths {
			for _, maxN := range caps {
				want := legacyQuery(cand, tree, self, maxN)
				got := index.Query(CandidateKeys(cand), self, maxN)
				if !reflect.DeepEqual(got, want) {
					t.Fatalf("candidate=%s selfPath=%q cap=%d: index path diverged from legacy\n got:  %+v\n want: %+v",
						cname, self, maxN, got, want)
				}
			}
		}
	}

	// Explicitly lock case (a): triple.go carries more process-windows than a single
	// copy (beta.go), so it ranks first with a strictly larger Windows count.
	res := index.Query(CandidateKeys(proc), "", 0)
	var triple, single int
	for _, m := range res {
		switch m.File {
		case "triple.go":
			triple = m.Windows
		case "beta.go":
			single = m.Windows
		}
	}
	if single == 0 {
		t.Fatalf("expected beta.go to match the process block, got results %+v", res)
	}
	if triple <= single {
		t.Fatalf("repeated-position aggregation not counted: triple.go windows=%d must exceed single copy beta.go windows=%d", triple, single)
	}
	if res[0].File != "triple.go" {
		t.Fatalf("file with three copies should rank first, got %q", res[0].File)
	}
}

// TestTreeIndexSelfPathNotBaked proves selfPath is a query-time argument, not baked
// into the index: the SAME prebuilt index excludes a path on one query and includes
// it on the next.
func TestTreeIndexSelfPathNotBaked(t *testing.T) {
	proc := namedBlock("process")
	tree := map[string]string{
		"one.go": "package a\n" + proc,
		"two.go": "package b\n" + proc,
	}
	index := BuildTreeIndex(tree)

	withSelf := index.Query(CandidateKeys(proc), "one.go", 0)
	for _, m := range withSelf {
		if m.File == "one.go" {
			t.Fatalf("selfPath one.go must be excluded at query time, got %+v", withSelf)
		}
	}
	withoutSelf := index.Query(CandidateKeys(proc), "", 0)
	sawOne := false
	for _, m := range withoutSelf {
		if m.File == "one.go" {
			sawOne = true
		}
	}
	if !sawOne {
		t.Fatalf("same index without selfPath must include one.go, got %+v", withoutSelf)
	}
}

// syntheticTree builds n distinct tokenizable files. Each holds a logic block whose
// threshold constant varies with the file index, so the files have overlapping but
// not identical window keys — a realistic tree for the scaling benchmark.
func syntheticTree(n int) map[string]string {
	tree := make(map[string]string, n)
	for i := 0; i < n; i++ {
		tree[fmt.Sprintf("pkg/file%03d.go", i)] = fmt.Sprintf(`package p
func f%d(items []int) int {
	total := 0
	for i := 0; i < len(items); i++ {
		if items[i] > %d {
			total += items[i] * 2
		} else {
			total -= items[i]
		}
	}
	return total
}
`, i, i%7)
	}
	return tree
}

// BenchmarkGuardTokenizeOncePerTree demonstrates the #4326 win directly: the
// whole-tree guard path tokenizes each tree file ONCE regardless of how many added
// files are checked against it, so the indexed path's cost is ~flat in the added
// count while the legacy per-file path re-tokenizes the whole tree per added file
// (cost grows linearly with the added count). Run:
//
//	go test ./internal/clonescan/ -run x -bench BenchmarkGuardTokenizeOncePerTree -benchmem
//
// legacy_perfile ns/op scales with added=K (K full-tree tokenizations); indexed_once
// ns/op is dominated by the single BuildTreeIndex and barely moves with K.
func BenchmarkGuardTokenizeOncePerTree(b *testing.B) {
	tree := syntheticTree(200)
	cands := make([]string, 0, 64)
	for i := 0; i < 64; i++ {
		cands = append(cands, fmt.Sprintf(`func c%d(items []int) int {
	total := 0
	for i := 0; i < len(items); i++ {
		if items[i] > %d {
			total += items[i] * 2
		} else {
			total -= items[i]
		}
	}
	return total
}
`, i, i%7))
	}

	for _, k := range []int{1, 8, 64} {
		sub := cands[:k]
		b.Run(fmt.Sprintf("legacy_perfile_added=%d", k), func(b *testing.B) {
			for n := 0; n < b.N; n++ {
				for j, c := range sub {
					_ = legacyQuery(c, tree, fmt.Sprintf("added%d.go", j), 5)
				}
			}
		})
		b.Run(fmt.Sprintf("indexed_once_added=%d", k), func(b *testing.B) {
			for n := 0; n < b.N; n++ {
				index := BuildTreeIndex(tree) // tokenize the tree exactly once
				for j, c := range sub {
					_ = index.Query(CandidateKeys(c), fmt.Sprintf("added%d.go", j), 5)
				}
			}
		})
	}
}
