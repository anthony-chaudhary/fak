package clonescan

import "testing"

// A candidate that shares NO qualifying window with sampleBlock: different control shape,
// different identifiers. Long enough to carry windows of its own, so it contributes to
// `total` while contributing nothing to `matched`.
const novelBlock = `
func classify(name string, limit int) string {
	seen := map[string]int{}
	for _, r := range name {
		seen[string(r)]++
		if seen[string(r)] > limit {
			return "over"
		}
	}
	if len(seen) == 0 {
		return "empty"
	}
	return "under"
}
`

// TestCoverageFullPasteMatchesEveryWindow pins the headline claim: a block pasted verbatim
// from the tree has every one of its own windows already present, so matched == total. This
// is the ratio a gate grades on, and a whole-function paste must reach the top of it.
func TestCoverageFullPasteMatchesEveryWindow(t *testing.T) {
	tree := map[string]string{
		"existing.go":  "package a\n" + sampleBlock,
		"unrelated.go": "package a\nfunc hello() string { return \"world\" }\n",
	}
	matched, total := Coverage(sampleBlock, tree, "")
	if total == 0 {
		t.Fatal("total = 0 for a block that exceeds the window size; fixture is too small")
	}
	if matched != total {
		t.Errorf("matched/total = %d/%d, want every window matched for a verbatim paste", matched, total)
	}
}

// TestCoverageExcludesSelfPath pins the exclusion the doc promises, and pins it on the RIGHT
// side of the fraction: naming the candidate's own path drops `matched` to zero while `total`
// is unchanged. A block that only exists at its own path is 0% copied, not ungradeable.
func TestCoverageExcludesSelfPath(t *testing.T) {
	tree := map[string]string{"mine.go": "package a\n" + sampleBlock}

	_, totalAll := Coverage(sampleBlock, tree, "")
	matched, total := Coverage(sampleBlock, tree, "mine.go")
	if matched != 0 {
		t.Errorf("matched = %d with self excluded, want 0 (a block is not its own copy)", matched)
	}
	if total != totalAll || total == 0 {
		t.Errorf("total = %d with self excluded, want the unchanged non-zero %d — exclusion must not shrink the denominator", total, totalAll)
	}
}

// TestCoverageTrivialCandidateReturnsNoVerdict pins the contract callers depend on to avoid
// dividing by zero: a candidate with no qualifying window returns (0, 0), which means NO
// VERDICT — deliberately indistinguishable in value from, but different in meaning to, a
// zero-coverage block. dupSeverity reads total == 0 as ungraded for exactly this reason.
func TestCoverageTrivialCandidateReturnsNoVerdict(t *testing.T) {
	tree := map[string]string{"existing.go": "package a\n" + sampleBlock}
	matched, total := Coverage("func f() {}\n", tree, "")
	if matched != 0 || total != 0 {
		t.Errorf("matched/total = %d/%d for a sub-window candidate, want 0/0 (no verdict)", matched, total)
	}
}

// TestCoveragePartialOverlapGradesBetween is the discrimination the whole grade rests on: a
// block that is half paste and half new must land strictly between "no copy" and "all copy".
// If this collapsed to matched == total, a large honest addition carrying one copied idiom
// would grade as a total paste — the exact false-escalation Coverage exists to prevent.
func TestCoveragePartialOverlapGradesBetween(t *testing.T) {
	tree := map[string]string{"existing.go": "package a\n" + sampleBlock}
	matched, total := Coverage(sampleBlock+novelBlock, tree, "")
	if total == 0 {
		t.Fatal("total = 0; fixture is too small to grade")
	}
	if matched == 0 {
		t.Fatal("matched = 0, but half the candidate is a verbatim copy of existing.go")
	}
	if matched >= total {
		t.Errorf("matched/total = %d/%d, want matched < total — the novel half must not count as copied", matched, total)
	}
}

// TestCoverageWrapperMatchesMethod pins the Query/TreeIndex.Query pairing the doc claims: the
// one-shot function and the build-once method must agree, so a caller grading many candidates
// against one index never gets a different number than the convenience wrapper.
func TestCoverageWrapperMatchesMethod(t *testing.T) {
	tree := map[string]string{"existing.go": "package a\n" + sampleBlock}
	wantMatched, wantTotal := Coverage(sampleBlock, tree, "")
	gotMatched, gotTotal := BuildTreeIndex(tree).Coverage(CandidateKeys(sampleBlock), "")
	if gotMatched != wantMatched || gotTotal != wantTotal {
		t.Errorf("method = %d/%d, wrapper = %d/%d; the two must agree", gotMatched, gotTotal, wantMatched, wantTotal)
	}
}
