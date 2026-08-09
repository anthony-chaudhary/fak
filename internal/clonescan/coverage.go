package clonescan

// coverage.go — the MAGNITUDE half of the clone query (#4328).
//
// Query answers "which tracked sites does this block clone", and Match.Windows ranks
// them by how many window hits each SIBLING carries. That number cannot grade the
// candidate: it counts occurrences on the other side of the comparison, and it grows
// with sheer size, so a large legitimate addition scores high purely for being large.
//
// Coverage asks the candidate-side question instead — "how much of what I just added
// is a copy" — as a FRACTION: distinct candidate windows that exist somewhere in the
// tree, over the candidate's own distinct qualifying windows. A whole-function paste
// approaches 1.0 no matter how big it is; a one-idiom overlap inside a large new file
// stays near 0. That is the ratio a commit-boundary gate can grade severity on without
// false-escalating big honest additions.

// Coverage reports how much of a candidate block already exists in the tree:
// `matched` distinct candidate window keys found in at least one tracked file, out of
// `total` distinct qualifying windows the candidate carries. matched <= total always,
// and total == 0 means the candidate has no qualifying window at all (too small, or
// pure data/declarations) — an ungradeable block, NOT a zero-coverage one, so callers
// must treat total == 0 as "no verdict" rather than dividing by it.
//
// selfPath is excluded exactly as in Query, so a block already committed at some path
// does not count itself as its own copy. Pass "" when the candidate is unwritten.
func (idx *TreeIndex) Coverage(want map[string]bool, selfPath string) (matched, total int) {
	total = len(want)
	if total == 0 {
		return 0, 0
	}
	for k := range want {
		for _, rel := range idx.files {
			if rel == selfPath || len(idx.byFile[rel][k]) == 0 {
				continue
			}
			matched++
			break // one tracked site is enough: coverage counts candidate windows, not hits
		}
	}
	return matched, total
}

// Coverage is the single-candidate wrapper over BuildTreeIndex + TreeIndex.Coverage,
// mirroring the Query/TreeIndex.Query pair. A caller grading many candidates against
// the same tree builds the index once and calls the method.
func Coverage(candidate string, tree map[string]string, selfPath string) (matched, total int) {
	return BuildTreeIndex(tree).Coverage(CandidateKeys(candidate), selfPath)
}
