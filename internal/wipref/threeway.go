package wipref

// threeway.go — the STALE-BASE half of restore/land (#4337). A checkpoint delta is
// `git diff <commit>^1 <commit>`: a patch whose context lines are anchored to the
// HEAD the crashed session was sitting on. The moment peers land on trunk, that base
// moves out from under the patch and a context-strict `git apply` rejects the WHOLE
// delta — so the normal case for a crashed session (its lane kept moving) reads as
// unrecoverable rather than as mergeable. `git apply --3way` reconstructs the patch's
// pre-image blobs from the object DB and performs a real three-way merge, which turns
// that reject into either a clean merge or a resolvable one with conflict markers.
//
// This file owns the DECISIONS, not the git calls: the ordered strict-then-three-way
// ladder, the widened state vocabulary, and the marker detector that decides whether a
// merged tree may be committed. The cmd shell (cmd/fak/wip.go) runs git and folds
// here, so every rule below is unit-testable without a repo — the same split the
// package doc states.
//
// THE STATE VOCABULARY is carried as plain strings, because that is what the shell
// already writes into wipLandResult.Materialized and what its JSON has always
// exposed. Three values predate this file; two are what a three-way tier makes
// expressible, and collapsing either of them back into an older value is precisely
// the misgrade #4337 is about:
//
//	applied               the context-strict apply took the delta as-is; the baseline
//	                      is unmoved, so no merge machinery ran and none of its
//	                      metadata is present.
//	present               the delta reverse-applies — it is already in the tree.
//	merged                NEW. Strict apply rejected (the base moved) but the
//	                      three-way merge resolved every hunk. Recoverable AND
//	                      committable: the delta now sits merged onto the moved base.
//	merged_with_conflicts NEW. The three-way merge ran and left conflict markers. The
//	                      delta IS recovered (the owner resolves it in place) but must
//	                      NOT be committed — this is the MERGE_CONFLICT refusal.
//	conflict              neither tier put the delta in the tree at all.
//
// Two facts about `git apply --3way`, both measured against git 2.51, shape the
// contract and neither is optional:
//
//  1. `--3way` IMPLIES `--index`. Left alone it STAGES what it merges, which would
//     break the "working tree only, never the index" invariant restore and land both
//     depend on (safecommit's prestaged-overlap guard reads the real index, and this
//     is a shared multi-session tree where staging a path can sweep a peer's file into
//     someone else's commit). The shell therefore points GIT_INDEX_FILE at a throwaway
//     index seeded from HEAD — see ThreeWayIndexSetup.
//  2. On a DIRTY working tree, `--3way` against the real index fails outright with
//     "does not match index", because the implied --index requires the file to match.
//     The throwaway index is thus not merely invariant-preserving, it is what makes
//     the three-way tier work at all in the case it exists to serve.
//
// Exit status cannot grade the result on its own: `--3way` returns 1 BOTH when it
// wrote conflict markers ("Applied patch to 'f' with conflicts.") and when it could
// not apply anything at all ("does not exist in index"). The distinguisher is the
// CONTENT of the touched files, which is why HasConflictMarkers is the load-bearing
// check rather than a convenience.

import "bytes"

// The state vocabulary, as written into wipLandResult.Materialized. Kept unexported:
// the shell already emits these as literals, and callers ask the two PREDICATES below
// rather than comparing to a constant — the predicate is the decision that matters,
// and it is the one that can stay fail-closed as the vocabulary grows.
const (
	stateApplied   = "applied"
	statePresent   = "present"
	stateMerged    = "merged"
	stateConflicts = "merged_with_conflicts"
	stateConflict  = "conflict"
)

// Committable reports whether land may turn a tree in this state into a commit.
// It is the guard behind the MERGE_CONFLICT refusal: committing a tree that holds
// conflict markers would ship them into trunk and grade a false OK, so a conflicted
// merge is refused here even though its delta IS recovered. Fail-closed — an
// unrecognized state is never committable.
func Committable(state string) bool {
	switch state {
	case stateApplied, statePresent, stateMerged:
		return true
	}
	return false
}

// InTree reports whether the delta actually reached the working tree in this state —
// true for a conflicted merge (the markers ARE the delta, resolvable in place) and
// false only where nothing was written. This is the recoverable/lost split, which is
// NOT the same cut as Committable: exactly one state, the conflicted merge, is
// recoverable but not committable, and conflating the two cuts is what makes a
// mergeable delta read as lost.
func InTree(state string) bool {
	switch state {
	case stateApplied, statePresent, stateMerged, stateConflicts:
		return true
	}
	return false
}

// ApplyRung is one rung of the strict-then-three-way ladder: the `git apply` argv tail
// to run, and the state a zero exit at that rung grades to.
type ApplyRung struct {
	// ThreeWay is whether this rung needs the throwaway index of ThreeWayIndexSetup.
	// Only a --3way rung does; the strict rung never touches the index at all.
	ThreeWay bool
	// Args is the argv AFTER "git", less the patch source, which the caller appends
	// ("-" for stdin, or a path).
	Args []string
	// OK is the state a zero exit at this rung grades to.
	OK string
}

// ApplyLadder is the ORDERED apply strategy: strict first, three-way only as a
// fallback. The ordering is load-bearing in both directions — an unmoved baseline
// must not pay for a merge or acquire merge metadata it did not need, and a moved
// baseline must not be reported as unrecoverable before the merge tier was tried.
// Returning it as data (rather than inlining the order at each call site) is what
// lets the ordering itself be asserted by a test.
func ApplyLadder() []ApplyRung {
	return []ApplyRung{
		{Args: []string{"apply", "--whitespace=nowarn"}, OK: stateApplied},
		{ThreeWay: true, Args: []string{"apply", "--3way", "--whitespace=nowarn"}, OK: stateMerged},
	}
}

// CheckLadder is ApplyLadder's read-only twin: the `git apply --check` rungs the
// forward/reverse discriminator (land's baseline test, reconcile's
// RECLAIM-vs-QUARANTINE test) walks. Reverse checks ask "is the delta already
// present", which is a question about THIS tree and never a merge, so the reverse
// ladder is strict-only — a delta that merely *merges* backwards is not present, and
// grading it present would make land skip a materialize it still owes.
func CheckLadder(reverse bool) []ApplyRung {
	if reverse {
		return []ApplyRung{{Args: []string{"apply", "--check", "-R"}, OK: statePresent}}
	}
	return []ApplyRung{
		{Args: []string{"apply", "--check"}, OK: stateApplied},
		{ThreeWay: true, Args: []string{"apply", "--check", "--3way"}, OK: stateMerged},
	}
}

// ThreeWayIndexSetup is the git invocation sequence that seeds the throwaway index a
// --3way rung runs against, to be run with GIT_INDEX_FILE pointed at a scratch path.
// read-tree populates it from HEAD; update-index --refresh restores the stat cache
// read-tree leaves empty, without which --3way rejects an untouched file as "does not
// match index". Both are reads of HEAD into a scratch file: the real index is never
// opened, so a peer's staged work cannot be disturbed.
func ThreeWayIndexSetup() [][]string {
	return [][]string{
		{"read-tree", "HEAD"},
		{"update-index", "--refresh"},
	}
}

// Conflict markers, as git writes them at the START of a line. A bare `=======` is NOT
// evidence: it is an ordinary setext/RST heading rule and appears throughout this
// repo's own docs, so requiring the OPENING and CLOSING markers together is what keeps
// the detector from quarantining innocent prose.
var (
	conflictOpen  = []byte("<<<<<<<")
	conflictClose = []byte(">>>>>>>")
)

// HasConflictMarkers reports whether data carries a git conflict region: an opening
// marker AND a closing marker, each at the start of a line. Line-anchored and
// pair-required, so a file that merely mentions a marker mid-line (a test fixture, a
// doc about merges) does not read as conflicted.
func HasConflictMarkers(data []byte) bool {
	return hasLinePrefix(data, conflictOpen) && hasLinePrefix(data, conflictClose)
}

// hasLinePrefix reports whether any line of data begins with prefix.
func hasLinePrefix(data, prefix []byte) bool {
	for rest := data; len(rest) > 0; {
		if bytes.HasPrefix(rest, prefix) {
			return true
		}
		nl := bytes.IndexByte(rest, '\n')
		if nl < 0 {
			return false
		}
		rest = rest[nl+1:]
	}
	return false
}

// ConflictedPaths returns the subset of order whose bytes carry conflict markers, in
// the caller's order so the refusal it feeds is deterministic. It takes CONTENT rather
// than paths because this package does no I/O; the shell reads the delta's file set
// after a merge rung and folds here.
func ConflictedPaths(order []string, content map[string][]byte) []string {
	out := make([]string, 0, len(order))
	for _, p := range order {
		if HasConflictMarkers(content[p]) {
			out = append(out, p)
		}
	}
	return out
}

// GradeApply folds a ladder walk into the one state that describes it.
//
// The conflicted-content test is checked FIRST and beats a zero exit status on
// purpose: content is the authority on whether a tree may be committed, and grading
// markers as anything other than a conflicted merge is the single failure that would
// let land ship them into trunk. Everything else is the ladder's own order — strict
// success is "applied", three-way success is "merged", and nothing having landed is
// "conflict".
func GradeApply(strictOK, threeWayClean bool, conflicted []string) string {
	switch {
	case len(conflicted) > 0:
		return stateConflicts
	case strictOK:
		return stateApplied
	case threeWayClean:
		return stateMerged
	default:
		return stateConflict
	}
}
