package safecommit

// RefusalReasons is the closed vocabulary of commit-STATE refusals — the reasons
// fak commit reports AFTER input and work-tree validation have passed, that an
// operator resolves per the commit-clean skill. It deliberately excludes the
// input/usage errors (NO_PATHS, EMPTY_MESSAGE) and the not-in-a-work-tree
// environment error (NOT_A_REPO), which are exit-2 / setup failures rather than
// the commit-state refusals a shared-trunk author acts on.
//
// This is the single source of truth the commit-clean SKILL.md refusal
// vocabulary binds to: a doc-drift test (reasons_doc_test.go) asserts every entry
// here is documented in the skill, so a newly-added reason cannot silently go
// undocumented. Order runs roughly pre-commit gates first, then post-commit
// integrity failures.
func RefusalReasons() []string {
	return []string{
		ReasonOffTrunk,
		ReasonMergeInProgress,
		ReasonNothingStaged,
		ReasonLockBusy,
		ReasonWindowFull,
		ReasonWriterLeaseHeld,
		ReasonStaleBaseDeletion,
		ReasonStaleUntrackedPath,
		ReasonSpuriousStagedDeletion,
		ReasonCachedRemoveWorktreePresent,
		ReasonPreStagedPathOverlap,
		ReasonCoreSelfModify,
		ReasonReviewRefuted,
		ReasonPathspecRace,
		ReasonMessageRace,
		ReasonSymlinkEscape,
		ReasonHookRefused,
		ReasonPushRejected,
	}
}

// Commit exit-code classes. Nothing-landed is TWO outcomes, not one, and a caller must be
// able to tell them apart without parsing prose (#5505 W4):
//
//   - ExitLockBusy — CONTENTION: the request never got as far as a verdict, because
//     another writer holds the commit lock, the lane window is saturated, or a sync-apply
//     window holds the worktree writer lease. The answer may differ next tick, so
//     retry-with-backoff is exactly the right response.
//   - ExitRefused — a VERDICT on the merits: nothing landed either, but the refusal is a
//     property of the request or of the tree (off-trunk HEAD, a core-lock path, a
//     pre-staged overlap, a stale base). Re-running the identical command cannot change
//     the answer; fix the named cause or replan.
//   - ExitPostCommitFailure — the commit RAN and its result is bad: halt and have a human
//     review, never auto-retry into a force-push or amend.
//
// ExitLockBusy keeps the historical value 3 deliberately. Exit 3 across `fak` means
// "blocked, come back later" (`fak resume admit` defers on 3, `fak loop admit` refuses on
// 3), and the fleet's landers already retry it with backoff — so the transient class is
// the one that must keep the old number. It was the VERDICT class that was mis-labelled
// as retryable, and it is the verdict class that moves to a new code: a lander that reads
// exit 3 as "retry me" now retries only things that can actually clear, and sees an
// unfamiliar code (which it halts on) for a refusal that never will. These are the exit
// codes cmd/fak's `fak commit` reports, mirrored in the commit-clean skill's "Exit codes"
// section; reasons_doc_test.go binds the two together so they cannot drift.
const (
	ExitLockBusy          = 3 // contention — nothing landed, retry with backoff
	ExitRefused           = 4 // a verdict — nothing landed, retrying cannot change it
	ExitPostCommitFailure = 1 // commit ran, result is bad — halt for review
)

// ContentionReasons is the sub-vocabulary of RefusalReasons() that means "the request
// never reached a verdict": a lock, a window or a lease was held by someone else. These
// and only these map to ExitLockBusy. Every other refusal reason is a verdict.
func ContentionReasons() []string {
	return []string{ReasonLockBusy, ReasonWindowFull, ReasonWriterLeaseHeld}
}

// RefusalExitCode classifies a closed-vocabulary refusal reason into its process exit
// code. ok is false for a reason outside RefusalReasons() (an input/usage or environment
// error, or an unknown string) — the caller keeps its own mapping for those. Keeping this
// next to RefusalReasons() makes the classification a single source of truth: a test
// asserts every RefusalReasons() entry is classified here, so a newly-added reason cannot
// silently fall through to the halt-class exit code and wedge a loop that treats exit 1 as
// "stop, human review" when the refusal was actually a blocked pre-commit attempt.
func RefusalExitCode(reason string) (code int, ok bool) {
	switch reason {
	case ReasonLockBusy, ReasonWindowFull, ReasonWriterLeaseHeld:
		return ExitLockBusy, true
	case ReasonOffTrunk, ReasonMergeInProgress, ReasonNothingStaged,
		ReasonStaleBaseDeletion, ReasonStaleUntrackedPath,
		ReasonSpuriousStagedDeletion, ReasonCachedRemoveWorktreePresent,
		ReasonPreStagedPathOverlap, ReasonCoreSelfModify, ReasonReviewRefuted:
		return ExitRefused, true
	case ReasonPathspecRace, ReasonMessageRace, ReasonSymlinkEscape,
		ReasonHookRefused, ReasonPushRejected:
		return ExitPostCommitFailure, true
	}
	return 0, false
}
