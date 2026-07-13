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

// Commit exit-code classes. A PRE-commit refusal means nothing landed: the caller
// (or a dispatch loop) may safely retry or replan. A POST-commit failure means the
// commit ran but its result is bad: halt and have a human review — never auto-retry
// into a force-push or amend. These are the exit codes cmd/fak's `fak commit` reports
// (mirrored in the commit-clean skill's "Exit codes" section) so a loop can branch on
// outcome without parsing prose.
const (
	ExitPreCommitRefusal  = 3 // nothing landed — safe to retry/replan
	ExitPostCommitFailure = 1 // commit ran, result is bad — halt for review
)

// RefusalExitCode classifies a closed-vocabulary refusal reason into its process exit
// code. ok is false for a reason outside RefusalReasons() (an input/usage or environment
// error, or an unknown string) — the caller keeps its own mapping for those. Keeping this
// next to RefusalReasons() makes the classification a single source of truth: a test
// asserts every RefusalReasons() entry is classified here, so a newly-added reason cannot
// silently fall through to the halt-class exit code and wedge a loop that treats exit 1 as
// "stop, human review" when the refusal was actually a retryable pre-commit block.
func RefusalExitCode(reason string) (code int, ok bool) {
	switch reason {
	case ReasonOffTrunk, ReasonMergeInProgress, ReasonNothingStaged,
		ReasonLockBusy, ReasonWindowFull, ReasonWriterLeaseHeld, ReasonStaleBaseDeletion,
		ReasonSpuriousStagedDeletion, ReasonCachedRemoveWorktreePresent,
		ReasonPreStagedPathOverlap, ReasonCoreSelfModify, ReasonReviewRefuted:
		return ExitPreCommitRefusal, true
	case ReasonPathspecRace, ReasonMessageRace, ReasonSymlinkEscape,
		ReasonHookRefused, ReasonPushRejected:
		return ExitPostCommitFailure, true
	}
	return 0, false
}
