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
