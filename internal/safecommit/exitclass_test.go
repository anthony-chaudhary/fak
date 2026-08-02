package safecommit

import "testing"

// The three exit-code classes, written as LITERALS on purpose. These tests are the wire
// contract `fak commit` publishes to its callers, so they must fail if the numbers move —
// not silently follow a renamed constant. They also compile against the pre-#5505 tree,
// which is what makes the -overlay non-vacuity proof possible: revert reasons.go to HEAD
// and TestContentionAndVerdictExitCodesAreDistinguishable fails as a test, not as a build
// error.
const (
	wantContentionExit = 3
	wantVerdictExit    = 4
	wantPostCommitExit = 1
)

// contentionWire is the "didn't get to ask" class: some other holder had the lock, the
// window or the writer lease, so the commit never reached a verdict.
var contentionWire = []string{
	ReasonLockBusy,
	ReasonWindowFull,
	ReasonWriterLeaseHeld,
}

// verdictWire is the "no" class: nothing landed AND re-running the identical command
// cannot change the answer.
var verdictWire = []string{
	ReasonOffTrunk,
	ReasonMergeInProgress,
	ReasonNothingStaged,
	ReasonStaleBaseDeletion,
	ReasonStaleUntrackedPath,
	ReasonSpuriousStagedDeletion,
	ReasonCachedRemoveWorktreePresent,
	ReasonPreStagedPathOverlap,
	ReasonCoreSelfModify,
	ReasonReviewRefuted,
}

// TestContentionAndVerdictExitCodesAreDistinguishable is the #5505 W4 witness: a caller
// must be able to tell "the lock was busy, retrying is exactly right" from "you were
// refused on the merits, retrying can never help" from the exit code alone. Before the
// split both classes returned 3, so a lander burned its whole retry budget on refusals
// that could never clear.
func TestContentionAndVerdictExitCodesAreDistinguishable(t *testing.T) {
	for _, busy := range contentionWire {
		busyCode, ok := RefusalExitCode(busy)
		if !ok {
			t.Fatalf("contention reason %q is unclassified", busy)
		}
		for _, verdict := range verdictWire {
			verdictCode, ok := RefusalExitCode(verdict)
			if !ok {
				t.Fatalf("verdict reason %q is unclassified", verdict)
			}
			if busyCode == verdictCode {
				t.Errorf("contention %q and verdict %q both exit %d — a caller cannot tell "+
					"transient contention from a hard refusal, so it either retries work "+
					"that never can land or parks work that could have", busy, verdict, busyCode)
			}
		}
	}
}

// TestContentionKeepsExitCodeThree pins the OLD code to its OLD meaning. Exit 3 was, and
// stays, the retryable "blocked, come back later" answer — the split moved the verdicts
// out to a new code, it did not repurpose 3 underneath the loops that already retry it.
func TestContentionKeepsExitCodeThree(t *testing.T) {
	for _, reason := range contentionWire {
		code, ok := RefusalExitCode(reason)
		if !ok || code != wantContentionExit {
			t.Errorf("RefusalExitCode(%q) = (%d, %v), want (%d, true) — exit 3 must keep "+
				"meaning retryable contention or every existing retry loop breaks",
				reason, code, ok, wantContentionExit)
		}
	}
}

// TestVerdictsExitFour pins the new code for the class that was mis-labelled retryable.
func TestVerdictsExitFour(t *testing.T) {
	for _, reason := range verdictWire {
		code, ok := RefusalExitCode(reason)
		if !ok || code != wantVerdictExit {
			t.Errorf("RefusalExitCode(%q) = (%d, %v), want (%d, true) — a refusal on the "+
				"merits must not share the retryable code", reason, code, ok, wantVerdictExit)
		}
	}
}

// TestPostCommitFailuresKeepExitCodeOne pins the third class, unchanged by the split: the
// commit RAN and its result is bad, so a caller halts for review instead of retrying.
func TestPostCommitFailuresKeepExitCodeOne(t *testing.T) {
	for _, reason := range []string{
		ReasonPathspecRace, ReasonMessageRace, ReasonSymlinkEscape,
		ReasonHookRefused, ReasonPushRejected,
	} {
		code, ok := RefusalExitCode(reason)
		if !ok || code != wantPostCommitExit {
			t.Errorf("RefusalExitCode(%q) = (%d, %v), want (%d, true)",
				reason, code, ok, wantPostCommitExit)
		}
	}
}

// TestEveryRefusalReasonLandsInExactlyOneWireClass closes the vocabulary against the wire:
// every reason fak commit can report is classified, and it is classified into one of the
// three published codes. A newly-added reason that forgets its class fails here rather
// than reaching a caller as an unclassified 0.
func TestEveryRefusalReasonLandsInExactlyOneWireClass(t *testing.T) {
	inContention := map[string]bool{}
	for _, r := range contentionWire {
		inContention[r] = true
	}
	for _, reason := range RefusalReasons() {
		code, ok := RefusalExitCode(reason)
		if !ok {
			t.Errorf("refusal reason %q is not classified by RefusalExitCode", reason)
			continue
		}
		switch code {
		case wantContentionExit:
			if !inContention[reason] {
				t.Errorf("refusal reason %q exits %d (the retryable contention code) but is "+
					"not a lock/window/lease holder — a caller will retry it forever",
					reason, wantContentionExit)
			}
		case wantVerdictExit, wantPostCommitExit:
			if inContention[reason] {
				t.Errorf("contention reason %q exits %d, want the retryable %d",
					reason, code, wantContentionExit)
			}
		default:
			t.Errorf("refusal reason %q exits %d, want one of %d/%d/%d",
				reason, code, wantContentionExit, wantVerdictExit, wantPostCommitExit)
		}
	}
}
