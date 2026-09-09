package issuepolicy

import "testing"

func TestBornMergedContract(t *testing.T) {
	// 1. Fully conformant candidate produces no flags and is dispatchable
	base := completeCandidate()
	bm := bornMerged(base)
	if !bm.PathsBounded || !bm.ClosureBindingValid || !bm.AtomicUnitValid || len(bm.Flags) != 0 {
		t.Fatalf("conformant bornMerged = %+v, want all valid with 0 flags", bm)
	}

	review := ReviewCandidate(base, Options{StrictBornMerged: true})
	if !review.OK || review.Dispatchability != Dispatchable || len(review.BornMerged.Flags) != 0 {
		t.Fatalf("conformant review = %+v, want dispatchable with 0 flags", review)
	}

	// 2. Path violations (missing, unbounded) produce expected flags
	t.Run("PathViolations", func(t *testing.T) {
		// Missing paths
		cMissing := completeCandidate()
		cMissing.Paths = nil
		bmMissing := bornMerged(cMissing)
		if bmMissing.PathsBounded || !containsString(bmMissing.Flags, BornMergedPathsMissing) {
			t.Fatalf("missing paths: expected %s, got %+v", BornMergedPathsMissing, bmMissing)
		}

		// Unbounded path with bare wildcard
		cWildcard := completeCandidate()
		cWildcard.Paths = []string{"internal/issuepolicy/*"}
		bmWildcard := bornMerged(cWildcard)
		if bmWildcard.PathsBounded || !containsString(bmWildcard.Flags, BornMergedPathsUnbounded) {
			t.Fatalf("wildcard path: expected %s, got %+v", BornMergedPathsUnbounded, bmWildcard)
		}

		// Unbounded path with ellipsis
		cEllipsis := completeCandidate()
		cEllipsis.Paths = []string{"internal/issuepolicy/..."}
		bmEllipsis := bornMerged(cEllipsis)
		if bmEllipsis.PathsBounded || !containsString(bmEllipsis.Flags, BornMergedPathsUnbounded) {
			t.Fatalf("ellipsis path: expected %s, got %+v", BornMergedPathsUnbounded, bmEllipsis)
		}
	})

	// 3. Closure binding / trailer violations produce expected flags
	t.Run("ClosureBindingViolations", func(t *testing.T) {
		// Missing closure binding
		cMissingClosure := completeCandidate()
		cMissingClosure.ClosureBinding = ""
		bmMissingClosure := bornMerged(cMissingClosure)
		if bmMissingClosure.ClosureBindingValid || !containsString(bmMissingClosure.Flags, BornMergedClosureBindingMissing) {
			t.Fatalf("missing closure: expected %s, got %+v", BornMergedClosureBindingMissing, bmMissingClosure)
		}

		// Closure binding without trailer pattern
		cMissingTrailer := completeCandidate()
		cMissingTrailer.ClosureBinding = "Commit resolves issue #1234 on main branch."
		bmMissingTrailer := bornMerged(cMissingTrailer)
		if bmMissingTrailer.ClosureBindingValid || !containsString(bmMissingTrailer.Flags, BornMergedTrailerMissing) {
			t.Fatalf("missing trailer: expected %s, got %+v", BornMergedTrailerMissing, bmMissingTrailer)
		}

		// Conventional trailers should validate
		cPrivateTrailer := completeCandidate()
		cPrivateTrailer.ClosureBinding = "Commit resolves #1234 (fak-private dispatch)"
		bmPrivateTrailer := bornMerged(cPrivateTrailer)
		if !bmPrivateTrailer.ClosureBindingValid || containsString(bmPrivateTrailer.Flags, BornMergedTrailerMissing) {
			t.Fatalf("private trailer: expected valid closure, got %+v", bmPrivateTrailer)
		}
	})

	// 4. Atomic unit violations produce expected flags
	t.Run("AtomicUnitViolations", func(t *testing.T) {
		// Oversized scale S2
		cScaleS2 := completeCandidate()
		cScaleS2.Scale = "S2"
		bmScaleS2 := bornMerged(cScaleS2)
		if bmScaleS2.AtomicUnitValid || !containsString(bmScaleS2.Flags, BornMergedAtomicUnitInvalid) {
			t.Fatalf("scale S2: expected %s, got %+v", BornMergedAtomicUnitInvalid, bmScaleS2)
		}

		// Oversized scale S3
		cScaleS3 := completeCandidate()
		cScaleS3.Scale = "S3"
		bmScaleS3 := bornMerged(cScaleS3)
		if bmScaleS3.AtomicUnitValid || !containsString(bmScaleS3.Flags, BornMergedAtomicUnitInvalid) {
			t.Fatalf("scale S3: expected %s, got %+v", BornMergedAtomicUnitInvalid, bmScaleS3)
		}

		// WorkUnit is "epic"
		cUnitEpic := completeCandidate()
		cUnitEpic.WorkUnit = "epic"
		bmUnitEpic := bornMerged(cUnitEpic)
		if bmUnitEpic.AtomicUnitValid || !containsString(bmUnitEpic.Flags, BornMergedAtomicUnitInvalid) {
			t.Fatalf("work unit epic: expected %s, got %+v", BornMergedAtomicUnitInvalid, bmUnitEpic)
		}

		// ExpectedSteps > 8
		cOversizedSteps := completeCandidate()
		cOversizedSteps.ExpectedSteps = 9
		bmOversizedSteps := bornMerged(cOversizedSteps)
		if bmOversizedSteps.AtomicUnitValid || !containsString(bmOversizedSteps.Flags, BornMergedAtomicUnitInvalid) {
			t.Fatalf("steps 9: expected %s, got %+v", BornMergedAtomicUnitInvalid, bmOversizedSteps)
		}

		// Valid scale declarations S0 and S1
		cScaleS0 := completeCandidate()
		cScaleS0.Scale = "S0"
		bmScaleS0 := bornMerged(cScaleS0)
		if !bmScaleS0.AtomicUnitValid || containsString(bmScaleS0.Flags, BornMergedAtomicUnitInvalid) {
			t.Fatalf("scale S0: expected valid atomic unit, got %+v", bmScaleS0)
		}

		cScaleS1 := completeCandidate()
		cScaleS1.Scale = "S1"
		bmScaleS1 := bornMerged(cScaleS1)
		if !bmScaleS1.AtomicUnitValid || containsString(bmScaleS1.Flags, BornMergedAtomicUnitInvalid) {
			t.Fatalf("scale S1: expected valid atomic unit, got %+v", bmScaleS1)
		}
	})

	// 5. Advisory mode (StrictBornMerged: false) attaches readout but preserves dispatchability
	t.Run("AdvisoryMode", func(t *testing.T) {
		cFlagged := completeCandidate()
		cFlagged.Paths = []string{"internal/issuepolicy/*"}
		cFlagged.ClosureBinding = "Commit without trailer"

		advReview := ReviewCandidate(cFlagged, Options{StrictBornMerged: false})
		if !advReview.OK || advReview.Dispatchability != Dispatchable {
			t.Fatalf("advisory mode: want Dispatchable OK, got dispatchability=%q OK=%v reasons=%v",
				advReview.Dispatchability, advReview.OK, advReview.Reasons)
		}
		if len(advReview.BornMerged.Flags) == 0 {
			t.Fatalf("advisory mode: want BornMerged flags attached, got none")
		}
		if containsString(advReview.Reasons, ReasonNotBornMerged) {
			t.Fatalf("advisory mode: ReasonNotBornMerged must NOT be in reasons: %v", advReview.Reasons)
		}
	})

	// 6. Strict mode (StrictBornMerged: true) holds candidate with ISSUE_NOT_BORN_MERGED and triage_only verdict
	t.Run("StrictMode", func(t *testing.T) {
		cFlagged := completeCandidate()
		cFlagged.Paths = []string{"internal/issuepolicy/*"}

		strictReview := ReviewCandidate(cFlagged, Options{StrictBornMerged: true})
		if strictReview.OK || strictReview.Dispatchability != TriageOnly {
			t.Fatalf("strict mode: want TriageOnly and !OK, got dispatchability=%q OK=%v",
				strictReview.Dispatchability, strictReview.OK)
		}
		if !containsString(strictReview.Reasons, ReasonNotBornMerged) {
			t.Fatalf("strict mode: want %s in reasons, got %v", ReasonNotBornMerged, strictReview.Reasons)
		}
	})
}
