package corelockgate

import (
	"context"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// TestHonestMaintainerNamingTheirOwnChangedFileCorrelates is the half that MUST NOT
// break. A maintainer who names one of the files they are actually changing has to
// pass trivially — on this surface the lock has no environment escape, so a
// correlation rule that fails this case would lock every maintainer out of
// internal/adjudicator/** permanently.
func TestHonestMaintainerNamingTheirOwnChangedFileCorrelates(t *testing.T) {
	changed := []string{
		"internal/adjudicator/oot_mention.go",
		"internal/adjudicator/oot_mention_test.go",
		"internal/adjudicator/outoftree.go",
	}
	for _, claim := range []string{
		"committed:internal/adjudicator/outoftree.go",
		"path:internal/adjudicator/oot_mention.go",
		// spellings an honest maintainer plausibly types
		`committed:internal\adjudicator\outoftree.go`,
		"committed:./internal/adjudicator/outoftree.go",
		"committed::/internal/adjudicator/outoftree.go",
		"committed:internal/adjudicator/OUTOFTREE.go",
		"committed: internal/adjudicator/outoftree.go ",
		// the containing directory counts: generous by design
		"committed:internal/adjudicator",
	} {
		got := CorrelateWitness(claim, changed)
		if got.Outcome != CorrelationCorrelated {
			t.Fatalf("claim %q over its own change must correlate, got %s (%s)", claim, got.Outcome, got.Reason)
		}
	}
}

// TestWitnessNamingAnUntouchedFileIsUncorrelated is the defect itself, taken
// verbatim from the real record: commit 525a596cd2 cleared CORE_SELF_MODIFY with
// `committed:internal/adjudicator/decide.go` while touching oot_mention.go,
// oot_mention_test.go and outoftree.go — and decide.go is tracked, so the resolver
// said CONFIRMED. The same note recorded the real pathset in `tree`, which is what
// makes the mismatch computable.
func TestWitnessNamingAnUntouchedFileIsUncorrelated(t *testing.T) {
	changed := []string{
		"internal/adjudicator/oot_mention.go",
		"internal/adjudicator/oot_mention_test.go",
		"internal/adjudicator/outoftree.go",
	}
	got := CorrelateWitness("committed:internal/adjudicator/decide.go", changed)
	if got.Outcome != CorrelationUncorrelated {
		t.Fatalf("a witness naming an untouched file must be uncorrelated, got %s (%s)", got.Outcome, got.Reason)
	}
	if !strings.Contains(got.Reason, "internal/adjudicator/decide.go") {
		t.Fatalf("the reading must name the claimed path:\n%s", got.Reason)
	}
	// Any tracked path in the repository clears the resolver today; the correlation
	// is what tells those apart from the change.
	if got := CorrelateWitness("committed:README.md", changed); got.Outcome != CorrelationUncorrelated {
		t.Fatalf("committed:README.md over an adjudicator change must be uncorrelated, got %s", got.Outcome)
	}
}

// TestSelfAuthoredTempFileWitnessIsUncorrelated covers the shape found in the
// record four times over: a `path:` claim on an absolute temp file the agent had
// just written itself. It cannot be a member of a repo-relative changed set, so this
// is a measurement rather than an abstain.
func TestSelfAuthoredTempFileWitnessIsUncorrelated(t *testing.T) {
	changed := []string{"internal/adjudicator/setpolicy_benchmark_test.go"}
	for _, claim := range []string{
		`path:C:\Users\USER\AppData\Local\Temp\fak-5146-adjudicator-witness.txt`,
		"path:/tmp/fak-core-witness.txt",
	} {
		got := CorrelateWitness(claim, changed)
		if got.Outcome != CorrelationUncorrelated {
			t.Fatalf("claim %q must be uncorrelated, got %s (%s)", claim, got.Outcome, got.Reason)
		}
		if !strings.Contains(got.Reason, "outside the repository") {
			t.Fatalf("the reading should name the out-of-repo cause:\n%s", got.Reason)
		}
	}
}

// TestUnjudgeableClaimsAbstain pins abstain-over-refute, the same care the resolver
// takes: "a bad/unknown ref is not evidence of absence". A claim kind that names no
// repository path, and a claim offered against no changed set, are Indeterminate —
// never an accusation.
func TestUnjudgeableClaimsAbstain(t *testing.T) {
	changed := []string{"internal/adjudicator/decide.go"}
	for _, claim := range []string{
		"ancestor:HEAD",
		"ancestor:origin/main",
		"commit:0f1e2d3c4b5a",
		"grep:add mention-disclosure to remedies",
		"notests:HEAD",
		"exec:{}",
		"nonsense",
		"",
		":",
	} {
		if got := CorrelateWitness(claim, changed); got.Outcome != CorrelationIndeterminate {
			t.Fatalf("claim %q must abstain, got %s (%s)", claim, got.Outcome, got.Reason)
		}
	}
	// A path-shaped claim with nothing to compare against also abstains.
	if got := CorrelateWitness("committed:internal/adjudicator/decide.go", nil); got.Outcome != CorrelationIndeterminate {
		t.Fatalf("an empty changed set must abstain, got %s (%s)", got.Outcome, got.Reason)
	}
}

// TestHistoryShapedClaimsSayWhyTheyCannotCorrelate is the honest half of the
// abstain: `ancestor:HEAD` is constant-true (a commit is its own ancestor) and no
// history-shaped claim can name a change that is not yet a commit. The reading
// abstains on the OUTCOME but must still record the structural reason, because that
// is the part a reader needs in order to act on it.
func TestHistoryShapedClaimsSayWhyTheyCannotCorrelate(t *testing.T) {
	got := CorrelateWitness("ancestor:HEAD", []string{"internal/adjudicator/decide.go"})
	if got.Kind != "ancestor" {
		t.Fatalf("kind = %q, want ancestor", got.Kind)
	}
	if !strings.Contains(got.Reason, "before that change is a commit") {
		t.Fatalf("the reading must explain why history cannot name the change:\n%s", got.Reason)
	}
}

// TestCorrelationStringIsRecordable pins the compact note form, including that an
// unmeasured zero value stringifies to "" so it is OMITTED from the record rather
// than written as a false "correlated".
func TestCorrelationStringIsRecordable(t *testing.T) {
	if s := (WitnessCorrelation{}).String(); s != "" {
		t.Fatalf("an unmeasured correlation must stringify empty, got %q", s)
	}
	s := CorrelateWitness("committed:README.md", []string{"internal/adjudicator/decide.go"}).String()
	if !strings.HasPrefix(s, string(CorrelationUncorrelated)+": ") {
		t.Fatalf("recorded form should lead with the outcome, got %q", s)
	}
}

// TestSamplePathsIsBounded keeps one decision-note line from becoming a pathset
// dump when a wide commit trips the lock.
func TestSamplePathsIsBounded(t *testing.T) {
	changed := []string{"a.go", "b.go", "c.go", "d.go", "e.go", "f.go", "g.go"}
	got := CorrelateWitness("committed:internal/adjudicator/decide.go", changed)
	if !strings.Contains(got.Reason, "(+3 more)") {
		t.Fatalf("the reason should elide the tail of a wide changed set:\n%s", got.Reason)
	}
	if strings.Contains(got.Reason, "g.go") {
		t.Fatalf("the reason should not dump every changed path:\n%s", got.Reason)
	}
}

// TestObserveHookReportsWithoutChangingTheVerdict is the observe-not-enforce
// contract at the gate. A CONFIRMED-but-uncorrelated witness must STILL clear the
// lock today — shipping enforcement in the same change that first measures the
// mismatch would refuse live work mid-flight — while the mismatch is reported.
func TestObserveHookReportsWithoutChangingTheVerdict(t *testing.T) {
	var seen WitnessCorrelation
	detail, fired := CheckCoreLockHardSelf(context.Background(), CoreLockCheck{
		Resolver: fixedResolver{outcome: abi.WitnessConfirmed},
		Changed:  []string{lockedPath},
		Witness:  "committed:README.md",
		Observe:  func(c WitnessCorrelation) { seen = c },
	})
	if fired {
		t.Fatalf("observation must not change the verdict yet, got refusal %q", detail)
	}
	if seen.Outcome != CorrelationUncorrelated {
		t.Fatalf("the gate should have observed the mismatch, got %s (%s)", seen.Outcome, seen.Reason)
	}

	// ... and the honest maintainer is observed as correlated, still cleared.
	seen = WitnessCorrelation{}
	if _, fired := CheckCoreLockHardSelf(context.Background(), CoreLockCheck{
		Resolver: fixedResolver{outcome: abi.WitnessConfirmed},
		Changed:  []string{lockedPath},
		Witness:  "committed:" + lockedPath,
		Observe:  func(c WitnessCorrelation) { seen = c },
	}); fired {
		t.Fatal("a maintainer naming their own changed file must still clear the lock")
	}
	if seen.Outcome != CorrelationCorrelated {
		t.Fatalf("the honest case must read correlated, got %s (%s)", seen.Outcome, seen.Reason)
	}
}

// TestNoObserverIsSafe pins that the hook is genuinely optional: every existing
// caller passes none and must behave exactly as before.
func TestNoObserverIsSafe(t *testing.T) {
	if _, fired := CheckCoreLockHardSelf(context.Background(), CoreLockCheck{
		Resolver: fixedResolver{outcome: abi.WitnessConfirmed},
		Changed:  []string{lockedPath},
		Witness:  "committed:README.md",
	}); fired {
		t.Fatal("a caller with no observer must be unaffected")
	}
}

// TestObserveIsNotCalledWithoutAResolution pins that the observation is a reading of
// a claim the gate actually RESOLVED. A missing claim never reaches the resolver, so
// there is nothing to correlate and nothing is reported.
func TestObserveIsNotCalledWithoutAResolution(t *testing.T) {
	called := false
	if _, fired := CheckCoreLockHardSelf(context.Background(), CoreLockCheck{
		Changed: []string{lockedPath},
		Observe: func(WitnessCorrelation) { called = true },
	}); !fired {
		t.Fatal("a missing witness must still refuse")
	}
	if called {
		t.Fatal("no claim was resolved, so no correlation should have been reported")
	}
}
