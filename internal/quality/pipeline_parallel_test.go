package quality

import (
	"strings"
	"testing"
)

// ppTestSeed and ppTestSteps pin the deterministic decode the pipeline-parity
// tests run: long enough that the defect token (3) and the resume boundary (5)
// both sit mid-sequence with a passing prefix and a checkable tail.
const (
	ppTestSeed     = 42
	ppTestSteps    = 8
	ppTestBoundary = 5
)

// TestPPParityFaithfulPipelinePasses is the parity contract on the happy path:
// a faithful two-stage pipeline equals the single-stage reference token for
// token, for EVERY resume boundary k in [0, steps] — including k == steps,
// where no resume triggers at all (the plain staged decode). A correct stage
// split and a correct boundary resume are invisible in the output.
func TestPPParityFaithfulPipelinePasses(t *testing.T) {
	c := PPParityCase(ppTestSeed, ppTestSteps)
	for k := 0; k <= ppTestSteps; k++ {
		res, err := RunCase(c, ReferenceRunner{}, PPPipelineEngine(k, ""), oraclesFor(t, c))
		if err != nil {
			t.Fatalf("RunCase (resume at %d): %v", k, err)
		}
		if !res.Pass {
			t.Fatalf("faithful pipeline resumed at boundary %d should equal single-stage; got %s", k, Explain(res))
		}
		if res.FailureBundle != nil {
			t.Fatalf("clean pipeline run (resume at %d) must not carry a failure bundle: %+v", k, res.FailureBundle)
		}
	}
}

// TestPPParityDropHandoffFailsAtAffectedToken is the dropped-handoff witness:
// a stage boundary that loses token ppDefectToken's activation leaves the
// prefix intact, and the oracle pins the first divergence to exactly that
// token, with the reference and engine tokens reported.
func TestPPParityDropHandoffFailsAtAffectedToken(t *testing.T) {
	c := PPParityCase(ppTestSeed, ppTestSteps)
	res, err := RunCase(c, ReferenceRunner{}, PPPipelineEngine(ppTestBoundary, "drop-handoff"), oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if res.Pass {
		t.Fatalf("drop-handoff engine must not pass; got %s", Explain(res))
	}
	fb := res.FailureBundle
	if fb == nil {
		t.Fatal("failing pipeline run must carry a failure bundle")
	}
	if fb.FailingOracle != "pp-parity" {
		t.Errorf("first failing oracle = %q, want pp-parity", fb.FailingOracle)
	}
	d := fb.FirstDivergence
	if d == nil || d.Index != ppDefectToken {
		t.Fatalf("expected first divergence at the dropped handoff's token %d, got %+v", ppDefectToken, d)
	}
	if want := c.Reference.Tokens[ppDefectToken]; d.Reference != want {
		t.Errorf("divergence reference token = %q, want %q", d.Reference, want)
	}
	if d.Engine == d.Reference {
		t.Errorf("divergence engine token %q must differ from the reference", d.Engine)
	}
	if !strings.Contains(fb.Detail, "stage boundary") {
		t.Errorf("detail should name the stage boundary; got %q", fb.Detail)
	}
	// The prefix BEFORE the defect must match: the failure localizes to the
	// handoff's token, not to the staged run as a whole.
	for i := 0; i < ppDefectToken; i++ {
		if fb.Engine.Tokens[i] != c.Reference.Tokens[i] {
			t.Fatalf("prefix token %d differs before the injected defect: reference %q, engine %q",
				i, c.Reference.Tokens[i], fb.Engine.Tokens[i])
		}
	}
}

// TestPPParityDupHandoffFailsAtAffectedToken is the duplicated-handoff
// witness: a boundary that delivers token ppDefectToken's activation to stage
// two twice (stage two runs its layers twice) diverges at exactly that token.
// The distinct per-layer constants guarantee the double application cannot
// cancel back to the reference computation.
func TestPPParityDupHandoffFailsAtAffectedToken(t *testing.T) {
	c := PPParityCase(ppTestSeed, ppTestSteps)
	res, err := RunCase(c, ReferenceRunner{}, PPPipelineEngine(ppTestBoundary, "dup-handoff"), oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if res.Pass {
		t.Fatalf("dup-handoff engine must not pass; got %s", Explain(res))
	}
	fb := res.FailureBundle
	if fb == nil {
		t.Fatal("failing pipeline run must carry a failure bundle")
	}
	d := fb.FirstDivergence
	if d == nil || d.Index != ppDefectToken {
		t.Fatalf("expected first divergence at the duplicated handoff's token %d, got %+v", ppDefectToken, d)
	}
	if want := c.Reference.Tokens[ppDefectToken]; d.Reference != want {
		t.Errorf("divergence reference token = %q, want %q", d.Reference, want)
	}
	if d.Engine == d.Reference {
		t.Errorf("divergence engine token %q must differ from the reference", d.Engine)
	}
}

// TestPPParityResumeStateLossFailsAtBoundary is the boundary-resume witness:
// a resume whose snapshot restore loses stage two's carried state decodes
// faithfully up to the boundary and diverges at exactly the resumed token —
// the first token stage two completes from its zeroed state.
func TestPPParityResumeStateLossFailsAtBoundary(t *testing.T) {
	c := PPParityCase(ppTestSeed, ppTestSteps)
	res, err := RunCase(c, ReferenceRunner{}, PPPipelineEngine(ppTestBoundary, "resume-state-loss"), oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if res.Pass {
		t.Fatalf("resume-state-loss engine must not pass; got %s", Explain(res))
	}
	fb := res.FailureBundle
	if fb == nil {
		t.Fatal("failing pipeline run must carry a failure bundle")
	}
	d := fb.FirstDivergence
	if d == nil || d.Index != ppTestBoundary {
		t.Fatalf("expected first divergence at the resumed token %d, got %+v", ppTestBoundary, d)
	}
	// Everything BEFORE the resume boundary round-tripped: the loss localizes
	// to the boundary, proving the snapshot itself carried the earlier state.
	for i := 0; i < ppTestBoundary; i++ {
		if fb.Engine.Tokens[i] != c.Reference.Tokens[i] {
			t.Fatalf("pre-resume token %d differs: reference %q, engine %q",
				i, c.Reference.Tokens[i], fb.Engine.Tokens[i])
		}
	}
}

// TestPPParitySnapshotRefusesWrongLength pins the fail-closed edge of the
// boundary snapshot: a truncated snapshot is refused with an error rather
// than restored as zeroes (which would silently reproduce the state-loss
// defect this child exists to catch).
func TestPPParitySnapshotRefusesWrongLength(t *testing.T) {
	st := ppNewModelState(ppTestSeed)
	snap := ppMarshalSnapshot(st, 7)
	if _, _, err := ppRestoreSnapshot(snap[:len(snap)-1]); err == nil {
		t.Fatal("restoring a truncated snapshot must fail, not zero-fill")
	}
	restored, act, err := ppRestoreSnapshot(snap)
	if err != nil {
		t.Fatalf("restoring a well-formed snapshot: %v", err)
	}
	if restored != st || act != 7 {
		t.Fatalf("snapshot round-trip mismatch: state %+v act %d, want %+v act 7", restored, act, st)
	}
}
