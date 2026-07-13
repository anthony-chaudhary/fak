package quality

import (
	"strings"
	"testing"
)

// resumeTestSeed / resumeTestSteps pin the deterministic decode the resume
// tests run. The defect tests assert their divergence lands exactly at the
// interrupt step, so each one first asserts the precondition that makes the
// boundary token distinguishable (no vocab collision at the tested index).
const (
	resumeTestSeed  = int64(42)
	resumeTestSteps = 8
)

// TestResumeParityFaithfulResumeAcrossInterruptPoints is the happy path at
// several k values, including both edges: interrupting at step k, serializing
// the state to bytes, restoring, and resuming yields a token stream identical
// to the uninterrupted reference — a correct resume is invisible in the output.
func TestResumeParityFaithfulResumeAcrossInterruptPoints(t *testing.T) {
	c := ResumeParityCase(resumeTestSeed, resumeTestSteps)
	ref := resumeDecode(resumeTestSeed, resumeTestSteps)
	for _, k := range []int{0, 1, 3, 5, resumeTestSteps - 1, resumeTestSteps} {
		res, err := RunCase(c, ResumeUninterruptedRunner{}, ResumeEngine(k, ""), oraclesFor(t, c))
		if err != nil {
			t.Fatalf("k=%d RunCase: %v", k, err)
		}
		if !res.Pass {
			t.Fatalf("k=%d faithful resume must pass; got %s", k, Explain(res))
		}
		if res.FailureBundle != nil {
			t.Fatalf("k=%d clean resume must not carry a failure bundle: %+v", k, res.FailureBundle)
		}
		// Token-identical, verified independently of the oracle.
		eng, err := ResumeEngine(k, "").Run(c)
		if err != nil {
			t.Fatalf("k=%d engine run: %v", k, err)
		}
		if len(eng.Tokens) != len(ref.Tokens) {
			t.Fatalf("k=%d resumed decode emitted %d tokens, want %d", k, len(eng.Tokens), len(ref.Tokens))
		}
		for i := range ref.Tokens {
			if eng.Tokens[i] != ref.Tokens[i] {
				t.Fatalf("k=%d token %d = %q, want %q", k, i, eng.Tokens[i], ref.Tokens[i])
			}
		}
	}
}

// TestResumeParityBoundaryTokenExact pins the resume boundary itself: for a
// faithful resume at k, the token AT index k — the first one decoded from the
// restored state — matches the reference exactly, and its neighbors are intact,
// so nothing was dropped or duplicated crossing the snapshot.
func TestResumeParityBoundaryTokenExact(t *testing.T) {
	const k = 3
	c := ResumeParityCase(resumeTestSeed, resumeTestSteps)
	ref := resumeDecode(resumeTestSeed, resumeTestSteps)
	eng, err := ResumeEngine(k, "").Run(c)
	if err != nil {
		t.Fatalf("engine run: %v", err)
	}
	if len(eng.Tokens) != resumeTestSteps {
		t.Fatalf("resumed decode emitted %d tokens, want %d (no drop/dup at the boundary)",
			len(eng.Tokens), resumeTestSteps)
	}
	for _, i := range []int{k - 1, k, k + 1} {
		if eng.Tokens[i] != ref.Tokens[i] {
			t.Errorf("boundary neighborhood token %d = %q, want %q", i, eng.Tokens[i], ref.Tokens[i])
		}
	}
}

// resumeRunDefect runs the pinned case against a defective engine and returns
// the failure bundle, asserting a failure was produced with a localized
// divergence at exactly step k carrying the expected reference/engine tokens
// and a Detail naming both.
func resumeRunDefect(t *testing.T, defect string, k int, wantEng string) {
	t.Helper()
	c := ResumeParityCase(resumeTestSeed, resumeTestSteps)
	res, err := RunCase(c, ResumeUninterruptedRunner{}, ResumeEngine(k, defect), oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if res.Pass {
		t.Fatalf("%s engine must not pass; got %s", defect, Explain(res))
	}
	fb := res.FailureBundle
	if fb == nil {
		t.Fatal("failing resume must carry a failure bundle")
	}
	if fb.FailingOracle != "resume-parity" {
		t.Errorf("first failing oracle = %q, want resume-parity", fb.FailingOracle)
	}
	d := fb.FirstDivergence
	if d == nil {
		t.Fatalf("%s failure must localize a first divergence", defect)
	}
	if d.Index != k {
		t.Fatalf("%s divergence index = %d, want the resume boundary %d (bundle: %+v)", defect, d.Index, k, d)
	}
	wantRef := c.Reference.Tokens[k]
	if d.Reference != wantRef {
		t.Errorf("divergence reference token = %q, want %q", d.Reference, wantRef)
	}
	if d.Engine != wantEng {
		t.Errorf("divergence engine token = %q, want %q", d.Engine, wantEng)
	}
	// Localized detail: the human-readable evidence names both tokens.
	if !strings.Contains(fb.Detail, `"`+wantRef+`"`) || !strings.Contains(fb.Detail, `"`+wantEng+`"`) {
		t.Errorf("detail must carry reference %q and engine %q tokens; got %q", wantRef, wantEng, fb.Detail)
	}
}

// TestResumeParityReseedFailsAtBoundary is the lost-state witness: a resume
// that ignores the restored snapshot and re-seeds from scratch silently
// replays generation from token 0 at position k — and the oracle pins the
// first divergence to exactly the resume boundary.
func TestResumeParityReseedFailsAtBoundary(t *testing.T) {
	const k = 3
	ref := resumeDecode(resumeTestSeed, resumeTestSteps)
	// Precondition: the boundary token must differ from token 0 for this seed,
	// or the replay-from-scratch defect would first surface later than k.
	if ref.Tokens[k] == ref.Tokens[0] {
		t.Fatalf("precondition: seed %d has ref[%d]==ref[0]==%q; pick a seed without this collision",
			resumeTestSeed, k, ref.Tokens[k])
	}
	resumeRunDefect(t, "reseed", k, ref.Tokens[0])
}

// TestResumeParityDropBoundaryFailsAtK is the lost-token witness: a resume that
// decodes the boundary token but loses its emission shifts every later token
// left, so the first divergence is at index k with the reference's k+1 token in
// the engine slot.
func TestResumeParityDropBoundaryFailsAtK(t *testing.T) {
	const k = 3
	ref := resumeDecode(resumeTestSeed, resumeTestSteps)
	// Precondition: adjacent tokens at the boundary must differ for this seed.
	if ref.Tokens[k] == ref.Tokens[k+1] {
		t.Fatalf("precondition: seed %d has ref[%d]==ref[%d]==%q; pick a seed without this collision",
			resumeTestSeed, k, k+1, ref.Tokens[k])
	}
	resumeRunDefect(t, "drop-boundary", k, ref.Tokens[k+1])
}

// TestResumeParityDupBoundaryFailsAtK is the replayed-step witness: a retry
// that re-emits the last pre-snapshot token puts the reference's k-1 token in
// the engine's k slot, and the oracle fails at exactly index k.
func TestResumeParityDupBoundaryFailsAtK(t *testing.T) {
	const k = 3
	ref := resumeDecode(resumeTestSeed, resumeTestSteps)
	// Precondition: the boundary token must differ from its predecessor.
	if ref.Tokens[k] == ref.Tokens[k-1] {
		t.Fatalf("precondition: seed %d has ref[%d]==ref[%d]==%q; pick a seed without this collision",
			resumeTestSeed, k, k-1, ref.Tokens[k])
	}
	resumeRunDefect(t, "dup-boundary", k, ref.Tokens[k-1])
}

// TestResumeSnapshotRoundTrip proves the serialization seam itself: a state
// marshaled and restored is field-identical and continues generation exactly as
// the original would, and a snapshot of the wrong length is refused rather than
// restored as zeroes (which would silently reproduce the reseed defect).
func TestResumeSnapshotRoundTrip(t *testing.T) {
	st := resumeNewState(resumeTestSeed)
	for i := 0; i < 3; i++ {
		st.next()
	}
	restored, err := resumeRestore(st.marshal())
	if err != nil {
		t.Fatalf("restore: %v", err)
	}
	if restored != st {
		t.Fatalf("round-tripped state = %+v, want %+v", restored, st)
	}
	// Identical continuation from both values.
	a, b := st, restored
	for i := 0; i < 5; i++ {
		ta, tb := a.next(), b.next()
		if ta != tb {
			t.Fatalf("continuation step %d: original %q, restored %q", i, ta, tb)
		}
	}
	if _, err := resumeRestore([]byte{1, 2, 3}); err == nil {
		t.Fatal("restoring a truncated snapshot must fail, not zero-fill")
	}
}
