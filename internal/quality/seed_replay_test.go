package quality

import (
	"bytes"
	"encoding/json"
	"testing"
)

// TestSeedReplaySameSeedReproduces is the replay contract on the happy path: an
// engine pinned to the case's seed reproduces the reference sequence exactly, and
// two independent same-seed runs are byte-identical — seeded stochastic generation
// is replayable evidence, not a one-off sample.
func TestSeedReplaySameSeedReproduces(t *testing.T) {
	c := SeedReplayCase(42)
	res, err := RunCase(c, ReferenceRunner{}, SeedReplayEngine(""), oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if !res.Pass {
		t.Fatalf("same-seed engine should replay exactly; got %s", Explain(res))
	}
	if res.FailureBundle != nil {
		t.Fatalf("clean replay must not carry a failure bundle: %+v", res.FailureBundle)
	}

	// Byte-identical replay: two independent runs of the same engine under the same
	// seed marshal to the same bytes — no hidden state, no ambient entropy.
	eng := SeedReplayEngine("")
	t1, err := eng.Run(c)
	if err != nil {
		t.Fatalf("first engine run: %v", err)
	}
	t2, err := eng.Run(c)
	if err != nil {
		t.Fatalf("second engine run: %v", err)
	}
	b1, err := json.Marshal(t1)
	if err != nil {
		t.Fatalf("marshal first trace: %v", err)
	}
	b2, err := json.Marshal(t2)
	if err != nil {
		t.Fatalf("marshal second trace: %v", err)
	}
	if !bytes.Equal(b1, b2) {
		t.Fatalf("two same-seed runs must be byte-identical:\n  first:  %s\n  second: %s", b1, b2)
	}
}

// TestSeedReplayStepBugFailsAtItsStep is the localized-defect witness: a
// step-dependent sampling bug at step 3 leaves the prefix intact and the oracle
// pins the first divergence to exactly that step, with the reference and engine
// tokens reported. Rotation in the vocab guarantees the mutant token differs.
func TestSeedReplayStepBugFailsAtItsStep(t *testing.T) {
	c := SeedReplayCase(42)
	res, err := RunCase(c, ReferenceRunner{}, SeedReplayEngine("step-bug"), oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if res.Pass {
		t.Fatalf("step-bug engine must not pass; got %s", Explain(res))
	}
	fb := res.FailureBundle
	if fb == nil {
		t.Fatal("failing replay must carry a failure bundle")
	}
	if fb.FailingOracle != "seed-replay" {
		t.Errorf("first failing oracle = %q, want seed-replay", fb.FailingOracle)
	}
	d := fb.FirstDivergence
	if d == nil || d.Index != seedReplayBugStep {
		t.Fatalf("expected first divergence at step %d, got %+v", seedReplayBugStep, d)
	}
	wantRef := c.Reference.Tokens[seedReplayBugStep]
	if d.Reference != wantRef {
		t.Errorf("divergence reference token = %q, want %q", d.Reference, wantRef)
	}
	if wantEng := seedReplayRotate(wantRef); d.Engine != wantEng {
		t.Errorf("divergence engine token = %q, want %q", d.Engine, wantEng)
	}
}

// TestSeedReplaySeedDriftFails is the unseeded-RNG witness: an engine that ignores
// the pinned seed (samples under seed+1) diverges from the reference, and the
// reported first divergence carries the actual tokens from each trace at that
// index — the evidence a human needs to see the two sampling paths split.
func TestSeedReplaySeedDriftFails(t *testing.T) {
	c := SeedReplayCase(42)
	res, err := RunCase(c, ReferenceRunner{}, SeedReplayEngine("seed-drift"), oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if res.Pass {
		t.Fatalf("seed-drift engine must not pass; got %s", Explain(res))
	}
	fb := res.FailureBundle
	if fb == nil {
		t.Fatal("failing replay must carry a failure bundle")
	}
	d := fb.FirstDivergence
	if d == nil {
		t.Fatal("seed-drift failure must localize a first divergence")
	}
	// The reported divergence must be the FIRST mismatch between the two traces,
	// computed independently here, and must carry each trace's token at that index.
	ref, eng := fb.Reference.Tokens, fb.Engine.Tokens
	first := -1
	for i := 0; i < len(ref) && i < len(eng); i++ {
		if ref[i] != eng[i] {
			first = i
			break
		}
	}
	if first < 0 {
		t.Fatalf("seed-drift traces should disagree token-wise; ref %v eng %v", ref, eng)
	}
	if d.Index != first {
		t.Errorf("divergence index = %d, want first mismatch %d", d.Index, first)
	}
	if d.Reference != ref[first] || d.Engine != eng[first] {
		t.Errorf("divergence tokens = ref %q eng %q, want ref %q eng %q",
			d.Reference, d.Engine, ref[first], eng[first])
	}
}

// TestSeedReplayIsSeedScoped documents the scope of the replay guarantee: a
// DIFFERENT seed may — and for this vocab does — produce a different sequence.
// Replay asserts same-seed reproducibility, not global constancy of generation.
func TestSeedReplayIsSeedScoped(t *testing.T) {
	n := SeedReplayCase(42).Params.MaxTokens
	a := seedReplaySample(42, n)
	b := seedReplaySample(1337, n)
	differ := len(a.Tokens) != len(b.Tokens)
	for i := 0; !differ && i < len(a.Tokens); i++ {
		differ = a.Tokens[i] != b.Tokens[i]
	}
	if !differ {
		t.Fatalf("seeds 42 and 1337 should sample different sequences over %d steps; both got %v", n, a.Tokens)
	}
	// And the oracle agrees: judged against a reference from another seed, the
	// engine's (correct, faithful) decode is a divergence, not a pass — the oracle
	// checks seed fidelity, not some seed-independent invariant.
	c := SeedReplayCase(42)
	c.Reference = b
	v := SeedReplay{}.Judge(seedReplaySample(1337, n), seedReplaySample(42, n), c)
	if v.Pass {
		t.Fatal("different-seed traces must not judge as a replay match")
	}
	if v.FirstDivergence == nil {
		t.Fatal("different-seed mismatch must still localize a first divergence")
	}
}
