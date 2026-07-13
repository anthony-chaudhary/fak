package quality

import (
	"fmt"
	"strings"
	"testing"
)

// TestPrefillParityChunkedMatchesMonolithic is the parity contract on the happy
// path: a chunked prefill that carries its accumulated state across chunk
// boundaries reproduces the monolithic decode token-identically at EVERY split
// point — chunk size 1 (a boundary after every token), interior sizes, a size
// equal to the prompt, larger than the prompt, and the degenerate <=0
// single-chunk form.
func TestPrefillParityChunkedMatchesMonolithic(t *testing.T) {
	c := PrefillParityCase()
	nPrompt := len(prefillTokens(c.Prompt))
	if nPrompt < 4 {
		t.Fatalf("case prompt must be long enough to chunk meaningfully; got %d tokens", nPrompt)
	}
	for _, k := range []int{0, 1, 2, 3, 4, 5, 7, nPrompt, nPrompt + 3} {
		t.Run(fmt.Sprintf("chunk-%d", k), func(t *testing.T) {
			res, err := RunCase(c, PrefillMonolithicRunner{}, PrefillEngine(k, ""), oraclesFor(t, c))
			if err != nil {
				t.Fatalf("RunCase: %v", err)
			}
			if !res.Pass {
				t.Fatalf("faithful chunked prefill at chunk size %d must match monolithic; got %s", k, Explain(res))
			}
			if res.FailureBundle != nil {
				t.Fatalf("clean parity run must not carry a failure bundle: %+v", res.FailureBundle)
			}
			// Token-identical, checked directly on the captured provenance rather
			// than trusting the pass fold alone.
			ref, eng := res.Provenance.Reference.Tokens, res.Provenance.Engine.Tokens
			if len(ref) != len(eng) {
				t.Fatalf("token counts differ: monolithic %d, chunked %d", len(ref), len(eng))
			}
			for i := range ref {
				if ref[i] != eng[i] {
					t.Fatalf("output token %d differs: monolithic %q, chunked %q", i, ref[i], eng[i])
				}
			}
		})
	}
}

// TestPrefillParityCaseReferenceIsMonolithic pins the case's declared golden
// trace to the monolithic path: judging the live monolithic runner against the
// case's Reference (replayed by the spine's ReferenceRunner) also passes, so the
// declared reference and the live monolithic decode are the same trace.
func TestPrefillParityCaseReferenceIsMonolithic(t *testing.T) {
	c := PrefillParityCase()
	res, err := RunCase(c, ReferenceRunner{}, PrefillMonolithicRunner{}, oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if !res.Pass {
		t.Fatalf("monolithic decode must reproduce the case's declared reference; got %s", Explain(res))
	}
}

// TestPrefillParityStateResetFailsAtFirstAffectedToken is the injected-defect
// witness: an engine that resets its accumulated prefill state at chunk
// boundaries decodes from the wrong state, the run fails, and the oracle pins
// the FIRST affected output token — with the monolithic and chunked tokens
// carried in both the divergence and the human-readable detail.
func TestPrefillParityStateResetFailsAtFirstAffectedToken(t *testing.T) {
	c := PrefillParityCase()
	res, err := RunCase(c, PrefillMonolithicRunner{}, PrefillEngine(3, prefillDefectReset), oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if res.Pass {
		t.Fatalf("state-reset engine must not pass parity; got %s", Explain(res))
	}
	fb := res.FailureBundle
	if fb == nil {
		t.Fatal("failing parity run must carry a failure bundle")
	}
	if fb.FailingOracle != "prefill-parity" {
		t.Errorf("first failing oracle = %q, want prefill-parity", fb.FailingOracle)
	}
	d := fb.FirstDivergence
	if d == nil {
		t.Fatal("parity failure must localize a first divergence")
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
		t.Fatalf("state-reset traces should disagree token-wise; ref %v eng %v", ref, eng)
	}
	if d.Index != first {
		t.Errorf("divergence index = %d, want first mismatch %d", d.Index, first)
	}
	if d.Reference != ref[first] || d.Engine != eng[first] {
		t.Errorf("divergence tokens = ref %q eng %q, want ref %q eng %q",
			d.Reference, d.Engine, ref[first], eng[first])
	}
	// Detail shows ref vs eng at the divergence, so a human reads exactly where
	// and how the chunked decode split from the monolithic one.
	for _, want := range []string{
		fmt.Sprintf("token %d", d.Index),
		fmt.Sprintf("%q", d.Reference),
		fmt.Sprintf("%q", d.Engine),
	} {
		if !strings.Contains(fb.Detail, want) {
			t.Errorf("detail %q missing %s", fb.Detail, want)
		}
	}
}

// TestPrefillParityResetNeedsABoundary scopes the defect: the same state-reset
// engine run as a SINGLE chunk (chunk size past the prompt length) has no
// boundary to reset at, so it decodes faithfully and passes. The injected bug is
// a chunk-BOUNDARY bug, and the gate trips exactly when a boundary exists to
// break.
func TestPrefillParityResetNeedsABoundary(t *testing.T) {
	c := PrefillParityCase()
	single := len(prefillTokens(c.Prompt)) + 1
	res, err := RunCase(c, PrefillMonolithicRunner{}, PrefillEngine(single, prefillDefectReset), oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if !res.Pass {
		t.Fatalf("single-chunk run has no boundary to reset at and must pass; got %s", Explain(res))
	}
}

// TestPrefillParityTruncatedDecodeLengthDivergence unit-tests the oracle's
// length rung directly: an engine trace that stops short of the reference fails
// with the divergence pinned at the first missing index and "<end>" on the
// engine side.
func TestPrefillParityTruncatedDecodeLengthDivergence(t *testing.T) {
	ref := Trace{Tokens: []string{"anchor", "beacon", "cobalt"}}
	eng := Trace{Tokens: []string{"anchor", "beacon"}}
	v := PrefillParity{}.Judge(ref, eng, QualityCase{})
	if v.Pass {
		t.Fatal("shorter engine trace must not pass parity")
	}
	d := v.FirstDivergence
	if d == nil || d.Index != 2 || d.Reference != "cobalt" || d.Engine != "<end>" {
		t.Fatalf("want divergence at index 2 with reference \"cobalt\" and engine \"<end>\", got %+v", d)
	}
}
