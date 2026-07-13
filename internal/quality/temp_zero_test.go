package quality

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

// TestTempZeroFaithfulSurfacesPass is the happy path of the temperature-zero
// contract: an engine that routes every request surface through the shared
// greedy decode passes, two independent runs are byte-identical, and — the
// clause case.go freezes for #4525 — the decode is identical REGARDLESS of the
// pinned seed, because a faithful temp-0 path never touches the sampler RNG.
func TestTempZeroFaithfulSurfacesPass(t *testing.T) {
	c := TZCase()
	res, err := RunCase(c, ReferenceRunner{}, TZEngine(""), oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if !res.Pass {
		t.Fatalf("faithful multi-surface engine should pass; got %s", Explain(res))
	}
	if res.FailureBundle != nil {
		t.Fatalf("clean run must not carry a failure bundle: %+v", res.FailureBundle)
	}

	// Byte-identical determinism: two independent runs marshal to the same bytes.
	eng := TZEngine("")
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
		t.Fatalf("two temp-0 runs must be byte-identical:\n  first:  %s\n  second: %s", b1, b2)
	}

	// Seed-independence: the same case under a different seed decodes identically.
	reseeded := c
	reseeded.Params.Seed = 999
	t3, err := eng.Run(reseeded)
	if err != nil {
		t.Fatalf("reseeded engine run: %v", err)
	}
	b3, err := json.Marshal(t3)
	if err != nil {
		t.Fatalf("marshal reseeded trace: %v", err)
	}
	if !bytes.Equal(b1, b3) {
		t.Fatalf("temp-0 decode must not depend on the seed:\n  seed %d: %s\n  seed %d: %s",
			c.Params.Seed, b1, reseeded.Params.Seed, b3)
	}
}

// TestTempZeroSamplingNoiseFailsNamedSurface is the localized-defect witness: a
// temp-0 path that still injects sampling noise on ONE surface fails, the first
// divergence pins the exact step, the Detail names the offending surface and
// index, and the other surfaces are proven to have matched — the failure
// localizes to "surface streaming, token 2", not "the run looked wrong".
func TestTempZeroSamplingNoiseFailsNamedSurface(t *testing.T) {
	c := TZCase()
	res, err := RunCase(c, ReferenceRunner{}, TZEngine("sampling-noise"), oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if res.Pass {
		t.Fatalf("sampling-noise engine must not pass; got %s", Explain(res))
	}
	fb := res.FailureBundle
	if fb == nil {
		t.Fatal("failing run must carry a failure bundle")
	}
	if fb.FailingOracle != "temp-zero-determinism" {
		t.Errorf("first failing oracle = %q, want temp-zero-determinism", fb.FailingOracle)
	}
	d := fb.FirstDivergence
	if d == nil || d.Index != tzNoiseStep {
		t.Fatalf("expected first divergence at step %d, got %+v", tzNoiseStep, d)
	}
	wantRef := c.Reference.Tokens[tzNoiseStep]
	if d.Reference != wantRef {
		t.Errorf("divergence reference token = %q, want %q", d.Reference, wantRef)
	}
	if wantEng := tzRotate(wantRef); d.Engine != wantEng {
		t.Errorf("divergence engine token = %q, want %q", d.Engine, wantEng)
	}
	if !strings.Contains(fb.Detail, `surface "`+tzNoisySurface+`"`) {
		t.Errorf("detail must name the offending surface %q; got %q", tzNoisySurface, fb.Detail)
	}

	// The defect is surface-local: every OTHER surface in the captured engine
	// trace still equals the greedy reference exactly.
	surfaces := tzParseSurfaces(fb.Engine)
	if len(surfaces) != len(tzSurfaces) {
		t.Fatalf("engine trace should carry %d surfaces, got %d", len(tzSurfaces), len(surfaces))
	}
	for _, s := range surfaces {
		if s.Surface == tzNoisySurface {
			continue
		}
		if _, _, _, diverged := tzFirstDiff(c.Reference.Tokens, s.Tokens); diverged {
			t.Errorf("surface %q should have matched the reference; tokens %v", s.Surface, s.Tokens)
		}
	}
}

// TestTempZeroPlainTraceAndScopeGuards covers the oracle's two edges: a plain
// single-path trace (no surface envelope) is judged as one surface — passing
// when greedy-equal, failing with a localized divergence when not — and a case
// that is not temperature-zero is refused rather than judged to a misleading
// verdict.
func TestTempZeroPlainTraceAndScopeGuards(t *testing.T) {
	c := TZCase()
	ref := c.Reference

	clean := Trace{Runner: "plain-engine", Tokens: ref.Tokens, Text: ref.Text}
	if v := (TZDeterminism{}).Judge(ref, clean, c); !v.Pass {
		t.Fatalf("plain greedy-equal trace should pass; got %+v", v)
	}

	mut := append([]string(nil), ref.Tokens...)
	mut[1] = tzRotate(mut[1])
	dirty := Trace{Runner: "plain-engine", Tokens: mut, Text: strings.Join(mut, " ")}
	v := (TZDeterminism{}).Judge(ref, dirty, c)
	if v.Pass {
		t.Fatal("plain diverging trace must fail")
	}
	if v.FirstDivergence == nil || v.FirstDivergence.Index != 1 {
		t.Fatalf("expected first divergence at token 1, got %+v", v.FirstDivergence)
	}

	sampled := c
	sampled.Params.Temperature = 0.7
	if v := (TZDeterminism{}).Judge(ref, clean, sampled); v.Pass {
		t.Fatal("a non-temp-0 case must be refused, not passed")
	} else if !strings.Contains(v.Detail, "temperature") {
		t.Errorf("refusal detail should mention temperature; got %q", v.Detail)
	}
}
