package quality

import (
	"strings"
	"testing"
)

// TestDtypeParityFaithfulLanesPass is the happy path: an engine whose FP32,
// BF16, and FP16 lanes carry only each dtype's honest grid rounding stays
// within every declared band, and the pass verdict accounts for all three
// budgets by name.
func TestDtypeParityFaithfulLanesPass(t *testing.T) {
	c := DtypeDeltaCase()
	res, err := RunCase(c, ReferenceRunner{}, DtypeDeltaEngine(""), oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if !res.Pass {
		t.Fatalf("faithful dtype lanes should pass; got %s", Explain(res))
	}
	if res.FailureBundle != nil {
		t.Fatalf("clean dtype run must not carry a failure bundle: %+v", res.FailureBundle)
	}
	if len(res.Verdicts) != 1 {
		t.Fatalf("expected one verdict, got %d", len(res.Verdicts))
	}
	for _, name := range []string{"fp32", "bf16", "fp16"} {
		if !strings.Contains(res.Verdicts[0].Detail, name) {
			t.Errorf("pass detail should account for dtype %s; got %q", name, res.Verdicts[0].Detail)
		}
	}
}

// TestDtypeParityFP16BlowoutFailsAtItsToken is the localized-defect witness:
// an fp16 lane pushed beyond its band fails, the first divergence pins the
// exact token index the drift occurred at, and the detail names BOTH the
// offending dtype and the token there.
func TestDtypeParityFP16BlowoutFailsAtItsToken(t *testing.T) {
	c := DtypeDeltaCase()
	res, err := RunCase(c, ReferenceRunner{}, DtypeDeltaEngine("fp16-blowout"), oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if res.Pass {
		t.Fatalf("fp16 lane beyond its band must not pass; got %s", Explain(res))
	}
	fb := res.FailureBundle
	if fb == nil {
		t.Fatal("failing dtype run must carry a failure bundle")
	}
	if fb.FailingOracle != "dtype-parity" {
		t.Errorf("first failing oracle = %q, want dtype-parity", fb.FailingOracle)
	}
	d := fb.FirstDivergence
	if d == nil || d.Index != dtFP16DefectStep {
		t.Fatalf("expected first divergence at token %d, got %+v", dtFP16DefectStep, d)
	}
	if d.Reference == d.Engine {
		t.Errorf("divergence must carry differing logit values; both %q", d.Reference)
	}
	if !strings.Contains(fb.Detail, "fp16") {
		t.Errorf("detail must name the offending dtype fp16; got %q", fb.Detail)
	}
	wantTok := c.Reference.Tokens[dtFP16DefectStep]
	if !strings.Contains(fb.Detail, `"`+wantTok+`"`) {
		t.Errorf("detail must name the offending token %q; got %q", wantTok, fb.Detail)
	}
}

// TestDtypeParityBandsArePerDtype proves the budgets are per-dtype and tighter
// for BF16 than FP16: the SAME mid-band delta (sized strictly between the two
// bands) fails when it appears on the bf16 lane and passes when it appears on
// the fp16 lane.
func TestDtypeParityBandsArePerDtype(t *testing.T) {
	c := DtypeDeltaCase()
	ref, err := ReferenceRunner{}.Run(c)
	if err != nil {
		t.Fatalf("reference run: %v", err)
	}

	bfEng, err := DtypeDeltaEngine("bf16-band").Run(c)
	if err != nil {
		t.Fatalf("bf16-band engine run: %v", err)
	}
	v := DtypeDelta{}.Judge(ref, bfEng, c)
	if v.Pass {
		t.Fatalf("mid-band delta on the bf16 lane must exceed bf16's tighter band; got %+v", v)
	}
	if !strings.Contains(v.Detail, "bf16") {
		t.Errorf("detail must name the offending dtype bf16; got %q", v.Detail)
	}
	if v.FirstDivergence == nil || v.FirstDivergence.Index != dtBF16DefectStep {
		t.Fatalf("expected first divergence at token %d, got %+v", dtBF16DefectStep, v.FirstDivergence)
	}

	lanes := dtCleanLanes(c.Reference.Logits)
	lanes["fp16"][dtBF16DefectStep][dtDefectCand] += dtMidBandBump
	fpEng := Trace{Tokens: c.Reference.Tokens, Text: dtEncodeLanes(lanes)}
	if v := (DtypeDelta{}).Judge(ref, fpEng, c); !v.Pass {
		t.Fatalf("the same mid-band delta must stay within fp16's looser band; got %+v", v)
	}
}

// TestDtypeParityMissingLaneFailsClosed: an engine payload that silently drops
// a declared dtype lane is a refusal, not a pass — a lane that was never
// executed cannot be within budget.
func TestDtypeParityMissingLaneFailsClosed(t *testing.T) {
	c := DtypeDeltaCase()
	ref, err := ReferenceRunner{}.Run(c)
	if err != nil {
		t.Fatalf("reference run: %v", err)
	}
	lanes := dtCleanLanes(c.Reference.Logits)
	delete(lanes, "fp16")
	eng := Trace{Tokens: c.Reference.Tokens, Text: dtEncodeLanes(lanes)}
	v := DtypeDelta{}.Judge(ref, eng, c)
	if v.Pass {
		t.Fatal("a payload missing the fp16 lane must fail closed")
	}
	if !strings.Contains(v.Detail, "fp16") {
		t.Errorf("detail must name the missing lane fp16; got %q", v.Detail)
	}
}
