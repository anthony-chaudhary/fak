package quality

import (
	"math"
	"strings"
	"testing"
)

func quantBudgetOracles(t *testing.T, c QualityCase) []Oracle {
	t.Helper()
	os, err := Lookup(c.Oracles)
	if err != nil {
		t.Fatalf("Lookup(%v): %v", c.Oracles, err)
	}
	return os
}

// TestQuantBudgetFaithfulEnginePasses is the happy path: a bit-faithful
// quantized decode (zero flips) agrees on every position, scores 1.0, and
// passes the declared budget with no failure bundle.
func TestQuantBudgetFaithfulEnginePasses(t *testing.T) {
	c := QuantBudgetCase("quant-budget-faithful", 0.98)
	res, err := RunCase(c, ReferenceRunner{}, QuantBudgetEngine{Label: "engine-quant-faithful"}, quantBudgetOracles(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if !res.Pass {
		t.Fatalf("bit-faithful quantized engine should pass; got %s", Explain(res))
	}
	if res.FailureBundle != nil {
		t.Fatalf("clean run must not carry a failure bundle: %+v", res.FailureBundle)
	}
	v := res.Verdicts[0]
	if v.Oracle != "quantization-budget" || v.Kind != "statistical" {
		t.Fatalf("verdict identity = %q/%q, want quantization-budget/statistical", v.Oracle, v.Kind)
	}
	if v.Score != 1 {
		t.Errorf("faithful agreement score = %v, want 1", v.Score)
	}
}

// TestQuantBudgetWithinBudgetDriftPasses is the tolerance half of the gate: a
// healthy quantization that flips 2 of 200 tokens (agreement 0.99) is NOT an
// exact match, yet stays above the declared 0.98 budget and passes — the
// statistical gate accepts bounded rounding drift an exact differential would
// red.
func TestQuantBudgetWithinBudgetDriftPasses(t *testing.T) {
	c := QuantBudgetCase("quant-budget-healthy-drift", 0.98)
	eng := QuantBudgetEngine{Label: "engine-quant-healthy", MismatchEvery: 100}
	res, err := RunCase(c, ReferenceRunner{}, eng, quantBudgetOracles(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if !res.Pass {
		t.Fatalf("0.99 agreement should pass a 0.98 budget; got %s", Explain(res))
	}
	v := res.Verdicts[0]
	if math.Abs(v.Score-0.99) > 1e-9 {
		t.Errorf("agreement score = %v, want 0.99 (198/200)", v.Score)
	}
	if !strings.Contains(v.Detail, "0.9900") || !strings.Contains(v.Detail, "0.9800") {
		t.Errorf("pass detail must report measured agreement vs budget; got %q", v.Detail)
	}
	// The drifted stream really does differ from the reference — the pass is a
	// tolerance verdict, not an accidental exact match.
	et, err := eng.Run(c)
	if err != nil {
		t.Fatalf("engine run: %v", err)
	}
	if c.Reference.Tokens[99] == et.Tokens[99] {
		t.Fatal("drift engine should have flipped token 99; the pass would otherwise be tautological")
	}
}

// TestQuantBudgetDefectFailsBelowBudget is the injected-defect witness: a
// defective quantization flipping every 20th token (agreement 0.95) drops below
// the declared 0.98 budget and fails, with Detail reporting measured agreement
// vs budget and FirstDivergence pinned to the first flipped position.
func TestQuantBudgetDefectFailsBelowBudget(t *testing.T) {
	c := QuantBudgetCase("quant-budget-defect", 0.98)
	res, err := RunCase(c, ReferenceRunner{}, QuantBudgetEngine{Label: "engine-quant-defect", MismatchEvery: 20}, quantBudgetOracles(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if res.Pass {
		t.Fatalf("0.95 agreement must not pass a 0.98 budget; got %s", Explain(res))
	}
	fb := res.FailureBundle
	if fb == nil {
		t.Fatal("failing run must carry a failure bundle")
	}
	if fb.FailingOracle != "quantization-budget" {
		t.Errorf("failing oracle = %q, want quantization-budget", fb.FailingOracle)
	}
	if fb.FailingKind != "statistical" {
		t.Errorf("failing kind = %q, want statistical", fb.FailingKind)
	}
	d := fb.FirstDivergence
	if d == nil || d.Index != 19 {
		t.Fatalf("first divergence should pin the first flipped position 19, got %+v", d)
	}
	if want := c.Reference.Tokens[19]; d.Reference != want || d.Engine != quantBudgetPerturb(want) {
		t.Errorf("divergence tokens = ref %q eng %q, want ref %q eng %q",
			d.Reference, d.Engine, want, quantBudgetPerturb(want))
	}
	if !strings.Contains(fb.Detail, "0.9500") || !strings.Contains(fb.Detail, "0.9800") {
		t.Errorf("fail detail must report measured agreement vs budget; got %q", fb.Detail)
	}
	v := res.Verdicts[0]
	if math.Abs(v.Score-0.95) > 1e-9 {
		t.Errorf("agreement score = %v, want 0.95 (190/200)", v.Score)
	}
}

// TestQuantBudgetTruncationCountsAgainstAgreement closes the truncation
// loophole: an engine that emits only a matching prefix cannot buy agreement by
// stopping early — missing positions are disagreements, the rate is measured
// over the longer stream, and the first divergence is the first missing index.
func TestQuantBudgetTruncationCountsAgainstAgreement(t *testing.T) {
	c := QuantBudgetCase("quant-budget-truncated", 0.98)
	ref := c.Reference
	eng := Trace{Runner: "engine-quant-truncated", Tokens: ref.Tokens[:150]}
	v := QuantBudget{}.Judge(ref, eng, c)
	if v.Pass {
		t.Fatalf("a 150/200 truncated stream must not pass a 0.98 budget; detail %q", v.Detail)
	}
	if math.Abs(v.Score-0.75) > 1e-9 {
		t.Errorf("agreement score = %v, want 0.75 (150/200)", v.Score)
	}
	if v.FirstDivergence == nil || v.FirstDivergence.Index != 150 {
		t.Fatalf("first divergence should be the first missing position 150, got %+v", v.FirstDivergence)
	}
	if v.FirstDivergence.Engine != "<end>" {
		t.Errorf("engine side of a missing position = %q, want <end>", v.FirstDivergence.Engine)
	}
}

// TestQuantBudgetDefaultBudgetApplies proves the budget is genuinely declared
// per case with a safe default: a case declaring no budget (MinScore 0) gates
// at the package default 0.98 and still fails a 0.95-agreement defect, while a
// case explicitly declaring a looser 0.90 budget passes the same defect —
// same traces, different declared tolerance, different verdict.
func TestQuantBudgetDefaultBudgetApplies(t *testing.T) {
	defect := QuantBudgetEngine{Label: "engine-quant-defect", MismatchEvery: 20}

	undeclared := QuantBudgetCase("quant-budget-default", 0)
	et, err := defect.Run(undeclared)
	if err != nil {
		t.Fatalf("engine run: %v", err)
	}
	v := QuantBudget{}.Judge(undeclared.Reference, et, undeclared)
	if v.Pass {
		t.Fatalf("0.95 agreement must fail the default 0.98 budget; detail %q", v.Detail)
	}
	if !strings.Contains(v.Detail, "0.9800") {
		t.Errorf("default-budget detail must name the 0.98 gate; got %q", v.Detail)
	}

	loose := QuantBudgetCase("quant-budget-loose", 0.90)
	v = QuantBudget{}.Judge(loose.Reference, et, loose)
	if !v.Pass {
		t.Fatalf("0.95 agreement should pass a declared 0.90 budget; detail %q", v.Detail)
	}
	if !strings.Contains(v.Detail, "0.9000") {
		t.Errorf("loose-budget detail must name the declared 0.90 gate; got %q", v.Detail)
	}
}

// TestQuantBudgetEmptyReferenceRefuses documents the malformed-input posture:
// with no reference tokens an agreement rate is undefined, so the oracle
// refuses (fails) with a diagnostic rather than passing vacuously.
func TestQuantBudgetEmptyReferenceRefuses(t *testing.T) {
	c := QuantBudgetCase("quant-budget-empty-ref", 0.98)
	v := QuantBudget{}.Judge(Trace{}, Trace{Tokens: []string{"tok000"}}, c)
	if v.Pass {
		t.Fatal("an empty reference must not judge as a pass")
	}
	if !strings.Contains(v.Detail, "reference carries no tokens") {
		t.Errorf("empty-reference detail should say why; got %q", v.Detail)
	}
}
