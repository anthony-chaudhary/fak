package main

import (
	"bytes"
	"math"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/negframe"
)

// TestFixtureCorpusIntegrity is the load-bearing test: it proves the fixture corpus is honestly
// paired -- every Arm A is genuinely negatively framed (negframe finds >=1 negation) and every
// Arm B is genuinely clean (negframe finds none). Without this, the whole A/B delta would rest
// on an unverified assumption instead of the deterministic classifier.
func TestFixtureCorpusIntegrity(t *testing.T) {
	if len(fixtures) < 10 {
		t.Fatalf("fixture corpus too small to be a meaningful sample: got %d pairs", len(fixtures))
	}
	seen := map[string]bool{}
	for _, fx := range fixtures {
		if seen[fx.ID] {
			t.Errorf("duplicate fixture id %q", fx.ID)
		}
		seen[fx.ID] = true

		fa := negframe.Classify("t/"+fx.ID+"/a", fx.ArmA)
		if len(fa) == 0 {
			t.Errorf("pair %q: Arm A (%q) carries no negframe finding -- not actually negative-framed", fx.ID, fx.ArmA)
		}
		fb := negframe.Classify("t/"+fx.ID+"/b", fx.ArmB)
		if len(fb) != 0 {
			t.Errorf("pair %q: Arm B (%q) still carries %d negframe finding(s) -- reframe leaked a negation", fx.ID, fx.ArmB, len(fb))
		}
	}
}

// TestComplianceProxyMonotonic pins the modeled cost function's three required properties: it is
// bounded to [floor, ceiling], strictly decreasing in each finding count, and returns the ceiling
// for a clean (0,0) directive.
func TestComplianceProxyMonotonic(t *testing.T) {
	if got := complianceProxy(0, 0); got != complianceCeiling {
		t.Errorf("complianceProxy(0,0) = %v, want ceiling %v", got, complianceCeiling)
	}
	prev := complianceProxy(0, 0)
	for m := 1; m <= 5; m++ {
		got := complianceProxy(m, 0)
		if got >= prev {
			t.Errorf("complianceProxy(%d,0) = %v, want strictly less than complianceProxy(%d,0) = %v", m, got, m-1, prev)
		}
		prev = got
	}
	prev = complianceProxy(0, 0)
	for j := 1; j <= 5; j++ {
		got := complianceProxy(0, j)
		if got >= prev {
			t.Errorf("complianceProxy(0,%d) = %v, want strictly less than complianceProxy(0,%d) = %v", j, got, j-1, prev)
		}
		prev = got
	}
	if got := complianceProxy(1000, 1000); got != complianceFloor {
		t.Errorf("complianceProxy(1000,1000) = %v, want floor %v", got, complianceFloor)
	}
}

// TestSignTestPValueKnownValues pins the exact binomial sign test against hand-computable cases.
func TestSignTestPValueKnownValues(t *testing.T) {
	cases := []struct {
		n, k int
		want float64
	}{
		{n: 0, k: 0, want: 1.0},
		{n: 4, k: 4, want: 1.0 / 16.0}, // only the all-successes outcome
		{n: 4, k: 0, want: 1.0},        // every outcome satisfies X>=0
		{n: 2, k: 1, want: 3.0 / 4.0},  // P(X>=1) for n=2: 1 - P(X=0)=1-1/4
		{n: 10, k: 10, want: 1.0 / 1024.0},
	}
	for _, c := range cases {
		got := signTestPValue(c.n, c.k)
		if math.Abs(got-c.want) > 1e-9 {
			t.Errorf("signTestPValue(%d,%d) = %v, want %v", c.n, c.k, got, c.want)
		}
	}
}

// TestRunExperimentDeterministic proves the harness has no hidden randomness: two runs over the
// same fixed fixture corpus must agree bit-for-bit on every aggregate.
func TestRunExperimentDeterministic(t *testing.T) {
	r1 := runExperiment()
	r2 := runExperiment()
	if r1.MeanA != r2.MeanA || r1.MeanB != r2.MeanB || r1.MeanDelta != r2.MeanDelta {
		t.Fatalf("runExperiment is not deterministic: r1=%+v r2=%+v", r1, r2)
	}
	if r1.SignTest != r2.SignTest {
		t.Fatalf("sign test not deterministic: r1=%+v r2=%+v", r1.SignTest, r2.SignTest)
	}
	if len(r1.Pairs) != len(fixtures) {
		t.Fatalf("expected %d pairs, got %d", len(fixtures), len(r1.Pairs))
	}
}

// TestThesisDirectionModeled is the headline claim of this harness, kept honest by its name: the
// MODELED proxy points the direction the thesis predicts across the whole fixture corpus. This
// is not a live-model result (see README.md) -- it is a pinned regression on the modeled proxy
// itself, so a future edit to the cost constants or fixtures cannot silently flip the reported
// direction without a test failure calling it out.
func TestThesisDirectionModeled(t *testing.T) {
	r := runExperiment()
	if r.MeanDelta <= 0 {
		t.Fatalf("modeled mean delta (B-A) = %v, want > 0", r.MeanDelta)
	}
	if r.SignTest.Favoring_B != r.SignTest.NUsed || r.SignTest.NUsed == 0 {
		t.Fatalf("sign test: %d/%d pairs favor Arm B, want all", r.SignTest.Favoring_B, r.SignTest.NUsed)
	}
	if r.SignTest.PValue > 0.05 {
		t.Fatalf("sign test p-value = %v, want <= 0.05 given every fixture pair favors Arm B", r.SignTest.PValue)
	}
}

// TestSelfCheckReportPass runs the same selfcheck the CLI's -selfcheck flag runs and requires it
// to PASS on the committed fixture corpus -- the "go run . -selfcheck" spine this issue asks for,
// pinned as a test so a regression is caught in CI, not just by eyeballing stdout.
func TestSelfCheckReportPass(t *testing.T) {
	var buf bytes.Buffer
	if ok := selfCheckReport(&buf, runExperiment()); !ok {
		t.Fatalf("selfCheckReport returned false; output:\n%s", buf.String())
	}
	if !bytes.Contains(buf.Bytes(), []byte("SELFCHECK PASS")) {
		t.Fatalf("selfcheck output missing PASS line:\n%s", buf.String())
	}
}

// TestPrintHumanIncludesProvenance proves the human report cannot be read without hitting the
// MODELED/OFFLINE label -- the honesty requirement this whole experiment is built around.
func TestPrintHumanIncludesProvenance(t *testing.T) {
	var buf bytes.Buffer
	printHuman(&buf, runExperiment())
	if !bytes.Contains(buf.Bytes(), []byte("MODELED / OFFLINE PROXY")) {
		t.Fatalf("human report missing provenance banner:\n%s", buf.String())
	}
}
