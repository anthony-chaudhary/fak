package cvregress

import "testing"

func testPowerSpec() PowerSpec {
	return PowerSpec{EffectSize: .5, StdDev: 1, Alpha: .05, Power: .8, Tails: 2, Seed: 7}
}

func TestPowerAnalysisUnderpoweredIsInconclusive(t *testing.T) {
	s := testPowerSpec()
	n, err := RequiredN(s)
	if err != nil {
		t.Fatal(err)
	}
	if n < 2 {
		t.Fatalf("bad required n %d", n)
	}
	r := Assess(s, n-1)
	v, ok, conclusive := StochasticVerdict(r, false)
	if conclusive || v != "INSUFFICIENT" {
		t.Fatalf("underpowered run passed: report=%+v verdict=%q ok=%v conclusive=%v", r, v, ok, conclusive)
	}
	r = Assess(s, n)
	v, ok, conclusive = StochasticVerdict(r, false)
	if !ok || !conclusive || v != "OK" {
		t.Fatalf("powered faithful run failed: %+v %q", r, v)
	}
}

func TestPowerSimulationReplaysAndMeetsTargets(t *testing.T) {
	s := testPowerSpec()
	n, err := RequiredN(s)
	if err != nil {
		t.Fatal(err)
	}
	a := Simulate(s, n, 5000, .08)
	b := Simulate(s, n, 5000, .08)
	if a != b {
		t.Fatalf("seeded simulation not replayable: %+v %+v", a, b)
	}
	if !a.MeetsTargets {
		t.Fatalf("simulation did not meet declared error/power targets: %+v", a)
	}
}

func TestPowerSpecRejectsMissingEvidence(t *testing.T) {
	for _, s := range []PowerSpec{{}, {EffectSize: .5, StdDev: 1, Alpha: .05, Power: .8}} {
		if _, err := RequiredN(s); err == nil {
			t.Fatalf("invalid/missing spec accepted: %+v", s)
		}
	}
}
