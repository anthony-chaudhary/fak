package swebench

import "testing"

func TestPrefillTokensFormula(t *testing.T) {
	// P=100, T=3, D=10, R=20, C=2
	g := Geometry{Prefix: 100, Turns: 3, Decode: 10, Result: 20}
	a, b, c := PrefillTokens(g, 2)
	// A per worker = (100) + (100+30) + (100+60) = 100+130+160 = 390; ×2 = 780
	if a != 780 {
		t.Errorf("A = %d want 780", a)
	}
	// B per worker = P + (T-1)*R = 100 + 2*20 = 140; ×2 = 280
	if b != 280 {
		t.Errorf("B = %d want 280", b)
	}
	// C = P + C*(T-1)*R = 100 + 2*40 = 180
	if c != 180 {
		t.Errorf("C = %d want 180", c)
	}
}

func TestPrefillTokensSingleWorkerBEqualsC(t *testing.T) {
	// at C=1 the cross-worker reuse vanishes: B == C; turn-tax (A/B) remains.
	g := Geometry{Prefix: 2500, Turns: 38, Decode: 200, Result: 400}
	a, b, c := PrefillTokens(g, 1)
	if b != c {
		t.Errorf("at workers=1, B(%d) should equal C(%d)", b, c)
	}
	if a <= b {
		t.Errorf("turn-tax should make A(%d) > B(%d)", a, b)
	}
}

func TestAggregatePrefillRatios(t *testing.T) {
	geoms := []Geometry{
		{Prefix: 100, Turns: 3, Decode: 10, Result: 20},
		{Prefix: 200, Turns: 5, Decode: 10, Result: 20},
	}
	agg := AggregatePrefill(geoms, 4)
	if agg.AOverC <= agg.BOverC {
		t.Errorf("A/C(%f) should exceed B/C(%f)", agg.AOverC, agg.BOverC)
	}
	if agg.AOverB <= 1 {
		t.Errorf("turn-tax A/B(%f) should exceed 1", agg.AOverB)
	}
	if agg.Instances != 2 || agg.Workers != 4 {
		t.Errorf("agg meta wrong: %+v", agg)
	}
}

func TestDescribeShape(t *testing.T) {
	d := NewDataset([]Instance{
		{InstanceID: "a__b-1", Difficulty: "<15min", ProblemStatement: "x"},
		{InstanceID: "a__b-2", Difficulty: "1-4hr", ProblemStatement: "y"},
		{InstanceID: "a__b-3"},
	})
	s := Describe(d, DefaultGeometryModel(), []int{1, 4})
	if s.Instances != 3 {
		t.Fatalf("instances %d", s.Instances)
	}
	if s.DifficultyDist["<15min"] != 1 || s.DifficultyDist["unknown"] != 1 {
		t.Errorf("difficulty dist wrong: %v", s.DifficultyDist)
	}
	if s.GeometrySources["difficulty"] != 2 || s.GeometrySources["default"] != 1 {
		t.Errorf("geometry sources wrong: %v", s.GeometrySources)
	}
	if len(s.Prefill) != 2 || s.Prefill[1].Workers != 4 {
		t.Errorf("prefill sweep wrong: %+v", s.Prefill)
	}
}
