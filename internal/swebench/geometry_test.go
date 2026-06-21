package swebench

import "testing"

func TestEstimateTokens(t *testing.T) {
	if EstimateTokens("") != 0 {
		t.Errorf("empty should be 0")
	}
	// a ~40-char sentence should land in a sane token range (not 0, not absurd)
	n := EstimateTokens("Fix the bug where the parser drops trailing commas.")
	if n < 6 || n > 30 {
		t.Errorf("token estimate out of sane range: %d", n)
	}
}

func TestDeriveGeometry(t *testing.T) {
	gm := DefaultGeometryModel()

	// difficulty-derived turns + problem folded into prefix
	in := Instance{InstanceID: "django__django-1", ProblemStatement: "make it work", Difficulty: "1-4hr"}
	g := gm.Derive(in)
	if g.Source != "difficulty" || g.Turns != 38 {
		t.Errorf("difficulty turns wrong: %+v", g)
	}
	if g.Prefix <= gm.BasePrefix || g.ProblemTokens == 0 {
		t.Errorf("problem not folded into prefix: %+v", g)
	}

	// unknown difficulty -> default, no problem statement -> base prefix only
	g2 := gm.Derive(Instance{InstanceID: "x__y-1"})
	if g2.Source != "default" || g2.Turns != gm.DefaultTurns || g2.Prefix != gm.BasePrefix {
		t.Errorf("default geometry wrong: %+v", g2)
	}

	// trajectory overrides difficulty
	gm.Trajectories = map[string]int{"django__django-1": 7}
	g3 := gm.Derive(in)
	if g3.Source != "trajectory" || g3.Turns != 7 {
		t.Errorf("trajectory override wrong: %+v", g3)
	}
}

func TestMaxContext(t *testing.T) {
	g := Geometry{Prefix: 100, Turns: 10, Decode: 5, Result: 15}
	if g.MaxContext() != 100+10*20 {
		t.Errorf("MaxContext = %d", g.MaxContext())
	}
}
