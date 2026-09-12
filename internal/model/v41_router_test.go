package model

import (
	"errors"
	"math"
	"reflect"
	"testing"
)

// v41OracleScore independently recomputes sqrt(softplus(z)) from the formula in
// the contract. It deliberately does NOT call any production router helper.
func v41OracleScore(z float32) float64 {
	zf := float64(z)
	softplus := math.Max(zf, 0) + math.Log1p(math.Exp(-math.Abs(zf)))
	return math.Sqrt(softplus)
}

// v41OracleSelectionScore is the score the production top-k ranks by:
// unbiased sqrt(softplus) plus the per-expert correction bias.
func v41OracleSelectionScore(z, bias float32) float64 {
	return v41OracleScore(z) + float64(bias)
}

// v41OracleTopK is an insertion-based top-k over the full width that ranks by
// descending selection score with the lower expert index winning ties. It is
// written from scratch here so the expected expert set is independent of the
// production sort.
func v41OracleTopK(logits, correctionBias []float32, k int) []int {
	selected := make([]int, 0, k)
	for i := range logits {
		bi := float32(0)
		if len(correctionBias) != 0 {
			bi = correctionBias[i]
		}
		si := v41OracleSelectionScore(logits[i], bi)
		pos := len(selected)
		for j, e := range selected {
			be := float32(0)
			if len(correctionBias) != 0 {
				be = correctionBias[e]
			}
			se := v41OracleSelectionScore(logits[e], be)
			if si > se || (si == se && i < e) {
				pos = j
				break
			}
		}
		if pos < k {
			selected = append(selected, 0)
			copy(selected[pos+1:], selected[pos:])
			selected[pos] = i
			if len(selected) > k {
				selected = selected[:k]
			}
		}
	}
	return selected
}

func v41TestCfg() v41RouterConfig { return v41DefaultRouterConfig() }

func v41WantClosed(t *testing.T, err error, field string) {
	t.Helper()
	if err == nil {
		t.Fatalf("expected *v4RouteError for %s, got nil", field)
	}
	var typed *v4RouteError
	if !errors.As(err, &typed) {
		t.Fatalf("error %T %v is not *v4RouteError", err, err)
	}
}

func v41Close(got float32, want float64) bool {
	return math.Abs(float64(got)-want) <= 1e-6
}

func TestV41RouterDefaultGeometry(t *testing.T) {
	want := v41RouterConfig{Experts: 384, TopK: 6, SharedCount: 1, RouteScale: 1.5}
	if got := v41DefaultRouterConfig(); !reflect.DeepEqual(got, want) {
		t.Fatalf("default config=%#v want=%#v", got, want)
	}
	consts := map[string]struct{ got, want int }{
		"V41RouterExperts":     {V41RouterExperts, 384},
		"V41RouterTopK":        {V41RouterTopK, 6},
		"V41RouterSharedCount": {V41RouterSharedCount, 1},
		"V41RouterMoEWidth":    {V41RouterMoEWidth, 2304},
	}
	for name, tc := range consts {
		if tc.got != tc.want {
			t.Fatalf("%s=%d want=%d", name, tc.got, tc.want)
		}
	}
	if V41RouterRouteScale != 1.5 {
		t.Fatalf("V41RouterRouteScale=%v want=1.5", V41RouterRouteScale)
	}
}

func TestV41RouterConfigFromOfficialConfig(t *testing.T) {
	_, cfg := readDeepSeekV41Config(t)
	got, err := v41RouterConfigFromConfig(cfg)
	if err != nil {
		t.Fatalf("official config rejected: %v", err)
	}
	want := v41RouterConfig{Experts: 384, TopK: 6, SharedCount: 1, RouteScale: 1.5}
	if got != want {
		t.Fatalf("from config=%#v want=%#v", got, want)
	}

	mutations := map[string]func(*Config){
		"NumExperts":          func(c *Config) { c.NumExperts = 383 },
		"NumExpertsPerTok":    func(c *Config) { c.NumExpertsPerTok = 5 },
		"NSharedExperts":      func(c *Config) { c.NSharedExperts = 2 },
		"RoutedScalingFactor": func(c *Config) { c.RoutedScalingFactor = 2.0 },
	}
	for name, mutate := range mutations {
		t.Run(name, func(t *testing.T) {
			mutated := cfg
			mutate(&mutated)
			_, err := v41RouterConfigFromConfig(mutated)
			v41WantClosed(t, err, name)
		})
	}
}

func TestV41RouterTop6SelectionMatchesIndependentOracle(t *testing.T) {
	cfg := v41TestCfg()
	logits := make([]float32, V41RouterExperts)
	bias := make([]float32, V41RouterExperts)
	for i := range logits {
		logits[i] = -8
	}
	// Natural top block, exactly six-way tied on selection score.
	for i := 100; i <= 105; i++ {
		logits[i] = 5
	}
	// Bias lifts expert 200 above the natural block; without bias it would not
	// be selected, which proves selection uses score+bias.
	logits[200] = 1
	bias[200] = 3

	picks, err := v41Route(logits, bias, cfg)
	if err != nil {
		t.Fatalf("route: %v", err)
	}
	if len(picks) != V41RouterTopK {
		t.Fatalf("picks=%d want=%d", len(picks), V41RouterTopK)
	}

	order := v41OracleTopK(logits, bias, V41RouterTopK)
	wantExperts := []int{200, 100, 101, 102, 103, 104}
	if !reflect.DeepEqual(order, wantExperts) {
		t.Fatalf("independent oracle order=%v want=%v", order, wantExperts)
	}
	for i, p := range picks {
		if p.expert != order[i] {
			t.Fatalf("pick[%d].expert=%d want=%d (oracle=%v)", i, p.expert, order[i], order)
		}
	}

	// Bias must have changed selection versus pure unbiased score.
	pure := v41OracleTopK(logits, nil, V41RouterTopK)
	if reflect.DeepEqual(pure, order) {
		t.Fatalf("bias did not change selection: pure=%v biased=%v", pure, order)
	}
	found200 := false
	for _, e := range pure {
		if e == 200 {
			found200 = true
		}
	}
	if found200 {
		t.Fatalf("expert 200 unexpectedly in unbiased top-k %v", pure)
	}

	var sum float64
	for _, e := range order {
		sum += v41OracleScore(logits[e])
	}
	for i, p := range picks {
		want := v41OracleScore(logits[order[i]]) / sum * 1.5
		if !v41Close(p.weight, want) {
			t.Fatalf("pick[%d].weight=%v want=%v", i, p.weight, want)
		}
	}

	// Returned picks must be sorted by descending selection score.
	prev := math.Inf(1)
	for _, p := range picks {
		s := v41OracleSelectionScore(logits[p.expert], bias[p.expert])
		if s > prev {
			t.Fatalf("picks not descending by selection score: pick expert=%d score=%v > prev=%v", p.expert, s, prev)
		}
		prev = s
	}
}

func TestV41RouterSelectedWeightNormalizationAndRouteScale(t *testing.T) {
	cfg := v41TestCfg()
	// Exactly six controlled experts are selected; the rest sit far below the
	// selection threshold. The oracle finds them independently.
	logits := make([]float32, V41RouterExperts)
	for i := range logits {
		logits[i] = -30
	}
	controlled := map[int]float32{5: 2.0, 40: 0.5, 77: -1.0, 150: 3.0, 260: -2.0, 383: 1.0}
	for i, z := range controlled {
		logits[i] = z
	}
	picks, err := v41Route(logits, nil, cfg)
	if err != nil {
		t.Fatalf("route: %v", err)
	}
	if len(picks) != V41RouterTopK {
		t.Fatalf("picks=%d want=%d", len(picks), V41RouterTopK)
	}

	order := v41OracleTopK(logits, nil, V41RouterTopK)
	if len(order) != V41RouterTopK {
		t.Fatalf("oracle picks=%d want=%d", len(order), V41RouterTopK)
	}
	var sumScore, sumWeight float64
	score := make(map[int]float64, V41RouterTopK)
	for _, e := range order {
		score[e] = v41OracleScore(logits[e])
		sumScore += score[e]
	}
	for _, p := range picks {
		sumWeight += float64(p.weight)
	}
	if math.Abs(sumWeight-float64(cfg.RouteScale)) > 1e-6 {
		t.Fatalf("sum(weights)=%v want RouteScale=%v", sumWeight, cfg.RouteScale)
	}
	for i, p := range picks {
		if p.expert != order[i] {
			t.Fatalf("pick[%d].expert=%d want=%d (oracle=%v)", i, p.expert, order[i], order)
		}
		want := score[p.expert] / sumScore * float64(cfg.RouteScale)
		if !v41Close(p.weight, want) {
			t.Fatalf("expert %d weight=%v want=%v", p.expert, p.weight, want)
		}
	}
}

func TestV41RouterSharedExpertAddedOnEveryToken(t *testing.T) {
	cfg := v41TestCfg()
	routed := []float32{1, 2, 3, 4}
	shared := []float32{0.5, -0.5, 2, -1}
	got, err := v41SharedExpertAdd(routed, shared, cfg)
	if err != nil {
		t.Fatalf("shared add: %v", err)
	}
	want := []float32{1.5, 1.5, 5, 3}
	for i := range want {
		if math.Abs(float64(got[i])-float64(want[i])) > 1e-6 {
			t.Fatalf("token1[%d]=%v want=%v", i, got[i], want[i])
		}
	}

	shared2 := []float32{-1, 3, 0, 8}
	got2, err := v41SharedExpertAdd(routed, shared2, cfg)
	if err != nil {
		t.Fatalf("shared add token2: %v", err)
	}
	want2 := []float32{0, 5, 3, 12}
	for i := range want2 {
		if math.Abs(float64(got2[i])-float64(want2[i])) > 1e-6 {
			t.Fatalf("token2[%d]=%v want=%v", i, got2[i], want2[i])
		}
	}

	// All-zero routed accumulator: the shared term must survive alone.
	zeros := []float32{0, 0, 0, 0}
	got3, err := v41SharedExpertAdd(zeros, shared, cfg)
	if err != nil {
		t.Fatalf("shared add zeros: %v", err)
	}
	for i := range shared {
		if math.Abs(float64(got3[i])-float64(shared[i])) > 1e-6 {
			t.Fatalf("zeros[%d]=%v want=%v", i, got3[i], shared[i])
		}
	}
}

func TestV41RouterSharedExpertRejectsMalformed(t *testing.T) {
	base := v41TestCfg()

	noShared := base
	noShared.SharedCount = 0
	if _, err := v41SharedExpertAdd([]float32{1, 2}, []float32{1, 2}, noShared); err == nil {
		t.Fatal("SharedCount==0 accepted")
	} else {
		v41WantClosed(t, err, "SharedCount==0")
	}

	if _, err := v41SharedExpertAdd([]float32{1, 2, 3}, []float32{1, 2}, base); err == nil {
		t.Fatal("length mismatch accepted")
	} else {
		v41WantClosed(t, err, "length mismatch")
	}

	if _, err := v41SharedExpertAdd([]float32{}, []float32{}, base); err == nil {
		t.Fatal("empty vectors accepted")
	} else {
		v41WantClosed(t, err, "empty")
	}

	nan := float32(math.NaN())
	if _, err := v41SharedExpertAdd([]float32{nan, 1}, []float32{1, 1}, base); err == nil {
		t.Fatal("NaN routed accepted")
	} else {
		v41WantClosed(t, err, "NaN routed")
	}
	if _, err := v41SharedExpertAdd([]float32{1, 1}, []float32{1, float32(math.Inf(1))}, base); err == nil {
		t.Fatal("Inf shared accepted")
	} else {
		v41WantClosed(t, err, "Inf shared")
	}
}

func TestV41RouterInvalidInputsFailExplicitly(t *testing.T) {
	base := v41TestCfg()
	valid384 := make([]float32, V41RouterExperts)

	t.Run("logits width 383", func(t *testing.T) {
		_, err := v41Route(make([]float32, 383), nil, base)
		v41WantClosed(t, err, "width 383")
	})
	t.Run("logits width 385", func(t *testing.T) {
		_, err := v41Route(make([]float32, 385), nil, base)
		v41WantClosed(t, err, "width 385")
	})
	t.Run("nil logits", func(t *testing.T) {
		_, err := v41Route(nil, nil, base)
		v41WantClosed(t, err, "nil")
	})
	t.Run("empty logits", func(t *testing.T) {
		_, err := v41Route([]float32{}, nil, base)
		v41WantClosed(t, err, "empty")
	})

	cfgMutations := map[string]func(*v41RouterConfig){
		"Experts":       func(c *v41RouterConfig) { c.Experts = 383 },
		"TopK":          func(c *v41RouterConfig) { c.TopK = 5 },
		"SharedCount":   func(c *v41RouterConfig) { c.SharedCount = 0 },
		"RouteScale0":   func(c *v41RouterConfig) { c.RouteScale = 0 },
		"RouteScaleNaN": func(c *v41RouterConfig) { c.RouteScale = float32(math.NaN()) },
	}
	for name, mutate := range cfgMutations {
		t.Run("cfg "+name, func(t *testing.T) {
			cfg := base
			mutate(&cfg)
			_, err := v41Route(valid384, nil, cfg)
			v41WantClosed(t, err, "cfg "+name)
		})
	}

	t.Run("non-finite logit NaN", func(t *testing.T) {
		logits := append([]float32(nil), valid384...)
		logits[7] = float32(math.NaN())
		_, err := v41Route(logits, nil, base)
		v41WantClosed(t, err, "NaN logit")
	})
	t.Run("non-finite logit +Inf", func(t *testing.T) {
		logits := append([]float32(nil), valid384...)
		logits[11] = float32(math.Inf(1))
		_, err := v41Route(logits, nil, base)
		v41WantClosed(t, err, "+Inf logit")
	})
	t.Run("non-finite bias", func(t *testing.T) {
		bias := make([]float32, V41RouterExperts)
		bias[13] = float32(math.NaN())
		_, err := v41Route(valid384, bias, base)
		v41WantClosed(t, err, "NaN bias")
	})

	// softplus underflows to zero for very negative finite logits, so every
	// selected unbiased score is 0 and the normalization sum is <= 0.
	t.Run("normalization sum <= 0", func(t *testing.T) {
		logits := make([]float32, V41RouterExperts)
		for i := range logits {
			logits[i] = -1e38
		}
		_, err := v41Route(logits, nil, base)
		v41WantClosed(t, err, "sum <= 0")
	})
}
