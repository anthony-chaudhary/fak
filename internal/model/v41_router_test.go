package model

import (
	"errors"
	"math"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
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

// v41OracleSharedAdd independently performs the routed-plus-shared combine the
// wrapper advertises: out[i] = routed[i] + shared[i] in increasing index order,
// written from scratch here so the expected bits do not depend on ffn.AddScaled.
func v41OracleSharedAdd(routed, shared []float32) []float32 {
	out := make([]float32, len(routed))
	for i := range routed {
		out[i] = routed[i] + shared[i]
	}
	return out
}

// TestV41SharedExpertAddScaledParity pins bit-for-bit equality between the
// migrated wrapper (which delegates its accumulation to ffn.AddScaled) and an
// independent original-arithmetic oracle, plus the signed-zero, single-output-
// allocation, input-ownership and error-order invariants the migration must
// preserve. It exercises the real v41SharedExpertAdd adapter its V4.1 forward and
// live-expert callers use, not a helper in isolation.
func TestV41SharedExpertAddScaledParity(t *testing.T) {
	cfg := v41TestCfg()

	t.Run("bits match original loop", func(t *testing.T) {
		routed := []float32{1.5, -2.25, 0.125, 3.0e30, -1e-30, 0, -0, 42, -7.75, 1e-45}
		shared := []float32{-0.5, 4.5, -0.125, 3.0e30, 1e-30, -0, 0, -42, -0.25, -1e-45}
		got, err := v41SharedExpertAdd(routed, shared, cfg)
		if err != nil {
			t.Fatalf("shared add: %v", err)
		}
		want := v41OracleSharedAdd(routed, shared)
		if len(got) != len(want) {
			t.Fatalf("len=%d want=%d", len(got), len(want))
		}
		for i := range want {
			if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
				t.Fatalf("out[%d] bits=%#08x want=%#08x (got=%v want=%v)",
					i, math.Float32bits(got[i]), math.Float32bits(want[i]), got[i], want[i])
			}
		}
	})

	t.Run("signed zero preserved", func(t *testing.T) {
		negZero := float32(math.Copysign(0, -1))
		posZero := float32(0)
		got, err := v41SharedExpertAdd([]float32{negZero, posZero}, []float32{negZero, negZero}, cfg)
		if err != nil {
			t.Fatalf("shared add: %v", err)
		}
		if math.Float32bits(got[0]) != math.Float32bits(negZero) {
			t.Fatalf("(-0)+(-0) bits=%#08x want=-0=%#08x", math.Float32bits(got[0]), math.Float32bits(negZero))
		}
		if math.Float32bits(got[1]) != math.Float32bits(posZero) {
			t.Fatalf("(+0)+(-0) bits=%#08x want=+0=%#08x", math.Float32bits(got[1]), math.Float32bits(posZero))
		}
	})

	t.Run("inputs unchanged and one fresh allocation", func(t *testing.T) {
		routed := []float32{1, 2, 3, 4, 5, 6, 7, 8}
		shared := []float32{8, 7, 6, 5, 4, 3, 2, 1}
		routedCopy := append([]float32(nil), routed...)
		sharedCopy := append([]float32(nil), shared...)
		got, err := v41SharedExpertAdd(routed, shared, cfg)
		if err != nil {
			t.Fatalf("shared add: %v", err)
		}
		for i := range routed {
			if math.Float32bits(routed[i]) != math.Float32bits(routedCopy[i]) {
				t.Fatalf("routed mutated at %d: %v want %v", i, routed[i], routedCopy[i])
			}
			if math.Float32bits(shared[i]) != math.Float32bits(sharedCopy[i]) {
				t.Fatalf("shared mutated at %d: %v want %v", i, shared[i], sharedCopy[i])
			}
		}
		// A fresh backing array: mutating the output must not alias an input.
		got[0] = -12345
		if routed[0] != routedCopy[0] || shared[0] != sharedCopy[0] {
			t.Fatal("output aliases an input operand")
		}

		allocs := testing.AllocsPerRun(50, func() {
			_, _ = v41SharedExpertAdd(routed, shared, cfg)
		})
		if allocs != 1 {
			t.Fatalf("shared add allocations=%v want 1", allocs)
		}
	})

	t.Run("error order and first non-finite result", func(t *testing.T) {
		nan := float32(math.NaN())
		inf := float32(math.Inf(1))

		// Input validation runs before any accumulation: routed is checked at
		// each index before shared, so a NaN in routed wins at the same index.
		_, err := v41SharedExpertAdd([]float32{nan, 1}, []float32{inf, 1}, cfg)
		var typed *v4RouteError
		if !errors.As(err, &typed) {
			t.Fatalf("err=%T %v want *v4RouteError", err, err)
		}
		if typed.Field != "routed" || typed.Reason != "non-finite value at 0" {
			t.Fatalf("field=%q reason=%q want routed at 0", typed.Field, typed.Reason)
		}

		// First non-finite *result* is reported in increasing index order by the
		// post-accumulation scan, after ffn.AddScaled has run.
		big := float32(math.MaxFloat32)
		_, err = v41SharedExpertAdd([]float32{1, big, 1}, []float32{1, big, 1}, cfg)
		if !errors.As(err, &typed) {
			t.Fatalf("err=%T %v want *v4RouteError", err, err)
		}
		if typed.Field != "shared_add" || typed.Reason != "non-finite result at 1" {
			t.Fatalf("field=%q reason=%q want shared_add at 1", typed.Field, typed.Reason)
		}
	})
}

// v41FullRouteNormalizationOracle keeps selection independent (insertion top-k)
// and stages the full profile's F32 denominator, division and scale separately.
// The thresholded scalar formula is independent from the stable production
// expression. This is not a GPU transcendental, denormal or reduction oracle.
func v41FullRouteNormalizationOracle(logits, bias []float32, topK int, scale float32) ([]int, []float64) {
	scores := make([]float32, len(logits))
	choice := make([]float32, len(logits))
	for i, z := range logits {
		// Independent thresholded softplus formula; staged tensor publication.
		softplus := z
		if z <= 20 {
			softplus = float32(math.Log1p(math.Exp(float64(z))))
		}
		scores[i] = float32(math.Sqrt(float64(softplus)))
		choice[i] = scores[i]
		if len(bias) != 0 {
			choice[i] += bias[i]
		}
	}
	picks := make([]int, 0, topK)
	for e := range scores {
		at := len(picks)
		for i, old := range picks {
			if choice[e] > choice[old] || (choice[e] == choice[old] && e < old) {
				at = i
				break
			}
		}
		if at < topK {
			picks = append(picks, 0)
			copy(picks[at+1:], picks[at:])
			picks[at] = e
			if len(picks) > topK {
				picks = picks[:topK]
			}
		}
	}
	var sum float32
	for _, e := range picks {
		sum = float32(sum + scores[e])
	}
	denominator := float32(sum + float32(1e-20))
	weights := make([]float64, len(picks))
	for i, e := range picks {
		divided := float32(scores[e] / denominator)
		weights[i] = float64(float32(divided * scale))
	}
	return picks, weights
}

// Runtime estimate only; unmeasured.
// fak-test:runtime fast est=100ms lane=default
func TestV41FullRouteNormalizationEpsilon(t *testing.T) {
	t.Parallel()
	cfg := v41TestCfg()
	for _, z := range []float32{-10000, -120, -104, -103, -96, -90, 0, 20, 21} {
		logits, bias := make([]float32, cfg.Experts), make([]float32, cfg.Experts)
		for i := range logits {
			logits[i] = z
		}
		for i := 0; i < cfg.TopK; i++ {
			bias[11+i] = float32(i + 1)
		}
		beforeLogits, beforeBias := append([]float32(nil), logits...), append([]float32(nil), bias...)
		got, err := v41RouteForGeometry(logits, bias, cfg, true)
		if err != nil {
			t.Fatalf("full logits=%g: %v", z, err)
		}
		ids, weights := v41FullRouteNormalizationOracle(logits, bias, cfg.TopK, cfg.RouteScale)
		var total float32
		for i, pick := range got {
			if pick.expert != ids[i] || pick.expert != 16-i || math.Float32bits(pick.weight) != math.Float32bits(float32(weights[i])) {
				t.Fatalf("logit=%g pick[%d]=%v want expert=%d weight=%g", z, i, pick, ids[i], weights[i])
			}
			total += pick.weight
		}
		if z <= -104 && total != 0 {
			t.Fatal("zero-score row acquired routing mass")
		}
		if z == -96 && !(total > 0 && total < 1) {
			t.Fatalf("tiny-positive epsilon witness vacuous: sum=%g", total)
		}
		legacy, legacyErr := v41Route(logits, bias, cfg)
		reduced, reducedErr := v41RouteForGeometry(logits, bias, cfg, false)
		if z == -10000 {
			if legacyErr == nil || reducedErr == nil {
				t.Fatal("legacy zero-sum refusal changed")
			}
		} else if legacyErr != nil || reducedErr != nil || !reflect.DeepEqual(legacy, reduced) {
			t.Fatal("reduced routing arithmetic changed")
		} else if z == -96 && reflect.DeepEqual(got, legacy) {
			t.Fatal("epsilon has no observable effect")
		}
		if !reflect.DeepEqual(logits, beforeLogits) || !reflect.DeepEqual(bias, beforeBias) {
			t.Fatal("router mutated borrowed operands")
		}
	}
	for _, invalid := range []float32{float32(math.NaN()), float32(math.Inf(1))} {
		logits := make([]float32, cfg.Experts)
		logits[0] = invalid
		if _, err := v41RouteForGeometry(logits, nil, cfg, true); err == nil {
			t.Fatal("full router admitted nonfinite logit")
		}
	}
}

// Actual full Prefill and both incremental branches must accept finite zero-score
// rows, using -120 logits whose legacy F64 softplus/sqrt remains nonzero.
// Callback gate logits are F32; unique correction biases fix the winners.
// A handled down callback observes the zero weighted operand before contraction.
// Runtime estimate only; unmeasured.
// fak-test:runtime medium est=3s lane=default
func TestV41FullRouteZeroScoresLiveOwners(t *testing.T) {
	for _, role := range []bool{false, true} {
		t.Run("role="+itoa(boolToIntV41Expert(role)), func(t *testing.T) {
			m := v41IncrementalExpertFixture(t, true, false)
			if !role {
				m.Cfg.DeepSeekV41.CompressRatios = []int{0}
				m.Cfg.DeepSeekV41.IndexSourceLayerIDs = nil
				m.Cfg.DeepSeekV41.KVSourceLayerIDs = nil
			}
			if full, err := v41ForwardGeometry(m.Cfg); err != nil || !full {
				t.Fatal("fixture is not full")
			}
			for l := 0; l < m.Cfg.NumLayers; l++ {
				bias := make([]float32, m.Cfg.NumExperts)
				for i := 0; i < m.Cfg.NumExpertsPerTok; i++ {
					bias[11+i] = float32(i + 1)
				}
				v41WriteTensorF32(t, m, layerName(l, "ffn.gate.e_score_correction_bias"), bias)
			}
			s := v41EngProjSession(t, m, compute.Default())
			defer s.Close()
			state := s.v41State()
			gateCalls, downCalls := 0, 0
			state.denseProjection = func(layer int, leaf string, input []float32, out, in, rows int) ([]float32, v41DenseProjectionOutcome, error) {
				if leaf != "ffn.gate.weight" {
					return nil, v41ProjectionDeclined, nil
				}
				gateCalls += rows
				logits := make([]float32, rows*out)
				for i := range logits {
					logits[i] = -120
				}
				return logits, v41ProjectionHandled, nil
			}
			state.expertGateUp = func(layer int, stem string, input []float32) ([]float32, v41ExpertGateUpOutcome, error) {
				fused := make([]float32, m.Cfg.MoEIntermediateSize)
				for i := range fused {
					fused[i] = 1
				}
				return fused, v41GateUpHandled, nil
			}
			state.expertDown = func(layer int, stem string, input []float32) ([]float32, v41ExpertDownOutcome, error) {
				downCalls++
				for _, value := range input {
					if value != 0 {
						t.Fatalf("nonzero weighted down operand=%g", value)
					}
				}
				return make([]float32, m.Cfg.HiddenSize), v41DownHandled, nil
			}
			for phase := 0; phase < 2; phase++ {
				beforeGate, beforeDown := gateCalls, downCalls
				tokens := 2
				var got []float32
				if phase == 0 {
					got = s.Prefill([]int{1, 2})
				} else {
					tokens = 1
					got = s.Step(3)
				}
				if len(got) == 0 || !s.v41IncrementalEligible() {
					t.Fatal("full route did not retain eligible continuation")
				}
				for _, value := range got {
					if !finite32(value) {
						t.Fatal("nonfinite live logits")
					}
				}
				if gateCalls-beforeGate != tokens*m.Cfg.NumLayers || downCalls-beforeDown != tokens*m.Cfg.NumLayers*m.Cfg.NumExpertsPerTok {
					t.Fatalf("phase=%d callback counts=%d/%d", phase, gateCalls-beforeGate, downCalls-beforeDown)
				}
			}
		})
	}
}

// Runtime estimate only; unmeasured.
// fak-test:runtime medium est=2s lane=default
func TestV41FullRouteSoftplusPublication(t *testing.T) {
	t.Parallel()
	cfg := v41TestCfg()
	logits, bias := make([]float32, cfg.Experts), make([]float32, cfg.Experts)
	for i := range logits {
		logits[i] = -120
	}
	for i := 0; i < cfg.TopK; i++ {
		bias[11+i] = float32(i + 1)
	}
	full, err := v41RouteForGeometry(logits, bias, cfg, true)
	if err != nil {
		t.Fatal(err)
	}
	reduced, err := v41Route(logits, bias, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if v41SqrtSoftplus(-120) <= 0 {
		t.Fatal("legacy underflow discriminator became vacuous")
	}
	for i := range full {
		if full[i].weight != 0 || reduced[i].weight <= 0 {
			t.Fatal("missing F32 softplus publication or changed reduced route")
		}
	}
	for i := range logits {
		logits[i] = math.MaxFloat32
	}
	extreme, err := v41RouteForGeometry(logits, nil, cfg, true)
	if err != nil {
		t.Fatalf("finite positive extreme refused: %v", err)
	}
	for _, pick := range extreme {
		if !finite32(pick.weight) || pick.weight <= 0 {
			t.Fatal("positive extreme lost finite routing mass")
		}
	}
	// Actual routed consumer: unit fused activation and identity-like down expose
	// the spurious nonzero route mass even after the BF16 weighted-input cast.
	m := v41IncrementalExpertFixture(t, true, false)
	input := make([]float32, m.Cfg.HiddenSize)
	seen := false
	_, err = m.v41FullRoutedExpert(0, "ffn.experts.16", input, full[0].weight, m.Cfg,
		func(int, string, []float32) ([]float32, v41ExpertGateUpOutcome, error) {
			x := make([]float32, m.Cfg.MoEIntermediateSize)
			for i := range x {
				x[i] = 1
			}
			return x, v41GateUpHandled, nil
		},
		func(_ int, _ string, x []float32) ([]float32, v41ExpertDownOutcome, error) {
			seen = true
			for _, v := range x {
				if v != 0 {
					t.Fatal("full score injected nonzero weighted operand")
				}
			}
			return make([]float32, m.Cfg.HiddenSize), v41DownHandled, nil
		}, nil, nil, false)
	if err != nil || !seen {
		t.Fatalf("live routed consumer unseen or failed: %v", err)
	}
	// Reconstruct the previous full score + epsilon, rather than reduced weights:
	// the omitted publication produces a small positive BF16 operand, not zero.
	legacyScore := v41SqrtSoftplus(-120)
	var sum float32
	for i := 0; i < cfg.TopK; i++ {
		sum += legacyScore
	}
	wrong := float32(legacyScore/float32(sum+float32(1e-20))) * cfg.RouteScale
	if v41RoundBF16(wrong) == 0 {
		t.Fatal("weighted BF16 consumer discriminator is vacuous")
	}
}
