package compute

import (
	"errors"
	"math"
	"sort"
	"testing"
)

const (
	mffTokens  = 5
	mffExperts = 6
	mffTopK    = 2
	mffHidden  = 4
	mffInter   = 16
)

// Token 2 ties experts 1 and 4 at the top; token 4 ties three experts.
var mffLogits = []float32{
	0.5, -1.0, 2.0, 0.25, 1.5, -0.5,
	3.0, 0.0, -2.0, 1.0, 0.0, 2.5,
	-1.0, 2.0, 0.5, 0.0, 2.0, -3.0,
	0.0, 0.0, 0.0, 4.0, -1.0, 1.0,
	1.0, 1.0, -0.5, 1.0, 0.0, -4.0,
}

func mffConfig() MoeFrontConfig {
	return MoeFrontConfig{
		NumExperts: mffExperts,
		TopK:       mffTopK,
		Hidden:     mffHidden,
		Inter:      mffInter,
		Scoring:    MoeFrontSoftmax,
		TileSize:   4,
		Activation: func(v float32) float32 { return float32(math.Max(0, float64(v))) },
	}
}

func mffWeights() (x, fc1, fc2, router []float32) {
	x = make([]float32, mffTokens*mffHidden)
	for i := range x {
		x[i] = float32((i*7)%11-5) / 4
	}
	fc1 = make([]float32, mffExperts*mffInter*mffHidden)
	for i := range fc1 {
		fc1[i] = float32((i*13)%17-8) / 8
	}
	fc2 = make([]float32, mffExperts*mffHidden*mffInter)
	for i := range fc2 {
		fc2[i] = float32((i*5)%9-4) / 8
	}
	router = make([]float32, mffExperts*mffHidden)
	for i := range router {
		router[i] = float32((i*3)%7-3) / 2
	}
	return x, fc1, fc2, router
}

// mffOracleRoute scores with the unshifted softmax/sigmoid formula and picks
// top-k with a full stable sort, independent of the selection loop under test.
func mffOracleRoute(logits []float32, tokens int, cfg MoeFrontConfig) ([]int32, []float32) {
	ids := make([]int32, 0, tokens*cfg.TopK)
	scales := make([]float32, 0, tokens*cfg.TopK)
	for t := 0; t < tokens; t++ {
		row := logits[t*cfg.NumExperts : (t+1)*cfg.NumExperts]
		scores := make([]float64, len(row))
		var sum float64
		for e, v := range row {
			if cfg.Scoring == MoeFrontSigmoid {
				scores[e] = 1 / (1 + math.Exp(-float64(v)))
			} else {
				scores[e] = math.Exp(float64(v))
				sum += scores[e]
			}
		}
		if cfg.Scoring == MoeFrontSoftmax {
			for e := range scores {
				scores[e] /= sum
			}
		}
		idx := make([]int, len(row))
		for e := range idx {
			idx[e] = e
		}
		sort.SliceStable(idx, func(a, b int) bool { return scores[idx[a]] > scores[idx[b]] })
		var picked float64
		for _, e := range idx[:cfg.TopK] {
			picked += scores[e]
		}
		for _, e := range idx[:cfg.TopK] {
			s := scores[e]
			if cfg.Renormalize {
				s /= picked
			}
			ids = append(ids, int32(e))
			scales = append(scales, float32(s))
		}
	}
	return ids, scales
}

// mffOracleForward is the dense token-major reference: no sort, no tables, no
// permutation.
func mffOracleForward(t *testing.T, x, fc1, fc2 []float32, ids []int32, scales []float32, tokens int, cfg MoeFrontConfig) []float32 {
	t.Helper()
	out := make([]float32, tokens*cfg.Hidden)
	for tok := 0; tok < tokens; tok++ {
		for k := 0; k < cfg.TopK; k++ {
			e := int(ids[tok*cfg.TopK+k])
			h := make([]float32, cfg.Inter)
			for i := range h {
				var acc float32
				for j := 0; j < cfg.Hidden; j++ {
					acc += fc1[(e*cfg.Inter+i)*cfg.Hidden+j] * x[tok*cfg.Hidden+j]
				}
				h[i] = cfg.Activation(acc)
			}
			if cfg.QuantizeFC1 {
				p, s, g, err := QuantizeNVFP4Reference(1, cfg.Inter, h)
				if err != nil {
					t.Fatal(err)
				}
				if h, err = DecodeNVFP4Reference(1, cfg.Inter, p, s, g); err != nil {
					t.Fatal(err)
				}
			}
			for j := 0; j < cfg.Hidden; j++ {
				var acc float32
				for i := 0; i < cfg.Inter; i++ {
					acc += fc2[(e*cfg.Hidden+j)*cfg.Inter+i] * h[i]
				}
				out[tok*cfg.Hidden+j] += scales[tok*cfg.TopK+k] * acc
			}
		}
	}
	return out
}

func mffClose(t *testing.T, what string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length %d want %d", what, len(got), len(want))
	}
	for i := range got {
		if d := math.Abs(float64(got[i] - want[i])); d > 1e-5*(1+math.Abs(float64(want[i]))) {
			t.Fatalf("%s[%d] = %v want %v", what, i, got[i], want[i])
		}
	}
}

func TestMoeFrontFusionRouteMatchesOracle(t *testing.T) {
	for _, tc := range []struct {
		name    string
		scoring MoeFrontScoring
		renorm  bool
	}{
		{"softmax", MoeFrontSoftmax, false},
		{"softmax-renorm", MoeFrontSoftmax, true},
		{"sigmoid-renorm", MoeFrontSigmoid, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := mffConfig()
			cfg.Scoring, cfg.Renormalize = tc.scoring, tc.renorm
			route, err := MoeFrontRouteWithSort(mffLogits, mffTokens, cfg)
			if err != nil {
				t.Fatal(err)
			}
			wantIDs, wantScales := mffOracleRoute(mffLogits, mffTokens, cfg)
			for i := range wantIDs {
				if route.ExpertIDs[i] != wantIDs[i] {
					t.Fatalf("ids[%d] = %d want %d (all %v want %v)", i, route.ExpertIDs[i], wantIDs[i], route.ExpertIDs, wantIDs)
				}
			}
			mffClose(t, "scales", route.Scales, wantScales)
			if tc.renorm {
				for tok := 0; tok < mffTokens; tok++ {
					s := route.Scales[tok*mffTopK] + route.Scales[tok*mffTopK+1]
					if math.Abs(float64(s)-1) > 1e-6 {
						t.Fatalf("token %d renormalized scales sum to %v", tok, s)
					}
				}
			}
		})
	}
}

func TestMoeFrontFusionTieBreaksLowerExpert(t *testing.T) {
	route, err := MoeFrontRouteWithSort(mffLogits, mffTokens, mffConfig())
	if err != nil {
		t.Fatal(err)
	}
	// Token 2: experts 1 and 4 tie at 2.0. Token 4: experts 0, 1, 3 tie at 1.0.
	if got := route.ExpertIDs[2*mffTopK : 3*mffTopK]; got[0] != 1 || got[1] != 4 {
		t.Fatalf("token 2 tie picked %v want [1 4]", got)
	}
	if got := route.ExpertIDs[4*mffTopK : 5*mffTopK]; got[0] != 0 || got[1] != 1 {
		t.Fatalf("token 4 tie picked %v want [0 1]", got)
	}
}

func TestMoeFrontFusionSortTablesSinglePass(t *testing.T) {
	cfg := mffConfig()
	route, err := MoeFrontRouteWithSort(mffLogits, mffTokens, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if !route.Tables.Describes(route.ExpertIDs, cfg.TileSize) {
		t.Fatal("routing tables do not describe the ids routing produced")
	}
	a := route.Tables.Align
	for e := 0; e < mffExperts; e++ {
		want := map[int32]bool{}
		for i, id := range route.ExpertIDs {
			if int(id) == e {
				want[int32(i)] = true
			}
		}
		got := map[int32]bool{}
		for p := a.Cumsum[e]; p < a.Cumsum[e+1]; p++ {
			if v := a.SortedTokenIDs[p]; v != a.Numel {
				got[v] = true
			}
			if int(a.ExpertIDs[int(p)/cfg.TileSize]) != e {
				t.Fatalf("slot %d in expert %d region has tile owner %d", p, e, a.ExpertIDs[int(p)/cfg.TileSize])
			}
		}
		if len(got) != len(want) {
			t.Fatalf("expert %d region %v want %v", e, got, want)
		}
		for v := range want {
			if !got[v] {
				t.Fatalf("expert %d region missing expanded row %d", e, v)
			}
		}
	}
}

func TestMoeFrontFusionDispatchRouterFirst(t *testing.T) {
	cfg := mffConfig()
	route, err := MoeFrontRouteWithSort(mffLogits, mffTokens, cfg)
	if err != nil {
		t.Fatal(err)
	}
	order := MoeFrontDispatchOrder(route.Tables.Align, cfg.TopK, cfg.TileSize)
	if len(order) != len(route.ExpertIDs) {
		t.Fatalf("order covers %d rows want %d", len(order), len(route.ExpertIDs))
	}
	seen := make([]bool, len(route.ExpertIDs))
	distinct := map[int]bool{}
	switches := 0
	tileOwner := map[int]int{}
	for i, w := range order {
		if seen[w.Expanded] {
			t.Fatalf("expanded row %d dispatched twice", w.Expanded)
		}
		seen[w.Expanded] = true
		if w.Expanded != w.Token*cfg.TopK+w.Slot || int(route.ExpertIDs[w.Expanded]) != w.Expert {
			t.Fatalf("work item %+v disagrees with route ids", w)
		}
		if owner, ok := tileOwner[w.Permuted/cfg.TileSize]; ok && owner != w.Expert {
			t.Fatalf("tile %d mixes experts %d and %d", w.Permuted/cfg.TileSize, owner, w.Expert)
		}
		tileOwner[w.Permuted/cfg.TileSize] = w.Expert
		distinct[w.Expert] = true
		if i > 0 {
			if w.Expert < order[i-1].Expert {
				t.Fatalf("expert order regresses at %d: %d after %d", i, w.Expert, order[i-1].Expert)
			}
			if w.Expert != order[i-1].Expert {
				switches++
			}
		}
	}
	if switches != len(distinct)-1 {
		t.Fatalf("router-first walk switches experts %d times, want distinct-1 = %d", switches, len(distinct)-1)
	}
	tokenMajorSwitches := 0
	for i := 1; i < len(route.ExpertIDs); i++ {
		if route.ExpertIDs[i] != route.ExpertIDs[i-1] {
			tokenMajorSwitches++
		}
	}
	if tokenMajorSwitches <= switches {
		t.Fatalf("fixture does not separate orders: token-major %d vs router-first %d switches", tokenMajorSwitches, switches)
	}
}

func TestMoeFrontFusionForwardMatchesDenseOracle(t *testing.T) {
	x, fc1, fc2, _ := mffWeights()
	for _, epi := range []MoeFrontEpilogue{MoeFrontFusedFinalize, MoeFrontUnpermute} {
		for _, quant := range []bool{false, true} {
			for _, tile := range []int{1, 4, 64} {
				cfg := mffConfig()
				cfg.Epilogue, cfg.QuantizeFC1, cfg.TileSize = epi, quant, tile
				res, err := MoeFrontForward(MoeFrontInput{X: x, Tokens: mffTokens, Logits: mffLogits, FC1: fc1, FC2: fc2}, cfg)
				if err != nil {
					t.Fatal(err)
				}
				if !res.ReusedSortTables {
					t.Fatalf("epi=%d quant=%v tile=%d: routing tables were not reused", epi, quant, tile)
				}
				ids, scales := mffOracleRoute(mffLogits, mffTokens, cfg)
				want := mffOracleForward(t, x, fc1, fc2, ids, scales, mffTokens, cfg)
				mffClose(t, "out", res.Out, want)
			}
		}
	}
}

func TestMoeFrontFusionFC1EpilogueQuantizes(t *testing.T) {
	x, fc1, fc2, _ := mffWeights()
	cfg := mffConfig()
	plain, err := MoeFrontForward(MoeFrontInput{X: x, Tokens: mffTokens, Logits: mffLogits, FC1: fc1, FC2: fc2}, cfg)
	if err != nil {
		t.Fatal(err)
	}
	cfg.QuantizeFC1 = true
	quant, err := MoeFrontForward(MoeFrontInput{X: x, Tokens: mffTokens, Logits: mffLogits, FC1: fc1, FC2: fc2}, cfg)
	if err != nil {
		t.Fatal(err)
	}
	differs := false
	for i := range plain.Out {
		if plain.Out[i] != quant.Out[i] {
			differs = true
		}
	}
	if !differs {
		t.Fatal("NVFP4 FC1 epilogue left latent_down input unchanged on a fixture with non-representable activations")
	}
}

func TestMoeFrontFusionOnDeviceLogitsMatchFixedLogits(t *testing.T) {
	x, fc1, fc2, router := mffWeights()
	cfg := mffConfig()
	logits := make([]float32, mffTokens*mffExperts)
	for tok := 0; tok < mffTokens; tok++ {
		for e := 0; e < mffExperts; e++ {
			var acc float32
			for j := 0; j < mffHidden; j++ {
				acc += router[e*mffHidden+j] * x[tok*mffHidden+j]
			}
			logits[tok*mffExperts+e] = acc
		}
	}
	onDevice, err := MoeFrontForward(MoeFrontInput{X: x, Tokens: mffTokens, RouterWeight: router, FC1: fc1, FC2: fc2}, cfg)
	if err != nil {
		t.Fatal(err)
	}
	ids, scales := mffOracleRoute(logits, mffTokens, cfg)
	for i := range ids {
		if onDevice.Route.ExpertIDs[i] != ids[i] {
			t.Fatalf("on-device ids %v want %v", onDevice.Route.ExpertIDs, ids)
		}
	}
	mffClose(t, "out", onDevice.Out, mffOracleForward(t, x, fc1, fc2, ids, scales, mffTokens, cfg))
}

func TestMoeFrontFusionStaleTablesAreDropped(t *testing.T) {
	x, fc1, fc2, _ := mffWeights()
	cfg := mffConfig()
	route, err := MoeFrontRouteWithSort(mffLogits, mffTokens, cfg)
	if err != nil {
		t.Fatal(err)
	}

	same := *route
	res, err := MoeFrontForward(MoeFrontInput{X: x, Tokens: mffTokens, Prerouted: &same, FC1: fc1, FC2: fc2}, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if !res.ReusedSortTables {
		t.Fatal("pre-routed batch with matching tables re-sorted")
	}
	mffClose(t, "prerouted", res.Out, mffOracleForward(t, x, fc1, fc2, route.ExpertIDs, route.Scales, mffTokens, cfg))

	// Replace token 0's experts (a forced router) but keep routing's tables.
	replaced := *route
	replaced.ExpertIDs = append([]int32(nil), route.ExpertIDs...)
	replaced.ExpertIDs[0], replaced.ExpertIDs[1] = 5, 3
	res, err = MoeFrontForward(MoeFrontInput{X: x, Tokens: mffTokens, Prerouted: &replaced, FC1: fc1, FC2: fc2}, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if res.ReusedSortTables {
		t.Fatal("tables reused for ids they do not describe")
	}
	want := mffOracleForward(t, x, fc1, fc2, replaced.ExpertIDs, replaced.Scales, mffTokens, cfg)
	mffClose(t, "replaced", res.Out, want)
	stale := mffOracleForward(t, x, fc1, fc2, route.ExpertIDs, route.Scales, mffTokens, cfg)
	differs := false
	for i := range want {
		if want[i] != stale[i] {
			differs = true
		}
	}
	if !differs {
		t.Fatal("fixture replacement does not change the output; stale-table reuse would go unnoticed")
	}

	retiled := cfg
	retiled.TileSize = 8
	res, err = MoeFrontForward(MoeFrontInput{X: x, Tokens: mffTokens, Prerouted: &same, FC1: fc1, FC2: fc2}, retiled)
	if err != nil {
		t.Fatal(err)
	}
	if res.ReusedSortTables {
		t.Fatal("tables built at tile 4 reused by a runner tuned for tile 8")
	}
	mffClose(t, "retiled", res.Out, stale)

	noTables := *route
	noTables.Tables = nil
	res, err = MoeFrontForward(MoeFrontInput{X: x, Tokens: mffTokens, Prerouted: &noTables, FC1: fc1, FC2: fc2}, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if res.ReusedSortTables {
		t.Fatal("reported reuse with no tables")
	}
	mffClose(t, "no-tables", res.Out, stale)
}

func TestMoeFrontFusionEpiloguePlacementAgrees(t *testing.T) {
	x, fc1, fc2, _ := mffWeights()
	cfg := mffConfig()
	cfg.QuantizeFC1 = true
	fused, err := MoeFrontForward(MoeFrontInput{X: x, Tokens: mffTokens, Logits: mffLogits, FC1: fc1, FC2: fc2}, cfg)
	if err != nil {
		t.Fatal(err)
	}
	cfg.Epilogue = MoeFrontUnpermute
	unperm, err := MoeFrontForward(MoeFrontInput{X: x, Tokens: mffTokens, Logits: mffLogits, FC1: fc1, FC2: fc2}, cfg)
	if err != nil {
		t.Fatal(err)
	}
	mffClose(t, "fused-vs-unpermute", fused.Out, unperm.Out)
}

func TestMoeFrontFusionRefusals(t *testing.T) {
	x, fc1, fc2, router := mffWeights()
	nan := append([]float32(nil), mffLogits...)
	nan[7] = float32(math.NaN())
	route, err := MoeFrontRouteWithSort(mffLogits, mffTokens, mffConfig())
	if err != nil {
		t.Fatal(err)
	}
	outOfRange := *route
	outOfRange.ExpertIDs = append([]int32(nil), route.ExpertIDs...)
	outOfRange.ExpertIDs[3] = mffExperts
	dup := *route
	dup.ExpertIDs = append([]int32(nil), route.ExpertIDs...)
	dup.ExpertIDs[1] = dup.ExpertIDs[0]

	base := MoeFrontInput{X: x, Tokens: mffTokens, FC1: fc1, FC2: fc2}
	with := func(f func(*MoeFrontInput)) MoeFrontInput { in := base; f(&in); return in }
	cases := []struct {
		name string
		in   MoeFrontInput
		cfg  func(*MoeFrontConfig)
		want error
	}{
		{"topk-over-experts", with(func(in *MoeFrontInput) { in.Logits = mffLogits }), func(c *MoeFrontConfig) { c.TopK = mffExperts + 1 }, ErrMoeFrontGeometry},
		{"quant-inter-unaligned", with(func(in *MoeFrontInput) { in.Logits = mffLogits }), func(c *MoeFrontConfig) { c.QuantizeFC1 = true; c.Inter = 8 }, ErrMoeFrontGeometry},
		{"nan-logit", with(func(in *MoeFrontInput) { in.Logits = nan }), nil, ErrMoeFrontRoute},
		{"no-source", base, nil, ErrMoeFrontRoute},
		{"two-sources", with(func(in *MoeFrontInput) { in.Logits = mffLogits; in.RouterWeight = router }), nil, ErrMoeFrontRoute},
		{"prerouted-out-of-range", with(func(in *MoeFrontInput) { in.Prerouted = &outOfRange }), nil, ErrMoeFrontRoute},
		{"prerouted-duplicate", with(func(in *MoeFrontInput) { in.Prerouted = &dup }), nil, ErrMoeFrontRoute},
		{"x-length", with(func(in *MoeFrontInput) { in.Logits = mffLogits; in.X = x[:len(x)-1] }), nil, ErrMoeFrontGeometry},
		{"tile-zero", with(func(in *MoeFrontInput) { in.Logits = mffLogits }), func(c *MoeFrontConfig) { c.TileSize = 0 }, ErrMoeFrontGeometry},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg := mffConfig()
			if tc.cfg != nil {
				tc.cfg(&cfg)
			}
			res, err := MoeFrontForward(tc.in, cfg)
			if !errors.Is(err, tc.want) || res != nil {
				t.Fatalf("got res=%v err=%v want %v", res, err, tc.want)
			}
		})
	}
}

func TestMoeFrontFusionEmptyBatch(t *testing.T) {
	_, fc1, fc2, router := mffWeights()
	res, err := MoeFrontForward(MoeFrontInput{Tokens: 0, X: []float32{}, RouterWeight: router, FC1: fc1, FC2: fc2}, mffConfig())
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Out) != 0 || len(res.Order) != 0 || len(res.Route.ExpertIDs) != 0 {
		t.Fatalf("empty batch produced out=%d order=%d ids=%d", len(res.Out), len(res.Order), len(res.Route.ExpertIDs))
	}
}
