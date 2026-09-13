package model

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"testing"
)

// v41_moe_dispatch_bench_test.go — the V4.1-shape synthetic MoE expert-dispatch
// driver. It proves the exact V4.1 routed geometry (384 experts / top-6 / 2304 MoE
// width / hidden 5120 / route scale 1.5) drives the routing and per-layer dispatch
// path with NO V4.1 weights: experts are synthetic Q4_K tensors, the token logits
// are deterministic. The benchmark reports a per-layer scheduling cost at the real
// width (6 routed experts' gate GEMVs of [2304,5120]) so the naive looped dispatch
// and a batched/grouped dispatch are directly comparable.
//
// Run: go test ./internal/model/ -run 'V41MoEDispatch' -count=1
// Bench: go test ./internal/model/ -run '^$' -bench 'V41MoEDispatch' -benchtime=1x

const (
	v41MoEDispatchFixtureSchema = "fak-v41-moe-dispatch-fixture/1"
	v41MoEDispatchHiddenSize    = 5120
	v41MoEDispatchFixtureLayers = 4
	v41MoEDispatchLogitsSeed    = 0x41dec0de
)

// v41MoEDispatchFixtureSHA256 pins the fixture bytes so any geometry/layer edit
// is a witnessed, deliberate change rather than silent drift.
const v41MoEDispatchFixtureSHA256 = "6cb6b1d04df59e1e1185a33589883a225a24402760575dbae75cd0826f3b05b8"

// v41MoEDispatchFixtureSource derives the fixture provenance from the pinned
// checkpoint identity so a revision bump can never desync the fixture silently.
var v41MoEDispatchFixtureSource = DeepSeekV41FlashModelID + "@" + DeepSeekV41FlashRevision

type v41MoEDispatchGeometry struct {
	NumExperts          int     `json:"num_experts"`
	NumExpertsPerTok    int     `json:"num_experts_per_tok"`
	NSharedExperts      int     `json:"n_shared_experts"`
	MoEIntermediateSize int     `json:"moe_intermediate_size"`
	HiddenSize          int     `json:"hidden_size"`
	RoutedScalingFactor float64 `json:"routed_scaling_factor"`
}

type v41MoEDispatchLayer struct {
	Layer         int     `json:"layer"`
	DispatchCost  float64 `json:"dispatch_cost_us"`
	RoutedExperts int     `json:"routed_experts"`
	Note          string  `json:"note,omitempty"`
}

type v41MoEDispatchFixture struct {
	Schema   string                    `json:"schema"`
	Source   string                    `json:"source"`
	Geometry v41MoEDispatchGeometry    `json:"geometry"`
	Layers   []v41MoEDispatchLayer     `json:"layers"`
	Metrics  map[string]map[string]any `json:"metrics"`
}

func readV41MoEDispatchFixture(t *testing.T) ([]byte, v41MoEDispatchFixture) {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("testdata", "v41_moe_dispatch_fixture.json"))
	if err != nil {
		t.Fatal(err)
	}
	var fx v41MoEDispatchFixture
	if err := json.Unmarshal(raw, &fx); err != nil {
		t.Fatalf("parse %s: %v", v41MoEDispatchFixtureSchema, err)
	}
	return raw, fx
}

// TestV41MoEDispatchFixtureMatchesPinnedConfig witnesses the fixture shape against
// the pinned geometry AND against the real official config parsed through
// v41RouterConfigFromConfig — so the synthetic driver can never silently drift.
func TestV41MoEDispatchFixtureMatchesPinnedConfig(t *testing.T) {
	if V41RouterExperts != 384 || V41RouterTopK != 6 || V41RouterSharedCount != 1 ||
		V41RouterMoEWidth != 2304 || V41RouterRouteScale != 1.5 {
		t.Fatalf("pinned constants drifted: experts=%d topk=%d shared=%d moe=%d scale=%g",
			V41RouterExperts, V41RouterTopK, V41RouterSharedCount, V41RouterMoEWidth, V41RouterRouteScale)
	}

	raw, fx := readV41MoEDispatchFixture(t)
	// Pin the fixture bytes: any edit to the geometry or layer entries changes
	// this digest, so the synthetic shape can never drift unnoticed.
	digest := v41MoEDispatchSHA256(raw)
	if digest != v41MoEDispatchFixtureSHA256 {
		t.Fatalf("fixture digest=%s want %s", digest, v41MoEDispatchFixtureSHA256)
	}
	if fx.Schema != v41MoEDispatchFixtureSchema {
		t.Fatalf("schema=%q want %q", fx.Schema, v41MoEDispatchFixtureSchema)
	}
	if fx.Source != v41MoEDispatchFixtureSource {
		t.Fatalf("source=%q want %q", fx.Source, v41MoEDispatchFixtureSource)
	}

	g := fx.Geometry
	if g.NumExperts != V41RouterExperts {
		t.Fatalf("fixture num_experts=%d want %d", g.NumExperts, V41RouterExperts)
	}
	if g.NumExpertsPerTok != V41RouterTopK {
		t.Fatalf("fixture num_experts_per_tok=%d want %d", g.NumExpertsPerTok, V41RouterTopK)
	}
	if g.NSharedExperts != V41RouterSharedCount {
		t.Fatalf("fixture n_shared_experts=%d want %d", g.NSharedExperts, V41RouterSharedCount)
	}
	if g.MoEIntermediateSize != V41RouterMoEWidth {
		t.Fatalf("fixture moe_intermediate_size=%d want %d", g.MoEIntermediateSize, V41RouterMoEWidth)
	}
	if g.HiddenSize != v41MoEDispatchHiddenSize {
		t.Fatalf("fixture hidden_size=%d want %d", g.HiddenSize, v41MoEDispatchHiddenSize)
	}
	if g.RoutedScalingFactor != float64(V41RouterRouteScale) {
		t.Fatalf("fixture routed_scaling_factor=%g want %g", g.RoutedScalingFactor, float64(V41RouterRouteScale))
	}

	_, official := readDeepSeekV41Config(t)
	cfg, err := v41RouterConfigFromConfig(official)
	if err != nil {
		t.Fatalf("official config rejected: %v", err)
	}
	if cfg.Experts != g.NumExperts || cfg.TopK != g.NumExpertsPerTok ||
		cfg.SharedCount != g.NSharedExperts || cfg.RouteScale != float32(g.RoutedScalingFactor) {
		t.Fatalf("fixture geometry %+v disagrees with official router config %+v", g, cfg)
	}
	// Cross-check the fixture against the parsed OFFICIAL config directly, not
	// just the constants it was derived from — this is the real drift witness.
	if official.NumExperts != g.NumExperts {
		t.Fatalf("official num_experts=%d fixture=%d", official.NumExperts, g.NumExperts)
	}
	if official.NumExpertsPerTok != g.NumExpertsPerTok {
		t.Fatalf("official num_experts_per_tok=%d fixture=%d", official.NumExpertsPerTok, g.NumExpertsPerTok)
	}
	if official.MoEIntermediateSize != g.MoEIntermediateSize {
		t.Fatalf("official moe_intermediate_size=%d fixture=%d", official.MoEIntermediateSize, g.MoEIntermediateSize)
	}
	if official.HiddenSize != g.HiddenSize {
		t.Fatalf("official hidden_size=%d fixture=%d", official.HiddenSize, g.HiddenSize)
	}
	if official.RoutedScalingFactor != g.RoutedScalingFactor {
		t.Fatalf("official routed_scaling_factor=%g fixture=%g", official.RoutedScalingFactor, g.RoutedScalingFactor)
	}
	lastNode := official.NumLayers - 1

	t.Run("fixture layers", func(t *testing.T) {
		if len(fx.Layers) != v41MoEDispatchFixtureLayers {
			t.Fatalf("layers=%d want %d representative entries", len(fx.Layers), v41MoEDispatchFixtureLayers)
		}
		seen := make(map[int]bool, len(fx.Layers))
		for _, l := range fx.Layers {
			if l.Layer < 0 {
				t.Fatalf("negative layer index %d", l.Layer)
			}
			if seen[l.Layer] {
				t.Fatalf("duplicate layer %d", l.Layer)
			}
			seen[l.Layer] = true
			if l.RoutedExperts != V41RouterTopK {
				t.Fatalf("layer %d routed_experts=%d want top-k %d", l.Layer, l.RoutedExperts, V41RouterTopK)
			}
			if l.DispatchCost < 0 {
				t.Fatalf("layer %d dispatch_cost_us=%g must be non-negative", l.Layer, l.DispatchCost)
			}
		}
		// The whole point is a per-layer label at the real 40-layer depth, not a
		// full-model run: verify the last layer is represented.
		if !seen[lastNode] {
			t.Fatalf("fixture layers %v omit last layer %d", fx.Layers, lastNode)
		}
	})

	t.Run("metrics labeled modeled", func(t *testing.T) {
		if len(fx.Metrics) == 0 {
			t.Fatal("metrics object missing")
		}
		unit, ok := fx.Metrics["per_layer_dispatch_cost"]
		if !ok {
			t.Fatalf("metrics missing per_layer_dispatch_cost; have %v", keysOfMap(fx.Metrics))
		}
		if unit["label"] != "MODELED" {
			t.Fatalf("per-layer dispatch cost label=%v want MODELED (no tokens/sec claim)", unit["label"])
		}
	})
}

// TestV41MoEDispatchAtV41Shape drives v41Route at the exact V4.1 geometry with
// deterministic synthetic logits and asserts the top-6 contract holds.
func TestV41MoEDispatchAtV41Shape(t *testing.T) {
	_, fx := readV41MoEDispatchFixture(t)
	cfg := v41RouterConfig{
		Experts:     fx.Geometry.NumExperts,
		TopK:        fx.Geometry.NumExpertsPerTok,
		SharedCount: fx.Geometry.NSharedExperts,
		RouteScale:  float32(fx.Geometry.RoutedScalingFactor),
	}
	if err := cfg.validate(); err != nil {
		t.Fatalf("fixture-derived cfg invalid: %v", err)
	}

	logits := v41MoEDispatchSyntheticLogits(V41RouterExperts, v41MoEDispatchLogitsSeed)
	if len(logits) != V41RouterExperts {
		t.Fatalf("logits width=%d want %d", len(logits), V41RouterExperts)
	}
	picks, err := v41Route(logits, nil, cfg)
	if err != nil {
		t.Fatalf("v41Route at V4.1 shape: %v", err)
	}
	if len(picks) != V41RouterTopK {
		t.Fatalf("picks=%d want exactly %d", len(picks), V41RouterTopK)
	}
	sum := v41MoEDispatchSumWeights(picks)
	seen := make(map[int]bool, len(picks))
	for _, p := range picks {
		if p.expert < 0 || p.expert >= V41RouterExperts {
			t.Fatalf("expert index %d out of range [0,%d)", p.expert, V41RouterExperts)
		}
		if seen[p.expert] {
			t.Fatalf("duplicate expert %d in picks", p.expert)
		}
		seen[p.expert] = true
		if !v41MoEDispatchFinite(float64(p.weight)) {
			t.Fatalf("non-finite weight for expert %d", p.expert)
		}
		if !(p.weight > 0) {
			t.Fatalf("expert %d weight=%g must be positive", p.expert, p.weight)
		}
	}
	if !v41MoEDispatchFinite(float64(sum)) || sum <= 0 {
		t.Fatalf("selected weight sum=%g must be finite and positive", sum)
	}

	// Determinism: same seed -> byte-identical picks across calls.
	again, err := v41Route(v41MoEDispatchSyntheticLogits(V41RouterExperts, v41MoEDispatchLogitsSeed), nil, cfg)
	if err != nil {
		t.Fatal(err)
	}
	for i := range picks {
		if picks[i] != again[i] {
			t.Fatalf("non-deterministic pick[%d]: %+v != %+v", i, picks[i], again[i])
		}
	}
}

// BenchmarkV41MoEDispatchPerLayer times one MoE layer's selected-expert dispatch at
// the V4.1 width. It routes a synthetic token, keeps only the 6 selected experts'
// gate tensors [2304,5120] (NOT all 384 — that would be gigabytes), and times them
// as 6 separate GEMVs ("looped") vs one fused [6*2304,5120] GEMV ("batched").
func BenchmarkV41MoEDispatchPerLayer(b *testing.B) {
	b.Run("looped", func(b *testing.B) {
		v41MoEBenchDispatch(b, V41RouterTopK, V41RouterMoEWidth, v41MoEDispatchHiddenSize, false)
	})
	b.Run("batched", func(b *testing.B) {
		v41MoEBenchDispatch(b, V41RouterTopK, V41RouterMoEWidth, v41MoEDispatchHiddenSize, true)
	})
}

// v41MoEBenchDispatch mirrors benchGLMExpertDispatch's looped/batched arms at the
// V4.1 K=6, MI=2304, H=5120 geometry and reports per-layer ms and resident weight.
func v41MoEBenchDispatch(b *testing.B, K, MI, H int, batched bool) {
	if err := (&v41RouterConfig{Experts: V41RouterExperts, TopK: V41RouterTopK,
		SharedCount: V41RouterSharedCount, RouteScale: V41RouterRouteScale}).validate(); err != nil {
		b.Fatal(err)
	}

	rng := rand.New(rand.NewSource(11))
	x := make([]float32, H)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	// Route a synthetic token, then restrict the timed set to the picked experts —
	// this is real dispatch work at the real shape without materializing 384 tensors.
	logits := v41MoEDispatchSyntheticLogits(V41RouterExperts, v41MoEDispatchLogitsSeed)
	picks, err := v41Route(logits, nil, v41DefaultRouterConfig())
	if err != nil {
		b.Fatal(err)
	}
	if len(picks) != K {
		b.Fatalf("route picks=%d want K=%d", len(picks), K)
	}

	weightGiB := float64(K*MI*H/qkK*q4kBlockBytes) / (1 << 30)
	b.ReportMetric(weightGiB, "weightGiB")
	b.ReportAllocs()

	if batched {
		big := buildQ4KExpertBench(rng, K*MI, H)
		y := make([]float32, K*MI)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			q4kMatRowsInto(big, x, y)
		}
	} else {
		gates := make([]*q4kTensor, K)
		for e := range gates {
			gates[e] = buildQ4KExpertBench(rng, MI, H)
		}
		y := make([]float32, MI)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			for _, g := range gates {
				q4kMatRowsInto(g, x, y)
			}
		}
	}
	b.StopTimer()
	perLayerMs := float64(b.Elapsed().Nanoseconds()) / float64(b.N) / 1e6
	b.ReportMetric(perLayerMs, "ms/layer")
}

// v41MoEDispatchSyntheticLogits is a fixed-seed, allocation-light logit generator
// so the routing witness is identical on every run and needs no V4.1 weights.
func v41MoEDispatchSyntheticLogits(n int, seed uint64) []float32 {
	s := seed
	out := make([]float32, n)
	for i := range out {
		s = s*6364136223846793005 + 1442695040888963407
		out[i] = float32(int32(s>>33)) / float32(1<<31)
	}
	return out
}

func v41MoEDispatchSumWeights(picks []routePick) float32 {
	var sum float32
	for _, p := range picks {
		sum += p.weight
	}
	return sum
}

func v41MoEDispatchFinite(v float64) bool {
	return !(v != v || v > 1e308 || v < -1e308)
}

func v41MoEDispatchSHA256(raw []byte) string {
	sum := sha256.Sum256(raw)
	return hex.EncodeToString(sum[:])
}

func keysOfMap[V any](m map[string]V) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}
