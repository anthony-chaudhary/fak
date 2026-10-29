package model

import (
	"errors"
	"math"
	"math/rand"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

func v41RoutedTopKTestConfig() Config {
	return Config{
		NumExperts:          V41RouterExperts,
		NumExpertsPerTok:    V41RouterTopK,
		NSharedExperts:      V41RouterSharedCount,
		RoutedScalingFactor: float64(V41RouterRouteScale),
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestV41RouterRoutedTopKOverrideAdmission(t *testing.T) {
	cases := []struct {
		raw  string
		want int
		bad  bool
	}{
		{raw: "", want: V41RouterTopK},
		{raw: " 5 ", want: 5},
		{raw: "1", want: 1},
		{raw: "6", want: 6},
		{raw: "0", bad: true},
		{raw: "7", bad: true},
		{raw: "-1", bad: true},
		{raw: "five", bad: true},
	}
	for _, tc := range cases {
		t.Run(tc.raw, func(t *testing.T) {
			t.Setenv(V41RoutedTopKEnv, tc.raw)
			cfg, err := v41RouterConfigFromConfig(v41RoutedTopKTestConfig())
			if tc.bad {
				if !errors.Is(err, ErrV41RoutedTopK) {
					t.Fatalf("err = %v, want ErrV41RoutedTopK", err)
				}
				if _, fullErr := v41RouterConfigFullGeometry(v41RoutedTopKTestConfig()); !errors.Is(fullErr, ErrV41RoutedTopK) || !errors.Is(fullErr, ErrV41ForwardStage) {
					t.Fatalf("full geometry err = %v, want ErrV41RoutedTopK and ErrV41ForwardStage", fullErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("admission: %v", err)
			}
			if cfg.TopK != tc.want || cfg.Experts != V41RouterExperts || cfg.SharedCount != V41RouterSharedCount || cfg.RouteScale != V41RouterRouteScale {
				t.Fatalf("cfg = %+v, want TopK %d over the published geometry", cfg, tc.want)
			}
		})
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestV41RouterRoutedTopKTruncatesAndRenormalises(t *testing.T) {
	rng := rand.New(rand.NewSource(4105))
	for trial := 0; trial < 32; trial++ {
		logits := make([]float32, V41RouterExperts)
		bias := make([]float32, V41RouterExperts)
		for i := range logits {
			logits[i] = float32(rng.NormFloat64() * 2)
			bias[i] = float32(rng.NormFloat64() * 0.1)
		}
		for _, full := range []bool{false, true} {
			base := v41DefaultRouterConfig()
			six, err := v41RouteForGeometry(logits, bias, base, full)
			if err != nil {
				t.Fatalf("top-6 route: %v", err)
			}
			for n := 1; n < V41RouterTopK; n++ {
				cfg := base
				cfg.TopK = n
				got, err := v41RouteForGeometry(logits, bias, cfg, full)
				if err != nil {
					t.Fatalf("top-%d route: %v", n, err)
				}
				if len(got) != n {
					t.Fatalf("top-%d returned %d picks", n, len(got))
				}
				var sixSum, gotSum float64
				for i := 0; i < n; i++ {
					sixSum += float64(six[i].weight)
				}
				for i, pick := range got {
					if pick.expert != six[i].expert {
						t.Fatalf("trial %d full=%v top-%d pick %d expert %d, want top-6 prefix expert %d", trial, full, n, i, pick.expert, six[i].expert)
					}
					gotSum += float64(pick.weight)
					// Renormalised over the shorter list: each weight is the
					// top-6 weight rescaled by the same prefix factor.
					want := float64(six[i].weight) / sixSum * float64(V41RouterRouteScale)
					if math.Abs(float64(pick.weight)-want) > 1e-5 {
						t.Fatalf("trial %d full=%v top-%d pick %d weight %g, want %g", trial, full, n, i, pick.weight, want)
					}
				}
				if math.Abs(gotSum-float64(V41RouterRouteScale)) > 1e-5 {
					t.Fatalf("trial %d full=%v top-%d weight sum %g, want route scale %g", trial, full, n, gotSum, V41RouterRouteScale)
				}
			}
		}
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestV41RouterRoutedTopKRefusesOutOfRangeConfig(t *testing.T) {
	logits := make([]float32, V41RouterExperts)
	for _, n := range []int{0, V41RouterTopK + 1} {
		cfg := v41DefaultRouterConfig()
		cfg.TopK = n
		_, err := v41Route(logits, nil, cfg)
		var routeErr *v4RouteError
		if !errors.As(err, &routeErr) || routeErr.Field != "v41_topk" {
			t.Fatalf("TopK=%d err = %v, want v41_topk route error", n, err)
		}
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestV41RouterRoutedTopKAutoRule(t *testing.T) {
	const gib = int64(1) << 30
	cases := []struct {
		name   string
		vram   int64
		known  bool
		shares bool
		want   int
	}{
		{name: "unknown_device", want: V41RouterTopK},
		{name: "unknown_but_shares_ram", shares: true, want: V41RouterTopK - 1},
		{name: "16gib_discrete", vram: 16 * gib, known: true, want: V41RouterTopK - 1},
		{name: "12gib_discrete", vram: 12 * gib, known: true, want: V41RouterTopK - 1},
		{name: "16gib_plus_one", vram: 16*gib + 1, known: true, want: V41RouterTopK},
		{name: "24gib_discrete", vram: 24 * gib, known: true, want: V41RouterTopK},
		{name: "32gib_discrete", vram: 32 * gib, known: true, want: V41RouterTopK},
		{name: "128gib_unified", vram: 128 * gib, known: true, shares: true, want: V41RouterTopK - 1},
		{name: "zero_reported", vram: 0, known: true, want: V41RouterTopK},
		{name: "small_but_unknown", vram: 8 * gib, known: false, want: V41RouterTopK},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := V41AutoRoutedTopK(tc.vram, tc.known, tc.shares); got != tc.want {
				t.Fatalf("V41AutoRoutedTopK(%d,%v,%v) = %d, want %d", tc.vram, tc.known, tc.shares, got, tc.want)
			}
		})
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestV41RouterRoutedTopKAutoResolvesThroughProbe(t *testing.T) {
	prev := v41RoutedTopKDeviceProbe
	t.Cleanup(func() { v41RoutedTopKDeviceProbe = prev })

	t.Setenv(V41RoutedTopKEnv, "AUTO")
	v41RoutedTopKDeviceProbe = nil
	cfg, err := v41RouterConfigFromConfig(v41RoutedTopKTestConfig())
	if err != nil || cfg.TopK != V41RouterTopK {
		t.Fatalf("auto without probe: TopK=%d err=%v, want published %d", cfg.TopK, err, V41RouterTopK)
	}

	v41RoutedTopKDeviceProbe = func() (int64, bool, bool) { return 16 << 30, true, false }
	cfg, err = v41RouterConfigFromConfig(v41RoutedTopKTestConfig())
	if err != nil || cfg.TopK != V41RouterTopK-1 {
		t.Fatalf("auto on 16 GiB: TopK=%d err=%v, want %d", cfg.TopK, err, V41RouterTopK-1)
	}

	v41RoutedTopKDeviceProbe = func() (int64, bool, bool) { return 24 << 30, true, false }
	cfg, err = v41RouterConfigFromConfig(v41RoutedTopKTestConfig())
	if err != nil || cfg.TopK != V41RouterTopK {
		t.Fatalf("auto on 24 GiB: TopK=%d err=%v, want %d", cfg.TopK, err, V41RouterTopK)
	}
}

// The device seam passes cfg.TopK to the backend primitive and must publish and
// validate rows*TopK picks for a reduced count, matching the host route.
// fak-test:runtime fast est=20ms lane=default
func TestV41RouterRoutedTopKDeviceSeamMatchesHost(t *testing.T) {
	rng := rand.New(rand.NewSource(41052))
	cfg := v41DefaultRouterConfig()
	cfg.TopK = V41RouterTopK - 1
	a, bias := v41RouterFixture(rng)
	c, _ := v41RouterFixture(rng)
	for _, full := range []bool{false, true} {
		b := newV41RouterTestBackend()
		lt := b.Upload(compute.NewF32(compute.Default(), []int{2, cfg.Experts}, append(append([]float32(nil), a...), c...)), compute.F32)
		bt := b.Upload(compute.NewF32(compute.Default(), []int{cfg.Experts}, bias), compute.F32)
		rows, err := v41DeviceRoute(b, lt, bt, 2, cfg, full, true)
		if err != nil {
			t.Fatalf("full=%v device route TopK=%d: %v", full, cfg.TopK, err)
		}
		for i, logits := range [][]float32{a, c} {
			want, err := v41RouteForGeometry(logits, bias, cfg, full)
			if err != nil {
				t.Fatal(err)
			}
			if len(rows[i]) != cfg.TopK || !reflect.DeepEqual(rows[i], want) {
				t.Fatalf("full=%v row %d device picks %v, host %v", full, i, rows[i], want)
			}
		}

		// Row-adapter path: one host row through the bound session primitive.
		s := &Session{M: &Model{}, Backend: newV41RouterTestBackend()}
		got, err := v41RouteForGeometryWith(3, a, bias, cfg, full, s.v41RouterTopKFunc())
		if err != nil {
			t.Fatalf("full=%v bound route TopK=%d: %v", full, cfg.TopK, err)
		}
		want, _ := v41RouteForGeometry(a, bias, cfg, full)
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("full=%v bound picks %v, host %v", full, got, want)
		}
	}
}
