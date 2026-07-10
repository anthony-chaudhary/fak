package deepseekv4moe

import (
	"errors"
	"testing"
)

func TestV4ConfigValidates(t *testing.T) {
	c := V4Config()
	if err := c.Validate(); err != nil {
		t.Fatalf("published V4 shape must be legal: %v", err)
	}
	if c.NumRoutedExperts != 384 || c.TopK != 6 || c.NumSharedExperts != 1 {
		t.Fatalf("V4 shape drifted: %+v", c)
	}
}

// The dispatch contract's failure modes: each malformed config must return its own
// specific, closed-set reason — not a generic error and never a silent pass.
func TestDispatchFailureModes(t *testing.T) {
	cases := []struct {
		name string
		mut  func(*Config)
		want error
	}{
		{"topk too wide", func(c *Config) { c.TopK = c.NumRoutedExperts + 1 }, ErrTopKWidth},
		{"topk zero", func(c *Config) { c.TopK = 0 }, ErrTopKWidth},
		{"missing router weight", func(c *Config) { c.HasRouterWeight = false }, ErrMissingRouter},
		{"shared mis-offloaded", func(c *Config) { c.SharedTreatedAsRouted = true }, ErrSharedMisOffload},
		{"no experts", func(c *Config) { c.NumRoutedExperts = 0 }, ErrExpertCount},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			c := V4Config()
			tc.mut(&c)
			if err := c.Validate(); !errors.Is(err, tc.want) {
				t.Errorf("Validate() = %v, want %v", err, tc.want)
			}
		})
	}
}

func TestRouteConservesSelectionsAndShared(t *testing.T) {
	const tokens = 4096
	r, err := Route(tokens, V4Config(), 0)
	if err != nil {
		t.Fatal(err)
	}
	sum := 0
	for _, l := range r.Loads {
		sum += l
	}
	if sum != tokens*TopK {
		t.Errorf("routed selections = %d, want tokens*topk = %d", sum, tokens*TopK)
	}
	if r.Shared != tokens {
		t.Errorf("shared expert must fire on every token: got %d, want %d", r.Shared, tokens)
	}
}

// A near-uniform router is balanced; a skewed one is detectably imbalanced. Same pure
// function, no RNG — the skew argument is the only difference.
func TestLoadBalanceDetectsSkew(t *testing.T) {
	const tokens = 4096
	bal, _ := Route(tokens, V4Config(), 0)
	skewed, _ := Route(tokens, V4Config(), 2.0)

	if lb := bal.LoadBalance(); lb >= 1.8 {
		t.Errorf("uniform routing should be balanced, got max/mean = %.3f", lb)
	}
	if lb := skewed.LoadBalance(); lb <= 2.0 {
		t.Errorf("skewed routing should be imbalanced, got max/mean = %.3f", lb)
	}
	if skewed.ActiveExperts() >= NumRoutedExperts {
		t.Errorf("skewed routing should starve some experts, active = %d", skewed.ActiveExperts())
	}
}

// Grouped scheduling never launches more kernels than naive, and strictly fewer when
// some experts are empty; the WORK is identical either way.
func TestGroupedNeverLaunchesMoreThanNaive(t *testing.T) {
	// Small batch => many empty experts => strict win for grouping.
	r, err := Route(32, V4Config(), 0)
	if err != nil {
		t.Fatal(err)
	}
	naive := r.Cost(NaivePerExpert)
	grouped := r.Cost(GroupedFused)

	if naive.Launches != NumRoutedExperts+NumSharedExperts {
		t.Errorf("naive must launch every expert + shared: got %d, want %d", naive.Launches, NumRoutedExperts+NumSharedExperts)
	}
	if grouped.Launches != r.ActiveExperts()+NumSharedExperts {
		t.Errorf("grouped must launch only active experts + shared: got %d, want %d", grouped.Launches, r.ActiveExperts()+NumSharedExperts)
	}
	if grouped.Launches >= naive.Launches {
		t.Errorf("grouped should launch strictly fewer with empty experts: grouped=%d naive=%d", grouped.Launches, naive.Launches)
	}
	if naive.WorkRows != grouped.WorkRows {
		t.Errorf("scheduling must not change the work: naive=%d grouped=%d", naive.WorkRows, grouped.WorkRows)
	}
}

// EP partition must be an exact, disjoint cover of every expert — the ranks=1
// bit-exactness invariant, and correct remainder spreading for uneven ranks.
func TestExpertParallelPlanExactCover(t *testing.T) {
	for _, ranks := range []int{1, 2, 3, 7, 8, 384} {
		bands, err := ExpertParallelPlan(NumRoutedExperts, ranks)
		if err != nil {
			t.Fatalf("ranks=%d: %v", ranks, err)
		}
		if len(bands) != ranks {
			t.Errorf("ranks=%d: got %d bands", ranks, len(bands))
		}
		seen := make([]bool, NumRoutedExperts)
		total := 0
		for _, band := range bands {
			for _, e := range band {
				if e < 0 || e >= NumRoutedExperts {
					t.Fatalf("ranks=%d: expert %d out of range", ranks, e)
				}
				if seen[e] {
					t.Fatalf("ranks=%d: expert %d covered twice", ranks, e)
				}
				seen[e] = true
				total++
			}
		}
		if total != NumRoutedExperts {
			t.Errorf("ranks=%d: covered %d of %d experts", ranks, total, NumRoutedExperts)
		}
	}
	if _, err := ExpertParallelPlan(NumRoutedExperts, 0); err == nil {
		t.Error("ranks=0 must error")
	}
	if _, err := ExpertParallelPlan(NumRoutedExperts, NumRoutedExperts+1); err == nil {
		t.Error("ranks > experts must error")
	}
}

func TestCommUnitsScaleWithTopKNotPool(t *testing.T) {
	const tokens = 1024
	r, _ := Route(tokens, V4Config(), 0)
	if got := r.CommUnits(1); got != 0 {
		t.Errorf("no fabric at ranks=1: got %d", got)
	}
	if got, want := r.CommUnits(8), tokens*TopK; got != want {
		t.Errorf("all-to-all traffic = %d, want tokens*topk = %d (scales with top-k, not 384)", got, want)
	}
}

func TestMetricsPercentileOrdering(t *testing.T) {
	r, _ := Route(2048, V4Config(), 0)
	m := r.ComputeMetrics(GroupedFused, 61, 8)
	if m.P95PerLayerWork < m.P50PerLayerWork {
		t.Errorf("p95 (%.2f) must be >= p50 (%.2f)", m.P95PerLayerWork, m.P50PerLayerWork)
	}
	if m.ExpertLoadBalance < 1.0 {
		t.Errorf("load balance must be >= 1.0, got %.3f", m.ExpertLoadBalance)
	}
	if m.CommUnits != 2048*TopK {
		t.Errorf("comm units = %d, want %d", m.CommUnits, 2048*TopK)
	}
}
