package model

import (
	"errors"
	"math"
	"math/rand"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

const mib = int64(1) << 20

// qwen38BandConfig is the Qwen3.8-27B geometry the band planner is sized for: 64 layers
// in [linear, linear, linear, full] periods, 4 KV heads of 256, and the GDN shapes that
// give 149.6 MiB of recurrent state over the 48 linear layers (FIT.md arithmetic).
func qwen38BandConfig() Config {
	cfg := Config{
		NumLayers: 64, HiddenSize: 5120, NumKVHeads: 4, HeadDim: 256,
		LinearConvKernelDim: 4, LinearKeyHeadDim: 128, LinearValueHeadDim: 128,
		LinearNumKeyHeads: 16, LinearNumValueHeads: 48,
	}
	cfg.LayerTypes = make([]string, cfg.NumLayers)
	for l := range cfg.LayerTypes {
		if l%4 == 3 {
			cfg.LayerTypes[l] = "full_attention"
		} else {
			cfg.LayerTypes[l] = "linear_attention"
		}
	}
	return cfg
}

// qwen38Q4KMCosts approximates the Q4_K_M file: ~232 MiB per layer, a 682 MiB Q4_K
// embedding and a 1,040 MiB Q6_K head, which sums close to the 16,314 MiB GGUF.
func qwen38Q4KMCosts(t *testing.T, ctx int) (Config, DeviceBandCosts) {
	t.Helper()
	cfg := qwen38BandConfig()
	w := make([]int64, cfg.NumLayers)
	for l := range w {
		w[l] = 228 * mib
		if cfg.LayerTypes[l] == "full_attention" {
			w[l] = 244 * mib
		}
	}
	costs, err := HybridDeviceBandCosts(cfg, w, 682*mib, 1040*mib, 300*mib, ctx, compute.KVPrecisionF32)
	if err != nil {
		t.Fatalf("HybridDeviceBandCosts: %v", err)
	}
	return cfg, costs
}

// fak-test:runtime fast est=1ms lane=default
func TestHybridDeviceBandCostsMatchesFitArithmetic(t *testing.T) {
	cfg, costs := qwen38Q4KMCosts(t, 8192)
	var kv, state int64
	full := 0
	for _, c := range costs.Layers {
		kv += c.KVBytes
		state += c.StateBytes
		if c.FullAttention {
			full++
			if c.StateBytes != 0 {
				t.Fatal("a full-attention layer was charged recurrent state")
			}
		} else if c.KVBytes != 0 {
			t.Fatal("a linear-attention layer was charged KV")
		}
	}
	if full != 16 {
		t.Fatalf("full-attention layers = %d, want 16", full)
	}
	// 16 rows x 3 tensors x 4 heads x 256 x 4 B = 196,608 B per token (FIT.md).
	if want := int64(8192) * 196608; kv != want {
		t.Fatalf("KV at 8192 tokens = %d, want %d", kv, want)
	}
	if want := int64(156893184); state != want {
		t.Fatalf("GDN state = %d, want %d (149.6 MiB)", state, want)
	}
	if _, err := HybridDeviceBandCosts(cfg, make([]int64, 3), 0, 0, 0, 1, compute.KVPrecisionF32); !errors.Is(err, ErrDeviceBandsInvalid) {
		t.Fatalf("short weight table err = %v, want ErrDeviceBandsInvalid", err)
	}
}

// laptopBudgets is the target box: the RTX 4070 Laptop (8 GiB, ~7 GiB free beside the
// display) as device 0 and two eGPUs as devices 1 and 2. The eGPU size is a parameter
// because the cards were first reported as 12 GiB and later enumerated as 16 GiB.
func laptopBudgets(egpuGiB int64) []DeviceBudget {
	gib := int64(1) << 30
	return []DeviceBudget{
		{Device: 0, Name: "RTX 4070 Laptop", FreeBytes: 7 * gib},
		{Device: 1, Name: "eGPU A", FreeBytes: egpuGiB*gib - gib},
		{Device: 2, Name: "eGPU B", FreeBytes: egpuGiB*gib - gib},
	}
}

func assertBandInvariants(t *testing.T, cfg Config, costs DeviceBandCosts, plan DeviceBandPlan) {
	t.Helper()
	if _, err := plan.PartitionPlan(cfg); err != nil {
		t.Fatalf("plan does not form a valid partition: %v", err)
	}
	var weights int64
	for i, b := range plan.Bands {
		if b.Stage.Hi-b.Stage.Lo <= 0 {
			t.Fatalf("band %d empty", i)
		}
		// Hybrid cuts land right after a full-attention layer, so KV and GDN state of a
		// period never straddle two devices and every band has full-attention rows.
		if !costs.Layers[b.Stage.Hi-1].FullAttention {
			t.Fatalf("band %d [%d,%d) ends on a linear-attention layer", i, b.Stage.Lo, b.Stage.Hi)
		}
		if b.FullAttentionLayers == 0 || b.LinearLayers != 3*b.FullAttentionLayers {
			t.Fatalf("band %d holds %d full / %d linear layers, want whole periods", i, b.FullAttentionLayers, b.LinearLayers)
		}
		if b.TotalBytes != b.WeightBytes+b.KVBytes+b.StateBytes+b.FixedBytes {
			t.Fatalf("band %d byte columns do not sum", i)
		}
		if b.TotalBytes > b.UsableBytes {
			t.Fatalf("band %d overfills device %d: %d > %d", i, b.Device, b.TotalBytes, b.UsableBytes)
		}
		weights += b.WeightBytes
	}
	var want int64
	for _, c := range costs.Layers {
		want += c.WeightBytes
	}
	if weights != want {
		t.Fatalf("bands hold %d weight bytes, model has %d", weights, want)
	}
}

// fak-test:runtime fast est=2ms lane=default
func TestPlanDeviceBandsLaptopThreeDevices(t *testing.T) {
	for _, egpu := range []int64{12, 16} {
		cfg, costs := qwen38Q4KMCosts(t, 8192)
		devs := laptopBudgets(egpu)
		plan, err := PlanDeviceBands(cfg, costs, devs, DeviceBandOptions{})
		if err != nil {
			t.Fatalf("egpu %d GiB: PlanDeviceBands: %v", egpu, err)
		}
		assertBandInvariants(t, cfg, costs, plan)
		for i, b := range plan.Bands {
			if b.Device != devs[i].Device {
				t.Fatalf("band %d on device %d, want pipeline order %d", i, b.Device, devs[i].Device)
			}
		}
		// The 8 GiB card must carry fewer layers than either larger eGPU.
		small := plan.Bands[0].Stage.Hi - plan.Bands[0].Stage.Lo
		for _, b := range plan.Bands[1:] {
			if small >= b.Stage.Hi-b.Stage.Lo {
				t.Fatalf("egpu %d GiB: 4070 band has %d layers, eGPU band %d", egpu, small, b.Stage.Hi-b.Stage.Lo)
			}
		}
		if got, want := plan.MaxUtilization(), bruteForceMinMax(cfg, costs, devs, DefaultDeviceBandHeadroom); got != want {
			t.Fatalf("egpu %d GiB: max utilization %v, brute force optimum %v", egpu, got, want)
		}
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestPlanDeviceBandsRefusesWhenNoSplitFits(t *testing.T) {
	cfg, costs := qwen38Q4KMCosts(t, 8192)
	gib := int64(1) << 30
	// One 16 GiB card cannot hold Q4_K_M (FIT.md: short by ~3.3 GiB).
	_, err := PlanDeviceBands(cfg, costs, []DeviceBudget{{Device: 1, FreeBytes: 15 * gib}}, DeviceBandOptions{})
	if !errors.Is(err, ErrDeviceBandsDoNotFit) {
		t.Fatalf("single eGPU err = %v, want ErrDeviceBandsDoNotFit", err)
	}
	// Two 8 GiB-class cards are still short once KV, GDN state and fixed bytes land.
	_, err = PlanDeviceBands(cfg, costs, []DeviceBudget{{Device: 0, FreeBytes: 7 * gib}, {Device: 1, FreeBytes: 7 * gib}}, DeviceBandOptions{})
	if !errors.Is(err, ErrDeviceBandsDoNotFit) {
		t.Fatalf("two small cards err = %v, want ErrDeviceBandsDoNotFit", err)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestPlanDeviceBandsRejectsInvalidRequests(t *testing.T) {
	cfg, costs := qwen38Q4KMCosts(t, 1024)
	gib := int64(1) << 30
	cases := map[string]struct {
		devs []DeviceBudget
		opts DeviceBandOptions
		want error
	}{
		"no devices":       {nil, DeviceBandOptions{}, ErrDeviceBandsInvalid},
		"duplicate device": {[]DeviceBudget{{Device: 1, FreeBytes: gib}, {Device: 1, FreeBytes: gib}}, DeviceBandOptions{}, ErrDeviceBandsInvalid},
		"zero free":        {[]DeviceBudget{{Device: 0, FreeBytes: 0}}, DeviceBandOptions{}, ErrDeviceBandsInvalid},
		"bad headroom":     {[]DeviceBudget{{Device: 0, FreeBytes: gib}}, DeviceBandOptions{Headroom: 1.5}, ErrDeviceBandsInvalid},
		// 64 layers in 16 periods give 16 legal bands at most.
		"too many devices": {manyDevices(17, 40*gib), DeviceBandOptions{}, ErrDeviceBandsTooFewCut},
	}
	for name, tc := range cases {
		if _, err := PlanDeviceBands(cfg, costs, tc.devs, tc.opts); !errors.Is(err, tc.want) {
			t.Errorf("%s: err = %v, want %v", name, err, tc.want)
		}
	}
	if _, err := PlanDeviceBands(cfg, costs, manyDevices(16, 40*gib), DeviceBandOptions{}); err != nil {
		t.Fatalf("16 devices over 16 periods: %v", err)
	}
}

func manyDevices(n int, free int64) []DeviceBudget {
	out := make([]DeviceBudget, n)
	for i := range out {
		out[i] = DeviceBudget{Device: i, FreeBytes: free}
	}
	return out
}

// fak-test:runtime fast est=20ms lane=default
func TestPlanDeviceBandsIsMinMaxOptimal(t *testing.T) {
	rng := rand.New(rand.NewSource(20261025))
	for trial := 0; trial < 60; trial++ {
		cfg := Config{NumLayers: 6 + rng.Intn(10), NumKVHeads: 2, HeadDim: 8}
		w := make([]int64, cfg.NumLayers)
		for l := range w {
			w[l] = int64(1+rng.Intn(50)) * mib
		}
		costs, err := HybridDeviceBandCosts(cfg, w, int64(rng.Intn(20))*mib, int64(rng.Intn(20))*mib, mib, 64, compute.KVPrecisionF32)
		if err != nil {
			t.Fatal(err)
		}
		devs := manyDevices(1+rng.Intn(4), 0)
		for i := range devs {
			devs[i].FreeBytes = int64(100+rng.Intn(400)) * mib
		}
		want := bruteForceMinMax(cfg, costs, devs, 0.9)
		plan, err := PlanDeviceBands(cfg, costs, devs, DeviceBandOptions{Headroom: 0.9})
		if err != nil && !errors.Is(err, ErrDeviceBandsDoNotFit) {
			t.Fatalf("trial %d: %v", trial, err)
		}
		if got := plan.MaxUtilization(); got != want {
			t.Fatalf("trial %d: planner max utilization %v, brute force %v", trial, got, want)
		}
		if (err == nil) != (want <= 1) {
			t.Fatalf("trial %d: fit verdict err=%v disagrees with optimum %v", trial, err, want)
		}
		// A dense model may cut at every boundary and charges KV to every layer.
		for i, b := range plan.Bands {
			if b.LinearLayers != 0 || b.FullAttentionLayers != b.Stage.Hi-b.Stage.Lo {
				t.Fatalf("trial %d band %d: dense layers miscounted", trial, i)
			}
		}
	}
}

// bruteForceMinMax enumerates every legal split and returns the best max utilization.
func bruteForceMinMax(cfg Config, costs DeviceBandCosts, devs []DeviceBudget, headroom float64) float64 {
	cuts := legalBandCuts(cfg, costs)
	n, D := len(costs.Layers), len(devs)
	best := math.Inf(1)
	var rec func(dev, lo, from int, worst float64)
	rec = func(dev, lo, from int, worst float64) {
		band := func(hi int) float64 {
			var b int64 = costs.FixedBytesPerDevice
			for l := lo; l < hi; l++ {
				c := costs.Layers[l]
				b += c.WeightBytes + c.KVBytes + c.StateBytes
			}
			if dev == 0 {
				b += costs.EmbedBytes
			}
			if dev == D-1 {
				b += costs.HeadBytes
			}
			return float64(b) / float64(int64(float64(devs[dev].FreeBytes)*headroom))
		}
		if dev == D-1 {
			best = math.Min(best, math.Max(worst, band(n)))
			return
		}
		for i := from; i < len(cuts); i++ {
			rec(dev+1, cuts[i], i+1, math.Max(worst, band(cuts[i])))
		}
	}
	rec(0, 0, 0, 0)
	return best
}
