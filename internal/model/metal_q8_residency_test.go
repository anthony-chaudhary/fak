package model

import (
	"os"
	"reflect"
	"testing"
	"unsafe"
)

func exactQwen38Q8Config() Config {
	types := make([]string, 64)
	for i := range types {
		if i%4 == 3 {
			types[i] = "full_attention"
		} else {
			types[i] = "linear_attention"
		}
	}
	return Config{ModelType: "qwen3_5_text", NumLayers: 64, LayerTypes: types}
}

func qwen35LinearLayerCount(cfg Config) int {
	count := 0
	for layer := range cfg.LayerTypes {
		if cfg.isLinearAttnLayer(layer) {
			count++
		}
	}
	return count
}

func qwen35Q8ProjectionCount(cfg Config) int {
	const linearProjections, fullAttentionProjections = 5, 2
	linear := qwen35LinearLayerCount(cfg)
	return linear*linearProjections + (len(cfg.LayerTypes)-linear)*fullAttentionProjections
}

func TestQwen38MetalQ8RuntimeInventoryMatchesLayerArchitecture(t *testing.T) {
	cfg := exactQwen38Q8Config()
	names, err := qwen38MetalQ8RuntimeNames(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if want := qwen35Q8ProjectionCount(cfg); len(names) != want {
		t.Fatalf("inventory=%d want %d projections derived from layer architecture", len(names), want)
	}
	seen := make(map[string]bool, len(names))
	for _, name := range names {
		if seen[name] {
			t.Fatalf("duplicate runtime name %q", name)
		}
		seen[name] = true
		if len(name) >= 4 && (name[:4] == "mtp." || name[:4] == "next") {
			t.Fatalf("trailing non-runtime tensor admitted: %q", name)
		}
	}
	for _, name := range []string{
		"model.layers.0.linear_attn.in_proj_qkv.weight",
		"model.layers.0.linear_attn.out_proj.weight",
		"model.layers.3.self_attn.q_proj.weight",
		"model.layers.3.self_attn.k_proj.weight",
	} {
		if !seen[name] {
			t.Fatalf("missing runtime name %q", name)
		}
	}
}

func TestQwen38MetalQ8RuntimeInventoryRefusesDrift(t *testing.T) {
	cfg := exactQwen38Q8Config()
	for _, mutate := range []func(*Config){
		func(c *Config) { c.NumLayers = 63 },
		func(c *Config) { c.LayerTypes = c.LayerTypes[:63] },
		func(c *Config) { c.LayerTypes[0] = "full_attention" },
		func(c *Config) { c.LayerTypes[0] = "mystery" },
	} {
		copyCfg := cfg
		copyCfg.LayerTypes = append([]string(nil), cfg.LayerTypes...)
		mutate(&copyCfg)
		if _, err := qwen38MetalQ8RuntimeNames(copyCfg); err == nil {
			t.Fatalf("drifted config admitted: layers=%d types=%v", copyCfg.NumLayers, copyCfg.LayerTypes[:1])
		}
	}
}

func TestQ8AliasBudgetCountsPhysicalOwnerOnceAndRefusesOverride(t *testing.T) {
	const GiB = int64(1) << 30
	if err := q8AliasFits(23*GiB, 27*GiB, ""); err != nil {
		t.Fatalf("single physical 23 GiB owner should fit 90%% of 27 GiB: %v", err)
	}
	for _, tc := range []struct {
		resident, total int64
		override        string
	}{
		{23 * GiB, 0, ""},
		{25 * GiB, 27 * GiB, ""},
		{23 * GiB, 27 * GiB, "1"},
		{23 * GiB, 27 * GiB, "0"},
	} {
		if err := q8AliasFits(tc.resident, tc.total, tc.override); err == nil {
			t.Fatalf("unproved alias admitted: resident=%d total=%d override=%q", tc.resident, tc.total, tc.override)
		}
	}
}

func TestBuildAllOrNothingRollsBackInReverse(t *testing.T) {
	var released []int
	_, err := buildAllOrNothing([]string{"a", "b", "c", "d"}, func(name string) (int, error) {
		if name == "d" {
			return 0, &MetalQ8ResidencyUnavailableError{Reason: "injected"}
		}
		return int(name[0]), nil
	}, func(v int) { released = append(released, v) })
	if err == nil {
		t.Fatal("injected failure admitted")
	}
	if want := []int{int('c'), int('b'), int('a')}; !reflect.DeepEqual(released, want) {
		t.Fatalf("rollback=%v want reverse %v", released, want)
	}
}

func TestNewQ8TensorOwnsPageAlignedByteIdenticalBacking(t *testing.T) {
	qt := newQ8Tensor(7, 64, 2)
	page := uintptr(os.Getpagesize())
	if uintptr(unsafe.Pointer(&qt.q[0]))%page != 0 || uintptr(unsafe.Pointer(&qt.d[0]))%page != 0 {
		t.Fatalf("unaligned owners: q=%#x d=%#x page=%d", uintptr(unsafe.Pointer(&qt.q[0])), uintptr(unsafe.Pointer(&qt.d[0])), page)
	}
	qt.q[17], qt.d[3] = -91, 0.375
	if qt.q[17] != -91 || qt.d[3] != 0.375 {
		t.Fatal("Q8 owner lost byte identity")
	}
}
