package main

import (
	"bytes"
	"flag"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/compute"
)

func TestNativeControlFlagsExposeSharedPublicHelp(t *testing.T) {
	fs := flag.NewFlagSet("native-controls", flag.ContinueOnError)
	var out bytes.Buffer
	fs.SetOutput(&out)
	registerNativeControlFlags(fs)
	fs.PrintDefaults()
	for _, name := range []string{
		"native-qwen-q4k-prefill-chunk-tokens",
		"native-qwen35-metal-gdn-sequence",
		"native-q4k-gateup-slab",
		"kv-precision",
		"native-prefix-profile",
		"vulkan-q4k-profile",
		"vulkan-stage-q4k",
	} {
		if !strings.Contains(out.String(), "-"+name) {
			t.Errorf("shared native help omitted --%s\n%s", name, out.String())
		}
	}
}

func TestServeNativeFlagsReachTypedPlannerConfig(t *testing.T) {
	t.Setenv("FAK_INKERNEL_QWEN_Q4K_PREFILL_CHUNK_TOKENS", "4096")
	t.Setenv("FAK_INKERNEL_QWEN35_METAL_GDN_SEQUENCE", "off")
	t.Setenv("FAK_Q4K_GATEUP_SLAB", "0")
	fs, sf := newServeFlagSet()
	if err := fs.Parse([]string{
		"--native-qwen-q4k-prefill-chunk-tokens", "2048",
		"--native-qwen35-metal-gdn-sequence",
		"--native-q4k-gateup-slab",
	}); err != nil {
		t.Fatal(err)
	}
	want := agent.InKernelPlannerConfig{
		QwenQ4KPrefillChunkTokens: 2048,
		Qwen35MetalGDNSequence:    true,
		Q4KGateUpOutputSlab:       true,
		// fak#13263: the serve planner is structurally opted into the continuous
		// batch-decode coalescer (same wiring `fak up` already applies, #1590), so
		// concurrent same-prefix turns share one batched forward. The coalescer's
		// device half (q4k && metal && qwen35-hybrid) still gates at run time.
		BatchDecode: true,
	}
	if got := serveNativePlannerConfig(sf); !reflect.DeepEqual(got, want) {
		t.Fatalf("typed planner config = %+v, want %+v", got, want)
	}
}

func TestServeNativeProfileAndVulkanFlagsReachProductionCallers(t *testing.T) {
	t.Setenv("FAK_PREFIX_PROFILE", "legacy-prefix.jsonl")
	t.Setenv("FAK_VULKAN_Q4K_PROFILE", "0")
	t.Setenv("FAK_VULKAN_STAGE_Q4K", "0")
	oldPrefix, oldVulkan, oldSlab := configureNativePrefixProfile, configureNativeVulkanQ4K, configureNativeQ4KSlab
	t.Cleanup(func() {
		configureNativePrefixProfile, configureNativeVulkanQ4K, configureNativeQ4KSlab = oldPrefix, oldVulkan, oldSlab
	})
	var prefix string
	configureNativePrefixProfile = func(path string) { prefix = path }
	var profile, stage bool
	configureNativeVulkanQ4K = func(_ compute.Backend, gotProfile, gotStage bool) bool {
		profile, stage = gotProfile, gotStage
		return true
	}
	var slab bool
	configureNativeQ4KSlab = func(enabled bool) { slab = enabled }
	fs, sf := newServeFlagSet()
	if err := fs.Parse([]string{"--native-prefix-profile", "profiles/prefix.jsonl", "--native-q4k-gateup-slab", "--vulkan-q4k-profile", "--vulkan-stage-q4k"}); err != nil {
		t.Fatal(err)
	}
	if err := applyNativeControls(nil, serveNativeControlConfig(sf)); err != nil {
		t.Fatal(err)
	}
	if prefix != "profiles/prefix.jsonl" || !profile || !stage || !slab {
		t.Fatalf("production callers got prefix=%q profile=%t stage=%t slab=%t", prefix, profile, stage, slab)
	}
}

func TestServeGPULayersFlagsReachTypedPlannerConfig(t *testing.T) {
	fs, sf := newServeFlagSet()
	if err := fs.Parse([]string{"--gpu-layers", "16"}); err != nil {
		t.Fatal(err)
	}
	if got := serveNativePlannerConfig(sf).DenseGPULayers; got != 16 {
		t.Fatalf("--gpu-layers: DenseGPULayers = %d, want 16", got)
	}

	fs2, sf2 := newServeFlagSet()
	if err := fs2.Parse([]string{"--native-gpu-layers", "24"}); err != nil {
		t.Fatal(err)
	}
	if got := serveNativePlannerConfig(sf2).DenseGPULayers; got != 24 {
		t.Fatalf("--native-gpu-layers: DenseGPULayers = %d, want 24", got)
	}
}

// fak-test:justify why=regression when=changed:cmd/fak/serve*.go
// fak-test:runtime fast est=50ms lane=default
func TestServeV41EngramLookAheadConfig(t *testing.T) {
	fs, sf := newServeFlagSet()
	if serveNativePlannerConfig(sf).V41EngramLookAhead {
		t.Fatal("look-ahead must default off")
	}
	if !serveNativePlannerConfig(sf).BatchDecode {
		t.Fatal("flag must not change existing serve batching default")
	}
	var help bytes.Buffer
	printServeHelp(&help, fs, "native")
	if !strings.Contains(help.String(), "native-v41-engram-lookahead") {
		t.Fatal("public flag absent from native help")
	}
	if err := fs.Parse([]string{"--native-v41-engram-lookahead"}); err != nil {
		t.Fatal(err)
	}
	cfg := serveNativePlannerConfig(sf)
	if !cfg.V41EngramLookAhead || !cfg.BatchDecode {
		t.Fatal("flag lost or silently switched serving mode")
	}
	got := parseEffectiveServeConfig(t, "", "--native-v41-engram-lookahead").Values["native_v41_engram_lookahead"]
	if got.Value != true || got.Source != "flag" {
		t.Fatalf("effective readback=%+v", got)
	}
	for _, explicit := range []bool{false, true} {
		sf := independentProxyComputeFlags(t, "--base-url", "http://127.0.0.1:18091/v1")
		if explicit {
			sf.explicit["native-v41-engram-lookahead"] = true
		} else {
			*sf.nativeV41EngramLookAhead = true
		}
		remote, err := (&serveRuntime{}).proxyOnlyCompute(sf, func(string) string { return "" })
		if err != nil || remote {
			t.Fatalf("enabled/explicit compute flag ignored: %t %v", remote, err)
		}
	}
	shared := flag.NewFlagSet("shared", flag.ContinueOnError)
	registerNativeControlFlags(shared)
	if shared.Lookup("native-v41-engram-lookahead") != nil {
		t.Fatal("serve-only route advertised by unsupported guard")
	}
}
