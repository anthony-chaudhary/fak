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
