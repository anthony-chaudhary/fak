package main

import (
	"flag"
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/modelengine"
)

var (
	configureNativePrefixProfile = model.SetPrefixProfilePath
	configureNativeVulkanQ4K     = compute.ConfigureVulkanQ4K
	configureNativeQ4KSlab       = modelengine.SetQ4KGateUpOutputSlab
)

type nativeControlFlags struct {
	prefillChunk      *int
	qwen35GDNSequence *bool
	q4kGateUpSlab     *bool
	prefixProfile     *string
	vulkanQ4KProfile  *bool
	vulkanStageQ4K    *bool
}

type nativeControlConfig struct {
	Planner          agent.InKernelPlannerConfig
	PrefixProfile    string
	VulkanQ4KProfile bool
	VulkanStageQ4K   bool
}

func registerNativeControlFlags(fs *flag.FlagSet) nativeControlFlags {
	return nativeControlFlags{
		prefillChunk:      fs.Int(nativeQwenQ4KPrefillChunkFlag, defaultNativeQwenQ4KPrefillChunk, "fak-native resident Qwen Q4_K prefill chunk ceiling in tokens (128..8192; default 512); validated before model load and stamped into native inference receipts; does not select another engine"),
		qwen35GDNSequence: fs.Bool("native-qwen35-metal-gdn-sequence", false, "enable the experimental Qwen3.5 Metal GDN preprojected sequence path"),
		q4kGateUpSlab:     fs.Bool("native-q4k-gateup-slab", false, "reuse the bounded Q4_K gate/up output slab within each native session"),
		prefixProfile:     fs.String("native-prefix-profile", "", "write native prefix-cache operation profiles to this JSONL path"),
		vulkanQ4KProfile:  fs.Bool("vulkan-q4k-profile", false, "enable Vulkan Q4_K timing profiles (requires a Vulkan backend)"),
		vulkanStageQ4K:    fs.Bool("vulkan-stage-q4k", false, "use Vulkan host-visible Q4_K staging (requires a Vulkan backend)"),
	}
}

func registerGuardNativeControlFlags(fs *flag.FlagSet) nativeControlFlags {
	return registerNativeControlFlags(fs)
}

func (f nativeControlFlags) config() nativeControlConfig {
	return nativeControlConfig{
		Planner: agent.InKernelPlannerConfig{
			QwenQ4KPrefillChunkTokens: *f.prefillChunk,
			Qwen35MetalGDNSequence:    *f.qwen35GDNSequence,
			Q4KGateUpOutputSlab:       *f.q4kGateUpSlab,
		},
		PrefixProfile:    *f.prefixProfile,
		VulkanQ4KProfile: *f.vulkanQ4KProfile,
		VulkanStageQ4K:   *f.vulkanStageQ4K,
	}
}

func applyNativeControls(backend compute.Backend, cfg nativeControlConfig) error {
	configureNativePrefixProfile(cfg.PrefixProfile)
	configureNativeQ4KSlab(cfg.Planner.Q4KGateUpOutputSlab)
	if !cfg.VulkanQ4KProfile && !cfg.VulkanStageQ4K {
		return nil
	}
	if !configureNativeVulkanQ4K(backend, cfg.VulkanQ4KProfile, cfg.VulkanStageQ4K) {
		return fmt.Errorf("--vulkan-q4k-profile/--vulkan-stage-q4k require an initialized Vulkan backend")
	}
	return nil
}

func serveNativeControlConfig(sf *serveFlags) nativeControlConfig {
	if sf == nil {
		return nativeControlConfig{}
	}
	gpuLayers := 0
	if sf.nativeGPULayers != nil {
		gpuLayers = *sf.nativeGPULayers
	}
	return nativeControlConfig{
		Planner: agent.InKernelPlannerConfig{
			QwenQ4KPrefillChunkTokens: *sf.nativeQwenQ4KPrefillChunk,
			Qwen35MetalGDNSequence:    *sf.nativeQwen35MetalGDNSequence,
			Q4KGateUpOutputSlab:       *sf.nativeQ4KGateUpOutputSlab,
			DenseGPULayers:            gpuLayers,
		},
		PrefixProfile:    *sf.nativePrefixProfile,
		VulkanQ4KProfile: *sf.vulkanQ4KProfile,
		VulkanStageQ4K:   *sf.vulkanStageQ4K,
	}
}

func serveNativePlannerConfig(sf *serveFlags) agent.InKernelPlannerConfig {
	return serveNativeControlConfig(sf).Planner
}
