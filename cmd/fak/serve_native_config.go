package main

import (
	"flag"
	"fmt"
	"os"
	"strings"

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
	kvPrecision       *string
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
		prefillChunk:      fs.Int(nativeQwenQ4KPrefillChunkFlag, defaultNativeQwenQ4KPrefillChunk, "Qwen Q4_K prefill chunk ceiling in tokens (128..8192; default 512). Validated before model load and stamped into native receipts."),
		qwen35GDNSequence: fs.Bool("native-qwen35-metal-gdn-sequence", false, "enable the experimental Qwen3.5 Metal GDN preprojected sequence path"),
		q4kGateUpSlab:     fs.Bool("native-q4k-gateup-slab", false, "reuse the bounded Q4_K gate/up output slab within each native session"),
		kvPrecision:       fs.String("kv-precision", "", "KV cache storage tier: f32 (default, exact) or q8_0 (dense mixed: f32 pre-RoPE K + q8_0 K/V; ~2x more context). Also settable via FAK_UP_KV_PRECISION."),
		prefixProfile:     fs.String("native-prefix-profile", "", "write native prefix-cache operation profiles to this JSONL path"),
		vulkanQ4KProfile:  fs.Bool("vulkan-q4k-profile", false, "enable Vulkan Q4_K timing profiles (requires a Vulkan backend)"),
		vulkanStageQ4K:    fs.Bool("vulkan-stage-q4k", false, "use Vulkan host-visible Q4_K staging (requires a Vulkan backend)"),
	}
}

func registerGuardNativeControlFlags(fs *flag.FlagSet) nativeControlFlags {
	return registerNativeControlFlags(fs)
}

func (f nativeControlFlags) config() nativeControlConfig {
	prec, _ := resolveKVPrecisionValue(*f.kvPrecision)
	return nativeControlConfig{
		Planner: agent.InKernelPlannerConfig{
			QwenQ4KPrefillChunkTokens: *f.prefillChunk,
			Qwen35MetalGDNSequence:    *f.qwen35GDNSequence,
			Q4KGateUpOutputSlab:       *f.q4kGateUpSlab,
			KVPrecision:               prec,
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
	prec, _ := resolveKVPrecisionValue(*sf.kvPrecision)
	return nativeControlConfig{
		Planner: agent.InKernelPlannerConfig{
			QwenQ4KPrefillChunkTokens: *sf.nativeQwenQ4KPrefillChunk,
			Qwen35MetalGDNSequence:    *sf.nativeQwen35MetalGDNSequence,
			Q4KGateUpOutputSlab:       *sf.nativeQ4KGateUpOutputSlab,
			DenseGPULayers:            gpuLayers,
			KVPrecision:               prec,
		},
		PrefixProfile:    *sf.nativePrefixProfile,
		VulkanQ4KProfile: *sf.vulkanQ4KProfile,
		VulkanStageQ4K:   *sf.vulkanStageQ4K,
	}
}

func serveNativePlannerConfig(sf *serveFlags) agent.InKernelPlannerConfig {
	return serveNativeControlConfig(sf).Planner
}

// resolveKVPrecisionValue maps the serve --kv-precision flag (with the documented
// FAK_UP_KV_PRECISION env fallback) to a model.KVPrecision tier. An empty value
// resolves to the exact f32 default; an unknown token is reported as parses=false so
// the caller can refuse at flag-validation time rather than silently serving f32.
func resolveKVPrecisionValue(flagValue string) (model.KVPrecision, bool) {
	raw := strings.TrimSpace(flagValue)
	if raw == "" {
		raw = strings.TrimSpace(os.Getenv("FAK_UP_KV_PRECISION"))
	}
	if raw == "" {
		// Return the zero value so an unset tier leaves InKernelPlannerConfig
		// byte-for-byte its historical zero. The planner reads "" as f32.
		return "", true
	}
	prec, err := model.ParseKVPrecision(raw)
	if err != nil {
		return "", false
	}
	// Publish the resolved tier so the fit-estimator seam (serve_model_fit.go) charges
	// the same density the engine will realize, without threading a param through every
	// sizing helper. An explicit f32 stays as the canonical tier.
	_ = os.Setenv("FAK_UP_KV_PRECISION", string(prec))
	return prec, true
}

// validateServeKVPrecision refuses an unknown --kv-precision token at serve startup,
// before any model load or listener bind, so a typo can never silently serve f32. An
// empty value (flag and env unset) is valid and means the exact f32 default.
func validateServeKVPrecision(flagValue string) error {
	if _, ok := resolveKVPrecisionValue(flagValue); !ok {
		raw := strings.TrimSpace(flagValue)
		if raw == "" {
			raw = strings.TrimSpace(os.Getenv("FAK_UP_KV_PRECISION"))
		}
		return fmt.Errorf("--kv-precision %q is invalid (want f32 or q8_0)", raw)
	}
	return nil
}
