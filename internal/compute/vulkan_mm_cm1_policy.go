package compute

import (
	"errors"
	"fmt"
	"os"
	"strconv"
	"sync"
	"sync/atomic"
)

// Prefill GEMM route (mm_cm1_*.spv): a multi-token
// matmul over a packed Q2_K or i-quant weight runs one tiled fp16 cooperative-matrix GEMM
// instead of one weight-streaming matvec per token. Every call that does not take the GEMM
// names its reason with one of these sentinels and is counted, so a fallback is never silent.
var (
	ErrVulkanMMCM1Disabled         = errors.New("compute: vulkan mm_cm1 prefill GEMM disabled")
	ErrVulkanMMCM1BelowThreshold   = errors.New("compute: vulkan mm_cm1 token count at or below threshold")
	ErrVulkanMMCM1UnsupportedDtype = errors.New("compute: vulkan mm_cm1 has no kernel for weight dtype")
	ErrVulkanMMCM1Unavailable      = errors.New("compute: vulkan mm_cm1 kernel unavailable on this device")
	ErrVulkanMMCM1Shape            = errors.New("compute: vulkan mm_cm1 refused matmul shape")
	ErrVulkanMMCM1Dispatch         = errors.New("compute: vulkan mm_cm1 dispatch not recorded")
)

const (
	vulkanMMCM1EnvEnable    = "FAK_VULKAN_MM_CM1"
	vulkanMMCM1EnvMinTokens = "FAK_VULKAN_MM_CM1_MIN_TOKENS"
	// Calls with tokens <= MinTokens keep the matvec kernel (decode and verify widths).
	vulkanMMCM1DefaultMinTokens = 8
	// Tile recorded in the activation line; must match BM/BN/BK in mm_cm1_quant.glsl.
	vulkanMMCM1Tile = "128x128x32"
)

// Shim status codes (vulkan_backend.h FVK_MM_CM1_*).
const (
	vulkanMMCM1StatusOK             = 0
	vulkanMMCM1StatusNotInitialized = 1
	vulkanMMCM1StatusNoCoopMat      = 2
	vulkanMMCM1StatusNoShaderF16    = 3
	vulkanMMCM1StatusNoF16Shape     = 4
	vulkanMMCM1StatusNoModule       = 5
	vulkanMMCM1StatusBadFormat      = 6
	vulkanMMCM1StatusBadShape       = 7
	vulkanMMCM1StatusDispatchFailed = 8
)

type vulkanMMCM1Format struct {
	id     int // fvk mm_cm1 format id (vulkan_shim.cpp FVK_MMCM1_*)
	kernel string
}

var vulkanMMCM1Formats = map[Dtype]vulkanMMCM1Format{
	Q2_K:    {id: 0, kernel: "mm_cm1_q2k"},
	IQ2_XXS: {id: 1, kernel: "mm_cm1_iq2xxs"},
	IQ2_XS:  {id: 2, kernel: "mm_cm1_iq2xs"},
	IQ2_S:   {id: 3, kernel: "mm_cm1_iq2s"},
	IQ3_XXS: {id: 4, kernel: "mm_cm1_iq3xxs"},
	IQ3_S:   {id: 5, kernel: "mm_cm1_iq3s"},
	IQ1_S:   {id: 6, kernel: "mm_cm1_iq1s"},
	IQ4_XS:  {id: 7, kernel: "mm_cm1_iq4xs"},
	Q4_K:    {id: 8, kernel: "mm_cm1_q4k"},
}

// VulkanMMCM1Config selects the prefill GEMM route. Enabled is the default
// (FAK_VULKAN_MM_CM1=0 turns it off); MinTokens is the largest token count that stays on matvec.
type VulkanMMCM1Config struct {
	Enabled   bool
	MinTokens int
}

func vulkanMMCM1ConfigFromEnv(getenv func(string) string) VulkanMMCM1Config {
	cfg := VulkanMMCM1Config{Enabled: getenv(vulkanMMCM1EnvEnable) != "0", MinTokens: vulkanMMCM1DefaultMinTokens}
	if s := getenv(vulkanMMCM1EnvMinTokens); s != "" {
		if n, err := strconv.Atoi(s); err == nil && n >= 1 {
			cfg.MinTokens = n
		}
	}
	return cfg
}

var (
	vulkanMMCM1ConfigOnce sync.Once
	vulkanMMCM1Config     atomic.Pointer[VulkanMMCM1Config]
)

func vulkanMMCM1CurrentConfig() VulkanMMCM1Config {
	vulkanMMCM1ConfigOnce.Do(func() {
		if vulkanMMCM1Config.Load() == nil {
			cfg := vulkanMMCM1ConfigFromEnv(os.Getenv)
			vulkanMMCM1Config.Store(&cfg)
		}
	})
	return *vulkanMMCM1Config.Load()
}

// vulkanMMCM1SetConfig replaces the route configuration and returns the previous one.
func vulkanMMCM1SetConfig(cfg VulkanMMCM1Config) VulkanMMCM1Config {
	prev := vulkanMMCM1CurrentConfig()
	vulkanMMCM1Config.Store(&cfg)
	return prev
}

// vulkanMMCM1Decide applies the host-side policy; a nil result still needs the device status.
func vulkanMMCM1Decide(cfg VulkanMMCM1Config, dt Dtype, tokens int) error {
	if !cfg.Enabled {
		return ErrVulkanMMCM1Disabled
	}
	if tokens <= cfg.MinTokens {
		return ErrVulkanMMCM1BelowThreshold
	}
	if _, ok := vulkanMMCM1Formats[dt]; !ok {
		return fmt.Errorf("%w: %v", ErrVulkanMMCM1UnsupportedDtype, dt)
	}
	return nil
}

func vulkanMMCM1StatusError(code int) error {
	switch code {
	case vulkanMMCM1StatusOK:
		return nil
	case vulkanMMCM1StatusNotInitialized:
		return fmt.Errorf("%w: backend not initialized", ErrVulkanMMCM1Unavailable)
	case vulkanMMCM1StatusNoCoopMat:
		return fmt.Errorf("%w: no KHR cooperative matrix with Wave32 subgroup control", ErrVulkanMMCM1Unavailable)
	case vulkanMMCM1StatusNoShaderF16:
		return fmt.Errorf("%w: shaderFloat16 not supported", ErrVulkanMMCM1Unavailable)
	case vulkanMMCM1StatusNoF16Shape:
		return fmt.Errorf("%w: no 16x16x16 fp16->fp32 subgroup cooperative matrix", ErrVulkanMMCM1Unavailable)
	case vulkanMMCM1StatusNoModule:
		return fmt.Errorf("%w: SPIR-V module missing or pipeline build failed", ErrVulkanMMCM1Unavailable)
	case vulkanMMCM1StatusBadFormat:
		return fmt.Errorf("%w: shim has no such format", ErrVulkanMMCM1UnsupportedDtype)
	case vulkanMMCM1StatusBadShape:
		return ErrVulkanMMCM1Shape
	case vulkanMMCM1StatusDispatchFailed:
		return ErrVulkanMMCM1Dispatch
	default:
		return fmt.Errorf("%w: unknown shim status %d", ErrVulkanMMCM1Unavailable, code)
	}
}

// VulkanMMCM1Stats counts prefill-GEMM routing decisions since process start.
type VulkanMMCM1Stats struct {
	GEMM             uint64
	Disabled         uint64
	BelowThreshold   uint64
	UnsupportedDtype uint64
	Unavailable      uint64
	Shape            uint64
	Dispatch         uint64
}

var vulkanMMCM1Counters struct {
	gemm, disabled, below, dtype, unavailable, shape, dispatch atomic.Uint64
	announced                                                  sync.Map
}

// VulkanMMCM1Snapshot returns the routing counters.
func VulkanMMCM1Snapshot() VulkanMMCM1Stats {
	c := &vulkanMMCM1Counters
	return VulkanMMCM1Stats{
		GEMM: c.gemm.Load(), Disabled: c.disabled.Load(), BelowThreshold: c.below.Load(),
		UnsupportedDtype: c.dtype.Load(), Unavailable: c.unavailable.Load(), Shape: c.shape.Load(),
		Dispatch: c.dispatch.Load(),
	}
}

// recordVulkanMMCM1 counts one routing decision. The first GEMM use and the first device
// refusal per dtype are also written to stderr, naming the kernel or the reason and the
// fallback kernel; policy outcomes (disabled, threshold) are counted only.
func recordVulkanMMCM1(dt Dtype, err error, fallback string) {
	c := &vulkanMMCM1Counters
	announce := func(key, line string) {
		if _, loaded := c.announced.LoadOrStore(key, true); !loaded {
			fmt.Fprintln(os.Stderr, line)
		}
	}
	switch {
	case err == nil:
		c.gemm.Add(1)
		cfg := vulkanMMCM1CurrentConfig()
		announce("gemm/"+dt.String(), fmt.Sprintf("fak-vulkan: mm_cm1 prefill GEMM active dtype=%v kernel=%s tile=%s min_tokens=%d",
			dt, vulkanMMCM1Formats[dt].kernel, vulkanMMCM1Tile, cfg.MinTokens))
	case errors.Is(err, ErrVulkanMMCM1Disabled):
		c.disabled.Add(1)
	case errors.Is(err, ErrVulkanMMCM1BelowThreshold):
		c.below.Add(1)
	case errors.Is(err, ErrVulkanMMCM1UnsupportedDtype):
		c.dtype.Add(1)
	case errors.Is(err, ErrVulkanMMCM1Shape):
		c.shape.Add(1)
		announce("shape/"+dt.String(), fmt.Sprintf("fak-vulkan: mm_cm1 fallback dtype=%v reason=%q fallback=%s", dt, err, fallback))
	case errors.Is(err, ErrVulkanMMCM1Dispatch):
		c.dispatch.Add(1)
		announce("dispatch/"+dt.String(), fmt.Sprintf("fak-vulkan: mm_cm1 fallback dtype=%v reason=%q fallback=%s", dt, err, fallback))
	default:
		c.unavailable.Add(1)
		announce("unavailable/"+dt.String(), fmt.Sprintf("fak-vulkan: mm_cm1 fallback dtype=%v reason=%q fallback=%s", dt, err, fallback))
	}
}
