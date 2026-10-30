package computebuild

import (
	"context"
	"os"
)

// CUDAV41DefaultsEnv opts a single-arch sm_120 build out of the default V4.1
// selection when set to "0". Any other value (or unset) keeps the default.
const CUDAV41DefaultsEnv = "FAK_CUDA_V41"

type cudaV41DefaultsKey struct{}

// cudaV41DefaultSelections is the default sm_120 V4.1 set: only candidates
// with a device witness on sm_120. Unwitnessed candidates (tail RoPE, BF16
// publish, mHC, Engram gate, routed SwiGLU, the pool-only compressor) stay
// explicit opt-in flags. The compressor default is the ds4 direct port
// (fak_cuda_v41_compressor); an explicit --v41-compressor-pool suppresses it
// because the two tags are mutually exclusive.
var cudaV41DefaultSelections = []struct {
	tag    string
	enable func(*CUDAConfig) *bool
	skip   func(*CUDAConfig) bool
}{
	{"fak_cuda_v41_compressor", func(c *CUDAConfig) *bool { return &c.EnableV41Compressor }, func(c *CUDAConfig) bool { return c.EnableV41CompressorPool }},
	{"fak_cuda_v41_router", func(c *CUDAConfig) *bool { return &c.EnableV41Router }, nil},
	{"fak_cuda_v41_shared_attention", func(c *CUDAConfig) *bool { return &c.EnableV41SharedAttention }, nil},
}

// CUDAV41DefaultTags lists the Go tags the default sm_120 build selects.
func CUDAV41DefaultTags() []string {
	tags := make([]string, 0, len(cudaV41DefaultSelections))
	for _, s := range cudaV41DefaultSelections {
		tags = append(tags, s.tag)
	}
	return tags
}

func cudaV41DefaultArch(arch string) bool {
	return arch == "sm_120" || arch == "120"
}

// ApplyCUDAV41Defaults turns on the default V4.1 selections for a single-arch
// sm_120 build unless DisableV41Defaults is set, and returns the tags it newly
// enabled. Fat multi-arch builds are left untouched: every V4.1 candidate
// requires an explicit sm_120 target. Idempotent.
func ApplyCUDAV41Defaults(cfg *CUDAConfig) []string {
	if cfg == nil || cfg.DisableV41Defaults || !cudaV41DefaultArch(cfg.Arch) {
		return nil
	}
	var applied []string
	for _, s := range cudaV41DefaultSelections {
		flag := s.enable(cfg)
		if *flag || (s.skip != nil && s.skip(cfg)) {
			continue
		}
		*flag = true
		applied = append(applied, s.tag)
	}
	cfg.V41Defaults = append(cfg.V41Defaults, applied...)
	return applied
}

// CUDAV41DefaultsDisabledByEnv reports the FAK_CUDA_V41=0 opt-out.
func CUDAV41DefaultsDisabledByEnv() bool {
	return os.Getenv(CUDAV41DefaultsEnv) == "0"
}

func cudaV41DefaultsFromContext(ctx context.Context) []string {
	tags, _ := ctx.Value(cudaV41DefaultsKey{}).([]string)
	return append([]string(nil), tags...)
}
