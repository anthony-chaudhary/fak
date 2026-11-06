package computebuild

import (
	"errors"
	"reflect"
	"testing"
)

// fak-test:runtime fast est=5ms lane=default
// Every V4.1 candidate gate shares one allowlist: single-arch sm_120 or sm_90
// (bare or sm_-prefixed) passes the arch gate; everything else, including fat
// and multi-arch builds, is refused with the closed ErrCUDAV41Arch sentinel.
func TestCUDAV41ArchAllowlist(t *testing.T) {
	if want := []string{"sm_120", "sm_90"}; !reflect.DeepEqual(cudaV41Arches, want) {
		t.Fatalf("cudaV41Arches = %v, want %v", cudaV41Arches, want)
	}
	prepares := map[string]func(*CUDAConfig) error{
		"bf16_publish":     prepareCUDAV41BF16Publish,
		"compressor":       prepareCUDAV41Compressor,
		"compressor_pool":  prepareCUDAV41CompressorPool,
		"engram_gate":      prepareCUDAV41EngramGate,
		"mhc":              prepareCUDAV41MHC,
		"routed_swiglu":    prepareCUDAV41RoutedSwiGLU,
		"router":           prepareCUDAV41Router,
		"shared_attention": prepareCUDAV41SharedAttention,
		"tail_rope":        prepareCUDAV41TailRoPE,
	}
	admitted := []string{"sm_120", "120", "sm_90", "90"}
	refused := []string{"", "sm_89", "89", "sm_80", "sm_90a", "sm_100", "sm_100a", "sm_103f", "sm_90,sm_120", "sm_120,sm_90"}
	for _, arch := range admitted {
		if !cudaV41ArchAdmitted(arch) || requireCUDAV41Arch(arch, "x") != nil {
			t.Fatalf("arch %q not admitted", arch)
		}
	}
	for _, arch := range refused {
		if cudaV41ArchAdmitted(arch) || !errors.Is(requireCUDAV41Arch(arch, "x"), ErrCUDAV41Arch) {
			t.Fatalf("arch %q admitted", arch)
		}
	}
	for name, prepare := range prepares {
		for _, arch := range admitted {
			// An empty repo root fails later (sources/arch matrix), never at the arch gate.
			if err := prepare(&CUDAConfig{Arch: arch, RepoRoot: t.TempDir()}); err == nil || errors.Is(err, ErrCUDAV41Arch) {
				t.Fatalf("%s arch %q: err=%v, want a post-gate failure", name, arch, err)
			}
		}
		for _, arch := range refused {
			if err := prepare(&CUDAConfig{Arch: arch, RepoRoot: t.TempDir()}); !errors.Is(err, ErrCUDAV41Arch) {
				t.Fatalf("%s arch %q: err=%v, want ErrCUDAV41Arch", name, arch, err)
			}
		}
	}
}
