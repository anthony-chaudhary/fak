package compute

import (
	"errors"
	"math"
	"testing"
)

// fak-test:runtime fast est=5ms lane=default
func TestDeepGemmHeuristicsDefaultOffAndFailClosed(t *testing.T) {
	t.Parallel()

	desc := SM90GemmDesc{
		GemmType:     SM90GemmNormal,
		KernelType:   SM90Kernel1D2D,
		M:            128,
		N:            4096,
		K:            4096,
		NumGroups:    1,
		ADType:       SM90DTypeFP8E4M3,
		BDType:       SM90DTypeFP8E4M3,
		CDDType:      SM90DTypeBF16,
		MajorA:       SM90MajorK,
		MajorB:       SM90MajorK,
		NumSMs:       132,
		CompiledDims: "nk",
	}

	// Zero-value options keep SM90 heuristics disabled by default.
	var defaultOpts SM90HeuristicsOptions
	if _, err := SelectSM90GemmConfig(defaultOpts, desc); !errors.Is(err, ErrSM90HeuristicsDisabled) {
		t.Fatalf("SelectSM90GemmConfig default off err = %v, want ErrSM90HeuristicsDisabled", err)
	}
	reg := NewSM90PrecompileRegistry(defaultOpts)
	if _, _, err := reg.ResolveOrCompile("sm90_fp8_gemm_1d2d", desc, nil); !errors.Is(err, ErrSM90HeuristicsDisabled) {
		t.Fatalf("ResolveOrCompile default off err = %v, want ErrSM90HeuristicsDisabled", err)
	}

	// Enabled rejects invalid descriptors cleanly.
	opts := SM90HeuristicsOptions{Enabled: true}
	oddSM := desc
	oddSM.NumSMs = 131
	if _, err := SelectSM90GemmConfig(opts, oddSM); !errors.Is(err, ErrSM90InvalidDesc) {
		t.Fatalf("odd NumSMs err = %v, want ErrSM90InvalidDesc", err)
	}

	badCombo := desc
	badCombo.ADType = SM90DTypeBF16
	badCombo.BDType = SM90DTypeBF16
	if _, err := SelectSM90GemmConfig(opts, badCombo); !errors.Is(err, ErrSM90InvalidDesc) {
		t.Fatalf("bf16 with 1D2D err = %v, want ErrSM90InvalidDesc", err)
	}

	bad1D1DMajor := desc
	bad1D1DMajor.KernelType = SM90Kernel1D1D
	bad1D1DMajor.CDDType = SM90DTypeFP32
	bad1D1DMajor.MajorB = SM90MajorMN
	if _, err := SelectSM90GemmConfig(opts, bad1D1DMajor); !errors.Is(err, ErrSM90InvalidDesc) {
		t.Fatalf("1D1D fp32 non-K-major err = %v, want ErrSM90InvalidDesc", err)
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestDeepGemmHeuristicsFixedShapeOracle(t *testing.T) {
	t.Parallel()

	opts := SM90HeuristicsOptions{Enabled: true}

	// Fixed-shape closed oracle over DeepGEMM SM90 heuristics (132-SM geometry).
	cases := []struct {
		name           string
		desc           SM90GemmDesc
		wantBlockM     int
		wantBlockN     int
		wantBlockK     int
		wantClusterSz  int
		wantMinStages  int
		wantTMAThreads int
		wantMathThread int
	}{
		{
			name: "fp8-1d2d-small-m16-nk4096",
			desc: SM90GemmDesc{
				GemmType:     SM90GemmNormal,
				KernelType:   SM90Kernel1D2D,
				M:            16,
				N:            4096,
				K:            4096,
				NumGroups:    1,
				ADType:       SM90DTypeFP8E4M3,
				BDType:       SM90DTypeFP8E4M3,
				CDDType:      SM90DTypeBF16,
				MajorA:       SM90MajorK,
				MajorB:       SM90MajorK,
				NumSMs:       132,
				CompiledDims: "nk",
			},
			wantBlockM:     16,
			wantBlockN:     32,
			wantBlockK:     128,
			wantClusterSz:  1, // 1 wave <= 132 SMs disables multicast
			wantMinStages:  4,
			wantTMAThreads: 128,
			wantMathThread: 128,
		},
		{
			name: "fp8-1d2d-large-m2048-n4096-k7168",
			desc: SM90GemmDesc{
				GemmType:     SM90GemmNormal,
				KernelType:   SM90Kernel1D2D,
				M:            2048,
				N:            4096,
				K:            7168,
				NumGroups:    1,
				ADType:       SM90DTypeFP8E4M3,
				BDType:       SM90DTypeFP8E4M3,
				CDDType:      SM90DTypeBF16,
				MajorA:       SM90MajorK,
				MajorB:       SM90MajorK,
				NumSMs:       132,
				CompiledDims: "nk",
			},
			wantBlockM:     256,
			wantBlockN:     128,
			wantBlockK:     128,
			wantClusterSz:  2, // 256 blocks = 2 waves > 1 enables multicast
			wantMinStages:  3,
			wantTMAThreads: 128,
			wantMathThread: 256,
		},
		{
			name: "fp8-1d1d-fp32-out-bank-conflict-n24-step",
			desc: SM90GemmDesc{
				GemmType:         SM90GemmNormal,
				KernelType:       SM90Kernel1D1D,
				M:                1024,
				N:                2048,
				K:                4096,
				NumGroups:        1,
				ADType:           SM90DTypeFP8E4M3,
				BDType:           SM90DTypeFP8E4M3,
				CDDType:          SM90DTypeFP32,
				MajorA:           SM90MajorK,
				MajorB:           SM90MajorK,
				WithAccumulation: true,
				NumSMs:           132,
				CompiledDims:     "nk",
			},
			wantBlockM:     128,
			wantBlockN:     136, // 24 + 7*16: bank-conflict avoidance start=24 step=16
			wantBlockK:     128,
			wantClusterSz:  1, // 1 wave <= 132 SMs disables multicast
			wantMinStages:  3,
			wantTMAThreads: 128,
			wantMathThread: 256,
		},
		{
			name: "bf16-nosf-m512-n4096-k4096",
			desc: SM90GemmDesc{
				GemmType:     SM90GemmNormal,
				KernelType:   SM90KernelNoSF,
				M:            512,
				N:            4096,
				K:            4096,
				NumGroups:    1,
				ADType:       SM90DTypeBF16,
				BDType:       SM90DTypeBF16,
				CDDType:      SM90DTypeBF16,
				MajorA:       SM90MajorK,
				MajorB:       SM90MajorK,
				NumSMs:       132,
				CompiledDims: "nk",
			},
			wantBlockM:     128,
			wantBlockN:     128,
			wantBlockK:     64, // 128*8/16
			wantClusterSz:  1,  // 1 wave <= 132 SMs disables multicast
			wantMinStages:  4,
			wantTMAThreads: 128,
			wantMathThread: 256,
		},
		{
			name: "fp8-batched-disables-multicast",
			desc: SM90GemmDesc{
				GemmType:     SM90GemmBatched,
				KernelType:   SM90Kernel1D2D,
				M:            1024,
				N:            1024,
				K:            2048,
				NumGroups:    8,
				ADType:       SM90DTypeFP8E4M3,
				BDType:       SM90DTypeFP8E4M3,
				CDDType:      SM90DTypeBF16,
				MajorA:       SM90MajorK,
				MajorB:       SM90MajorK,
				NumSMs:       132,
				CompiledDims: "nk",
			},
			wantBlockM:     256,
			wantBlockN:     128,
			wantBlockK:     128,
			wantClusterSz:  1, // Batched disables multicast even when num_waves == 2 > 1
			wantMinStages:  3,
			wantTMAThreads: 128,
			wantMathThread: 256,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg, err := SelectSM90GemmConfig(opts, tc.desc)
			if err != nil {
				t.Fatalf("SelectSM90GemmConfig: %v", err)
			}
			// Pin selection deterministically across repeated evaluations.
			cfg2, err := SelectSM90GemmConfig(opts, tc.desc)
			if err != nil {
				t.Fatalf("second SelectSM90GemmConfig: %v", err)
			}
			if cfg != cfg2 {
				t.Fatalf("non-deterministic selection: %+v != %+v", cfg, cfg2)
			}

			if cfg.Layout.BlockM != tc.wantBlockM || cfg.Layout.BlockN != tc.wantBlockN || cfg.Layout.BlockK != tc.wantBlockK {
				t.Fatalf("tile = (%d,%d,%d), want (%d,%d,%d)",
					cfg.Layout.BlockM, cfg.Layout.BlockN, cfg.Layout.BlockK,
					tc.wantBlockM, tc.wantBlockN, tc.wantBlockK)
			}
			if cfg.Layout.ClusterSize() != tc.wantClusterSz {
				t.Fatalf("cluster_size = %d, want %d", cfg.Layout.ClusterSize(), tc.wantClusterSz)
			}
			if cfg.Pipeline.NumStages < tc.wantMinStages {
				t.Fatalf("num_stages = %d, want >= %d", cfg.Pipeline.NumStages, tc.wantMinStages)
			}
			if cfg.Pipeline.SMEMSize > sm90SmemCapacityBytes {
				t.Fatalf("smem %d exceeds SM90 capacity %d", cfg.Pipeline.SMEMSize, sm90SmemCapacityBytes)
			}
			if cfg.Launch.NumTMAThreads != tc.wantTMAThreads || cfg.Launch.NumMathThreads != tc.wantMathThread {
				t.Fatalf("threads = (%d,%d), want (%d,%d)",
					cfg.Launch.NumTMAThreads, cfg.Launch.NumMathThreads,
					tc.wantTMAThreads, tc.wantMathThread)
			}
			if cfg.Storage.SwizzleAMode%64 != 0 || cfg.Storage.SwizzleBMode%64 != 0 {
				t.Fatalf("swizzle modes (%d,%d) must be multiples of 64", cfg.Storage.SwizzleAMode, cfg.Storage.SwizzleBMode)
			}

			// Verify selected candidate achieves minimum modeled cycle count across all valid candidates.
			cands, err := SM90LayoutCandidates(opts, tc.desc)
			if err != nil {
				t.Fatalf("SM90LayoutCandidates: %v", err)
			}
			minCycles := int64(math.MaxInt64)
			for _, cand := range cands {
				info := sm90LayoutInfo(tc.desc, cand)
				if info.NumCycles < minCycles {
					minCycles = info.NumCycles
				}
			}
			if cfg.Info.NumCycles != minCycles {
				t.Fatalf("selected cycles %d != min candidate cycles %d", cfg.Info.NumCycles, minCycles)
			}
		})
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestDeepGemmHeuristicsPrecompileManifestAndHitShortCircuit(t *testing.T) {
	t.Parallel()

	reg := NewSM90PrecompileRegistry(SM90HeuristicsOptions{Enabled: true})
	compileCalls := 0
	compileFn := func(sig SM90KernelSignature) error {
		compileCalls++
		return nil
	}

	base := SM90GemmDesc{
		GemmType:     SM90GemmNormal,
		KernelType:   SM90Kernel1D2D,
		N:            4096,
		K:            7168,
		NumGroups:    1,
		ADType:       SM90DTypeFP8E4M3,
		BDType:       SM90DTypeFP8E4M3,
		CDDType:      SM90DTypeBF16,
		MajorA:       SM90MajorK,
		MajorB:       SM90MajorK,
		NumSMs:       132,
		CompiledDims: "nk",
	}

	// Precompile across M buckets where dynamic M collapses to CompiledM=0.
	// M=1920 and M=2048 both select (256, 128, 128) with identical specialization
	// keys, so Precompile must compile once and hit on the duplicate signature.
	m1920 := base
	m1920.M = 1920
	m2048 := base
	m2048.M = 2048
	m16 := base
	m16.M = 16

	stats, err := reg.Precompile("sm90_fp8_gemm_1d2d", []SM90GemmDesc{m1920, m2048, m16}, compileFn)
	if err != nil {
		t.Fatalf("Precompile: %v", err)
	}
	if stats.CompileCalls != 2 || stats.ManifestEntries != 2 || stats.CacheHits != 1 {
		t.Fatalf("after precompile stats = %+v, want CompileCalls=2 CacheHits=1 ManifestEntries=2", stats)
	}

	// Subsequent runtime lookup for M=2000 (same tile config, dynamic M -> CompiledM=0)
	// must hit the registry and short-circuit compileFn completely.
	m2000 := base
	m2000.M = 2000
	beforeCalls := compileCalls
	entry, hit, err := reg.ResolveOrCompile("sm90_fp8_gemm_1d2d", m2000, func(sig SM90KernelSignature) error {
		t.Fatalf("registry hit must short-circuit compileFn, got compile call for %s", sig.Key())
		return nil
	})
	if err != nil {
		t.Fatalf("ResolveOrCompile hit: %v", err)
	}
	if !hit {
		t.Fatal("expected registry hit for precompiled signature")
	}
	if compileCalls != beforeCalls {
		t.Fatalf("compileCalls grew from %d to %d on cache hit", beforeCalls, compileCalls)
	}
	if entry.Signature.CompiledM != 0 || entry.Signature.CompiledN != 4096 || entry.Signature.CompiledK != 7168 {
		t.Fatalf("compiled dims = (%d,%d,%d), want (0,4096,7168)",
			entry.Signature.CompiledM, entry.Signature.CompiledN, entry.Signature.CompiledK)
	}
	manifest := reg.Manifest()
	if len(manifest) != 2 {
		t.Fatalf("manifest len = %d, want 2", len(manifest))
	}
	if gotStats := reg.Stats(); gotStats.CacheHits != 2 || gotStats.CompileCalls != 2 {
		t.Fatalf("final stats = %+v, want CompileCalls=2 CacheHits=2", gotStats)
	}
}
