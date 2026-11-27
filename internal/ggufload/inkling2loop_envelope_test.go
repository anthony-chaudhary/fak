package ggufload

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

func testEnvelopeHParams() model.Inkling2LoopHParams {
	local := make([]bool, 24)
	for i := range local {
		local[i] = i%4 != 3
	}
	return model.Inkling2LoopHParams{
		Layers: 24, CoreStart: 6, CoreEnd: 18, Passes: 3,
		Hidden: 2304, Heads: 16, KVHeads: 8, HeadDim: 128,
		SlidingWindow: 512, ConvKernel: 4,
		LocalLayer: local,
	}
}

func ptrKVDtype(d Inkling2KVDType) *Inkling2KVDType { return &d }

func TestInkling2ServeEnvelope(t *testing.T) {
	hp := testEnvelopeHParams()

	// Hardware envelopes across 12 GiB (5070 eGPU), 16 GiB (5080), 24 GiB (5090 Laptop),
	// 32 GiB (5090), 40 GiB A100, and 80 GiB A100, plus f32/f16/fp8/nvfp4 capabilities.
	cases := []struct {
		name           string
		freeBytes      int64
		caps           Inkling2ServeCaps
		overrides      Inkling2ServeOverrides
		wantKVDtype    Inkling2KVDType
		wantDefChunk   int
		wantMinChunk   int
		wantMinCtx     int
		wantMaxCtx     int
		wantMinC       int
		wantMaxC       int
		checkC1Exact   bool
		wantC1AtMaxCtx bool
	}{
		{
			name:         "12GiB_5070_eGPU_f16",
			freeBytes:    1800 << 20, // ~1.76 GiB free after weights
			caps:         Inkling2ServeCaps{BackendName: "cuda", BatchEligible: true, F16Ready: true},
			wantKVDtype:  Inkling2KVF16,
			wantDefChunk: 512,
			wantMinChunk: 128,
			wantMinCtx:   4096,
			wantMaxCtx:   16384,
			wantMinC:     1,
			wantMaxC:     4,
			checkC1Exact: true,
		},
		{
			name:         "12GiB_5070_eGPU_nvfp4_KL",
			freeBytes:    1800 << 20,
			caps:         Inkling2ServeCaps{BackendName: "cuda", BatchEligible: true, F16Ready: true, FP8Ready: true, FP8KLQualified: true, NVFP4Ready: true, NVFP4KLQualified: true},
			wantKVDtype:  Inkling2KVNVFP4,
			wantDefChunk: 512,
			wantMinChunk: 128,
			wantMinCtx:   4096,
			wantMaxCtx:   16384,
			wantMinC:     1,
			wantMaxC:     8,
			checkC1Exact: true,
		},
		{
			name:         "16GiB_5080_f32",
			freeBytes:    2534 << 20, // ~2.47 GiB free after weights
			caps:         Inkling2ServeCaps{BackendName: "cuda", BatchEligible: true},
			wantKVDtype:  Inkling2KVF32,
			wantDefChunk: 512,
			wantMinChunk: 128,
			wantMinCtx:   4096,
			wantMaxCtx:   16384,
			wantMinC:     1,
			wantMaxC:     4,
			checkC1Exact: true,
		},
		{
			name:         "16GiB_5080_f16",
			freeBytes:    2534 << 20,
			caps:         Inkling2ServeCaps{BackendName: "cuda", BatchEligible: true, F16Ready: true},
			wantKVDtype:  Inkling2KVF16,
			wantDefChunk: 512,
			wantMinChunk: 128,
			wantMinCtx:   16384,
			wantMaxCtx:   16384,
			wantMinC:     2,
			wantMaxC:     4,
			checkC1Exact: true,
		},
		{
			name:         "16GiB_5080_fp8_KL",
			freeBytes:    2534 << 20,
			caps:         Inkling2ServeCaps{BackendName: "cuda", BatchEligible: true, F16Ready: true, FP8Ready: true, FP8KLQualified: true},
			wantKVDtype:  Inkling2KVFP8,
			wantDefChunk: 512,
			wantMinChunk: 128,
			wantMinCtx:   16384,
			wantMaxCtx:   16384,
			wantMinC:     2,
			wantMaxC:     8,
			checkC1Exact: true,
		},
		{
			name:         "16GiB_5080_fp8_unqualified_stays_f16",
			freeBytes:    2534 << 20,
			caps:         Inkling2ServeCaps{BackendName: "cuda", BatchEligible: true, F16Ready: true, FP8Ready: true, FP8KLQualified: false},
			wantKVDtype:  Inkling2KVF16,
			wantDefChunk: 512,
			wantMinChunk: 128,
			wantMinCtx:   16384,
			wantMaxCtx:   16384,
			wantMinC:     2,
			wantMaxC:     4,
			checkC1Exact: true,
		},
		{
			name:         "24GiB_5090_Laptop_f16",
			freeBytes:    10 << 30,
			caps:         Inkling2ServeCaps{BackendName: "cuda", BatchEligible: true, F16Ready: true},
			wantKVDtype:  Inkling2KVF16,
			wantDefChunk: 1024,
			wantMinChunk: 128,
			wantMinCtx:   65536,
			wantMaxCtx:   262144,
			wantMinC:     8,
			wantMaxC:     16,
			checkC1Exact: true,
		},
		{
			name:         "24GiB_5090_Laptop_fp8_KL",
			freeBytes:    10 << 30,
			caps:         Inkling2ServeCaps{BackendName: "cuda", BatchEligible: true, F16Ready: true, FP8Ready: true, FP8KLQualified: true},
			wantKVDtype:  Inkling2KVFP8,
			wantDefChunk: 1024,
			wantMinChunk: 128,
			wantMinCtx:   131072,
			wantMaxCtx:   262144,
			wantMinC:     12,
			wantMaxC:     16,
			checkC1Exact: true,
		},
		{
			name:         "32GiB_5090_f16",
			freeBytes:    18 << 30,
			caps:         Inkling2ServeCaps{BackendName: "cuda", BatchEligible: true, F16Ready: true},
			wantKVDtype:  Inkling2KVF16,
			wantDefChunk: 1024,
			wantMinChunk: 128,
			wantMinCtx:   262144,
			wantMaxCtx:   262144,
			wantMinC:     16,
			wantMaxC:     Inkling2LoopMaxBatch,
			checkC1Exact: true,
		},
		{
			name:         "32GiB_5090_nvfp4_KL",
			freeBytes:    18 << 30,
			caps:         Inkling2ServeCaps{BackendName: "cuda", BatchEligible: true, F16Ready: true, FP8Ready: true, FP8KLQualified: true, NVFP4Ready: true, NVFP4KLQualified: true},
			wantKVDtype:  Inkling2KVNVFP4,
			wantDefChunk: 1024,
			wantMinChunk: 128,
			wantMinCtx:   262144,
			wantMaxCtx:   262144,
			wantMinC:     32,
			wantMaxC:     Inkling2LoopMaxBatch,
			checkC1Exact: true,
		},
		{
			name:           "40GiB_A100_f16",
			freeBytes:      26 << 30,
			caps:           Inkling2ServeCaps{BackendName: "cuda", BatchEligible: true, F16Ready: true},
			wantKVDtype:    Inkling2KVF16,
			wantDefChunk:   1024,
			wantMinChunk:   128,
			wantMinCtx:     262144,
			wantMaxCtx:     262144,
			wantMinC:       24,
			wantMaxC:       Inkling2LoopMaxBatch,
			wantC1AtMaxCtx: true,
		},
		{
			name:           "80GiB_A100_f16",
			freeBytes:      64 << 30,
			caps:           Inkling2ServeCaps{BackendName: "cuda", BatchEligible: true, F16Ready: true},
			wantKVDtype:    Inkling2KVF16,
			wantDefChunk:   1024,
			wantMinChunk:   128,
			wantMinCtx:     262144,
			wantMaxCtx:     262144,
			wantMinC:       Inkling2LoopMaxBatch,
			wantMaxC:       Inkling2LoopMaxBatch,
			wantC1AtMaxCtx: true,
		},
		{
			name:         "80GiB_A100_f32_caps",
			freeBytes:    64 << 30,
			caps:         Inkling2ServeCaps{BackendName: "cuda", BatchEligible: true},
			wantKVDtype:  Inkling2KVF32,
			wantDefChunk: 1024,
			wantMinChunk: 128,
			wantMinCtx:   16384,
			wantMaxCtx:   16384,
			wantMinC:     24,
			wantMaxC:     Inkling2LoopMaxBatch,
			checkC1Exact: true,
		},
		{
			name:         "Pin_RequestedCtx_overrides_auto",
			freeBytes:    64 << 30,
			caps:         Inkling2ServeCaps{BackendName: "cuda", BatchEligible: true, F16Ready: true},
			overrides:    Inkling2ServeOverrides{RequestedCtx: 24576},
			wantKVDtype:  Inkling2KVF16,
			wantDefChunk: 1024,
			wantMinChunk: 128,
			wantMinCtx:   24576,
			wantMaxCtx:   24576,
			wantMinC:     Inkling2LoopMaxBatch,
			wantMaxC:     Inkling2LoopMaxBatch,
		},
		{
			name:         "Pin_EnvCtx_overrides_auto",
			freeBytes:    64 << 30,
			caps:         Inkling2ServeCaps{BackendName: "cuda", BatchEligible: true, F16Ready: true},
			overrides:    Inkling2ServeOverrides{EnvCtx: 8192},
			wantKVDtype:  Inkling2KVF16,
			wantDefChunk: 1024,
			wantMinChunk: 128,
			wantMinCtx:   8192,
			wantMaxCtx:   8192,
			wantMinC:     Inkling2LoopMaxBatch,
			wantMaxC:     Inkling2LoopMaxBatch,
		},
		{
			name:         "Pin_KVDtype_and_PrefillChunk_and_BatchRows",
			freeBytes:    64 << 30,
			caps:         Inkling2ServeCaps{BackendName: "cuda", BatchEligible: true, F16Ready: true, FP8Ready: true, FP8KLQualified: true},
			overrides:    Inkling2ServeOverrides{KVDtype: ptrKVDtype(Inkling2KVF32), PrefillChunk: 256, ServeBatchRows: 4},
			wantKVDtype:  Inkling2KVF32,
			wantDefChunk: 256,
			wantMinChunk: 128,
			wantMinCtx:   16384,
			wantMaxCtx:   16384,
			wantMinC:     4,
			wantMaxC:     4,
		},
		{
			name:         "Pin_BatchDisabled_forces_C1",
			freeBytes:    64 << 30,
			caps:         Inkling2ServeCaps{BackendName: "cuda", BatchEligible: true, F16Ready: true},
			overrides:    Inkling2ServeOverrides{BatchDisabled: true},
			wantKVDtype:  Inkling2KVF16,
			wantDefChunk: 1024,
			wantMinChunk: 128,
			wantMinCtx:   262144,
			wantMaxCtx:   262144,
			wantMinC:     1,
			wantMaxC:     1,
		},
		{
			name:         "UnknownFree_falls_back_safely",
			freeBytes:    compute.FreeUnknown,
			caps:         Inkling2ServeCaps{BackendName: "cuda", BatchEligible: true, F16Ready: true},
			wantKVDtype:  Inkling2KVF16,
			wantDefChunk: 512,
			wantMinChunk: 128,
			wantMinCtx:   4096,
			wantMaxCtx:   4096,
			wantMinC:     1,
			wantMaxC:     1,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := ResolveInkling2ServeEnvelope(hp, tc.freeBytes, tc.caps, tc.overrides)
			if got.KVDtype != tc.wantKVDtype {
				t.Fatalf("KVDtype = %s, want %s", got.KVDtype, tc.wantKVDtype)
			}
			if got.DefaultChunk != tc.wantDefChunk {
				t.Fatalf("DefaultChunk = %d, want %d", got.DefaultChunk, tc.wantDefChunk)
			}
			if got.MinChunk != tc.wantMinChunk {
				t.Fatalf("MinChunk = %d, want %d", got.MinChunk, tc.wantMinChunk)
			}
			if got.MaxCtx < tc.wantMinCtx || got.MaxCtx > tc.wantMaxCtx {
				t.Fatalf("MaxCtx = %d, want in [%d, %d]", got.MaxCtx, tc.wantMinCtx, tc.wantMaxCtx)
			}
			if got.TargetConcurrency < tc.wantMinC || got.TargetConcurrency > tc.wantMaxC {
				t.Fatalf("TargetConcurrency = %d, want in [%d, %d]", got.TargetConcurrency, tc.wantMinC, tc.wantMaxC)
			}
			if tc.wantC1AtMaxCtx && got.LongCtxC1Threshold != got.MaxCtx {
				t.Fatalf("LongCtxC1Threshold = %d, want MaxCtx (%d)", got.LongCtxC1Threshold, got.MaxCtx)
			}
			if tc.checkC1Exact {
				if got.LongCtxC1Threshold > 0 && !inkling2EnvelopeBatchFits(hp, 2, got.LongCtxC1Threshold, got.MinChunk, got.KVDtype, tc.freeBytes) {
					t.Fatalf("C=2 at LongCtxC1Threshold=%d MinChunk=%d should fit in %d bytes", got.LongCtxC1Threshold, got.MinChunk, tc.freeBytes)
				}
				if next := got.LongCtxC1Threshold + inkling2BatchCtxAlign; next <= got.MaxCtx {
					if inkling2EnvelopeBatchFits(hp, 2, next, got.MinChunk, got.KVDtype, tc.freeBytes) {
						t.Fatalf("C=2 at LongCtxC1Threshold+%d=%d MinChunk=%d should NOT fit in %d bytes", inkling2BatchCtxAlign, next, got.MinChunk, tc.freeBytes)
					}
				}
			}
		})
	}
}
