package ggufload

import (
	"os"
	"slices"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

type fakeLowBitSpecBackend struct {
	fakeHalfSpecBackend
	compute.Inkling2VerifyFP8KVDevice
	compute.Inkling2VerifyNVFP4KVDevice
	name string
}

func (b fakeLowBitSpecBackend) Name() string {
	if b.name != "" {
		return b.name
	}
	return "cuda"
}

func (fakeLowBitSpecBackend) Ink2Alloc(n int) (compute.Tensor, error) {
	return compute.Tensor{Dtype: compute.F32}, nil
}
func (fakeLowBitSpecBackend) Free(compute.Tensor)                        {}
func (fakeLowBitSpecBackend) Ink2Write(compute.Ink2Ref, []float32) error { return nil }

func (fakeLowBitSpecBackend) Ink2PFP8KVReady() error { return nil }
func (fakeLowBitSpecBackend) Ink2QKVPostFP8(compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2AttnParams, int, *compute.Ink2KVConv) error {
	return nil
}
func (fakeLowBitSpecBackend) Ink2AttendFP8(compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2AttnParams, int, int, int) error {
	return nil
}
func (fakeLowBitSpecBackend) Ink2PQKVPostFP8(compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2AttnParams, int, int, int) error {
	return nil
}
func (fakeLowBitSpecBackend) Ink2PAttendFP8(compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2AttnParams, int, int, int, int, int) error {
	return nil
}

func (fakeLowBitSpecBackend) Ink2PNVFP4KVReady() error { return nil }
func (fakeLowBitSpecBackend) Ink2QKVPostNVFP4(compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2AttnParams, int, *compute.Ink2KVConv) error {
	return nil
}
func (fakeLowBitSpecBackend) Ink2AttendNVFP4(compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2AttnParams, int, int, int) error {
	return nil
}
func (fakeLowBitSpecBackend) Ink2PQKVPostNVFP4(compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2AttnParams, int, int, int) error {
	return nil
}
func (fakeLowBitSpecBackend) Ink2PAttendNVFP4(compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2AttnParams, int, int, int, int, int) error {
	return nil
}

func (fakeLowBitSpecBackend) Ink2PAsymKVReady() error { return nil }
func (fakeLowBitSpecBackend) Ink2QKVPostAsym(compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2AttnParams, int, *compute.Ink2KVConv) error {
	return nil
}
func (fakeLowBitSpecBackend) Ink2AttendAsym(compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2AttnParams, int, int, int) error {
	return nil
}
func (fakeLowBitSpecBackend) Ink2PQKVPostAsym(compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2AttnParams, int, int, int) error {
	return nil
}
func (fakeLowBitSpecBackend) Ink2PAttendAsym(compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2AttnParams, int, int, int, int, int) error {
	return nil
}

type fakeLowBitMemBackend struct {
	fakeLowBitSpecBackend
	freeBytes int64
}

func (b fakeLowBitMemBackend) DeviceMemoryInfo() (total, free int64) {
	return b.freeBytes, b.freeBytes
}

// fak-test:runtime fast est=50ms
func TestInkling2SpecLowBitKVGating(t *testing.T) {
	be := fakeLowBitSpecBackend{}
	for _, mode := range []string{"fp8", "nvfp4", "fp8k_nvfp4v"} {
		lookup := func(k string) (string, bool) {
			if k == "FAK_INKLING2_KV" {
				return mode, true
			}
			return "", false
		}
		if inkling2SpecAutoDisabledForKV(be, lookup) {
			t.Fatalf("inkling2SpecAutoDisabledForKV(%s) = true, want false", mode)
		}
	}
	f16Lookup := func(k string) (string, bool) {
		if k == "FAK_INKLING2_KV" {
			return "f16", true
		}
		return "", false
	}
	if !inkling2SpecAutoDisabledForKV(be, f16Lookup) {
		t.Fatal("inkling2SpecAutoDisabledForKV(f16) = false, want true")
	}

	hp := model.Inkling2LoopHParams{Hidden: 64, KVHeads: 2, HeadDim: 32, ConvKernel: 4}
	for _, tc := range []struct {
		name string
		res  *inkling2Resident
		want bool
	}{
		{name: "fp8", res: &inkling2Resident{be: be, dev: be, hp: hp, batched: true, kvFP8: true, hist: make([]ink2Ref, 1), output: ink2Mat{rows: 16}}, want: true},
		{name: "nvfp4", res: &inkling2Resident{be: be, dev: be, hp: hp, batched: true, kvNVFP4: true, hist: make([]ink2Ref, 1), output: ink2Mat{rows: 16}}, want: true},
		{name: "fp8k_nvfp4v", res: &inkling2Resident{be: be, dev: be, hp: hp, batched: true, kvAsym: true, hist: make([]ink2Ref, 1), output: ink2Mat{rows: 16}}, want: true},
		{name: "f16", res: &inkling2Resident{be: be, dev: be, hp: hp, batched: true, kvHalf: true, hist: make([]ink2Ref, 1), output: ink2Mat{rows: 16}}, want: false},
	} {
		err := tc.res.enableSpec(4)
		if tc.want && err != nil {
			t.Fatalf("enableSpec(%s) unexpected error: %v", tc.name, err)
		}
		if !tc.want && err == nil {
			t.Fatalf("enableSpec(%s) = nil, want refusal", tc.name)
		}
	}
}

// fak-test:runtime fast est=50ms
func TestInkling2LowBitGraphEligibility(t *testing.T) {
	be := fakeLowBitSpecBackend{name: "cuda"}
	defaultLookup := func(string) (string, bool) { return "", false }
	offLookup := func(k string) (string, bool) {
		if k == "FAK_INKLING2_GRAPH" {
			return "0", true
		}
		return "", false
	}
	for _, dtype := range []Inkling2KVDType{Inkling2KVF32, Inkling2KVF16, Inkling2KVFP8, Inkling2KVNVFP4, Inkling2KVAsym} {
		if !inkling2GraphEligible(be, false, dtype, defaultLookup) {
			t.Fatalf("inkling2GraphEligible(cuda, %s) = false, want true", dtype)
		}
		if inkling2GraphEligible(be, true, dtype, defaultLookup) {
			t.Fatalf("inkling2GraphEligible(hostRouting, %s) = true, want false", dtype)
		}
		if inkling2GraphEligible(be, false, dtype, offLookup) {
			t.Fatalf("inkling2GraphEligible(FAK_INKLING2_GRAPH=0, %s) = true, want false", dtype)
		}
	}
	metal := fakeLowBitSpecBackend{name: "metal"}
	if inkling2GraphEligible(metal, false, Inkling2KVNVFP4, defaultLookup) {
		t.Fatal("inkling2GraphEligible(metal) = true, want false")
	}

	s := &inkling2Resident{be: be, graph: true, batched: true, kvNVFP4: true}
	if !s.verifyGraphEligible() {
		t.Fatal("verifyGraphEligible(nvfp4) = false, want true")
	}
	sHalf := &inkling2Resident{be: be, graph: true, batched: true, kvHalf: true}
	if sHalf.verifyGraphEligible() {
		t.Fatal("verifyGraphEligible(f16) = true, want false")
	}
}

// fak-test:runtime fast est=50ms
func TestInkling2LocalSWAKVBudgetAndSavings(t *testing.T) {
	for _, tc := range []struct {
		in   string
		want Inkling2KVDType
	}{
		{in: "", want: Inkling2KVF32},
		{in: "f32", want: Inkling2KVF32},
		{in: "f16", want: Inkling2KVF16},
		{in: "fp8", want: Inkling2KVFP8},
	} {
		got, err := inkling2LocalSWAKVDTypeFrom(func(k string) (string, bool) {
			if k == "FAK_INKLING2_LOCAL_SWA_KV" && tc.in != "" {
				return tc.in, true
			}
			return "", false
		})
		if err != nil || got != tc.want {
			t.Fatalf("inkling2LocalSWAKVDTypeFrom(%q) = (%v, %v), want (%v, nil)", tc.in, got, err, tc.want)
		}
	}
	if _, err := inkling2LocalSWAKVDTypeFrom(func(string) (string, bool) { return "bad", true }); err == nil {
		t.Fatal("inkling2LocalSWAKVDTypeFrom(bad) = nil, want error")
	}

	// 40 local SWA layers (W=1024, chunk=128 -> cap=1152), KVHeads=4, HeadDim=256.
	local := make([]bool, 48)
	for i := 0; i < 40; i++ {
		local[i] = true
	}
	hp := model.Inkling2LoopHParams{
		Layers:        48,
		Passes:        1,
		Hidden:        2048,
		Heads:         16,
		KVHeads:       4,
		HeadDim:       256,
		SlidingWindow: 1024,
		ConvKernel:    4,
		LocalLayer:    local,
	}
	const mib = 1 << 20
	f32Bytes := inkling2ResidentKVBytesForLocal(hp, 131072, true, 128, Inkling2KVNVFP4, Inkling2KVF32)
	f16Bytes := inkling2ResidentKVBytesForLocal(hp, 131072, true, 128, Inkling2KVNVFP4, Inkling2KVF16)
	fp8Bytes := inkling2ResidentKVBytesForLocal(hp, 131072, true, 128, Inkling2KVNVFP4, Inkling2KVFP8)

	savedF16MiB := float64(f32Bytes-f16Bytes) / mib
	savedFP8MiB := float64(f32Bytes-fp8Bytes) / mib
	if savedF16MiB < 179 || savedF16MiB > 181 {
		t.Fatalf("local SWA f16 savings = %.2f MiB, want ~180 MiB", savedF16MiB)
	}
	if savedFP8MiB < 260 || savedFP8MiB > 360 {
		t.Fatalf("local SWA fp8 savings = %.2f MiB, want 260-360 MiB", savedFP8MiB)
	}

	// Verify auto-ctx and ring fit stay in sync through inkling2ResidentKVBudget.
	be := fakeLowBitMemBackend{freeBytes: 6 * (1 << 30)}
	other := inkling2ResidentOtherBytes(hp, inkling2ArenaFloats, 128, 0)
	autoCtx, err := inkling2ResolveAutoCtxForLocal(be, hp, true, 128, other, Inkling2KVNVFP4, Inkling2KVFP8)
	if err != nil {
		t.Fatalf("inkling2ResolveAutoCtxForLocal: %v", err)
	}
	if _, err := inkling2FitResidentKVForLocal(be, hp, autoCtx, true, 128, Inkling2KVNVFP4, Inkling2KVFP8, other); err != nil {
		t.Fatalf("inkling2FitResidentKVForLocal(%d): %v", autoCtx, err)
	}
}

// fak-test:runtime fast est=200ms
func TestInkling2CUDASpecLowBitKV(t *testing.T) {
	path := os.Getenv("FAK_INKLING2_GGUF")
	if path == "" || os.Getenv("FAK_INKLING2_SPEC_LOWBIT_KV") == "" {
		t.Skip("FAK_INKLING2_GGUF / FAK_INKLING2_SPEC_LOWBIT_KV not set")
	}
	be, ok := compute.Lookup("cuda")
	if !ok || be == nil || be.Name() != "cuda" {
		t.Fatal("FAK_INKLING2_GGUF set but the cuda backend is not registered")
	}
	m, err := OpenInkling2LoopDevice(path, be)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	t.Setenv("FAK_INKLING2_BATCHED_PREFILL", "1")

	prompt := []int{1, 2, 3, 4, 5, 6, 7, 8}
	n := 64
	for _, kvMode := range []string{"fp8", "nvfp4", "fp8k_nvfp4v"} {
		t.Run(kvMode, func(t *testing.T) {
			t.Setenv("FAK_INKLING2_KV", kvMode)
			ref, _, err := inkling2ResidentGreedy(m, be, prompt, n)
			if err != nil {
				t.Fatal(err)
			}
			got, sst, err := inkling2ResidentSpecGreedy(m, be, prompt, n, 4, func(*inkling2Resident) (inkling2Drafter, error) {
				return &inkling2OracleDrafter{ref: ref, corrupt: func(i int) bool { return i%7 == 3 }}, nil
			})
			if err != nil {
				t.Fatal(err)
			}
			if sst.Verifies == 0 {
				t.Fatalf("%s: expected speculative verifies > 0, got 0", kvMode)
			}
			if !slices.Equal(ref, got) {
				t.Fatalf("%s diverges from plain greedy: got %v, want %v", kvMode, got, ref)
			}
		})
	}
}
