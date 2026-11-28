//go:build cuda && cgo

package compute

import (
	"math"
	"math/rand"
	"os"
	"testing"
)

// fak-test:runtime integration est=5s lane=optin
func TestCUDAInk2PNVFP4SlabMajorBitwise(t *testing.T) {
	cb, ok := Pick("cuda").(*cudaBackend)
	if !ok {
		t.Fatal("CUDA backend is required")
	}
	t.Setenv("FAK_INKLING2_PREFILL_OLD", "")
	t.Setenv("FAK_INKLING2_PREFILL_NVFP4_W4A4", "0")
	const nexp, tokens, k, pairs = 16, 56, 8, 56 * 8
	rng := rand.New(rand.NewSource(73))
	alloc := func(t *testing.T, n int) Ink2Ref {
		t.Helper()
		buf, err := cb.Ink2Alloc(n)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { cb.Free(buf) })
		return Ink2At(buf, 0)
	}
	write := func(t *testing.T, dst Ink2Ref, src []float32) {
		t.Helper()
		if err := cb.Ink2Write(dst, src); err != nil {
			t.Fatal(err)
		}
	}
	id := make([]float32, pairs)
	for tok := 0; tok < tokens; tok++ {
		for slot := 1; slot < k; slot++ {
			id[tok*k+slot] = math.Float32frombits(uint32(1 + (tok*5+slot)%12))
		}
	}
	esc := make([]float32, nexp)
	for i := range esc {
		esc[i] = 0.5 + rng.Float32()
	}
	dids, ds := alloc(t, len(id)), alloc(t, len(esc))
	scratch := alloc(t, Ink2PGroupScratch(pairs, nexp))
	write(t, dids, id)
	write(t, ds, esc)
	checkProfile := os.Getenv("FAK_CUDA_TIMESTAMP_PROFILE") == "1"
	bucketReady := false
	for _, tc := range []struct {
		name           string
		rows, in, xDiv int
	}{
		{"inkling2_m768_k2304", 768, 2304, k},
		{"inkling2_m2304_k768", 2304, 768, 1},
		{"inkling2_m2304_k2304", 2304, 2304, k},
		{"inkling2_m2304_k1152", 2304, 1152, 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			packed := make([]byte, nexp*tc.rows*tc.in/2)
			_, _ = rng.Read(packed)
			scales := make([]byte, nexp*tc.rows*tc.in/16)
			for i := range scales {
				scales[i] = byte(0x20 + rng.Intn(0x30))
			}
			wRow := cudaNVFP4Resident(t, cb, []int{nexp * tc.rows, tc.in}, packed, scales, 1)
			defer cb.Free(wRow)
			wSlab := cudaNVFP4Resident(t, cb, []int{nexp * tc.rows, tc.in}, packed, scales, 1)
			defer cb.Free(wSlab)

			x := make([]float32, pairs/tc.xDiv*tc.in)
			for i := range x {
				x[i] = rng.Float32()*4 - 2
			}
			dx, y := alloc(t, len(x)), alloc(t, pairs*tc.rows)
			write(t, dx, x)
			poison := make([]float32, pairs*tc.rows)
			for i := range poison {
				poison[i] = math.Float32frombits(0x7fc00000)
			}

			runGrouped := func(w Tensor, bm128Flag, slabFlag, wantKernel string) []float32 {
				t.Helper()
				t.Setenv("FAK_INKLING2_NVFP4_BM128", bm128Flag)
				t.Setenv(CUDAInk2PNVFP4SlabEnv, slabFlag)
				write(t, y, poison)
				before := KernelProfileSnapshot()
				ok, err := cb.Ink2PExpertGrouped(w, tc.rows, nexp, dids, dx, tc.xDiv, tc.in, y, pairs, ds, scratch, bucketReady)
				if !ok || err != nil {
					t.Fatalf("slab=%s bm128=%s accepted=%t err=%v", slabFlag, bm128Flag, ok, err)
				}
				bucketReady = true
				got, err := cb.Ink2Read(y, len(poison))
				if err != nil || len(got) != len(poison) {
					t.Fatalf("slab=%s output length %d err=%v", slabFlag, len(got), err)
				}
				if checkProfile {
					delta := KernelProfileDelta(before, KernelProfileSnapshot())
					if delta == nil {
						t.Fatal("CUDA launch profile unavailable")
					}
					found := false
					for _, row := range delta.Rows {
						if row.Kernel == wantKernel && row.GX == uint32(tc.rows/128) && row.Calls == 1 {
							found = true
							break
						}
					}
					if !found {
						t.Fatalf("missing %s grid.x=%d: %v", wantKernel, tc.rows/128, delta.Rows)
					}
				}
				return got
			}

			want := runGrouped(wRow, "1", "0", "k_ink2p_expert_nvfp4_v2_bm128")
			got := runGrouped(wSlab, "0", "1", "k_ink2p_expert_nvfp4_v2_slab128")
			for i, v := range want {
				a, b := math.Float32bits(v), math.Float32bits(got[i])
				if a != b || math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
					t.Fatalf("grouped pass1 pair=%d row=%d rowMajor=%08x slabMajor=%08x", i/tc.rows, i%tc.rows, a, b)
				}
			}

			// Second grouped call on wSlab must be idempotent (no double-repack).
			got2 := runGrouped(wSlab, "0", "1", "k_ink2p_expert_nvfp4_v2_slab128")
			for i, v := range want {
				a, b := math.Float32bits(v), math.Float32bits(got2[i])
				if a != b {
					t.Fatalf("grouped pass2 pair=%d row=%d rowMajor=%08x slabMajor=%08x", i/tc.rows, i%tc.rows, a, b)
				}
			}

			// Ordinary decode GEMV (Ink2ExpertMatVec) on wSlab must also be bitwise-exact to wRow.
			dxDec := alloc(t, k*tc.in)
			write(t, dxDec, x[:k*tc.in])
			yDec := alloc(t, k*tc.rows)
			poisonDec := poison[:k*tc.rows]
			runDecode := func(w Tensor) []float32 {
				t.Helper()
				write(t, yDec, poisonDec)
				ok, err := cb.Ink2ExpertMatVec(w, tc.rows, dids, dxDec, tc.in, yDec, k, ds, nil)
				if !ok || err != nil {
					t.Fatalf("decode Ink2ExpertMatVec accepted=%t err=%v", ok, err)
				}
				out, err := cb.Ink2Read(yDec, len(poisonDec))
				if err != nil || len(out) != len(poisonDec) {
					t.Fatalf("decode readback len=%d err=%v", len(out), err)
				}
				return out
			}
			wantDec := runDecode(wRow)
			gotDec := runDecode(wSlab)
			for i, v := range wantDec {
				a, b := math.Float32bits(v), math.Float32bits(gotDec[i])
				if a != b || math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
					t.Fatalf("decode slot=%d row=%d rowMajor=%08x slabMajor=%08x", i/tc.rows, i%tc.rows, a, b)
				}
			}
		})
	}
}
