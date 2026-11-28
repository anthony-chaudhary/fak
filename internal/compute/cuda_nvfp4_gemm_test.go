package compute

import (
	"errors"
	"math"
	"math/rand"
	"testing"
)

// fak-test:runtime fast est=200ms
func TestCUDAInk2NVFP4W4A4GEMM(t *testing.T) {
	t.Run("env_gating", func(t *testing.T) {
		t.Setenv(CUDAInk2W4A4GEMMEnv, "")
		if Ink2W4A4GEMMEnabled() {
			t.Fatal("expected W4A4 GEMM disabled when env unset")
		}
		if Ink2W4A4GEMMEligible(Ink2W4A4Prefill, 128, 2304, 2304) {
			t.Fatal("expected prefill ineligible when env unset")
		}

		t.Setenv(CUDAInk2W4A4GEMMEnv, "0")
		if Ink2W4A4GEMMEligible(Ink2W4A4Verify, 16, 2304, 2304) {
			t.Fatal("expected verify ineligible when env=0")
		}

		t.Setenv(CUDAInk2W4A4GEMMEnv, "1")
		if !Ink2W4A4GEMMEnabled() {
			t.Fatal("expected W4A4 GEMM enabled when env=1")
		}
		if Ink2W4A4GEMMEligible(Ink2W4A4Prefill, 1, 2304, 2304) || Ink2W4A4GEMMEligible(Ink2W4A4Prefill, 15, 2304, 2304) {
			t.Fatal("single-token decode and T<16 prefill must stay off W4A4 GEMM")
		}
		if !Ink2W4A4GEMMEligible(Ink2W4A4Prefill, 16, 2304, 2304) {
			t.Fatal("expected T=16 prefill eligible")
		}
		if Ink2W4A4GEMMEligible(Ink2W4A4Verify, 3, 2304, 2304) {
			t.Fatal("expected K<4 verify ineligible")
		}
		if !Ink2W4A4GEMMEligible(Ink2W4A4Verify, 4, 2304, 2304) {
			t.Fatal("expected K=4 verify eligible")
		}
		if Ink2W4A4GEMMEligible(Ink2W4A4Prefill, 64, 2304, 2312) {
			t.Fatal("expected K not divisible by 16 ineligible")
		}
	})

	t.Run("inkling2_shapes_and_parity", func(t *testing.T) {
		t.Setenv(CUDAInk2W4A4GEMMEnv, "1")
		rng := rand.New(rand.NewSource(120))
		// Cover Inkling-2 Hidden=2304 and MoE FFN dims across M in {4, 16, 128, 512},
		// plus tail rows/cols and K % 64 != 0.
		cases := []struct {
			name        string
			m, n, k     int
			wantTailM   int
			wantTailN   int
			wantTailK   int
			maxRelL2Err float64
		}{
			{"verify_k4_hidden2304", 4, 64, 2304, 4, 0, 0, 0.18},
			{"prefill_m16_hidden2304", 16, 128, 2304, 0, 0, 0, 0.18},
			{"prefill_m128_moe_ffn", 128, 64, 1152, 0, 0, 0, 0.18},
			{"prefill_m512_moe_ffn", 512, 32, 512, 0, 32, 0, 0.18},
			{"tail_m17_n37_k80", 17, 37, 80, 1, 37, 16, 0.20},
			{"tail_m5_n13_k48", 5, 13, 48, 5, 13, 48, 0.22},
		}

		for _, tc := range cases {
			t.Run(tc.name, func(t *testing.T) {
				plan, err := PlanInk2W4A4GEMM(tc.m, tc.n, tc.k)
				if err != nil {
					t.Fatalf("PlanInk2W4A4GEMM: %v", err)
				}
				if plan.TailM != tc.wantTailM || plan.TailN != tc.wantTailN || plan.TailK != tc.wantTailK {
					t.Fatalf("plan tails = (%d,%d,%d), want (%d,%d,%d)",
						plan.TailM, plan.TailN, plan.TailK, tc.wantTailM, tc.wantTailN, tc.wantTailK)
				}
				if plan.KPad%ink2W4A4TileK != 0 || plan.KPad < tc.k {
					t.Fatalf("invalid KPad=%d for K=%d", plan.KPad, tc.k)
				}

				wPacked := make([]byte, tc.n*tc.k/2)
				rng.Read(wPacked)
				wScales := make([]byte, tc.n*tc.k/NVFP4BlockSize)
				for i := range wScales {
					wScales[i] = byte(0x28 + rng.Intn(0x20))
				}
				const wGlobal = float32(0.25)

				x := make([]float32, tc.m*tc.k)
				for i := range x {
					x[i] = float32(rng.NormFloat64())
				}

				got, err := GemmInk2W4A4Reference(tc.n, tc.k, tc.m, wPacked, wScales, wGlobal, x)
				if err != nil {
					t.Fatalf("GemmInk2W4A4Reference: %v", err)
				}
				ref, err := GemmInk2W4A4FP32Reference(tc.n, tc.k, tc.m, wPacked, wScales, wGlobal, x)
				if err != nil {
					t.Fatalf("GemmInk2W4A4FP32Reference: %v", err)
				}
				if len(got) != tc.m*tc.n || len(ref) != tc.m*tc.n {
					t.Fatalf("output len mismatch: got=%d ref=%d want=%d", len(got), len(ref), tc.m*tc.n)
				}

				var num, den float64
				for i := range got {
					if math.IsNaN(float64(got[i])) || math.IsInf(float64(got[i]), 0) {
						t.Fatalf("non-finite output at %d: %v", i, got[i])
					}
					d := float64(got[i]) - float64(ref[i])
					num += d * d
					den += float64(ref[i]) * float64(ref[i])
				}
				relL2 := math.Sqrt(num / (den + 1e-12))
				if relL2 > tc.maxRelL2Err {
					t.Fatalf("relL2 = %.4f exceeds bound %.4f", relL2, tc.maxRelL2Err)
				}
			})
		}
	})

	t.Run("scale_overflow_and_errors", func(t *testing.T) {
		const m, n, k = 4, 8, 64
		wPacked := make([]byte, n*k/2)
		wScales := make([]byte, n*k/NVFP4BlockSize)
		for i := range wPacked {
			wPacked[i] = 0x11
		}
		for i := range wScales {
			wScales[i] = 0x38 // 1.0 in UE4M3FN
		}

		// Extreme activation magnitudes near float32 ceiling + zero blocks + subnormals.
		x := make([]float32, m*k)
		for i := 0; i < 16; i++ {
			x[i] = math.MaxFloat32 / 8
			x[k+i] = -math.MaxFloat32 / 8
			x[2*k+i] = 1e-38
		}
		// Row 3 stays all zeros.
		act, err := QuantizeInk2W4A4Activations(m, k, x)
		if err != nil {
			t.Fatalf("QuantizeInk2W4A4Activations extreme: %v", err)
		}
		for i, code := range act.Scales {
			if code > ink2W4A4MaxE4M3Code {
				t.Fatalf("scale[%d] = 0x%02x exceeds max finite UE4M3FN 0x%02x", i, code, ink2W4A4MaxE4M3Code)
			}
		}
		for i, rs := range act.RowScales {
			if !nvfp4Finite(rs) {
				t.Fatalf("rowScale[%d] non-finite: %v", i, rs)
			}
		}
		if act.RowScales[3] != 0 {
			t.Fatalf("zero row scale = %v, want 0", act.RowScales[3])
		}

		// With small wGlobal, GEMM must stay finite even at extreme activation scale.
		out, err := GemmInk2W4A4Reference(n, k, m, wPacked, wScales, 1e-6, x)
		if err != nil {
			t.Fatalf("GemmInk2W4A4Reference extreme: %v", err)
		}
		for i, v := range out {
			if !nvfp4Finite(v) {
				t.Fatalf("output[%d] non-finite: %v", i, v)
			}
		}

		// Non-finite input must be rejected.
		x[0] = float32(math.NaN())
		if _, err := QuantizeInk2W4A4Activations(m, k, x); !errors.Is(err, ErrNVFP4Input) {
			t.Fatalf("NaN input err = %v, want ErrNVFP4Input", err)
		}
		x[0] = 1.0
		if _, err := GemmInk2W4A4Reference(n, k, m, wPacked, wScales, -1.0, x); !errors.Is(err, ErrNVFP4Scale) {
			t.Fatalf("negative wGlobal err = %v, want ErrNVFP4Scale", err)
		}
	})
}
