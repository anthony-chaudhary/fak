package compute

import (
	"bytes"
	"errors"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/mathx"
)

// fak-test:runtime fast est=5ms lane=default

// TestSiluMulFp8QuantMatchesOracle pins the vLLM fused swiglu + per-token-group
// FP8 borrow from csrc/libtorch_stable/quantization/activation_kernels.cu:276
// (silu_mul_fp8_quant_deep_gemm_kernel) with the act_and_mul_kernel structure
// (csrc/libtorch_stable/activation_kernels.cu:102) @
// b6ff8a2f509cc7ac9c58176f5115a836aa1e08bd (Apache-2.0):
//
//	(a) amax-driven ue8m0 scale and e4m3fn RNE payload vs the independent
//	    f64 Go reference on a known row, with hand-pinned scale bytes and
//	    payload codes;
//	(b) the 128-wide group boundary with a ragged tail: H need not be a
//	    multiple of 128, the tail group counts only its own elements, and a
//	    large tail element drives its own group scale;
//	(c) non-finite refusal matching the host sentinel: NaN/infinite operands
//	    and finite operands whose fused product overflows f32 all refuse with
//	    ErrSiluMulFp8NonFinite, while finite extremes pass.
//
// CPU/in-memory results only: [SW-VERIFIED], no GPU or throughput claim.
func TestSiluMulFp8QuantMatchesOracle(t *testing.T) {
	t.Parallel()

	t.Run("known_row_scale_and_rounding", func(t *testing.T) {
		t.Parallel()
		const hidden = 256
		gate := make([]float32, hidden)
		up := make([]float32, hidden)
		// Group 0: gate = 0 -> silu(0) = 0 -> every product is exactly zero
		// regardless of up; the amax floors to EPS.
		for i := 0; i < 128; i++ {
			up[i] = float32(i)
		}
		// Group 1: gate = 1, up pattern in {-3..3}/4.
		for i := 128; i < hidden; i++ {
			gate[i] = 1
			up[i] = float32((i-128)%7-3) * 0.25
		}
		codes, scales, err := SiluMulFp8QuantRow(gate, up)
		if err != nil {
			t.Fatalf("SiluMulFp8QuantRow returned %v, want nil", err)
		}
		if len(codes) != hidden || len(scales) != 2 {
			t.Fatalf("got %d codes / %d scales, want %d / 2",
				len(codes), len(scales), hidden)
		}
		// Hand-pinned scale bytes:
		//   group 0: amax = 0 floors to EPS (bf16 raw 0x2E5C widened) ->
		//            ceil(log2(EPS/448)) = -43 -> 127-43 = 84.
		//   group 1: amax = silu(1)*0.75 -> ceil(log2(amax/448)) = -9 -> 118.
		if scales[0] != 84 {
			t.Errorf("scale[0] = %d, want 84 (EPS floor)", scales[0])
		}
		if scales[1] != 118 {
			t.Errorf("scale[1] = %d, want 118 (amax-driven)", scales[1])
		}
		// Independent f64 cross-check of both exponents.
		s1 := float32(float64(1.0) / (1 + math.Exp(-1.0)))
		if got := smfqRefExponent(0); got != int(scales[0])-127 {
			t.Errorf("group 0 exponent = %d, f64 reference wants %d",
				int(scales[0])-127, got)
		}
		if got := smfqRefExponent(float64(s1) * 0.75); got != int(scales[1])-127 {
			t.Errorf("group 1 exponent = %d, f64 reference wants %d",
				int(scales[1])-127, got)
		}
		// Hand-pinned payload codes (k := (i-128)%7-3, res = silu(1)*k/4,
		// scaled by 2^9 under the group scale):
		//   k=0  -> res 0           -> 0x00
		//   k=3  -> res 280.726...  -> 0x79 (nearest e4m3 is 288)
		//   k=-3 -> res -280.726... -> 0xF9
		//   k=1  -> res 93.575...   -> 0x6C (nearest e4m3 is 96)
		//   k=-1 -> res -93.575...  -> 0xEC
		//   k=2  -> res 187.151...  -> 0x74 (nearest e4m3 is 192)
		//   k=-2 -> res -187.151... -> 0xF4
		pins := map[int]byte{128: 0xF9, 129: 0xF4, 130: 0xEC, 131: 0x00,
			132: 0x6C, 133: 0x74, 134: 0x79}
		for i, want := range pins {
			if codes[i] != want {
				t.Errorf("codes[%d] = 0x%02X, want 0x%02X", i, codes[i], want)
			}
		}
		// Full cross-check of every code against the f64 value-set walk.
		exp1 := int(scales[1]) - 127
		for i := 128; i < hidden; i++ {
			res := float32(float64(s1) * float64(up[i]))
			want := smfqRefCode(float64(res) * math.Ldexp(1, -exp1))
			if codes[i] != want {
				t.Errorf("codes[%d] = 0x%02X, f64 reference wants 0x%02X",
					i, codes[i], want)
			}
		}
		for i := 0; i < 128; i++ {
			if codes[i] != 0x00 {
				t.Fatalf("codes[%d] = 0x%02X, want 0x00 on the zero group", i, codes[i])
			}
		}
	})

	t.Run("group_boundary_ragged_tail", func(t *testing.T) {
		t.Parallel()
		// hidden = 300: groups [0,128), [128,256), [256,300) — the last is a
		// 44-element ragged tail.
		const hidden = 300
		gate := make([]float32, hidden)
		up := make([]float32, hidden)
		for i := 0; i < 128; i++ {
			gate[i], up[i] = 1, 1 // g0: amax = silu(1) -> exp -9 -> 118
		}
		// g1 left at zeros: amax floors to EPS -> 84.
		for i := 256; i < hidden; i++ {
			gate[i], up[i] = 1, 0.001 // g2: amax ~7.31e-4 -> exp -19 -> 108
		}
		codes, scales, err := SiluMulFp8QuantRow(gate, up)
		if err != nil {
			t.Fatalf("SiluMulFp8QuantRow returned %v, want nil", err)
		}
		if len(codes) != 300 || len(scales) != 3 {
			t.Fatalf("got %d codes / %d scales, want 300 / 3",
				len(codes), len(scales))
		}
		wantScales := []byte{118, 84, 108}
		if !bytes.Equal(scales, wantScales) {
			t.Errorf("scales = %v, want %v", scales, wantScales)
		}
		// Every ragged-tail code matches the f64 reference; the tail counts
		// only its 44 present elements.
		s1 := float32(float64(1.0) / (1 + math.Exp(-1.0)))
		for i := 256; i < hidden; i++ {
			res := float32(float64(s1) * float64(up[i]))
			want := smfqRefCode(float64(res) * math.Ldexp(1, -(int(scales[2])-127)))
			if codes[i] != want {
				t.Errorf("codes[%d] = 0x%02X, f64 reference wants 0x%02X",
					i, codes[i], want)
			}
		}
		// A large element at the last valid index drives its own group scale:
		// amax = silu(1)*1000 -> exp 1 -> byte 128, siblings untouched.
		upBig := append([]float32(nil), up...)
		upBig[299] = 1000
		codesBig, scalesBig, err := SiluMulFp8QuantRow(gate, upBig)
		if err != nil {
			t.Fatalf("SiluMulFp8QuantRow(big tail) returned %v, want nil", err)
		}
		wantBig := []byte{118, 84, 128}
		if !bytes.Equal(scalesBig, wantBig) {
			t.Errorf("scales(big tail) = %v, want %v", scalesBig, wantBig)
		}
		resBig := float32(float64(s1) * 1000)
		wantBigCode := smfqRefCode(float64(resBig) * math.Ldexp(1, -1))
		if codesBig[299] != wantBigCode {
			t.Errorf("codes[299] = 0x%02X, f64 reference wants 0x%02X",
				codesBig[299], wantBigCode)
		}
		// hidden = 129: one full group plus a one-element tail.
		gate3 := make([]float32, 129)
		up3 := make([]float32, 129)
		gate3[128], up3[128] = 1, 5 // tail amax = silu(1)*5 -> exp -6 -> 121
		_, scales3, err := SiluMulFp8QuantRow(gate3, up3)
		if err != nil {
			t.Fatalf("SiluMulFp8QuantRow(129) returned %v, want nil", err)
		}
		if !bytes.Equal(scales3, []byte{84, 121}) {
			t.Errorf("scales(129) = %v, want [84 121]", scales3)
		}
	})

	t.Run("batch_addressing_act_and_mul", func(t *testing.T) {
		t.Parallel()
		// act_and_mul layout: token t reads gate = x[t*2H : t*2H+H] and
		// up = x[t*2H+H : t*3H]; the batch call must equal the per-row
		// oracle on the split halves.
		const tokens, hidden = 3, 300
		x := make([]float32, tokens*2*hidden)
		for tk := 0; tk < tokens; tk++ {
			for i := 0; i < hidden; i++ {
				x[tk*2*hidden+i] = float32(tk + 1)
				x[tk*2*hidden+hidden+i] = float32((i*31+tk*7)%13-6) * 0.125
			}
		}
		codes, scales, err := SiluMulFp8Quant(x, tokens, hidden)
		if err != nil {
			t.Fatalf("SiluMulFp8Quant returned %v, want nil", err)
		}
		if len(codes) != tokens*hidden || len(scales) != tokens*3 {
			t.Fatalf("got %d codes / %d scales, want %d / %d",
				len(codes), len(scales), tokens*hidden, tokens*3)
		}
		ng := SiluMulFp8NumGroups(hidden)
		for tk := 0; tk < tokens; tk++ {
			row := x[tk*2*hidden : (tk+1)*2*hidden]
			rc, rs, err := SiluMulFp8QuantRow(row[:hidden], row[hidden:])
			if err != nil {
				t.Fatalf("row %d: %v", tk, err)
			}
			if !bytes.Equal(rc, codes[tk*hidden:(tk+1)*hidden]) {
				t.Fatalf("token %d: batch codes diverge from the row oracle", tk)
			}
			if !bytes.Equal(rs, scales[tk*ng:(tk+1)*ng]) {
				t.Fatalf("token %d: batch scales diverge from the row oracle", tk)
			}
		}
	})

	t.Run("non_finite_refusal", func(t *testing.T) {
		t.Parallel()
		const hidden = 256
		mk := func() ([]float32, []float32) {
			return make([]float32, hidden), make([]float32, hidden)
		}
		nan := math.Float32frombits(0x7fc00000)
		posInf := math.Float32frombits(0x7f800000)
		negInf := math.Float32frombits(0xff800000)
		operandCases := []struct {
			name string
			mut  func(g, u []float32)
		}{
			{"gate_nan", func(g, u []float32) { g[0] = nan }},
			{"up_pos_inf", func(g, u []float32) { u[5] = posInf }},
			{"gate_neg_inf", func(g, u []float32) { g[100] = negInf }},
		}
		for _, c := range operandCases {
			g, u := mk()
			c.mut(g, u)
			if _, _, err := SiluMulFp8QuantRow(g, u); !errors.Is(err, ErrSiluMulFp8NonFinite) {
				t.Errorf("%s: errors.Is(%v, ErrSiluMulFp8NonFinite) = false", c.name, err)
			}
		}
		// Finite operands whose fused product overflows f32 refuse too:
		// silu(1e38) ~= 1e38, and 1e38 * 1e38 is +Inf.
		g, u := mk()
		for i := range g {
			g[i], u[i] = 1e38, 1e38
		}
		if _, _, err := SiluMulFp8QuantRow(g, u); !errors.Is(err, ErrSiluMulFp8NonFinite) {
			t.Errorf("overflow product: errors.Is(%v, ErrSiluMulFp8NonFinite) = false", err)
		}
		// The finite extreme passes: amax = MaxFloat32 -> exp 120 -> byte 247,
		// and every code lands on 0x78 ((1-2^-24)*256 rounds up to 256).
		g, u = mk()
		for i := range g {
			g[i], u[i] = math.MaxFloat32, 1
		}
		codes, scales, err := SiluMulFp8QuantRow(g, u)
		if err != nil {
			t.Fatalf("MaxFloat32 row refused: %v", err)
		}
		for i, s := range scales {
			if s != 247 {
				t.Errorf("scale[%d] = %d, want 247", i, s)
			}
		}
		for i, c := range codes {
			if c != 0x78 {
				t.Fatalf("code[%d] = 0x%02X, want 0x78", i, c)
			}
		}
		// Shape sentinels close the admission contract.
		if _, _, err := SiluMulFp8QuantRow(nil, nil); !errors.Is(err, ErrSiluMulFp8Shape) {
			t.Errorf("empty row: errors.Is(%v, ErrSiluMulFp8Shape) = false", err)
		}
		if _, _, err := SiluMulFp8QuantRow(make([]float32, 4),
			make([]float32, 5)); !errors.Is(err, ErrSiluMulFp8Shape) {
			t.Errorf("length mismatch: errors.Is(%v, ErrSiluMulFp8Shape) = false", err)
		}
		if _, _, err := SiluMulFp8Quant(make([]float32, 8), 0, 4); !errors.Is(err, ErrSiluMulFp8Empty) {
			t.Errorf("zero tokens: errors.Is(%v, ErrSiluMulFp8Empty) = false", err)
		}
		if _, _, err := SiluMulFp8Quant(make([]float32, 12), 2, 4); !errors.Is(err, ErrSiluMulFp8Shape) {
			t.Errorf("short batch: errors.Is(%v, ErrSiluMulFp8Shape) = false", err)
		}
		if _, _, err := SiluMulFp8Quant(make([]float32, 8), 1, 0); !errors.Is(err, ErrSiluMulFp8Shape) {
			t.Errorf("zero hidden: errors.Is(%v, ErrSiluMulFp8Shape) = false", err)
		}
	})

	t.Run("encoder_closure", func(t *testing.T) {
		t.Parallel()
		// Closed round values pinning RNE + saturation of the e4m3fn encoder.
		cases := []struct {
			in   float32
			want byte
		}{
			{448, 0x7E}, {-448, 0xFE}, {448.0001, 0x7E}, {-448.0001, 0xFE},
			{0, 0x00}, {math.Float32frombits(0x80000000), 0x80}, // -0.0 -> signed zero
			{1, 0x38}, {0.5, 0x30}, {224, 0x76}, {447.9, 0x7E},
			{float32(math.Ldexp(1, -9)), 0x01},       // smallest subnormal
			{float32(math.Ldexp(1, -10)), 0x00},      // exact tie rounds to even (0)
			{float32(1.5 * math.Ldexp(1, -9)), 0x02}, // tie rounds to even (2)
			{float32(0.5 * math.Ldexp(1, -9)), 0x00}, // tie rounds to even (0)
			{float32(7.5 * math.Ldexp(1, -9)), 0x08}, // top-of-band tie rolls to 2^-6
		}
		for _, c := range cases {
			if got := siluMulFp8EncodeE4M3(c.in); got != c.want {
				t.Errorf("encode(%v) = 0x%02X, want 0x%02X", c.in, got, c.want)
			}
		}
		// Round-trip closure: every finite e4m3fn byte decodes to a value
		// that re-encodes to the same byte.
		for b := 0; b < 256; b++ {
			if b == 0x7F || b == 0xFF {
				continue // NaN codes
			}
			v := mathx.DecodeE4M3(byte(b))
			if got := siluMulFp8EncodeE4M3(v); got != byte(b) {
				t.Errorf("round trip: decode(0x%02X) = %v re-encodes to 0x%02X",
					b, v, got)
			}
		}
	})

	t.Run("source_contract", func(t *testing.T) {
		t.Parallel()
		cu, err := os.ReadFile(filepath.FromSlash("silu_mul_fp8_quant.cu"))
		if err != nil {
			t.Fatal(err)
		}
		cuText := string(bytes.ReplaceAll(cu, []byte("\r\n"), []byte("\n")))
		cuCode := stripCComments(cuText)
		for token, want := range map[string]int{
			"__global__ void k_silu_mul_fp8_quant":     1,
			`extern "C" int fcuda_silu_mul_fp8_quant`:  1,
			"atomicMin(status + 2, op_fault)":          1,
			"atomicMin(status + 3, mid_fault)":         1,
			"__shfl_xor_sync(0xFFFFFFFFu, amax, mask)": 1,
		} {
			if got := strings.Count(cuCode, token); got != want {
				t.Errorf("silu_mul_fp8_quant.cu has %d occurrences of %q, want %d",
					got, token, want)
			}
		}
		if !strings.Contains(cuText, "b6ff8a2f509cc7ac9c58176f5115a836aa1e08bd") {
			t.Error("silu_mul_fp8_quant.cu lacks the upstream vLLM pin citation")
		}
		if !strings.Contains(cuText, "silu_mul_fp8_quant_deep_gemm_kernel") {
			t.Error("silu_mul_fp8_quant.cu lacks the adapted-kernel citation")
		}
		if !strings.Contains(cuText, "act_and_mul_kernel") {
			t.Error("silu_mul_fp8_quant.cu lacks the structure citation")
		}
	})
}

// smfqRefExponent is the independent f64 reference for the group scale
// exponent: ceil(log2(floor(amax)/448)) without any bit manipulation. It
// agrees with the f32-bit library path for every finite amax because the
// f32 rounding of amax/448 can never cross a power-of-two boundary in the
// wrong direction.
func smfqRefExponent(amax float64) int {
	f := float64(SiluMulFp8AmaxFloor)
	if amax < f {
		amax = f
	}
	return int(math.Ceil(math.Log2(amax / 448.0)))
}

// smfqRefCode is the independent f64 reference for one e4m3fn code: it walks
// the closed set of 128 representable magnitudes (8 subnormal + 120 normal,
// the 0x7F/0xFF NaN codes excluded) and picks the nearest to |x|, ties to
// even mantissa, saturating at +/-448. Structurally different from the
// bit-manipulation encoder it cross-checks.
func smfqRefCode(x float64) byte {
	sign := byte(0)
	if x < 0 {
		sign = 0x80
		x = -x
	}
	if x >= 448 {
		return sign | 0x7E
	}
	type cand struct {
		val   float64
		code  byte
		evenM bool
	}
	var cands []cand
	for m := 0; m < 8; m++ {
		cands = append(cands, cand{math.Ldexp(float64(m)/8, 1-7), byte(m), m%2 == 0})
	}
	for e := 1; e <= 15; e++ {
		for m := 0; m < 8; m++ {
			if e == 15 && m == 7 {
				continue // 0x7F NaN code
			}
			cands = append(cands, cand{math.Ldexp(1+float64(m)/8, e-7),
				byte(e<<3 | m), m%2 == 0})
		}
	}
	best, bestD := 0, math.Inf(1)
	for i, c := range cands {
		d := math.Abs(c.val - x)
		switch {
		case d < bestD:
			best, bestD = i, d
		case d == bestD && c.evenM && !cands[best].evenM:
			best = i
		}
	}
	return sign | cands[best].code
}
