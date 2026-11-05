package compute

import (
	"bytes"
	"errors"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

// fak-test:runtime fast est=5ms lane=default

// TestRmsNormFp8QuantMatchesOracle pins the vLLM fused RMSNorm + static-scale
// FP8 E4M3 activation quant borrow from csrc/layernorm_quant_kernels.cu @
// ae065918640a198a56f6f17eb45e251c3229fe41 (Apache-2.0):
//
//	(a) norm+scale+round matches the Go reference on known rows, with
//	    hand-pinned e4m3 bytes on exact-representable values;
//	(b) the residual-add variant equals add-then-norm byte for byte;
//	(c) a zero-variance row refuses with the host sentinel
//	    (errors.Is on ErrRmsNormFp8ZeroVariance, never error prose).
//
// CPU/in-memory results only: [SW-VERIFIED], no GPU or throughput claim.
func TestRmsNormFp8QuantMatchesOracle(t *testing.T) {
	t.Parallel()

	t.Run("known_rows_hand_pinned", func(t *testing.T) {
		t.Parallel()
		// width 4, two rows sharing one static per-tensor scale (0.5 ->
		// scale_inv 2), eps 0. Row 0: x=[1,1,1,1] -> sumsq 4, mean 1 ->
		// inv 1; n = x*inv*w = [1, 0.5, -1, 2]; q = n*scale_inv =
		// [2, 1, -2, 4]: e4m3 codes 0x40, 0x38, 0xC0, 0x48 (all exact).
		// Row 1: x=[1,1,1,-1] -> same sumsq/mean/inv; n = [1, 0.5, -1, -2];
		// q = [2, 1, -2, -4]: 0x40, 0x38, 0xC0, 0xC8.
		args := RmsNormFp8QuantArgs{
			X:       []float32{1, 1, 1, 1, 1, 1, 1, -1},
			Weight:  []float32{1, 0.5, -1, 2},
			Scale:   0.5,
			Epsilon: 0,
		}
		got, err := RmsNormFp8Quant(args)
		if err != nil {
			t.Fatalf("RmsNormFp8Quant returned %v, want nil", err)
		}
		want := []byte{0x40, 0x38, 0xC0, 0x48, 0x40, 0x38, 0xC0, 0xC8}
		if !bytes.Equal(got.Codes, want) {
			t.Fatalf("codes = % X, want % X", got.Codes, want)
		}
		if got.ResidualOut != nil {
			t.Fatalf("plain variant must not report a residual, got %v", got.ResidualOut)
		}
	})

	t.Run("general_row_matches_reference", func(t *testing.T) {
		t.Parallel()
		// 3 rows x width 64, deterministic mixed-sign/magnitude fill spanning
		// 1e-2..1e2, nonzero weights, eps > 0: every code must equal the
		// independent in-test reference, and dequantizing each code must land
		// within half the e4m3 bucket spacing of the pre-encode value.
		const rows, width = 3, 64
		x := make([]float32, rows*width)
		w := make([]float32, width)
		for r := 0; r < rows; r++ {
			for i := 0; i < width; i++ {
				mag := math.Pow(10, float64(i%5)-2)
				x[r*width+i] = float32(math.Sin(float64(r*31+i)*0.9) * mag)
			}
		}
		for i := range w {
			w[i] = float32(0.5 + float64(i%7)/4.0)
		}
		const scale, eps = 0.25, 1e-5
		got, err := RmsNormFp8Quant(RmsNormFp8QuantArgs{X: x, Weight: w, Scale: scale, Epsilon: eps})
		if err != nil {
			t.Fatalf("RmsNormFp8Quant returned %v, want nil", err)
		}
		refCodes, refQ := rfqReferenceRow(x, w, scale, eps)
		for i := range got.Codes {
			if got.Codes[i] != refCodes[i] {
				t.Fatalf("dim %d: code 0x%02X, reference 0x%02X", i, got.Codes[i], refCodes[i])
			}
		}
		for i, q := range refQ {
			a := math.Abs(float64(q))
			if a >= 448 {
				wantSat := byte(0x7E)
				if q < 0 {
					wantSat = 0xFE
				}
				if got.Codes[i] != wantSat {
					t.Fatalf("dim %d: q=%v must saturate to 0x%02X, got 0x%02X",
						i, q, wantSat, got.Codes[i])
				}
				continue
			}
			dec := float64(DecodeE4M3(got.Codes[i]))
			if diff := math.Abs(dec - float64(q)); diff > float64(rfqHalfSpacing(q)) {
				t.Fatalf("dim %d: |decode(code)-q| = %g exceeds half the e4m3 bucket spacing %g",
					i, diff, rfqHalfSpacing(q))
			}
		}
	})

	t.Run("encoder_rne_closure", func(t *testing.T) {
		t.Parallel()
		// Closed round values pinning RNE + saturation on the OCP e4m3fn
		// grid, including the subnormal step 2^-9 and the promotion of a
		// rounded-up top subnormal to the smallest normal 2^-6 (code 0x08).
		// The 2^-13 case documents the deliberate divergence from
		// dsv41kEncodeE4M3's m*2^-13 subnormal grid: on the OCP grid 2^-13
		// is nearer to zero than to 2^-9, so it encodes to 0x00.
		cases := []struct {
			in   float32
			want byte
		}{
			{448, 0x7E}, {-448, 0xFE}, {448.0001, 0x7E}, {-448.0001, 0xFE},
			{447.9, 0x7E}, {224, 0x76},
			{0, 0x00}, {math.Float32frombits(0x80000000), 0x80}, // -0.0 -> signed zero
			{1, 0x38}, {2, 0x40}, {0.5, 0x30},
			{float32(math.Ldexp(1, -9)), 0x01},   // smallest subnormal (OCP)
			{float32(math.Ldexp(1, -10)), 0x00},  // exact tie rounds to even (0)
			{float32(math.Ldexp(3, -10)), 0x02},  // exact tie rounds to even (2)
			{float32(math.Ldexp(7, -9)), 0x07},   // top subnormal, exact
			{float32(math.Ldexp(15, -10)), 0x08}, // tie at 7.5*2^-9 promotes (even)
			{float32(math.Ldexp(31, -11)), 0x08}, // 7.75*2^-9 rounds up, promotes
			{float32(math.Ldexp(1, -6)), 0x08},   // smallest normal, exact
			{float32(math.Ldexp(1, -13)), 0x00},  // OCP grid: nearer to zero
		}
		for _, c := range cases {
			if got := rfqEncodeE4M3(c.in); got != c.want {
				t.Errorf("encode(%v) = 0x%02X, want 0x%02X", c.in, got, c.want)
			}
		}
	})

	t.Run("fused_add_equals_add_then_norm", func(t *testing.T) {
		t.Parallel()
		const rows, width = 3, 16
		x := make([]float32, rows*width)
		res := make([]float32, rows*width)
		w := make([]float32, width)
		for i := range x {
			x[i] = float32(math.Cos(float64(i)*0.37)) * float32(1+(i%4))
			res[i] = float32(math.Sin(float64(i)*1.13)) * float32(1-float64(i%3)*0.25)
		}
		for i := range w {
			w[i] = float32(0.25 + float64(i%5)*0.5)
		}
		const scale, eps = 0.75, 2e-5
		fused, err := RmsNormFp8Quant(RmsNormFp8QuantArgs{
			X: x, Weight: w, Residual: res, Scale: scale, Epsilon: eps,
		})
		if err != nil {
			t.Fatalf("fused RmsNormFp8Quant returned %v, want nil", err)
		}
		// Add-then-norm: materialize the sum, run the plain variant on it.
		sum := make([]float32, rows*width)
		for i := range sum {
			sum[i] = x[i] + res[i]
		}
		plain, err := RmsNormFp8Quant(RmsNormFp8QuantArgs{
			X: sum, Weight: w, Scale: scale, Epsilon: eps,
		})
		if err != nil {
			t.Fatalf("plain RmsNormFp8Quant returned %v, want nil", err)
		}
		if !bytes.Equal(fused.Codes, plain.Codes) {
			t.Fatalf("fused codes diverge from add-then-norm codes")
		}
		if !reflect.DeepEqual(fused.ResidualOut, sum) {
			t.Fatalf("fused residual-out diverges from the materialized add")
		}
	})

	t.Run("zero_variance_row_refusal", func(t *testing.T) {
		t.Parallel()
		// Row 0 is live; row 1 is all-zero -> its f32 squared sum is zero and
		// the call must refuse with the host sentinel before any codes exist.
		args := RmsNormFp8QuantArgs{
			X:       []float32{1, 2, 3, 4, 0, 0, 0, 0},
			Weight:  []float32{1, 1, 1, 1},
			Scale:   1,
			Epsilon: 1e-6,
		}
		got, err := RmsNormFp8Quant(args)
		if !errors.Is(err, ErrRmsNormFp8ZeroVariance) {
			t.Fatalf("errors.Is(%v, ErrRmsNormFp8ZeroVariance) = false, want true", err)
		}
		if got != nil {
			t.Fatalf("refused call must return no result, got %+v", got)
		}
		// Signed zeros still sum to zero variance.
		zneg := math.Float32frombits(0x80000000) // -0.0
		if _, err := RmsNormFp8Quant(RmsNormFp8QuantArgs{
			X: []float32{0, zneg, 0, zneg}, Weight: []float32{1, 1, 1, 1}, Scale: 1,
		}); !errors.Is(err, ErrRmsNormFp8ZeroVariance) {
			t.Fatalf("signed-zero row: errors.Is(%v, ErrRmsNormFp8ZeroVariance) = false", err)
		}
		// The fused variant refuses the same way when x + residual cancels.
		if _, err := RmsNormFp8Quant(RmsNormFp8QuantArgs{
			X:        []float32{1, -1, 2, -2},
			Residual: []float32{-1, 1, -2, 2},
			Weight:   []float32{1, 1, 1, 1},
			Scale:    1,
		}); !errors.Is(err, ErrRmsNormFp8ZeroVariance) {
			t.Fatalf("fused cancelling row: errors.Is(%v, ErrRmsNormFp8ZeroVariance) = false", err)
		}
	})

	t.Run("sentinel_errors", func(t *testing.T) {
		t.Parallel()
		checks := []struct {
			name     string
			err      error
			sentinel error
		}{
			{"empty_weight", rfqErr(RmsNormFp8QuantArgs{X: []float32{1}, Scale: 1}), ErrRmsNormFp8Geometry},
			{"short_x", rfqErr(RmsNormFp8QuantArgs{X: []float32{1, 2}, Weight: []float32{1, 1, 1}, Scale: 1}), ErrRmsNormFp8Geometry},
			{"residual_mismatch", rfqErr(RmsNormFp8QuantArgs{X: []float32{1, 1}, Residual: []float32{1}, Weight: []float32{1, 1}, Scale: 1}), ErrRmsNormFp8Geometry},
			{"zero_scale", rfqErr(RmsNormFp8QuantArgs{X: []float32{1, 1}, Weight: []float32{1, 1}, Scale: 0}), ErrRmsNormFp8Params},
			{"nan_scale", rfqErr(RmsNormFp8QuantArgs{X: []float32{1, 1}, Weight: []float32{1, 1}, Scale: float32(math.NaN())}), ErrRmsNormFp8Params},
			{"inf_scale", rfqErr(RmsNormFp8QuantArgs{X: []float32{1, 1}, Weight: []float32{1, 1}, Scale: float32(math.Inf(1))}), ErrRmsNormFp8Params},
			{"negative_eps", rfqErr(RmsNormFp8QuantArgs{X: []float32{1, 1}, Weight: []float32{1, 1}, Scale: 1, Epsilon: -1}), ErrRmsNormFp8Params},
			{"nan_x", rfqErr(RmsNormFp8QuantArgs{X: []float32{float32(math.NaN()), 1}, Weight: []float32{1, 1}, Scale: 1}), ErrRmsNormFp8NonFinite},
			{"inf_residual", rfqErr(RmsNormFp8QuantArgs{X: []float32{1, 1}, Residual: []float32{1, float32(math.Inf(1))}, Weight: []float32{1, 1}, Scale: 1}), ErrRmsNormFp8NonFinite},
			{"inf_weight", rfqErr(RmsNormFp8QuantArgs{X: []float32{1, 1}, Weight: []float32{1, float32(math.Inf(-1))}, Scale: 1}), ErrRmsNormFp8NonFinite},
		}
		for _, c := range checks {
			if !errors.Is(c.err, c.sentinel) {
				t.Errorf("%s: errors.Is(%v, %v) = false, want true", c.name, c.err, c.sentinel)
			}
		}
	})

	t.Run("source_contract", func(t *testing.T) {
		t.Parallel()
		cu, err := os.ReadFile(filepath.FromSlash("rmsnorm_fp8_quant.cu"))
		if err != nil {
			t.Fatal(err)
		}
		cuText := string(bytes.ReplaceAll(cu, []byte("\r\n"), []byte("\n")))
		cuCode := stripCComments(cuText)
		for token, want := range map[string]int{
			"__global__ void k_rfq_":                    2,
			`extern "C" int fcuda_rmsnorm_fp8_quant`:    2,
			`extern "C" size_t fcuda_rmsnorm_fp8_quant`: 2,
			"rfq_e4m3_encode":                           3,
			"g_rfq_launches":                            3,
		} {
			if got := strings.Count(cuCode, token); got != want {
				t.Errorf("rmsnorm_fp8_quant.cu has %d occurrences of %q, want %d",
					got, token, want)
			}
		}
		for sym := range map[string]bool{
			"__fadd_rn":  true,
			"__fmul_rn":  true,
			"__fdiv_rn":  true,
			"rfq_geom":   true,
			"rfq_params": true,
		} {
			if !strings.Contains(cuCode, sym) {
				t.Errorf("rmsnorm_fp8_quant.cu missing bit-exactness helper %q", sym)
			}
		}
		if !strings.Contains(cuText, "ae065918640a198a56f6f17eb45e251c3229fe41") {
			t.Error("rmsnorm_fp8_quant.cu lacks the upstream vLLM pin citation")
		}
	})
}

// rfqErr runs RmsNormFp8Quant and returns only the error, for the closed
// sentinel table.
func rfqErr(args RmsNormFp8QuantArgs) error {
	_, err := RmsNormFp8Quant(args)
	return err
}

// rfqReferenceRow is the in-test reference for one plain RmsNormFp8Quant call:
// the same pinned formula (sequential f32 squared sum, exact
// reciprocal-of-sqrt, (x*inv)*w*scale_inv, RNE e4m3) spelled out step by step
// so a transcription bug in RmsNormFp8Quant shows up as a divergence. It also
// returns the pre-encode q values for the closed quantization-bound check.
func rfqReferenceRow(x []float32, w []float32, scale, eps float32) (codes []byte, q []float32) {
	width := len(w)
	invScale := float32(1.0) / scale
	rows := len(x) / width
	codes = make([]byte, len(x))
	q = make([]float32, len(x))
	for r := 0; r < rows; r++ {
		base := r * width
		var ss float32
		for i := 0; i < width; i++ {
			vv := x[base+i]
			ss = ss + vv*vv
		}
		denom := ss/float32(width) + eps
		inv := float32(1.0) / float32(math.Sqrt(float64(denom)))
		for i := 0; i < width; i++ {
			ni := x[base+i] * inv
			ni = ni * w[i]
			qi := ni * invScale
			q[base+i] = qi
			codes[base+i] = rfqEncodeE4M3(qi)
		}
	}
	return codes, q
}

// rfqHalfSpacing returns half the OCP e4m3fn bucket spacing at |q|: the top
// bucket [256,448] has spacing 32, normal buckets [2^j, 2^(j+1)) have spacing
// 2^(j-3), and the subnormal range below 2^-6 has spacing 2^-9. A round-to-
// nearest-even encode therefore always lands within this bound of q.
func rfqHalfSpacing(q float32) float32 {
	a := math.Abs(float64(q))
	if a >= 256 {
		return 16
	}
	if a < math.Ldexp(1, -6) {
		return float32(math.Ldexp(1, -10))
	}
	j := int(math.Floor(math.Log2(a)))
	return float32(math.Ldexp(1, j-4))
}
