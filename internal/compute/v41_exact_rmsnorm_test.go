package compute

import (
	"math"
	"math/rand"
	"os"
	"strings"
	"testing"
)

// v41ExactRMSNormInvScaleMirror is a line-for-line uint64 mirror of the
// shader's integer route for float32(1/sqrt(float64(y))): a restoring integer
// square root rounded to 53 bits, a restoring long division of 2^106 rounded to
// 53 bits, then RNE to 24 bits. Every step is exact, so the double rounding of
// the host formula is reproduced rather than approximated.
func v41ExactRMSNormInvScaleMirror(y float32) float32 {
	b := math.Float32bits(y)
	if b >= 0x7f800000 {
		return 0
	}
	m := uint64(b & 0x7fffff)
	e := int(b>>23) - 150
	if b>>23 == 0 {
		if m == 0 {
			return float32(math.Inf(1))
		}
		e = -149
		for m&0x800000 == 0 {
			m <<= 1
			e--
		}
	} else {
		m |= 0x800000
	}
	if e&1 != 0 {
		m <<= 1
		e--
	}
	var root, rem uint64
	step := func(pair uint64) {
		rem = rem<<2 | pair
		trial := root<<2 | 1
		root <<= 1
		if rem >= trial {
			rem -= trial
			root |= 1
		}
	}
	for i := 12; i >= 0; i-- {
		step((m >> (2 * uint(i))) & 3)
	}
	for i := 0; i < 42; i++ {
		step(0)
	}
	shift := uint(1)
	if root >= 1<<54 {
		shift = 2
	}
	s := root >> shift
	low, half := root&(1<<shift-1), uint64(1)<<(shift-1)
	es := e/2 - 42 + int(shift)
	if low > half || (low == half && (rem != 0 || s&1 == 1)) {
		s++
	}
	if s == 1<<53 {
		s = 1 << 52
		es++
	}
	var q, r uint64
	for i := 106; i >= 0; i-- {
		r <<= 1
		if i == 106 {
			r |= 1
		}
		q <<= 1
		if r >= s {
			r -= s
			q |= 1
		}
	}
	qq, lowq := q>>1, q&1
	eq := -105 - es
	if lowq == 1 && (r != 0 || qq&1 == 1) {
		qq++
	}
	if qq == 1<<53 {
		qq = 1 << 52
		eq++
	}
	mant, low29 := qq>>29, qq&(1<<29-1)
	f := eq + 29
	if low29 > 1<<28 || (low29 == 1<<28 && mant&1 == 1) {
		mant++
	}
	if mant == 1<<24 {
		mant = 1 << 23
		f++
	}
	return math.Float32frombits(uint32(f+23+127)<<23 | uint32(mant&0x7fffff))
}

func v41ExactRMSNormHostInvScale(y float32) float32 {
	return float32(1.0 / math.Sqrt(float64(y)))
}

// Ref #13668. The integer route the exact kernel runs equals the host's
// float64-sqrt-and-divide formula: every binade edge, every subnormal edge, and
// a dense random census. FAK_V41_EXACT_RMSNORM_EXHAUSTIVE=1 checks all 2^31
// positive finite binary32 inputs (minutes; offline witness).
// fak-test:runtime fast est=400ms lane=default
func TestV41ExactRMSNormInvScaleMirrorsHost(t *testing.T) {
	check := func(bitsY uint32) {
		y := math.Float32frombits(bitsY)
		got, want := v41ExactRMSNormInvScaleMirror(y), v41ExactRMSNormHostInvScale(y)
		if math.Float32bits(got) != math.Float32bits(want) {
			t.Fatalf("y=%#08x (%g): integer route %#08x, host %#08x", bitsY, y, math.Float32bits(got), math.Float32bits(want))
		}
	}
	check(0x7f800000)
	for exp := uint32(0); exp < 255; exp++ {
		for _, mant := range []uint32{0, 1, 2, 3, 0x400000, 0x7ffffd, 0x7ffffe, 0x7fffff} {
			if exp == 0 && mant == 0 {
				continue
			}
			check(exp<<23 | mant)
		}
	}
	rng := rand.New(rand.NewSource(13668))
	for i := 0; i < 1<<18; i++ {
		check(uint32(rng.Int63n(0x7f800000-1)) + 1)
	}
	if os.Getenv("FAK_V41_EXACT_RMSNORM_EXHAUSTIVE") == "1" {
		for b := uint32(1); b < 0x7f800000; b++ {
			check(b)
		}
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestV41ExactRMSNormDomainAndValidation(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		width int
		eps   float32
		want  bool
	}{
		{128, 1e-6, true}, {512, 1e-6, true}, {1, 1e-6, true}, {V41ExactRMSNormMaxWidth, 1, true},
		{0, 1e-6, false}, {384, 1e-6, false}, {2 * V41ExactRMSNormMaxWidth, 1e-6, false},
		{128, 0, false}, {128, -1e-6, false}, {128, float32(math.NaN()), false}, {128, float32(math.Inf(1)), false},
		{128, math.Float32frombits(0x00400000), false}, {128, math.Float32frombits(0x00800000), true},
	} {
		if got := V41ExactRMSNormDomain(tc.width, tc.eps); got != tc.want {
			t.Errorf("domain(%d, %g) = %v, want %v", tc.width, tc.eps, got, tc.want)
		}
	}
	ref := Default()
	w := NewF32(ref, []int{128}, make([]float32, 128))
	for _, tc := range []struct {
		name     string
		x        Tensor
		rows     int
		accepted bool
	}{
		{"vector", NewF32(ref, []int{128}, make([]float32, 128)), 1, true},
		{"matrix", NewF32(ref, []int{3, 128}, make([]float32, 384)), 3, true},
		{"width mismatch", NewF32(ref, []int{2, 64}, make([]float32, 128)), 0, false},
		{"rank three", NewF32(ref, []int{1, 1, 128}, make([]float32, 128)), 0, false},
		{"empty rows", NewF32(ref, []int{0, 128}, nil), 0, false},
	} {
		rows, width, err := validateV41ExactRMSNorm(tc.x, w, 1e-6)
		if (err == nil) != tc.accepted || (tc.accepted && (rows != tc.rows || width != 128)) {
			t.Errorf("%s: rows=%d width=%d err=%v", tc.name, rows, width, err)
		}
	}
	if _, _, err := validateV41ExactRMSNorm(NewF32(ref, []int{96}, make([]float32, 96)), NewF32(ref, []int{96}, make([]float32, 96)), 1e-6); err == nil {
		t.Error("non-power-of-two width admitted")
	}
}

// The shader is the device half of the mirror above; pin the steps a later
// edit could silently loosen (fused or reassociated arithmetic, a tree sum,
// the approximate inversesqrt) so the mirror keeps describing it.
// fak-test:runtime fast est=2ms lane=default
func TestV41ExactRMSNormShaderKeepsExactRoute(t *testing.T) {
	t.Parallel()
	src, err := os.ReadFile("shaders/v41_exact_rmsnorm.comp")
	if err != nil {
		t.Fatal(err)
	}
	shader := string(src)
	for _, token := range []string{
		"spirv_execution_mode(extensions = [\"SPV_KHR_float_controls\"], capabilities = [4464], 4459, 32);",
		"precise float sq = v * v;", "precise float next = ss + sq;",
		"precise float mean = ss * pc.invWidth;", "precise float y = mean + pc.eps;",
		"for (int i = 106; i >= 0; --i)", "precise float scaled = X[base + i] * inv;", "precise float out_ = scaled * W[i];",
	} {
		if !strings.Contains(shader, token) {
			t.Errorf("exact RMSNorm shader lost %q", token)
		}
	}
	for _, banned := range []string{"inversesqrt", "sqrt(", "fma(", "/ float", "subgroup", "shared float red"} {
		if strings.Contains(shader, banned) {
			t.Errorf("exact RMSNorm shader uses %q", banned)
		}
	}
}
