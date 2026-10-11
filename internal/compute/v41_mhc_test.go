package compute

import (
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// fak-test:runtime fast est=1ms lane=default
func TestV41MHCContract(t *testing.T) {
	t.Parallel()
	f32 := func(shape ...int) Tensor { return Tensor{Dtype: F32, Layout: RowMajor, Shape: shape} }
	const rows, hidden = 3, 8
	valid := []struct {
		op       V41MHCOp
		operands []Tensor
		out      []int
	}{
		{V41MHCOpInitial, []Tensor{f32(rows, hidden)}, []int{rows, 4, hidden}},
		{V41MHCOpSplit, []Tensor{f32(rows, 24), f32(3), f32(24)}, []int{rows, 24}},
		{V41MHCOpPre, []Tensor{f32(rows, 4, hidden), f32(rows, 24)}, []int{rows, hidden}},
		{V41MHCOpPost, []Tensor{f32(rows, hidden), f32(rows, 4, hidden), f32(rows, 24)}, []int{rows, 4, hidden}},
	}
	for _, tc := range valid {
		in, out, err := validateV41MHC(tc.op, rows, hidden, 20, 1e-6, tc.operands...)
		if err != nil || len(in) != len(tc.operands) || !equalInts(out, tc.out) {
			t.Fatalf("%s valid geometry: in=%d out=%v err=%v", tc.op, len(in), out, err)
		}
		// Every operand is shape-, dtype-, layout- and quant-checked.
		for i := range tc.operands {
			for _, mutate := range []func(*Tensor){
				func(x *Tensor) { x.Shape = append([]int{1}, x.Shape...) },
				func(x *Tensor) { x.Dtype = BF16 },
				func(x *Tensor) { x.Layout = ColMajor },
				func(x *Tensor) { x.Quant = &QuantSpec{} },
			} {
				bad := append([]Tensor(nil), tc.operands...)
				bad[i].Shape = append([]int(nil), bad[i].Shape...)
				mutate(&bad[i])
				if _, _, err := validateV41MHC(tc.op, rows, hidden, 20, 1e-6, bad...); err == nil {
					t.Errorf("%s accepted malformed operand %d: %+v", tc.op, i, bad[i])
				}
			}
		}
		if _, _, err := validateV41MHC(tc.op, rows, hidden, 20, 1e-6, tc.operands[:len(tc.operands)-1]...); err == nil {
			t.Errorf("%s accepted a missing operand", tc.op)
		}
	}
	for _, bad := range []struct {
		op                  V41MHCOp
		rows, hidden, iters int
		eps                 float32
	}{
		{V41MHCOpInitial, 0, hidden, 1, 1e-6},
		{V41MHCOpPre, rows, 0, 1, 1e-6},
		{V41MHCOpPost, rows, 1 << 30, 1, 1e-6},
		{V41MHCOpSplit, rows, 0, 0, 1e-6},
		{V41MHCOpSplit, rows, 0, 1, 0},
		{V41MHCOpSplit, rows, 0, 1, float32(math.NaN())},
		{V41MHCOpSplit, rows, 0, 1, float32(math.Inf(1))},
		{V41MHCOp(9), rows, hidden, 1, 1e-6},
	} {
		if _, _, err := validateV41MHC(bad.op, bad.rows, bad.hidden, bad.iters, bad.eps); err == nil {
			t.Errorf("accepted invalid geometry %+v", bad)
		}
	}
}

func equalInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// The oracles below are an independent, test-only transcription of the host
// model's mHC arithmetic (internal/model/v41_mhc.go): explicit F32 rounding of
// every product and sum in stream order, F64 exp/sigmoid narrowed to F32, and
// integer BF16 round-to-nearest-even. No production backend can select them.
func v41MHCRefBF16(v float32) float32 {
	bits := math.Float32bits(v)
	bits += 0x7fff + ((bits >> 16) & 1)
	return math.Float32frombits(bits & 0xffff0000)
}

func v41MHCRefInitial(x []float32, rows, hidden int) []float32 {
	out := make([]float32, rows*4*hidden)
	for r := 0; r < rows; r++ {
		for h := 0; h < 4; h++ {
			for d := 0; d < hidden; d++ {
				out[(r*4+h)*hidden+d] = v41MHCRefBF16(x[r*hidden+d])
			}
		}
	}
	return out
}

func v41MHCRefSigmoid(x float32) float32 { return float32(1 / (1 + math.Exp(float64(-x)))) }

func v41MHCRefSplit(mixes, scale, base []float32, rows, iters int, eps float32) []float32 {
	out := make([]float32, rows*24)
	for r := 0; r < rows; r++ {
		m, o := mixes[r*24:(r+1)*24], out[r*24:(r+1)*24]
		for j := 0; j < 4; j++ {
			o[j] = v41MHCRefSigmoid(float32(m[j]*scale[0])+base[j]) + eps
			o[4+j] = 2 * v41MHCRefSigmoid(float32(m[4+j]*scale[1])+base[4+j])
		}
		c := o[8:]
		for i := range c {
			c[i] = float32(m[8+i]*scale[2]) + base[8+i]
		}
		for row := 0; row < 4; row++ {
			mx := c[row*4]
			for col := 1; col < 4; col++ {
				mx = max(mx, c[row*4+col])
			}
			var sum float32
			for col := 0; col < 4; col++ {
				c[row*4+col] = float32(math.Exp(float64(c[row*4+col] - mx)))
				sum += c[row*4+col]
			}
			for col := 0; col < 4; col++ {
				c[row*4+col] = c[row*4+col]/sum + eps
			}
		}
		cols := func() {
			for col := 0; col < 4; col++ {
				var sum float32
				for row := 0; row < 4; row++ {
					sum += c[row*4+col]
				}
				for row := 0; row < 4; row++ {
					c[row*4+col] /= sum + eps
				}
			}
		}
		cols()
		for iter := 1; iter < iters; iter++ {
			for row := 0; row < 4; row++ {
				var sum float32
				for col := 0; col < 4; col++ {
					sum += c[row*4+col]
				}
				for col := 0; col < 4; col++ {
					c[row*4+col] /= sum + eps
				}
			}
			cols()
		}
	}
	return out
}

func v41MHCRefPre(streams, coeffs []float32, rows, hidden int) []float32 {
	out := make([]float32, rows*hidden)
	for r := 0; r < rows; r++ {
		for d := 0; d < hidden; d++ {
			var acc float32
			for h := 0; h < 4; h++ {
				acc += float32(coeffs[r*24+h] * streams[(r*4+h)*hidden+d])
			}
			out[r*hidden+d] = v41MHCRefBF16(acc)
		}
	}
	return out
}

func v41MHCRefPost(x, streams, coeffs []float32, rows, hidden int) []float32 {
	out := make([]float32, rows*4*hidden)
	for r := 0; r < rows; r++ {
		k := coeffs[r*24 : (r+1)*24]
		for dst := 0; dst < 4; dst++ {
			for d := 0; d < hidden; d++ {
				var acc float32
				for src := 0; src < 4; src++ {
					acc += float32(k[8+src*4+dst] * streams[(r*4+src)*hidden+d])
				}
				acc += float32(k[4+dst] * v41MHCRefBF16(x[r*hidden+d]))
				out[(r*4+dst)*hidden+d] = v41MHCRefBF16(acc)
			}
		}
	}
	return out
}

// fak-test:runtime fast est=1ms lane=default
func TestV41MHCReferenceSinkhornIsDoublyStochastic(t *testing.T) {
	t.Parallel()
	mixes, scale, base := v41MHCSplitFixture(2)
	coeffs := v41MHCRefSplit(mixes, scale, base, 2, 20, 1e-6)
	for r := 0; r < 2; r++ {
		c := coeffs[r*24+8 : r*24+24]
		for i := 0; i < 4; i++ {
			var row, col float64
			for j := 0; j < 4; j++ {
				row += float64(c[i*4+j])
				col += float64(c[j*4+i])
			}
			// Columns are normalized last; rows of the deliberately
			// near-degenerate 90-wide softmax converge slowly within 20 steps.
			if math.Abs(row-1) > 0.1 || math.Abs(col-1) > 1e-5 {
				t.Fatalf("row %d: Sinkhorn sums row=%g col=%g", r, row, col)
			}
		}
		for j := 0; j < 4; j++ {
			if pre := coeffs[r*24+j]; pre <= 0 || pre > 1+1e-5 {
				t.Fatalf("pre[%d]=%g outside (0,1]", j, pre)
			}
			if post := coeffs[r*24+4+j]; post < 0 || post > 2 {
				t.Fatalf("post[%d]=%g outside [0,2]", j, post)
			}
		}
	}
}

// v41MHCSplitFixture spans saturated and central sigmoid arguments and
// softmax rows with large spreads, so exp range reduction is exercised.
func v41MHCSplitFixture(rows int) (mixes, scale, base []float32) {
	mixes = make([]float32, rows*24)
	for i := range mixes {
		mixes[i] = float32((i*37)%41-20) * 0.173
	}
	if rows > 0 {
		mixes[0], mixes[1] = 60, -60 // saturated pre arguments
		mixes[8], mixes[9] = 45, -45 // a 90-wide softmax row
	}
	scale = []float32{0.75, 1.25, 1.5}
	base = make([]float32, 24)
	for i := range base {
		base[i] = float32(i%5-2) * 0.0625
	}
	return mixes, scale, base
}

// fak-test:runtime fast est=5ms lane=default
// Source witness only: no compiler invocation, SPIR-V fabrication, or GPU proof.
func TestV41MHCSourceContract(t *testing.T) {
	t.Parallel()
	read := func(path string) string {
		t.Helper()
		raw, err := os.ReadFile(filepath.FromSlash(path))
		if err != nil {
			t.Fatal(err)
		}
		return string(raw)
	}
	shader := read("shaders/v41_mhc.comp")
	for _, want := range []string{
		"int mode;", "int rows;", "int hidden;", "int iters;", "float eps;",
		"bits += 0x7fffu + ((bits >> 16) & 1u);", "bits & 0xffff0000u",
		"precise float p = B[m + h] * A[s + h * H];",
		"precise float x = C[m + 4u + dst] * bf16(A[row * H + d]);",
		"layout(std430, binding = 3) writeonly buffer OutBuf",
	} {
		if !strings.Contains(shader, want) {
			t.Fatalf("v41_mhc.comp lacks %q", want)
		}
	}
	shim := read("vulkan_shim.cpp")
	for _, want := range []string{
		`P("v41_mhc.spv"), 4, 5 * sizeof(int)`,
		"struct Push { int32_t mode, rows, hidden, iters; float eps; }",
		"for (int i = 0; i < 3; ++i) if (aliases(bufs[3], bufs[i])) return 2;",
	} {
		if !strings.Contains(shim, want) {
			t.Fatalf("vulkan_shim.cpp lacks %q", want)
		}
	}
	if !strings.Contains(read("build_vulkan.ps1"), `$shaders += "v41_mhc"`) {
		t.Fatal("build_vulkan.ps1 does not compile v41_mhc")
	}
}
