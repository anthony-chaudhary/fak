package compute

// kernel_matrix_test.go — adversarial tests authored from the mixed-precision kernel-matrix
// SPEC. They assert what the spec REQUIRES (unevaluated `test-case-03/04/05`): the undeclared-
// pair fail-closed contract, the per-backend resolution contract, the cpu-ref dispatch
// completeness, the recorded cosine floors, the KernelMatrixFor dispatch, and pair uniqueness.
//
// The tests deliberately probe BEYOND the implementation's happy path: an empty matrix, a
// non-nil backend with an unknown name, and a weight dtype the cpu-ref switch does not handle
// (which must panic — that panic is the proof the matrix is complete, not a bug).

import (
	"math"
	"testing"
)

// kmMatrices returns every per-backend matrix plus an empty one, with stable names for
// subtest labels and failure messages.
func kmMatrices() []struct {
	name string
	m    KernelMatrix
} {
	return []struct {
		name string
		m    KernelMatrix
	}{
		{"cpu-ref", CPURefKernelMatrix()},
		{"cuda", CUDAKernelMatrix()},
		{"vulkan", VulkanKernelMatrix()},
		{"rocm", ROCmKernelMatrix()},
		{"metal", MetalKernelMatrix()},
		{"w8a8", W8A8KernelMatrix()},
		{"empty", KernelMatrix{}},
	}
}

// kmRecordedFloor is the SPEC's recorded Approx cosine floor for one explicit (matrix, weight,
// activation) row. Test 4 asserts these literally — the values are transcribed from the spec,
// NOT from any implementation constant (the tagged cuda constants are invisible in the default
// build). Rows absent from this map are only held to the general (Approx, 0<CosineMin<=1) rule.
var kmRecordedFloor = map[string]float64{
	"cuda|q8_0|q8_0":   0.999,
	"cuda|q4_k|f32":    0.995,
	"cuda|q2_k|f32":    0.995,
	"cuda|q2_0|f32":    0.999,
	"cuda|f16|f16":     0.997,
	"cuda|fp8|f32":     0.999,
	"w8a8|q8_0|q8_0":   0.999,
	"vulkan|q8_0|q8_0": 0.999,
	"vulkan|q4_k|f32":  0.995,
	"rocm|fp4|f16":     0.995,
	"metal|q8_0|q8_0":  0.999,
}

// kmFloorKey names one row of kmRecordedFloor.
func kmFloorKey(matrix string, p Pair) string {
	return matrix + "|" + p.Weight.String() + "|" + p.Activation.String()
}

// TestKernelMatrixUndeclaredPairFailsClosed asserts a combination a matrix does NOT declare
// returns (Pair{}, false) — the zero Pair and no guessing. It probes several real matrices with
// weights that are definitely absent from all of them, and an empty matrix with a plausible pair.
func TestKernelMatrixUndeclaredPairFailsClosed(t *testing.T) {
	undeclared := []struct {
		name string
		w, a Dtype
	}{
		{"bf16/bf16", BF16, BF16},
		{"iq3_s/iq3_s", IQ3_S, IQ3_S},
		{"iq3_xxs/f32", IQ3_XXS, F32},
	}
	for _, m := range kmMatrices() {
		for _, u := range undeclared {
			t.Run(m.name+"/"+u.name, func(t *testing.T) {
				got, ok := m.m.Supports(u.w, u.a)
				if ok {
					t.Fatalf("Supports(%s, %s) = ok true; want false for an undeclared pair", u.w, u.a)
				}
				if got != (Pair{}) {
					t.Fatalf("Supports(%s, %s) returned %+v; want the zero Pair{}", u.w, u.a, got)
				}
			})
		}
	}

	// A plausible pair against an EMPTY matrix must also fail closed.
	var empty KernelMatrix
	got, ok := empty.Supports(F32, F32)
	if ok || got != (Pair{}) {
		t.Fatalf("empty KernelMatrix.Supports(F32, F32) = (%+v, %v); want (Pair{}, false)", got, ok)
	}
}

// TestKernelMatrixEveryDeclaredPairResolves asserts every declared row round-trips through
// Supports with all fields equal, and that the same weight paired with an undeclared activation
// is refused (the pair key is (Weight, Activation), not Weight alone).
func TestKernelMatrixEveryDeclaredPairResolves(t *testing.T) {
	undeclaredActivation := BF16 // present in no matrix's activation column
	for _, m := range kmMatrices() {
		if len(m.m.Pairs) == 0 {
			continue
		}
		for i, p := range m.m.Pairs {
			t.Run(m.name+"/row", func(t *testing.T) {
				got, ok := m.m.Supports(p.Weight, p.Activation)
				if !ok {
					t.Fatalf("row %d: Supports(%s, %s) = ok false; the declared pair must resolve", i, p.Weight, p.Activation)
				}
				if got != p {
					t.Fatalf("row %d: Supports(%s, %s) = %+v; want the declared %+v (all fields)", i, p.Weight, p.Activation, got, p)
				}
				// The same weight with an undeclared activation must NOT resolve.
				if _, ok := m.m.Supports(p.Weight, undeclaredActivation); ok && p.Activation != undeclaredActivation {
					t.Fatalf("row %d: Supports(%s, %s) = ok true; activation column must be part of the key", i, p.Weight, undeclaredActivation)
				}
			})
		}
	}
}

// TestKernelMatrixCPURefWeightSetIsExact asserts the cpu-ref matrix's weight-dtype set is
// EXACTLY the eight dtypes cpuref's MatMul switch handles — no more, no fewer — and that each
// named weight dtype is actually executable (no panic). A weight dtype the switch does not
// handle must NOT appear in the matrix, and is proven to panic when dispatched.
func TestKernelMatrixCPURefWeightSetIsExact(t *testing.T) {
	// SPEC-TRANSCRIBED, NOT derived: this map is a hand copy of the weight dtypes the
	// MatMul / BatchedMatMul switch arms in cpuref.go actually handle (F32, Q8_0, Q4_K,
	// Q5_K, Q6_K, Q2_K, IQ2_XXS, Q2_0). Because it is a SECOND COPY of the same knowledge,
	// it can drift in lockstep with a bug rather than catch one — it MUST be re-audited
	// against that switch whenever the switch's arms change.
	want := map[Dtype]bool{
		F32: true, Q8_0: true, Q4_K: true, Q5_K: true, Q6_K: true, Q2_K: true, IQ2_XXS: true, Q2_0: true,
	}
	got := map[Dtype]bool{}
	for _, p := range CPURefKernelMatrix().Pairs {
		got[p.Weight] = true
	}
	for d := range want {
		if !got[d] {
			t.Errorf("cpu-ref matrix missing weight dtype %s", d)
		}
	}
	for d := range got {
		if !want[d] {
			t.Errorf("cpu-ref matrix claims weight dtype %s, which is not an executable arm", d)
		}
	}

	// Every declared cpu-ref weight dtype must execute without panicking.
	for d := range want {
		t.Run("exec/"+d.String(), func(t *testing.T) {
			w := kmWeightTensor(cpu(), d, 256)
			x := NewF32(cpu(), []int{256}, make([]float32, 256))
			defer func() {
				if r := recover(); r != nil {
					t.Fatalf("cpu-ref MatMul panicked on declared weight dtype %s: %v", d, r)
				}
			}()
			_ = cpu().MatMul(w, x)
		})
	}

	// A weight dtype NOT in the matrix must not be silently widened to an f32 fallback: the
	// dispatch panics, and the matrix does not name it. The panic IS the completeness proof.
	for _, d := range []Dtype{BF16, IQ3_S} {
		if _, ok := CPURefKernelMatrix().Supports(d, F32); ok {
			t.Errorf("cpu-ref matrix claims unsupported weight dtype %s", d)
		}
		t.Run("panic/"+d.String(), func(t *testing.T) {
			w := kmWeightTensor(cpu(), F32, 256) // well-formed f32 buffer...
			w.Dtype = d                          // ...dispatched as an unsupported weight dtype
			x := NewF32(cpu(), []int{256}, make([]float32, 256))
			panicked := false
			func() {
				defer func() {
					if recover() != nil {
						panicked = true
					}
				}()
				_ = cpu().MatMul(w, x)
			}()
			if !panicked {
				t.Fatalf("cpu-ref MatMul did not panic on unsupported weight dtype %s; the matrix would be incomplete", d)
			}
		})
	}
}

// kmF32 builds a small deterministic f32 vector.
func kmF32(n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32(i%7) - 3
	}
	return out
}

// kmWeightTensor builds a well-formed [1,in] weight Tensor of the requested dtype using the
// exported host constructors, with deterministic non-degenerate bytes. `in` is 256 for the
// k-quant/i-quant widths and must be a multiple of 32 for q8_0/q2_0.
func kmWeightTensor(be Backend, d Dtype, in int) Tensor {
	shape := []int{1, in}
	switch d {
	case F32:
		return NewF32(be, shape, kmF32(in))
	case Q8_0:
		const block = 32
		codes := make([]int8, in)
		for i := range codes {
			codes[i] = int8(i%7 - 3)
		}
		scales := make([]float32, in/block)
		for i := range scales {
			scales[i] = 0.25
		}
		return NewQ8(be, shape, codes, scales, block)
	case Q4_K:
		return NewQ4K(be, shape, make([]byte, (in/256)*q4kSuperBlock))
	case Q5_K:
		return NewQ5K(be, shape, make([]byte, (in/256)*176))
	case Q6_K:
		return NewQ6K(be, shape, make([]byte, (in/256)*210))
	case Q2_K:
		return NewQ2K(be, shape, make([]byte, (in/256)*q2kSuperBlock))
	case IQ2_XXS:
		return NewIQ2XXS(be, shape, make([]byte, (in/256)*iq2xxsSuperBlock))
	case Q2_0:
		const block = 32
		packed := make([]byte, in/4)
		scales := make([]float32, in/block)
		for i := range scales {
			scales[i] = 0.5
		}
		return NewQ2(be, shape, packed, scales, block)
	default:
		panic("kmWeightTensor: no host constructor for " + d.String())
	}
}

// TestKernelMatrixApproxPairsCarryRecordedCosineFloor asserts the spec's recorded floors appear
// literally, every Reference row carries CosineMin 0, and every Approx row is in (0, 1].
func TestKernelMatrixApproxPairsCarryRecordedCosineFloor(t *testing.T) {
	const eps = 1e-9
	for _, m := range kmMatrices() {
		for i, p := range m.m.Pairs {
			// class / floor consistency, independent of the recorded table
			if p.Class == Reference {
				if p.CosineMin != 0 {
					t.Errorf("%s row %d (%s/%s): Reference CosineMin = %v; want 0", m.name, i, p.Weight, p.Activation, p.CosineMin)
				}
			} else {
				if p.Class != Approx {
					t.Errorf("%s row %d (%s/%s): non-Reference class = %v; want Approx", m.name, i, p.Weight, p.Activation, p.Class)
				}
				if p.CosineMin <= 0 || p.CosineMin > 1 {
					t.Errorf("%s row %d (%s/%s): Approx CosineMin = %v; want 0 < CosineMin <= 1", m.name, i, p.Weight, p.Activation, p.CosineMin)
				}
			}
			// recorded floor, where the spec names one
			if want, ok := kmRecordedFloor[kmFloorKey(m.name, p)]; ok {
				if math.Abs(p.CosineMin-want) >= eps {
					t.Errorf("%s row %d (%s/%s): CosineMin = %v; want the recorded floor %v", m.name, i, p.Weight, p.Activation, p.CosineMin, want)
				}
			}
		}
	}
}

// TestKernelMatrixForDispatch asserts the dispatcher fails closed for nil and unknown backends,
// and agrees with CPURefKernelMatrix for Default() on a reference host.
func TestKernelMatrixForDispatch(t *testing.T) {
	if got := KernelMatrixFor(nil); len(got.Pairs) != 0 {
		t.Fatalf("KernelMatrixFor(nil) declared %d pairs; want an empty matrix", len(got.Pairs))
	}

	if def := Default(); def != nil {
		want := CPURefKernelMatrix()
		got := KernelMatrixFor(def)
		if len(got.Pairs) != len(want.Pairs) {
			t.Fatalf("KernelMatrixFor(Default=%q) has %d pairs; CPU-ref has %d", def.Name(), len(got.Pairs), len(want.Pairs))
		}
		for i := range want.Pairs {
			if got.Pairs[i] != want.Pairs[i] {
				t.Fatalf("KernelMatrixFor(Default) row %d = %+v; CPU-ref = %+v", i, got.Pairs[i], want.Pairs[i])
			}
		}
	}

	// Unknown backend name, non-nil backend: the pre-existing foreignBackend double (collective_test.go)
	// wraps the reference kernels but reports Name()=="foreign", so it exercises the unknown-name
	// arm without building a fresh 20-method fake.
	unknown := foreignBackend{cpu()}
	if got := KernelMatrixFor(unknown); len(got.Pairs) != 0 {
		t.Fatalf("KernelMatrixFor(unknown-name backend %q) declared %d pairs; want an empty matrix (fail closed)", unknown.Name(), len(got.Pairs))
	}
	if _, ok := KernelMatrixFor(unknown).Supports(F32, F32); ok {
		t.Fatal("Supports on the unknown-backend matrix returned ok true; the empty matrix must fail closed")
	}
}

// TestKernelMatrixSupportsFirstMatch asserts Supports' documented first-match contract: when
// two declared pairs share the same (Weight, Activation), declaration order decides, and the
// FIRST one is returned. It also probes the Dtype zero value (Dtype(0) == F32) and the empty
// matrix's fail-closed result. All matrices here are SYNTHETIC — no GPU or backend is needed.
func TestKernelMatrixSupportsFirstMatch(t *testing.T) {
	// Two rows with the same key but different other fields: the first must win.
	first := Pair{Weight: F32, Activation: F32, Accumulate: F32, Class: Reference, CosineMin: 0}
	second := Pair{Weight: F32, Activation: F32, Accumulate: F32, Class: Approx, CosineMin: 0.9}
	two := KernelMatrix{Pairs: []Pair{first, second}}

	got, ok := two.Supports(F32, F32)
	if !ok {
		t.Fatalf("Supports(F32, F32) on a two-row synthetic matrix = ok false; want true")
	}
	if got != first {
		t.Fatalf("Supports(F32, F32) = %+v; want the FIRST declared row %+v (declaration order decides)", got, first)
	}
	if got != two.Pairs[0] {
		t.Fatalf("Supports(F32, F32) = %+v; want Pairs[0] %+v", got, two.Pairs[0])
	}

	// A single-pair matrix returns that pair.
	single := KernelMatrix{Pairs: []Pair{first}}
	if got, ok := single.Supports(F32, F32); !ok || got != first {
		t.Fatalf("Supports(F32, F32) on a single-pair matrix = (%+v, %v); want (%+v, true)", got, ok, first)
	}

	// The Dtype zero value is F32: Supports(Dtype(0), Dtype(0)) must resolve to the first
	// F32/F32 row of the synthetic matrix.
	if Dtype(0) != F32 {
		t.Fatalf("Dtype(0) = %s; this test assumes the zero value is F32", Dtype(0))
	}
	if got, ok := single.Supports(Dtype(0), Dtype(0)); !ok || got != first {
		t.Fatalf("Supports(Dtype(0), Dtype(0)) = (%+v, %v); want (%+v, true)", got, ok, first)
	}

	// An EMPTY matrix fails closed even for the zero-value key.
	var empty KernelMatrix
	if got, ok := empty.Supports(Dtype(0), Dtype(0)); ok || got != (Pair{}) {
		t.Fatalf("empty KernelMatrix.Supports(Dtype(0), Dtype(0)) = (%+v, %v); want (Pair{}, false)", got, ok)
	}
}

// TestKernelMatrixNoDuplicatePairs asserts no matrix declares the same (Weight, Activation)
// twice — a duplicate would make Supports' first-match order observable.
func TestKernelMatrixNoDuplicatePairs(t *testing.T) {
	for _, m := range kmMatrices() {
		seen := map[[2]Dtype]int{}
		for i, p := range m.m.Pairs {
			key := [2]Dtype{p.Weight, p.Activation}
			if first, dup := seen[key]; dup {
				t.Errorf("%s declares (%s, %s) at rows %d and %d; duplicates make Supports order-dependent", m.name, p.Weight, p.Activation, first, i)
			}
			seen[key] = i
		}
	}
}
