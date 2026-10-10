//go:build linux && rocm && cgo

package compute

import (
	"encoding/binary"
	"errors"
	"go/ast"
	"go/parser"
	"go/token"
	"math"
	"math/rand"
	"os"
	"reflect"
	"testing"
	"unsafe"
)

func rocmRequired(t *testing.T) Backend {
	t.Helper()
	b, ok := Lookup("rocm")
	if !ok {
		if os.Getenv("FAK_ROCM_REQUIRE_DEVICE") == "1" {
			t.Fatal("FAK_ROCM_REQUIRE_DEVICE=1: native ROCm backend unavailable")
		}
		t.Skip("native ROCm backend unavailable")
	}
	return b
}

func rocmResident(b Backend, shape []int, data []float32) Tensor {
	return b.Upload(NewF32(Default(), shape, append([]float32(nil), data...)), F32)
}

func rocmClose(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if math.IsNaN(float64(a[i])) || math.IsNaN(float64(b[i])) || math.IsInf(float64(a[i]), 0) || math.IsInf(float64(b[i]), 0) {
			return false
		}
		if math.Abs(float64(a[i]-b[i])) > 2e-4 {
			return false
		}
	}
	return true
}

func rocmCosine(a, b []float32) float64 {
	if len(a) == 0 || len(a) != len(b) {
		return math.NaN()
	}
	var dot, na, nb float64
	for i := range a {
		x, y := float64(a[i]), float64(b[i])
		if math.IsNaN(x) || math.IsNaN(y) || math.IsInf(x, 0) || math.IsInf(y, 0) {
			return math.NaN()
		}
		dot += x * y
		na += x * x
		nb += y * y
	}
	if na == 0 || nb == 0 {
		return math.NaN()
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}

func TestROCmBackendCoreBehavior(t *testing.T) {
	b, ref := rocmRequired(t), Default()
	if b.Name() != "rocm" || b.Class() != Approx || !b.Caps().DeviceMemory {
		t.Fatalf("identity/caps: %s %s %+v", b.Name(), b.Class(), b.Caps())
	}
	if total, free, ok := DeviceMemoryInfo(b); !ok || total <= 0 || free < 0 || free > total {
		t.Fatalf("device memory: total=%d free=%d known=%v", total, free, ok)
	}
	x := rocmResident(b, []int{4}, []float32{1, -2, 3, .5})
	if h, ok := b.Host(x); ok || h != nil {
		t.Fatal("resident tensor became host-addressable")
	}
	clone, err := b.(TensorCloner).CloneTensor(x)
	if err != nil {
		t.Fatal(err)
	}
	b.Free(x)
	if got := b.Read(clone); !rocmClose(got, []float32{1, -2, 3, .5}) {
		t.Fatalf("clone lifetime: %v", got)
	}
	w := []float32{1, 2, 3, 4, -1, .5, 2, -2, 0, 1, 0, 1}
	wr, wd := rocmResident(ref, []int{3, 4}, w), rocmResident(b, []int{3, 4}, w)
	xr, xd := rocmResident(ref, []int{4}, []float32{.5, -1, 2, 3}), rocmResident(b, []int{4}, []float32{.5, -1, 2, 3})
	if want, got := ref.Read(ref.MatMul(wr, xr)), b.Read(b.MatMul(wd, xd)); !rocmClose(want, got) {
		t.Fatalf("matmul want=%v got=%v", want, got)
	}
	tailW := make([]float32, 259*4)
	for i := range tailW {
		tailW[i] = float32(i%17-8) / 13
	}
	if want, got := ref.Read(ref.MatMul(rocmResident(ref, []int{259, 4}, tailW), xr)), b.Read(b.MatMul(rocmResident(b, []int{259, 4}, tailW), xd)); !rocmClose(want, got) {
		t.Fatal("259-row matmul tail mismatch")
	}
	panel := []float32{.5, -1, 2, 3, 2, 1, -1, .25}
	if want, got := ref.Read(ref.BatchedMatMul(wr, rocmResident(ref, []int{2, 4}, panel), 2)), b.Read(b.BatchedMatMul(wd, rocmResident(b, []int{2, 4}, panel), 2)); !rocmClose(want, got) {
		t.Fatalf("batched want=%v got=%v", want, got)
	}
	weights := []float32{1, .9, 1.1, 1}
	for name, op := range map[string]func(Backend) []float32{
		"rms": func(q Backend) []float32 {
			return q.Read(q.RMSNorm(rocmResident(q, []int{4}, panel[:4]), rocmResident(q, []int{4}, weights), 1e-5))
		},
		"rope": func(q Backend) []float32 { return q.Read(q.RoPE(rocmResident(q, []int{4}, panel[:4]), 7, 2, 2, 10000)) },
		"swiglu": func(q Backend) []float32 {
			return q.Read(q.SwiGLU(rocmResident(q, []int{4}, panel[:4]), rocmResident(q, []int{4}, panel[4:])))
		},
	} {
		if want, got := op(ref), op(b); !rocmClose(want, got) {
			t.Fatalf("%s want=%v got=%v", name, want, got)
		}
	}
	d := rocmResident(b, []int{4}, []float32{1, 2, 3, 4})
	b.AddInPlace(d, rocmResident(b, []int{4}, []float32{4, 3, 2, 1}))
	b.AddBias(d, rocmResident(b, []int{2}, []float32{1, -1}))
	if got := b.Read(d); !rocmClose(got, []float32{6, 4, 6, 4}) {
		t.Fatalf("add/bias: %v", got)
	}
	if got := b.Argmax(rocmResident(b, []int{7}, []float32{-3, 9, 9, 1, 2, 8, 8})); got != 1 {
		t.Fatalf("argmax first tie/tail=%d", got)
	}
}

// TestROCmBackendDSAIndexSelectHostExact is the GPU-FREE witness for rocmBackend.DSAIndexSelect:
// the method is a deliberate HOST delegation (the indexer drives a discrete top-k, so the selected
// set must be bit-identical to the cpu-ref ? see dsa.go's selection-on-host rationale), so it can be
// exercised with a bare backend and HOST-resident tensors, touching no HIP call and needing no device.
// It requires only the rocm build tag (this file is linux && rocm && cgo), not physical hardware; on a
// tag build without a GPU it still runs, and it pins the selected POSITIONS to the independent f64
// oracle hostIndexSelectF64 (dsa_index_test.go) across the same geometry sweep the cpu-ref test uses.
func TestROCmBackendDSAIndexSelectHostExact(t *testing.T) {
	r := &rocmBackend{name: "rocm"}
	scale := float32(1.0 / math.Sqrt(8))
	cases := []struct {
		name            string
		nH, indexDim    int
		nKeys, queryPos int
		topK            int
		tie             bool
	}{
		{"small", 4, 8, 6, 5, 2, false},
		{"topk-ge-keys", 4, 8, 3, 2, 8, false},
		{"causal-mask", 4, 8, 10, 4, 3, false},
		{"near-tie", 2, 4, 8, 7, 3, true},
		{"single-head", 1, 16, 12, 11, 5, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			indexQ := make([]float32, tc.nH*tc.indexDim)
			for i := range indexQ {
				indexQ[i] = float32(math.Sin(float64(i)*0.7+1.3)) * 0.5
			}
			indexK := make([]float32, tc.nKeys*tc.indexDim)
			for i := range indexK {
				indexK[i] = float32(math.Cos(float64(i)*0.37+0.2)) * 0.5
			}
			weights := make([]float32, tc.nH)
			for h := range weights {
				weights[h] = float32(0.3 + 0.1*float64(h))
			}
			if tc.tie {
				copy(indexK[2*tc.indexDim:3*tc.indexDim], indexK[1*tc.indexDim:2*tc.indexDim])
			}

			want := hostIndexSelectF64(indexQ, indexK, weights, tc.nKeys, tc.nH, tc.indexDim, tc.queryPos, tc.topK, scale)

			// Host-resident (Default()) tensors: rocmBackend.Read returns their host slice directly,
			// so no device buffer and no HIP call is reached.
			qt := NewF32(Default(), []int{tc.nH * tc.indexDim}, indexQ)
			kt := NewF32(Default(), []int{tc.nKeys * tc.indexDim}, indexK)
			wt := NewF32(Default(), []int{tc.nH}, weights)
			got := r.DSAIndexSelect(qt, kt, wt, tc.nKeys, tc.nH, tc.indexDim, tc.queryPos, tc.topK, scale)

			if len(got) != len(want) {
				t.Fatalf("selection length: got %d want %d (got=%v want=%v)", len(got), len(want), got, want)
			}
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("selection[%d]: got %d want %d (full got=%v want=%v) ? NOT selection-stable", i, got[i], want[i], got, want)
				}
			}
			seen := map[int]bool{}
			for _, p := range got {
				if p < 0 || p > tc.queryPos {
					t.Fatalf("selected non-causal position %d (queryPos=%d)", p, tc.queryPos)
				}
				if seen[p] {
					t.Fatalf("selected duplicate position %d", p)
				}
				seen[p] = true
			}
			t.Logf("rocm DSAIndexSelect (host-delegated) %s: selection==host-f64 exactly: %v", tc.name, got)
		})
	}
}

func TestROCmBackendKVCloneEvict(t *testing.T) {
	b, ref := rocmRequired(t), Default()
	kvCfg := KVConfig{NumLayers: 1, NumKVHeads: 1, HeadDim: 4, RopeTheta: 10000}
	kv := b.NewKV(kvCfg)
	kr := ref.NewKV(kvCfg)
	defer kv.(interface{ Free() }).Free()
	for p := 0; p < 3; p++ {
		raw := rocmResident(b, []int{4}, []float32{float32(p), 1, 2, 3})
		rope := b.RoPE(raw, p, 1, 4, 10000)
		val := rocmResident(b, []int{4}, []float32{4, 3, 2, float32(p)})
		kv.AppendKV(0, raw, rope, val, p)
		rawR := rocmResident(ref, []int{4}, []float32{float32(p), 1, 2, 3})
		kr.AppendKV(0, rawR, ref.RoPE(rawR, p, 1, 4, 10000), rocmResident(ref, []int{4}, []float32{4, 3, 2, float32(p)}), p)
	}
	if want, got := ref.Read(kr.KeysView(0)), b.Read(kv.KeysView(0)); !rocmClose(want, got) {
		t.Fatalf("KV keys want=%v got=%v", want, got)
	}
	qR := rocmResident(ref, []int{8}, []float32{1, 2, 3, 4, 4, 3, 2, 1})
	qD := rocmResident(b, []int{8}, []float32{1, 2, 3, 4, 4, 3, 2, 1})
	if want, got := ref.Read(ref.Attention(qR, kr, 0, true, 2, .37)), b.Read(b.Attention(qD, kv, 0, true, 2, .37)); !rocmClose(want, got) {
		t.Fatalf("attention want=%v got=%v", want, got)
	}
	copy := kv.Clone()
	defer copy.(interface{ Free() }).Free()
	if kv.Evict(1, 1) != 1 || kr.Evict(1, 1) != 1 || kv.Len() != 2 || copy.Len() != 3 {
		t.Fatalf("clone/evict lens original=%d clone=%d", kv.Len(), copy.Len())
	}
	if want, got := ref.Read(kr.KeysView(0)), b.Read(kv.KeysView(0)); !rocmClose(want, got) {
		t.Fatalf("evicted keys want=%v got=%v", want, got)
	}
	if want, got := ref.Read(kr.ValuesView(0)), b.Read(kv.ValuesView(0)); !rocmClose(want, got) {
		t.Fatalf("evicted values want=%v got=%v", want, got)
	}
	if h, ok := b.Host(copy.KeysView(0)); ok || h != nil {
		t.Fatal("KV view became host-addressable")
	}
	if len(b.Read(copy.ValuesView(0))) != copy.Len()*kvCfg.NumKVHeads*kvCfg.HeadDim {
		t.Fatal("clone changed after source eviction")
	}
}

func TestROCmBackendPackedMatMulPaths(t *testing.T) {
	b, ref := rocmRequired(t), Default()
	x := make([]float32, 256)
	for i := range x {
		x[i] = float32(i%11-5) / 7
	}
	q8Weights := append([]float32(nil), x...)
	for i := range x {
		q8Weights = append(q8Weights, -x[255-i]*.73)
	}
	q8 := QuantizeQ8(ref, []int{2, 256}, q8Weights, 32)
	q8Codes := q8.Buf().(HostBuffer).I8()
	q8Dequant := make([]float32, len(q8Codes))
	for i, code := range q8Codes {
		q8Dequant[i] = float32(code) * q8.Quant.Scale[(i/256)*8+(i%256)/32]
	}
	q8Oracle := make([]float32, 2)
	for row := range q8Oracle {
		for col, xv := range x {
			q8Oracle[row] += q8Dequant[row*256+col] * xv
		}
	}
	rng := rand.New(rand.NewSource(12755))
	q4raw := make([]byte, 288)
	for off := 0; off < len(q4raw); off += 144 {
		randQ4KBlockC(rng, q4raw[off:off+144])
	}
	q5raw := make([]byte, 352)
	rng.Read(q5raw)
	for off := 0; off < len(q5raw); off += 176 {
		binary.LittleEndian.PutUint16(q5raw[off:], 0x3000)
		binary.LittleEndian.PutUint16(q5raw[off+2:], 0x2c00)
	}
	q6raw := make([]byte, 420)
	rng.Read(q6raw)
	for off := 0; off < len(q6raw); off += 210 {
		binary.LittleEndian.PutUint16(q6raw[off+208:], 0x3000)
	}
	for _, tc := range []struct {
		name string
		host Tensor
		dt   Dtype
	}{{"q8", q8, Q8_0}, {"q4k", NewQ4K(ref, []int{2, 256}, q4raw), Q4_K}, {"q5k", NewQ5K(ref, []int{2, 256}, q5raw), Q5_K}, {"q6k", NewQ6K(ref, []int{2, 256}, q6raw), Q6_K}} {
		cpuWant := ref.Read(ref.MatMul(tc.host, rocmResident(ref, []int{256}, x)))
		got := b.Read(b.MatMul(b.Upload(tc.host, tc.dt), rocmResident(b, []int{256}, x)))
		want := cpuWant
		if tc.dt == Q8_0 {
			want = q8Oracle
		}
		if len(got) != len(want) {
			t.Fatalf("%s length want=%d got=%d", tc.name, len(want), len(got))
		}
		if cos := rocmCosine(want, got); math.IsNaN(cos) || cos < 0.999 {
			t.Fatalf("%s cosine=%v want=%v got=%v", tc.name, cos, want, got)
		}
		for i := range want {
			delta := math.Abs(float64(want[i] - got[i]))
			bound := 2e-4 + 2e-5*math.Abs(float64(want[i]))
			if math.IsNaN(delta) || math.IsInf(delta, 0) || delta > bound {
				t.Fatalf("%s[%d] want=%v got=%v abs=%v bound=%v", tc.name, i, want[i], got[i], delta, bound)
			}
		}
		if tc.dt == Q8_0 {
			if len(cpuWant) != len(got) {
				t.Fatalf("q8 CPU comparison length want=%d got=%d", len(cpuWant), len(got))
			}
			for i := range cpuWant {
				delta := math.Abs(float64(cpuWant[i] - got[i]))
				if math.IsNaN(delta) || math.IsInf(delta, 0) || delta > 0.05 {
					t.Fatalf("q8 CPU-quantized activation[%d] want=%v got=%v abs=%v", i, cpuWant[i], got[i], delta)
				}
			}
		}
	}
}

// fak-test:runtime fast est=1ms
func TestROCmBackendWeightDtypeAdmission(t *testing.T) {
	r := &rocmBackend{}
	for _, tc := range []struct {
		dtype Dtype
		want  bool
	}{
		{F32, true}, {Q8_0, true}, {Q4_K, true}, {Q5_K, true}, {Q6_K, true},
		{F16, false}, {BF16, false}, {I8, false}, {I4, false}, {FP8, false},
		{Q2_0, false}, {Q2_K, false}, {Q3_K, false}, {IQ3_XXS, false},
		{IQ3_S, false}, {IQ2_XXS, false}, {FP4, false}, {IQ4_XS, false},
		{IQ2_S, false}, {IQ2_XS, false}, {IQ1_S, false}, {Dtype(255), false},
	} {
		if got := r.SupportsDeviceWeightDtype(tc.dtype); got != tc.want {
			t.Errorf("dtype %s (%d): support = %v, want %v", tc.dtype, tc.dtype, got, tc.want)
		}
	}
}

// fak-test:runtime fast est=1ms
func TestROCmBackendRejectsWeightLayoutBeforeDeviceAccess(t *testing.T) {
	for _, tc := range []struct {
		name   string
		layout Layout
		rows   int
	}{
		{"decode_colmajor", ColMajor, 1},
		{"decode_tiled", Tiled, 1},
		{"batch_colmajor", ColMajor, 2},
		{"batch_tiled", Tiled, 2},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r := &rocmBackend{}
			// Metadata-only operands must be rejected before any device access or allocation.
			w := Tensor{Dtype: F32, Layout: tc.layout, Shape: []int{2, 32}}
			x := Tensor{Dtype: F32, Layout: RowMajor, Shape: []int{tc.rows, 32}}
			defer func() {
				got := recover()
				err, ok := got.(error)
				if !ok || err.Error() != "rocm: matmul requires row-major weight tensor" {
					t.Fatalf("panic = %v, want weight-layout rejection before device access", got)
				}
			}()
			if tc.rows == 1 {
				r.MatMul(w, x)
			} else {
				r.BatchedMatMul(w, x, tc.rows)
			}
		})
	}
}

// Runtime estimates are unmeasured; this control requires the ROCm build only.
// fak-test:runtime fast est=1ms lane=default
func TestROCmLimitedSwiGLUInvalidLimit(t *testing.T) {
	r := &rocmBackend{}
	for _, limit := range []float32{0, -1, float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1))} {
		func() {
			defer func() {
				value := recover()
				err, ok := value.(error)
				if !ok || err.Error() != "rocm: limited swiglu requires finite positive limit" {
					t.Errorf("limit=%g refusal=%v", limit, value)
				}
			}()
			// Empty operands ensure invalid limits refuse before device access.
			r.SwiGLUWithLimit(Tensor{}, Tensor{}, limit)
		}()
	}
}

// Runtime estimate only, unmeasured. Requires an actual ROCm backend/device;
// absence follows rocmRequired's explicit skip/FAK_ROCM_REQUIRE_DEVICE policy.
// fak-test:runtime medium est=2s lane=default
func TestROCmLimitedSwiGLUOwnedPublication(t *testing.T) {
	b := rocmRequired(t)
	operation, ok := b.(interface {
		SwiGLUWithLimit(Tensor, Tensor, float32) Tensor
	})
	if !ok {
		t.Fatal("ROCm lacks optional limited SwiGLU")
	}
	const n = 518
	gate, up, want := make([]float32, n), make([]float32, n), make([]float32, n)
	gs := []float32{-9, -3, -2, -0.25, 0, 0.25, 2, 3, 9}
	us := []float32{9, -9, 3, -3, 0, 1, -1, 2, -2}
	for i := range gate {
		gate[i], up[i] = gs[i%len(gs)], us[i%len(us)]
		g, u := float64(gate[i]), float64(up[i])
		if g > 2 {
			g = 2
		}
		if u > 2 {
			u = 2
		}
		if u < -2 {
			u = -2
		}
		want[i] = float32((g / (1 + math.Exp(-g))) * u)
	}
	gd, ud := rocmResident(b, []int{2, 259}, gate), rocmResident(b, []int{2, 259}, up)
	defer b.Free(gd)
	defer b.Free(ud)
	out := operation.SwiGLUWithLimit(gd, ud, 2)
	defer b.Free(out)
	if out.Backend() != b || out.Dtype != F32 || len(out.Shape) != 2 || out.Shape[0] != 2 || out.Shape[1] != 259 || out.Buf() == gd.Buf() || out.Buf() == ud.Buf() {
		t.Fatal("limited SwiGLU did not publish an independently owned matching tensor")
	}
	if got := b.Read(out); !rocmClose(got, want) {
		t.Fatal("asymmetric clamp / launch tail differs from scalar oracle")
	}
	if !rocmClose(b.Read(gd), gate) || !rocmClose(b.Read(ud), up) {
		t.Fatal("limited operation changed input operands")
	}
	// -9 remains -9; a symmetric gate clamp would produce a materially different result.
	wrong := float32((-2 / (1 + math.Exp(2))) * 2)
	if math.Abs(float64(want[0]-wrong)) < 0.1 {
		t.Fatal("negative gate discriminator is vacuous")
	}
	// Exact shape equality, not merely equal element count; failures must leave operands usable.
	bad := ud
	bad.Shape = []int{1, n}
	func() {
		defer func() {
			if recover() == nil {
				t.Error("equal-numel different shapes admitted")
			}
		}()
		operation.SwiGLUWithLimit(gd, bad, 2)
	}()
	bad = ud
	bad.Layout = ColMajor
	func() {
		defer func() {
			if recover() == nil {
				t.Error("non-row-major input admitted")
			}
		}()
		operation.SwiGLUWithLimit(gd, bad, 2)
	}()
	host := NewF32(Default(), []int{2, 259}, up)
	func() {
		defer func() {
			if recover() == nil {
				t.Error("foreign host operand admitted")
			}
		}()
		operation.SwiGLUWithLimit(gd, host, 2)
	}()
	if !rocmClose(b.Read(gd), gate) || !rocmClose(b.Read(ud), up) {
		t.Fatal("validation refusal damaged input operands")
	}
	// Ordered comparisons preserve NaN instead of silently replacing it by limit.
	ng, nu := rocmResident(b, []int{2}, []float32{float32(math.NaN()), 1}), rocmResident(b, []int{2}, []float32{1, float32(math.NaN())})
	defer b.Free(ng)
	defer b.Free(nu)
	ny := operation.SwiGLUWithLimit(ng, nu, 2)
	defer b.Free(ny)
	nanValues := b.Read(ny)
	if len(nanValues) != 2 {
		t.Fatal("NaN publication width changed")
	}
	for _, v := range nanValues {
		if !math.IsNaN(float64(v)) {
			t.Fatal("clamp hid a NaN operand")
		}
	}
}

// fak-test:runtime fast est=5ms lane=default
// Source-authored estimate, unmeasured. Injected ownership operations only: no
// device lookup, HIP allocation, copy, synchronization or native fault injection.
func TestROCmCloneStorageUnwindsUnpublishedAllocations(t *testing.T) {
	allocationFailure := &DeviceAllocError{Bytes: 8, Site: "clone-scales", Class: MemoryWeights}
	copyFailure := errors.New("injected native D2D failure")
	unknownFailure := &struct{ name string }{"unknown native panic"}
	cleanupFailure := &struct{ name string }{"cleanup failure"}
	for _, tc := range []struct {
		name                             string
		allocPanic, copyError, copyPanic int
		cleanupPanic                     bool
		wantRelease                      []int
	}{
		{name: "first-allocation", allocPanic: 1},
		{name: "second-allocation", allocPanic: 2, wantRelease: []int{1}},
		{name: "second-allocation-cleanup-panic", allocPanic: 2, cleanupPanic: true, wantRelease: []int{1}},
		{name: "first-copy", copyError: 1, wantRelease: []int{1}},
		{name: "second-copy", copyError: 2, wantRelease: []int{2, 1}},
		{name: "second-copy-both-cleanups-panic", copyError: 2, cleanupPanic: true, wantRelease: []int{2, 1}},
		{name: "unknown-copy-panic", copyPanic: 2, cleanupPanic: true, wantRelease: []int{2, 1}},
		{name: "success"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var sourceData [64]byte
			var sourceScales [8]byte
			var destinationData [64]byte
			var destinationScales [8]byte
			source := &rocmBuf{ptr: unsafe.Pointer(&sourceData[0]), scales: unsafe.Pointer(&sourceScales[0]), n: 64, scalesN: 8, class: MemoryWeights, owned: true}
			original := *source
			var allocated []*rocmBuf
			allocCalls, copyCalls := 0, 0
			var released []int
			allocate := func(n int, class MemoryClass, site string) *rocmBuf {
				allocCalls++
				if (allocCalls == 1 && (n != 64 || site != "clone")) || (allocCalls == 2 && (n != 8 || site != "clone-scales")) || class != MemoryWeights {
					t.Fatalf("unexpected allocation %d: %d %s %s", allocCalls, n, class, site)
				}
				if allocCalls == tc.allocPanic {
					panic(allocationFailure)
				}
				ptr := unsafe.Pointer(&destinationData[0])
				if allocCalls == 2 {
					ptr = unsafe.Pointer(&destinationScales[0])
				}
				b := &rocmBuf{ptr: ptr, n: n, class: class, owned: true}
				allocated = append(allocated, b)
				return b
			}
			copyDevice := func(dst, src unsafe.Pointer, n int, site string) error {
				copyCalls++
				if copyCalls == 1 {
					if dst != unsafe.Pointer(&destinationData[0]) || src != source.ptr || n != 64 || site != "clone" {
						t.Fatal("data copy contract changed")
					}
				} else if dst != unsafe.Pointer(&destinationScales[0]) || src != source.scales || n != 8 || site != "clone scales" {
					t.Fatal("scale copy contract changed")
				}
				if copyCalls == tc.copyError {
					return copyFailure
				}
				if copyCalls == tc.copyPanic {
					panic(unknownFailure)
				}
				return nil
			}
			release := func(b *rocmBuf) {
				index := 0
				for i, owned := range allocated {
					if b == owned {
						index = i + 1
					}
				}
				if index == 0 || b == source {
					t.Fatal("attempted to release unowned/source storage")
				}
				if b.scales != nil {
					t.Fatal("unpublished auxiliary storage was prematurely bundled")
				}
				released = append(released, index)
				if tc.cleanupPanic {
					panic(cleanupFailure)
				}
				b.ptr = nil
			}
			var got *rocmBuf
			var err error
			var panicked any
			func() {
				defer func() { panicked = recover() }()
				got, err = cloneROCmStorage(source, allocate, copyDevice, release)
			}()
			if !reflect.DeepEqual(released, tc.wantRelease) {
				t.Fatalf("release attempts=%v want=%v", released, tc.wantRelease)
			}
			if source.ptr != original.ptr || source.scales != original.scales || source.n != original.n || source.scalesN != original.scalesN || source.freed != 0 {
				t.Fatal("source ownership changed")
			}
			switch {
			case tc.allocPanic != 0:
				if panicked != allocationFailure || got != nil || err != nil {
					t.Fatalf("allocation cause replaced: panic=%v result=%v err=%v", panicked, got, err)
				}
			case tc.copyPanic != 0:
				if panicked != unknownFailure || got != nil || err != nil {
					t.Fatalf("unknown panic identity lost: %v", panicked)
				}
			case tc.copyError != 0:
				if panicked != nil || err != copyFailure || got != nil {
					t.Fatalf("copy error replaced: panic=%v result=%v err=%v", panicked, got, err)
				}
			default:
				if panicked != nil || err != nil || got != allocated[0] || got.scales != allocated[1].ptr || got.scalesN != 8 || got.ptr == source.ptr || got.scales == source.scales || allocCalls != 2 || copyCalls != 2 {
					t.Fatalf("successful clone ownership transfer failed: panic=%v result=%v err=%v", panicked, got, err)
				}
			}
		})
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestROCmCloneStorageWithoutAuxiliaryBufferTransfersOnce(t *testing.T) {
	var sourceByte, destByte byte
	source := &rocmBuf{ptr: unsafe.Pointer(&sourceByte), n: 1, class: MemoryActivation, owned: true}
	dest := &rocmBuf{ptr: unsafe.Pointer(&destByte), n: 1, class: MemoryActivation, owned: true}
	allocations, copies, releases := 0, 0, 0
	got, err := cloneROCmStorage(source, func(n int, class MemoryClass, site string) *rocmBuf {
		allocations++
		if n != 1 || class != MemoryActivation || site != "clone" {
			t.Fatal("F32 allocation contract changed")
		}
		return dest
	}, func(dst, src unsafe.Pointer, n int, site string) error {
		copies++
		if dst != dest.ptr || src != source.ptr || n != 1 || site != "clone" {
			t.Fatal("F32 copy contract changed")
		}
		return nil
	}, func(*rocmBuf) { releases++ })
	if err != nil || got != dest || allocations != 1 || copies != 1 || releases != 0 || got.scales != nil || got.scalesN != 0 {
		t.Fatalf("unexpected F32 clone result=%v err=%v operations=%d/%d/%d", got, err, allocations, copies, releases)
	}
}

// fak-test:runtime fast est=5ms lane=default
// Estimate unmeasured. Operations are injected Go ownership controls, not HIP.
func TestROCmQ8UploadStorageUnwindsEveryUnpublishedOwner(t *testing.T) {
	allocationFailure := &DeviceAllocError{Bytes: 4, Site: "upload-q8-scales", Class: MemoryWeights}
	uploadFailure := errors.New("injected native H2D error")
	unknownFailure := &struct{ name string }{"unknown upload panic"}
	cleanupFailure := &struct{ name string }{"secondary release panic"}
	for _, tc := range []struct {
		name                    string
		allocPanic, uploadPanic int
		unknown, cleanupPanic   bool
		wantRelease             []int
	}{
		{name: "first-allocation", allocPanic: 1},
		{name: "scale-allocation", allocPanic: 2, wantRelease: []int{1}},
		{name: "scale-allocation-cleanup-panic", allocPanic: 2, cleanupPanic: true, wantRelease: []int{1}},
		{name: "codes-upload", uploadPanic: 1, wantRelease: []int{2, 1}},
		{name: "scales-upload", uploadPanic: 2, wantRelease: []int{2, 1}},
		{name: "codes-upload-both-cleanups-panic", uploadPanic: 1, cleanupPanic: true, wantRelease: []int{2, 1}},
		{name: "unknown-scales-upload-both-cleanups-panic", uploadPanic: 2, unknown: true, cleanupPanic: true, wantRelease: []int{2, 1}},
		{name: "success"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			codes := make([]int8, 32)
			for i := range codes {
				codes[i] = int8(i%7 - 3)
			}
			scales := []float32{.25}
			originalCodes := append([]int8(nil), codes...)
			originalScales := append([]float32(nil), scales...)
			var codeStorage [32]byte
			var scaleStorage [4]byte
			var allocated []*rocmBuf
			var released []int
			allocCalls, uploadCalls := 0, 0
			allocate := func(n int, class MemoryClass, site string) *rocmBuf {
				allocCalls++
				if class != MemoryWeights || (allocCalls == 1 && (n != 32 || site != "upload-q8")) || (allocCalls == 2 && (n != 4 || site != "upload-q8-scales")) {
					t.Fatal("allocation metadata changed")
				}
				if allocCalls == tc.allocPanic {
					panic(allocationFailure)
				}
				ptr := unsafe.Pointer(&codeStorage[0])
				if allocCalls == 2 {
					ptr = unsafe.Pointer(&scaleStorage[0])
				}
				b := &rocmBuf{ptr: ptr, n: n, class: class, owned: true}
				allocated = append(allocated, b)
				return b
			}
			upload := func(dst, src unsafe.Pointer, n int, site string) {
				uploadCalls++
				if uploadCalls == 1 {
					if dst != unsafe.Pointer(&codeStorage[0]) || src != unsafe.Pointer(&codes[0]) || n != 32 || site != "upload q8 codes" {
						t.Fatal("codes transfer changed")
					}
				} else if dst != unsafe.Pointer(&scaleStorage[0]) || src != unsafe.Pointer(&scales[0]) || n != 4 || site != "upload q8 scales" {
					t.Fatal("scale transfer changed")
				}
				if uploadCalls == tc.uploadPanic {
					if tc.unknown {
						panic(unknownFailure)
					}
					panic(uploadFailure)
				}
			}
			release := func(b *rocmBuf) {
				index := 0
				for i, owned := range allocated {
					if owned == b {
						index = i + 1
					}
				}
				if index == 0 || b.scales != nil {
					t.Fatal("release lost separate temporary ownership")
				}
				released = append(released, index)
				if tc.cleanupPanic {
					panic(cleanupFailure)
				}
				b.ptr = nil
			}
			var got *rocmBuf
			var panicked any
			func() {
				defer func() { panicked = recover() }()
				got = uploadROCmQ8Storage(codes, scales, 4, allocate, upload, release)
			}()
			if !reflect.DeepEqual(released, tc.wantRelease) {
				t.Fatalf("release attempts=%v want=%v", released, tc.wantRelease)
			}
			if !reflect.DeepEqual(codes, originalCodes) || !reflect.DeepEqual(scales, originalScales) {
				t.Fatal("host source changed")
			}
			var wantPanic any
			if tc.allocPanic != 0 {
				wantPanic = allocationFailure
			} else if tc.uploadPanic != 0 {
				wantPanic = uploadFailure
				if tc.unknown {
					wantPanic = unknownFailure
				}
			}
			if panicked != wantPanic {
				t.Fatalf("panic=%v want identical %v", panicked, wantPanic)
			}
			if wantPanic != nil {
				if got != nil {
					t.Fatal("failed upload published storage")
				}
			} else if got != allocated[0] || got.ptr != unsafe.Pointer(&codeStorage[0]) || got.scales != allocated[1].ptr || got.scalesN != 4 || got.n != 32 || !got.owned || allocCalls != 2 || uploadCalls != 2 {
				t.Fatal("successful combined ownership transfer changed")
			}
		})
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestROCmQ8UploadStoragePreservesZeroElementBehavior(t *testing.T) {
	var storage [2]byte
	allocations, uploads, releases := 0, 0, 0
	got := uploadROCmQ8Storage(nil, nil, 0, func(n int, class MemoryClass, site string) *rocmBuf {
		if n != 0 || class != MemoryWeights {
			t.Fatal("zero-size allocation request changed")
		}
		if (allocations == 0 && site != "upload-q8") || (allocations == 1 && site != "upload-q8-scales") {
			t.Fatal("allocation site changed")
		}
		b := &rocmBuf{ptr: unsafe.Pointer(&storage[allocations]), n: 1, class: class, owned: true}
		allocations++
		return b
	}, func(unsafe.Pointer, unsafe.Pointer, int, string) { uploads++ }, func(*rocmBuf) { releases++ })
	if allocations != 2 || uploads != 0 || releases != 0 || got.ptr != unsafe.Pointer(&storage[0]) || got.scales != unsafe.Pointer(&storage[1]) || got.scalesN != 0 {
		t.Fatal("zero-size storage ownership changed")
	}
}

// fak-test:runtime fast est=5ms lane=default
// Unmeasured estimate. The transfer/release operations below are pure Go fault
// controls; no native allocation, HIP transfer or device execution is attempted.
func TestROCmSingleUploadFailureRetainsPrimaryAndReleasesOwner(t *testing.T) {
	primaryError := errors.New("injected ROCm H2D failure")
	primaryUnknown := &struct{ name string }{"unknown upload failure"}
	secondary := errors.New("injected native Free failure")
	for _, dt := range []Dtype{F32, Q4_K, Q5_K, Q6_K} {
		bytes := map[Dtype]int{F32: 16, Q4_K: 144, Q5_K: 176, Q6_K: 210}[dt]
		for _, cleanupPanic := range []bool{false, true} {
			for _, primary := range []any{primaryError, primaryUnknown} {
				storage := make([]byte, bytes)
				owned := &rocmBuf{ptr: unsafe.Pointer(&storage[0]), n: bytes, class: MemoryWeights, owned: true}
				host := make([]byte, bytes)
				host[0], host[len(host)-1] = 17, 29
				original := append([]byte(nil), host...)
				transfers, releases := 0, 0
				published := false
				var got any
				func() {
					defer func() { got = recover() }()
					finishROCmUpload(owned, func() {
						transfers++
						if owned.n != len(host) || owned.class != MemoryWeights || owned.scales != nil {
							t.Fatal("upload owner metadata changed")
						}
						panic(primary)
					}, func(b *rocmBuf) {
						releases++
						if b != owned || b.ptr == unsafe.Pointer(&host[0]) {
							t.Fatal("released wrong owner or host source")
						}
						if cleanupPanic {
							panic(secondary)
						}
						b.ptr = nil
					})
					published = true
				}()
				if got != primary || transfers != 1 || releases != 1 || published || !reflect.DeepEqual(host, original) {
					t.Fatalf("%s cleanupPanic=%v: cause=%v transfer/release=%d/%d published=%v", dt, cleanupPanic, got, transfers, releases, published)
				}
			}
		}
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestROCmSingleUploadSuccessAndEmptyTransferPreserveOwnership(t *testing.T) {
	for _, n := range []int{0, 16, 144, 176, 210} {
		storage := make([]byte, max(n, 1))
		owned := &rocmBuf{ptr: unsafe.Pointer(&storage[0]), n: len(storage), class: MemoryWeights, owned: true}
		before := owned.ptr
		transfers, releases := 0, 0
		finishROCmUpload(owned, func() {
			// This is the same existing len(f)>0 / want>0 guard retained in Upload.
			if n > 0 {
				transfers++
			}
		}, func(*rocmBuf) { releases++ })
		wantTransfers := 0
		if n > 0 {
			wantTransfers = 1
		}
		if transfers != wantTransfers || releases != 0 || owned.ptr != before || !owned.owned || owned.class != MemoryWeights || owned.n != max(n, 1) {
			t.Fatalf("bytes=%d transfers=%d releases=%d owner=%+v", n, transfers, releases, owned)
		}
	}
}

// fak-test:runtime fast est=10ms lane=default
// Static production-coupling control; this is not a native Upload execution test.
func TestROCmSingleUploadGuardOwnsBothProductionTransferBlocks(t *testing.T) {
	f, err := parser.ParseFile(token.NewFileSet(), "rocm_backend.go", nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	var upload *ast.FuncDecl
	for _, decl := range f.Decls {
		if fn, ok := decl.(*ast.FuncDecl); ok && fn.Name.Name == "Upload" {
			upload = fn
		}
	}
	if upload == nil {
		t.Fatal("Upload owner missing")
	}
	checked := 0
	ast.Inspect(upload.Body, func(n ast.Node) bool {
		clause, ok := n.(*ast.CaseClause)
		if !ok {
			return true
		}
		var guarded bool
		for _, expr := range clause.List {
			if id, ok := expr.(*ast.Ident); ok && (id.Name == "F32" || id.Name == "Q4_K" || id.Name == "Q5_K" || id.Name == "Q6_K") {
				guarded = true
			}
		}
		if !guarded {
			return true
		}
		calls, nativeCalls := 0, 0
		ast.Inspect(clause, func(child ast.Node) bool {
			call, ok := child.(*ast.CallExpr)
			if !ok {
				return true
			}
			id, ok := call.Fun.(*ast.Ident)
			if !ok || id.Name != "finishROCmUpload" {
				return true
			}
			calls++
			if len(call.Args) != 3 {
				t.Fatal("upload guard operation contract changed")
			}
			transfer, ok := call.Args[1].(*ast.FuncLit)
			if !ok {
				t.Fatal("expected original inline transfer block")
			}
			ast.Inspect(transfer.Body, func(inner ast.Node) bool {
				if c, ok := inner.(*ast.CallExpr); ok {
					if sel, ok := c.Fun.(*ast.SelectorExpr); ok && sel.Sel.Name == "frocm_h2d" {
						nativeCalls++
					}
				}
				return true
			})
			return true
		})
		if calls != 1 || nativeCalls != 1 {
			t.Fatalf("production upload arm guards=%d native H2D blocks=%d", calls, nativeCalls)
		}
		checked++
		return false
	})
	if checked != 2 {
		t.Fatalf("guarded dtype arms=%d want F32 plus grouped Q4_K/Q5_K/Q6_K", checked)
	}
}

// fak-test:runtime fast est=10ms lane=default
// Estimate unmeasured. Only injected Go allocation/copy/release operations run.
func TestROCmKVCloneUnwindsEveryUnpublishedRow(t *testing.T) {
	allocFailure := &DeviceAllocError{Bytes: 16, Site: "clone KV", Class: MemoryKVCache}
	copyFailure := errors.New("injected KV D2D error")
	unknown := &struct{ name string }{"unknown KV copy panic"}
	secondary := errors.New("secondary KV cleanup error")
	for _, operation := range []string{"allocate", "copy", "unknown-copy"} {
		for _, at := range []int{1, 3, 6} {
			for _, cleanupFails := range []bool{false, true} {
				var sourceBytes [6][32]byte
				var destinationBytes [6][16]byte
				source := &rocmKV{cfg: KVConfig{NumLayers: 2, NumKVHeads: 1, HeadDim: 2, RopeTheta: 10000, WindowPerLayer: []int{4, 0}}, K: make([]rocmRows, 2), Kraw: make([]rocmRows, 2), V: make([]rocmRows, 2), pos: []int{3, 4}}
				var sourceRows []rocmRows
				for l := 0; l < 2; l++ {
					for j, row := range []*rocmRows{&source.K[l], &source.Kraw[l], &source.V[l]} {
						i := l*3 + j
						sourceBytes[i][0] = byte(i + 1)
						*row = rocmRows{ptr: unsafe.Pointer(&sourceBytes[i][0]), rows: 2, cap: 4, generation: uint64(10 + i)}
						sourceRows = append(sourceRows, *row)
					}
				}
				originalBytes := sourceBytes
				allocations, copies := 0, 0
				var acquired []unsafe.Pointer
				var released []unsafe.Pointer
				var result *rocmKV
				var panicked any
				func() {
					defer func() { panicked = recover() }()
					result = source.cloneWithOperations(func(n int, class MemoryClass, site string) *rocmBuf {
						allocations++
						if n != 16 || class != MemoryKVCache {
							t.Fatal("clone allocation contract changed")
						}
						if operation == "allocate" && allocations == at {
							panic(allocFailure)
						}
						ptr := unsafe.Pointer(&destinationBytes[allocations-1][0])
						acquired = append(acquired, ptr)
						return &rocmBuf{ptr: ptr, n: n, class: class, owned: true}
					}, func(dst, src unsafe.Pointer, n int, site string) {
						copies++
						if dst != acquired[len(acquired)-1] || src != sourceRows[copies-1].ptr || n != 16 {
							t.Fatal("clone copy source/order changed")
						}
						if operation != "allocate" && copies == at {
							if operation == "unknown-copy" {
								panic(unknown)
							}
							panic(copyFailure)
						}
						copy((*[16]byte)(dst)[:], (*[16]byte)(src)[:])
					}, func(ptr unsafe.Pointer) {
						for _, row := range sourceRows {
							if ptr == row.ptr {
								t.Fatal("released source row")
							}
						}
						released = append(released, ptr)
						if cleanupFails {
							panic(secondary)
						}
					})
				}()
				var want any = copyFailure
				if operation == "allocate" {
					want = allocFailure
				}
				if operation == "unknown-copy" {
					want = unknown
				}
				if panicked != want || result != nil {
					t.Fatalf("%s at%d cleanup=%v: lost primary/publication: %v", operation, at, cleanupFails, panicked)
				}
				if len(released) != len(acquired) {
					t.Fatalf("release attempts=%d acquired=%d", len(released), len(acquired))
				}
				for i, ptr := range released {
					if ptr != acquired[len(acquired)-1-i] {
						t.Fatal("cleanup skipped or repeated destination")
					}
				}
				for l := 0; l < 2; l++ {
					for j, row := range []rocmRows{source.K[l], source.Kraw[l], source.V[l]} {
						if row != sourceRows[l*3+j] {
							t.Fatal("source metadata/generation changed")
						}
					}
				}
				if sourceBytes != originalBytes || !reflect.DeepEqual(source.pos, []int{3, 4}) || !reflect.DeepEqual(source.cfg.WindowPerLayer, []int{4, 0}) {
					t.Fatal("source bytes/positions/config changed")
				}
			}
		}
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestROCmKVCloneSuccessPreservesSourceAndSkipsEmptyRows(t *testing.T) {
	var src [3][16]byte
	var dst [3][16]byte
	source := &rocmKV{cfg: KVConfig{NumLayers: 2, NumKVHeads: 1, HeadDim: 2, RopeTheta: 10000, WindowPerLayer: []int{8, 0}}, K: make([]rocmRows, 2), Kraw: make([]rocmRows, 2), V: make([]rocmRows, 2), pos: []int{0}}
	for i, row := range []*rocmRows{&source.K[0], &source.Kraw[0], &source.V[0]} {
		src[i][0] = byte(i + 3)
		*row = rocmRows{ptr: unsafe.Pointer(&src[i][0]), rows: 1, cap: 2, generation: 7}
	}
	allocations, copies, releases := 0, 0, 0
	result := source.cloneWithOperations(func(n int, class MemoryClass, site string) *rocmBuf {
		if n != 8 || class != MemoryKVCache {
			t.Fatal("clone capacity should equal written rows")
		}
		ptr := unsafe.Pointer(&dst[allocations][0])
		allocations++
		return &rocmBuf{ptr: ptr, n: n, class: class, owned: true}
	}, func(to, from unsafe.Pointer, n int, site string) {
		copies++
		copy((*[8]byte)(to)[:], (*[8]byte)(from)[:])
	}, func(unsafe.Pointer) { releases++ })
	if allocations != 3 || copies != 3 || releases != 0 || result == source || !reflect.DeepEqual(result.cfg, source.cfg) || !reflect.DeepEqual(result.pos, source.pos) {
		t.Fatal("success/empty-layer clone contract changed")
	}
	for i, row := range []rocmRows{result.K[0], result.Kraw[0], result.V[0]} {
		if row.ptr != unsafe.Pointer(&dst[i][0]) || row.ptr == unsafe.Pointer(&src[i][0]) || row.rows != 1 || row.cap != 1 || row.generation != 0 || dst[i][0] != src[i][0] {
			t.Fatal("clone row ownership/capacity/generation changed")
		}
	}
	if result.K[1] != (rocmRows{}) || result.Kraw[1] != (rocmRows{}) || result.V[1] != (rocmRows{}) {
		t.Fatal("empty rows allocated")
	}
	result.pos[0] = 9
	result.cfg.WindowPerLayer[0] = 99
	if source.pos[0] != 0 || source.cfg.WindowPerLayer[0] != 8 || source.K[0].generation != 7 || source.K[0].cap != 2 {
		t.Fatal("clone metadata aliases source")
	}
}

// fak-test:runtime fast est=10ms lane=default
// Static production coupling; no native Clone operation is executed.
func TestROCmKVCloneCallsOwnershipLeafUnderExistingLock(t *testing.T) {
	f, err := parser.ParseFile(token.NewFileSet(), "rocm_kv.go", nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	calls, locks, unlocks := 0, 0, 0
	for _, decl := range f.Decls {
		fn, ok := decl.(*ast.FuncDecl)
		if !ok || fn.Name.Name != "Clone" {
			continue
		}
		ast.Inspect(fn.Body, func(n ast.Node) bool {
			if call, ok := n.(*ast.CallExpr); ok {
				if sel, ok := call.Fun.(*ast.SelectorExpr); ok {
					switch sel.Sel.Name {
					case "cloneWithOperations":
						calls++
					case "Lock":
						locks++
					case "Unlock":
						unlocks++
					case "Free":
						t.Error("Clone must not reacquire rocmMu through public Free")
					}
				}
			}
			return true
		})
	}
	if calls != 1 || locks != 1 || unlocks != 1 {
		t.Fatalf("clone leaf/lock/unlock=%d/%d/%d", calls, locks, unlocks)
	}
}

// fak-test:runtime fast est=5ms lane=default
// Unmeasured estimate. Injected prefix-copy ownership controls, no HIP execution.
func TestROCmKVGrowthCopyFailureKeepsOldRowAndReleasesNewBuffer(t *testing.T) {
	copyFailure := errors.New("injected native KV prefix-copy failure")
	unknown := &struct{ name string }{"unknown prefix-copy panic"}
	cleanupFailure := errors.New("secondary new-buffer Free failure")
	for _, primary := range []any{copyFailure, unknown} {
		for _, cleanupFails := range []bool{false, true} {
			var oldData [16]byte
			oldData[0] = 37
			var newData [32]byte
			row := rocmRows{ptr: unsafe.Pointer(&oldData[0]), rows: 2, cap: 2, generation: 11}
			original := row
			originalData := oldData
			owned := &rocmBuf{ptr: unsafe.Pointer(&newData[0]), n: 32, class: MemoryKVCache, owned: true}
			copies, releases, retirements := 0, 0, 0
			var panicked any
			func() {
				defer func() { panicked = recover() }()
				finishROCmKVGrowthCopy(owned, func() {
					copies++
					copy(newData[:16], oldData[:])
					panic(primary)
				}, func(b *rocmBuf) {
					releases++
					if b != owned || b.ptr == row.ptr {
						t.Fatal("released source or wrong allocation")
					}
					if cleanupFails {
						panic(cleanupFailure)
					}
					b.ptr = nil
				})
				// The real grow reaches old retirement only after the guard returns.
				retirements++
			}()
			if panicked != primary || copies != 1 || releases != 1 || retirements != 0 || row != original || oldData != originalData {
				t.Fatalf("copy failure contract: cause=%v copies/releases/retirements=%d/%d/%d row=%+v", panicked, copies, releases, retirements, row)
			}
		}
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestROCmKVGrowthCopySuccessRetainsDestinationForOriginalCommit(t *testing.T) {
	var oldData [8]byte
	oldData[0] = 51
	var newData [16]byte
	row := rocmRows{ptr: unsafe.Pointer(&oldData[0]), rows: 1, cap: 1, generation: 4}
	before := row
	owned := &rocmBuf{ptr: unsafe.Pointer(&newData[0]), n: 16, class: MemoryKVCache, owned: true}
	releases := 0
	finishROCmKVGrowthCopy(owned, func() { copy(newData[:8], oldData[:]) }, func(*rocmBuf) { releases++ })
	if releases != 0 || newData[0] != 51 || row != before || owned.ptr != unsafe.Pointer(&newData[0]) || !owned.owned {
		t.Fatal("successful prefix copy changed source or released destination")
	}
}

// fak-test:runtime fast est=10ms lane=default
// Static coupling and order control, not an executed grow/native failure witness.
func TestROCmKVGrowthCopyGuardPrecedesLogicalRetirement(t *testing.T) {
	f, err := parser.ParseFile(token.NewFileSet(), "rocm_kv.go", nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	var copyGuard, oldFree, retirement token.Pos
	guardCalls, copies := 0, 0
	isField := func(expr ast.Expr, owner, field string) bool {
		selector, ok := expr.(*ast.SelectorExpr)
		if !ok || selector.Sel.Name != field {
			return false
		}
		id, ok := selector.X.(*ast.Ident)
		return ok && id.Name == owner
	}
	for _, decl := range f.Decls {
		fn, ok := decl.(*ast.FuncDecl)
		if !ok || fn.Name.Name != "grow" {
			continue
		}
		ast.Inspect(fn.Body, func(n ast.Node) bool {
			switch node := n.(type) {
			case *ast.CallExpr:
				if sel, ok := node.Fun.(*ast.SelectorExpr); ok && sel.Sel.Name == "finishGrowthRetirement" {
					retirement = node.Pos()
				}
				if id, ok := node.Fun.(*ast.Ident); ok && id.Name == "finishROCmKVGrowthCopy" {
					copyGuard = node.Pos()
					guardCalls++
					if len(node.Args) != 3 {
						t.Fatal("growth guard arguments changed")
					}
					owned, ok := node.Args[0].(*ast.Ident)
					if !ok || owned.Name != "nb" {
						t.Fatal("growth guard must own only the new buffer")
					}
				}
				if sel, ok := node.Fun.(*ast.SelectorExpr); ok && sel.Sel.Name == "frocm_free" {
					oldFree = node.Pos()
				}
				if sel, ok := node.Fun.(*ast.SelectorExpr); ok && sel.Sel.Name == "frocm_d2d" {
					copies++
					if len(node.Args) != 3 || !isField(node.Args[0], "nb", "ptr") || !isField(node.Args[1], "row", "ptr") {
						t.Fatal("growth copy source/destination changed")
					}
					size, ok := node.Args[2].(*ast.CallExpr)
					if !ok || len(size.Args) != 1 {
						t.Fatal("growth copy byte count changed")
					}
					used, ok := size.Args[0].(*ast.Ident)
					if !ok || used.Name != "used" {
						t.Fatal("growth must copy only the used prefix")
					}
				}
			}
			return true
		})
	}
	if guardCalls != 1 || copies != 1 || copyGuard == 0 || retirement <= copyGuard || oldFree <= retirement {
		t.Fatalf("guard/retirement-leaf/native-callback order=%d/%d/%d calls=%d", copyGuard, retirement, oldFree, guardCalls)
	}
}

func rocmKVRetirementFixture() (*rocmKV, [6]unsafe.Pointer) {
	be := &rocmBackend{}
	k := &rocmKV{be: be, cfg: KVConfig{NumLayers: 2, NumKVHeads: 1, HeadDim: 2, RopeTheta: 10000}, K: make([]rocmRows, 2), Kraw: make([]rocmRows, 2), V: make([]rocmRows, 2), pos: []int{0, 1}}
	var pointers [6]unsafe.Pointer
	for i, row := range []*rocmRows{&k.K[0], &k.Kraw[0], &k.V[0], &k.K[1], &k.Kraw[1], &k.V[1]} {
		data := new([16]byte)
		data[0] = byte(i + 1)
		pointers[i] = unsafe.Pointer(&data[0])
		*row = rocmRows{ptr: pointers[i], rows: 2, cap: 2, generation: 7}
	}
	return k, pointers
}

func rocmKVRetirementPanic(fn func()) (cause any) {
	defer func() { cause = recover() }()
	fn()
	return nil
}

// fak-test:runtime fast est=10ms lane=default
// Synthetic owner operations only; no HIP allocation, free, copy or dispatch.
func TestROCmKVOldRetirementFailurePoisonsAllConsumers(t *testing.T) {
	primaryError := errors.New("ambiguous old native Free")
	unknown := &struct{ label string }{"unknown old Free panic"}
	secondary := errors.New("secondary cleanup failure")
	for _, primary := range []any{primaryError, unknown} {
		for _, cleanupFails := range []bool{false, true} {
			k, pointers := rocmKVRetirementFixture()
			oldView, otherView := k.KeysView(0), k.ValuesView(1)
			var replacement [32]byte
			owned := &rocmBuf{ptr: unsafe.Pointer(&replacement[0]), n: 32, owned: true, class: MemoryKVCache}
			oldCalls, newCalls := 0, 0
			got := rocmKVRetirementPanic(func() {
				k.finishGrowthRetirement(&k.K[0], owned, 4, func(ptr unsafe.Pointer) {
					oldCalls++
					if ptr != pointers[0] || k.K[0].ptr != nil || oldView.buf.(*rocmBuf).Ready() {
						t.Fatal("old handle not detached before retirement")
					}
					panic(primary)
				}, func(b *rocmBuf) {
					newCalls++
					if b != owned || rocmKVRetirementPanic(k.ensureUsable) != primary || otherView.buf.(*rocmBuf).Ready() {
						t.Fatal("primary poison was not published before cleanup")
					}
					if cleanupFails {
						panic(secondary)
					}
					b.ptr = nil
				})
			})
			if got != primary || oldCalls != 1 || newCalls != 1 || k.K[0].ptr != nil || k.K[0].generation != 8 {
				t.Fatalf("wrong failure ownership: cause=%v old/new=%d/%d", got, oldCalls, newCalls)
			}
			// Metadata remains readable; these are logical bytes, not evidence
			// that the ambiguous native allocation was physically reclaimed.
			if k.Len() != 2 || !reflect.DeepEqual(k.Pos(), []int{0, 1}) || k.ResidentBytes() != 96 || k.KVConfig().NumLayers != 2 {
				t.Fatal("failure changed diagnostic metadata")
			}
			var query [2]float32
			q := makeTensor(k.be, F32, RowMajor, []int{2}, nil, &rocmBuf{ptr: unsafe.Pointer(&query[0]), n: 8})
			for name, fn := range map[string]func(){
				"append":         func() { k.AppendKV(0, Tensor{}, Tensor{}, Tensor{}, 2) },
				"grow-fast-path": func() { k.grow(&k.K[0], 0, "poisoned") },
				"evict":          func() { k.Evict(0, 1) },
				"clone":          func() { k.Clone() },
				"keys":           func() { k.KeysView(0) },
				"values":         func() { k.ValuesView(1) },
				"attention":      func() { k.be.Attention(q, k, 0, true, 1, 1) },
			} {
				if cause := rocmKVRetirementPanic(fn); cause != primary {
					t.Fatalf("%s did not retain primary: %v", name, cause)
				}
			}
			for _, view := range []Tensor{oldView, otherView} {
				if view.buf.(*rocmBuf).Ready() || rocmKVRetirementPanic(func() { k.be.Read(view) }) == nil {
					t.Fatal("borrowed view remained usable")
				}
			}
			calls := make(map[unsafe.Pointer]int)
			if cause := rocmKVRetirementPanic(func() { k.freeWithOperation(func(ptr unsafe.Pointer) { calls[ptr]++; panic(secondary) }) }); cause != nil {
				t.Fatalf("poison cleanup replaced primary: %v", cause)
			}
			if calls[pointers[0]] != 0 || len(calls) != 5 {
				t.Fatalf("ambiguous old handle retried or survivors skipped: %v", calls)
			}
			for _, ptr := range pointers[1:] {
				if calls[ptr] != 1 {
					t.Fatal("survivor not attempted exactly once")
				}
			}
			k.freeWithOperation(func(unsafe.Pointer) { t.Fatal("cleanup retried a detached handle") })
			k.Free() // no handles remain: the real public method must not call HIP.
			if k.Len() != 0 || k.ResidentBytes() != 0 || rocmKVRetirementPanic(k.ensureUsable) != primary {
				t.Fatal("cleanup reset poison or logical metadata")
			}
		}
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestROCmKVFreeReportsFirstFailureAfterEveryAttempt(t *testing.T) {
	for _, first := range []any{errors.New("first Free failure"), &struct{ label string }{"unknown Free panic"}} {
		k, pointers := rocmKVRetirementFixture()
		views := []Tensor{k.KeysView(0), k.ValuesView(1)}
		calls := make(map[unsafe.Pointer]int)
		cause := rocmKVRetirementPanic(func() {
			k.freeWithOperation(func(ptr unsafe.Pointer) {
				calls[ptr]++
				for _, rows := range [][]rocmRows{k.K, k.Kraw, k.V} {
					for _, row := range rows {
						if row.ptr == ptr {
							t.Fatal("Free attempted before detachment")
						}
					}
				}
				if ptr == pointers[0] {
					panic(first)
				}
				panic(errors.New("later Free failure"))
			})
		})
		if cause != first || len(calls) != 6 || rocmKVRetirementPanic(k.ensureUsable) != first {
			t.Fatalf("first failure lost or cleanup incomplete: %v calls=%d", cause, len(calls))
		}
		for _, n := range calls {
			if n != 1 {
				t.Fatal("duplicate retirement")
			}
		}
		for _, v := range views {
			if v.buf.(*rocmBuf).Ready() {
				t.Fatal("Free retained borrowed view")
			}
		}
		k.freeWithOperation(func(unsafe.Pointer) { t.Fatal("repeated Free retried ambiguous pointer") })
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestROCmKVRetirementSuccessKeepsReusableEmptyStore(t *testing.T) {
	k, pointers := rocmKVRetirementFixture()
	oldView, otherView := k.KeysView(0), k.ValuesView(1)
	var replacement [32]byte
	owned := &rocmBuf{ptr: unsafe.Pointer(&replacement[0]), n: 32, owned: true}
	retired := 0
	k.finishGrowthRetirement(&k.K[0], owned, 4, func(ptr unsafe.Pointer) {
		if ptr != pointers[0] || oldView.buf.(*rocmBuf).Ready() {
			t.Fatal("successful retirement order changed")
		}
		retired++
	}, func(*rocmBuf) { t.Fatal("successful growth freed replacement") })
	if retired != 1 || k.K[0].ptr != owned.ptr || k.K[0].cap != 4 || k.K[0].rows != 2 || k.Len() != 2 || !otherView.buf.(*rocmBuf).Ready() {
		t.Fatal("successful growth metadata changed")
	}
	k.ensureUsable()
	calls := 0
	k.freeWithOperation(func(unsafe.Pointer) { calls++ })
	if calls != 6 || k.Len() != 0 || k.ResidentBytes() != 0 {
		t.Fatal("successful Free did not empty store")
	}
	k.ensureUsable()
	k.freeWithOperation(func(unsafe.Pointer) { t.Fatal("successful repeated Free retried handles") })
	// Empty store can publish a fresh row again without retiring any old pointer.
	var fresh [32]byte
	owned = &rocmBuf{ptr: unsafe.Pointer(&fresh[0]), n: 32, owned: true}
	k.finishGrowthRetirement(&k.K[0], owned, 4, func(unsafe.Pointer) { t.Fatal("empty store retired old pointer") }, func(*rocmBuf) { t.Fatal("empty-store growth failed") })
	if k.K[0].ptr != owned.ptr || !k.KeysView(0).buf.(*rocmBuf).Ready() {
		t.Fatal("empty store is no longer reusable")
	}
}

// fak-test:runtime fast est=10ms lane=default
// Structural guard/caller coupling only, not a native execution witness.
func TestROCmKVRetirementProductionGuardsAndFreeOwner(t *testing.T) {
	for file, owners := range map[string][]string{
		"rocm_kv.go":      {"grow", "AppendKV", "view", "KeysView", "ValuesView", "Evict", "cloneWithOperations"},
		"rocm_backend.go": {"Attention"},
	} {
		f, err := parser.ParseFile(token.NewFileSet(), file, nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		for _, name := range owners {
			found := false
			for _, d := range f.Decls {
				fn, ok := d.(*ast.FuncDecl)
				if !ok || fn.Name.Name != name {
					continue
				}
				var guard, work token.Pos
				ast.Inspect(fn.Body, func(n ast.Node) bool {
					call, ok := n.(*ast.CallExpr)
					if !ok {
						return true
					}
					sel, ok := call.Fun.(*ast.SelectorExpr)
					if !ok {
						return true
					}
					if sel.Sel.Name == "ensureUsable" && guard == 0 {
						guard = call.Pos()
					}
					if (sel.Sel.Name == "tensor" || sel.Sel.Name == "alloc" || sel.Sel.Name == "frocm_kv_evict_f32" || sel.Sel.Name == "frocm_kv_append_f32" || sel.Sel.Name == "frocm_attention_f32") && work == 0 {
						work = call.Pos()
					}
					return true
				})
				if guard == 0 || (work != 0 && guard >= work) {
					t.Fatalf("%s.%s missing guard before device work", file, name)
				}
				found = true
			}
			if !found {
				t.Fatalf("missing production consumer %s", name)
			}
		}
	}
	f, err := parser.ParseFile(token.NewFileSet(), "rocm_kv.go", nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	calls, native := 0, 0
	for _, d := range f.Decls {
		fn, ok := d.(*ast.FuncDecl)
		if !ok || fn.Name.Name != "Free" {
			continue
		}
		ast.Inspect(fn.Body, func(n ast.Node) bool {
			call, ok := n.(*ast.CallExpr)
			if !ok {
				return true
			}
			if sel, ok := call.Fun.(*ast.SelectorExpr); ok {
				if sel.Sel.Name == "freeWithOperation" {
					calls++
				}
				if sel.Sel.Name == "frocm_free" {
					native++
				}
			}
			return true
		})
	}
	if calls != 1 || native != 1 {
		t.Fatalf("public Free ownership leaf/native calls=%d/%d", calls, native)
	}
}
