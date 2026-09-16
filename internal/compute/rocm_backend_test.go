//go:build linux && rocm && cgo

package compute

import (
	"encoding/binary"
	"math"
	"math/rand"
	"os"
	"testing"
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
	kv := b.NewKV(KVConfig{NumLayers: 1, NumKVHeads: 1, HeadDim: 4, RopeTheta: 10000})
	kr := ref.NewKV(KVConfig{NumLayers: 1, NumKVHeads: 1, HeadDim: 4, RopeTheta: 10000})
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
	if len(b.Read(copy.ValuesView(0))) != 12 {
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
