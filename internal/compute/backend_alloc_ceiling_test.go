package compute

import (
	"sort"
	"testing"
)

// backend_alloc_ceiling_test.go — TICKET-07 of the hot-path copy sweep (Ref #13876).
//
// A per-call copy of a static weight is ONE allocation of the whole weight's bytes, so an
// allocation-COUNT ceiling (TestCPURefMatMulKQuantReadsWeightInPlace) can miss it: a lone
// `make([]byte, len(weight))` is a single alloc and slips under a count ceiling of three. The
// exercise that actually catches the class is a BYTE ceiling: one cpu-ref MatMul must allocate far
// fewer bytes than the weight it reads, for every dtype the backend admits. A future dtype is
// caught without a test edit because the admitted set is read from SupportsDeviceWeightDtype rather
// than a hand-written list.
//
// Device caveat: the Vulkan upload path is behind the `vulkan` build tag and needs a live device,
// so it is not covered by this default-build gate. It remains a named residual of TICKET-07.

// allocCeilingDtypeUniverse is every weight dtype the cpu-ref backend could admit: the explicit
// MatMul switch cases plus every registered native i-quant format. A new raw IQ format joins
// rawIQFormats and is covered automatically; a new switch case is a one-line addition here.
func allocCeilingDtypeUniverse() []Dtype {
	seen := map[Dtype]bool{}
	dts := []Dtype{F32, Q8_0, Q4_K, Q5_K, Q6_K, Q2_K, Q3_K, IQ2_XXS, Q2_0}
	for _, dt := range dts {
		seen[dt] = true
	}
	for dt := range rawIQFormats {
		if !seen[dt] {
			dts = append(dts, dt)
		}
	}
	sort.Slice(dts, func(i, j int) bool { return dts[i] < dts[j] })
	return dts
}

// allocCeilingWeights builds one host weight for every admitted cpu-ref MatMul dtype, extending the
// packed-only fixture with the two unquantized formats the switch also admits (F32, Q8_0).
func allocCeilingWeights(out, in int) map[Dtype]Tensor {
	ws := cpuRefPackedWeights(out, in)
	be := &cpuBackend{}
	w := make([]float32, out*in)
	for i := range w {
		w[i] = float32(i%5) - 2
	}
	ws[F32] = NewF32(be, []int{out, in}, append([]float32(nil), w...))
	ws[Q8_0] = QuantizeQ8(be, []int{out, in}, w, 32)
	return ws
}

// allocCeilingWeightBytes sizes a host weight in bytes: the floor a per-call whole-weight copy must
// allocate and the ceiling one MatMul must stay below.
func allocCeilingWeightBytes(w Tensor) int {
	hb, ok := w.buf.(HostBuffer)
	if !ok {
		return 0
	}
	if w.Dtype == F32 {
		return len(hb.F32()) * 4
	}
	n := len(hb.I8())
	if w.Quant != nil {
		n += len(w.Quant.Scale) * 4
	}
	return n
}

// TestAllocCeilingCPURefMatMulReadsWeightInPlace asserts one MatMul and one BatchedMatMul allocate
// fewer bytes than the weight they read, for every admitted dtype. A per-call weight copy — the
// pre-19eab47217 i8AsBytes shape — allocates at least the weight's bytes and turns this red.
// fak-test:runtime fast est=3s
func TestAllocCeilingCPURefMatMulReadsWeightInPlace(t *testing.T) {
	const out, in, p = 64, 512, 2
	be := &cpuBackend{}
	x := cpuRefBenchX(in)
	xb := cpuRefBenchX(p * in)
	ws := allocCeilingWeights(out, in)

	for _, dt := range allocCeilingDtypeUniverse() {
		if !be.SupportsDeviceWeightDtype(dt) {
			if _, ok := ws[dt]; ok {
				t.Fatalf("%s has a fixture but SupportsDeviceWeightDtype rejected it", dt)
			}
			continue
		}
		w, ok := ws[dt]
		if !ok {
			t.Fatalf("admitted dtype %s has no fixture: add it to the ceiling table", dt)
		}
		weightBytes := allocCeilingWeightBytes(w)
		if weightBytes <= 0 {
			t.Fatalf("%s: cannot size the weight (%d bytes)", dt, weightBytes)
		}
		mm := testing.Benchmark(func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				be.MatMul(w, x)
			}
		})
		if per := mm.AllocedBytesPerOp(); per >= int64(weightBytes) {
			t.Errorf("%s MatMul allocates %d B/op >= the %d-byte weight: a per-call weight copy is being made", dt, per, weightBytes)
		}
		bm := testing.Benchmark(func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				be.BatchedMatMul(w, xb, p)
			}
		})
		if per := bm.AllocedBytesPerOp(); per >= int64(weightBytes) {
			t.Errorf("%s BatchedMatMul allocates %d B/op >= the %d-byte weight: a per-call weight copy is being made", dt, per, weightBytes)
		}
	}
}

// TestAllocCeilingDetectsSeededWeightCopy proves the byte ceiling rejects the copy class it exists
// to catch: a per-call whole-weight byte copy (the pre-19eab47217 i8AsBytes shape) allocates at
// least the weight's bytes, above the ceiling.
// fak-test:runtime fast est=1s
func TestAllocCeilingDetectsSeededWeightCopy(t *testing.T) {
	const out, in = 64, 512
	w := allocCeilingWeights(out, in)[Q4_K]
	weightBytes := allocCeilingWeightBytes(w)
	raw := w.buf.(HostBuffer).I8()
	cp := testing.Benchmark(func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			_ = i8CopyForTest(raw)
		}
	})
	if per := cp.AllocedBytesPerOp(); per < int64(weightBytes) {
		t.Fatalf("seeded copy allocates %d B/op < weight %d bytes; the ceiling would not catch a copy", per, weightBytes)
	}
}
