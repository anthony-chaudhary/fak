//go:build darwin && arm64 && cgo

package metalgemm

import (
	"encoding/binary"
	"fmt"
	"math"
	"testing"
)

// q6kMulMvReference is an independent CPU transcription of the 210-byte Q6_K layout (ql[128],
// qh[64], signed int8 scales[16], f16 d @208) with float64 accumulation. It is the oracle for both
// q6k_gemv (legacy) and q6k_mul_mv, so neither Metal kernel grades itself.
func q6kMulMvReference(raw []byte, out, in int, x []float32) []float32 {
	nblk := in / 256
	y := make([]float32, out)
	for o := 0; o < out; o++ {
		var acc float64
		for b := 0; b < nblk; b++ {
			blk := raw[(o*nblk+b)*210:]
			d := float64(math.Float32frombits(q4kTestF16Bits(binary.LittleEndian.Uint16(blk[208:210]))))
			xb := x[b*256:]
			for half := 0; half < 2; half++ {
				ql := blk[64*half:]
				qh := blk[128+32*half:]
				sc := blk[192+8*half:]
				n := 128 * half
				for l := 0; l < 32; l++ {
					is := l / 16
					q1 := int(ql[l]&0x0f|((qh[l]>>0)&3)<<4) - 32
					q2 := int(ql[l+32]&0x0f|((qh[l]>>2)&3)<<4) - 32
					q3 := int(ql[l]>>4|((qh[l]>>4)&3)<<4) - 32
					q4 := int(ql[l+32]>>4|((qh[l]>>6)&3)<<4) - 32
					acc += d * float64(int8(sc[is+0])) * float64(q1) * float64(xb[n+l])
					acc += d * float64(int8(sc[is+2])) * float64(q2) * float64(xb[n+l+32])
					acc += d * float64(int8(sc[is+4])) * float64(q3) * float64(xb[n+l+64])
					acc += d * float64(int8(sc[is+6])) * float64(q4) * float64(xb[n+l+96])
				}
			}
		}
		y[o] = float32(acc)
	}
	return y
}

// mulMvShape is one parity shape. The lists below cover the tail guard (out not a multiple of the
// 4 rows per threadgroup), nblk below and not a multiple of the 4 Q4_K lane groups / 2 Q6_K block
// parities, and the real Qwen2.5-7B decode projections.
type mulMvShape struct{ out, in int }

func requireMulMvParity(t *testing.T, label string, want, got []float32) {
	t.Helper()
	cosine, maxRel := q4kTestCosineMaxRel(want, got)
	if cosine < 0.99999 || maxRel > 5e-3 {
		t.Fatalf("%s: cosine=%.9f maxRel=%g, want cosine >= 0.99999 and maxRel <= 5e-3", label, cosine, maxRel)
	}
	t.Logf("%s: cosine=%.9f maxRel=%g", label, cosine, maxRel)
}

// TestQ4KMulMvP1Parity pins q4k_mul_mv (and the legacy q4k_gemv control) against the independent
// CPU reference, and proves the executed identity of explicit and DEFAULT requests.
// fak-test:runtime medium est=6s lane=default
func TestQ4KMulMvP1Parity(t *testing.T) {
	if !Available() {
		t.Skip("Metal unavailable")
	}
	defer ResetQ4K()
	defer SetQ4KGEMVKernelDefault(SetQ4KGEMVKernelDefault(GEMVKernelMulMv))

	for i, s := range []mulMvShape{
		{130, 256},    // nblk=1: three of four lane groups idle; tail of 2 rows
		{7, 1280},     // nblk=5: uneven lane-group split; 7 rows = one full + one partial threadgroup
		{513, 512},    // nblk=2, odd out
		{130, 3584},   // nblk=14 (7B hidden), tail
		{130, 18944},  // nblk=74 (7B intermediate), tail
		{512, 3584},   // 7B k/v projection
		{3584, 3584},  // 7B q/o projection
		{18944, 3584}, // 7B gate/up
		{3584, 18944}, // 7B down
	} {
		s := s
		t.Run(fmt.Sprintf("%dx%d", s.out, s.in), func(t *testing.T) {
			raw := q4kTestRaw(s.out, s.in, uint64(0x13599+i))
			x := q4kTestVector(s.in, int64(13599+i))
			w := UploadQ4K(raw, s.out, s.in)
			if w == nil {
				t.Fatal("UploadQ4K returned nil")
			}
			defer w.Release()
			want := q4kVectorizedReference(raw, s.out, s.in, x)

			mulmv := make([]float32, s.out)
			if ex := w.gemvWithEventsMode(x, mulmv, nil, q4kGEMVModeMulMv); ex != q4kGEMVExecutedMulMv {
				t.Fatalf("explicit mul_mv executed %v, want %v", ex, q4kGEMVExecutedMulMv)
			}
			legacy := make([]float32, s.out)
			if ex := w.gemvWithEventsMode(x, legacy, nil, q4kGEMVModeScalar); ex != q4kGEMVExecutedScalar {
				t.Fatalf("explicit legacy executed %v, want %v", ex, q4kGEMVExecutedScalar)
			}
			def := make([]float32, s.out)
			if ex := w.gemvWithEvents(x, def, nil); ex != q4kGEMVExecutedMulMv {
				t.Fatalf("DEFAULT executed %v, want mul_mv", ex)
			}
			requireMulMvParity(t, "mul_mv vs CPU", want, mulmv)
			requireMulMvParity(t, "legacy vs CPU", want, legacy)
			requireMulMvParity(t, "mul_mv vs legacy", legacy, mulmv)
			for i := range def {
				if def[i] != mulmv[i] {
					t.Fatalf("DEFAULT row %d = %v, explicit mul_mv = %v; same kernel must be bit-equal", i, def[i], mulmv[i])
				}
			}
		})
	}
}

// TestQ6KMulMvP1Parity pins q6k_mul_mv (and the legacy q6k_gemv control) against the independent
// float64 CPU reference, including the Q6_K head-like and down-projection shapes.
// fak-test:runtime medium est=3s lane=default
func TestQ6KMulMvP1Parity(t *testing.T) {
	if !Available() {
		t.Skip("Metal unavailable")
	}
	defer ResetQ4K()
	defer SetQ4KGEMVKernelDefault(SetQ4KGEMVKernelDefault(GEMVKernelMulMv))

	for i, s := range []mulMvShape{
		{130, 256},   // nblk=1: one block parity idle; tail
		{7, 768},     // nblk=3: odd block count
		{513, 3584},  // nblk=14, odd out
		{4096, 3584}, // Q6_K head-like slice
		{3584, 18944},
	} {
		s := s
		t.Run(fmt.Sprintf("%dx%d", s.out, s.in), func(t *testing.T) {
			raw := q6kTestRaw(s.out, s.in, uint64(0x6b13599+i))
			x := q4kTestVector(s.in, int64(613599+i))
			w := UploadQ6K(raw, s.out, s.in)
			if w == nil {
				t.Fatal("UploadQ6K returned nil")
			}
			defer w.Release()
			want := q6kMulMvReference(raw, s.out, s.in, x)

			mulmv := make([]float32, s.out)
			if ex := w.gemvWithEventsMode(x, mulmv, nil, q4kGEMVModeMulMv); ex != q4kGEMVExecutedMulMv {
				t.Fatalf("explicit mul_mv executed %v, want %v", ex, q4kGEMVExecutedMulMv)
			}
			legacy := make([]float32, s.out)
			if ex := w.gemvWithEventsMode(x, legacy, nil, q4kGEMVModeScalar); ex != q4kGEMVExecutedScalar {
				t.Fatalf("explicit legacy executed %v, want %v", ex, q4kGEMVExecutedScalar)
			}
			def := make([]float32, s.out)
			observation := NewExecutionObservation(ExecutionQ6KGEMV)
			w.GEMVWithEvents(x, def, observation)
			requireCompletedExecution(t, observation, ExecutionQ6KGEMV)
			requireMulMvParity(t, "mul_mv vs CPU", want, mulmv)
			requireMulMvParity(t, "legacy vs CPU", want, legacy)
			for i := range def {
				if def[i] != mulmv[i] {
					t.Fatalf("DEFAULT row %d = %v, explicit mul_mv = %v; DEFAULT must run mul_mv", i, def[i], mulmv[i])
				}
			}
		})
	}
}

// TestQ4KGEMVKernelEnvParse pins the FAK_Q4K_GEMV_KERNEL vocabulary: the Go-facing name
// "scalar" and any case of "legacy" roll back, mul_mv spellings keep the default, and anything
// else is reported unrecognized (the native resolver logs it) instead of silently meaning mul_mv.
// fak-test:runtime fast est=10ms lane=default
func TestQ4KGEMVKernelEnvParse(t *testing.T) {
	for _, tc := range []struct {
		v     string
		want  GEMVKernel
		known bool
	}{
		{"", GEMVKernelMulMv, true},
		{"legacy", GEMVKernelScalar, true},
		{"LEGACY", GEMVKernelScalar, true},
		{"scalar", GEMVKernelScalar, true},
		{"Scalar", GEMVKernelScalar, true},
		{"mul_mv", GEMVKernelMulMv, true},
		{"mulmv", GEMVKernelMulMv, true},
		{"default", GEMVKernelMulMv, true},
		{"0", GEMVKernelMulMv, false},
		{"q4k_gemv", GEMVKernelMulMv, false},
		{"legacy ", GEMVKernelMulMv, false},
	} {
		got, known := q4kGEMVKernelFromEnv(tc.v)
		if got != tc.want || known != tc.known {
			t.Errorf("FAK_Q4K_GEMV_KERNEL=%q -> (%v, %v), want (%v, %v)", tc.v, got, known, tc.want, tc.known)
		}
	}
}

// TestQ4KGEMVDefaultKernelSelector proves the process-default seam: mul_mv by default, the legacy
// kernel after SetQ4KGEMVKernelDefault(GEMVKernelScalar) (the FAK_Q4K_GEMV_KERNEL=legacy
// override), an explicit SetGEMVUseVectorized request still honored, and an unavailable explicit
// request never substituting another kernel.
// fak-test:runtime fast est=300ms lane=default
func TestQ4KGEMVDefaultKernelSelector(t *testing.T) {
	if !Available() {
		t.Skip("Metal unavailable")
	}
	defer ResetQ4K()
	defer SetGEMVUseVectorized(false)
	defer SetQ4KGEMVKernelDefault(SetQ4KGEMVKernelDefault(GEMVKernelMulMv))

	const out, in = 130, 3584
	raw := q4kTestRaw(out, in, 0x5e1ec7)
	x := q4kTestVector(in, 5101)
	w := UploadQ4K(raw, out, in)
	if w == nil {
		t.Fatal("UploadQ4K returned nil")
	}
	defer w.Release()
	q6 := UploadQ6K(q6kTestRaw(out, in, 0x5e1ec8), out, in)
	if q6 == nil {
		t.Fatal("UploadQ6K returned nil")
	}
	defer q6.Release()
	y := make([]float32, out)

	if got := Q4KGEMVKernelDefault(); got != GEMVKernelMulMv {
		t.Fatalf("Q4KGEMVKernelDefault() = %v, want mul_mv", got)
	}
	if ex := w.gemvWithEvents(x, y, nil); ex != q4kGEMVExecutedMulMv {
		t.Fatalf("default Q4_K executed %v, want mul_mv", ex)
	}
	if ex := q6.gemvWithEventsMode(x, y, nil, q4kGEMVModeDefault); ex != q4kGEMVExecutedMulMv {
		t.Fatalf("default Q6_K executed %v, want mul_mv", ex)
	}

	if prev := SetQ4KGEMVKernelDefault(GEMVKernelScalar); prev != GEMVKernelMulMv {
		t.Fatalf("SetQ4KGEMVKernelDefault previous = %v, want mul_mv", prev)
	}
	if got := Q4KGEMVKernelDefault(); got != GEMVKernelScalar {
		t.Fatalf("legacy Q4KGEMVKernelDefault() = %v, want legacy", got)
	}
	if ex := w.gemvWithEvents(x, y, nil); ex != q4kGEMVExecutedScalar {
		t.Fatalf("legacy-default Q4_K executed %v, want legacy", ex)
	}
	if ex := q6.gemvWithEventsMode(x, y, nil, q4kGEMVModeDefault); ex != q4kGEMVExecutedScalar {
		t.Fatalf("legacy-default Q6_K executed %v, want legacy", ex)
	}
	// An explicit mul_mv request is independent of the process default.
	if ex := w.gemvWithEventsMode(x, y, nil, q4kGEMVModeMulMv); ex != q4kGEMVExecutedMulMv {
		t.Fatalf("explicit mul_mv under legacy default executed %v", ex)
	}
	SetQ4KGEMVKernelDefault(GEMVKernelMulMv)

	SetGEMVUseVectorized(true)
	if ex := w.gemvWithEvents(x, y, nil); ex != q4kGEMVExecutedVectorized {
		t.Fatalf("SetGEMVUseVectorized(true) executed %v, want vectorized", ex)
	}
	SetGEMVUseVectorized(false)

	untouched := make([]float32, out)
	for i := range untouched {
		untouched[i] = 13599
	}
	if ex := w.gemvWithEventsMode(x, untouched, nil, q4kGEMVModeVectorizedUnavailable); ex != q4kGEMVNotExecuted {
		t.Fatalf("unavailable request executed %v, want none", ex)
	}
	for i := range untouched {
		if untouched[i] != 13599 {
			t.Fatal("unavailable request mutated output")
		}
	}
}

// TestQ4KGEMVGroupAndFusedMLPMulMvParity proves the grouped GEMV and fused Q6_K-down MLP default
// paths (which bind the default P=1 kernel per dispatch) agree between mul_mv and legacy.
// fak-test:runtime medium est=1s lane=default
func TestQ4KGEMVGroupAndFusedMLPMulMvParity(t *testing.T) {
	if !Available() {
		t.Skip("Metal unavailable")
	}
	defer ResetQ4K()
	defer SetQ4KGEMVKernelDefault(SetQ4KGEMVKernelDefault(GEMVKernelMulMv))

	const hidden, inter = 3584, 1280 // intermediate kept small; nblk=5 exercises the uneven split
	x := q4kTestVector(hidden, 7101)
	for i := range x {
		x[i] *= 0.05
	}
	qRaw := q4kTestRaw(513, hidden, 0x7101)
	kRaw := q4kTestRaw(130, hidden, 0x7102)
	q := UploadQ4K(qRaw, 513, hidden)
	k := UploadQ4K(kRaw, 130, hidden)
	gate := UploadQ4K(q4kTestRaw(inter, hidden, 0x7103), inter, hidden)
	up := UploadQ4K(q4kTestRaw(inter, hidden, 0x7104), inter, hidden)
	down := UploadQ6K(q6kTestRaw(hidden, inter, 0x7105), hidden, inter)
	if q == nil || k == nil || gate == nil || up == nil || down == nil {
		t.Fatal("upload failed")
	}
	defer q.Release()
	defer k.Release()
	defer gate.Release()
	defer up.Release()
	defer down.Release()

	run := func(kernel GEMVKernel) ([][]float32, []float32) {
		SetQ4KGEMVKernelDefault(kernel)
		if got := Q4KGEMVKernelDefault(); got != kernel {
			t.Fatalf("default kernel = %v, want %v", got, kernel)
		}
		group := GEMVGroup([]*Q4KWeight{q, k}, x)
		y := make([]float32, hidden)
		if !FusedMLPQ6Down(gate, up, down, x, y) {
			t.Fatal("FusedMLPQ6Down declined")
		}
		return group, y
	}
	legacyGroup, legacyMLP := run(GEMVKernelScalar)
	mulmvGroup, mulmvMLP := run(GEMVKernelMulMv)
	requireMulMvParity(t, "group q mul_mv vs CPU", q4kVectorizedReference(qRaw, 513, hidden, x), mulmvGroup[0])
	requireMulMvParity(t, "group k mul_mv vs CPU", q4kVectorizedReference(kRaw, 130, hidden, x), mulmvGroup[1])
	requireMulMvParity(t, "group q mul_mv vs legacy", legacyGroup[0], mulmvGroup[0])
	requireMulMvParity(t, "group k mul_mv vs legacy", legacyGroup[1], mulmvGroup[1])
	requireMulMvParity(t, "fused MLP q6down mul_mv vs legacy", legacyMLP, mulmvMLP)
}

// TestProjectionGraphGEMVMulMvRouteParity is the graph-route witness: a P=1 SetGEMVDecode graph
// executes q4k_mul_mv / q6k_mul_mv by default (receipt Q4KGEMVKernels/Q6KGEMVKernels), agrees with
// the legacy kernels pinned through SetGEMVKernel(GEMVKernelScalar) and through the process
// legacy default, and still honors an explicit SetGEMVVectorized(true) for Q4_K.
// fak-test:runtime medium est=1s lane=default
func TestProjectionGraphGEMVMulMvRouteParity(t *testing.T) {
	if !Available() {
		t.Skip("Metal unavailable")
	}
	defer ResetQ4K()
	defer SetQ4KGEMVKernelDefault(SetQ4KGEMVKernelDefault(GEMVKernelMulMv))

	const in = 3584
	x := q4kTestVector(in, 8101)
	for i := range x {
		x[i] *= 0.05
	}
	sq := UploadQ4K(q4kTestRaw(in, in, 0x8101), in, in)     // square: feeds the chained Q6_K projection
	tail := UploadQ4K(q4kTestRaw(130, in, 0x8102), 130, in) // tail guard
	head := UploadQ6K(q6kTestRaw(513, in, 0x8103), 513, in) // odd out Q6_K, chained from sq
	if sq == nil || tail == nil || head == nil {
		t.Fatal("upload failed")
	}
	defer sq.Release()
	defer tail.Release()
	defer head.Release()

	type graphRun struct {
		outputs  [][]float32
		receipt  GraphReceipt
		q4k, q6k GEMVKernelSet
	}
	run := func(name string, configure func(*ProjectionGraph) bool) graphRun {
		t.Helper()
		g, err := BeginProjectionGraph(x, nil, nil, 1, in)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		defer g.Free()
		if !g.SetGEMVDecode() {
			t.Fatalf("%s: P=1 GEMV route unavailable", name)
		}
		if configure != nil && !configure(g) {
			t.Fatalf("%s: kernel configuration refused", name)
		}
		r1, err := g.EncodeQ4K(sq)
		if err != nil {
			t.Fatal(err)
		}
		r2, err := g.EncodeQ4K(tail)
		if err != nil {
			t.Fatal(err)
		}
		r3, err := g.EncodeQ6KFrom(head, r1)
		if err != nil {
			t.Fatal(err)
		}
		outputs, receipt, err := g.FinishRead(r2, r3)
		if err != nil {
			t.Fatal(err)
		}
		if !receipt.Committed || !receipt.CompletedWait || receipt.Encoders != 3 {
			t.Fatalf("%s receipt=%+v, want 3 completed encoders", name, receipt)
		}
		t.Logf("%s: q4k=%v q6k=%v gpu_ms=%.3f", name, receipt.Q4KGEMVKernels, receipt.Q6KGEMVKernels, receipt.GPUMilliseconds)
		return graphRun{outputs: outputs, receipt: receipt, q4k: receipt.Q4KGEMVKernels, q6k: receipt.Q6KGEMVKernels}
	}
	only := func(k GEMVKernel) GEMVKernelSet { return GEMVKernelSet(1) << uint(k) }

	def := run("default", nil)
	if def.q4k != only(GEMVKernelMulMv) || def.q6k != only(GEMVKernelMulMv) {
		t.Fatalf("default graph executed q4k=%v q6k=%v, want mul_mv for both", def.q4k, def.q6k)
	}
	legacy := run("legacy-pinned", func(g *ProjectionGraph) bool { return g.SetGEMVKernel(GEMVKernelScalar) })
	if legacy.q4k != only(GEMVKernelScalar) || legacy.q6k != only(GEMVKernelScalar) {
		t.Fatalf("legacy-pinned graph executed q4k=%v q6k=%v, want legacy", legacy.q4k, legacy.q6k)
	}
	pinned := run("mulmv-pinned", func(g *ProjectionGraph) bool { return g.SetGEMVKernel(GEMVKernelMulMv) })
	if pinned.q4k != only(GEMVKernelMulMv) || pinned.q6k != only(GEMVKernelMulMv) {
		t.Fatalf("mulmv-pinned graph executed q4k=%v q6k=%v", pinned.q4k, pinned.q6k)
	}
	vec := run("vectorized", func(g *ProjectionGraph) bool { return g.SetGEMVVectorized(true) })
	if vec.q4k != only(GEMVKernelVectorized) || vec.q6k != only(GEMVKernelMulMv) {
		t.Fatalf("vectorized graph executed q4k=%v q6k=%v, want vectorized Q4_K and mul_mv Q6_K", vec.q4k, vec.q6k)
	}
	SetQ4KGEMVKernelDefault(GEMVKernelScalar)
	envLegacy := run("process-legacy", nil)
	if envLegacy.q4k != only(GEMVKernelScalar) || envLegacy.q6k != only(GEMVKernelScalar) {
		t.Fatalf("process-legacy graph executed q4k=%v q6k=%v, want legacy", envLegacy.q4k, envLegacy.q6k)
	}
	SetQ4KGEMVKernelDefault(GEMVKernelMulMv)

	for _, cand := range []struct {
		name string
		got  graphRun
	}{{"default", def}, {"mulmv-pinned", pinned}, {"vectorized", vec}, {"process-legacy", envLegacy}} {
		requireMulMvParity(t, cand.name+" q4k tail vs legacy", legacy.outputs[0], cand.got.outputs[0])
		requireMulMvParity(t, cand.name+" chained q6k vs legacy", legacy.outputs[1], cand.got.outputs[1])
	}
}

// mulMvBenchShape is one Qwen2.5-7B decode projection for the kernel-only GB/s benchmark.
type mulMvBenchShape struct {
	name    string
	q6      bool
	out, in int
}

// BenchmarkQ4KGEMVKernelGBs measures kernel-only P=1 GEMV bandwidth (weight bytes / GPU seconds)
// at Qwen2.5-7B shapes for the legacy and mul_mv kernels. The "graph" arm encodes graphEncodes
// projections into ONE ProjectionGraph and divides by the receipt's GPU window, so launch and
// host sync are excluded; the "perop" arm reports the per-command-buffer GPU window of single
// GEMV calls. The GPU is single tenant: run it alone (go test -run XXX -bench Q4KGEMVKernelGBs).
func BenchmarkQ4KGEMVKernelGBs(b *testing.B) {
	if !Available() {
		b.Skip("Metal unavailable")
	}
	defer ResetQ4K()
	defer SetQ4KGEMVKernelDefault(SetQ4KGEMVKernelDefault(GEMVKernelMulMv))
	const graphEncodes = 32
	shapes := []mulMvBenchShape{
		{name: "q4k_3584x3584", out: 3584, in: 3584},
		{name: "q4k_512x3584", out: 512, in: 3584},
		{name: "q4k_18944x3584", out: 18944, in: 3584},
		{name: "q4k_3584x18944", out: 3584, in: 18944},
		{name: "q6k_152064x3584_head", q6: true, out: 152064, in: 3584},
		{name: "q6k_3584x18944_down", q6: true, out: 3584, in: 18944},
	}
	for _, s := range shapes {
		x := q4kTestVector(s.in, 9101)
		for i := range x {
			x[i] *= 0.05
		}
		var (
			q4         *Q4KWeight
			q6         *Q6KWeight
			weightByte float64
		)
		if s.q6 {
			q6 = UploadQ6K(q6kTestRaw(s.out, s.in, 0x9101), s.out, s.in)
			weightByte = float64(s.out) * float64(s.in/256) * 210
		} else {
			q4 = UploadQ4K(q4kTestRaw(s.out, s.in, 0x9101), s.out, s.in)
			weightByte = float64(s.out) * float64(s.in/256) * 144
		}
		if q4 == nil && q6 == nil {
			b.Fatalf("%s: upload failed", s.name)
		}
		for _, kernel := range []GEMVKernel{GEMVKernelScalar, GEMVKernelMulMv} {
			b.Run(fmt.Sprintf("%s/%s/graph", s.name, kernel), func(b *testing.B) {
				var gpuSec, bytes float64
				for i := 0; i < b.N; i++ {
					g, err := BeginProjectionGraph(x, nil, nil, 1, s.in)
					if err != nil {
						b.Fatal(err)
					}
					if !g.SetGEMVDecode() || !g.SetGEMVKernel(kernel) {
						g.Free()
						b.Fatalf("%s: kernel %v unavailable", s.name, kernel)
					}
					for n := 0; n < graphEncodes; n++ {
						var err error
						if s.q6 {
							_, err = g.EncodeQ6K(q6)
						} else {
							_, err = g.EncodeQ4K(q4)
						}
						if err != nil {
							g.Free()
							b.Fatal(err)
						}
					}
					receipt, err := g.Finish()
					g.Free()
					if err != nil || !receipt.TimingAvailable {
						b.Fatalf("graph finish: err=%v receipt=%+v", err, receipt)
					}
					gpuSec += receipt.GPUMilliseconds / 1e3
					bytes += graphEncodes * weightByte
				}
				if gpuSec > 0 {
					b.ReportMetric(bytes/gpuSec/1e9, "GB/s")
					b.ReportMetric(gpuSec/float64(b.N*graphEncodes)*1e6, "gpu-us/op")
				}
			})
			b.Run(fmt.Sprintf("%s/%s/perop", s.name, kernel), func(b *testing.B) {
				mode, _ := q4kGEMVModeFor(kernel)
				y := make([]float32, s.out)
				var gpuSec, bytes float64
				for i := 0; i < b.N; i++ {
					var observation *ExecutionObservation
					if s.q6 {
						observation = NewExecutionObservation(ExecutionQ6KGEMV)
						if ex := q6.gemvWithEventsMode(x, y, observation, mode); ex != kernel {
							b.Fatalf("executed %v, want %v", ex, kernel)
						}
					} else {
						observation = NewExecutionObservation(ExecutionQ4KGEMV)
						if ex := q4.gemvWithEventsMode(x, y, observation, mode); ex != kernel {
							b.Fatalf("executed %v, want %v", ex, kernel)
						}
					}
					snapshot, err := observation.Snapshot()
					if err != nil {
						b.Fatal(err)
					}
					for _, event := range snapshot.Events {
						gpuSec += event.GPUMilliseconds / 1e3
					}
					bytes += weightByte
				}
				if gpuSec > 0 {
					b.ReportMetric(bytes/gpuSec/1e9, "GB/s")
					b.ReportMetric(gpuSec/float64(b.N)*1e6, "gpu-us/op")
				}
			})
		}
		if q4 != nil {
			q4.Release()
		}
		if q6 != nil {
			q6.Release()
		}
	}
}

// TestFusedMLPQ6DownFollowsP1KernelDefault proves the fused Q4_K gate/up + Q6_K down MLP
// (FusedMLPQ6DownFast, also FusedMLPFast) runs its down projection on the process-default P=1
// kernel — q6k_mul_mv by default, q6k_gemv under the legacy default — on both its fused
// single-command-buffer branch and its mg_q6k_gemv_mode fallback, and stays at parity with the
// unfused FusedMLPQ6Down under either family.
// fak-test:runtime fast est=500ms lane=default
func TestFusedMLPQ6DownFollowsP1KernelDefault(t *testing.T) {
	if !Available() {
		t.Skip("Metal unavailable")
	}
	if err := EnsureFusedSwiGLUPipeline(); err != nil {
		t.Skipf("fused SwiGLU pipeline unavailable: %v", err)
	}
	defer ResetQ4K()
	defer SetQ4KGEMVKernelDefault(SetQ4KGEMVKernelDefault(GEMVKernelMulMv))

	const in, inter = 512, 768 // Q6_K down needs in=inter on a 256-element super-block
	gate := UploadQ4K(q4kTestRaw(inter, in, 0x13599a), inter, in)
	up := UploadQ4K(q4kTestRaw(inter, in, 0x13599b), inter, in)
	down := UploadQ6K(q6kTestRaw(in, inter, 0x13599c), in, inter)
	if gate == nil || up == nil || down == nil {
		t.Fatal("weight upload failed")
	}
	x := q4kTestVector(in, 13599)
	for _, family := range []GEMVKernel{GEMVKernelMulMv, GEMVKernelScalar} {
		SetQ4KGEMVKernelDefault(family)
		want := Q4KGEMVKernelDefault()
		if family == GEMVKernelMulMv && want != GEMVKernelMulMv {
			t.Skipf("mul_mv pipeline unavailable (default resolves to %v)", want)
		}
		yRef := make([]float32, in)
		if !FusedMLPQ6Down(gate, up, down, x, yRef) {
			t.Fatalf("%v: unfused FusedMLPQ6Down failed", family)
		}
		yFast := make([]float32, in)
		if !FusedMLPQ6DownFast(gate, up, down, x, yFast) {
			t.Fatalf("%v: FusedMLPQ6DownFast failed", family)
		}
		got, branch := fusedMLPQ6DownLast()
		t.Logf("%v default: fused down projection executed %v on branch %d", family, got, branch)
		if got != want {
			t.Fatalf("%v default: fused down projection executed %v (branch %d), want %v", family, got, branch, want)
		}
		if cos, maxRel := q4kTestCosineMaxRel(yRef, yFast); cos < 0.999999 {
			t.Fatalf("%v default: fused vs unfused cosine %.8f (maxRel %.6g)", family, cos, maxRel)
		}
	}
}
