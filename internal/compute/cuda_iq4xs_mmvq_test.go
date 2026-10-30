package compute

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// cuda_iq4xs_mmvq_test.go — resident IQ4_XS CUDA GEMV / MMVQ / dequant-split GEMM
// (cuda/iq4xs_mmvq.cuh, ported from llama.cpp @571d0d54, MIT).
//
// Runs in both builds. Pure Go: the C ABI/source contract plus thread-for-thread replays of
// the three kernels (k_iq4xs_gemv lanes, vec_dot_iq4_xs_q8_1 over every (kbx, iqs) that
// mul_mat_vec_q visits, k_iq4xs_dequant_split's (il, ib) threads) against the CPU oracle
// rawIQRowDot / iq4xsDequantSuperBlock. With -tags cuda and a reachable device: on-device
// parity for every arm. FAK_CUDA_REQUIRE_DEVICE=1 turns a missing device into a failure.

// simIQ4XSLaneRow mirrors k_iq4xs_gemv for one row: lane (ib, il) reads sub-block ib's scale
// and bytes 16*ib+4*il .. +3, dotting the low nibble with x[32*ib+4*il+j] and the high with +16.
func simIQ4XSLaneRow(row []byte, x []float32) float32 {
	nsb := len(row) / iq4xsSuperBlock
	var lanes [32]float32
	for lane := 0; lane < 32; lane++ {
		ib, il := lane>>2, lane&3
		var sum float32
		for sb := 0; sb < nsb; sb++ {
			blk := row[sb*iq4xsSuperBlock : (sb+1)*iq4xsSuperBlock]
			d := f16At(blk) * float32(iq4xsSubScale(blk, ib))
			xs := x[sb*256+32*ib+4*il:]
			q4 := blk[8+16*ib+4*il:]
			var part float32
			for j := 0; j < 4; j++ {
				part += kvaluesIQ4NL[q4[j]&0x0f]*xs[j] + kvaluesIQ4NL[q4[j]>>4]*xs[j+16]
			}
			sum += d * part
		}
		lanes[lane] = sum
	}
	var total float32
	for _, v := range lanes {
		total += v
	}
	return total
}

// iq4xsSubScale is the kernel's sub_scale: the 6-bit (scales_l nibble | scales_h pair) - 32.
func iq4xsSubScale(blk []byte, ib int) int {
	scalesH := binary.LittleEndian.Uint16(blk[2:])
	lo := int(blk[4+ib/2]>>(4*uint(ib%2))) & 0xf
	hi := int(scalesH>>(2*uint(ib))) & 3
	return (lo | hi<<4) - 32
}

// iq4xsMMVQRowModel replays vec_dot_iq4_xs_q8_1 for one weight row over every (kbx, iqs)
// mul_mat_vec_q visits (qi/vdr = 8 threads per super-block, iqs = 0, 4, .., 28), in integer
// arithmetic with a float64 epilogue. qs/ds are one activation row's q8_1 codes and scales.
func iq4xsMMVQRowModel(row []byte, qs []int8, ds []float32) float64 {
	acc := 0.0
	for kbx := 0; kbx*iq4xsSuperBlock < len(row); kbx++ {
		blk := row[kbx*iq4xsSuperBlock : (kbx+1)*iq4xsSuperBlock]
		for iqs := 0; iqs < 32; iqs += 4 {
			ib := iqs / 4
			q8 := qs[(kbx*8+ib)*32:]
			sumi := int32(0)
			for j := 0; j < 4; j++ {
				for k := 0; k < 4; k++ {
					b := blk[8+4*(iqs+j)+k]
					sumi += int32(int8(kvaluesIQ4NL[b&0x0f])) * int32(q8[4*j+k])
					sumi += int32(int8(kvaluesIQ4NL[b>>4])) * int32(q8[16+4*j+k])
				}
			}
			ls := (int(blk[4+iqs/8]>>uint(iqs&4)) & 0x0f) | (int(binary.LittleEndian.Uint16(blk[2:])>>uint(iqs/2))&3)<<4
			sumi *= int32(ls - 32)
			acc += float64(f16At(blk)) * float64(ds[kbx*8+ib]) * float64(sumi)
		}
	}
	return acc
}

// iq4xsDequantSplitModel mirrors k_iq4xs_dequant_split's 32 threads for one super-block and
// returns the f32 value each thread splits (hi + lo reconstructs it to fp16-pair precision),
// failing if a weight is written twice or never.
func iq4xsDequantSplitModel(t *testing.T, blk []byte) []float32 {
	t.Helper()
	out := make([]float32, 256)
	seen := make([]bool, 256)
	for tid := 0; tid < 32; tid++ {
		il, ib := tid/8, tid%8
		base := 32*ib + 4*il
		q4 := blk[8+16*ib+4*il:]
		d := f16At(blk) * float32(iq4xsSubScale(blk, ib))
		for j := 0; j < 4; j++ {
			for _, w := range [2]struct {
				idx int
				v   float32
			}{{base + j, d * kvaluesIQ4NL[q4[j]&0x0f]}, {base + j + 16, d * kvaluesIQ4NL[q4[j]>>4]}} {
				if seen[w.idx] {
					t.Fatalf("thread %d rewrites weight %d", tid, w.idx)
				}
				seen[w.idx] = true
				out[w.idx] = w.v
			}
		}
	}
	for i, s := range seen {
		if !s {
			t.Fatalf("weight %d never written", i)
		}
	}
	return out
}

func iq4xsTestRaw(out, in int, seed uint64) []byte {
	return iqGEMVTestRaw(out, in, iq4xsSuperBlock, seed)
}

// fak-test:runtime fast est=20ms lane=default
func TestIQ4XSCUDASourceContract(t *testing.T) {
	read := func(p string) string {
		b, err := os.ReadFile(p)
		if err != nil {
			t.Fatal(err)
		}
		return string(b)
	}
	src := read(filepath.Join("cuda", "iq4xs_mmvq.cuh"))
	for _, sym := range []string{
		"571d0d540df04f25298d0e159e520d9fc62ed121", "vec_dot_iq4_xs_q8_1", "get_int_from_table_16",
		"VDR_IQ4_XS_Q8_1_MMVQ = 4", "dequantize_block_iq4_xs", "static_assert(sizeof(block_iq4_xs) == 136",
		`FCUDA_TS("k_iq4xs_gemv"`, `FCUDA_TS("k_iq4xs_mmvq_gemv"`, `FCUDA_TS("k_iq4xs_dequant_split"`,
		"g_q4k_decode_arm != 1", "#define FCUDA_IQ4XS_DEQUANT_MIN_P 32", "fp16_compensated_gemm(",
		`extern "C" int fcuda_iq4xs_matmul_f32(`,
	} {
		if !strings.Contains(src, sym) {
			t.Errorf("cuda/iq4xs_mmvq.cuh lost %q", sym)
		}
	}
	if !strings.Contains(read("cuda_kernels.cu"), "#include \"cuda/iq_gemv.cuh\"\n#include \"cuda/iq4xs_mmvq.cuh\"") {
		t.Error("cuda_kernels.cu must include cuda/iq4xs_mmvq.cuh after cuda/iq_gemv.cuh (fcuda_kvalues_iq4nl)")
	}
	hdr := read("cuda_backend.h")
	for _, sym := range []string{
		"int fcuda_iq4xs_matmul_f32(const uint8_t *dW, const float *dX, float *dY, int out, int in, int P);",
		"unsigned long long fcuda_iq4xs_mmvq_launch_count(void);",
		"unsigned long long fcuda_iq4xs_dequant_gemm_launch_count(void);",
	} {
		if !strings.Contains(hdr, sym) {
			t.Errorf("cuda_backend.h lost %q", sym)
		}
	}
	if !strings.Contains(read(filepath.Join("..", "..", "THIRD_PARTY_NOTICES.md")), "cuda/iq4xs_mmvq.cuh") {
		t.Error("THIRD_PARTY_NOTICES.md does not attribute cuda/iq4xs_mmvq.cuh")
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestIQ4XSCUDALaneSimulationMatchesCPUOracle(t *testing.T) {
	const out, in = 37, 1024
	raw := iq4xsTestRaw(out, in, 0x1a4)
	x := iqGEMVTestX(in, 2)
	rowBytes := in / 256 * iq4xsSuperBlock
	scratch := make([]float32, 256)
	got, want := make([]float32, out), make([]float32, out)
	for o := 0; o < out; o++ {
		row := raw[o*rowBytes : (o+1)*rowBytes]
		got[o] = simIQ4XSLaneRow(row, x)
		want[o] = rawIQRowDot(rawIQFormats[IQ4_XS], row, x, scratch)
	}
	if e := iqGEMVRelErr(got, want); e > 1e-5 {
		t.Fatalf("IQ4_XS lane simulation rel err %g > 1e-5", e)
	}
}

// TestIQ4XSCUDAMMVQReplayMatchesCPUOracle: the integer replay of vec_dot_iq4_xs_q8_1 equals
// the dequantized-weight dot with the dequantized q8_1 activation (exact up to f64 rounding),
// and stays within the q8_1 activation-quantization budget of the f32 oracle.
// fak-test:runtime fast est=50ms lane=default
func TestIQ4XSCUDAMMVQReplayMatchesCPUOracle(t *testing.T) {
	const out, in = 29, 2048
	raw := iq4xsTestRaw(out, in, 0x9e4)
	x := iqGEMVTestX(in, 5)
	qs, ds := q3kQ8_1(x, func(v float32) float32 { return v })
	xq := make([]float32, in)
	for i := range xq {
		xq[i] = float32(qs[i]) * ds[i/32]
	}
	rowBytes := in / 256 * iq4xsSuperBlock
	w := DequantRawIQ(IQ4_XS, out, in, raw)
	replay, exactQ, oracle := make([]float32, out), make([]float32, out), make([]float32, out)
	for o := 0; o < out; o++ {
		replay[o] = float32(iq4xsMMVQRowModel(raw[o*rowBytes:(o+1)*rowBytes], qs, ds))
		var a, b float64
		for i := 0; i < in; i++ {
			a += float64(w[o*in+i]) * float64(xq[i])
			b += float64(w[o*in+i]) * float64(x[i])
		}
		exactQ[o], oracle[o] = float32(a), float32(b)
	}
	if e := iqGEMVRelErr(replay, exactQ); e > 1e-5 {
		t.Fatalf("MMVQ replay vs q8_1-dequant dot rel err %g > 1e-5", e)
	}
	if c := cosine(oracle, replay); c < 0.999 {
		t.Fatalf("MMVQ replay vs f32 oracle cosine %.6f < 0.999", c)
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestIQ4XSCUDADequantSplitReplayMatchesCPUOracle(t *testing.T) {
	raw := iq4xsTestRaw(3, 512, 0x51)
	for b := 0; b*iq4xsSuperBlock < len(raw); b++ {
		blk := raw[b*iq4xsSuperBlock : (b+1)*iq4xsSuperBlock]
		got := iq4xsDequantSplitModel(t, blk)
		want := make([]float32, 256)
		iq4xsDequantSuperBlock(want, blk)
		for i := range want {
			if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
				t.Fatalf("block %d weight %d: kernel model %g != oracle %g", b, i, got[i], want[i])
			}
		}
	}
}

// iq4xsLaunchCounter is the cuda-build launch witness (cuda_iq4xs.go).
type iq4xsLaunchCounter interface {
	iq4xsLaunchesForTest() (mmvq, dequantGEMM uint64)
}

// TestIQ4XSCUDADeviceParity gates device MatMul (P=1) / BatchedMatMul (P>1) over resident IQ4_XS
// against the CPU oracle through the same NewRawIQ bytes, on each arm: the f32 GEMV (P < 32),
// the q8_1 MMVQ (P <= 3 under the mmvq decode arm) and the dequant-split compensated fp16 GEMM
// (P >= 32). Each case asserts which arm ran via the launch counters.
// fak-test:runtime fast est=4s lane=default
func TestIQ4XSCUDADeviceParity(t *testing.T) {
	be, ok := Lookup("cuda")
	if !ok {
		if os.Getenv("FAK_CUDA_REQUIRE_DEVICE") == "1" {
			t.Fatal("FAK_CUDA_REQUIRE_DEVICE=1 but the cuda backend is not registered")
		}
		t.Skip("cuda backend not registered (build without -tags cuda or no device)")
	}
	counter, ok := be.(iq4xsLaunchCounter)
	if !ok {
		t.Fatal("cuda backend lacks the IQ4_XS launch witness")
	}
	if !BackendSupportsDeviceWeightDtype(be, IQ4_XS) {
		t.Fatal("cuda backend does not admit IQ4_XS device weights")
	}
	ref := Default()
	before := CUDAQ4KDecodeArmInUse()
	defer func() { _ = SetCUDAQ4KDecodeArm(before) }()
	shapes := []struct{ out, in int }{{321, 256}, {321, 2304}, {1024, 5120}, {5120, 1024}}
	for si, sh := range shapes {
		raw := iq4xsTestRaw(sh.out, sh.in, uint64(0x4c5+si))
		devW := be.Upload(NewRawIQ(be, IQ4_XS, []int{sh.out, sh.in}, raw), IQ4_XS)
		refW := NewRawIQ(ref, IQ4_XS, []int{sh.out, sh.in}, raw)
		for _, P := range []int{1, 2, 3, 8, 31, 32, 64} {
			for _, arm := range []CUDAQ4KDecodeArm{CUDAQ4KDecodeF32, CUDAQ4KDecodeMMVQ} {
				if arm == CUDAQ4KDecodeMMVQ && P > 3 {
					continue
				}
				t.Run(fmt.Sprintf("%dx%d/P%d/%s", sh.out, sh.in, P, arm), func(t *testing.T) {
					if err := SetCUDAQ4KDecodeArm(arm); err != nil {
						t.Fatalf("SetCUDAQ4KDecodeArm(%s): %v", arm, err)
					}
					x := make([]float32, 0, P*sh.in)
					for tok := 0; tok < P; tok++ {
						x = append(x, iqGEMVTestX(sh.in, tok)...)
					}
					xShape := []int{P, sh.in}
					if P == 1 {
						xShape = []int{sh.in}
					}
					m0, g0 := counter.iq4xsLaunchesForTest()
					devX := be.Upload(NewF32(be, xShape, x), F32)
					var devY, refY Tensor
					if P == 1 {
						devY = be.MatMul(devW, devX)
						refY = ref.MatMul(refW, NewF32(ref, xShape, x))
					} else {
						devY = be.BatchedMatMul(devW, devX, P)
						refY = ref.BatchedMatMul(refW, NewF32(ref, xShape, x), P)
					}
					got, want := be.Read(devY), ref.Read(refY)
					be.Free(devY)
					be.Free(devX)
					m1, g1 := counter.iq4xsLaunchesForTest()
					wantM, wantG := uint64(0), uint64(0)
					switch {
					case P >= 32:
						wantG = 1
					case arm == CUDAQ4KDecodeMMVQ:
						wantM = 1
					}
					if m1-m0 != wantM || g1-g0 != wantG {
						t.Fatalf("launches mmvq=%d gemm=%d, want %d/%d", m1-m0, g1-g0, wantM, wantG)
					}
					if len(got) != len(want) {
						t.Fatalf("len %d != %d", len(got), len(want))
					}
					tol := 1e-4
					if wantM == 1 {
						tol = 1e-2 // q8_1 activation quantization
					}
					cos := cosine(want, got)
					if e := iqGEMVRelErr(got, want); e > tol || cos < 0.9999 {
						t.Fatalf("rel err %g (gate %g), cosine %.8f (gate 0.9999)", e, tol, cos)
					}
					for r := 0; r < P; r++ {
						if a, b := argmaxF32(got[r*sh.out:(r+1)*sh.out]), argmaxF32(want[r*sh.out:(r+1)*sh.out]); a != b {
							t.Fatalf("row %d argmax got=%d want=%d", r, a, b)
						}
					}
				})
			}
		}
		be.Free(devW)
	}
}

// TestCUDAIQ4XSABBA times resident IQ4_XS against the Q8_0 representation it replaces (the same
// weights dequantized and re-quantized on upload) at the Qwen3.8-27B FFN shapes, in A B B A
// order per arm pair, P=1 decode (f32 GEMV and q8_1 MMVQ arms) and P=512 prefill (dequant-split
// GEMM). Opt-in with FAK_CUDA_IQ4XS_ABBA=1; it logs ms/call and weight-stream GB/s only and
// asserts nothing about speed.
// fak-test:runtime slow est=60s
func TestCUDAIQ4XSABBA(t *testing.T) {
	if os.Getenv("FAK_CUDA_IQ4XS_ABBA") != "1" {
		t.Skip("set FAK_CUDA_IQ4XS_ABBA=1 to time the IQ4_XS arms")
	}
	be, ok := Lookup("cuda")
	if !ok {
		t.Fatal("cuda backend not registered")
	}
	before := CUDAQ4KDecodeArmInUse()
	defer func() { _ = SetCUDAQ4KDecodeArm(before) }()
	type arm struct {
		name string
		w    Tensor
		dec  CUDAQ4KDecodeArm
		wb   float64
	}
	for _, sh := range []struct{ out, in int }{{17408, 5120}, {5120, 17408}} {
		raw := iq4xsTestRaw(sh.out, sh.in, 0xabba)
		iq := be.Upload(NewRawIQ(be, IQ4_XS, []int{sh.out, sh.in}, raw), IQ4_XS)
		q8 := be.Upload(NewF32(be, []int{sh.out, sh.in}, DequantRawIQ(IQ4_XS, sh.out, sh.in, raw)), Q8_0)
		iqBytes, q8Bytes := float64(len(raw)), float64(sh.out)*float64(sh.in)*1.125
		for _, P := range []int{1, 512} {
			x := make([]float32, 0, P*sh.in)
			for tok := 0; tok < P; tok++ {
				x = append(x, iqGEMVTestX(sh.in, tok)...)
			}
			xShape := []int{P, sh.in}
			if P == 1 {
				xShape = []int{sh.in}
			}
			devX := be.Upload(NewF32(be, xShape, x), F32)
			arms := []arm{{"q8_0", q8, CUDAQ4KDecodeF32, q8Bytes}, {"iq4_xs", iq, CUDAQ4KDecodeF32, iqBytes}}
			if P == 1 {
				arms = append(arms, arm{"q8_0", q8, CUDAQ4KDecodeMMVQ, q8Bytes}, arm{"iq4_xs", iq, CUDAQ4KDecodeMMVQ, iqBytes})
			}
			iters := 200
			if P > 1 {
				iters = 20
			}
			run := func(a arm) float64 {
				if err := SetCUDAQ4KDecodeArm(a.dec); err != nil {
					t.Fatal(err)
				}
				call := func() Tensor {
					if P == 1 {
						return be.MatMul(a.w, devX)
					}
					return be.BatchedMatMul(a.w, devX, P)
				}
				be.Free(call()) // warm
				var last Tensor
				start := time.Now()
				for i := 0; i < iters; i++ {
					if last.Shape != nil {
						be.Free(last)
					}
					last = call()
				}
				_ = be.Read(last)
				el := time.Since(start)
				be.Free(last)
				return float64(el.Microseconds()) / 1000 / float64(iters)
			}
			for i := 0; i+1 < len(arms); i += 2 {
				a, b := arms[i], arms[i+1]
				msA1, msB1, msB2, msA2 := run(a), run(b), run(b), run(a)
				msA, msB := (msA1+msA2)/2, (msB1+msB2)/2
				t.Logf("ABBA %dx%d P=%d arm=%s: %s %.4f ms (%.4f,%.4f) %.1f GB/s | %s %.4f ms (%.4f,%.4f) %.1f GB/s | iq4_xs/q8_0 time %.3fx",
					sh.out, sh.in, P, a.dec, a.name, msA, msA1, msA2, a.wb/msA/1e6, b.name, msB, msB1, msB2, b.wb/msB/1e6, msB/msA)
			}
			be.Free(devX)
		}
		be.Free(iq)
		be.Free(q8)
	}
}
