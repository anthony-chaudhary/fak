package compute

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/kquantbits"
)

// cuda_q3k_mmvq_test.go — resident Q3_K CUDA MMVQ (cuda/q3k_mmvq.cuh, ported from llama.cpp
// vecdotq.cuh / mmvq.cu / convert.cu @571d0d54, MIT). Pure Go, both builds: the C ABI and
// source contract, plus thread-for-thread replays of the ported vec_dot_q3_K_q8_1 and
// dequantize_block_q3_K against the CPU Q3_K oracle (q3kRowDot / q3kDequantSuperBlock).
// Device parity and the GB/s table live in cuda_q3k_device_test.go (-tags cuda).

// q3kTestRaw fills out rows of in/256 Q3_K super-blocks with xorshift bytes and a bounded
// f16 d in the last two bytes, so every decoded weight is finite.
func q3kTestRaw(out, in int, seed uint64) []byte {
	raw := make([]byte, out*(in/q3kSuper)*q3kSuperBlock)
	state := seed | 1
	for i := range raw {
		state ^= state << 13
		state ^= state >> 7
		state ^= state << 17
		raw[i] = byte(state)
	}
	dBits := [...]uint16{0x2c00, 0x3000, 0x3400, 0x3800} // 0.0625 .. 0.5
	for b, base := 0, 0; base < len(raw); b, base = b+1, base+q3kSuperBlock {
		binary.LittleEndian.PutUint16(raw[base+q3kSuperBlock-2:], dBits[b%len(dBits)])
	}
	return raw
}

// q3kTestX returns P*in activations in [-0.5, 0.5) from a fixed LCG.
func q3kTestX(P, in int, seed uint64) []float32 {
	x := make([]float32, P*in)
	state := seed
	for i := range x {
		state = state*6364136223846793005 + 1442695040888963407
		x[i] = float32(state>>40)/float32(1<<24) - 0.5
	}
	return x
}

// q3kQ8_1 replays quantize_q8_1: per 32 values d = amax/127, q = roundf(x/d), and the f16
// copy of d the dot reads back (block_q8_1.ds.x).
func q3kQ8_1(x []float32, roundD func(float32) float32) ([]int8, []float32) {
	qs := make([]int8, len(x))
	ds := make([]float32, len(x)/32)
	for g := range ds {
		xs := x[g*32 : g*32+32]
		var amax float32
		for _, v := range xs {
			amax = float32(math.Max(float64(amax), math.Abs(float64(v))))
		}
		if amax == 0 {
			continue
		}
		d := amax / 127
		for l, v := range xs {
			qs[g*32+l] = int8(math.Round(float64(v / d)))
		}
		ds[g] = roundD(d)
	}
	return qs, ds
}

// q3kMMVQRowModel replays vec_dot_q3_K_q8_1 for one weight row over every (kbx, iqs) the
// mul_mat_vec_q threads visit (qi/vdr = 16 threads per super-block), in integer arithmetic
// with a float64 epilogue. qs/ds are one activation row's q8_1 codes and scales.
func q3kMMVQRowModel(row []byte, qs []int8, ds []float32) float64 {
	const qi3k, qi8 = 16, 8
	le32 := func(b []byte) int32 { return int32(binary.LittleEndian.Uint32(b)) }
	acc := 0.0
	for kbx := 0; kbx*q3kSuperBlock < len(row); kbx++ {
		blk := row[kbx*q3kSuperBlock : (kbx+1)*q3kSuperBlock]
		hmask, qlow, scales := blk[0:32], blk[32:96], blk[96:108]
		d3 := float64(math.Float32frombits(kquantbits.F16BitsToF32Bits(binary.LittleEndian.Uint16(blk[108:110]))))
		for iqs := 0; iqs < qi3k; iqs++ {
			bq8Offset := 4 * (iqs / (qi3k / 2))
			scaleOffset := iqs - iqs%qi8 + (iqs%qi8)/(qi8/2)
			vl := le32(qlow[4*iqs:])
			vh := ^le32(hmask[4*(iqs%(qi3k/2)):]) >> uint(bq8Offset)
			sumf := 0.0
			for i := 0; i < 4; i++ {
				isc := scaleOffset + 2*i
				scLow := int32(scales[isc%8]>>(4*(isc/8))) & 0xF
				scHigh := (int32(scales[8+isc%4]>>(2*(isc/4))) & 3) << 4
				sc := (scLow | scHigh) - 32
				vil := (vl >> uint(2*i)) & 0x03030303
				vih := ((vh >> uint(i)) << 2) & 0x04040404
				g := kbx*8 + bq8Offset + i
				u := qs[g*32+4*(iqs%qi8) : g*32+4*(iqs%qi8)+4]
				var dot int32
				for b := 0; b < 4; b++ {
					vi := int32(int8(byte(vil>>(8*b)))) - int32(int8(byte(vih>>(8*b))))
					dot += vi * int32(u[b])
				}
				sumf += float64(ds[g]) * float64(dot*sc)
			}
			acc += d3 * sumf
		}
	}
	return acc
}

// q3kDequantThreadModel replays dequantize_block_q3_K's 64-thread mapping for one super-block.
func q3kDequantThreadModel(blk []byte) []float32 {
	y := make([]float32, q3kSuper)
	scales, qs, hm := blk[96:108], blk[32:96], blk[0:32]
	dAll := math.Float32frombits(kquantbits.F16BitsToF32Bits(binary.LittleEndian.Uint16(blk[108:110])))
	for t := 0; t < 64; t++ {
		r := t / 4
		tid := r / 2
		is0 := r % 2
		l0 := 16*is0 + 4*(t%4)
		n := tid / 4
		j := tid - 4*n
		m := byte(1) << uint(4*n+j)
		is := 8*n + 2*j + is0
		shift := uint(2 * j)
		var us int8
		switch {
		case is < 4:
			us = int8((scales[is] & 0xF) | (((scales[is+8] >> 0) & 3) << 4))
		case is < 8:
			us = int8((scales[is] & 0xF) | (((scales[is+4] >> 2) & 3) << 4))
		case is < 12:
			us = int8((scales[is-8] >> 4) | (((scales[is] >> 4) & 3) << 4))
		default:
			us = int8((scales[is-8] >> 4) | (((scales[is-4] >> 6) & 3) << 4))
		}
		dl := dAll * float32(int(us)-32)
		for l := l0; l < l0+4; l++ {
			code := int8((qs[32*n+l] >> shift) & 3)
			if hm[l]&m == 0 {
				code -= 4
			}
			y[128*n+32*j+l] = dl * float32(code)
		}
	}
	return y
}

// fak-test:runtime fast est=20ms lane=default
func TestQ3KCUDASourceContract(t *testing.T) {
	wd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	root := findRepoRootForTest(t, wd)
	header, err := os.ReadFile(filepath.Join(root, "internal", "compute", "cuda_backend.h"))
	if err != nil {
		t.Fatal(err)
	}
	for _, decl := range []string{
		"int fcuda_q3k_matmul_f32(const uint8_t *dQ3K, const float *dX, float *dY, int out, int in, int P);",
		"unsigned long long fcuda_q3k_mmvq_launch_count(void);",
	} {
		if !strings.Contains(string(header), decl) {
			t.Errorf("cuda_backend.h missing %q", decl)
		}
	}
	src := cudaSourceClosure(t, filepath.Join(root, "internal", "compute", "cuda_kernels.cu"))
	for _, sym := range []string{
		"571d0d540df04f25298d0e159e520d9fc62ed121", "vec_dot_q3_K_q8_1_impl_mmvq", "vec_dot_q3_K_q8_1",
		"dequantize_block_q3_K", "k_q3k_mmvq_gemv", "k_q3k_dequant", "static int fcuda_q3k_panel_f32(", "g_q4k_decode_arm != 1",
		`extern "C" int fcuda_q3k_matmul_f32(`,
	} {
		if !strings.Contains(src, sym) {
			t.Errorf("cuda source closure missing %q", sym)
		}
	}
	notices, err := os.ReadFile(filepath.Join(root, "THIRD_PARTY_NOTICES.md"))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(notices), "llama.cpp Q3_K MMVQ decode GEMV") {
		t.Error("THIRD_PARTY_NOTICES.md missing the llama.cpp Q3_K MMVQ decode GEMV entry")
	}
}

// TestQ3KCUDAMMVQReplayMatchesCPUOracle pins the ported index arithmetic: with activations that
// are integers with amax 127 in every 32-group, q8_1 is exact (d = 1), so the replay of every
// thread's vec_dot_q3_K_q8_1 must equal q3kRowDot up to float summation order.
//
// fak-test:runtime fast est=50ms lane=default
func TestQ3KCUDAMMVQReplayMatchesCPUOracle(t *testing.T) {
	const out, in = 7, 2304
	raw := q3kTestRaw(out, in, 0x3c0de)
	x := make([]float32, in)
	state := uint64(0x51)
	for i := range x {
		state = state*6364136223846793005 + 1442695040888963407
		x[i] = float32(int(state>>57) - 64) // [-64, 63]
	}
	for g := 0; g < in/32; g++ {
		x[g*32+int(g%32)] = 127
		if g%2 == 1 {
			x[g*32+int(g%32)] = -127
		}
	}
	qs, ds := q3kQ8_1(x, func(d float32) float32 { return d })
	for g, d := range ds {
		if d != 1 {
			t.Fatalf("group %d d = %v, want exact 1", g, d)
		}
	}
	rowBytes := (in / q3kSuper) * q3kSuperBlock
	scratch := make([]float32, q3kSuper)
	for o := 0; o < out; o++ {
		row := raw[o*rowBytes : (o+1)*rowBytes]
		want := float64(q3kRowDot(row, x, scratch))
		got := q3kMMVQRowModel(row, qs, ds)
		if diff := math.Abs(got - want); diff > 1e-5*math.Max(1, math.Abs(want)) {
			t.Fatalf("row %d: mmvq replay %.9g, q3kRowDot %.9g", o, got, want)
		}
	}
}

// TestQ3KCUDADequantReplayMatchesCPUOracle checks dequantize_block_q3_K's thread mapping
// bit-for-bit against q3kDequantSuperBlock.
//
// fak-test:runtime fast est=10ms lane=default
func TestQ3KCUDADequantReplayMatchesCPUOracle(t *testing.T) {
	raw := q3kTestRaw(4, 1024, 0xde03)
	want := make([]float32, q3kSuper)
	for b := 0; b*q3kSuperBlock < len(raw); b++ {
		blk := raw[b*q3kSuperBlock : (b+1)*q3kSuperBlock]
		q3kDequantSuperBlock(want, blk)
		got := q3kDequantThreadModel(blk)
		for i := range want {
			if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
				t.Fatalf("block %d elem %d: thread model %v, oracle %v", b, i, got[i], want[i])
			}
		}
	}
}
