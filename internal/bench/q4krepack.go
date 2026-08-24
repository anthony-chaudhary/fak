// q4krepack.go — the #5285 spike: an online (load-time) repack of a Q4_K weight
// matrix into an N-wide COLUMN-INTERLEAVED block layout, with N picked by RUNTIME
// CPU-feature detection, plus the A/B that says whether the layout alone pays.
//
// WHY THIS EXISTS. llama.cpp @571d0d54 re-serializes a row-major quant weight at
// set_tensor time into `block_q4_Kx8`-style interleaved records (repack.cpp:143-205,
// picker at :4528, feature branch at :4573-4599) so the CPU GEMM fills its SIMD lanes
// with ONE contiguous load instead of N gathers. The repack is paid once at load; every
// later token's matmul reads the streaming layout. fak has no CPU-side equivalent — it
// holds the GGUF Q4_K bytes verbatim (internal/model/quant_q4k.go: "the resident bytes
// ARE the GGUF bytes") and only repacks for the GPU at upload. #5285 asks for ONE quant
// type, measured, before anyone generalizes.
//
// WHAT IS MEASURED — AND WHAT IS NOT. This arm isolates the LAYOUT axis: both kernels
// below are pure Go, do bit-identical arithmetic, and differ only in memory order, so
// the ratio is attributable to the layout and nothing else. It does NOT measure fak's
// production Q4_K CPU path, which on amd64/arm64 is hand-written assembly living inside
// internal/model (q4kMatRowsRangeArch, q4kReduceRow); those symbols are package-private
// and are a different, faster absolute bar. Read a win here as "the layout is worth a
// matching SIMD kernel", never as "this beats fak's shipped decode".
//
// FENCE. Numbers land OBSERVED with the host named (FAK_BENCH_HW). This arm asserts and
// flips NO gate; its only hard assertion is bit-identity between the two layouts.
package bench

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"time"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// Q4KRepackSchema versions the layout A/B artifact.
const Q4KRepackSchema = "fak.q4krepack.v1"

const q4kRepackFence = "OBSERVED layout A/B (#5285): pure-Go row-major vs column-interleaved Q4_K GEMV, " +
	"bit-identical arithmetic, single goroutine, min-of-iterations. Isolates the LAYOUT axis only — it does " +
	"NOT measure fak's shipped Q4_K CPU path (the AVX2/NEON assembly inside internal/model). Host named via " +
	"FAK_BENCH_HW; NO gate asserted or flipped."

// Q4_K super-block geometry, byte-for-byte the GGUF record internal/model holds resident
// (quant_q4k.go: 2 (d f16) + 2 (dmin f16) + 12 (6-bit packed scales/mins) + 128 nibble
// bytes = 144 bytes per 256 weights).
const (
	// Q4KSuperBlockWeights is how many weights share one Q4_K record.
	Q4KSuperBlockWeights = 256
	// Q4KBlockBytes is one Q4_K super-block's byte cost.
	Q4KBlockBytes = 4 + q4kScaleBytes + q4kQSBytes

	q4kScaleBytes = 12
	q4kQSBytes    = Q4KSuperBlockWeights / 2
	// q4kHeadBytes is the per-record d+dmin+scales prefix that precedes the nibble bytes.
	q4kHeadBytes = 4 + q4kScaleBytes
)

// Q4KInterleaveWidth reports the column-interleave width this host should repack Q4_K
// weights into, and the one-line reason. The width comes from RUNTIME CPU-feature
// detection, not a build tag: model.Q8DecodeKernel() reports the tier fak resolved once
// at init through its own stdlib-only CPUID+XGETBV probe (internal/model/quant_amd64.go)
// or the arm64 HWCAP probe, so a binary shipped to an unknown CPU picks its own layout —
// which is the property #5285 is actually about.
func Q4KInterleaveWidth() (width int, why string) {
	kernel, _ := model.Q8DecodeKernel()
	return q4kWidthForKernel(kernel)
}

// q4kWidthForKernel maps fak's resolved decode-kernel tier to an interleave width,
// following llama.cpp's picker (repack.cpp:4573-4599): AVX2 -> 8 lanes, NEON+dotprod -> 4,
// no vector unit -> 1 (identity, i.e. do not repack).
//
// KNOWN NARROWING. llama.cpp splits arm64 further — NEON+i8mm takes the 4x8 record, plain
// dotprod the 4x4 — but fak's arm64 tier ladder (quant_arm64.go) exposes only
// scalar/neon/neon-amort and has no i8mm tier yet, so both arm64 SIMD tiers map to 4 here.
// Widening to 4x8 is gated on that tier landing, not on this file.
func q4kWidthForKernel(kernel string) (width int, why string) {
	switch kernel {
	case "avx512", "avx2":
		return 8, "8-wide: host decode kernel is " + kernel + " (256-bit f32 lanes; llama.cpp's AVX2 8x8 record)"
	case "neon", "neon-amort":
		return 4, "4-wide: host decode kernel is " + kernel + " (NEON dotprod; llama.cpp's 4x4 record, no i8mm tier in fak yet)"
	default:
		return 1, "1-wide (identity, no repack): host decode kernel is " + kernel + " — no vector lanes to fill"
	}
}

// q4kValidate is the shared precondition for both repack directions.
func q4kValidate(b []byte, out, in, width int) error {
	if width < 1 {
		return fmt.Errorf("bench: Q4_K interleave width %d < 1", width)
	}
	if out < 0 || in <= 0 {
		return fmt.Errorf("bench: Q4_K shape [%d,%d] not positive", out, in)
	}
	if in%Q4KSuperBlockWeights != 0 {
		return fmt.Errorf("bench: Q4_K reduction dim %d is not a multiple of %d", in, Q4KSuperBlockWeights)
	}
	if want := out * (in / Q4KSuperBlockWeights) * Q4KBlockBytes; len(b) != want {
		return fmt.Errorf("bench: Q4_K payload is %d bytes, want %d for shape [%d,%d]", len(b), want, out, in)
	}
	return nil
}

// q4kInterleave moves every field of every Q4_K record between the row-major and the
// width-interleaved layout. The two directions share ONE body on purpose: the layout is a
// pure byte PERMUTATION (same length, nothing derived), and a single body is what makes
// the round-trip witness (TestQ4KRepackIsPurePermutation) able to catch an offset bug
// instead of two hand-written inverses agreeing on the same mistake.
//
// LAYOUT. Rows are grouped `width` at a time. Group g holds rows [g*width, g*width+w),
// where w is `width` except in a short tail group. Because every group before g is full,
// group g starts at exactly g*width*rowBytes — the packed buffer is the same length as the
// row-major one. Inside a group, each super-block index b holds the w lanes field by field:
//
//	[ d     x w ][ dmin  x w ][ scales x w ][ qs byte 0 x w ][ qs byte 1 x w ] ... [ qs byte 127 x w ]
//	  2*w bytes    2*w bytes    12*w bytes    w bytes           w bytes             w bytes
//
// so a kernel that wants lane-l's nibble byte j for all l reads w CONTIGUOUS bytes — the
// "no gather" property #5285 targets.
func q4kInterleave(rowMajor, packed []byte, out, in, width int, toPacked bool) {
	nblk := in / Q4KSuperBlockWeights
	rowBytes := nblk * Q4KBlockBytes
	for o := 0; o < out; o++ {
		g, lane := o/width, o%width
		w := width
		if rest := out - g*width; rest < w {
			w = rest
		}
		groupStart := g * width * rowBytes
		for b := 0; b < nblk; b++ {
			rec := rowMajor[o*rowBytes+b*Q4KBlockBytes:][:Q4KBlockBytes]
			base := groupStart + b*w*Q4KBlockBytes
			d := packed[base+2*lane:][:2]
			dmin := packed[base+2*w+2*lane:][:2]
			sc := packed[base+4*w+q4kScaleBytes*lane:][:q4kScaleBytes]
			qs := base + q4kHeadBytes*w
			if toPacked {
				copy(d, rec[0:2])
				copy(dmin, rec[2:4])
				copy(sc, rec[4:q4kHeadBytes])
				for j := 0; j < q4kQSBytes; j++ {
					packed[qs+j*w+lane] = rec[q4kHeadBytes+j]
				}
				continue
			}
			copy(rec[0:2], d)
			copy(rec[2:4], dmin)
			copy(rec[4:q4kHeadBytes], sc)
			for j := 0; j < q4kQSBytes; j++ {
				rec[q4kHeadBytes+j] = packed[qs+j*w+lane]
			}
		}
	}
}

// RepackQ4KInterleaved is the load-time repack: it takes the raw row-major Q4_K payload of
// an [out, in] weight (exactly the bytes internal/model holds resident) and returns the
// width-interleaved permutation of it. Width 1 is the identity, so a scalar host can call
// this unconditionally and get its bytes back unchanged.
func RepackQ4KInterleaved(raw []byte, out, in, width int) ([]byte, error) {
	if err := q4kValidate(raw, out, in, width); err != nil {
		return nil, err
	}
	packed := make([]byte, len(raw))
	q4kInterleave(raw, packed, out, in, width, true)
	return packed, nil
}

// UnrepackQ4KInterleaved inverts RepackQ4KInterleaved. It exists for the round-trip
// witness — proving the repack invents and drops no byte is the cheapest strong
// correctness statement available for a permutation.
func UnrepackQ4KInterleaved(packed []byte, out, in, width int) ([]byte, error) {
	if err := q4kValidate(packed, out, in, width); err != nil {
		return nil, err
	}
	raw := make([]byte, len(packed))
	q4kInterleave(raw, packed, out, in, width, false)
	return raw, nil
}

// q4kDequantRecord writes one super-block's 256 weights into dst. This is
// internal/model's q4kDequantSuperBlock loop shape reproduced at the bench seam because
// that symbol is package-private; the ARITHMETIC is not copied — the 6-bit scale/min
// unpack and the f16 widen are fak's own exported model.GetScaleMinK4 /
// model.F16BitsToF32Bits, so this cannot drift away from the resident path's numerics.
func q4kDequantRecord(dst []float32, rec []byte) {
	d := math.Float32frombits(model.F16BitsToF32Bits(binary.LittleEndian.Uint16(rec[0:])))
	dmin := math.Float32frombits(model.F16BitsToF32Bits(binary.LittleEndian.Uint16(rec[2:])))
	scales := rec[4:q4kHeadBytes]
	q := rec[q4kHeadBytes:Q4KBlockBytes]
	qi, is := 0, 0
	for j := 0; j < Q4KSuperBlockWeights; j += 64 {
		sc, m := model.GetScaleMinK4(is, scales)
		d1, m1 := d*float32(sc), dmin*float32(m)
		sc, m = model.GetScaleMinK4(is+1, scales)
		d2, m2 := d*float32(sc), dmin*float32(m)
		for l := 0; l < 32; l++ {
			dst[j+l] = d1*float32(q[qi+l]&0x0f) - m1
		}
		for l := 0; l < 32; l++ {
			dst[j+32+l] = d2*float32(q[qi+l]>>4) - m2
		}
		qi += 32
		is += 2
	}
}

// q4kGemvRowMajor is the A side: fak's portable resident-Q4_K decode GEMV shape
// (internal/model's q4kMatRowsRange) — for each output row, for each super-block, dequant
// 256 f32 into an L1-resident scratch and run a stride-4 four-accumulator dot, summing
// across super-blocks in row order. Single goroutine so the number is the kernel, not the
// scheduler.
func q4kGemvRowMajor(raw []byte, out, in int, x, y []float32) {
	nblk := in / Q4KSuperBlockWeights
	rowBytes := nblk * Q4KBlockBytes
	buf := make([]float32, Q4KSuperBlockWeights)
	for o := 0; o < out; o++ {
		row := raw[o*rowBytes:]
		var acc float32
		for b := 0; b < nblk; b++ {
			q4kDequantRecord(buf, row[b*Q4KBlockBytes:(b+1)*Q4KBlockBytes])
			xs := x[b*Q4KSuperBlockWeights:]
			var s0, s1, s2, s3 float32
			for i := 0; i < Q4KSuperBlockWeights; i += 4 {
				s0 += buf[i] * xs[i]
				s1 += buf[i+1] * xs[i+1]
				s2 += buf[i+2] * xs[i+2]
				s3 += buf[i+3] * xs[i+3]
			}
			acc += (s0 + s1) + (s2 + s3)
		}
		y[o] = acc
	}
}

// q4kGemvInterleaved is the B side: the same GEMV over the width-interleaved layout,
// shaped the way an N-lane SIMD kernel would run it. Every nibble load in the dequant
// stage reads `w` CONTIGUOUS lane bytes (the no-gather property), the scratch is
// lane-minor so the writes are contiguous too, and the dot stage's inner loop is the lane
// axis — i.e. the loop a vector unit would replace with one register.
//
// BIT-IDENTITY IS THE POINT. Per output row the arithmetic is unchanged from
// q4kGemvRowMajor: same dequant, same stride-4 four-accumulator dot, same super-block
// order. Only the memory order moves, so max|Δ| must be exactly 0
// (TestQ4KInterleavedGemvIsBitIdentical). That is what makes the timing ratio
// attributable to the layout alone.
//
// FULL WIDTH-8 GROUPS TAKE THE SPECIALIZATION. This generic body keeps its per-lane state
// (accumulators, per-lane scales/mins) in heap slices sized at run time, so every innermost
// access pays a bounds check and the lane loop cannot unroll — a cost of Go's slice
// discipline, NOT of the layout, and it was charging the interleaved side most of its
// measured deficit. q4kGemvInterleaved8 runs the same arithmetic over fixed [8]float32
// arrays instead; on this host (linux/amd64, avx512 tier, 2048x6144) that moved the ratio
// from 0.644x to 0.865x, which is what lets the residual gap be read as the layout's own
// scalar cost rather than as an artifact. The generic body still runs every other width and
// the short tail group.
func q4kGemvInterleaved(packed []byte, out, in, width int, x, y []float32) {
	row0 := 0
	if width == 8 {
		row0 = q4kGemvInterleaved8(packed, out, in, x, y)
	}
	nblk := in / Q4KSuperBlockWeights
	rowBytes := nblk * Q4KBlockBytes
	buf := make([]float32, width*Q4KSuperBlockWeights) // lane-minor: buf[i*width+lane]
	acc := make([]float32, width)
	dLo := make([]float32, width)
	mLo := make([]float32, width)
	dHi := make([]float32, width)
	mHi := make([]float32, width)
	s0 := make([]float32, width)
	s1 := make([]float32, width)
	s2 := make([]float32, width)
	s3 := make([]float32, width)

	for ; row0 < out; row0 += width {
		w := width
		if rest := out - row0; rest < w {
			w = rest
		}
		// Every group before row0 was full, so group row0/width starts at exactly
		// row0*rowBytes — the same invariant the repack's offset math relies on, which is
		// what lets the width-8 specialization hand off mid-buffer.
		groupStart := row0 * rowBytes
		for l := 0; l < w; l++ {
			acc[l] = 0
		}
		q4kAccumulateInterleavedGroup(packed, groupStart, nblk, w, width, x,
			acc[:w], dLo[:w], mLo[:w], dHi[:w], mHi[:w], s0[:w], s1[:w], s2[:w], s3[:w], buf)
		for l := 0; l < w; l++ {
			y[row0+l] = acc[l]
		}
	}
}

// q4kGemvInterleaved8 runs the FULL 8-row groups of the interleaved GEMV and returns the
// first row it did not handle (the start of the short tail group, which the generic body
// finishes). It is the pure-Go stand-in for llama.cpp's block_q4_Kx8 kernel: the lane state
// lives in fixed [8]float32 arrays and every innermost loop has a constant bound, so the
// compiler keeps lanes in registers, elides the bounds checks, and unrolls — the things a
// vector register would give for free and that the generic slice body cannot express.
//
// The arithmetic, operand order, and accumulation order are byte-for-byte the generic
// body's, so bit-identity against q4kGemvRowMajor still holds and
// TestQ4KInterleavedGemvIsBitIdentical covers this path at width 8 (its shapes include
// full-group, full-plus-tail, and out<width cases).
func q4kGemvInterleaved8(packed []byte, out, in int, x, y []float32) int {
	const w = 8
	nblk := in / Q4KSuperBlockWeights
	rowBytes := nblk * Q4KBlockBytes
	var buf [w * Q4KSuperBlockWeights]float32 // lane-minor: buf[i*w+lane]

	row0 := 0
	for ; row0+w <= out; row0 += w {
		var acc, dLo, mLo, dHi, mHi, s0, s1, s2, s3 [w]float32
		groupStart := row0 * rowBytes
		for b := 0; b < nblk; b++ {
			base := groupStart + b*w*Q4KBlockBytes
			dOff, mOff, scOff := base, base+2*w, base+4*w
			qsOff := base + q4kHeadBytes*w

			for j := 0; j < Q4KSuperBlockWeights; j += 64 {
				is := j / 32
				for l := 0; l < w; l++ {
					d := math.Float32frombits(model.F16BitsToF32Bits(binary.LittleEndian.Uint16(packed[dOff+2*l:])))
					dmin := math.Float32frombits(model.F16BitsToF32Bits(binary.LittleEndian.Uint16(packed[mOff+2*l:])))
					sc := packed[scOff+q4kScaleBytes*l:][:q4kScaleBytes]
					a, c := model.GetScaleMinK4(is, sc)
					dLo[l], mLo[l] = d*float32(a), dmin*float32(c)
					a, c = model.GetScaleMinK4(is+1, sc)
					dHi[l], mHi[l] = d*float32(a), dmin*float32(c)
				}
				qi := (j / 64) * 32
				for k := 0; k < 32; k++ {
					lanes := (*[w]byte)(packed[qsOff+(qi+k)*w:]) // w contiguous bytes: no gather
					lo := (*[w]float32)(buf[(j+k)*w:])
					hi := (*[w]float32)(buf[(j+32+k)*w:])
					for l := 0; l < w; l++ {
						q := lanes[l]
						lo[l] = dLo[l]*float32(q&0x0f) - mLo[l]
						hi[l] = dHi[l]*float32(q>>4) - mHi[l]
					}
				}
			}

			xs := x[b*Q4KSuperBlockWeights:]
			s0, s1, s2, s3 = [w]float32{}, [w]float32{}, [w]float32{}, [w]float32{}
			for i := 0; i < Q4KSuperBlockWeights; i += 4 {
				b0 := (*[w]float32)(buf[i*w:])
				b1 := (*[w]float32)(buf[(i+1)*w:])
				b2 := (*[w]float32)(buf[(i+2)*w:])
				b3 := (*[w]float32)(buf[(i+3)*w:])
				x0, x1, x2, x3 := xs[i], xs[i+1], xs[i+2], xs[i+3]
				for l := 0; l < w; l++ {
					s0[l] += b0[l] * x0
					s1[l] += b1[l] * x1
					s2[l] += b2[l] * x2
					s3[l] += b3[l] * x3
				}
			}
			for l := 0; l < w; l++ {
				acc[l] += (s0[l] + s1[l]) + (s2[l] + s3[l])
			}
		}
		for l := 0; l < w; l++ {
			y[row0+l] = acc[l]
		}
	}
	return row0
}

func q4kAccumulateInterleavedGroup(
	packed []byte,
	groupStart, blocks, lanes, stride int,
	x, acc, dLo, mLo, dHi, mHi, s0, s1, s2, s3, buf []float32,
) {
	for block := 0; block < blocks; block++ {
		base := groupStart + block*lanes*Q4KBlockBytes
		dOff, mOff, scOff := base, base+2*lanes, base+4*lanes
		qsOff := base + q4kHeadBytes*lanes

		for j := 0; j < Q4KSuperBlockWeights; j += 64 {
			scaleIndex := j / 32
			q4kInterleavedScales(packed, dOff, mOff, scOff, scaleIndex, dLo, mLo, dHi, mHi)
			quantIndex := (j / 64) * 32
			for k := 0; k < 32; k++ {
				quantLanes := packed[qsOff+(quantIndex+k)*lanes:][:lanes]
				lo := buf[(j+k)*stride:][:lanes]
				hi := buf[(j+32+k)*stride:][:lanes]
				q4kInterleavedQuants(quantLanes, lo, hi, dLo, mLo, dHi, mHi)
			}
		}

		xBlock := x[block*Q4KSuperBlockWeights:]
		q4kAccumulateInterleaved(acc, s0, s1, s2, s3, buf, xBlock, stride)
	}
}

func q4kInterleavedScales(packed []byte, dOff, mOff, scOff, scaleIndex int, dLo, mLo, dHi, mHi []float32) {
	for lane := range dLo {
		d := math.Float32frombits(model.F16BitsToF32Bits(binary.LittleEndian.Uint16(packed[dOff+2*lane:])))
		dmin := math.Float32frombits(model.F16BitsToF32Bits(binary.LittleEndian.Uint16(packed[mOff+2*lane:])))
		scales := packed[scOff+q4kScaleBytes*lane:][:q4kScaleBytes]
		scale, min := model.GetScaleMinK4(scaleIndex, scales)
		dLo[lane], mLo[lane] = d*float32(scale), dmin*float32(min)
		scale, min = model.GetScaleMinK4(scaleIndex+1, scales)
		dHi[lane], mHi[lane] = d*float32(scale), dmin*float32(min)
	}
}

func q4kInterleavedQuants(lanes []byte, lo, hi, dLo, mLo, dHi, mHi []float32) {
	for lane, q := range lanes {
		lo[lane] = dLo[lane]*float32(q&0x0f) - mLo[lane]
		hi[lane] = dHi[lane]*float32(q>>4) - mHi[lane]
	}
}

func q4kAccumulateInterleaved(acc, s0, s1, s2, s3, buf, x []float32, stride int) {
	clear(s0)
	clear(s1)
	clear(s2)
	clear(s3)
	for i := 0; i < Q4KSuperBlockWeights; i += 4 {
		b0 := buf[i*stride:][:len(acc)]
		b1 := buf[(i+1)*stride:][:len(acc)]
		b2 := buf[(i+2)*stride:][:len(acc)]
		b3 := buf[(i+3)*stride:][:len(acc)]
		x0, x1, x2, x3 := x[i], x[i+1], x[i+2], x[i+3]
		for lane := range acc {
			s0[lane] += b0[lane] * x0
			s1[lane] += b1[lane] * x1
			s2[lane] += b2[lane] * x2
			s3[lane] += b3[lane] * x3
		}
	}
	for lane := range acc {
		acc[lane] += (s0[lane] + s1[lane]) + (s2[lane] + s3[lane])
	}
}

// Q4KRepackConfig sizes one A/B run. Zero fields take the defaults in withQ4KDefaults, so
// a bare Q4KRepackConfig{} is a valid full run.
type Q4KRepackConfig struct {
	// Out and In are the weight shape [out, in]; In must be a multiple of 256.
	// Defaults are a GLM-5.2-shaped offloaded expert slice (#5285's stated target).
	Out int `json:"out"`
	In  int `json:"in"`
	// Iters is how many timed GEMVs each side runs (min-of-iterations is reported).
	Iters int `json:"iters"`
	// Width is the interleave width; 0 => Q4KInterleaveWidth() (runtime-detected).
	Width int `json:"width"`
	// Seed makes the synthetic weight deterministic across hosts and reruns.
	Seed int64 `json:"seed"`
}

func withQ4KDefaults(cfg Q4KRepackConfig) Q4KRepackConfig {
	if cfg.Out <= 0 {
		cfg.Out = 2048
	}
	if cfg.In <= 0 {
		cfg.In = 6144
	}
	if cfg.Iters <= 0 {
		cfg.Iters = 25
	}
	if cfg.Seed == 0 {
		cfg.Seed = 0x5285
	}
	return cfg
}

// Q4KRepackReport is the self-verifying A/B artifact.
type Q4KRepackReport struct {
	Schema   string `json:"schema"`
	Issue    string `json:"issue"`
	Fence    string `json:"fence"`
	Hardware string `json:"hardware"`
	GOOS     string `json:"goos"`
	GOARCH   string `json:"goarch"`
	NumCPU   int    `json:"num_cpu"`

	// DecodeKernel is the runtime-detected tier the width was selected from.
	DecodeKernel string `json:"decode_kernel"`
	Width        int    `json:"width"`
	WidthWhy     string `json:"width_why"`

	Out         int `json:"out"`
	In          int `json:"in"`
	Iters       int `json:"iters"`
	WeightBytes int `json:"weight_bytes"`

	// RepackNS is the one-time load-time cost of the interleave for this weight.
	RepackNS int64 `json:"repack_ns"`
	// RepackRoundTripOK is true when unrepack(repack(raw)) == raw byte for byte.
	RepackRoundTripOK bool `json:"repack_round_trip_ok"`

	RowMajorNS    int64 `json:"row_major_ns_per_gemv"`
	InterleavedNS int64 `json:"interleaved_ns_per_gemv"`
	// SpeedupX is row-major / interleaved: >1 means the interleaved layout is faster.
	SpeedupX float64 `json:"speedup_x"`
	// BreakEvenGEMVs is how many GEMVs it takes to repay RepackNS, or -1 when the
	// interleaved layout is not faster (no amortization exists to compute).
	BreakEvenGEMVs float64 `json:"break_even_gemvs"`

	MaxAbsDelta  float64 `json:"max_abs_delta"`
	BitIdentical bool    `json:"bit_identical"`

	Verdict string `json:"verdict"`
	Digest  string `json:"digest"`
}

// computeDigest returns the SHA-256 hex of the canonical report JSON with Digest cleared,
// so a reader can recompute it and confirm the witness was not edited after the fact.
func (r *Q4KRepackReport) computeDigest() string {
	return computeReportDigest(r, &r.Digest)
}

// VerifyDigest recomputes the digest and reports whether it still matches.
func (r Q4KRepackReport) VerifyDigest() bool {
	want := r.Digest
	got := (&r).computeDigest()
	return want != "" && want == got
}

// randQ4KWeight builds a deterministic synthetic row-major Q4_K payload. Any byte pattern
// is a legal 6-bit scales/mins field and a legal nibble field, so those are random; d and
// dmin are drawn as bounded f16 exponents only to keep Inf/NaN out of the dot, which would
// make the bit-identity check vacuous (NaN != NaN).
func randQ4KWeight(rng *rand.Rand, out, in int) []byte {
	nblk := in / Q4KSuperBlockWeights
	raw := make([]byte, out*nblk*Q4KBlockBytes)
	for i := 0; i < out*nblk; i++ {
		rec := raw[i*Q4KBlockBytes:][:Q4KBlockBytes]
		rng.Read(rec[4:])
		binary.LittleEndian.PutUint16(rec[0:], randF16Bits(rng))
		binary.LittleEndian.PutUint16(rec[2:], randF16Bits(rng))
	}
	return raw
}

// randF16Bits draws a finite half-precision value in roughly ±[2^-3, 2^2): sign random,
// exponent field held in [12,16] (never 0 or 31, so never subnormal/Inf/NaN), mantissa
// random.
func randF16Bits(rng *rand.Rand) uint16 {
	sign := uint16(rng.Intn(2)) << 15
	exp := uint16(12+rng.Intn(5)) << 10
	return sign | exp | uint16(rng.Intn(1<<10))
}

// RunQ4KRepackAB builds a synthetic Q4_K weight, repacks it into the runtime-selected
// interleaved layout, and times both GEMVs against the same activation. Both sides are
// warmed once and then run Iters times, alternating A/B/A/B so a thermal or scheduler
// drift hits both equally; the reported per-GEMV time is the MINIMUM over iterations (the
// least-contaminated estimator on a shared host, and this checkout is a live multi-session
// tree). It returns an error only on a malformed shape — a slower interleave is a RESULT,
// not a failure.
func RunQ4KRepackAB(cfg Q4KRepackConfig) (Q4KRepackReport, error) {
	cfg = withQ4KDefaults(cfg)
	width, why := cfg.Width, ""
	kernel, _ := model.Q8DecodeKernel()
	if width <= 0 {
		width, why = Q4KInterleaveWidth()
	} else {
		_, why = q4kWidthForKernel(kernel)
		why = fmt.Sprintf("width %d pinned by config (runtime detection would say: %s)", width, why)
	}

	rng := rand.New(rand.NewSource(cfg.Seed))
	raw := randQ4KWeight(rng, cfg.Out, cfg.In)
	x := make([]float32, cfg.In)
	for i := range x {
		// Mixed magnitudes and signs so a lane- or affine-min mix-up shows as a delta
		// rather than hiding inside small uniform noise.
		x[i] = float32(rng.NormFloat64()) * float32(1+(i%9))
	}

	start := time.Now()
	packed, err := RepackQ4KInterleaved(raw, cfg.Out, cfg.In, width)
	repackNS := time.Since(start).Nanoseconds()
	if err != nil {
		return Q4KRepackReport{}, err
	}

	back, err := UnrepackQ4KInterleaved(packed, cfg.Out, cfg.In, width)
	if err != nil {
		return Q4KRepackReport{}, err
	}
	roundTrip := string(back) == string(raw)

	yA := make([]float32, cfg.Out)
	yB := make([]float32, cfg.Out)
	q4kGemvRowMajor(raw, cfg.Out, cfg.In, x, yA)
	q4kGemvInterleaved(packed, cfg.Out, cfg.In, width, x, yB)

	maxDelta, identical := 0.0, true
	for o := 0; o < cfg.Out; o++ {
		if d := math.Abs(float64(yA[o]) - float64(yB[o])); d > maxDelta {
			maxDelta = d
		}
		if math.Float32bits(yA[o]) != math.Float32bits(yB[o]) {
			identical = false
		}
	}

	bestA, bestB := int64(math.MaxInt64), int64(math.MaxInt64)
	for i := 0; i < cfg.Iters; i++ {
		t := time.Now()
		q4kGemvRowMajor(raw, cfg.Out, cfg.In, x, yA)
		if ns := time.Since(t).Nanoseconds(); ns < bestA {
			bestA = ns
		}
		t = time.Now()
		q4kGemvInterleaved(packed, cfg.Out, cfg.In, width, x, yB)
		if ns := time.Since(t).Nanoseconds(); ns < bestB {
			bestB = ns
		}
	}

	rep := Q4KRepackReport{
		Schema:            Q4KRepackSchema,
		Issue:             "#5285",
		Fence:             q4kRepackFence,
		Hardware:          hardwareLabel(),
		GOOS:              runtime.GOOS,
		GOARCH:            runtime.GOARCH,
		NumCPU:            runtime.NumCPU(),
		DecodeKernel:      kernel,
		Width:             width,
		WidthWhy:          why,
		Out:               cfg.Out,
		In:                cfg.In,
		Iters:             cfg.Iters,
		WeightBytes:       len(raw),
		RepackNS:          repackNS,
		RepackRoundTripOK: roundTrip,
		RowMajorNS:        bestA,
		InterleavedNS:     bestB,
		MaxAbsDelta:       maxDelta,
		BitIdentical:      identical,
		BreakEvenGEMVs:    -1,
	}
	if bestB > 0 {
		rep.SpeedupX = float64(bestA) / float64(bestB)
	}
	switch {
	case bestB < bestA:
		rep.BreakEvenGEMVs = float64(repackNS) / float64(bestA-bestB)
		rep.Verdict = fmt.Sprintf("interleaved layout FASTER: %.3fx, repack repaid after %.0f GEMVs",
			rep.SpeedupX, rep.BreakEvenGEMVs)
	case bestB > bestA:
		rep.Verdict = fmt.Sprintf("interleaved layout SLOWER: %.3fx — the layout alone does not pay in pure Go "+
			"on this host; the llama.cpp win needs the matching SIMD kernel, not the repack", rep.SpeedupX)
	default:
		rep.Verdict = "interleaved layout NEUTRAL: identical minimum times"
	}
	if !identical {
		rep.Verdict = "REFUSED to interpret timing: the two layouts disagree numerically (max|Δ|=" +
			fmt.Sprintf("%g", maxDelta) + ") — the repack or the kernel is wrong"
	}
	rep.Digest = rep.computeDigest()
	return rep, nil
}
