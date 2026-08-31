package ggufload

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"runtime"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/model"
)

func alignment(meta map[string]Value) (uint64, error) {
	align := uint64(defaultAlign)
	if v, ok := meta["general.alignment"]; ok {
		got, ok := valueUint64(v)
		if !ok {
			return 0, fmt.Errorf("gguf: general.alignment is not an unsigned integer")
		}
		align = got
	}
	if align == 0 || align%8 != 0 {
		return 0, fmt.Errorf("gguf: invalid alignment %d", align)
	}
	return align, nil
}

func alignOffset(off, align uint64) uint64 {
	return off + (align-(off%align))%align
}

// tensorOnDiskBytes is the best-effort on-disk payload size of a tensor for load-progress
// accounting: tensorPayloadBytes, or 0 if its shape/type is not byte-sizable. It never
// errors — a 0 from an exotic tensor only understates the running GB, not the percentage.
func tensorOnDiskBytes(t TensorInfo) int64 {
	n, err := tensorPayloadBytes(t)
	if err != nil {
		return 0
	}
	return int64(n)
}

func tensorPayloadBytes(t TensorInfo) (uint64, error) {
	elems, err := tensorElems(t)
	if err != nil {
		return 0, err
	}
	payload := func(units, bytesPerUnit uint64) (uint64, error) {
		if units > math.MaxUint64/bytesPerUnit {
			return 0, fmt.Errorf("gguf: tensor %s type %s payload size overflows uint64", t.Name, t.Type)
		}
		return units * bytesPerUnit, nil
	}
	switch t.Type {
	case TensorF32:
		return payload(elems, 4)
	case TensorF16, TensorBF16:
		return payload(elems, 2)
	case TensorQ4_0:
		if elems%qk4 != 0 {
			return 0, fmt.Errorf("gguf: tensor %s Q4_0 element count %d is not a multiple of %d", t.Name, elems, qk4)
		}
		return payload(elems/qk4, blockQ4_0Bytes)
	case TensorQ4_1:
		if elems%qk4 != 0 {
			return 0, fmt.Errorf("gguf: tensor %s Q4_1 element count %d is not a multiple of %d", t.Name, elems, qk4)
		}
		return payload(elems/qk4, blockQ4_1Bytes)
	case TensorQ5_0:
		if elems%qk5 != 0 {
			return 0, fmt.Errorf("gguf: tensor %s Q5_0 element count %d is not a multiple of %d", t.Name, elems, qk5)
		}
		return payload(elems/qk5, blockQ5_0Bytes)
	case TensorQ5_1:
		if elems%qk5 != 0 {
			return 0, fmt.Errorf("gguf: tensor %s Q5_1 element count %d is not a multiple of %d", t.Name, elems, qk5)
		}
		return payload(elems/qk5, blockQ5_1Bytes)
	case TensorQ8_0:
		if elems%qk8_0 != 0 {
			return 0, fmt.Errorf("gguf: tensor %s Q8_0 element count %d is not a multiple of %d", t.Name, elems, qk8_0)
		}
		return payload(elems/qk8_0, blockQ8_0Bytes)
	case TensorQ2_K:
		if elems%qkK != 0 {
			return 0, fmt.Errorf("gguf: tensor %s Q2_K element count %d is not a multiple of %d", t.Name, elems, qkK)
		}
		return payload(elems/qkK, blockQ2KBytes)
	case TensorQ3_K:
		if elems%qkK != 0 {
			return 0, fmt.Errorf("gguf: tensor %s Q3_K element count %d is not a multiple of %d", t.Name, elems, qkK)
		}
		return payload(elems/qkK, blockQ3KBytes)
	case TensorQ4_K:
		if elems%qkK != 0 {
			return 0, fmt.Errorf("gguf: tensor %s Q4_K element count %d is not a multiple of %d", t.Name, elems, qkK)
		}
		return payload(elems/qkK, blockQ4KBytes)
	case TensorQ5_K:
		if elems%qkK != 0 {
			return 0, fmt.Errorf("gguf: tensor %s Q5_K element count %d is not a multiple of %d", t.Name, elems, qkK)
		}
		return payload(elems/qkK, blockQ5KBytes)
	case TensorQ6_K:
		if elems%qkK != 0 {
			return 0, fmt.Errorf("gguf: tensor %s Q6_K element count %d is not a multiple of %d", t.Name, elems, qkK)
		}
		return payload(elems/qkK, blockQ6KBytes)
	case TensorIQ2_XXS:
		if elems%qkK != 0 {
			return 0, fmt.Errorf("gguf: tensor %s IQ2_XXS element count %d is not a multiple of %d", t.Name, elems, qkK)
		}
		return payload(elems/qkK, blockIQ2XXSBytes)
	case TensorIQ2_XS:
		if elems%qkK != 0 {
			return 0, fmt.Errorf("gguf: tensor %s IQ2_XS element count %d is not a multiple of %d", t.Name, elems, qkK)
		}
		return payload(elems/qkK, blockIQ2XSBytes)
	case TensorIQ1_S:
		if elems%qkK != 0 {
			return 0, fmt.Errorf("gguf: tensor %s IQ1_S element count %d is not a multiple of %d", t.Name, elems, qkK)
		}
		return payload(elems/qkK, blockIQ1SBytes)
	case TensorIQ2_S:
		if elems%qkK != 0 {
			return 0, fmt.Errorf("gguf: tensor %s IQ2_S element count %d is not a multiple of %d", t.Name, elems, qkK)
		}
		return payload(elems/qkK, blockIQ2SBytes)
	case TensorIQ1_M:
		if elems%qkK != 0 {
			return 0, fmt.Errorf("gguf: tensor %s IQ1_M element count %d is not a multiple of %d", t.Name, elems, qkK)
		}
		return payload(elems/qkK, blockIQ1MBytes)
	case TensorMXFP4:
		if elems%qkMXFP4 != 0 {
			return 0, fmt.Errorf("gguf: tensor %s MXFP4 element count %d is not a multiple of %d", t.Name, elems, qkMXFP4)
		}
		return payload(elems/qkMXFP4, blockMXFP4Bytes)
	case TensorIQ4_NL:
		if elems%qkIQ4NL != 0 {
			return 0, fmt.Errorf("gguf: tensor %s IQ4_NL element count %d is not a multiple of %d", t.Name, elems, qkIQ4NL)
		}
		return payload(elems/qkIQ4NL, blockIQ4NLBytes)
	case TensorIQ3_S:
		if elems%qkIQ3S != 0 {
			return 0, fmt.Errorf("gguf: tensor %s IQ3_S element count %d is not a multiple of %d", t.Name, elems, qkIQ3S)
		}
		return payload(elems/qkIQ3S, blockIQ3SBytes)
	case TensorIQ4_XS:
		if elems%qkK != 0 {
			return 0, fmt.Errorf("gguf: tensor %s IQ4_XS element count %d is not a multiple of %d", t.Name, elems, qkK)
		}
		return payload(elems/qkK, blockIQ4XSBytes)
	case TensorIQ3_XXS:
		if elems%qkK != 0 {
			return 0, fmt.Errorf("gguf: tensor %s IQ3_XXS element count %d is not a multiple of %d", t.Name, elems, qkK)
		}
		return payload(elems/qkK, blockIQ3XXSBytes)
	case TensorQ2_0:
		if elems%128 != 0 {
			return 0, fmt.Errorf("gguf: tensor %s Q2_0 element count %d is not a multiple of 128", t.Name, elems)
		}
		return payload(elems/128, blockQ2_0Bytes)
	case TensorQ1_0:
		if elems%128 != 0 {
			return 0, fmt.Errorf("gguf: tensor %s Q1_0 element count %d is not a multiple of 128", t.Name, elems)
		}
		return payload(elems/128, blockQ1_0Bytes)
	case TensorHQQ4:
		if elems%qkHQQ4 != 0 {
			return 0, fmt.Errorf("gguf: tensor %s HQQ4 element count %d is not a multiple of %d", t.Name, elems, qkHQQ4)
		}
		return payload(elems/qkHQQ4, blockHQQ4Bytes)
	default:
		return 0, fmt.Errorf("gguf: tensor %s type %s does not have a simple payload", t.Name, t.Type)
	}
}

func tensorElems(t TensorInfo) (uint64, error) {
	if len(t.Dims) == 0 {
		return 0, fmt.Errorf("gguf: tensor %s has no dimensions", t.Name)
	}
	n := uint64(1)
	for _, d := range t.Dims {
		if d == 0 {
			return 0, fmt.Errorf("gguf: tensor %s has zero dimension", t.Name)
		}
		if n > math.MaxUint64/d {
			return 0, fmt.Errorf("gguf: tensor %s element count overflows uint64", t.Name)
		}
		n *= d
	}
	return n, nil
}

// reuseF32 returns a length-n float32 slice backed by buf when buf's capacity allows, else
// a fresh allocation. The caller overwrites every returned element, so the reused tail is
// not zeroed — and never leaks into the result, whose length is exactly n.
func reuseF32(buf []float32, n int) []float32 {
	if cap(buf) >= n {
		return buf[:n]
	}
	return make([]float32, n)
}

const (
	dequantParallelMinBlocks       = 4096
	dequantParallelBlocksPerWorker = 4096
)

func dequantParallelWorkers(blocks int) int {
	if activeParallelLoads.Load() != 0 {
		return 1
	}
	if blocks < dequantParallelMinBlocks {
		return 1
	}
	workers := dequantWorkers()
	if workers < 2 {
		return 1
	}
	maxByWork := (blocks + dequantParallelBlocksPerWorker - 1) / dequantParallelBlocksPerWorker
	if workers > maxByWork {
		workers = maxByWork
	}
	if workers > blocks {
		workers = blocks
	}
	if workers < 2 {
		return 1
	}
	return workers
}

func dequantWorkers() int {
	n := runtime.GOMAXPROCS(0)
	if n < 1 {
		return 1
	}
	return n
}

func dequantBlocks(out []float32, raw []byte, qk, blockBytes int, body func([]float32, []byte)) {
	blocks := len(out) / qk
	workers := dequantParallelWorkers(blocks)
	if workers <= 1 {
		body(out, raw)
		return
	}

	var wg sync.WaitGroup
	for w := 1; w < workers; w++ {
		lo := blocks * w / workers
		hi := blocks * (w + 1) / workers
		wg.Add(1)
		go func() {
			defer wg.Done()
			body(out[lo*qk:hi*qk], raw[lo*blockBytes:hi*blockBytes])
		}()
	}
	hi := blocks / workers
	body(out[:hi*qk], raw[:hi*blockBytes])
	wg.Wait()
}

// dequantScalarBlocks owns the shared scalar-decoder block iteration and indexing.
// body receives exactly one decoded output block and its corresponding packed bytes,
// keeping format-specific kernels local without duplicating stride arithmetic.
func dequantScalarBlocks(out []float32, raw []byte, qk, blockBytes int, body func([]float32, []byte)) {
	for block := 0; block < len(out)/qk; block++ {
		body(out[block*qk:(block+1)*qk], raw[block*blockBytes:(block+1)*blockBytes])
	}
}

// dequantKQuantBody returns the per-chunk dequant body the K-quant and IQ3_XXS load
// paths hand to dequantBlocks. It runs the SIMD arch unpack when the CPU provides it
// (arch returns true after vectorizing the chunk) and falls back to the scalar unpack
// otherwise. Wrapping arch-then-scalar as a dequantBlocks body is the #1130 seam:
// dequantBlocks fans a tensor's blocks across cores (#1102's parFor) and each core
// vectorizes its own block-aligned chunk here, so per-core SIMD throughput composes
// with across-core parallelism. Quants without an arch kernel keep passing their
// scalar body straight to dequantBlocks, so the default path is untouched where the
// SIMD feature is absent (arch is the noasm stub that always declines).
func dequantKQuantBody(arch func([]float32, []byte) bool, scalar func([]float32, []byte)) func([]float32, []byte) {
	return func(out []float32, raw []byte) {
		if !arch(out, raw) {
			scalar(out, raw)
		}
	}
}

// dequantF32 decodes a GGUF tensor's raw payload into a freshly-allocated f32 slice.
func dequantF32(t TensorInfo, raw []byte) ([]float32, error) {
	return dequantF32Into(nil, t, raw)
}

// dequantF32Into decodes a GGUF tensor's raw payload to f32, writing into scratch when it
// has the capacity (else allocating). The dequant writes every returned element for every
// supported type, so the reused buffer's prior contents never leak. The returned slice
// aliases scratch's backing array on reuse, so a caller recycling one buffer across many
// tensors MUST finish consuming the result before the next dequantF32Into overwrites it.
// Passing nil always allocates — the historical dequantF32 behavior every other caller keeps.
//
// This is the GGUF->Q8 quant-on-load page-churn fix (#440): the quant-on-load path
// dequantizes each tensor only long enough to re-quantize it, so a 27B checkpoint's 800+
// throwaway elems*4 f32 buffers — each faulting in fresh zeroed pages the GC then unmaps —
// collapse to one reused arena grown to the largest tensor.
func dequantF32Into(scratch []float32, t TensorInfo, raw []byte) ([]float32, error) {
	elems, err := tensorElems(t)
	if err != nil {
		return nil, err
	}
	if elems > uint64(math.MaxInt) {
		return nil, fmt.Errorf("gguf: tensor %s element count overflows int", t.Name)
	}
	out := reuseF32(scratch, int(elems))
	switch t.Type {
	case TensorF32:
		if err := checkFloatPayload(t, raw, len(out)*4, "f32"); err != nil {
			return nil, err
		}
		for i := range out {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
		}
	case TensorF16:
		if err := checkFloatPayload(t, raw, len(out)*2, "f16"); err != nil {
			return nil, err
		}
		for i := range out {
			out[i] = f16At(raw, i*2)
		}
	case TensorBF16:
		if err := checkFloatPayload(t, raw, len(out)*2, "bf16"); err != nil {
			return nil, err
		}
		for i := range out {
			out[i] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(raw[i*2:])) << 16)
		}
	case TensorQ4_0:
		if _, err := checkQuantPayload(t, elems, raw, qk4, blockQ4_0Bytes, "Q4_0"); err != nil {
			return nil, err
		}
		dequantQ4_0(out, raw)
	case TensorQ2_0:
		if _, err := checkQuantPayload(t, elems, raw, 128, blockQ2_0Bytes, "Q2_0"); err != nil {
			return nil, err
		}
		dequantQ2_0Scalar(out, raw)
	case TensorQ1_0:
		if _, err := checkQuantPayload(t, elems, raw, 128, blockQ1_0Bytes, "Q1_0"); err != nil {
			return nil, err
		}
		dequantQ1_0Scalar(out, raw)
	case TensorHQQ4:
		if _, err := checkQuantPayload(t, elems, raw, qkHQQ4, blockHQQ4Bytes, "HQQ4"); err != nil {
			return nil, err
		}
		dequantHQQ4Scalar(out, raw)
	case TensorQ4_1:
		if _, err := checkQuantPayload(t, elems, raw, qk4, blockQ4_1Bytes, "Q4_1"); err != nil {
			return nil, err
		}
		dequantQ4_1(out, raw)
	case TensorQ5_0:
		if _, err := checkQuantPayload(t, elems, raw, qk5, blockQ5_0Bytes, "Q5_0"); err != nil {
			return nil, err
		}
		dequantQ5_0(out, raw)
	case TensorQ5_1:
		if _, err := checkQuantPayload(t, elems, raw, qk5, blockQ5_1Bytes, "Q5_1"); err != nil {
			return nil, err
		}
		dequantQ5_1(out, raw)
	case TensorQ8_0:
		if _, err := checkQuantPayload(t, elems, raw, qk8_0, blockQ8_0Bytes, "Q8_0"); err != nil {
			return nil, err
		}
		dequantQ8_0(out, raw)
	case TensorQ2_K:
		if _, err := checkQuantPayload(t, elems, raw, qkK, blockQ2KBytes, "Q2_K"); err != nil {
			return nil, err
		}
		dequantQ2K(out, raw)
	case TensorQ3_K:
		if _, err := checkQuantPayload(t, elems, raw, qkK, blockQ3KBytes, "Q3_K"); err != nil {
			return nil, err
		}
		dequantQ3K(out, raw)
	case TensorQ4_K:
		if _, err := checkQuantPayload(t, elems, raw, qkK, blockQ4KBytes, "Q4_K"); err != nil {
			return nil, err
		}
		dequantQ4K(out, raw)
	case TensorQ5_K:
		if _, err := checkQuantPayload(t, elems, raw, qkK, blockQ5KBytes, "Q5_K"); err != nil {
			return nil, err
		}
		dequantQ5K(out, raw)
	case TensorQ6_K:
		if _, err := checkQuantPayload(t, elems, raw, qkK, blockQ6KBytes, "Q6_K"); err != nil {
			return nil, err
		}
		dequantQ6K(out, raw)
	case TensorIQ2_XXS:
		if _, err := checkQuantPayload(t, elems, raw, qkK, blockIQ2XXSBytes, "IQ2_XXS"); err != nil {
			return nil, err
		}
		dequantIQ2XXS(out, raw)
	case TensorIQ2_XS:
		if _, err := checkQuantPayload(t, elems, raw, qkK, blockIQ2XSBytes, "IQ2_XS"); err != nil {
			return nil, err
		}
		dequantIQ2XS(out, raw)
	case TensorIQ1_S:
		if _, err := checkQuantPayload(t, elems, raw, qkK, blockIQ1SBytes, "IQ1_S"); err != nil {
			return nil, err
		}
		dequantIQ1S(out, raw)
	case TensorIQ2_S:
		if _, err := checkQuantPayload(t, elems, raw, qkK, blockIQ2SBytes, "IQ2_S"); err != nil {
			return nil, err
		}
		dequantIQ2S(out, raw)
	case TensorIQ1_M:
		if _, err := checkQuantPayload(t, elems, raw, qkK, blockIQ1MBytes, "IQ1_M"); err != nil {
			return nil, err
		}
		dequantIQ1M(out, raw)
	case TensorMXFP4:
		if _, err := checkQuantPayload(t, elems, raw, qkMXFP4, blockMXFP4Bytes, "MXFP4"); err != nil {
			return nil, err
		}
		dequantMXFP4(out, raw)
	case TensorIQ4_NL:
		if _, err := checkQuantPayload(t, elems, raw, qkIQ4NL, blockIQ4NLBytes, "IQ4_NL"); err != nil {
			return nil, err
		}
		dequantIQ4NL(out, raw)
	case TensorIQ3_S:
		if _, err := checkQuantPayload(t, elems, raw, qkIQ3S, blockIQ3SBytes, "IQ3_S"); err != nil {
			return nil, err
		}
		dequantIQ3S(out, raw)
	case TensorIQ4_XS:
		if _, err := checkQuantPayload(t, elems, raw, qkK, blockIQ4XSBytes, "IQ4_XS"); err != nil {
			return nil, err
		}
		dequantIQ4XS(out, raw)
	case TensorIQ3_XXS:
		if _, err := checkQuantPayload(t, elems, raw, qkK, blockIQ3XXSBytes, "IQ3_XXS"); err != nil {
			return nil, err
		}
		dequantIQ3XXS(out, raw)
	default:
		return nil, fmt.Errorf("gguf: tensor %s type %d cannot dequantize to f32 yet", t.Name, t.Type)
	}
	return out, nil
}

// checkQuantPayload validates that a quantized tensor's raw payload is a whole number of
// blocks (elems divisible by qk) and exactly the size those blocks pack to, returning that
// expected byte count. It is the shared block-shape guard the per-type dequant cases all ran
// inline; label is the quant name in the error text (byte-identical to the inlined checks).
// checkFloatPayload verifies a non-quantized (f32/f16/bf16) tensor's raw byte count matches the
// element count its decode loop will write. want is len(out) scaled by the element byte width;
// label names the dtype in the error message.
func checkFloatPayload(t TensorInfo, raw []byte, want int, label string) error {
	if len(raw) != want {
		return fmt.Errorf("gguf: tensor %s %s payload has %d bytes, want %d", t.Name, label, len(raw), want)
	}
	return nil
}

func checkQuantPayload(t TensorInfo, elems uint64, raw []byte, qk, blockBytes uint64, label string) (int, error) {
	if elems%qk != 0 {
		return 0, fmt.Errorf("gguf: tensor %s %s element count %d is not a multiple of %d", t.Name, label, elems, qk)
	}
	want := int(elems / qk * blockBytes)
	if len(raw) != want {
		return 0, fmt.Errorf("gguf: tensor %s %s payload has %d bytes, want %d", t.Name, label, len(raw), want)
	}
	return want, nil
}

// dequantQ4_0 expands the legacy GGML Q4_0 32-element block. Each block is a
// little-endian f16 scale d followed by qk4/2 bytes of packed 4-bit codes (two
// nibbles per byte). The GGML layout (dequantize_row_q4_0) is interleaved: the low
// nibble of byte j is element j, the high nibble is element j+qk4/2, and each code is
// re-centered by -8 before scaling: y = (nibble-8)*d. This is the 4-bit sibling of
// dequantQ5_0 with no 5th high bit.
func dequantQ2_0Scalar(out []float32, raw []byte) {
	for block, base := 0, 0; block < len(out)/128; block, base = block+1, base+blockQ2_0Bytes {
		d := f16At(raw, base)
		qs := raw[base+2 : base+blockQ2_0Bytes]
		for j := 0; j < 128; j++ {
			q := (qs[j/4] >> (2 * uint(j%4))) & 0x03
			out[block*128+j] = float32(int(q)-1) * d
		}
	}
}

// dequantQ1_0Scalar expands the PrismML Q1_0 (g128) 1-bit binary block — the
// Bonsai-27B 1-bit sibling of Q2_0 (#4871): one little-endian f16 scale d
// followed by 128 contiguous 1-bit codes, eight low-to-high codes per byte.
// The code cardinality is 2 (binary, not ternary): 0=-1, 1=+1, so
// y = (2*code-1)*d. 18 bytes per 128-element block = ~1.125 bpw.
func dequantQ1_0Scalar(out []float32, raw []byte) {
	for block, base := 0, 0; block < len(out)/128; block, base = block+1, base+blockQ1_0Bytes {
		d := f16At(raw, base)
		qs := raw[base+2 : base+blockQ1_0Bytes]
		for j := 0; j < 128; j++ {
			q := (qs[j/8] >> uint(j%8)) & 1
			out[block*128+j] = float32(2*int(q)-1) * d
		}
	}
}

// dequantHQQ4Scalar expands a 4-bit HQQ (Half-Quadratic Quantization) group — the
// Bonsai VLM vision-tower quant (#4876). Unlike the ggml k-quants, HQQ reconstructs
// in the QUANTIZED domain: y = scale*(q - zero), where q is the raw 4-bit code
// (0..15) and BOTH scale and zero are learned fp16 per-group parameters. Each
// qkHQQ4=64-element group packs a little-endian f16 scale, a little-endian f16 zero,
// then qkHQQ4/2 bytes of split-half interleaved 4-bit codes — the low nibble of byte
// j is element j, the high nibble is element j+qkHQQ4/2 (the Q4_0/Q4_1 nibble order,
// NOT re-centered by -8). This is the pure HQQ dequant math (mobiusml/hqq
// Quantizer.dequantize, W_r=(W_q-zero)*scale); the group size (64), the fp16
// scale/zero encoding, and the GGUF block/tag layout are the invalidating
// assumptions to confirm against a real Bonsai mmproj header (see TensorHQQ4).
func dequantHQQ4Scalar(out []float32, raw []byte) {
	for block := 0; block < len(out)/qkHQQ4; block++ {
		base := block * blockHQQ4Bytes
		scale := f16At(raw, base)
		zero := f16At(raw, base+2)
		qs := raw[base+4 : base+blockHQQ4Bytes]
		yi := block * qkHQQ4
		for j := 0; j < qkHQQ4/2; j++ {
			q0 := float32(int(qs[j] & 0x0f))
			q1 := float32(int(qs[j] >> 4))
			out[yi+j] = scale * (q0 - zero)
			out[yi+j+qkHQQ4/2] = scale * (q1 - zero)
		}
	}
}

// IQ1/IQ2 decoding is owned by internal/model so GGUF load-time dequantization and the
// resident QuantBuilder matvec path share one pinned, fak-native implementation.
func dequantIQ2XXS(out []float32, raw []byte) { model.DequantIQ2XXS(out, raw) }
func dequantIQ2XS(out []float32, raw []byte)  { model.DequantIQ2XS(out, raw) }
func dequantIQ2S(out []float32, raw []byte)   { model.DequantIQ2S(out, raw) }
func dequantIQ1S(out []float32, raw []byte)   { model.DequantIQ1S(out, raw) }
func dequantIQ1M(out []float32, raw []byte)   { model.DequantIQ1M(out, raw) }

func dequantQ4_0(out []float32, raw []byte) {
	dequantBlocks(out, raw, qk4, blockQ4_0Bytes, dequantQ4_0Scalar)
}

func dequantQ4_0Scalar(out []float32, raw []byte) {
	for block := 0; block < len(out)/qk4; block++ {
		base := block * blockQ4_0Bytes
		d := f16At(raw, base)
		qs := raw[base+2 : base+blockQ4_0Bytes]
		yi := block * qk4
		for j := 0; j < qk4/2; j++ {
			x0 := int(qs[j]&0x0f) - 8
			x1 := int(qs[j]>>4) - 8
			out[yi+j] = float32(x0) * d
			out[yi+j+qk4/2] = float32(x1) * d
		}
	}
}

// kvaluesMXFP4 maps a 4-bit E2M1 (FP4) code to its value, stored as 2x the real
// FP4 magnitude so the table is exact integers; the ×0.5 that restores the true
// E2M1 values {0,.5,1,1.5,2,3,4,6} is folded into the E8M0 scale by e8m0ToF32Half
// (which yields 2^(e-128) rather than 2^(e-127)). This matches GGML's
// kvalues_mxfp4 + GGML_E8M0_TO_FP32_HALF pairing for gpt-oss weights.
var kvaluesMXFP4 = [16]float32{0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12}

// e8m0ToF32Half decodes an E8M0 shared-exponent scale byte to 2^(e-128) — the
// half-scaled power that pairs with the doubled kvaluesMXFP4 table so that
// kvaluesMXFP4[code] * e8m0ToF32Half(e) == fp4(code) * 2^(e-127).
func e8m0ToF32Half(e uint8) float32 {
	return float32(math.Ldexp(1, int(e)-128))
}

// dequantMXFP4 expands the MXFP4 (gpt-oss) 32-element block: a 1-byte E8M0 shared
// scale followed by qkMXFP4/2 bytes of packed 4-bit E2M1 codes. The GGML layout
// (dequantize_row_mxfp4) interleaves like Q4_0 — the low nibble of byte j is
// element j, the high nibble is element j+qkMXFP4/2 — and each code indexes the
// E2M1 value table scaled by the block's half-scaled E8M0 exponent.
func dequantMXFP4(out []float32, raw []byte) {
	dequantBlocks(out, raw, qkMXFP4, blockMXFP4Bytes, dequantMXFP4Scalar)
}

func dequantMXFP4Scalar(out []float32, raw []byte) {
	for block := 0; block < len(out)/qkMXFP4; block++ {
		base := block * blockMXFP4Bytes
		d := e8m0ToF32Half(raw[base])
		qs := raw[base+1 : base+blockMXFP4Bytes]
		yi := block * qkMXFP4
		for j := 0; j < qkMXFP4/2; j++ {
			out[yi+j] = kvaluesMXFP4[qs[j]&0x0f] * d
			out[yi+j+qkMXFP4/2] = kvaluesMXFP4[qs[j]>>4] * d
		}
	}
}

// kvaluesIQ4NL is GGML's non-linear 4-bit codebook (kvalues_iq4nl): a 4-bit code
// indexes one of 16 fixed int8 reconstruction levels, spaced non-uniformly to put
// finer resolution near zero. IQ4_NL and IQ4_XS share this single table — they differ
// only in how the per-block scale is encoded, not in the codebook itself.
var kvaluesIQ4NL = [16]float32{-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113}

// dequantIQ4NL expands the GGML IQ4_NL 32-element block: a little-endian f16 scale d
// followed by qkIQ4NL/2 bytes of packed 4-bit codes. The GGML layout
// (dequantize_row_iq4_nl) is sequential — byte j holds element 2j in its low nibble and
// element 2j+1 in its high nibble — and each code indexes kvaluesIQ4NL before the block
// scale: y = d*kvaluesIQ4NL[code].
func dequantIQ4NL(out []float32, raw []byte) {
	dequantBlocks(out, raw, qkIQ4NL, blockIQ4NLBytes, dequantIQ4NLScalar)
}

func dequantIQ4NLScalar(out []float32, raw []byte) {
	for block := 0; block < len(out)/qkIQ4NL; block++ {
		base := block * blockIQ4NLBytes
		d := f16At(raw, base)
		qs := raw[base+2 : base+blockIQ4NLBytes]
		yi := block * qkIQ4NL
		for j := 0; j < qkIQ4NL/2; j++ {
			out[yi+2*j] = d * kvaluesIQ4NL[qs[j]&0x0f]
			out[yi+2*j+1] = d * kvaluesIQ4NL[qs[j]>>4]
		}
	}
}

// dequantIQ4XS expands the GGML IQ4_XS 256-element super-block: a little-endian f16
// super-scale d, a little-endian u16 high-bit scale field scales_h, qkK/64 low-bit scale
// bytes scales_l, then qkK/2 bytes of packed 4-bit codes. The GGML layout
// (dequantize_row_iq4_xs) splits the super-block into eight 32-element sub-blocks; each
// sub-block ib carries a 6-bit scale ls — its low 4 bits from a scales_l nibble, its high
// 2 bits from a scales_h field — applied as dl = d*(ls-32). Within a sub-block byte j holds
// element j in its low nibble and element j+16 in its high nibble: y = dl*kvaluesIQ4NL[code].
func dequantIQ4XS(out []float32, raw []byte) {
	dequantBlocks(out, raw, qkK, blockIQ4XSBytes, dequantIQ4XSScalar)
}

func dequantIQ4XSScalar(out []float32, raw []byte) {
	for block := 0; block < len(out)/qkK; block++ {
		base := block * blockIQ4XSBytes
		d := f16At(raw, base)
		scalesH := binary.LittleEndian.Uint16(raw[base+2:])
		scalesL := raw[base+4 : base+4+qkK/64]
		qs := raw[base+4+qkK/64 : base+blockIQ4XSBytes]
		yi := block * qkK
		for ib := 0; ib < qkK/32; ib++ {
			lo := int(scalesL[ib/2]>>(4*uint(ib%2))) & 0x0f
			hi := int((scalesH >> (2 * uint(ib))) & 3)
			ls := lo | hi<<4
			dl := d * float32(ls-32)
			sub := qs[ib*16 : ib*16+16]
			off := yi + ib*32
			for j := 0; j < 16; j++ {
				out[off+j] = dl * kvaluesIQ4NL[sub[j]&0x0f]
				out[off+j+16] = dl * kvaluesIQ4NL[sub[j]>>4]
			}
		}
	}
}

// dequantIQ3XXS expands the GGML IQ3_XXS 256-element super-block (dequantize_row_iq3_xxs):
// one f16 super-scale d, then qkK/4=64 grid-index bytes and qkK/8=32 scale/sign bytes. Each of
// the eight 32-element sub-blocks reads a u32 (top 4 bits = scale nibble, low 28 bits = four
// 7-bit sign selectors) and 8 grid-index bytes. The sub-block scale is db = d*(0.5+nibble)*0.5.
// For each of the 4 lane-pairs l, two grid indices select two iq3xxsGrid entries; each entry is
// a uint32 whose 4 little-endian bytes are 4 magnitudes; the 7-bit selector ksignsIQ2XS[sel]
// gives an 8-bit sign mask (bit j flips output j). Layout matches ggml exactly so the f32 is
// bit-faithful to llama.cpp's IQ3_XXS dequant.
func dequantIQ3XXS(out []float32, raw []byte) {
	dequantBlocks(out, raw, qkK, blockIQ3XXSBytes, dequantKQuantBody(dequantIQ3XXSArch, dequantIQ3XXSScalar))
}

func dequantIQ3XXSScalar(out []float32, raw []byte) {
	dequantScalarBlocks(out, raw, qkK, blockIQ3XXSBytes, func(out []float32, raw []byte) {
		d := f16At(raw, 0)
		qs := raw[2 : 2+qkK/4]                 // 64 grid-index bytes
		sas := raw[2+qkK/4 : blockIQ3XXSBytes] // 32 scale/sign bytes (8 u32)
		for ib32 := 0; ib32 < qkK/32; ib32++ {
			aux32 := binary.LittleEndian.Uint32(sas[4*ib32:])
			db := d * (0.5 + float32(aux32>>28)) * 0.5
			gi := ib32 * 8 // 8 grid-index bytes per sub-block
			off := ib32 * 32
			for l := 0; l < 4; l++ {
				signs := ksignsIQ2XS[(aux32>>(7*uint(l)))&127]
				g1 := iq3xxsGrid[qs[gi+2*l+0]]
				g2 := iq3xxsGrid[qs[gi+2*l+1]]
				for j := 0; j < 4; j++ {
					s1 := float32(1)
					if signs&(1<<uint(j)) != 0 {
						s1 = -1
					}
					s2 := float32(1)
					if signs&(1<<uint(j+4)) != 0 {
						s2 = -1
					}
					out[off+l*8+j] = db * float32(byte(g1>>(8*uint(j)))) * s1
					out[off+l*8+j+4] = db * float32(byte(g2>>(8*uint(j)))) * s2
				}
			}
		}
	})
}

// dequantQ4_1 expands the legacy GGML Q4_1 32-element block: a little-endian f16
// scale d, then a little-endian f16 min m, then qk4/2 bytes of packed 4-bit codes.
// The GGML layout (dequantize_row_q4_1) keeps the same low/high-nibble interleave as
// Q4_0 but the codes are NOT re-centered — they carry an affine min: y = nibble*d + m.
func dequantQ4_1(out []float32, raw []byte) {
	dequantBlocks(out, raw, qk4, blockQ4_1Bytes, dequantQ4_1Scalar)
}

func dequantQ4_1Scalar(out []float32, raw []byte) {
	dequantScalarBlocks(out, raw, qk4, blockQ4_1Bytes, func(out []float32, raw []byte) {
		d := f16At(raw, 0)
		m := f16At(raw, 2)
		qs := raw[4:blockQ4_1Bytes]
		for j := 0; j < qk4/2; j++ {
			x0 := int(qs[j] & 0x0f)
			x1 := int(qs[j] >> 4)
			out[j] = float32(x0)*d + m
			out[j+qk4/2] = float32(x1)*d + m
		}
	})
}

func dequantQ5_0(out []float32, raw []byte) {
	dequantBlocks(out, raw, qk5, blockQ5_0Bytes, dequantQ5_0Scalar)
}

func dequantQ5_0Scalar(out []float32, raw []byte) {
	dequantScalarBlocks(out, raw, qk5, blockQ5_0Bytes, func(out []float32, raw []byte) {
		d := f16At(raw, 0)
		qh := binary.LittleEndian.Uint32(raw[2:])
		qs := raw[6:blockQ5_0Bytes]
		for j := 0; j < qk5/2; j++ {
			xh0 := byte(((qh >> uint(j)) << 4) & 0x10)
			xh1 := byte((qh >> uint(j+12)) & 0x10)
			x0 := int((qs[j]&0x0f)|xh0) - 16
			x1 := int((qs[j]>>4)|xh1) - 16
			out[j] = float32(x0) * d
			out[j+qk5/2] = float32(x1) * d
		}
	})
}

func dequantQ5_1(out []float32, raw []byte) {
	dequantBlocks(out, raw, qk5, blockQ5_1Bytes, dequantQ5_1Scalar)
}

func dequantQ5_1Scalar(out []float32, raw []byte) {
	dequantScalarBlocks(out, raw, qk5, blockQ5_1Bytes, func(out []float32, raw []byte) {
		d := f16At(raw, 0)
		m := f16At(raw, 2)
		qh := binary.LittleEndian.Uint32(raw[4:])
		qs := raw[8:blockQ5_1Bytes]
		for j := 0; j < qk5/2; j++ {
			xh0 := byte(((qh >> uint(j)) << 4) & 0x10)
			xh1 := byte((qh >> uint(j+12)) & 0x10)
			x0 := int((qs[j] & 0x0f) | xh0)
			x1 := int((qs[j] >> 4) | xh1)
			out[j] = float32(x0)*d + m
			out[j+qk5/2] = float32(x1)*d + m
		}
	})
}

func dequantQ2K(out []float32, raw []byte) {
	dequantBlocks(out, raw, qkK, blockQ2KBytes, dequantQ2KScalar)
}

func dequantQ2KScalar(out []float32, raw []byte) {
	dequantScalarBlocks(out, raw, qkK, blockQ2KBytes, func(out []float32, raw []byte) {
		scales := raw[:qkK/16]
		q := raw[qkK/16 : qkK/16+qkK/4]
		dm := qkK/16 + qkK/4
		d := f16At(raw, dm)
		min := f16At(raw, dm+2)
		qi := 0
		is := 0
		for n := 0; n < qkK; n += 128 {
			shift := uint(0)
			for j := 0; j < 4; j++ {
				sc := scales[is]
				is++
				dl, ml := d*float32(sc&0x0f), min*float32(sc>>4)
				for l := 0; l < 16; l++ {
					out[n+j*32+l] = dl*float32((q[qi+l]>>shift)&3) - ml
				}

				sc = scales[is]
				is++
				dl, ml = d*float32(sc&0x0f), min*float32(sc>>4)
				for l := 0; l < 16; l++ {
					out[n+j*32+16+l] = dl*float32((q[qi+16+l]>>shift)&3) - ml
				}
				shift += 2
			}
			qi += 32
		}
	})
}

func dequantQ3K(out []float32, raw []byte) {
	dequantBlocks(out, raw, qkK, blockQ3KBytes, dequantQ3KScalar)
}

func dequantQ3KScalar(out []float32, raw []byte) {
	dequantScalarBlocks(out, raw, qkK, blockQ3KBytes, func(out []float32, raw []byte) {
		hmask := raw[:qkK/8]
		q := raw[qkK/8 : qkK/8+qkK/4]
		scales := unpackQ3KScales(raw[qkK/8+qkK/4 : qkK/8+qkK/4+kScaleSize])
		d := f16At(raw, blockQ3KBytes-2)
		qi := 0
		is := 0
		mask := byte(1)
		for n := 0; n < qkK; n += 128 {
			shift := uint(0)
			for j := 0; j < 4; j++ {
				dl := d * float32(scales[is]-32)
				is++
				for l := 0; l < 16; l++ {
					code := int8((q[qi+l] >> shift) & 3)
					if hmask[l]&mask == 0 {
						code -= 4
					}
					out[n+j*32+l] = dl * float32(code)
				}

				dl = d * float32(scales[is]-32)
				is++
				for l := 0; l < 16; l++ {
					code := int8((q[qi+16+l] >> shift) & 3)
					if hmask[16+l]&mask == 0 {
						code -= 4
					}
					out[n+j*32+16+l] = dl * float32(code)
				}
				shift += 2
				mask <<= 1
			}
			qi += 32
		}
	})
}

func unpackQ3KScales(raw []byte) [16]int8 {
	const (
		kmask1 = uint32(0x03030303)
		kmask2 = uint32(0x0f0f0f0f)
	)
	aux0 := binary.LittleEndian.Uint32(raw[0:4])
	aux1 := binary.LittleEndian.Uint32(raw[4:8])
	aux2 := binary.LittleEndian.Uint32(raw[8:12])
	tmp := aux2
	words := [4]uint32{
		(aux0 & kmask2) | (((tmp >> 0) & kmask1) << 4),
		(aux1 & kmask2) | (((tmp >> 2) & kmask1) << 4),
		((aux0 >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4),
		((aux1 >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4),
	}
	var scales [16]int8
	for i, word := range words {
		for j := 0; j < 4; j++ {
			scales[i*4+j] = int8(byte(word >> (8 * j)))
		}
	}
	return scales
}

func dequantQ4K(out []float32, raw []byte) {
	dequantBlocks(out, raw, qkK, blockQ4KBytes, dequantKQuantBody(dequantQ4KArch, dequantQ4KScalar))
}

func dequantQ4KScalar(out []float32, raw []byte) {
	dequantScalarBlocks(out, raw, qkK, blockQ4KBytes, func(out []float32, raw []byte) {
		d := f16At(raw, 0)
		min := f16At(raw, 2)
		scales := raw[4 : 4+kScaleSize]
		q := raw[4+kScaleSize : blockQ4KBytes]
		qi := 0
		is := 0
		for j := 0; j < qkK; j += 64 {
			d1, m1, d2, m2 := scaleMinPairK4(d, min, is, scales)
			for l := 0; l < 32; l++ {
				out[j+l] = d1*float32(q[qi+l]&0x0f) - m1
			}
			for l := 0; l < 32; l++ {
				out[j+32+l] = d2*float32(q[qi+l]>>4) - m2
			}
			qi += 32
			is += 2
		}
	})
}

// scaleMinPairK4 decodes the two 6-bit (scale,min) sub-block fields at indices is and is+1
// and folds them with the super-block d/min into the (d1,m1,d2,m2) the Q4_K and Q5_K kernels
// both consume per 64-element stride. It is pure code motion of the identical 4-line preamble
// shared by dequantQ4K and dequantQ5K.
func scaleMinPairK4(d, min float32, is int, scales []byte) (d1, m1, d2, m2 float32) {
	sc, m := model.GetScaleMinK4(is, scales)
	d1, m1 = d*float32(sc), min*float32(m)
	sc, m = model.GetScaleMinK4(is+1, scales)
	d2, m2 = d*float32(sc), min*float32(m)
	return d1, m1, d2, m2
}

func dequantQ5K(out []float32, raw []byte) {
	dequantBlocks(out, raw, qkK, blockQ5KBytes, dequantKQuantBody(dequantQ5KArch, dequantQ5KScalar))
}

func dequantQ5KScalar(out []float32, raw []byte) {
	dequantScalarBlocks(out, raw, qkK, blockQ5KBytes, func(out []float32, raw []byte) {
		d := f16At(raw, 0)
		min := f16At(raw, 2)
		scales := raw[4 : 4+kScaleSize]
		qh := raw[4+kScaleSize : 4+kScaleSize+qkK/8]
		ql := raw[4+kScaleSize+qkK/8 : blockQ5KBytes]
		qi := 0
		is := 0
		u1, u2 := byte(1), byte(2)
		for j := 0; j < qkK; j += 64 {
			d1, m1, d2, m2 := scaleMinPairK4(d, min, is, scales)
			for l := 0; l < 32; l++ {
				hi := byte(0)
				if qh[l]&u1 != 0 {
					hi = 16
				}
				out[j+l] = d1*float32((ql[qi+l]&0x0f)+hi) - m1
			}
			for l := 0; l < 32; l++ {
				hi := byte(0)
				if qh[l]&u2 != 0 {
					hi = 16
				}
				out[j+32+l] = d2*float32((ql[qi+l]>>4)+hi) - m2
			}
			qi += 32
			is += 2
			u1 <<= 2
			u2 <<= 2
		}
	})
}

func dequantQ6K(out []float32, raw []byte) {
	dequantBlocks(out, raw, qkK, blockQ6KBytes, dequantKQuantBody(dequantQ6KArch, dequantQ6KScalar))
}

func dequantQ8_0(out []float32, raw []byte) {
	dequantBlocks(out, raw, qk8_0, blockQ8_0Bytes, dequantQ8_0Scalar)
}

func dequantQ8_0Scalar(out []float32, raw []byte) {
	dequantScalarBlocks(out, raw, qk8_0, blockQ8_0Bytes, func(out []float32, raw []byte) {
		d := f16At(raw, 0)
		for j := 0; j < qk8_0; j++ {
			out[j] = float32(int8(raw[2+j])) * d
		}
	})
}

func dequantQ6KScalar(out []float32, raw []byte) {
	dequantScalarBlocks(out, raw, qkK, blockQ6KBytes, func(out []float32, raw []byte) {
		ql := raw[:qkK/2]
		qh := raw[qkK/2 : qkK/2+qkK/4]
		scales := raw[qkK/2+qkK/4 : qkK/2+qkK/4+qkK/16]
		d := f16At(raw, blockQ6KBytes-2)
		qlOff, qhOff, scOff := 0, 0, 0
		for n := 0; n < qkK; n += 128 {
			for l := 0; l < 32; l++ {
				is := l / 16
				q1 := int8((ql[qlOff+l+0]&0x0f)|(((qh[qhOff+l]>>0)&3)<<4)) - 32
				q2 := int8((ql[qlOff+l+32]&0x0f)|(((qh[qhOff+l]>>2)&3)<<4)) - 32
				q3 := int8((ql[qlOff+l+0]>>4)|(((qh[qhOff+l]>>4)&3)<<4)) - 32
				q4 := int8((ql[qlOff+l+32]>>4)|(((qh[qhOff+l]>>6)&3)<<4)) - 32
				out[n+l+0] = d * float32(int8(scales[scOff+is+0])) * float32(q1)
				out[n+l+32] = d * float32(int8(scales[scOff+is+2])) * float32(q2)
				out[n+l+64] = d * float32(int8(scales[scOff+is+4])) * float32(q3)
				out[n+l+96] = d * float32(int8(scales[scOff+is+6])) * float32(q4)
			}
			qlOff += 64
			qhOff += 32
			scOff += 8
		}
	})
}

// f16At decodes the little-endian IEEE-754 half stored at raw[off:off+2] into a float32.
// It is the single reader for every per-block GGUF scale/min field — the dozen-plus
// dequant kernels all read their f16 scales through here so the conversion lives in one
// place. Behavior is identical to the inlined
// f16At(raw, off) it replaces.
func f16At(raw []byte, off int) float32 {
	return math.Float32frombits(model.F16BitsToF32Bits(binary.LittleEndian.Uint16(raw[off:])))
}

type countingReader struct {
	r io.Reader
	n int64
}

func (r *countingReader) readFull(b []byte) error {
	if _, err := io.ReadFull(r.r, b); err != nil {
		return err
	}
	r.n += int64(len(b))
	return nil
}

// readLE reads size little-endian bytes from r (advancing its byte counter) and decodes them
// with dec. It is the shared body of the u32/u64 fixed-width readers; the zero value of T is
// returned on a short read.
func readLE[T any](r *countingReader, size int, dec func([]byte) T) (T, error) {
	b := make([]byte, size)
	if err := r.readFull(b); err != nil {
		var zero T
		return zero, err
	}
	return dec(b), nil
}

func (r *countingReader) u32() (uint32, error) {
	return readLE(r, 4, binary.LittleEndian.Uint32)
}

func (r *countingReader) u64() (uint64, error) {
	return readLE(r, 8, binary.LittleEndian.Uint64)
}

func (r *countingReader) str() (string, error) {
	n, err := r.u64()
	if err != nil {
		return "", err
	}
	if n > maxStringBytes {
		return "", fmt.Errorf("string too large: %d bytes", n)
	}
	b := make([]byte, int(n))
	if err := r.readFull(b); err != nil {
		return "", err
	}
	return string(b), nil
}

func (r *countingReader) valueType() (ValueType, error) {
	u, err := r.u32()
	return ValueType(u), err
}

func (r *countingReader) value(typ ValueType) (Value, error) {
	switch typ {
	case TypeUint8:
		var b [1]byte
		if err := r.readFull(b[:]); err != nil {
			return Value{}, err
		}
		return Value{Type: typ, Value: b[0]}, nil
	case TypeInt8:
		var b [1]byte
		if err := r.readFull(b[:]); err != nil {
			return Value{}, err
		}
		return Value{Type: typ, Value: int8(b[0])}, nil
	case TypeUint16:
		var b [2]byte
		if err := r.readFull(b[:]); err != nil {
			return Value{}, err
		}
		return Value{Type: typ, Value: binary.LittleEndian.Uint16(b[:])}, nil
	case TypeInt16:
		var b [2]byte
		if err := r.readFull(b[:]); err != nil {
			return Value{}, err
		}
		return Value{Type: typ, Value: int16(binary.LittleEndian.Uint16(b[:]))}, nil
	case TypeUint32:
		v, err := r.u32()
		return Value{Type: typ, Value: v}, err
	case TypeInt32:
		v, err := r.u32()
		return Value{Type: typ, Value: int32(v)}, err
	case TypeFloat32:
		v, err := r.u32()
		return Value{Type: typ, Value: math.Float32frombits(v)}, err
	case TypeBool:
		var b [1]byte
		if err := r.readFull(b[:]); err != nil {
			return Value{}, err
		}
		if b[0] > 1 {
			return Value{}, fmt.Errorf("invalid bool byte %d", b[0])
		}
		return Value{Type: typ, Value: b[0] == 1}, nil
	case TypeString:
		s, err := r.str()
		return Value{Type: typ, Value: s}, err
	case TypeArray:
		elem, err := r.valueType()
		if err != nil {
			return Value{}, err
		}
		n, err := r.u64()
		if err != nil {
			return Value{}, err
		}
		if n > uint64(math.MaxInt) {
			return Value{}, fmt.Errorf("array too large: %d elements", n)
		}
		items := make([]Value, int(n))
		for i := range items {
			items[i], err = r.value(elem)
			if err != nil {
				return Value{}, fmt.Errorf("array element %d: %w", i, err)
			}
		}
		return Value{Type: typ, Value: items}, nil
	case TypeUint64:
		v, err := r.u64()
		return Value{Type: typ, Value: v}, err
	case TypeInt64:
		v, err := r.u64()
		return Value{Type: typ, Value: int64(v)}, err
	case TypeFloat64:
		v, err := r.u64()
		return Value{Type: typ, Value: math.Float64frombits(v)}, err
	default:
		return Value{}, fmt.Errorf("unsupported value type %d", typ)
	}
}

// dequantIQ3S expands GGML IQ3_S super-blocks. Each 256-value block stores one
// f16 scale, 64 low grid-index bytes, 8 high-bit bytes, 32 sign masks, and four
// packed 4-bit subscales. The indexing and sign order intentionally mirror
// llama.cpp dequantize_row_iq3_s so that file-format parity has one oracle.
func dequantIQ3S(out []float32, raw []byte) {
	const (
		qsOffset     = 2
		qhOffset     = qsOffset + 64
		signsOffset  = qhOffset + 8
		scalesOffset = signsOffset + 32
	)
	dequantScalarBlocks(out, raw, qkIQ3S, blockIQ3SBytes, func(out []float32, raw []byte) {
		d := f16At(raw, 0)
		qs := raw[qsOffset:qhOffset]
		qh := raw[qhOffset:signsOffset]
		signs := raw[signsOffset:scalesOffset]
		scales := raw[scalesOffset:blockIQ3SBytes]
		for pair := 0; pair < 4; pair++ {
			scaleByte := scales[pair]
			for half := 0; half < 2; half++ {
				ib32 := pair*2 + half
				nibble := scaleByte & 0x0f
				if half != 0 {
					nibble = scaleByte >> 4
				}
				db := d * float32(1+2*int(nibble))
				qBase := ib32 * 8
				high := qh[ib32]
				signBase := ib32 * 4
				outBase := ib32 * 32
				for group := 0; group < 4; group++ {
					idx1 := uint16(qs[qBase+2*group]) | (uint16(high)<<uint(8-2*group))&0x100
					idx2 := uint16(qs[qBase+2*group+1]) | (uint16(high)<<uint(7-2*group))&0x100
					grid1, grid2 := iq3SGrid[idx1], iq3SGrid[idx2]
					sign := signs[signBase+group]
					for j := 0; j < 4; j++ {
						v1 := float32(byte(grid1 >> uint(8*j)))
						v2 := float32(byte(grid2 >> uint(8*j)))
						if sign&(1<<uint(j)) != 0 {
							v1 = -v1
						}
						if sign&(1<<uint(j+4)) != 0 {
							v2 = -v2
						}
						out[outBase+group*8+j] = db * v1
						out[outBase+group*8+j+4] = db * v2
					}
				}
			}
		}
	})
}
