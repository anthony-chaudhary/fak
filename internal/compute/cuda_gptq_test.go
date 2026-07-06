//go:build cuda

package compute

import (
	"encoding/binary"
	"testing"
)

// cuda_gptq_test.go — the `-tags cuda` witness for issue #3030: the native packed GPTQ device
// GEMV (GPTQMatMul, kernel fcuda_gptq_gemv) gets the SAME cpuref-parity gate the Q4_K (#485) and
// AWQ (#926) 4-bit dequant-fused lanes use — argmax-exact + cosine ≥ cudaGPTQCosineMin. #300
// shipped the CPU-resident GPTQ loader/session (internal/model/gptq.go); this closes the GPU
// remainder: a native packed GPTQ path for Llama/Mistral-shaped projection/head weights that runs
// on the device and matches the CPU-resident oracle within the recorded 4-bit floor.
//
// GPTQ is AutoGPTQ/GPTQModel's int32-packed weight-only format: 4/8-bit codes packed pack=32/bits
// per int32 along the INPUT dim (qweight [in/pack, out]), per-GROUP int32-packed zero-points
// (qzeros [nGroups, out/pack]), per-group f32 scales [nGroups, out], and the zero+1 convention
// (weight[o,i] = (code(i,o) - (zero(g,o)+1)) · scale[g,o]). The kernel dequant-fuses the unpack
// into the GEMV tile and accumulates in F32. The reference is an f32 dequant of the SAME packed
// bytes (dequantGPTQWeight), so this gate isolates the device tile's arithmetic — reduction-order
// drift against the host GPTQ dequant — NOT the true-f32→4-bit reconstruction error (the same
// isolation discipline as the Q4_K / AWQ gates, and it mirrors internal/model/gptq.go bit-for-bit).
//
// RESIDENCY NOTE: like the AWQ witness, the packed quant tensors ride the Q4_K raw-byte upload
// channel (cb.Upload(newQ4KHost(...), Q4_K)) — a straight H2D of the little-endian int32 bytes whose
// Q4_K dtype label is COSMETIC: GPTQMatMul reads only the device pointers + the explicit
// out/in/bits/groupSize/nGroups dims, never the dtype/QuantSpec, so the bytes reach the kernel
// verbatim. A future production uploadGPTQ would reuse the same dallocWeight primitive with honest
// I4/I32 labels. This witness exercises the grouped (desc_act-off) path; the optional g_idx pointer
// is NULL-checked in the kernel and Go binding for the desc_act path (a follow-up witness).
//
// HARDWARE: the realized cosine and the decode-throughput-vs-llama.cpp comparison need a CUDA node —
// the explicit residual of this build+commit handoff (the win32 dev host has no CUDA toolkit / GPU).
// On the dev host these skip cleanly via cudaOrSkip, fail-closed, naming the missing runner. Run
// them on a GPU node:  go test -tags cuda ./internal/compute -run TestCUDAGPTQ -v
//                      go test -tags cuda ./internal/compute -run x -bench BenchmarkCUDAGPTQ -benchmem
// The Go here type-checks under `go vet -tags cuda`.

// u32ToBytesLE serializes an int32-packed slice to little-endian bytes for the raw-byte upload
// channel. The device (x86/CUDA) is little-endian, so the uint32 values reproduce verbatim when
// the kernel re-reads the buffer as uint32_t*.
func u32ToBytesLE(v []uint32) []byte {
	b := make([]byte, len(v)*4)
	for i, x := range v {
		binary.LittleEndian.PutUint32(b[i*4:], x)
	}
	return b
}

// u32 advances the deterministic lcg and returns its high 32 bits — a valid packed GPTQ word
// (every 32-bit pattern packs `pack` in-range 4/8-bit codes, since pack·bits == 32 exactly).
func (s *lcg) u32() uint32 {
	*s = *s*6364136223846793005 + 1442695040888963407
	return uint32(*s >> 32)
}

// randGPTQ authors random but VALID packed GPTQ operands for a [out,in] weight. Because pack·bits
// == 32 exactly (4-bit: 8×4, 8-bit: 4×8), EVERY uint32 is a valid packing of `pack` in-range codes,
// so qweight/qzeros are just random uint32 words. scales are small positive per-group f32.
func randGPTQ(g *lcg, out, in, bits, groupSize int) (qweight, qzeros []uint32, scales []float32, nGroups int) {
	pack := 32 / bits
	nGroups = (in + groupSize - 1) / groupSize
	qweight = make([]uint32, (in/pack)*out)
	for i := range qweight {
		qweight[i] = g.u32()
	}
	qzeros = make([]uint32, nGroups*(out/pack))
	for i := range qzeros {
		qzeros[i] = g.u32()
	}
	scales = make([]float32, nGroups*out)
	for i := range scales {
		scales[i] = 0.01 + absf(g.f())*0.03
	}
	return qweight, qzeros, scales, nGroups
}

// dequantGPTQWeight dequantizes a whole [out,in] packed GPTQ weight to f32 (row-major, w[o*in+i]),
// mirroring the device k_gptq_gemv unpack — and internal/model/gptq.go's qt.weight() — bit-for-bit:
// group = i/groupSize (clamped); code = (qweight[(i/pack)*out+o] >> (i%pack·bits)) & mask;
// zero = ((qzeros[g·out/pack + o/pack] >> (o%pack·bits)) & mask) + 1; w = (code - zero)·scale[g·out+o].
// This is the f32 Reference the device GPTQ GEMV is held to.
func dequantGPTQWeight(qweight, qzeros []uint32, scales []float32, out, in, bits, groupSize, nGroups int) []float32 {
	pack := 32 / bits
	mask := uint32(1<<uint(bits)) - 1
	outPack := out / pack
	w := make([]float32, out*in)
	for o := 0; o < out; o++ {
		for i := 0; i < in; i++ {
			g := i / groupSize
			if g >= nGroups {
				g = nGroups - 1
			}
			wv := qweight[(i/pack)*out+o]
			code := (wv >> (uint(i%pack) * uint(bits))) & mask
			zv := qzeros[g*outPack+o/pack]
			zero := ((zv >> (uint(o%pack) * uint(bits))) & mask) + 1
			w[o*in+i] = (float32(int32(code)) - float32(int32(zero))) * scales[g*out+o]
		}
	}
	return w
}

// uploadGPTQResident stages the packed GPTQ operands resident: qweight and qzeros ride the Q4_K
// raw-byte channel (see the file's RESIDENCY NOTE for why the cosmetic Q4_K label is correct),
// scales go through the F32 upload. Returns (qweight, qzeros, scales) device tensors.
func uploadGPTQResident(cb *cudaBackend, qweight, qzeros []uint32, scales []float32, out, in, pack, nGroups int) (Tensor, Tensor, Tensor) {
	wW := cb.Upload(newQ4KHost(cb, []int{in / pack, out}, u32ToBytesLE(qweight)), Q4_K)
	zW := cb.Upload(newQ4KHost(cb, []int{nGroups, out / pack}, u32ToBytesLE(qzeros)), Q4_K)
	sW := cb.Upload(NewF32(cb, []int{nGroups, out}, scales), F32)
	return wW, zW, sW
}

// TestCUDAGPTQMatMulApproxMatchesRef — native packed GPTQ 4-bit decode GEMV with the dequant fused
// into the tile, vs the cpuref f32 matmul over an f32 dequant of the SAME packed operands. Its OWN
// gate instance: argmax-exact (activation aligned to the dominant channel) + cosine over the
// non-dominant channels ≥ the RECORDED cudaGPTQCosineMin. Isolates the device GPTQ tile: a wrong
// code/zero unpack, group index, or scale apply collapses it.
func TestCUDAGPTQMatMulApproxMatchesRef(t *testing.T) {
	cb := cudaOrSkip(t)
	ref := Default()
	var seed lcg = 0x6712
	g := &seed
	out, in, bits, groupSize := 320, 256, 4, 128 // in/pack=32, out/pack=40, nGroups=2
	pack := 32 / bits
	qweight, qzeros, scales, nGroups := randGPTQ(g, out, in, bits, groupSize)
	w := dequantGPTQWeight(qweight, qzeros, scales, out, in, bits, groupSize, nGroups) // f32 reference weight
	target := dominantRow(w, out, in)
	x := alignActToRow(w, out, in, target)

	yRef := ref.Read(ref.MatMul(mkResident(ref, []int{out, in}, w), mkResident(ref, []int{in}, x)))

	wW, zW, sW := uploadGPTQResident(cb, qweight, qzeros, scales, out, in, pack, nGroups)
	yCu := cb.Read(cb.GPTQMatMul(wW, zW, sW, Tensor{}, mkResident(cb, []int{in}, x), out, in, bits, groupSize, nGroups))
	if len(yRef) != out || len(yCu) != out {
		t.Fatalf("shape ref=%d cu=%d want %d", len(yRef), len(yCu), out)
	}

	if a := argmaxF32(yRef); a != target {
		t.Fatalf("reference argmax %d != constructed dominant channel %d", a, target)
	}
	aCu := cb.Argmax(mkResident(cb, []int{out}, yCu))
	if aCu != argmaxF32(yCu) || argmaxF32(yCu) != argmaxF32(yRef) {
		t.Fatalf("GPTQ argmax-exact failed: ref=%d cudaHost=%d cudaKernel=%d", argmaxF32(yRef), argmaxF32(yCu), aCu)
	}
	c := cosine(nonTarget(yRef, out, target), nonTarget(yCu, out, target))
	if c < cudaGPTQCosineMin {
		t.Fatalf("GPTQ MatMul cosine %.6f < recorded GPTQ gate %.6f (cudaGPTQCosineMin)", c, cudaGPTQCosineMin)
	}
	t.Logf("#3030 GPTQ MatMul: cosine=%.8f maxAbs=%.2e gate=%.4f argmax-exact bits=%d groups=%d (device=%s tier=%s class=%s)",
		c, maxAbsDelta(yRef, yCu), cudaGPTQCosineMin, bits, nGroups, cb.Name(), cb.Tier(), cb.Class())
}

// BenchmarkCUDAGPTQMatMul — the decode GEMV with a resident packed GPTQ 4-bit weight (dequant fused
// into the tile), at a realistic Llama projection size (fp16BenchDims out/in). The acceptance script
// turns this ns/op into the fak-GPTQ-vs-llama.cpp-GPTQ decode throughput comparison the #3030 gate
// asks for (within-2x). This records fak's number; the same-host llama.cpp GPTQ decode is captured
// alongside it on the GPU node — neither is asserted here.
func BenchmarkCUDAGPTQMatMul(b *testing.B) {
	cb := cudaTBOrSkip(b)
	out, in, _ := fp16BenchDims() // 4096 x 4096
	bits, groupSize := 4, 128
	pack := 32 / bits
	var seed lcg = 3030
	g := &seed
	qweight, qzeros, scales, nGroups := randGPTQ(g, out, in, bits, groupSize)
	wW, zW, sW := uploadGPTQResident(cb, qweight, qzeros, scales, out, in, pack, nGroups)
	x := mkResident(cb, []int{in}, rscale(g, in, 1.0))
	cb.Read(cb.GPTQMatMul(wW, zW, sW, Tensor{}, x, out, in, bits, groupSize, nGroups)) // warm: pool + weight cache
	cb.Recycle()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cb.Read(cb.GPTQMatMul(wW, zW, sW, Tensor{}, x, out, in, bits, groupSize, nGroups))
		cb.Recycle()
	}
	b.StopTimer()
}
