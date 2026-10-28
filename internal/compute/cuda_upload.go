//go:build cuda && cgo

package compute

/*
#cgo CFLAGS: -I${SRCDIR}
#cgo LDFLAGS: -L${SRCDIR} -lfakcuda -lcudart -lcublas -lcublasLt -lstdc++ -lm
#include <stdlib.h>
#include "cuda_backend.h"
*/
import "C"

import "unsafe"

// uploadQ8 narrows an f32 host weight [out,in] to a resident Q8_0 weight at H2D (#485): the f32 is
// quantized to per-block(32) int8 codes + f32 scales using the EXACT cpuref QuantizeQ8 scheme
// (d=amax/127, q8round), and both narrow operands are uploaded — codes to the buffer's ptr, scales
// to its scale side-channel. The f32 weight never becomes resident, so the VRAM footprint is the
// int8 size (codes) + a thin per-block scale band, not the f32 size. `in` must be divisible by 32.
func cudaUploadQuantPayload(buf *cudaBuf, codes []int8, scales []float32) {
	C.fcuda_h2d(buf.ptr, unsafe.Pointer(&codes[0]), C.size_t(len(codes)))
	C.fcuda_h2d(buf.scales, unsafe.Pointer(&scales[0]), C.size_t(len(scales)*F32.Bytes()))
}

func (c *cudaBackend) uploadQ8(t Tensor, hb HostBuffer, f []float32, hp uintptr) Tensor {
	if len(t.Shape) != 2 {
		panic(&CUDAOpError{
			Op:    "Upload",
			Site:  "upload-q8",
			Msg:   "cuda Upload(_, Q8_0) expects a 2-D [out,in] weight (got rank " + itoaC(len(t.Shape)) + ")",
			Class: MemoryWeights,
		})
	}
	out, in := t.Shape[0], t.Shape[1]
	blk := q8DeviceBlock
	nblk := in / blk
	if len(f) == 0 { // degenerate empty weight — mirror the F32 path's len==0 tolerance
		res, _ := c.devQ8(t.Shape, blk, out*nblk)
		return res
	}
	codes := make([]int8, out*in)
	scales := make([]float32, out*nblk)
	for o := 0; o < out; o++ {
		for b := 0; b < nblk; b++ {
			base := o*in + b*blk
			var amax float32
			for i := 0; i < blk; i++ {
				if a := absf(f[base+i]); a > amax {
					amax = a
				}
			}
			d := amax / 127
			scales[o*nblk+b] = d
			if d == 0 {
				continue
			}
			inv := 1.0 / d
			for i := 0; i < blk; i++ {
				codes[base+i] = q8round(f[base+i] * inv)
			}
		}
	}
	res, buf := c.devQ8(t.Shape, blk, len(scales))
	cudaUploadQuantPayload(buf, codes, scales)
	c.accountImmutableWeightUpload(len(codes)+len(scales)*F32.Bytes(), buf)
	buf.host, buf.hostKeep, buf.hostDt, buf.hostLo = hp, hb, Q8_0, t.Layout
	uploadCache[ucKey{hp, Q8_0, t.Layout}] = res
	return res
}

// cudaQ8UploadSource retains both immutable host operands and their geometry.
// A codes address alone does not identify independently supplied Q8 scales.
// Embedding the original owner preserves HostBuffer; Free clears hostKeep.
type cudaQ8UploadSource struct {
	HostBuffer
	scales []float32
	rows   int
	cols   int
	block  int
}

// uploadQ8Resident copies an ALREADY-quantized Q8_0 host weight resident with NO re-quantization:
// the int8 codes (HostBuffer.I8()) go to the buffer's ptr and the per-block(32) f32 scales
// (QuantSpec.Scale) to its scale side-channel — the SAME resident layout uploadQ8 produces, so
// k_q8_gemm consumes it unchanged. This is the memory-lean decode path (the HAL's weightHALQ8
// hands a NewQ8 host tensor); the f32-narrowing uploadQ8 above is the witness/quant-at-upload path.
func (c *cudaBackend) uploadQ8Resident(t Tensor, hb HostBuffer) Tensor {
	if len(t.Shape) != 2 {
		panic(&CUDAOpError{
			Op:    "Upload",
			Site:  "upload-q8-resident",
			Msg:   "cuda Upload(Q8_0 host) expects a 2-D [out,in] weight (got rank " + itoaC(len(t.Shape)) + ")",
			Class: MemoryWeights,
		})
	}
	if t.Quant == nil || t.Quant.Scale == nil {
		panic(&CUDAOpError{
			Op:    "Upload",
			Site:  "upload-q8-resident",
			Msg:   "cuda Upload(Q8_0 host) requires QuantSpec.Scale (per-block f32 scales)",
			Class: MemoryWeights,
		})
	}
	out, in := t.Shape[0], t.Shape[1]
	blk := t.Quant.Block
	// DP4A consumes four codes at a time; partial blocks would lose their tail.
	if blk <= 0 || blk%4 != 0 || in%blk != 0 {
		panic(&CUDAOpError{
			Op:    "Upload",
			Site:  "upload-q8-resident",
			Msg:   "cuda Upload(Q8_0 host) needs QuantSpec.Block divisible by 4 and exactly dividing in (block=" + itoaC(blk) + ")",
			Class: MemoryWeights,
		})
	}
	codes := hb.I8()
	scales := t.Quant.Scale
	nblk := in / blk
	if len(codes) != 0 && len(codes) != out*in {
		panic(&CUDAOpError{
			Op:    "Upload",
			Site:  "upload-q8-resident",
			Msg:   "cuda Upload(Q8_0 host) code length " + itoaC(len(codes)) + " != out*in",
			Class: MemoryWeights,
		})
	}
	if len(scales) != out*nblk {
		panic(&CUDAOpError{
			Op:    "Upload",
			Site:  "upload-q8-resident",
			Msg:   "cuda Upload(Q8_0 host) scale length " + itoaC(len(scales)) + " != out*(in/block)",
			Class: MemoryWeights,
		})
	}
	var hp uintptr
	if len(codes) > 0 {
		hp = uintptr(unsafe.Pointer(&codes[0]))
		// element-count guard against host-address reuse (see uploadClass F32 path).
		if cached, ok := uploadCache[ucKey{hp, Q8_0, t.Layout}]; ok && cached.Numel() == t.Numel() {
			buf, ok := cached.buf.(*cudaBuf)
			if ok && buf != nil && cached.Quant != nil && cached.Quant.Block == blk &&
				len(cached.Shape) == 2 && cached.Shape[0] == out && cached.Shape[1] == in {
				if source, ok := buf.hostKeep.(*cudaQ8UploadSource); ok && source != nil &&
					source.rows == out && source.cols == in && source.block == blk &&
					len(source.scales) == len(scales) && len(scales) > 0 && &source.scales[0] == &scales[0] {
					return cached
				}
			}
		}
	}
	res, buf := c.devQ8(t.Shape, blk, len(scales))
	if len(codes) > 0 {
		cudaUploadQuantPayload(buf, codes, scales)
		c.accountImmutableWeightUpload(len(codes)+len(scales)*F32.Bytes(), buf)
		buf.host, buf.hostKeep, buf.hostDt, buf.hostLo = hp, &cudaQ8UploadSource{
			HostBuffer: hb, scales: scales, rows: out, cols: in, block: blk,
		}, Q8_0, t.Layout
		uploadCache[ucKey{hp, Q8_0, t.Layout}] = res
	}
	return res
}

// uploadQ2Resident copies an ALREADY-packed ternary Q2_0 host weight resident with NO
// re-quantization (#4872): the 2-bit codes (HostBuffer.I8(), out*in/4 bytes) go to the buffer's
// ptr and the per-block(=Block) f32 scales (QuantSpec.Scale) to its scale side-channel — the same
// resident (codes ptr + scales) shape uploadQ8Resident produces, so k_q2_0_gemm consumes it
// unchanged. The weight stays 0.25 byte/elem in VRAM; no dequant-to-f32 round trip.
func (c *cudaBackend) uploadQ2Resident(t Tensor, hb HostBuffer) Tensor {
	if len(t.Shape) != 2 {
		panic(&CUDAOpError{
			Op:    "Upload",
			Site:  "upload-q2-resident",
			Msg:   "cuda Upload(Q2_0 host) expects a 2-D [out,in] weight (got rank " + itoaC(len(t.Shape)) + ")",
			Class: MemoryWeights,
		})
	}
	if t.Quant == nil || t.Quant.Scale == nil {
		panic(&CUDAOpError{
			Op:    "Upload",
			Site:  "upload-q2-resident",
			Msg:   "cuda Upload(Q2_0 host) requires QuantSpec.Scale (per-block f32 scales)",
			Class: MemoryWeights,
		})
	}
	out, in := t.Shape[0], t.Shape[1]
	blk := t.Quant.Block
	if blk <= 0 || in%blk != 0 {
		panic(&CUDAOpError{
			Op:    "Upload",
			Site:  "upload-q2-resident",
			Msg:   "cuda Upload(Q2_0 host) needs in divisible by QuantSpec.Block (block=" + itoaC(blk) + ")",
			Class: MemoryWeights,
		})
	}
	if in%4 != 0 {
		panic(&CUDAOpError{
			Op:    "Upload",
			Site:  "upload-q2-resident",
			Msg:   "cuda Upload(Q2_0 host) needs in divisible by 4 (2-bit codes, 4/byte)",
			Class: MemoryWeights,
		})
	}
	codes := hb.I8()
	scales := t.Quant.Scale
	nblk := in / blk
	if len(codes) != 0 && len(codes) != out*in/4 {
		panic(&CUDAOpError{
			Op:    "Upload",
			Site:  "upload-q2-resident",
			Msg:   "cuda Upload(Q2_0 host) code length " + itoaC(len(codes)) + " != out*in/4",
			Class: MemoryWeights,
		})
	}
	if len(scales) != out*nblk {
		panic(&CUDAOpError{
			Op:    "Upload",
			Site:  "upload-q2-resident",
			Msg:   "cuda Upload(Q2_0 host) scale length " + itoaC(len(scales)) + " != out*(in/block)",
			Class: MemoryWeights,
		})
	}
	var hp uintptr
	if len(codes) > 0 {
		hp = uintptr(unsafe.Pointer(&codes[0]))
		// element-count guard against host-address reuse (see uploadClass F32 path).
		if cached, ok := uploadCache[ucKey{hp, Q2_0, t.Layout}]; ok && cached.Numel() == t.Numel() {
			return cached
		}
	}
	res, buf := c.devQ2(t.Shape, blk, len(scales))
	if len(codes) > 0 {
		cudaUploadQuantPayload(buf, codes, scales)
		c.accountImmutableWeightUpload(len(codes)+len(scales)*F32.Bytes(), buf)
		buf.host, buf.hostKeep, buf.hostDt, buf.hostLo = hp, hb, Q2_0, t.Layout
		uploadCache[ucKey{hp, Q2_0, t.Layout}] = res
	}
	return res
}

// devQ2 allocates a resident packed-ternary Q2_0 weight: an out*in/4-byte codes buffer (ptr, 2-bit
// codes, 4/byte) plus an nScales*4-byte f32 scale side-channel (scales). The Tensor carries a
// QuantSpec{Block} so the GEMM reconstructs nblk = in/block. Uses dev-family weight allocs so the
// resident-weight cache is never recycled out from under it.
func (c *cudaBackend) devQ2(shape []int, block, nScales int) (Tensor, *cudaBuf) {
	return c.devQ2WithOperations(shape, block, nScales, c.dallocWeight,
		func(bytes int) *cudaBuf { return c.dallocClass(bytes, MemoryWeights, "q2-scale") },
		func(ptr unsafe.Pointer) { C.fcuda_free(ptr) },
	)
}

// devQ2WithOperations runs under the upload owner's cudaMu. It owns both
// allocations until Tensor construction succeeds. Only the code allocation has
// placement charges under the existing CUDA policy; scales remain uncharged.
// This transaction ends before H2D and upload-cache publication.
func (c *cudaBackend) devQ2WithOperations(shape []int, block, nScales int, allocateCodes, allocateScales func(int) *cudaBuf, release func(unsafe.Pointer)) (Tensor, *cudaBuf) {
	out, in := shape[0], shape[1]
	var owners [2]*cudaBuf
	published := false
	defer func() {
		if published {
			return
		}
		for i, owner := range owners {
			if owner == nil {
				continue
			}
			ptr := owner.ptr
			owner.ptr = nil
			if i == 0 {
				c.dlUsed -= owner.budgetedWeightBytes
				owner.budgetedWeightBytes = 0
				if owner.managedWeight {
					c.managedN--
					owner.managedWeight = false
				}
				owner.scales = nil
				owner.scalesN = 0
			}
			if ptr != nil {
				func() {
					// Preserve the original construction panic and attempt the
					// other owned allocation even if cleanup itself panics.
					defer func() { _ = recover() }()
					release(ptr)
				}()
			}
		}
	}()
	owners[0] = allocateCodes(out * in / 4) // packed 2-bit codes, four per byte
	owners[1] = allocateScales(nScales * 4)
	buf, scales := owners[0], owners[1]
	buf.scales = scales.ptr
	buf.scalesN = scales.n
	q := &QuantSpec{Block: block, Axis: 2, Bits: 2, Symmetric: true}
	result := makeTensor(c, Q2_0, RowMajor, append([]int(nil), shape...), q, buf)
	published = true
	return result, buf
}

// uploadQ4K copies raw Q4_K super-block bytes resident (#485). The host tensor carries the bytes
// in its HostBuffer.I8() view (one int8 per byte); they are already narrow (144 bytes / 256 elems),
// so there is no quantize or dtype-narrow step — just an H2D into a uint8 VRAM buffer the
// dequant-fused GEMM tile consumes. Cached on (host ptr, Q4_K, layout) like every other upload.
func (c *cudaBackend) uploadRawKQuant(t Tensor, hb HostBuffer) Tensor {
	raw := hb.I8()
	var hp uintptr
	if len(raw) > 0 {
		hp = uintptr(unsafe.Pointer(&raw[0]))
		// element-count guard against host-address reuse (see uploadClass F32 path).
		if cached, ok := uploadCache[ucKey{hp, t.Dtype, t.Layout}]; ok && cached.Numel() == t.Numel() {
			return cached
		}
	}
	res, buf := c.devRawKQuant(t.Dtype, t.Shape, len(raw))
	if len(raw) > 0 {
		C.fcuda_h2d(buf.ptr, unsafe.Pointer(&raw[0]), C.size_t(len(raw)))
		c.accountImmutableWeightUpload(len(raw), buf)
		buf.host, buf.hostKeep, buf.hostDt, buf.hostLo = hp, hb, t.Dtype, t.Layout
		uploadCache[ucKey{hp, t.Dtype, t.Layout}] = res
	}
	return res
}

// devQ8 allocates a resident Q8_0 weight: an out*in-byte int8 codes buffer (ptr) plus an
// nScales*4-byte f32 scale side-channel (scales). The Tensor carries a QuantSpec{Block} so the
// GEMM kernel reconstructs nblk = in/block. Weights use dev-family allocs, never devTr, so the
// resident-weight cache is never recycled out from under them.
func (c *cudaBackend) devQ8(shape []int, block, nScales int) (Tensor, *cudaBuf) {
	return c.devQ8WithOperations(shape, block, nScales, c.dallocWeight,
		func(bytes int) *cudaBuf { return c.dallocClass(bytes, MemoryWeights, "q8-scale") },
		func(ptr unsafe.Pointer) { C.fcuda_free(ptr) },
	)
}

// devQ8WithOperations runs under the upload owner's cudaMu. It owns both
// allocations until Tensor construction succeeds. Only the code allocation has
// placement charges under the existing CUDA policy; scales remain uncharged.
// This transaction ends before H2D and upload-cache publication.
func (c *cudaBackend) devQ8WithOperations(shape []int, block, nScales int, allocateCodes, allocateScales func(int) *cudaBuf, release func(unsafe.Pointer)) (Tensor, *cudaBuf) {
	out, in := shape[0], shape[1]
	var owners [2]*cudaBuf
	published := false
	defer func() {
		if published {
			return
		}
		for i, owner := range owners {
			if owner == nil {
				continue
			}
			ptr := owner.ptr
			owner.ptr = nil
			if i == 0 {
				c.dlUsed -= owner.budgetedWeightBytes
				owner.budgetedWeightBytes = 0
				if owner.managedWeight {
					c.managedN--
					owner.managedWeight = false
				}
				owner.scales = nil
				owner.scalesN = 0
			}
			if ptr != nil {
				func() {
					// Preserve the original construction panic and attempt the
					// other owned allocation even if cleanup itself panics.
					defer func() { _ = recover() }()
					release(ptr)
				}()
			}
		}
	}()
	owners[0] = allocateCodes(out * in) // int8 codes, 1 byte each
	owners[1] = allocateScales(nScales * 4)
	buf, scales := owners[0], owners[1]
	buf.scales = scales.ptr
	buf.scalesN = scales.n
	q := &QuantSpec{Block: block, Axis: 2, Bits: 8, Symmetric: true}
	result := makeTensor(c, Q8_0, RowMajor, append([]int(nil), shape...), q, buf)
	published = true
	return result, buf
}

// uploadF16 narrows an f32 host weight to a resident F16 weight at H2D (#484). The f32 is staged
// in a transient device buffer, converted to __half by a device kernel — row-major in place, or
// ColMajor transpose-repacked ([out,in] -> col-major) — and the stage is freed. The narrow runs
// on device (one conversion implementation, identical numerics to the GEMM's own half cast),
// never on the host.
func (c *cudaBackend) uploadF16(t Tensor, hb HostBuffer, f []float32, hp uintptr) Tensor {
	out, buf := c.devF16(t.Shape, t.Layout)
	if len(f) == 0 {
		return out
	}
	stage := c.allocateF16UploadStage(buf, len(f)*4,
		func(bytes int) *cudaBuf { return c.dallocClass(bytes, MemoryScratchpad, "f16-stage") },
		func(ptr unsafe.Pointer) { C.fcuda_free(ptr) },
	)
	C.fcuda_h2d(stage.ptr, unsafe.Pointer(&f[0]), C.size_t(len(f)*4))
	if t.Layout == ColMajor && len(t.Shape) == 2 {
		C.fcuda_f32_to_f16_T(buf.ptr, (*C.float)(stage.ptr), C.int(t.Shape[0]), C.int(t.Shape[1]))
	} else {
		C.fcuda_f32_to_f16(buf.ptr, (*C.float)(stage.ptr), C.int(len(f)))
	}
	C.fcuda_free(stage.ptr)
	c.accountImmutableWeightUpload(len(f)*F32.Bytes(), buf)
	buf.host, buf.hostKeep, buf.hostDt, buf.hostLo = hp, hb, F16, t.Layout
	uploadCache[ucKey{hp, F16, t.Layout}] = out
	return out
}

// allocateF16UploadStage owns the unpublished destination only while the scratch
// allocation is pending. Caller holds cudaMu. A failed allocation must not strand
// the destination or its logical placement charge. H2D, conversion, stage release
// and upload-cache publication occur after this boundary and are unchanged.
func (c *cudaBackend) allocateF16UploadStage(dst *cudaBuf, bytes int, allocate func(int) *cudaBuf, release func(unsafe.Pointer)) *cudaBuf {
	allocated := false
	defer func() {
		if allocated {
			return
		}
		ptr := dst.ptr
		dst.ptr = nil
		c.dlUsed -= dst.budgetedWeightBytes
		dst.budgetedWeightBytes = 0
		if dst.managedWeight {
			c.managedN--
			dst.managedWeight = false
		}
		if ptr != nil {
			func() {
				// Cleanup is best effort and must preserve the allocation panic.
				defer func() { _ = recover() }()
				release(ptr)
			}()
		}
	}()
	stage := allocate(bytes)
	allocated = true
	return stage
}
