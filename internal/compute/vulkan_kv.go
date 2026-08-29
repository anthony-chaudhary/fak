//go:build vulkan && windows

package compute

/*
#include <stdlib.h>
#include "vulkan_backend.h"
*/
import "C"

import (
	"strconv"
	"unsafe"
)

// NewKV creates an empty device-resident KV cache sized for cfg.NumLayers, with the
// pre-RoPE keys, post-RoPE keys, and values each held in their own per-layer slices.
func (v *vulkanBackend) NewKV(cfg KVConfig) KVStore {
	k := &vulkanKV{be: v, cfg: cfg}
	k.K = make([]vslice, cfg.NumLayers)
	k.Kraw = make([]vslice, cfg.NumLayers)
	k.V = make([]vslice, cfg.NumLayers)
	return k
}

type vslice struct {
	ptr      unsafe.Pointer
	len, cap int
}

func (v *vulkanBackend) growAppend(d *vslice, srcPtr unsafe.Pointer, nFloats int, what string) {
	if d.len+nFloats > d.cap {
		ncap := d.cap*2 + nFloats
		np := v.dallocKVFor(ncap*F32.Bytes(), what).ptr
		if d.len > 0 {
			C.fvk_d2d(unsafe.Pointer(np), d.ptr, C.size_t(d.len*4))
		}
		if d.ptr != nil {
			C.fvk_free(d.ptr)
		}
		d.ptr = unsafe.Pointer(np)
		d.cap = ncap
	}
	// append the new row at byte offset d.len within the (possibly grown) layer buffer.
	// d.ptr is an OPAQUE Buffer* handle, not a base address, so the destination offset must
	// be expressed to the shim (fvk_d2d_off) — pointer arithmetic on d.ptr would be garbage.
	C.fvk_d2d_off(d.ptr, C.size_t(d.len*4), srcPtr, C.size_t(nFloats*4))
	d.len += nFloats
}

type vulkanKV struct {
	be   *vulkanBackend
	cfg  KVConfig
	K    []vslice
	Kraw []vslice
	V    []vslice
	pos  []int
}

func (k *vulkanKV) stride() int { return k.cfg.NumKVHeads * k.cfg.HeadDim }

func (k *vulkanKV) ResidentBytes() int64 {
	return kvResidentBytes(len(k.K), len(k.pos), func(layer int) (int, int, int) {
		return k.K[layer].len, k.Kraw[layer].len, k.V[layer].len
	})
}

func (k *vulkanKV) AppendKV(layer int, kRaw, kRoPE, val Tensor, pos int) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	w := k.stride()
	k.be.growAppend(&k.Kraw[layer], kRaw.buf.(*vulkanBuf).ptr, w, "KV pre-RoPE key cache layer "+strconv.Itoa(layer))
	k.be.growAppend(&k.K[layer], kRoPE.buf.(*vulkanBuf).ptr, w, "KV key cache layer "+strconv.Itoa(layer))
	k.be.growAppend(&k.V[layer], val.buf.(*vulkanBuf).ptr, w, "KV value cache layer "+strconv.Itoa(layer))
	if layer == 0 {
		k.pos = append(k.pos, pos)
	}
}

// AppendKVRoPE appends one position, applying RoPE on-device: it stores the pre-RoPE key
// (so Evict can reposition it), rotates it in place to form the post-RoPE key, and stores
// that and the value row.
func (k *vulkanKV) AppendKVRoPE(layer int, kRaw, val Tensor, pos, nHeads, headDim int, theta float64) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	if nHeads != k.cfg.NumKVHeads || headDim != k.cfg.HeadDim {
		panic("compute: vulkan AppendKVRoPE shape does not match KV config")
	}
	w := k.stride()
	kRawPtr := kRaw.buf.(*vulkanBuf).ptr
	k.be.growAppend(&k.Kraw[layer], kRawPtr, w, "KV pre-RoPE key cache layer "+strconv.Itoa(layer))
	C.fvk_rope_f32(kRawPtr, C.int(pos), C.int(nHeads), C.int(headDim), C.double(theta))
	k.be.growAppend(&k.K[layer], kRawPtr, w, "KV key cache layer "+strconv.Itoa(layer))
	k.be.growAppend(&k.V[layer], val.buf.(*vulkanBuf).ptr, w, "KV value cache layer "+strconv.Itoa(layer))
	if layer == 0 {
		k.pos = append(k.pos, pos)
	}
}

// Len reports the number of positions currently cached.
func (k *vulkanKV) Len() int   { return len(k.pos) }
func (k *vulkanKV) Pos() []int { return append([]int(nil), k.pos...) }

func (k *vulkanKV) KeysView(layer int) Tensor {
	w := k.stride()
	n := k.K[layer].len / w
	return makeTensor(k.be, F32, RowMajor, []int{n, w}, nil, &vulkanBuf{ptr: k.K[layer].ptr, n: k.K[layer].len * 4, class: MemoryKVCache})
}

// ValuesView returns a flat [pos, nKV*hd] device tensor viewing the layer's cached value
// rows, without copying the underlying storage.
func (k *vulkanKV) ValuesView(layer int) Tensor {
	w := k.stride()
	n := k.V[layer].len / w
	return makeTensor(k.be, F32, RowMajor, []int{n, w}, nil, &vulkanBuf{ptr: k.V[layer].ptr, n: k.V[layer].len * 4, class: MemoryKVCache})
}

// Evict removes [from, from+n) from every layer and compacts the survivors, re-RoPE-ing
// each shifted key from its stored pre-RoPE copy so the cache is byte-for-byte what it
// would be had the span never been seen; it returns the number of positions removed.
func (k *vulkanKV) Evict(from, n int) int {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	if from < 0 || n <= 0 || from >= len(k.pos) {
		return 0
	}
	end := from + n
	if end > len(k.pos) {
		end = len(k.pos)
	}
	w := k.stride()
	hd, nKV := k.cfg.HeadDim, k.cfg.NumKVHeads
	for l := 0; l < k.cfg.NumLayers; l++ {
		K := k.readVS(&k.K[l])
		Kraw := k.readVS(&k.Kraw[l])
		V := k.readVS(&k.V[l])
		K = append(K[:from*w], K[end*w:]...)
		Kraw = append(Kraw[:from*w], Kraw[end*w:]...)
		V = append(V[:from*w], V[end*w:]...)
		newPos := append(append([]int(nil), k.pos[:from]...), k.pos[end:]...)
		for i := range newPos {
			if newPos[i] != i {
				cos, sin := ropeRow(k.cfg.RopeTheta, hd, i)
				for h := 0; h < nKV; h++ {
					dst := K[i*w+h*hd : i*w+(h+1)*hd]
					copy(dst, Kraw[i*w+h*hd:i*w+(h+1)*hd])
					applyRope(dst, cos, sin)
				}
			}
		}
		k.writeVS(&k.K[l], K, "KV key cache rewrite layer "+strconv.Itoa(l))
		k.writeVS(&k.Kraw[l], Kraw, "KV pre-RoPE key cache rewrite layer "+strconv.Itoa(l))
		k.writeVS(&k.V[l], V, "KV value cache rewrite layer "+strconv.Itoa(l))
	}
	k.pos = append(k.pos[:from], k.pos[end:]...)
	for i := range k.pos {
		k.pos[i] = i
	}
	return end - from
}

// Clone deep-copies the cache (each layer's key, pre-RoPE key, and value buffers copied
// D2D into fresh device allocations) so a forked decode can reuse a shared prefix.
func (k *vulkanKV) Clone() KVStore {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	n := &vulkanKV{be: k.be, cfg: k.cfg,
		K: make([]vslice, len(k.K)), Kraw: make([]vslice, len(k.Kraw)), V: make([]vslice, len(k.V)),
		pos: append([]int(nil), k.pos...)}
	cp := func(dst, src *vslice, what string) {
		if src.len == 0 {
			return
		}
		np := k.be.dallocKVFor(src.len*F32.Bytes(), what).ptr
		C.fvk_d2d(unsafe.Pointer(np), src.ptr, C.size_t(src.len*4))
		dst.ptr, dst.len, dst.cap = unsafe.Pointer(np), src.len, src.len
	}
	for l := range k.K {
		cp(&n.K[l], &k.K[l], "KV key cache clone layer "+strconv.Itoa(l))
		cp(&n.Kraw[l], &k.Kraw[l], "KV pre-RoPE key cache clone layer "+strconv.Itoa(l))
		cp(&n.V[l], &k.V[l], "KV value cache clone layer "+strconv.Itoa(l))
	}
	return n
}

// Free releases every per-layer key, pre-RoPE key, and value device buffer and clears
// the position list, returning all VRAM the cache held.
func (k *vulkanKV) Free() {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	releaseKVDeviceSlices(k.K, k.Kraw, k.V, &k.pos, func(d *vslice) {
		releaseDeviceSlice(&d.ptr, &d.len, &d.cap, func(pointer unsafe.Pointer) { C.fvk_free(pointer) })
	})
}

func (k *vulkanKV) readVS(d *vslice) []float32 {
	return readDeviceFloats(d.len, func(out []float32) {
		C.fvk_d2h(unsafe.Pointer(&out[0]), d.ptr, C.size_t(d.len*4))
	})
}

func (k *vulkanKV) writeVS(d *vslice, data []float32, what string) {
	need := len(data)
	if need > d.cap {
		if d.ptr != nil {
			C.fvk_free(d.ptr)
		}
		d.ptr = k.be.dallocKVFor(need*F32.Bytes(), what).ptr
		d.cap = need
	}
	if need > 0 {
		C.fvk_h2d(d.ptr, unsafe.Pointer(&data[0]), C.size_t(need*4))
	}
	d.len = need
}
