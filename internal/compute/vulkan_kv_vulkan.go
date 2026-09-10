//go:build vulkan && (windows || linux) && cgo

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
	backing  *vulkanKVBacking
}

// vulkanKVBacking is the allocation shared by metadata-only KV clones. All access is
// serialized by vulkanMu. highWater is the greatest float offset written: a clone at
// highWater may append into unused capacity without affecting readers whose local len ends
// at the old high-water mark. A writer whose len trails highWater must detach first because
// another fork already owns the tail it would overwrite.
type vulkanKVBacking struct {
	ptr       unsafe.Pointer
	cap       int
	refs      int
	highWater int
}

func (d *vslice) adoptBacking() {
	if d.backing == nil && d.ptr != nil {
		d.backing = &vulkanKVBacking{ptr: d.ptr, cap: d.cap, refs: 1, highWater: d.len}
	}
}

func (v *vulkanBackend) makeVSliceWritable(d *vslice, need int, preserve bool, what string) {
	ncap := d.cap
	if ncap < need {
		ncap = d.cap*2 + (need - d.len)
		if ncap < need {
			ncap = need
		}
	}
	v.makeVSliceWritableCapacity(d, need, ncap, preserve, what)
}

// makeVSliceWritableCapacity is the exact-capacity form used by callers that already
// applied their own guarded growth policy, such as Qwen sequence prefill.
func (v *vulkanBackend) makeVSliceWritableCapacity(d *vslice, need, ncap int, preserve bool, what string) {
	d.adoptBacking()
	if d.backing != nil && d.backing.refs == 1 && need <= d.cap {
		// Any tail beyond this sole owner's visible length is unreachable and may be
		// reclaimed by its next append or rewrite.
		d.backing.highWater = d.len
		return
	}
	if d.backing == nil && need <= d.cap {
		return
	}
	if ncap < need {
		ncap = need
	}
	var np unsafe.Pointer
	if ncap > 0 {
		np = v.dallocKVFor(ncap*F32.Bytes(), what).ptr
		if preserve && d.len > 0 {
			C.fvk_d2d(np, d.ptr, C.size_t(d.len*4))
		}
	}
	d.releaseBacking()
	d.ptr, d.cap = np, ncap
	if np != nil {
		d.backing = &vulkanKVBacking{ptr: np, cap: ncap, refs: 1, highWater: d.len}
	}
}

func (d *vslice) releaseBacking() {
	d.adoptBacking()
	if d.backing != nil {
		d.backing.refs--
		if d.backing.refs == 0 && d.backing.ptr != nil {
			C.fvk_free(d.backing.ptr)
		}
	}
	d.ptr, d.cap, d.backing = nil, 0, nil
}

func (v *vulkanBackend) growAppend(d *vslice, srcPtr unsafe.Pointer, nFloats int, what string) {
	need := d.len + nFloats
	d.adoptBacking()
	// Shared storage is appendable only at the allocation's frontier. The first fork to
	// claim the tail advances highWater; every sibling still at the old len then detaches.
	if d.backing != nil && d.backing.refs == 1 && need <= d.cap {
		d.backing.highWater = d.len // discard an unreachable tail left by a released sibling
	} else if d.backing != nil && d.len == d.backing.highWater && need <= d.cap {
		// safe append into bytes outside every other owner's visible range
	} else {
		v.makeVSliceWritable(d, need, true, what)
	}
	// append the new row at byte offset d.len within the (possibly grown) layer buffer.
	// d.ptr is an OPAQUE Buffer* handle, not a base address, so the destination offset must
	// be expressed to the shim (fvk_d2d_off) — pointer arithmetic on d.ptr would be garbage.
	C.fvk_d2d_off(d.ptr, C.size_t(d.len*4), srcPtr, C.size_t(nFloats*4))
	d.len += nFloats
	d.adoptBacking()
	if d.backing != nil && d.len > d.backing.highWater {
		d.backing.highWater = d.len
	}
}

type vulkanKV struct {
	be         *vulkanBackend
	cfg        KVConfig
	K          []vslice
	Kraw       []vslice
	V          []vslice
	pos        []int
	scratchpad *VulkanKVScratchpad
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

// Clone shares immutable visible prefixes. Writers detach only when they would overwrite a
// tail another fork has claimed; the first append at the shared high-water is zero-copy.
func (k *vulkanKV) Clone() KVStore {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	n := &vulkanKV{be: k.be, cfg: k.cfg,
		K: make([]vslice, len(k.K)), Kraw: make([]vslice, len(k.Kraw)), V: make([]vslice, len(k.V)),
		pos: append([]int(nil), k.pos...)}
	share := func(dst, src *vslice) {
		src.adoptBacking()
		if src.backing != nil && src.backing.refs == 1 {
			// A sole owner may have survived a divergent sibling. Its invisible tail is
			// not part of the new clone's prefix and must not reserve the frontier.
			src.backing.highWater = src.len
		}
		*dst = *src
		if src.backing != nil {
			src.backing.refs++
		}
	}
	for l := range k.K {
		share(&n.K[l], &k.K[l])
		share(&n.Kraw[l], &k.Kraw[l])
		share(&n.V[l], &k.V[l])
	}
	return n
}

// Free releases every per-layer key, pre-RoPE key, and value device buffer and clears
// the position list, returning all VRAM the cache held.
func (k *vulkanKV) Free() {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	if k.scratchpad != nil {
		if k.scratchpad.DeviceBufK != nil {
			if b, ok := k.scratchpad.DeviceBufK.(*vulkanBuf); ok && b != nil && b.ptr != nil {
				C.fvk_free(b.ptr)
			}
			k.scratchpad.DeviceBufK = nil
		}
		if k.scratchpad.DeviceBufV != nil {
			if b, ok := k.scratchpad.DeviceBufV.(*vulkanBuf); ok && b != nil && b.ptr != nil {
				C.fvk_free(b.ptr)
			}
			k.scratchpad.DeviceBufV = nil
		}
		k.scratchpad = nil
	}
	releaseKVDeviceSlices(k.K, k.Kraw, k.V, &k.pos, func(d *vslice) {
		d.releaseBacking()
		d.len = 0
	})
}

// InitScratchpad initializes or reconfigures the dequant-once scratchpad for this KV cache.
func (k *vulkanKV) InitScratchpad(arch string, format QuantizedKVType, nPos int) (*VulkanKVScratchpad, error) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	if k.scratchpad != nil && k.scratchpad.NumPos == nPos && k.scratchpad.Format == format {
		k.scratchpad.ResetReuse()
		return k.scratchpad, nil
	}
	if k.scratchpad != nil {
		k.scratchpad.Free()
		k.scratchpad = nil
	}
	sp, err := NewVulkanKVScratchpad(k.be, arch, format, nPos, k.cfg.NumKVHeads, k.cfg.HeadDim)
	if err != nil {
		return nil, err
	}
	k.scratchpad = sp
	return sp, nil
}

// EnsureScratchpad returns the cached scratchpad or allocates a new one if dimensions or format changed.
func (k *vulkanKV) EnsureScratchpad(arch string, format QuantizedKVType, nPos int) (*VulkanKVScratchpad, error) {
	return k.InitScratchpad(arch, format, nPos)
}

// Scratchpad returns the currently allocated scratchpad, or nil if none.
func (k *vulkanKV) Scratchpad() *VulkanKVScratchpad {
	return k.scratchpad
}

// ResetScratchpad resets usage counters on the cached scratchpad.
func (k *vulkanKV) ResetScratchpad() {
	if k.scratchpad != nil {
		k.scratchpad.ResetReuse()
	}
}

func (k *vulkanKV) readVS(d *vslice) []float32 {
	return readDeviceFloats(d.len, func(out []float32) {
		C.fvk_d2h(unsafe.Pointer(&out[0]), d.ptr, C.size_t(d.len*4))
	})
}

func (k *vulkanKV) writeVS(d *vslice, data []float32, what string) {
	need := len(data)
	// Rewrites start at offset zero, so any shared owner must detach even when capacity fits.
	d.adoptBacking()
	if d.backing != nil && d.backing.refs > 1 {
		k.be.makeVSliceWritable(d, need, false, what)
	} else if need > d.cap {
		k.be.makeVSliceWritable(d, need, false, what)
	}
	if need > 0 {
		C.fvk_h2d(d.ptr, unsafe.Pointer(&data[0]), C.size_t(need*4))
	}
	d.len = need
	d.adoptBacking()
	if d.backing != nil {
		d.backing.highWater = need
	}
}
