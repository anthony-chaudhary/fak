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
//
// The compaction is DEVICE-RESIDENT: the survivors of K/Kraw/V are shifted down in VRAM
// through a disjoint device scratch buffer, and each relocated key is re-rotated by the
// same on-device RoPE kernel AppendKV used, so no K/Kraw/V byte crosses the host boundary.
// This mirrors cudaKV.Evict (#479). The pre-change Vulkan path read all three planes to the
// host and rewrote every layer, turning sliding-window maintenance into O(layers x context)
// host traffic (#12401). The prefix [0,from) is left byte-for-byte untouched - only the
// suffix is repositioned - which is the write-time quarantine asymmetry (MODEL-ARCH-SEAM
// section 3, O1-O3): a span evicted before the query attends vanishes, but one evicted after
// downstream tokens already attended cannot be un-seen.
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
	removed := end - from
	if removed <= 0 {
		return 0
	}
	w := k.stride()
	hd, nKV := k.cfg.HeadDim, k.cfg.NumKVHeads
	fromF, endF := from*w, end*w
	tailFloats := (len(k.pos) - end) * w // survivors after the span (shared by K/Kraw/V)
	// Survivor positions after compaction: the prefix keeps its index, the suffix shifts down.
	newPos := append(append([]int(nil), k.pos[:from]...), k.pos[end:]...)

	// Rewriting a relocated layer in place must not clobber another owner's visible prefix,
	// so take private storage for any layer still shared (clone) before mutating it.
	for l := 0; l < k.cfg.NumLayers; l++ {
		k.detachForRewrite(&k.K[l], l, "kv-key-evict layer ")
		k.detachForRewrite(&k.Kraw[l], l, "kv-pre-rope-key-evict layer ")
		k.detachForRewrite(&k.V[l], l, "kv-value-evict layer ")
	}

	// One reused scratch buffer for the leftward shift. An in-place device-to-device copy of
	// overlapping regions is undefined, so the tail is staged through disjoint VRAM; a null
	// tail means the span reached the end and nothing survives after it.
	var scratch unsafe.Pointer
	if tailFloats > 0 {
		scratch = k.be.dallocKVFor(tailFloats*F32.Bytes(), "kv-evict-scratch").ptr
		if scratch == nil {
			panic(&DeviceAllocError{Bytes: tailFloats * F32.Bytes(), Site: "evict-scratch", Class: MemoryScratchpad})
		}
	}
	for l := 0; l < k.cfg.NumLayers; l++ {
		compactVSlice(&k.K[l], fromF, endF, tailFloats, scratch)
		compactVSlice(&k.Kraw[l], fromF, endF, tailFloats, scratch)
		compactVSlice(&k.V[l], fromF, endF, tailFloats, scratch)
		for i := range newPos {
			if newPos[i] == i {
				continue // prefix survivor: position unchanged, post-RoPE K stays byte-for-byte
			}
			// K[i] <- Kraw[i] (disjoint buffers, no overlap) then one in-place rotation at i.
			C.fvk_d2d_range(k.K[l].ptr, C.size_t(i*w*4), k.Kraw[l].ptr, C.size_t(i*w*4), C.size_t(w*4))
			C.fvk_rope_f32(k.K[l].ptr, C.int(i), C.int(nKV), C.int(hd), C.double(k.cfg.RopeTheta))
		}
	}
	if scratch != nil {
		C.fvk_free(scratch)
	}
	k.pos = append(k.pos[:from], k.pos[end:]...)
	for i := range k.pos {
		k.pos[i] = i
	}
	return removed
}

// detachForRewrite gives a layer private device storage when the slice is still shared with
// a clone, so an in-place eviction cannot corrupt a sibling's visible prefix. A sole owner
// keeps its allocation; the shift is bounded by the survivors it already owns.
func (k *vulkanKV) detachForRewrite(d *vslice, layer int, site string) {
	d.adoptBacking()
	if d.backing != nil && d.backing.refs > 1 {
		k.be.makeVSliceWritable(d, d.len, false, site+strconv.Itoa(layer))
	}
}

// compactVSlice removes the float span [fromF,endF) from a position-major device buffer in
// place by shifting its tailFloats-long tail down through a caller-supplied disjoint scratch.
// A direct leftward device-to-device copy would overlap (src and dst intersect), which
// VkBufferCopy leaves undefined; staging through scratch is well-defined and never touches
// the host. d.ptr is an opaque Buffer* handle, so offsets are byte offsets, not pointer math.
func compactVSlice(d *vslice, fromF, endF, tailFloats int, scratch unsafe.Pointer) {
	if tailFloats > 0 {
		C.fvk_d2d_range(scratch, 0, d.ptr, C.size_t(endF*4), C.size_t(tailFloats*4))
		C.fvk_d2d_range(d.ptr, C.size_t(fromF*4), scratch, 0, C.size_t(tailFloats*4))
	}
	d.len -= endF - fromF
	d.adoptBacking()
	if d.backing != nil && d.len > d.backing.highWater {
		d.backing.highWater = d.len
	}
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
