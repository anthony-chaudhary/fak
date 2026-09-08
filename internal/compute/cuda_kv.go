//go:build cuda && cgo

package compute

/*
#include <stdlib.h>
#include "cuda_backend.h"
*/
import "C"

import (
	"fmt"
	"sync/atomic"
	"unsafe"
)

// cudaKVMaxPos is the fixed cache capacity (in positions) each device KV preallocates, so
// AppendKV never reallocs — a hard requirement for CUDA-graph capture (a cudaMalloc during
// capture is illegal). 1024 covers the decode benchmarks; a longer-context serve raises it
// to the context budget via SetCUDAGraphKVCapacity so a real prompt never grows the cache
// mid-capture. Read only inside the graphEnabled NewKV prealloc, so a plain const-like var.
var cudaKVMaxPos = 1024

// NewKV creates a device-resident KV store for cfg's geometry; under graph capture it
// preallocates a fixed cudaKVMaxPos capacity (no mid-token cudaMalloc), otherwise it stays growable.
func (c *cudaBackend) NewKV(cfg KVConfig) KVStore {
	k := &cudaKV{be: c, cfg: cfg}
	k.K = make([]dslice, cfg.NumLayers)
	k.Kraw = make([]dslice, cfg.NumLayers)
	k.V = make([]dslice, cfg.NumLayers)
	if graphEnabled {
		// Graph capture forbids a cudaMalloc mid-token, so preallocate a fixed capacity
		// the cache never has to realloc within. Default (non-graph) path stays growable
		// and lean (no per-session preallocation).
		k.maxPos = cudaKVMaxPos
		capF := k.maxPos * cfg.NumKVHeads * cfg.HeadDim
		for l := 0; l < cfg.NumLayers; l++ {
			k.K[l] = dslice{ptr: k.be.dallocKV(capF*F32.Bytes(), "kv-key-prealloc layer "+itoaC(l)).ptr, cap: capF}
			k.Kraw[l] = dslice{ptr: k.be.dallocKV(capF*F32.Bytes(), "kv-pre-rope-key-prealloc layer "+itoaC(l)).ptr, cap: capF}
			k.V[l] = dslice{ptr: k.be.dallocKV(capF*F32.Bytes(), "kv-value-prealloc layer "+itoaC(l)).ptr, cap: capF}
		}
	}
	return k
}

func (c *cudaBackend) dallocKV(nbytes int, site string) *cudaBuf {
	if site == "" {
		site = "kv-cache"
	}
	return c.dallocClass(nbytes, MemoryKVCache, site)
}

// dslice is a growable VRAM float buffer (len/cap in floats).
type dslice struct {
	ptr      unsafe.Pointer
	len, cap int
}

func (c *cudaBackend) growAppend(d *dslice, srcPtr unsafe.Pointer, nFloats int, site string) {
	if d.len+nFloats > d.cap {
		ncap := d.cap*2 + nFloats
		np := c.dallocKV(ncap*F32.Bytes(), site).ptr
		if d.len > 0 {
			C.fcuda_d2d(unsafe.Pointer(np), d.ptr, C.size_t(d.len*4))
		}
		if d.ptr != nil {
			C.fcuda_free(d.ptr)
		}
		d.ptr = unsafe.Pointer(np)
		d.cap = ncap
	}
	// kernel-form append (scalar offset) instead of a cudaMemcpy to a moving pointer, so a
	// captured decode graph stays reusable via cudaGraphExecUpdate as the cache grows.
	C.fcuda_kv_write((*C.float)(d.ptr), (*C.float)(srcPtr), C.int(d.len), C.int(nFloats))
	d.len += nFloats
}

type cudaKV struct {
	be     *cudaBackend
	cfg    KVConfig
	maxPos int // fixed capacity in positions (preallocated so AppendKV never reallocs)
	K      []dslice
	Kraw   []dslice
	V      []dslice
	pos    []int
}

func (k *cudaKV) stride() int { return k.cfg.NumKVHeads * k.cfg.HeadDim }

func (k *cudaKV) ResidentBytes() int64 {
	return kvResidentBytes(len(k.K), len(k.pos), func(layer int) (int, int, int) {
		return k.K[layer].len, k.Kraw[layer].len, k.V[layer].len
	})
}

func (k *cudaKV) AppendKV(layer int, kRaw, kRoPE, v Tensor, pos int) {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	w := k.stride()
	// Preflight all three sources before the first append so a poisoned second or
	// third operand cannot leave a partially advanced KV row.
	kRawBuf := k.be.cudaBufForSubmit(kRaw)
	kRoPEBuf := k.be.cudaBufForSubmit(kRoPE)
	vBuf := k.be.cudaBufForSubmit(v)
	k.be.growAppend(&k.Kraw[layer], kRawBuf.ptr, w, "kv-pre-rope-key-grow layer "+itoaC(layer))
	k.be.growAppend(&k.K[layer], kRoPEBuf.ptr, w, "kv-key-grow layer "+itoaC(layer))
	k.be.growAppend(&k.V[layer], vBuf.ptr, w, "kv-value-grow layer "+itoaC(layer))
	if layer == 0 {
		k.pos = append(k.pos, pos)
	}
}

// Len reports the number of cached positions (entries per layer).
func (k *cudaKV) Len() int   { return len(k.pos) }
func (k *cudaKV) Pos() []int { return append([]int(nil), k.pos...) }

func (k *cudaKV) KeysView(layer int) Tensor {
	w := k.stride()
	n := k.K[layer].len / w
	return makeTensor(k.be, F32, RowMajor, []int{n, w}, nil, &cudaBuf{ptr: k.K[layer].ptr, n: k.K[layer].len * 4, class: MemoryKVCache})
}

// ValuesView returns a device handle onto the layer's cached values as a flat [pos, nKV*hd]
// tensor (a VRAM view, not a host copy — Host on it stays (nil,false)).
func (k *cudaKV) ValuesView(layer int) Tensor {
	w := k.stride()
	n := k.V[layer].len / w
	return makeTensor(k.be, F32, RowMajor, []int{n, w}, nil, &cudaBuf{ptr: k.V[layer].ptr, n: k.V[layer].len * 4, class: MemoryKVCache})
}

// Evict compacts the cache ON-GPU — no host round-trip (#479). For every layer it shifts
// the survivors of K/Kraw/V down past the [from,from+n) span, then re-derives the post-RoPE
// K of each survivor whose absolute position changed by a SINGLE rotation of its (already
// device-resident) Kraw at the NEW index — the very kernel AppendKV used, so a device evict
// is bit-identical to a device run that never saw the span (the Approx-gate witness). The
// prefix [0,from) is left byte-for-byte untouched; that asymmetry — only the suffix is
// repositioned — is the write-time quarantine witness (MODEL-ARCH-SEAM §3, O1–O3): a span
// evicted before the query attends vanishes, but one evicted after downstream tokens already
// attended cannot be un-seen. The KV never leaves VRAM, so Host() on these tensors stays
// (nil,false). The host round-trip this replaces lived on cpuKV.Evict / earlier cudaKV.
func (k *cudaKV) Evict(from, n int) int {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	if from < 0 || n <= 0 || from >= len(k.pos) {
		return 0
	}
	end := from + n
	if end > len(k.pos) {
		end = len(k.pos)
	}
	w := k.stride()
	hd, nKV := k.cfg.HeadDim, k.cfg.NumKVHeads
	fromF, endF := from*w, end*w
	tailFloats := (len(k.pos) - end) * w // survivors after the span (shared by K/Kraw/V)
	// survivor positions after compaction: prefix keeps its index, suffix shifts down.
	newPos := append(append([]int(nil), k.pos[:from]...), k.pos[end:]...)
	// One reused scratch buffer for the leftward shift: an in-place device-to-device copy of
	// overlapping regions is undefined, so the tail is staged through disjoint VRAM. Stream
	// ordering (everything on g_stream) serializes the per-layer reuse correctly.
	var scratch unsafe.Pointer
	if tailFloats > 0 {
		scratch = unsafe.Pointer(C.fcuda_malloc(C.size_t(tailFloats * 4)))
		if scratch == nil {
			panic(&DeviceAllocError{Bytes: tailFloats * 4, Site: "evict-scratch", Class: MemoryScratchpad})
		}
	}
	for l := 0; l < k.cfg.NumLayers; l++ {
		k.be.compactDS(&k.K[l], fromF, endF, tailFloats, scratch)
		k.be.compactDS(&k.Kraw[l], fromF, endF, tailFloats, scratch)
		k.be.compactDS(&k.V[l], fromF, endF, tailFloats, scratch)
		for i := range newPos {
			if newPos[i] == i {
				continue // prefix survivor: position unchanged, post-RoPE K stays byte-for-byte
			}
			// K[i] <- Kraw[i] (disjoint buffers, no overlap) then one in-place rotation at i.
			kRow := offsetF(k.K[l].ptr, i*w)
			C.fcuda_d2d(kRow, offsetF(k.Kraw[l].ptr, i*w), C.size_t(w*4))
			C.fcuda_rope_f32((*C.float)(kRow), C.int(i), C.int(nKV), C.int(hd), C.double(k.cfg.RopeTheta))
		}
	}
	if scratch != nil {
		C.fcuda_free(scratch)
	}
	k.pos = append(k.pos[:from], k.pos[end:]...)
	for i := range k.pos {
		k.pos[i] = i
	}
	return end - from
}

// offsetF advances a device pointer by nFloats f32 elements. The KV buffers are C-allocated
// (cudaMalloc), not Go-managed memory, so this is the correct way to address a sub-row and
// is outside the GC's purview (the vet unsafeptr concern is for Go-heap pointers, not these).
func offsetF(p unsafe.Pointer, nFloats int) unsafe.Pointer {
	return unsafe.Pointer(uintptr(p) + uintptr(nFloats)*4)
}

// compactDS removes the float span [fromF,endF) from a position-major device buffer in place
// by shifting its tailFloats-long tail down through a caller-supplied disjoint scratch. A
// direct leftward device-to-device copy would overlap (src and dst intersect), which
// cudaMemcpy leaves undefined; staging through scratch is well-defined and never touches the
// host. Both copies ride g_stream, so they stay ordered against each other and the re-RoPE.
func (c *cudaBackend) compactDS(d *dslice, fromF, endF, tailFloats int, scratch unsafe.Pointer) {
	if tailFloats > 0 {
		C.fcuda_d2d(scratch, offsetF(d.ptr, endF), C.size_t(tailFloats*4))
		C.fcuda_d2d(offsetF(d.ptr, fromF), scratch, C.size_t(tailFloats*4))
	}
	d.len -= endF - fromF
}

// Clone deep-copies the cache device-to-device (a fresh VRAM allocation per layer for K/Kraw/V
// plus the position list), so a forked session reuses the prefix without sharing storage.
func (k *cudaKV) Clone() KVStore {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	n := &cudaKV{be: k.be, cfg: k.cfg,
		K: make([]dslice, len(k.K)), Kraw: make([]dslice, len(k.Kraw)), V: make([]dslice, len(k.V)),
		pos: append([]int(nil), k.pos...)}
	cp := func(dst, src *dslice, site string) {
		if src.len == 0 {
			return
		}
		np := k.be.dallocKV(src.len*F32.Bytes(), site).ptr
		C.fcuda_d2d(unsafe.Pointer(np), src.ptr, C.size_t(src.len*4))
		dst.ptr, dst.len, dst.cap = unsafe.Pointer(np), src.len, src.len
	}
	for l := range k.K {
		cp(&n.K[l], &k.K[l], "kv-key-clone layer "+itoaC(l))
		cp(&n.Kraw[l], &k.Kraw[l], "kv-pre-rope-key-clone layer "+itoaC(l))
		cp(&n.V[l], &k.V[l], "kv-value-clone layer "+itoaC(l))
	}
	return n
}

// SnapshotToHost copies the complete CUDA KV owner into ordinary host DRAM,
// including pre-RoPE Kraw. The source remains resident until the caller
// explicitly frees/demotes it, preserving stage-before-evict ordering.
func (k *cudaKV) SnapshotToHost() (KVHostSnapshot, error) {
	if k == nil {
		return KVHostSnapshot{}, fmt.Errorf("cuda: cannot snapshot nil KV store")
	}
	cudaMu.Lock()
	defer cudaMu.Unlock()
	copyDS := func(src dslice) []float32 {
		out := make([]float32, src.len)
		if len(out) > 0 {
			C.fcuda_d2h(unsafe.Pointer(&out[0]), src.ptr, C.size_t(len(out)*F32.Bytes()))
			atomic.AddUint64(&k.be.fenceGen, 1)
		}
		return out
	}
	out := KVHostSnapshot{
		Config: cloneKVConfig(k.cfg),
		Pos:    append([]int(nil), k.pos...),
		K:      make([][]float32, len(k.K)),
		KRaw:   make([][]float32, len(k.Kraw)),
		V:      make([][]float32, len(k.V)),
	}
	for layer := range k.K {
		out.K[layer] = copyDS(k.K[layer])
		out.KRaw[layer] = copyDS(k.Kraw[layer])
		out.V[layer] = copyDS(k.V[layer])
	}
	return out, out.Validate()
}

// RestoreKVFromHost bulk-copies a complete host image back into fresh CUDA KV
// allocations. It is the inverse of SnapshotToHost; no per-token forward runs.
func (c *cudaBackend) RestoreKVFromHost(state KVHostSnapshot) (out KVStore, err error) {
	if err := state.Validate(); err != nil {
		return nil, err
	}
	cudaMu.Lock()
	defer cudaMu.Unlock()
	k, ok := c.NewKV(cloneKVConfig(state.Config)).(*cudaKV)
	if !ok || k == nil {
		return nil, fmt.Errorf("cuda: NewKV returned an incompatible store during host restore")
	}
	k.pos = append([]int(nil), state.Pos...)
	freePartial := func() {
		free := func(d *dslice) {
			if d.ptr != nil {
				C.fcuda_free(d.ptr)
				d.ptr = nil
			}
			d.len, d.cap = 0, 0
		}
		for layer := range k.K {
			free(&k.K[layer])
			free(&k.Kraw[layer])
			free(&k.V[layer])
		}
	}
	defer func() {
		if r := recover(); r != nil {
			freePartial()
			err = fmt.Errorf("cuda: restore KV from host: %v", r)
			out = nil
		}
	}()
	copyHost := func(dst *dslice, src []float32, site string) {
		if len(src) == 0 {
			return
		}
		if dst.ptr == nil || dst.cap < len(src) {
			if dst.ptr != nil {
				C.fcuda_free(dst.ptr)
			}
			buf := c.dallocKV(len(src)*F32.Bytes(), site)
			dst.ptr, dst.cap = buf.ptr, len(src)
		}
		C.fcuda_h2d(dst.ptr, unsafe.Pointer(&src[0]), C.size_t(len(src)*F32.Bytes()))
		dst.len = len(src)
	}
	for layer := range state.K {
		copyHost(&k.K[layer], state.K[layer], "kv-key-host-restore layer "+itoaC(layer))
		copyHost(&k.Kraw[layer], state.KRaw[layer], "kv-pre-rope-key-host-restore layer "+itoaC(layer))
		copyHost(&k.V[layer], state.V[layer], "kv-value-host-restore layer "+itoaC(layer))
	}
	return k, nil
}

// Free releases every layer's K/Kraw/V VRAM buffer and clears the position list.
func (k *cudaKV) Free() {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	releaseKVDeviceSlices(k.K, k.Kraw, k.V, &k.pos, func(d *dslice) {
		releaseDeviceSlice(&d.ptr, &d.len, &d.cap, func(pointer unsafe.Pointer) { C.fcuda_free(pointer) })
	})
}

// itoaC is a tiny int->string for the tier label (avoids importing strconv into the
// build-tagged file's surface).
func itoaC(n int) string {
	if n == 0 {
		return "0"
	}
	neg := n < 0
	if neg {
		n = -n
	}
	var b [20]byte
	i := len(b)
	for n > 0 {
		i--
		b[i] = byte('0' + n%10)
		n /= 10
	}
	if neg {
		i--
		b[i] = '-'
	}
	return string(b[i:])
}
