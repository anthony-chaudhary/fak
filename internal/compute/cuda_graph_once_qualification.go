//go:build cuda && cgo && fak_cuda_capture_once_qualification

package compute

/*
#include "cuda_backend.h"
*/
import "C"

import (
	"fmt"
	"os"
	"sync/atomic"
	"unsafe"
)

func (c *cudaBackend) graphOnceCheck(code C.int) {
	if code == 0 {
		return
	}
	c.faultLatch.Observe(DeviceFaultExecution, "capture-once", int(code))
	panic(fmt.Errorf("cuda capture-once failed code=%d; no eager retry", int(code)))
}

func (c *cudaBackend) graphOnceReleaseLocked() {
	s := c.graphOnce
	if s == nil {
		return
	}
	if cudaGraphOnceOwner != c {
		panic("cuda capture-once release owner mismatch")
	}
	if s.capturing {
		panic("cuda capture-once release during capture")
	}
	c.graphOnceCheck(C.fak_graph_once_release(&s.owner))
	// Native exec is gone and its final fence succeeded before ANY unpin/free.
	for _, b := range s.buffers {
		b.graphOncePinned = false
		if b.host != 0 {
			delete(uploadCache, ucKey{b.host, b.hostDt, b.hostLo})
		}
		if b.ptr != nil {
			C.fcuda_free(b.ptr)
			b.ptr = nil
		}
		b.hostKeep = nil
	}
	c.graphOnce = nil
	cudaGraphOnceOwner = nil // only after checked native release and matching ownership
}
func (c *cudaBackend) GraphOnceInvalidate() {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	c.graphOnceReleaseLocked()
}
func (c *cudaBackend) graphOnceInvalidateKVLocked(k *cudaKV) {
	if c != nil && c.graphOnce != nil && c.graphOnce.kv == k {
		c.graphOnceReleaseLocked()
	}
}

func graphOnceSameKV(s *cudaGraphOnceState, pos int) bool {
	k := s.kv
	if k == nil || pos != s.next || pos < 0 || pos >= k.maxPos || len(k.pos) != pos {
		return false
	}
	if len(k.K) != len(s.keys) || len(k.Kraw) != len(s.raw) || len(k.V) != len(s.values) {
		return false
	}
	w := k.stride()
	for l := range k.K {
		for _, pair := range [][2]dslice{{k.K[l], s.keys[l]}, {k.Kraw[l], s.raw[l]}, {k.V[l], s.values[l]}} {
			a, b := pair[0], pair[1]
			if a.ptr != b.ptr || a.cap != b.cap || a.dtype != b.dtype || a.len != pos*w {
				return false
			}
		}
	}
	return true
}

// Replay returns used only after the checked graph fence. A refusal happens
// before input copying/enqueue; every selected failure panics without fallback.
func (c *cudaBackend) GraphOnceReplay(input Tensor, kv KVStore, pos int) (Tensor, bool) {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	s := c.graphOnce
	if s == nil {
		return Tensor{}, false
	}
	if cudaGraphOnceOwner != c || s.invalidated {
		panic("cuda capture-once invalid replay owner")
	}
	k, ok := kv.(*cudaKV)
	if !cudaAsyncGraphOnceAllowed() || !ok || k != s.kv || !graphOnceSameKV(s, pos) || s.capturing || s.output.buf == nil ||
		input.Backend() != c || input.Dtype != F32 || input.Numel() != s.input.Numel() || !graphEnabled || os.Getenv("FAK_CUDA_CAPTURE_ONCE") != "1" {
		c.graphOnceReleaseLocked()
		return Tensor{}, false
	}
	if err := c.faultLatch.Admit("capture-once"); err != nil {
		panic(err)
	}
	for b := range s.borrowed {
		if b.ptr == nil || b.invalidForSubmit() {
			c.graphOnceReleaseLocked()
			panic("cuda capture-once invalid borrowed dependency")
		}
	}
	src := c.cudaBufForSubmit(input)
	dst := c.cudaBufForSubmit(s.input)
	if src.n != dst.n || src.device != 0 || src.managed {
		c.graphOnceReleaseLocked()
		return Tensor{}, false
	}
	c.graphOnceCheck(C.fak_graph_once_copy_input(s.owner, dst.ptr, src.ptr, C.size_t(src.n)))
	c.graphOnceCheck(C.fak_graph_once_replay(s.owner, C.int(pos)))
	cudaGraphOnceWitnessReplays++
	// No host KV mutation before successful completion. Every dense layer was
	// captured exactly once and appends exactly one row on each replay.
	w := k.stride()
	for l := range k.K {
		k.K[l].len += w
		k.Kraw[l].len += w
		k.V[l].len += w
	}
	k.pos = append(k.pos, pos)
	s.next++
	atomic.AddUint64(&c.fenceGen, 1)
	cudaGraphReplays.Add(1)
	return s.output, true
}

func (c *cudaBackend) GraphOnceBegin(input Tensor, kv KVStore, pos int) bool {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	if !cudaAsyncGraphOnceAllowed() || !graphEnabled || cudaKernelProfileOn || os.Getenv("FAK_CUDA_CAPTURE_ONCE") != "1" {
		return false
	}
	if cudaGraphOnceOwner != nil && cudaGraphOnceOwner != c {
		panic("cuda capture-once shared stream already owned")
	}
	k, ok := kv.(*cudaKV)
	if !ok || k.be != c || k.maxPos <= 0 || k.maxPos >= 2147483647 || pos < 0 || pos >= k.maxPos || len(k.pos) != pos || input.Backend() != c || input.Dtype != F32 {
		return false
	}
	if c.graphOnce != nil {
		c.graphOnceReleaseLocked()
	}
	for i, p := range k.pos {
		if i != p {
			return false
		}
	}
	w := k.stride()
	if w <= 0 || w > 2147483647 || k.maxPos > int(^uint(0)>>1)/w || len(k.K) == 0 {
		return false
	}
	for l := range k.K {
		for _, d := range []dslice{k.K[l], k.Kraw[l], k.V[l]} {
			if d.dtype != F32 || d.len != pos*w || d.cap < k.maxPos*w || d.ptr == nil {
				return false
			}
		}
	}
	b := c.cudaBufForSubmit(input)
	if b.device != 0 || b.managed || b.scales != nil {
		return false
	}
	if err := c.faultLatch.Admit("capture-once"); err != nil {
		panic(err)
	}
	C.fcuda_graph_prewarm(C.int(graphPrewarmDepth))
	s := &cudaGraphOnceState{kv: k, input: input, next: pos + 1, keys: append([]dslice(nil), k.K...), raw: append([]dslice(nil), k.Kraw...), values: append([]dslice(nil), k.V...)}
	c.graphOnce = s
	cudaGraphOnceOwner = c
	c.graphOnceCheck(C.fak_graph_once_create(C.int(k.maxPos), &s.owner))
	c.graphOnceCheck(C.fak_graph_once_begin(s.owner, C.int(pos)))
	s.capturing = true
	c.capturing = true
	c.graphOncePinLocked(b)
	return true
}
func (c *cudaBackend) GraphOnceEnd() {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	s := c.graphOnce
	if s == nil || !s.capturing {
		panic("cuda capture-once end without capture")
	}
	if cudaGraphOnceOwner != c || s.invalidated {
		panic("cuda capture-once invalidated before initial launch")
	}
	for b := range s.borrowed {
		if b.ptr == nil || b.invalidForSubmit() {
			s.invalidated = true
			panic("cuda capture-once borrowed dependency lost before initial launch")
		}
	}
	s.capturing = false
	c.capturing = false
	c.graphOnceCheck(C.fak_graph_once_end(s.owner))
	cudaGraphOnceWitnessCaptures++
	atomic.AddUint64(&c.fenceGen, 1)
	cudaGraphReplays.Add(1)
}
func (c *cudaBackend) GraphOnceSave(output Tensor) {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	s := c.graphOnce
	if s == nil || s.capturing || output.Dtype != F32 {
		panic("cuda capture-once invalid output")
	}
	b := c.cudaBufForSubmit(output)
	if !b.graphOncePinned {
		panic("cuda capture-once output outside arena")
	}
	s.output = output
}
func (c *cudaBackend) GraphOnceAbort() {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	s := c.graphOnce
	if s == nil || !s.capturing {
		return
	}
	// Dedicated native abort closes capture without launching a partial token.
	s.capturing = false
	c.capturing = false
	c.graphOnceCheck(C.fak_graph_once_abort(s.owner))
	c.graphOnceReleaseLocked()
}
func (c *cudaBackend) graphOnceRoPELocked(x Tensor, heads, hd int, theta float64) bool {
	s := c.graphOnce
	if s == nil || !s.capturing {
		return false
	}
	c.graphOnceCheck(C.fak_graph_once_rope(s.owner, c.cf(x), C.int(heads), C.int(hd), C.double(theta)))
	return true
}
func (c *cudaBackend) graphOnceKVWriteLocked(d *dslice, src unsafe.Pointer, width int) bool {
	s := c.graphOnce
	if s == nil || !s.capturing {
		return false
	}
	c.graphOnceCheck(C.fak_graph_once_kv_write(s.owner, (*C.float)(d.ptr), (*C.float)(src), C.int(width)))
	return true
}
func (c *cudaBackend) graphOnceAttentionLocked(q, out Tensor, k *cudaKV, layer, heads, kvHeads, hd int, scale float32) bool {
	s := c.graphOnce
	if s == nil || !s.capturing {
		return false
	}
	if s.kv != k {
		panic("cuda capture-once foreign KV")
	}
	c.graphOnceCheck(C.fak_graph_once_attention(s.owner, c.cf(q), (*C.float)(k.K[layer].ptr), (*C.float)(k.V[layer].ptr), c.cf(out), C.int(heads), C.int(kvHeads), C.int(hd), C.float(scale)))
	return true
}

// Resolve the actual shared-stream owner, not the receiver of Free/Recycle.
// uploadCache can return the same borrowed cudaBuf to multiple backend objects.
func (c *cudaBackend) graphOnceFreeDependencyLocked(b *cudaBuf) {
	owner := cudaGraphOnceOwner
	if owner == nil {
		return
	}
	s := owner.graphOnce
	if s == nil {
		panic("cuda capture-once missing shared owner state")
	}
	if _, ok := s.borrowed[b]; !ok {
		return
	}
	if s.capturing {
		// Even a caller recovering this panic cannot launch a now-invalid capture.
		s.invalidated = true
		panic("cuda: freeing a captured borrowed weight during capture")
	}
	owner.graphOnceReleaseLocked()
}

// Qualification-only observability. Counters advance only after actual native
// completion succeeds; this seam never enables capture or synchronizes a device.
var cudaGraphOnceWitnessCaptures, cudaGraphOnceWitnessReplays uint64

type CUDAGraphOnceQualificationSnapshot struct {
	Captures, Replays                                            uint64
	Owner, Input, Output                                         uintptr
	Arena, KV                                                    []uintptr
	Next, Borrowed                                               int
	Capturing, Invalidated, ArenaPinned, DenseF32KV, SharedOwner bool
}

func (c *cudaBackend) GraphOnceQualificationSnapshot() CUDAGraphOnceQualificationSnapshot {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	out := CUDAGraphOnceQualificationSnapshot{Captures: cudaGraphOnceWitnessCaptures, Replays: cudaGraphOnceWitnessReplays}
	s := c.graphOnce
	if s == nil {
		return out
	}
	out.Owner = uintptr(s.owner)
	out.Next = s.next
	out.Borrowed = len(s.borrowed)
	out.Capturing = s.capturing
	out.Invalidated = s.invalidated
	out.SharedOwner = cudaGraphOnceOwner == c
	if b, ok := s.input.buf.(*cudaBuf); ok && b != nil {
		out.Input = uintptr(b.ptr)
	}
	if b, ok := s.output.buf.(*cudaBuf); ok && b != nil {
		out.Output = uintptr(b.ptr)
	}
	out.ArenaPinned = len(s.buffers) > 0
	for _, b := range s.buffers {
		out.Arena = append(out.Arena, uintptr(b.ptr))
		out.ArenaPinned = out.ArenaPinned && b.ptr != nil && b.graphOncePinned
	}
	out.DenseF32KV = s.kv != nil && len(s.kv.K) > 0
	if s.kv != nil {
		for l := range s.kv.K {
			for _, b := range []dslice{s.kv.K[l], s.kv.Kraw[l], s.kv.V[l]} {
				out.KV = append(out.KV, uintptr(b.ptr))
				out.DenseF32KV = out.DenseF32KV && b.ptr != nil && b.dtype == F32
			}
		}
	}
	return out
}
