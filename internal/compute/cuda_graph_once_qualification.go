//go:build cuda && cgo && fak_cuda_capture_once_qualification

package compute

/*
#include "cuda_backend.h"
// Qualification-local ABI; not a general arbitrary-pointer free.
int fak_graph_once_arena_release_checked(void *expected, size_t bytes, int f32_input);
int fak_graph_once_arena_failed(int status);
*/
import "C"

import (
	"fmt"
	"sync/atomic"
	"unsafe"
)

// Shared-stream runtime consent is explicit and defaults OFF. Access, including
// both per-token reads below, is serialized by the existing cudaMu.
var cudaCaptureOnceMode cudaCaptureOnceConfiguration

func (c *cudaBackend) configureCUDACaptureOnce(enabled bool) error {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	if c == nil || c != cudaDev {
		return ErrCUDACaptureOnceUnsupported
	}
	ownerLive := cudaGraphOnceOwner != nil || c.graphOnce != nil || c.capturing
	return cudaCaptureOnceMode.set(enabled, cudaAsyncGraphOnceAllowed(), graphEnabled, cudaKernelProfileOn, ownerLive)
}

func (c *cudaBackend) graphOnceCheck(code C.int) {
	if code == 0 {
		return
	}
	c.faultLatch.Observe(DeviceFaultExecution, "capture-once", int(code))
	panic(fmt.Errorf("cuda capture-once failed code=%d; no eager retry", int(code)))
}

// A weight-class arena member is only the F32 input already checked by Begin.
// Quantized weights, secondary devices, side channels and residency-budget
// owners are never silently passed to this dedicated physical retirement seam.
func graphOnceArenaBufferLegal(b *cudaBuf, input bool) bool {
	return b != nil && b.ptr != nil && b.n > 0 && b.device == 0 && b.scales == nil && b.scalesN == 0 &&
		b.budgetedWeightBytes == 0 && !b.managedWeight &&
		(b.class == MemoryScratchpad || b.class == MemoryActivation || (input && b.class == MemoryWeights && !b.managed)) &&
		(b.host == 0 || (input && b.hostDt == F32))
}

// Metadata rebinding cannot be repaired by releasing whatever a mutable buffer
// now points at. Keep the original receipt/root and close shared admission before
// launch, input copy or retirement. Native tickets detect tracked reservation
// replacement, not untracked reuse, content changes or device-context identity.
func (c *cudaBackend) graphOnceRequireDependenciesLocked() {
	s := c.graphOnce
	if s == nil || !s.graphOnceDependenciesValid() {
		if s != nil {
			s.invalidated = true
			s.dependencyInvalidated = true
		}
		c.graphOnceCheck(C.fak_graph_once_arena_failed(-75092))
	}
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
	if err := c.faultLatch.Admit("capture-once-release"); err != nil {
		panic(err)
	}
	c.graphOnceRequireDependenciesLocked()
	// Root the remaining arena in the shared owner even if a caller catches a
	// cleanup panic. An interrupted/partial release can never resume replay.
	s.invalidated = true
	for _, b := range s.buffers {
		if b != nil && b.ptr == nil && !b.graphOncePinned {
			continue
		} // already retired
		if !graphOnceArenaBufferLegal(b, b == s.input.buf) || !b.graphOncePinned {
			c.graphOnceCheck(C.fak_graph_once_arena_failed(-75091))
		}
	}
	c.graphOnceCheck(C.fak_graph_once_release(&s.owner))
	// Native graph, exec, fixed quant scratch and position are gone and the final
	// fence succeeded. Each arena pointer/root remains pinned until ITS free succeeds.
	for _, b := range s.buffers {
		if b.ptr == nil {
			continue
		}
		input := C.int(0)
		if b == s.input.buf {
			input = 1
		}
		c.graphOnceCheck(C.fak_graph_once_arena_release_checked(b.ptr, C.size_t(b.n), input))
		b.ptr = nil
		b.graphOncePinned = false
		if b.host != 0 {
			key := ucKey{b.host, b.hostDt, b.hostLo}
			if cached, ok := uploadCache[key]; ok && cached.buf == b {
				delete(uploadCache, key)
			}
		}
		b.hostKeep = nil
	}
	c.graphOnce = nil
	cudaGraphOnceOwner = nil // only after every exact pointer has a successful receipt
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
	if _, ok := s.graphOnceKVSlotsValid(); !ok {
		return false
	}
	k := s.kv
	if pos != s.next || pos < 0 || pos >= s.kvCapacity || len(k.pos) != pos {
		return false
	}
	w := s.kvHeads * s.kvHeadDim // bounded and frozen by the one-shot binding
	for l := 0; l < s.kvLayers; l++ {
		for _, row := range [3]dslice{k.K[l], k.Kraw[l], k.V[l]} {
			if row.len != pos*w {
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
	if cudaGraphOnceOwner != c {
		panic("cuda capture-once invalid replay owner")
	}
	c.graphOnceRequireDependenciesLocked()
	if s.invalidated {
		panic("cuda capture-once invalid replay owner")
	}
	k, ok := kv.(*cudaKV)
	if !cudaAsyncGraphOnceAllowed() || !ok || k != s.kv || !graphOnceSameKV(s, pos) || s.capturing || s.output.buf == nil ||
		input.Backend() != c || input.Dtype != F32 || input.Numel() != s.input.Numel() || !graphEnabled || !cudaCaptureOnceMode.enabled {
		c.graphOnceReleaseLocked()
		return Tensor{}, false
	}
	if err := c.faultLatch.Admit("capture-once"); err != nil {
		panic(err)
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
	if !cudaAsyncGraphOnceAllowed() || !graphEnabled || cudaKernelProfileOn || !cudaCaptureOnceMode.enabled {
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
	s := &cudaGraphOnceState{kv: k, input: input, next: pos + 1}
	if !s.graphOnceBindKVLocked(c, pos) {
		return false
	}
	b := c.cudaBufForSubmit(input)
	if b.device != 0 || b.managed || b.scales != nil || !graphOnceArenaBufferLegal(b, true) {
		return false
	}
	if err := c.faultLatch.Admit("capture-once"); err != nil {
		panic(err)
	}
	C.fcuda_graph_prewarm(C.int(graphPrewarmDepth))
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
	if cudaGraphOnceOwner != c {
		panic("cuda capture-once invalidated before initial launch")
	}
	c.graphOnceRequireDependenciesLocked()
	if s.invalidated {
		panic("cuda capture-once invalidated before initial launch")
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
	c.graphOnceRequireDependenciesLocked()
	b := c.cudaBufForSubmit(output)
	d, ok := s.dependencies[b]
	if !ok || !d.pinned || !d.matches(b) {
		s.invalidated = true
		s.dependencyInvalidated = true
		c.graphOnceCheck(C.fak_graph_once_arena_failed(-75092))
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
