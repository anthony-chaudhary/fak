//go:build cuda && cgo && fak_cuda_v41_compressor && !fak_cuda_v41_compressor_pool && fak_cuda_v41_compressor_pending

package compute

/*
#include <stddef.h>
int fak_v41_compressor_pending_retain(void *, size_t);
int fak_v41_compressor_pending_finish(void);
int fak_v41_compressor_pending_commit(void);
void fak_v41_compressor_pending_fail(void);
*/
import "C"

import (
	"fmt"
	"unsafe"
)

// All entries are actual owner handles from the connected ProjectStore caller,
// not guessed pointer identities. cudaMu protects producer scope and inventory.
type cudaV41CompressorPendingEntry struct {
	buffer          *cudaBuf
	ring, release   bool
	pointer, scales unsafe.Pointer
}

var cudaV41CompressorPending struct {
	owner             *cudaBackend
	entries           [2048]cudaV41CompressorPendingEntry
	count             int
	producing, failed bool
}

func cudaV41CompressorPendingIndex(b *cudaBuf) int {
	p := &cudaV41CompressorPending
	for i := 0; i < p.count; i++ {
		if p.entries[i].buffer == b {
			return i
		}
	}
	return -1
}

func (c *cudaBackend) cudaV41CompressorHoldLocked(b *cudaBuf, ring bool) {
	p := &cudaV41CompressorPending
	if c == nil || c != cudaDev || c.capturing || p.failed || (p.owner != nil && p.owner != c) ||
		b == nil || b.ptr == nil || b.device != 0 || b.managed || b.n <= 0 || (b.be != nil && b.be != c) {
		panic(fmt.Errorf("compressor pending: invalid owner or device storage"))
	}
	if err := b.invalidStateError("compressor pending"); err != nil {
		panic(err)
	}
	i := cudaV41CompressorPendingIndex(b)
	if i >= 0 {
		if p.entries[i].release {
			panic(fmt.Errorf("compressor pending: released operand"))
		}
		p.entries[i].ring = p.entries[i].ring || ring
	} else {
		if p.count == len(p.entries) {
			panic(fmt.Errorf("compressor pending: owner capacity exhausted"))
		}
		p.owner = c
		p.entries[p.count] = cudaV41CompressorPendingEntry{buffer: b, ring: ring, pointer: b.ptr, scales: b.scales}
		p.count++ // publish Go ownership before entering the possibly failing native seam
	}
	if status := int(C.fak_v41_compressor_pending_retain(b.ptr, C.size_t(b.n))); status != 0 {
		panic(fmt.Errorf("compressor retain: %d", status))
	}
	if b.scales != nil {
		if b.scalesN <= 0 {
			panic(fmt.Errorf("compressor scale extent invalid"))
		}
		if status := int(C.fak_v41_compressor_pending_retain(b.scales, C.size_t(b.scalesN))); status != 0 {
			panic(fmt.Errorf("compressor scales retain: %d", status))
		}
	}
}

func cudaV41CompressorFailLocked() {
	cudaV41CompressorPending.failed = true
	C.fak_v41_compressor_pending_fail()
}

func cudaV41CompressorProducedLocked(c *cudaBackend, b *cudaBuf) {
	if cudaV41CompressorPending.producing {
		c.cudaV41CompressorHoldLocked(b, false)
	}
}

func cudaV41CompressorSubmitLocked(b *cudaBuf) {
	p := &cudaV41CompressorPending
	if p.failed {
		panic(fmt.Errorf("compressor pending: terminal stream"))
	}
	if i := cudaV41CompressorPendingIndex(b); i >= 0 && p.entries[i].release {
		panic(fmt.Errorf("compressor pending: use after Free"))
	}
}

// Free of normal producer tensors queues ownership release without waiting.
// Ring replacement/teardown is an explicit completion boundary instead.
func cudaV41CompressorDeferFreeLocked(c *cudaBackend, b *cudaBuf) bool {
	p := &cudaV41CompressorPending
	if p.failed {
		// Cleanup must preserve the original selected error, not replace it.
		// Native terminal state already refuses every free; keep Go handles too.
		return true
	}
	i := cudaV41CompressorPendingIndex(b)
	if i < 0 {
		return false
	}
	if p.entries[i].ring {
		cudaV41CompressorDrainLocked(c)
		return false
	}
	p.entries[i].release = true
	return true
}

func cudaV41CompressorReadBeginLocked(c *cudaBackend) bool {
	p := &cudaV41CompressorPending
	if p.count == 0 {
		return false
	}
	if p.owner != c || p.failed || c.capturing {
		cudaV41CompressorFailLocked()
		panic(fmt.Errorf("compressor pending: invalid completion owner"))
	}
	for i := 0; i < p.count; i++ {
		e := &p.entries[i]
		if e.buffer.ptr != e.pointer || e.buffer.scales != e.scales {
			cudaV41CompressorFailLocked()
			panic(fmt.Errorf("compressor pending: external owner detached retained storage"))
		}
	}
	cudaV41CompressorPeerDrainLocked(c)
	if status := int(C.fak_v41_compressor_pending_finish()); status != 0 {
		cudaV41CompressorFailLocked()
		panic(fmt.Errorf("compressor pending completion: %d", status))
	}
	return true
}

func cudaV41CompressorReadEndLocked(c *cudaBackend, pending, success bool) {
	if !pending {
		return
	}
	if !success {
		cudaV41CompressorFailLocked()
		return
	}
	if status := int(C.fak_v41_compressor_pending_commit()); status != 0 {
		cudaV41CompressorFailLocked()
		panic(fmt.Errorf("compressor pending commit: %d", status))
	}
	p := &cudaV41CompressorPending
	retired := p.entries
	count := p.count
	clear(p.entries[:])
	p.count = 0
	p.owner = nil
	for i := 0; i < count; i++ {
		if retired[i].release {
			c.cudaFreeLocked(Tensor{buf: retired[i].buffer})
		}
	}
}

func cudaV41CompressorDrainLocked(c *cudaBackend) {
	pending := cudaV41CompressorReadBeginLocked(c)
	cudaV41CompressorReadEndLocked(c, pending, true)
}

// Connected producer entry: no other goroutine can enqueue inside this scope.
// Existing MatMul arithmetic is reused; devTr publishes its output owner before
// MatMul crosses the native ABI. No unknown caller submissions are adopted.
func (c *cudaBackend) V41CompressorRingProjectStore(kv, scores, wkv, wscore, x Tensor, slot, ratio, width int, emit bool) (out Tensor, err error) {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	defer func() {
		cudaV41CompressorPending.producing = false
		if recovered := recover(); recovered != nil {
			err = fmt.Errorf("compressor ProjectStore: %v", recovered)
		}
		if err != nil && cudaV41CompressorPending.count > 0 {
			cudaV41CompressorFailLocked()
			out = Tensor{}
		}
	}()
	if cudaQ4KAsyncReleaseBlockedLocked() {
		return Tensor{}, fmt.Errorf("compressor ProjectStore: terminal Q4 owner")
	}
	if err = c.cudaV41CompressorState("V41CompressorRingProjectStore"); err != nil {
		return Tensor{}, err
	}
	if err = ValidateV41CompressorRingAlloc(ratio, width); err != nil {
		return Tensor{}, err
	}
	if slot < 0 || slot >= ratio || x.Dtype != F32 || x.Layout != RowMajor || len(x.Shape) != 1 || x.Shape[0] <= 0 {
		return Tensor{}, ErrV41CompressorRingGeometry
	}
	for _, w := range []Tensor{wkv, wscore} {
		if len(w.Shape) != 2 || w.Shape[0] != width || w.Shape[1] != x.Shape[0] {
			return Tensor{}, ErrV41CompressorRingGeometry
		}
	}
	probe := Tensor{Dtype: F32, Layout: RowMajor, Shape: []int{width}}
	if err = ValidateV41CompressorRingStore(kv, scores, probe, probe, slot, ratio, width); err != nil {
		return Tensor{}, err
	}
	for _, t := range []Tensor{kv, scores, wkv, wscore, x} {
		b, ok := t.buf.(*cudaBuf)
		if !ok || t.be != c {
			return Tensor{}, fmt.Errorf("compressor producer owner mismatch")
		}
		c.cudaV41CompressorHoldLocked(b, t.Buf() == kv.Buf() || t.Buf() == scores.Buf())
	}
	cudaV41CompressorPending.producing = true
	a := c.cudaMatMulLocked(wkv, x)
	b := c.cudaMatMulLocked(wscore, x)
	cudaV41CompressorPending.producing = false
	if a.Dtype != F32 || b.Dtype != F32 || a.Numel() != width || b.Numel() != width {
		return Tensor{}, fmt.Errorf("compressor producer shape mismatch")
	}
	out, err = c.cudaV41CompressorRingStoreLocked(kv, scores, a, b, slot, ratio, width, emit)
	if err == nil {
		cudaV41CompressorDeferFreeLocked(c, a.buf.(*cudaBuf))
		cudaV41CompressorDeferFreeLocked(c, b.buf.(*cudaBuf))
	}
	return out, err
}

// Commit must not drop ring references while a peer still owns submitted work.
// On any peer failure, poison ring admission before unwinding the ORIGINAL panic.
// No release occurs here; even partially completed peer batches keep live buffers.
func cudaV41CompressorPeerDrainLocked(c *cudaBackend) {
	complete := false
	defer func() {
		if !complete {
			cudaV41CompressorFailLocked()
		}
	}()
	cudaQ4KAsyncRequireDrainLocked(c)
	complete = true
}
