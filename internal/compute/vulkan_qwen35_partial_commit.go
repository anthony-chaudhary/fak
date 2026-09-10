//go:build vulkan && (windows || linux) && cgo

package compute

/*
#include <stdlib.h>
#include "vulkan_backend.h"
*/
import "C"

import (
	"fmt"
	"unsafe"
)

type vulkanQwen35ReplayProjection struct {
	layer                 int
	mixed, z, beta, alpha Tensor
}

// vulkanQwen35PrefixReplay owns only detached transient projection buffers.
// Layer weights, live state, and KV remain owned by the session. The pre-round
// states supplied to CommitPrefix remain owned by the model snapshot.
type vulkanQwen35PrefixReplay struct {
	backend       *vulkanBackend
	kv            *vulkanKV
	startPos      int
	tokens        int
	kvWidth       int
	convDim       int
	valueDim      int
	numKeyHeads   int
	numValueHeads int
	keyHeadDim    int
	valueHeadDim  int
	convKernel    int
	rmsEpsilon    float32
	layers        []Qwen35SequenceLayer
	states        []Qwen35SequenceState
	projections   []vulkanQwen35ReplayProjection
	owned         []*vulkanBuf
	closed        bool
}

func (*vulkanBackend) Qwen35SequencePrefixReplayPath() string {
	return Qwen35SequencePrefixReplayPath
}

// qwen35DetachReplayBuffersLocked transfers transient buffer ownership to the
// checkpoint. The caller holds vulkanMu and has already fenced all users.
func (v *vulkanBackend) qwen35DetachReplayBuffersLocked(projections []vulkanQwen35ReplayProjection) ([]*vulkanBuf, error) {
	want := make(map[*vulkanBuf]bool, len(projections)*4)
	for _, p := range projections {
		for _, t := range []Tensor{p.mixed, p.z, p.beta, p.alpha} {
			b, ok := t.buf.(*vulkanBuf)
			if !ok || b == nil || b.ptr == nil || want[b] {
				return nil, qwen35VulkanSequenceError("prefix-replay-capture", p.layer, "projection buffer is missing, invalid, or aliased")
			}
			want[b] = true
		}
	}
	found := 0
	for _, b := range v.transient {
		if want[b] {
			found++
		}
	}
	if found != len(want) {
		return nil, qwen35VulkanSequenceError("prefix-replay-capture", -1, "projection buffer is outside request transient ownership")
	}
	retained := v.transient[:0]
	owned := make([]*vulkanBuf, 0, len(want))
	for _, b := range v.transient {
		if want[b] {
			owned = append(owned, b)
			delete(want, b)
			continue
		}
		retained = append(retained, b)
	}
	v.transient = retained
	return owned, nil
}

func qwen35ReplayProjectionTensors(projections []vulkanQwen35ReplayProjection) []Tensor {
	keep := make([]Tensor, 0, len(projections)*4)
	for _, p := range projections {
		keep = append(keep, p.mixed, p.z, p.beta, p.alpha)
	}
	return keep
}

func (p *vulkanQwen35PrefixReplay) Close() {
	if p == nil || p.backend == nil {
		return
	}
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	if p.closed {
		return
	}
	for _, b := range p.owned {
		if b != nil && b.ptr != nil {
			p.backend.recycleTransientLocked(b)
			b.ptr = nil
		}
	}
	p.owned = nil
	p.projections = nil
	p.layers = nil
	p.states = nil
	p.closed = true
}

func (p *vulkanQwen35PrefixReplay) CommitPrefix(accepted int, before []Qwen35SequenceState) error {
	if p == nil || p.backend == nil {
		return qwen35VulkanSequenceError("prefix-replay-preflight", -1, "nil prefix replay checkpoint")
	}
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	if err := p.validateCommitLocked(accepted, before); err != nil {
		return err
	}
	if status := int(C.fvk_submission_status()); status != 0 {
		return qwen35VulkanSequenceError("prefix-replay-admission", -1, fmt.Sprintf("Vulkan submission fault %d", status))
	}

	start := len(p.backend.transient)
	batchOpen := false
	defer func() {
		if batchOpen {
			C.fvk_batch_flush_status()
		}
		p.backend.qwen35VulkanSequenceReleaseLocked(start)
	}()

	C.fvk_batch_begin()
	batchOpen = true
	var operationErr error
	for _, projection := range p.projections {
		live := p.states[projection.layer]
		base := before[projection.layer]
		liveConv := live.Conv.buf.(*vulkanBuf)
		liveRecurrent := live.Recurrent.buf.(*vulkanBuf)
		baseConv := base.Conv.buf.(*vulkanBuf)
		baseRecurrent := base.Recurrent.buf.(*vulkanBuf)
		C.fvk_d2d(liveConv.ptr, baseConv.ptr, C.size_t(liveConv.n))
		C.fvk_d2d(liveRecurrent.ptr, baseRecurrent.ptr, C.size_t(liveRecurrent.n))

		core, _ := p.backend.devTr([]int{accepted, p.valueDim}, F32)
		layer := p.layers[projection.layer]
		code := C.fvk_qwen35_gdn_preprojected_f32(
			p.backend.vp(projection.mixed), p.backend.vp(projection.z), p.backend.vp(projection.beta), p.backend.vp(projection.alpha),
			p.backend.vp(layer.GDNConv), p.backend.vp(layer.GDNALog), p.backend.vp(layer.GDNDTBias), p.backend.vp(layer.GDNNorm),
			liveConv.ptr, liveRecurrent.ptr, p.backend.vp(core),
			C.int(accepted), C.int(p.convDim), C.int(p.numKeyHeads), C.int(p.numValueHeads), C.int(p.keyHeadDim), C.int(p.valueHeadDim), C.int(p.convKernel), C.float(p.rmsEpsilon),
		)
		if code != 0 {
			operationErr = qwen35VulkanSequenceError("prefix-replay-gdn", projection.layer, fmt.Sprintf("Vulkan operation failed with status %d", int(code)))
			break
		}
	}
	status := int(C.fvk_batch_flush_status())
	batchOpen = false
	if operationErr != nil {
		return operationErr
	}
	if status != 0 {
		return qwen35VulkanSequenceError("prefix-replay-fence", -1, fmt.Sprintf("Vulkan submission failed with status %d", status))
	}

	// All validation and device work completed before this infallible metadata
	// cut. Rejected suffix bytes remain unreachable and may be overwritten by
	// the next sole-owner append after the transaction snapshot closes.
	end := (p.startPos + accepted) * p.kvWidth
	activeAttention := len(p.layers) / 4
	for i := 0; i < activeAttention; i++ {
		p.kv.K[i].len = end
		p.kv.Kraw[i].len = end
		p.kv.V[i].len = end
	}
	p.kv.pos = p.kv.pos[:p.startPos+accepted]
	return nil
}

func (p *vulkanQwen35PrefixReplay) validateCommitLocked(accepted int, before []Qwen35SequenceState) error {
	fail := func(layer int, reason string) error {
		return qwen35VulkanSequenceError("prefix-replay-preflight", layer, reason)
	}
	if p.closed {
		return fail(-1, "prefix replay checkpoint is closed")
	}
	if accepted < 1 || accepted >= p.tokens {
		return fail(-1, fmt.Sprintf("accepted prefix %d must be between 1 and %d", accepted, p.tokens-1))
	}
	if len(before) != len(p.layers) || len(p.states) != len(p.layers) || len(p.projections) == 0 {
		return fail(-1, "layer, state, or projection count mismatch")
	}
	activeAttention := len(p.layers) / 4
	if p.kv == nil || p.kv.be != p.backend || p.kv.Len() != p.startPos+p.tokens || len(p.kv.K) < activeAttention || len(p.kv.Kraw) < activeAttention || len(p.kv.V) < activeAttention {
		return fail(-1, "live KV ownership, geometry, or speculative length mismatch")
	}
	for i, pos := range p.kv.pos {
		if pos != i {
			return fail(-1, "live KV positions are not a contiguous prefix")
		}
	}
	fullEnd := (p.startPos + p.tokens) * p.kvWidth
	seen := make(map[unsafe.Pointer]bool, len(p.projections)*8)
	owned := make(map[*vulkanBuf]bool, len(p.owned))
	for _, b := range p.owned {
		if b == nil || b.ptr == nil || owned[b] {
			return fail(-1, "retained projection ownership is invalid")
		}
		owned[b] = true
	}
	for i := 0; i < activeAttention; i++ {
		for _, cache := range []*vslice{&p.kv.K[i], &p.kv.Kraw[i], &p.kv.V[i]} {
			if cache.ptr == nil || cache.len != fullEnd || cache.cap < cache.len || seen[cache.ptr] {
				return fail(-1, "live KV tail is malformed or aliased")
			}
			seen[cache.ptr] = true
		}
	}
	wantLayer := 0
	for _, projection := range p.projections {
		for wantLayer < len(p.layers) && !p.layers[wantLayer].Linear {
			wantLayer++
		}
		if wantLayer >= len(p.layers) || projection.layer != wantLayer {
			return fail(projection.layer, "retained projections do not cover linear layers in order")
		}
		wantLayer++
		shapes := []struct {
			t Tensor
			s []int
		}{{projection.mixed, []int{p.tokens, p.convDim}}, {projection.z, []int{p.tokens, p.valueDim}}, {projection.beta, []int{p.tokens, p.numValueHeads}}, {projection.alpha, []int{p.tokens, p.numValueHeads}}}
		for _, item := range shapes {
			b, ok := item.t.buf.(*vulkanBuf)
			floats, shapeOK := qwen35VulkanSequenceSize(item.s...)
			if !ok || !shapeOK || b == nil || b.ptr == nil || b.n != floats*F32.Bytes() || !owned[b] || item.t.Backend() != p.backend || item.t.Dtype != F32 || item.t.Layout != RowMajor || !qwen35SameShape(item.t.Shape, item.s) || seen[b.ptr] {
				return fail(projection.layer, "retained projection shape, ownership, or alias is invalid")
			}
			seen[b.ptr] = true
		}
		statePairs := []struct {
			live, base Tensor
			shape      []int
		}{
			{p.states[projection.layer].Conv, before[projection.layer].Conv, []int{p.convKernel - 1, p.convDim}},
			{p.states[projection.layer].Recurrent, before[projection.layer].Recurrent, []int{p.numValueHeads, p.keyHeadDim, p.valueHeadDim}},
		}
		for _, pair := range statePairs {
			live, liveOK := pair.live.buf.(*vulkanBuf)
			base, baseOK := pair.base.buf.(*vulkanBuf)
			floats, shapeOK := qwen35VulkanSequenceSize(pair.shape...)
			wantBytes := floats * F32.Bytes()
			if !liveOK || !baseOK || !shapeOK || live == nil || base == nil || live.ptr == nil || base.ptr == nil || live.ptr == base.ptr || live.n != wantBytes || base.n != wantBytes || pair.live.Backend() != p.backend || pair.base.Backend() != p.backend || pair.live.Dtype != F32 || pair.base.Dtype != F32 || pair.live.Layout != RowMajor || pair.base.Layout != RowMajor || !qwen35SameShape(pair.live.Shape, pair.shape) || !qwen35SameShape(pair.base.Shape, pair.shape) || seen[live.ptr] || seen[base.ptr] {
				return fail(projection.layer, "live and baseline GDN states are invalid or aliased")
			}
			seen[live.ptr] = true
			seen[base.ptr] = true
		}
	}
	for ; wantLayer < len(p.layers); wantLayer++ {
		if p.layers[wantLayer].Linear {
			return fail(wantLayer, "retained projections omit a linear layer")
		}
	}
	if len(owned) != len(p.projections)*4 {
		return fail(-1, "retained projection ownership count mismatch")
	}
	return nil
}

func qwen35SameShape(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
