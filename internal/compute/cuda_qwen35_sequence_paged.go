//go:build cuda && cgo

package compute

/*
#include "cuda_backend.h"
#include "cuda_kv_precision.h"
extern int fak_qwen35_causal_attention_panel_f32(const float*, const float*, const float*, float*, int, int, int, int, int, float);
*/
import "C"

import (
	"fmt"
	"runtime"
	"unsafe"
)

// This operation owns preparation and all paged sequence mutations under the
// existing cudaMu/ForwardExecutionOwner, including eager independent batch lanes.
// It is not a graph capability.
// Retained operations are rooted before any native allocation/submission.
type cudaQwen35PagedSequence struct {
	c                                                         *cudaBackend
	k                                                         *cudaPagedKV
	start, tokens, end, width                                 int
	table                                                     []int32
	positions                                                 []int
	frontiers                                                 []int
	blocks                                                    pagedKVBlocks
	planes                                                    [3][]unsafe.Pointer
	capacity                                                  int
	tableBuf                                                  *cudaBuf
	gatherBuf                                                 *cudaBuf
	fresh                                                     []*cudaBuf
	old                                                       []cudaQwen35PagedRetirement
	tablePin                                                  runtime.Pinner
	pinned                                                    bool
	started, prepared, executing, drained, retired, completed bool
	failure                                                   error
	next                                                      *cudaQwen35PagedSequence
}
type cudaQwen35PagedRetirement struct {
	ptr       unsafe.Pointer
	bytes     int
	attempted bool
	retired   bool
}

var cudaQwen35PagedOwners *cudaQwen35PagedSequence

func (p *cudaQwen35PagedSequence) reject(reason string) error {
	return qwen35SequenceFailure("paged-kv-preflight", -1, reason)
}
func (p *cudaQwen35PagedSequence) validateHostLocked(req Qwen35SequencePrefillRequest) error {
	if p == nil || p.c == nil || p.k == nil || p.k.pool == nil || p.k.pool.be != p.c {
		return qwen35SequenceFailure("paged-kv-preflight", -1, "paged KV backend identity mismatch")
	}
	k, pool := p.k, p.k.pool
	if p.c.capturing || p.c.graphOnce != nil || p.c.promptPanel || cudaKernelProfileOn ||
		req.CapturePrefixReplay || len(req.FeatureLayers) != 0 {
		return p.reject("paged sequence requires eager, profile-off, non-replay execution")
	}
	layers := len(req.Layers) / 4
	if k.cfg.Precision != KVPrecisionF32 || k.cfg.NumLayers != layers ||
		k.cfg.NumKVHeads != req.NumKVHeads || k.cfg.HeadDim != req.HeadDim ||
		pool.layers != layers || pool.nKV != req.NumKVHeads || pool.hd != req.HeadDim ||
		pool.blockTokens != CUDAPagedKVBlockTokens || len(k.lens) != layers ||
		len(pool.K) != layers || len(pool.Kraw) != layers || len(pool.V) != layers {
		return p.reject("paged KV compact layers or F32 geometry mismatch")
	}
	p.start, p.tokens, p.end, p.width = req.StartPos, len(req.TokenIDs), req.StartPos+len(req.TokenIDs), req.NumKVHeads*req.HeadDim
	if p.width <= 0 || p.end <= 0 || p.end > int(qwen35GDNMaxCInt)/p.width ||
		pool.blockFloats != pool.blockTokens*p.width || pool.capBlocks < 0 ||
		pool.capBlocks > int(qwen35GDNMaxCInt)/pool.blockFloats ||
		len(pool.blocks.refs) != pool.capBlocks || pool.blocks.live < 0 || pool.blocks.live > pool.capBlocks ||
		len(k.pos) != p.start || len(k.table) != (p.start+pool.blockTokens-1)/pool.blockTokens {
		return p.reject("paged KV extent/frontier exceeds native bounds")
	}
	for i, pos := range k.pos {
		if pos != i {
			return p.reject("paged sequence requires an unevicted canonical prefix")
		}
	}
	for _, n := range k.lens {
		if n != p.start {
			return p.reject("paged compact layer frontier mismatch")
		}
	}
	free := make(map[int32]bool)
	for _, id := range pool.blocks.free {
		if id < 0 || int(id) >= pool.capBlocks || free[id] || pool.blocks.refs[id] != 0 {
			return p.reject("malformed paged free list")
		}
		free[id] = true
	}
	live := 0
	for id, refs := range pool.blocks.refs {
		if refs < 0 || (refs == 0) != free[int32(id)] {
			return p.reject("malformed paged reference inventory")
		}
		if refs > 0 {
			live++
		}
	}
	if live != pool.blocks.live {
		return p.reject("paged live count disagrees with references")
	}
	seen := make(map[int32]bool)
	for _, id := range k.table {
		if id < 0 || int(id) >= pool.capBlocks || pool.blocks.refs[id] <= 0 || seen[id] {
			return p.reject("malformed paged table")
		}
		seen[id] = true
	}
	if pool.copyFailure.Load() != 0 {
		return p.reject("paged pool is terminal")
	}
	for _, plane := range [][]unsafe.Pointer{pool.Kraw, pool.K, pool.V} {
		for _, ptr := range plane {
			if (pool.capBlocks == 0) != (ptr == nil) {
				return p.reject("paged arena capacity/base mismatch")
			}
		}
	}
	if k.dTable == nil && k.dTableCap != 0 || k.dTable != nil && (k.dTableCap < len(k.table) || k.dTableCap <= 0 || k.dTableCap > int(qwen35GDNMaxCInt)/4) {
		return p.reject("paged device table capacity mismatch")
	}
	// Pre-existing borrowed tensor views must not alias an arena that preparation
	// may replace or a later layer will mutate. Check real byte extents, not just bases.
	type span struct {
		base  uintptr
		bytes int
	}
	var owners []span
	add := func(ptr unsafe.Pointer, bytes int) bool {
		if ptr == nil {
			return bytes == 0
		}
		base := uintptr(ptr)
		if bytes <= 0 || uintptr(bytes) > ^uintptr(0)-base {
			return false
		}
		for _, s := range owners {
			if base < s.base+uintptr(s.bytes) && s.base < base+uintptr(bytes) {
				return false
			}
		}
		owners = append(owners, span{base, bytes})
		return true
	}
	arenaBytes, ok := qwen35GDNShapeBytes([]int{pool.capBlocks, pool.blockFloats}, 4)
	if pool.capBlocks == 0 {
		arenaBytes, ok = 0, true
	}
	if !ok {
		return p.reject("paged arena bytes overflow")
	}
	for _, plane := range [][]unsafe.Pointer{pool.Kraw, pool.K, pool.V} {
		for _, ptr := range plane {
			if !add(ptr, arenaBytes) {
				return p.reject("paged arena ownership aliases")
			}
		}
	}
	if k.dTable != nil && !add(k.dTable, k.dTableCap*4) {
		return p.reject("paged table aliases arena")
	}
	inputs := []Tensor{req.TokenEmbedding, req.OutputNorm, req.Output}
	for i, l := range req.Layers {
		inputs = append(inputs, l.InputNorm, l.PostNorm, l.Gate, l.Up, l.Down, l.Q, l.K, l.V, l.O, l.QNorm, l.KNorm,
			l.GDNInQKV, l.GDNInZ, l.GDNInB, l.GDNInA, l.GDNConv, l.GDNALog, l.GDNDTBias, l.GDNNorm, l.GDNOut)
		if i < len(req.States) {
			inputs = append(inputs, req.States[i].Conv, req.States[i].Recurrent)
		}
	}
	for _, t := range inputs {
		b, ok := t.buf.(*cudaBuf)
		if !ok || b == nil || b.ptr == nil {
			continue
		}
		if b.scalesN < 0 || (b.scales == nil) != (b.scalesN == 0) {
			return p.reject("operand scale pointer/extent mismatch")
		}
		if t.Dtype == Q8_0 || t.Dtype == Q2_0 {
			if t.Quant == nil || t.Quant.Block <= 0 || len(t.Shape) != 2 ||
				t.Shape[1] <= 0 || t.Shape[1]%t.Quant.Block != 0 || b.scales == nil {
				return p.reject("quantized operand scale geometry mismatch")
			}
			need, ok := qwen35GDNShapeBytes([]int{t.Shape[0], t.Shape[1] / t.Quant.Block}, 4)
			if !ok || b.scalesN < need {
				return p.reject("quantized operand scale capacity mismatch")
			}
		}
		for _, part := range []struct {
			name  string
			ptr   unsafe.Pointer
			bytes int
		}{
			{"operand", b.ptr, b.n}, {"operand scale", b.scales, b.scalesN},
		} {
			if part.ptr == nil {
				continue
			}
			base := uintptr(part.ptr)
			if part.bytes <= 0 || uintptr(part.bytes) > ^uintptr(0)-base {
				return p.reject(part.name + " byte range overflow")
			}
			for _, s := range owners {
				if base < s.base+uintptr(s.bytes) && s.base < base+uintptr(part.bytes) {
					return p.reject(part.name + " aliases paged owner")
				}
			}
		}
	}
	return nil
}

func (p *cudaQwen35PagedSequence) failLocked(site string, code int) error {
	if code == 0 {
		code = -75821
	}
	if p.failure == nil {
		p.failure = &CUDAOpError{Op: "Qwen35SequencePrefill", Site: site, Code: code, Class: MemoryKVCache, Msg: "paged sequence stopped; exact owners retained"}
	}
	C.fcuda_kv_snapshot_failed(C.int(code))
	p.k.pool.stopCopyLocked(int32(code), p.k)
	p.c.faultLatch.Observe(DeviceFaultExecution, site, code)
	return p.failure
}
func (p *cudaQwen35PagedSequence) statusLocked(site string, code int) error {
	if code == 0 {
		code = int(C.fak_v41_checked_stream_terminal())
	}
	if code != 0 {
		return p.failLocked(site, code)
	}
	return nil
}
func (p *cudaQwen35PagedSequence) abortLocked() {
	if p == nil || !p.started || p.completed {
		return
	}
	// No recover: preserve panic(nil)/Goexit. Unknown completion must close native
	// cleanup before the enclosing sequence releases any transient.
	if !p.drained || p.executing {
		p.failLocked("paged-sequence-unwind", -75822)
	} else {
		p.k.pool.stopCopyLocked(-75822, p.k)
	}
}
func (p *cudaQwen35PagedSequence) allocateLocked(bytes int, site string) (*cudaBuf, error) {
	b, err := p.c.dallocDeviceOnlyClass(bytes, MemoryKVCache, site)
	if err != nil {
		if p.failure == nil {
			p.failure = err
		}
		return nil, err
	}
	p.fresh = append(p.fresh, b) // root before the next fallible boundary
	if err := p.statusLocked(site, int(C.fcuda_paged_kv_owner_checked(b.ptr, C.size_t(bytes)))); err != nil {
		return nil, err
	}
	return b, nil
}
func (p *cudaQwen35PagedSequence) copyLocked(dst, src unsafe.Pointer, bytes int) error {
	p.drained = false
	return p.statusLocked("paged-reserve-copy", int(C.fcuda_kv_clone_d2d_checked(dst, src, C.size_t(bytes))))
}
func (p *cudaQwen35PagedSequence) prepareLocked() (err error) {
	if p.started {
		return p.reject("paged operation preparation repeated")
	}
	p.started = true
	p.next, cudaQwen35PagedOwners = cudaQwen35PagedOwners, p
	normal := false
	defer func() {
		if !normal {
			p.abortLocked()
		}
	}()
	k, pool := p.k, p.k.pool
	if err = p.statusLocked("paged-reserve-entry", 0); err != nil {
		return err
	}
	oldBytes := pool.capBlocks * pool.blockFloats * 4
	for _, plane := range [][]unsafe.Pointer{pool.Kraw, pool.K, pool.V} {
		for _, ptr := range plane {
			if ptr != nil {
				if err = p.statusLocked("paged-arena-owner", int(C.fcuda_paged_kv_owner_checked(ptr, C.size_t(oldBytes)))); err != nil {
					return err
				}
			}
		}
	}
	if k.dTable != nil {
		if err = p.statusLocked("paged-table-owner", int(C.fcuda_paged_kv_owner_checked(k.dTable, C.size_t(k.dTableCap*4)))); err != nil {
			return err
		}
	}
	p.blocks = pagedKVBlocks{refs: append([]int32(nil), pool.blocks.refs...), free: append([]int32(nil), pool.blocks.free...), live: pool.blocks.live}
	p.table = append([]int32(nil), k.table...)
	pages := (p.end + pool.blockTokens - 1) / pool.blockTokens
	sharedTail := p.start%pool.blockTokens != 0 && p.blocks.shared(p.table[len(p.table)-1])
	need := pages - len(p.table)
	if sharedTail {
		need++
	}
	p.capacity = pool.capBlocks
	if need > len(p.blocks.free) {
		p.capacity = pagedKVGrowBlocks(pool.capBlocks, pool.capBlocks+need-len(p.blocks.free))
		if p.capacity <= 0 || p.capacity > int(qwen35GDNMaxCInt)/pool.blockFloats {
			return p.reject("reserved arena exceeds native indexing")
		}
		p.blocks.grow(p.capacity)
	}
	tailOld, tailNew := int32(-1), int32(-1)
	if sharedTail {
		tailOld = p.table[len(p.table)-1]
		var ok bool
		tailNew, ok = p.blocks.alloc()
		if !ok {
			return p.reject("staged COW reservation exhausted")
		}
		p.table[len(p.table)-1] = tailNew
		p.blocks.release(tailOld)
	}
	for len(p.table) < pages {
		id, ok := p.blocks.alloc()
		if !ok {
			return p.reject("staged page reservation exhausted")
		}
		p.table = append(p.table, id)
	}
	p.positions = make([]int, p.end)
	for i := range p.positions {
		p.positions[i] = i
	}
	p.frontiers = append([]int(nil), k.lens...)
	oldPlanes := [3][]unsafe.Pointer{pool.Kraw, pool.K, pool.V}
	sites := [3]string{"qwen35-paged-raw-arena", "qwen35-paged-key-arena", "qwen35-paged-value-arena"}
	bytes, ok := qwen35GDNShapeBytes([]int{p.capacity, pool.blockFloats}, 4)
	if !ok {
		return p.reject("staged arena byte size overflow")
	}
	for plane := range p.planes {
		p.planes[plane] = append([]unsafe.Pointer(nil), oldPlanes[plane]...)
		if p.capacity == pool.capBlocks {
			continue
		}
		for l, old := range oldPlanes[plane] {
			var b *cudaBuf
			b, err = p.allocateLocked(bytes, sites[plane])
			if err != nil {
				return err
			}
			p.planes[plane][l] = b.ptr
			if old != nil {
				p.old = append(p.old, cudaQwen35PagedRetirement{ptr: old, bytes: oldBytes})
				if err = p.copyLocked(b.ptr, old, oldBytes); err != nil {
					return err
				}
			}
		}
	}
	if sharedTail {
		for _, plane := range p.planes {
			for _, base := range plane {
				if err = p.copyLocked(unsafe.Add(base, int(tailNew)*pool.blockFloats*4), unsafe.Add(base, int(tailOld)*pool.blockFloats*4), pool.blockFloats*4); err != nil {
					return err
				}
			}
		}
	}
	p.tableBuf, err = p.allocateLocked(len(p.table)*4, "qwen35-paged-table")
	if err != nil {
		return err
	}
	if k.dTable != nil {
		p.old = append(p.old, cudaQwen35PagedRetirement{ptr: k.dTable, bytes: k.dTableCap * 4})
	}
	p.tablePin.Pin(&p.table[0])
	p.pinned = true
	p.drained = false
	if err = p.statusLocked("paged-table-h2d", int(C.fcuda_h2d_checked(p.tableBuf.ptr, unsafe.Pointer(&p.table[0]), C.size_t(len(p.table)*4)))); err != nil {
		return err
	}
	if err = p.c.qwen35SequenceSyncLocked(); err != nil {
		if p.failure == nil {
			p.failure = err
		}
		return err
	}
	if err = p.statusLocked("paged-reserve-fence", 0); err != nil {
		return err
	}
	p.drained = true
	p.tablePin.Unpin()
	p.pinned = false
	// All existing clones resolve arenas through this one shared pool. Publish
	// every arena and reference inventory together under cudaMu after completion.
	pool.Kraw, pool.K, pool.V = p.planes[0], p.planes[1], p.planes[2]
	pool.capBlocks, pool.blocks = p.capacity, p.blocks
	k.table, k.dTable, k.dTableCap, k.dirty = p.table, p.tableBuf.ptr, len(p.table), false
	k.seqLenDev = -1
	for i := range p.old {
		old := &p.old[i]
		old.attempted = true
		var disposition C.int
		status := int(C.fcuda_paged_kv_release_checked(old.ptr, C.size_t(old.bytes), &disposition))
		old.retired = disposition == C.FCUDA_PAGED_KV_RETIRED
		if disposition != C.FCUDA_PAGED_KV_RETIRED && status == 0 {
			status = -75823
		}
		if err = p.statusLocked("paged-old-owner-retirement", status); err != nil {
			return err
		}
	}
	gatherBytes, valid := qwen35GDNShapeBytes([]int{2, p.end, p.width}, 4)
	if !valid {
		return p.reject("gather scratch bytes overflow")
	}
	p.gatherBuf, err = p.allocateLocked(gatherBytes, "qwen35-paged-gather")
	if err != nil {
		if p.failure == nil {
			p.failure = err
		}
		return err
	}
	p.prepared = true
	normal = true
	return nil
}

func (p *cudaQwen35PagedSequence) appendLocked(layer int, kRaw, kRoPE, value Tensor) (err error) {
	if !p.prepared || p.retired || p.completed || layer < 0 || layer >= len(p.frontiers) || p.frontiers[layer] != p.start {
		return p.reject("paged append frontier mismatch")
	}
	normal := false
	defer func() {
		if !normal {
			p.abortLocked()
		}
	}()
	for _, item := range []struct {
		name string
		t    Tensor
	}{{"paged raw key", kRaw}, {"paged key", kRoPE}, {"paged value", value}} {
		if err = p.c.validateQwen35SequenceTensor(item.name, item.t, false, p.tokens, p.width); err != nil {
			return err
		}
	}
	p.executing = true
	p.drained = false
	for _, run := range pagedKVRuns(p.start, p.tokens, p.k.pool.blockTokens) {
		if err = p.statusLocked("paged-write-admission", 0); err != nil {
			return err
		}
		offset := run.src * p.width
		status := int(C.fcuda_paged_kv_write_f32((*C.float)(p.planes[1][layer]), (*C.float)(p.planes[2][layer]), (*C.float)(p.planes[0][layer]),
			(*C.float)(offsetF(kRoPE.buf.(*cudaBuf).ptr, offset)), (*C.float)(offsetF(value.buf.(*cudaBuf).ptr, offset)), (*C.float)(offsetF(kRaw.buf.(*cudaBuf).ptr, offset)),
			C.int(p.table[run.blk]), C.int(run.off), C.int(run.n), C.int(p.k.cfg.NumKVHeads), C.int(p.k.cfg.HeadDim), C.int(p.k.pool.blockTokens)))
		if err = p.statusLocked("paged-write", status); err != nil {
			return err
		}
	}
	p.frontiers[layer] = p.end
	normal = true
	return nil
}
func (p *cudaQwen35PagedSequence) attentionLocked(q Tensor, layer, heads int, scale float32) (out Tensor, b *cudaBuf, err error) {
	return p.attentionIntoLocked(q, layer, heads, scale, Tensor{})
}

// A supplied output is a borrowed row of the batch's live activation panel.
// An empty output preserves the single-sequence allocation and gather order.
func (p *cudaQwen35PagedSequence) attentionIntoLocked(q Tensor, layer, heads int, scale float32, out Tensor) (_ Tensor, b *cudaBuf, err error) {
	if !p.prepared || p.retired || p.completed || layer < 0 || layer >= len(p.frontiers) || p.frontiers[layer] != p.end {
		return out, nil, p.reject("paged attention frontier mismatch")
	}
	normal := false
	defer func() {
		if !normal {
			p.abortLocked()
		}
	}()
	if err = validateQwen35CausalAttentionPanelGeometry(p.tokens, p.start, heads, p.k.cfg.NumKVHeads, p.k.cfg.HeadDim); err != nil {
		return out, nil, err
	}
	if err = validateQwen35CausalAttentionPanelScale(scale); err != nil {
		return out, nil, err
	}
	if err = p.c.validateQwen35SequenceTensor("paged query", q, false, p.tokens, heads*p.k.cfg.HeadDim); err != nil {
		return out, nil, err
	}
	if out.Buf() != nil {
		if err = p.c.validateQwen35SequenceTensor("paged output", out, false, p.tokens, heads*p.k.cfg.HeadDim); err != nil {
			return out, nil, err
		}
	}
	kp := p.gatherBuf.ptr
	vp := offsetF(kp, p.end*p.width)
	p.executing = true
	p.drained = false
	for i, dst := range []unsafe.Pointer{kp, vp} {
		if err = p.statusLocked("paged-gather-admission", 0); err != nil {
			return out, nil, err
		}
		keyLayout := 0
		plane := 2
		if i == 0 {
			keyLayout = 1
			plane = 1
		}
		status := int(C.fcuda_paged_kv_snapshot_gather_checked((*C.float)(dst), (*C.float)(p.planes[plane][layer]), (*C.int)(p.tableBuf.ptr), C.int(p.end), C.int(p.k.cfg.NumKVHeads), C.int(p.k.cfg.HeadDim), C.int(p.k.pool.blockTokens), C.int(keyLayout)))
		if err = p.statusLocked("paged-gather", status); err != nil {
			return out, nil, err
		}
	}
	if out.Buf() == nil {
		out, b, err = p.c.qwen35SequenceAllocLocked([]int{p.tokens, heads * p.k.cfg.HeadDim}, "paged-causal-attention")
		if err != nil {
			return out, b, err
		}
	}
	if err = p.statusLocked("paged-attention-admission", 0); err != nil {
		return out, b, err
	}
	status := int(C.fak_qwen35_causal_attention_panel_f32(p.c.cf(q), (*C.float)(kp), (*C.float)(vp), p.c.cf(out), C.int(p.tokens), C.int(p.start), C.int(heads), C.int(p.k.cfg.NumKVHeads), C.int(p.k.cfg.HeadDim), C.float(scale)))
	if err = p.statusLocked("paged-attention", status); err != nil {
		return out, b, err
	}
	normal = true
	return out, b, nil
}
func (p *cudaQwen35PagedSequence) completeLocked() error {
	if err := p.retireLocked(); err != nil {
		return err
	}
	p.publishLocked()
	return nil
}

// Retire all fallible resources without making the logical sequence reusable.
// A batch retires every lane before publishing any lane.
func (p *cudaQwen35PagedSequence) retireLocked() error {
	if !p.prepared || p.retired || p.completed || !p.drained || p.executing {
		return p.reject("paged completion repeated, unprepared or not drained")
	}
	if err := p.statusLocked("paged-completion", 0); err != nil {
		return err
	}
	for _, frontier := range p.frontiers {
		if frontier != p.end {
			return p.reject(fmt.Sprintf("incomplete paged layer: %d, want %d", frontier, p.end))
		}
	}
	// Gather scratch has completed and is private to this operation. Retire it
	// before publishing logical frontiers, honoring disposition even on failure.
	p.old = append(p.old, cudaQwen35PagedRetirement{ptr: p.gatherBuf.ptr, bytes: p.gatherBuf.n, attempted: true})
	var disposition C.int
	status := int(C.fcuda_paged_kv_release_checked(p.gatherBuf.ptr, C.size_t(p.gatherBuf.n), &disposition))
	p.old[len(p.old)-1].retired = disposition == C.FCUDA_PAGED_KV_RETIRED
	if disposition != C.FCUDA_PAGED_KV_RETIRED && status == 0 {
		status = -75823
	}
	if err := p.statusLocked("paged-gather-retirement", status); err != nil {
		return err
	}
	p.retired = true
	return nil
}

// Called only after successful retirement of the complete group under cudaMu.
// Keep this phase allocation-free and incapable of submitting native work.
func (p *cudaQwen35PagedSequence) publishLocked() {
	p.k.lens, p.k.pos = p.frontiers, p.positions
	p.drained = true
	p.executing = false
	p.completed = true
	for link := &cudaQwen35PagedOwners; *link != nil; link = &(*link).next {
		if *link == p {
			*link = p.next
			break
		}
	}
	p.next = nil
	p.fresh = nil
	p.old = nil
}
