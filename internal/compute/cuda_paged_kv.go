//go:build cuda && cgo

package compute

/*
#include <stdlib.h>
#include "cuda_backend.h"
*/
import "C"

import (
	"fmt"
	"sync"
	"unsafe"
)

// cudaKVPagePool owns the device arenas of the opt-in paged KV store: per layer one K,
// one pre-RoPE K and one V arena of capBlocks pages in the vendored vLLM block layout.
// Every paged store of one backend and geometry draws from the same pool, so a cloned
// store maps its source's pages instead of copying them. Arenas only grow; freed pages
// return to the free list and are reused, never to the device allocator.
type cudaKVPagePool struct {
	be          *cudaBackend
	layers      int
	nKV, hd     int
	blockTokens int
	blockFloats int // floats per page per plane per layer
	capBlocks   int
	blocks      pagedKVBlocks
	K, Kraw, V  []unsafe.Pointer
}

type cudaPagedPoolKey struct {
	be              *cudaBackend
	layers, nKV, hd int
}

var (
	cudaPagedPoolsMu sync.Mutex
	cudaPagedPools   = map[cudaPagedPoolKey]*cudaKVPagePool{}
)

// pagedPool returns the backend's shared pool for cfg's geometry. It takes no cudaMu
// (NewKV runs under it during host restore) and allocates nothing until the first page.
func (c *cudaBackend) pagedPool(cfg KVConfig) *cudaKVPagePool {
	key := cudaPagedPoolKey{be: c, layers: cfg.NumLayers, nKV: cfg.NumKVHeads, hd: cfg.HeadDim}
	cudaPagedPoolsMu.Lock()
	defer cudaPagedPoolsMu.Unlock()
	if p := cudaPagedPools[key]; p != nil {
		return p
	}
	p := &cudaKVPagePool{be: c, layers: cfg.NumLayers, nKV: cfg.NumKVHeads, hd: cfg.HeadDim,
		blockTokens: CUDAPagedKVBlockTokens, blockFloats: CUDAPagedKVBlockTokens * cfg.NumKVHeads * cfg.HeadDim,
		K: make([]unsafe.Pointer, cfg.NumLayers), Kraw: make([]unsafe.Pointer, cfg.NumLayers), V: make([]unsafe.Pointer, cfg.NumLayers)}
	cudaPagedPools[key] = p
	return p
}

// CUDAPagedKVPageStats is the device-side accounting of one paged KV pool.
type CUDAPagedKVPageStats struct {
	LiveBlocks     int   `json:"live_blocks"`
	CapacityBlocks int   `json:"capacity_blocks"`
	BlockTokens    int   `json:"block_tokens"`
	LiveBytes      int64 `json:"live_bytes"`
	ArenaBytes     int64 `json:"arena_bytes"`
}

func (p *cudaKVPagePool) bytesPerBlock() int64 {
	return int64(p.layers) * 3 * int64(p.blockFloats) * 4
}

func (p *cudaKVPagePool) stats() CUDAPagedKVPageStats {
	return CUDAPagedKVPageStats{LiveBlocks: p.blocks.live, CapacityBlocks: p.capBlocks, BlockTokens: p.blockTokens,
		LiveBytes: int64(p.blocks.live) * p.bytesPerBlock(), ArenaBytes: int64(p.capBlocks) * p.bytesPerBlock()}
}

// growLocked raises every arena to hold need pages. Existing page ids keep their data.
// Caller holds cudaMu; a paged store is never graph-captured, so allocating is legal.
func (p *cudaKVPagePool) growLocked(need int) {
	n := pagedKVGrowBlocks(p.capBlocks, need)
	bytes := n * p.blockFloats * 4
	oldBytes := p.capBlocks * p.blockFloats * 4
	move := func(arena []unsafe.Pointer, site string) {
		for l := range arena {
			np := p.be.dallocKV(bytes, site+" layer "+itoaC(l)).ptr
			if arena[l] != nil {
				C.fcuda_d2d(np, arena[l], C.size_t(oldBytes))
				// A retired arena size is never requested again, so return it to the
				// driver (after the stream drains the copy) instead of the exact-size pool.
				if C.fcuda_discard_unsubmitted(arena[l], C.size_t(oldBytes)) != 0 {
					C.fcuda_free(arena[l])
				}
			}
			arena[l] = np
		}
	}
	move(p.K, "kv-paged-key-arena")
	move(p.Kraw, "kv-paged-pre-rope-key-arena")
	move(p.V, "kv-paged-value-arena")
	p.capBlocks = n
	p.blocks.grow(n)
}

func (p *cudaKVPagePool) allocLocked() int32 {
	id, ok := p.blocks.alloc()
	if !ok {
		p.growLocked(p.capBlocks + 1)
		id, _ = p.blocks.alloc()
	}
	return id
}

// copyBlockLocked duplicates page src into dst in every arena (copy-on-write).
func (p *cudaKVPagePool) copyBlockLocked(dst, src int32) {
	bytes := C.size_t(p.blockFloats * 4)
	for _, arena := range [][]unsafe.Pointer{p.K, p.Kraw, p.V} {
		for l := range arena {
			C.fcuda_d2d(unsafe.Add(arena[l], int(dst)*p.blockFloats*4), unsafe.Add(arena[l], int(src)*p.blockFloats*4), bytes)
		}
	}
}

// cudaPagedKV is the page-granular CUDA KV store (FAK_CUDA_PAGED_KV=1). Attention reads
// through its block table with the vendored vLLM paged-attention kernel; Clone retains
// the source's pages instead of copying device bytes, and a writer copies a shared tail
// page before its first write (copy-on-write).
type cudaPagedKV struct {
	pool  *cudaKVPagePool
	cfg   KVConfig
	table []int32
	lens  []int // positions written per layer
	pos   []int

	dTable    unsafe.Pointer // device copy of table
	dTableCap int
	dirty     bool
	dSeqLen   unsafe.Pointer // device int read by the kernel as seq_lens[0]
	seqLenDev int
	scratch   unsafe.Pointer // v2 partition scratch
	scratchF  int64
	gK, gV    unsafe.Pointer // KeysView / ValuesView gather buffers
	gKF, gVF  int
}

func (c *cudaBackend) newPagedKV(cfg KVConfig) *cudaPagedKV {
	return &cudaPagedKV{pool: c.pagedPool(cfg), cfg: cloneKVConfig(cfg), lens: make([]int, cfg.NumLayers), seqLenDev: -1}
}

func (k *cudaPagedKV) KVConfig() KVConfig      { return cloneKVConfig(k.cfg) }
func (k *cudaPagedKV) GraphCaptureReady() bool { return false }
func (k *cudaPagedKV) Len() int                { return len(k.pos) }
func (k *cudaPagedKV) Pos() []int              { return append([]int(nil), k.pos...) }
func (k *cudaPagedKV) stride() int             { return k.cfg.NumKVHeads * k.cfg.HeadDim }
func (k *cudaPagedKV) BlockTable() []int32     { return append([]int32(nil), k.table...) }
func (k *cudaPagedKV) PageStats() CUDAPagedKVPageStats {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	return k.pool.stats()
}

// ResidentBytes counts only pages this store holds alone; a page shared with a clone is
// charged to neither, so summing stores never double-counts a shared prefix.
func (k *cudaPagedKV) ResidentBytes() int64 {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	var own int64
	for _, id := range k.table {
		if !k.pool.blocks.shared(id) {
			own++
		}
	}
	return own*k.pool.bytesPerBlock() + int64(len(k.pos))*8
}

// SharedBytes is the device payload this store maps but shares with another store.
func (k *cudaPagedKV) SharedBytes() int64 {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	var n int64
	for _, id := range k.table {
		if k.pool.blocks.shared(id) {
			n++
		}
	}
	return n * k.pool.bytesPerBlock()
}

// writablePageLocked returns logical page blk ready for a write: allocated if new, or a
// private copy if it is still shared with a clone.
func (k *cudaPagedKV) writablePageLocked(blk int) int32 {
	switch {
	case blk == len(k.table):
		k.table = append(k.table, k.pool.allocLocked())
		k.dirty = true
	case blk > len(k.table):
		panic("cuda: paged KV append skipped a page")
	case k.pool.blocks.shared(k.table[blk]):
		fresh := k.pool.allocLocked()
		k.pool.copyBlockLocked(fresh, k.table[blk])
		k.pool.blocks.release(k.table[blk])
		k.table[blk] = fresh
		k.dirty = true
	}
	return k.table[blk]
}

// appendRowsLocked writes n rows [n, nKV*hd] of each plane at the layer's next positions.
func (k *cudaPagedKV) appendRowsLocked(layer int, kRaw, kRoPE, v unsafe.Pointer, n int) {
	if layer < 0 || layer >= len(k.lens) || n <= 0 {
		panic("cuda: paged KV append layer or row count out of range")
	}
	w := k.stride()
	for _, run := range pagedKVRuns(k.lens[layer], n, k.pool.blockTokens) {
		id := k.writablePageLocked(run.blk)
		off := run.src * w * 4
		if rc := C.fcuda_paged_kv_write_f32((*C.float)(k.pool.K[layer]), (*C.float)(k.pool.V[layer]), (*C.float)(k.pool.Kraw[layer]),
			(*C.float)(unsafe.Add(kRoPE, off)), (*C.float)(unsafe.Add(v, off)), (*C.float)(unsafe.Add(kRaw, off)),
			C.int(id), C.int(run.off), C.int(run.n), C.int(k.cfg.NumKVHeads), C.int(k.cfg.HeadDim), C.int(k.pool.blockTokens)); rc != 0 {
			panic(fmt.Sprintf("cuda: paged KV write failed (%d)", int(rc)))
		}
	}
	k.lens[layer] += n
}

func (k *cudaPagedKV) AppendKV(layer int, kRaw, kRoPE, v Tensor, pos int) {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	if kRaw.Dtype != F32 || kRoPE.Dtype != F32 || v.Dtype != F32 {
		panic("cuda: KV append requires F32 source rows")
	}
	be := k.pool.be
	a, b, c := be.cudaBufForSubmit(kRaw), be.cudaBufForSubmit(kRoPE), be.cudaBufForSubmit(v)
	k.appendRowsLocked(layer, a.ptr, b.ptr, c.ptr, 1)
	if layer == 0 {
		k.pos = append(k.pos, pos)
	}
}

// syncDeviceTableLocked uploads the block table when it changed and sets the device
// sequence length the kernel reads. g_stream is a blocking stream, so the synchronous
// table upload is ordered after every kernel already reading the old table.
func (k *cudaPagedKV) syncDeviceTableLocked(seqLen int) {
	if k.dirty || k.dTable == nil {
		if len(k.table) > k.dTableCap || k.dTable == nil {
			if k.dTable != nil {
				C.fcuda_free(k.dTable)
			}
			n := pagedKVGrowBlocks(k.dTableCap, len(k.table))
			k.dTable = k.pool.be.dallocKV(n*4, "kv-paged-block-table").ptr
			k.dTableCap = n
		}
		if len(k.table) > 0 {
			C.fcuda_h2d(k.dTable, unsafe.Pointer(&k.table[0]), C.size_t(len(k.table)*4))
		}
		k.dirty = false
	}
	if k.dSeqLen == nil {
		k.dSeqLen = k.pool.be.dallocKV(4, "kv-paged-seq-len").ptr
		k.seqLenDev = -1
	}
	if k.seqLenDev != seqLen {
		if rc := C.fcuda_paged_store_int((*C.int)(k.dSeqLen), C.int(seqLen)); rc != 0 {
			panic(fmt.Sprintf("cuda: paged KV seq-len store failed (%d)", int(rc)))
		}
		k.seqLenDev = seqLen
	}
}

// attentionLocked is decode attention of q [nH*hd] over the layer's pages.
func (c *cudaBackend) pagedAttentionLocked(q Tensor, k *cudaPagedKV, layer, grp int, scale float32) Tensor {
	hd, nKV := k.cfg.HeadDim, k.cfg.NumKVHeads
	nH := grp * nKV
	seqLen := k.lens[layer]
	if seqLen <= 0 {
		panic("cuda: paged attention over an empty layer")
	}
	k.syncDeviceTableLocked(seqLen)
	need := int64(C.fcuda_paged_attention_scratch_floats(1, C.int(nH), C.int(hd), C.int(seqLen)))
	if need > k.scratchF {
		if k.scratch != nil {
			C.fcuda_free(k.scratch)
		}
		n := need + need/2
		k.scratch = c.dallocKV(int(n)*4, "kv-paged-attention-partitions").ptr
		k.scratchF = n
	}
	out, _ := c.devTr([]int{nH * hd}, F32)
	if rc := C.fcuda_paged_attention_f32(c.cf(out), c.cf(q), (*C.float)(k.pool.K[layer]), (*C.float)(k.pool.V[layer]),
		(*C.int)(k.dTable), (*C.int)(k.dSeqLen), 1, C.int(nH), C.int(nKV), C.int(hd),
		C.int(len(k.table)), C.int(seqLen), C.float(scale), (*C.float)(k.scratch), C.longlong(k.scratchF)); rc != 0 {
		panic(fmt.Sprintf("cuda: paged attention launch failed (%d)", int(rc)))
	}
	return out
}

// gatherLocked unpacks the layer's first n rows of one plane into dst (row-major).
func (k *cudaPagedKV) gatherLocked(dst unsafe.Pointer, arena unsafe.Pointer, layer, n int, keyLayout bool) {
	if n <= 0 {
		return
	}
	k.syncDeviceTableLocked(k.lens[layer])
	kl := 0
	if keyLayout {
		kl = 1
	}
	if rc := C.fcuda_paged_kv_gather_f32((*C.float)(dst), (*C.float)(arena), (*C.int)(k.dTable), C.int(n),
		C.int(k.cfg.NumKVHeads), C.int(k.cfg.HeadDim), C.int(k.pool.blockTokens), C.int(kl)); rc != 0 {
		panic(fmt.Sprintf("cuda: paged KV gather failed (%d)", int(rc)))
	}
}

func (k *cudaPagedKV) viewLocked(buf *unsafe.Pointer, bufF *int, arena unsafe.Pointer, layer int, keyLayout bool, site string) Tensor {
	w := k.stride()
	n := k.lens[layer]
	if n*w > *bufF {
		if *buf != nil {
			C.fcuda_free(*buf)
		}
		f := pagedKVGrowBlocks(*bufF/w, n) * w
		*buf = k.pool.be.dallocKV(f*4, site).ptr
		*bufF = f
	}
	k.gatherLocked(*buf, arena, layer, n, keyLayout)
	return makeTensor(k.pool.be, F32, RowMajor, []int{n, w}, nil, &cudaBuf{ptr: *buf, n: n * w * 4, class: MemoryKVCache})
}

// KeysView gathers the layer's post-RoPE keys into a row-major [pos, nKV*hd] device
// buffer the store owns. The view stays valid until the next KeysView on this store.
func (k *cudaPagedKV) KeysView(layer int) Tensor {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	return k.viewLocked(&k.gK, &k.gKF, k.pool.K[layer], layer, true, "kv-paged-keys-view")
}

// ValuesView is KeysView's counterpart for values.
func (k *cudaPagedKV) ValuesView(layer int) Tensor {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	return k.viewLocked(&k.gV, &k.gVF, k.pool.V[layer], layer, false, "kv-paged-values-view")
}

// Clone maps the source's pages into a new store: every page gains a reference and no
// device KV byte is copied. The first write either store makes into a shared page copies
// that one page first.
func (k *cudaPagedKV) Clone() KVStore {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	n := &cudaPagedKV{pool: k.pool, cfg: cloneKVConfig(k.cfg), table: append([]int32(nil), k.table...),
		lens: append([]int(nil), k.lens...), pos: append([]int(nil), k.pos...), dirty: true, seqLenDev: -1}
	for _, id := range n.table {
		k.pool.blocks.retain(id)
	}
	return n
}

// releaseLocked drops this store's page references and its private device buffers.
func (k *cudaPagedKV) releaseLocked() {
	for _, id := range k.table {
		k.pool.blocks.release(id)
	}
	k.table = nil
	for _, p := range []*unsafe.Pointer{&k.dTable, &k.dSeqLen, &k.scratch, &k.gK, &k.gV} {
		if *p != nil {
			C.fcuda_free(*p)
			*p = nil
		}
	}
	k.dTableCap, k.scratchF, k.gKF, k.gVF, k.seqLenDev = 0, 0, 0, 0, -1
	for l := range k.lens {
		k.lens[l] = 0
	}
	k.pos = nil
}

func (k *cudaPagedKV) Free() {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	k.releaseLocked()
}

// SnapshotToHost gathers every plane of every layer into host rows.
func (k *cudaPagedKV) SnapshotToHost() (KVHostSnapshot, error) {
	if k == nil {
		return KVHostSnapshot{}, fmt.Errorf("cuda: cannot snapshot nil KV store")
	}
	cudaMu.Lock()
	defer cudaMu.Unlock()
	w := k.stride()
	out := KVHostSnapshot{Config: cloneKVConfig(k.cfg), Pos: append([]int(nil), k.pos...),
		K: make([][]float32, k.cfg.NumLayers), KRaw: make([][]float32, k.cfg.NumLayers), V: make([][]float32, k.cfg.NumLayers)}
	maxN := 0
	for _, n := range k.lens {
		maxN = max(maxN, n)
	}
	var tmp unsafe.Pointer
	if maxN > 0 {
		tmp = k.pool.be.dallocKV(maxN*w*4, "kv-paged-snapshot").ptr
		defer C.fcuda_free(tmp)
	}
	read := func(arena unsafe.Pointer, layer int, keyLayout bool) []float32 {
		n := k.lens[layer]
		rows := make([]float32, n*w)
		if n > 0 {
			k.gatherLocked(tmp, arena, layer, n, keyLayout)
			C.fcuda_d2h(unsafe.Pointer(&rows[0]), tmp, C.size_t(len(rows)*4))
		}
		return rows
	}
	for l := 0; l < k.cfg.NumLayers; l++ {
		out.K[l] = read(k.pool.K[l], l, true)
		out.KRaw[l] = read(k.pool.Kraw[l], l, true)
		out.V[l] = read(k.pool.V[l], l, false)
	}
	return out, out.Validate()
}

// loadFromHostLocked replaces the store's contents with a host image. Caller holds cudaMu.
func (k *cudaPagedKV) loadFromHostLocked(state KVHostSnapshot) {
	k.releaseLocked()
	w := k.stride()
	maxN := 0
	for l := range state.K {
		maxN = max(maxN, len(state.K[l])/w)
	}
	if maxN > 0 {
		tmp := k.pool.be.dallocKV(3*maxN*w*4, "kv-paged-host-restore").ptr
		defer C.fcuda_free(tmp)
		for l := range state.K {
			n := len(state.K[l]) / w
			if n == 0 {
				continue
			}
			kp, rp, vp := tmp, unsafe.Add(tmp, maxN*w*4), unsafe.Add(tmp, 2*maxN*w*4)
			C.fcuda_h2d(kp, unsafe.Pointer(&state.K[l][0]), C.size_t(n*w*4))
			C.fcuda_h2d(rp, unsafe.Pointer(&state.KRaw[l][0]), C.size_t(n*w*4))
			C.fcuda_h2d(vp, unsafe.Pointer(&state.V[l][0]), C.size_t(n*w*4))
			k.appendRowsLocked(l, rp, kp, vp, n)
		}
	}
	k.pos = append([]int(nil), state.Pos...)
}

func (c *cudaBackend) restorePagedKVFromHost(state KVHostSnapshot) (out KVStore, err error) {
	if err := state.Validate(); err != nil {
		return nil, err
	}
	k := c.newPagedKV(cloneKVConfig(state.Config))
	cudaMu.Lock()
	defer cudaMu.Unlock()
	defer func() {
		if r := recover(); r != nil {
			k.releaseLocked()
			out, err = nil, fmt.Errorf("cuda: restore paged KV from host: %v", r)
		}
	}()
	k.loadFromHostLocked(state)
	return k, nil
}

// Evict removes [from, from+n) with the contiguous store's exact semantics (prefix pages
// untouched in value, survivors re-rotated from pre-RoPE K at their new index). It is not
// yet device-native: it round-trips through host DRAM and a temporary contiguous store.
func (k *cudaPagedKV) Evict(from, n int) int {
	if from < 0 || n <= 0 || from >= k.Len() {
		return 0
	}
	snap, err := k.SnapshotToHost()
	if err != nil {
		panic(fmt.Sprintf("cuda: paged KV evict snapshot: %v", err))
	}
	tmp, err := k.pool.be.restoreContiguousKVFromHost(snap)
	if err != nil {
		panic(fmt.Sprintf("cuda: paged KV evict staging: %v", err))
	}
	removed := tmp.Evict(from, n)
	after, err := tmp.(KVHostSnapshotter).SnapshotToHost()
	tmp.Free()
	if err != nil {
		panic(fmt.Sprintf("cuda: paged KV evict readback: %v", err))
	}
	cudaMu.Lock()
	defer cudaMu.Unlock()
	k.loadFromHostLocked(after)
	return removed
}

// CUDAPagedKVPageStatsFor reports the shared page pool behind kv when it is a CUDA paged
// store.
func CUDAPagedKVPageStatsFor(kv KVStore) (CUDAPagedKVPageStats, bool) {
	pk, ok := kv.(*cudaPagedKV)
	if !ok || pk == nil {
		return CUDAPagedKVPageStats{}, false
	}
	return pk.PageStats(), true
}

// pagedPanelAppendLocked is the prompt panel's multi-row append into pages.
func (c *cudaBackend) pagedPanelAppendLocked(k *cudaPagedKV, layer int, kRaw, kRoPE, v Tensor, start, tokens int) {
	w := k.stride()
	n := tokens * w
	if kRaw.Dtype != F32 || kRoPE.Dtype != F32 || v.Dtype != F32 {
		cudaPromptPanelFail("prompt-panel-kv-dtype", "KV append requires F32 source rows", 0)
	}
	if kRaw.Numel() != n || kRoPE.Numel() != n || v.Numel() != n {
		cudaPromptPanelFail("prompt-panel-kv-geometry", "prompt panel KV rows do not match tokens x kv width", 0)
	}
	if layer < 0 || layer >= len(k.lens) || k.lens[layer] != start {
		cudaPromptPanelFail("prompt-panel-kv-position", "prompt panel KV append does not start at the cached length", 0)
	}
	a, b, d := c.cudaBufForSubmit(kRaw), c.cudaBufForSubmit(kRoPE), c.cudaBufForSubmit(v)
	k.appendRowsLocked(layer, a.ptr, b.ptr, d.ptr, tokens)
	if layer == 0 {
		for t := 0; t < tokens; t++ {
			k.pos = append(k.pos, start+t)
		}
	}
}

// pagedPanelAttentionLocked gathers the layer's rows into contiguous scratch and runs the
// existing causal panel kernel over them (prefill is not the paged kernel's job).
func (c *cudaBackend) pagedPanelAttentionLocked(q Tensor, k *cudaPagedKV, layer, tokens, start, grp int, scale float32) Tensor {
	hd, nKV := k.cfg.HeadDim, k.cfg.NumKVHeads
	nH := grp * nKV
	w := nKV * hd
	if q.Numel() != tokens*nH*hd || layer < 0 || layer >= len(k.lens) || k.lens[layer] != start+tokens {
		cudaPromptPanelFail("prompt-panel-attention-geometry", "prompt panel attention query or cache length mismatch", 0)
	}
	n := start + tokens
	tmp := c.dallocKV(2*n*w*4, "kv-paged-panel-gather").ptr
	defer C.fcuda_free(tmp)
	kp, vp := tmp, unsafe.Add(tmp, n*w*4)
	k.gatherLocked(kp, k.pool.K[layer], layer, n, true)
	k.gatherLocked(vp, k.pool.V[layer], layer, n, false)
	out, _ := c.devTr([]int{tokens, nH * hd}, F32)
	if rc := int(C.fcuda_causal_attention_panel_f32(c.cf(q), (*C.float)(kp), (*C.float)(vp), c.cf(out),
		C.int(tokens), C.int(start), C.int(nH), C.int(nKV), C.int(hd), C.float(scale))); rc != 0 {
		cudaPromptPanelFail("prompt-panel-attention-launch", "paged prompt panel attention launch failed", rc)
	}
	return out
}
