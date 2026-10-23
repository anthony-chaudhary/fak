//go:build cuda && cgo

package compute

/*
#include "cuda_backend.h"
*/
import "C"

import "unsafe"

// cuda_prompt_panel.go is the CUDA half of the dense HAL prompt panel (model.promptPanelBackend).
// Every panel op is the per-row form of the op a decode step issues, so a row's bits never
// depend on the panel's row count or on how the prompt was chunked: the Q8_0 GEMM quantizes and
// dots each row exactly as the P=1 GEMV does (fcuda_q8_matmul_rows_f32, never the P>=128
// dequant + cuBLAS path), RoPE keeps k_rope's arithmetic, and attention keeps
// k_flash_attention's recurrence. Panel scratch lives in its own free list, released at
// PromptPanelEnd, so panel-sized buffers never reach the decode pool.

// cudaScratchAllocFailed and cudaSplitScratchAllocFailed are FCUDA_SCRATCH_ALLOC_FAILED and
// FCUDA_SPLIT_SCRATCH_ALLOC_FAILED in cuda_kernels.cu.
const (
	cudaScratchAllocFailed      = 80001
	cudaSplitScratchAllocFailed = 80002
)

func cudaPromptPanelFail(site, msg string, code int) {
	panic(&CUDAOpError{Op: "PromptPanel", Site: site, Msg: msg, Code: code, Class: MemoryActivation})
}

// cudaGEMMScratch names the scratch one GEMM call can miss: main is its f32 dequant copy (or
// per-row activation codes and scales), split one fp16 half of the compensated weight.
type cudaGEMMScratch struct{ main, split int }

// cudaGEMMError maps a non-zero fcuda GEMM status to a typed error: a scratch allocation miss
// is a DeviceAllocError for the allocation that missed, any other status a CUDAOpError. The
// GEMM launched nothing, so its output holds no result.
func cudaGEMMError(op string, rc int, s cudaGEMMScratch) error {
	switch rc {
	case cudaScratchAllocFailed:
		return &DeviceAllocError{Bytes: s.main, Site: op + " gemm scratch", Class: MemoryScratchpad}
	case cudaSplitScratchAllocFailed:
		return &DeviceAllocError{Bytes: s.split, Site: op + " gemm fp16 split scratch", Class: MemoryScratchpad}
	}
	return &CUDAOpError{Op: op, Site: "gemm-status", Msg: "gemm refused its arguments", Code: rc, Class: MemoryActivation}
}

func cudaGEMMStatus(op string, rc int, s cudaGEMMScratch) {
	if rc != 0 {
		panic(cudaGEMMError(op, rc, s))
	}
}

// cudaDequantScratch is the f32 weight copy and one fp16 weight half the P>=128 dequant + fp16
// GEMM allocates.
func cudaDequantScratch(out, in int) cudaGEMMScratch {
	return cudaGEMMScratch{main: out * in * F32.Bytes(), split: out * in * F16.Bytes()}
}

// cudaQ8RowsScratch is the int8 activation codes plus f32 block scales of the per-row Q8 GEMM.
func cudaQ8RowsScratch(rows, in, block int) cudaGEMMScratch {
	return cudaGEMMScratch{main: rows*in + rows*(in/block)*F32.Bytes()}
}

// cudaQ8MatMulScratch is the scratch fcuda_q8_matmul_f32 needs for rows rows.
func cudaQ8MatMulScratch(out, in, rows, block int) cudaGEMMScratch {
	if rows >= 128 {
		return cudaDequantScratch(out, in)
	}
	return cudaQ8RowsScratch(rows, in, block)
}

// cudaKQuantScratch is the scratch the Q4_K / Q2_K GEMM needs for rows rows.
func cudaKQuantScratch(out, in, rows int) cudaGEMMScratch {
	if rows >= 128 {
		return cudaDequantScratch(out, in)
	}
	return cudaGEMMScratch{}
}

// dallocPromptPanel serves panel scratch from the panel free list. A miss is a typed
// DeviceAllocError: panel scratch never falls back to managed memory. Caller holds cudaMu.
func (c *cudaBackend) dallocPromptPanel(nbytes int, class MemoryClass, site string) *cudaBuf {
	p := C.fcuda_malloc_panel(C.size_t(nbytes))
	if p == nil {
		panic(&DeviceAllocError{Bytes: nbytes, Site: "prompt-panel " + site, Class: class})
	}
	return &cudaBuf{ptr: unsafe.Pointer(p), n: nbytes, class: class}
}

// PromptPanelRowExact reports the weight dtypes whose panel GEMM is bitwise the decode GEMV
// per row. Only Q8_0 has that per-row kernel today; every other dtype keeps the token loop.
func (c *cudaBackend) PromptPanelRowExact(dt Dtype) bool { return dt == Q8_0 }

// PromptPanelBegin routes scratch and activation allocations to the panel free list.
func (c *cudaBackend) PromptPanelBegin() {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	c.promptPanel = true
	c.promptPanelRows = 0
	c.promptPanelBegins++
	C.fcuda_panel_peak_reset()
}

// PromptPanelChunk starts a chunk of rows rows. It drops freed transient handles, and when the
// row count differs from the previous chunk's it first frees the cached buffers of that size,
// so a tail chunk never holds full-chunk buffers beside its own.
func (c *cudaBackend) PromptPanelChunk(rows int) {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	c.compactTransientLocked()
	if c.promptPanelRows != 0 && rows != c.promptPanelRows {
		C.fcuda_panel_release()
	}
	c.promptPanelRows = rows
}

// PromptPanelEnd closes the scope and frees every pooled panel buffer after draining the stream.
func (c *cudaBackend) PromptPanelEnd() {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	c.promptPanel = false
	c.promptPanelRows = 0
	c.promptPanelLastLive = int64(C.fcuda_panel_release())
	c.compactTransientLocked()
}

// CUDAPromptPanelCounters are the panel engagement witness: scopes opened, per-row panel GEMMs
// issued, and panel buffers still held when the last scope closed.
type CUDAPromptPanelCounters struct {
	Begins, MatMuls uint64
	LastReleaseLive int64
}

// CUDADebugPromptPanelCounters reads CUDAPromptPanelCounters under the allocator lock.
func (c *cudaBackend) CUDADebugPromptPanelCounters() CUDAPromptPanelCounters {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	return CUDAPromptPanelCounters{Begins: c.promptPanelBegins, MatMuls: c.promptPanelMatMuls, LastReleaseLive: c.promptPanelLastLive}
}

// CUDADebugSetGraphEnabled flips the process-wide CUDA graph gate (FAK_CUDA_GRAPH) and returns
// its previous value, so a device test can exercise graph capture whatever the env says. Set it
// only between sessions: it also decides whether NewKV preallocates a fixed graph capacity.
func (c *cudaBackend) CUDADebugSetGraphEnabled(on bool) bool {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	old := graphEnabled
	graphEnabled = on
	return old
}

// cudaArmScratchAllocFaultForTest makes the nth next dequant + fp16 GEMM scratch allocation
// miss (1 = the f32 dequant copy, 2 = the first fp16 split half); 0 disarms.
func cudaArmScratchAllocFaultForTest(nth int) {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	C.fcuda_scratch_alloc_test_fault(C.int(nth))
}

// PromptPanelMatMul is the BitExact panel GEMM: X [tokens, in] -> Y [tokens, out] for a Q8_0
// weight, every row through the per-row path whatever tokens is.
func (c *cudaBackend) PromptPanelMatMul(w, x Tensor, tokens int) Tensor {
	if w.Dtype != Q8_0 || len(w.Shape) != 2 || tokens <= 0 || x.Numel() != tokens*w.Shape[1] {
		cudaPromptPanelFail("prompt-panel-matmul-geometry", "panel GEMM needs a Q8_0 [out,in] weight and a [tokens,in] panel", 0)
	}
	cudaMu.Lock()
	defer cudaMu.Unlock()
	out, in := w.Shape[0], w.Shape[1]
	wb := c.cudaBufForSubmit(w)
	y, _ := c.devTr([]int{tokens, out}, F32)
	cudaGEMMStatus("PromptPanelMatMul", int(C.fcuda_q8_matmul_rows_f32((*C.int8_t)(wb.ptr), (*C.float)(wb.scales),
		c.cf(x), c.cf(y), C.int(out), C.int(in), C.int(tokens), C.int(w.Quant.Block))), cudaQ8RowsScratch(tokens, in, w.Quant.Block))
	c.promptPanelMatMuls++
	return y
}

// PromptPanelRoPE rotates row t of x [tokens, nHeads*headDim] at position start+t.
func (c *cudaBackend) PromptPanelRoPE(x Tensor, tokens, start, nHeads, headDim int, theta float64) Tensor {
	if tokens <= 0 || start < 0 || nHeads <= 0 || headDim <= 0 || headDim%2 != 0 || x.Numel() != tokens*nHeads*headDim {
		cudaPromptPanelFail("prompt-panel-rope-geometry", "rope panel shape does not match tokens x heads x head_dim", 0)
	}
	cudaMu.Lock()
	defer cudaMu.Unlock()
	y, _ := c.devTr([]int{tokens, nHeads * headDim}, F32)
	if rc := int(C.fcuda_rope_panel_f32(c.cf(x), c.cf(y), C.int(tokens), C.int(start), C.int(nHeads), C.int(headDim), C.double(theta))); rc != 0 {
		cudaPromptPanelFail("prompt-panel-rope-launch", "rope panel launch failed", rc)
	}
	return y
}

// PromptPanelAppendKV appends tokens contiguous rows to layer's raw-key, key and value caches.
func (c *cudaBackend) PromptPanelAppendKV(kv KVStore, layer int, kRaw, kRoPE, v Tensor, start, tokens int) {
	ck, ok := kv.(*cudaKV)
	if !ok || ck == nil {
		cudaPromptPanelFail("prompt-panel-kv-store", "prompt panel KV append needs the CUDA KV store", 0)
	}
	if layer < 0 || layer >= len(ck.K) || tokens <= 0 {
		cudaPromptPanelFail("prompt-panel-kv-geometry", "prompt panel KV append layer or token count out of range", 0)
	}
	cudaMu.Lock()
	defer cudaMu.Unlock()
	w := ck.stride()
	n := tokens * w
	if kRaw.Numel() != n || kRoPE.Numel() != n || v.Numel() != n {
		cudaPromptPanelFail("prompt-panel-kv-geometry", "prompt panel KV rows do not match tokens x kv width", 0)
	}
	if ck.K[layer].len != start*w {
		cudaPromptPanelFail("prompt-panel-kv-position", "prompt panel KV append does not start at the cached length", 0)
	}
	kRawBuf := c.cudaBufForSubmit(kRaw)
	kRoPEBuf := c.cudaBufForSubmit(kRoPE)
	vBuf := c.cudaBufForSubmit(v)
	c.growAppend(&ck.Kraw[layer], kRawBuf.ptr, n, "kv-pre-rope-key-grow layer "+itoaC(layer))
	c.growAppend(&ck.K[layer], kRoPEBuf.ptr, n, "kv-key-grow layer "+itoaC(layer))
	c.growAppend(&ck.V[layer], vBuf.ptr, n, "kv-value-grow layer "+itoaC(layer))
	if layer == 0 {
		for t := 0; t < tokens; t++ {
			ck.pos = append(ck.pos, start+t)
		}
	}
}

// PromptPanelAttention attends query row t of q [tokens, nH*hd] to cache positions [0, start+t].
func (c *cudaBackend) PromptPanelAttention(q Tensor, kv KVStore, layer, tokens, start, grp int, scale float32) Tensor {
	ck, ok := kv.(*cudaKV)
	if !ok || ck == nil {
		cudaPromptPanelFail("prompt-panel-attention-store", "prompt panel attention needs the CUDA KV store", 0)
	}
	if layer < 0 || layer >= len(ck.K) || tokens <= 0 || start < 0 || grp <= 0 {
		cudaPromptPanelFail("prompt-panel-attention-geometry", "prompt panel attention layer, tokens or group out of range", 0)
	}
	cudaMu.Lock()
	defer cudaMu.Unlock()
	hd, nKV := ck.cfg.HeadDim, ck.cfg.NumKVHeads
	nH := grp * nKV
	if q.Numel() != tokens*nH*hd || ck.K[layer].len != (start+tokens)*nKV*hd || ck.V[layer].len != ck.K[layer].len {
		cudaPromptPanelFail("prompt-panel-attention-geometry", "prompt panel attention query or cache length mismatch", 0)
	}
	out, _ := c.devTr([]int{tokens, nH * hd}, F32)
	if rc := int(C.fcuda_causal_attention_panel_f32(c.cf(q), (*C.float)(ck.K[layer].ptr), (*C.float)(ck.V[layer].ptr),
		c.cf(out), C.int(tokens), C.int(start), C.int(nH), C.int(nKV), C.int(hd), C.float(scale))); rc != 0 {
		cudaPromptPanelFail("prompt-panel-attention-launch", "prompt panel attention launch failed", rc)
	}
	return out
}

// PromptPanelRow copies row of a [rows, width] panel into a new [width] tensor. The copy is a
// decode-pool transient even inside the panel scope: it outlives PromptPanelEnd as the
// last-token hidden the logits head reads.
func (c *cudaBackend) PromptPanelRow(x Tensor, row, width int) Tensor {
	if width <= 0 || row < 0 || (row+1)*width > x.Numel() {
		cudaPromptPanelFail("prompt-panel-row-geometry", "prompt panel row outside the panel", 0)
	}
	cudaMu.Lock()
	defer cudaMu.Unlock()
	src := c.cudaBufForSubmit(x)
	inPanel := c.promptPanel
	c.promptPanel = false
	y, ybuf := c.devTr([]int{width}, F32)
	c.promptPanel = inPanel
	C.fcuda_d2d(ybuf.ptr, unsafe.Add(src.ptr, row*width*F32.Bytes()), C.size_t(width*F32.Bytes()))
	return y
}

// CUDAPoolStats is the allocator witness for panel tests: cached (free) decode-pool bytes and
// non-empty buckets, the largest such bucket, cached panel bytes, panel buffers still held, and
// the last panel scope's peak of panel bytes held plus cached.
type CUDAPoolStats struct {
	PooledBytes, Buckets, LargestBucket, PanelCachedBytes, PanelLive, PanelPeakBytes int64
}

// CUDADebugPoolStats reads CUDAPoolStats under the allocator lock.
func (c *cudaBackend) CUDADebugPoolStats() CUDAPoolStats {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	var pooled, buckets, largest, panel, live, peak C.size_t
	C.fcuda_pool_stats(&pooled, &buckets, &largest, &panel, &live, &peak)
	return CUDAPoolStats{PooledBytes: int64(pooled), Buckets: int64(buckets), LargestBucket: int64(largest),
		PanelCachedBytes: int64(panel), PanelLive: int64(live), PanelPeakBytes: int64(peak)}
}
