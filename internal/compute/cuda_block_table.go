package compute

import (
	"sync"
	"sync/atomic"
)

// CUDABlockID is an identifier for a physical memory block in device memory.
type CUDABlockID int64

// CUDAPageBlock represents an allocated physical page block backed by device or host memory (#10723).
type CUDAPageBlock struct {
	ID       CUDABlockID
	Tokens   int
	Stride   int
	Bytes    int64
	refCount int64
}

// RefCount returns the current reference count of the physical page block.
func (b *CUDAPageBlock) RefCount() int64 {
	if b == nil {
		return 0
	}
	return atomic.LoadInt64(&b.refCount)
}

// IncRef increments the reference count.
func (b *CUDAPageBlock) IncRef() int64 {
	if b == nil {
		return 0
	}
	return atomic.AddInt64(&b.refCount, 1)
}

// DecRef decrements the reference count.
func (b *CUDAPageBlock) DecRef() int64 {
	if b == nil {
		return 0
	}
	return atomic.AddInt64(&b.refCount, -1)
}

// CUDAPagedPrefillPath names the dtype path a CUDA block table pages its KV memory on
// during prefill (oss-port-flashinfer-fp8-prefill-6393, PR #3321).
type CUDAPagedPrefillPath string

const (
	// CUDAPagedPrefillGeneric is the default paged path: one F32 element per slot
	// (4 bytes). Every arch not admitted to the FP8 branch pages on it, including
	// unknown compute capabilities — the generic path is the always-available fallback.
	CUDAPagedPrefillGeneric CUDAPagedPrefillPath = "generic"
	// CUDAPagedPrefillFP8 is the arch-gated paged path: one FP8 (E4M3) element per slot
	// (1 byte). It is enabled only where the arch admits it (sm_120 today). It changes
	// byte accounting, never block topology: the same page boundaries, IDs, and
	// refcounts the generic path produces.
	CUDAPagedPrefillFP8 CUDAPagedPrefillPath = "fp8"
)

// elementBytes is the per-slot width of the path's elements.
func (p CUDAPagedPrefillPath) elementBytes() int {
	if p == CUDAPagedPrefillFP8 {
		return 1
	}
	return 4
}

// The FP8 paged branch's arch admission. INSPIRE-ONLY port of the principle behind
// flashinfer-ai/flashinfer's SM120 FMHA paged prefill
// (flashinfer/attention/cute_dsl/sm120_fmha.py, Apache-2.0): an FP8 paged prefill
// path exists only where the architecture covers it; every other arch falls back to
// the generic path. This file adds no kernel — it selects and accounts the path on
// the host side.
const (
	cudaFP8PagedPrefillMajor = 12
	cudaFP8PagedPrefillMinor = 0
)

// SelectCUDAPagedPrefillPath reports which paged-prefill dtype path a device of
// compute capability (major, minor) runs. Only sm_120 is admitted to the FP8
// branch; every other known arch, and any unknown capability (major <= 0), keeps
// the generic path. Pure: no cgo, no GPU.
func SelectCUDAPagedPrefillPath(major, minor int) CUDAPagedPrefillPath {
	if major > 0 && major == cudaFP8PagedPrefillMajor && minor == cudaFP8PagedPrefillMinor {
		return CUDAPagedPrefillFP8
	}
	return CUDAPagedPrefillGeneric
}

// CUDABlockTable represents a logical-to-physical block table for a session's KV cache (#10723, epic #2236).
// Instead of a single flat device allocation per layer, KV memory is divided into fixed-size
// physical blocks. Cloning a session duplicates only the block table pointers and increments
// refcounts on underlying blocks, achieving zero-copy prefix sharing.
type CUDABlockTable struct {
	mu        sync.Mutex
	blockSize int // tokens per block
	stride    int // elements per token per layer
	blocks    []*CUDAPageBlock
	nextID    int64
	path      CUDAPagedPrefillPath // paged-prefill dtype path; generic unless arch-admitted to FP8
}

// NewCUDABlockTable creates a block table with the given page block token capacity,
// on the generic paged-prefill path.
func NewCUDABlockTable(blockSize, stride int) *CUDABlockTable {
	return newCUDABlockTable(blockSize, stride, CUDAPagedPrefillGeneric)
}

// NewCUDABlockTableForArch creates a block table whose paged-prefill dtype path is
// selected by the device's compute capability: sm_120 admits the FP8 paged branch,
// every other arch (and an unknown capability) keeps the generic path.
func NewCUDABlockTableForArch(blockSize, stride, major, minor int) *CUDABlockTable {
	return newCUDABlockTable(blockSize, stride, SelectCUDAPagedPrefillPath(major, minor))
}

func newCUDABlockTable(blockSize, stride int, path CUDAPagedPrefillPath) *CUDABlockTable {
	if blockSize <= 0 {
		blockSize = 16 // standard PagedAttention default
	}
	if path == "" {
		path = CUDAPagedPrefillGeneric
	}
	return &CUDABlockTable{
		blockSize: blockSize,
		stride:    stride,
		path:      path,
	}
}

// AppendLogicalTokens adds tokens to the logical block table, allocating new blocks as page boundaries are crossed.
func (t *CUDABlockTable) AppendLogicalTokens(count int) []*CUDAPageBlock {
	t.mu.Lock()
	defer t.mu.Unlock()

	var allocated []*CUDAPageBlock
	remaining := count
	for remaining > 0 {
		var active *CUDAPageBlock
		if len(t.blocks) > 0 {
			last := t.blocks[len(t.blocks)-1]
			// Check if last block has room and is not shared (CoW)
			if last.Tokens < t.blockSize && last.RefCount() <= 1 {
				active = last
			}
		}

		if active == nil {
			// Allocate new block
			t.nextID++
			active = &CUDAPageBlock{
				ID:       CUDABlockID(t.nextID),
				Stride:   t.stride,
				refCount: 1,
			}
			t.blocks = append(t.blocks, active)
			allocated = append(allocated, active)
		}

		room := t.blockSize - active.Tokens
		toAdd := remaining
		if toAdd > room {
			toAdd = room
		}
		active.Tokens += toAdd
		active.Bytes += int64(toAdd * t.stride * t.path.elementBytes())
		remaining -= toAdd
	}
	return allocated
}

// Clone creates a shallow copy of the block table and increments refcounts on all physical blocks (#10723).
// Physical memory is shared zero-copy; subsequent appends to the clone allocate new blocks on page boundaries.
func (t *CUDABlockTable) Clone() *CUDABlockTable {
	t.mu.Lock()
	defer t.mu.Unlock()

	clonedBlocks := make([]*CUDAPageBlock, len(t.blocks))
	for i, b := range t.blocks {
		b.IncRef()
		clonedBlocks[i] = b
	}

	return &CUDABlockTable{
		blockSize: t.blockSize,
		stride:    t.stride,
		blocks:    clonedBlocks,
		nextID:    t.nextID,
		path:      t.path,
	}
}

// Blocks returns the slice of page blocks.
func (t *CUDABlockTable) Blocks() []*CUDAPageBlock {
	t.mu.Lock()
	defer t.mu.Unlock()
	return append([]*CUDAPageBlock(nil), t.blocks...)
}

// PagedPrefillPath returns the dtype path this block table pages on.
func (t *CUDABlockTable) PagedPrefillPath() CUDAPagedPrefillPath {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.path
}

// TokenCount returns the total logical tokens mapped.
func (t *CUDABlockTable) TokenCount() int {
	t.mu.Lock()
	defer t.mu.Unlock()
	total := 0
	for _, b := range t.blocks {
		total += b.Tokens
	}
	return total
}

// SharedBytes returns the physical device memory shared with other sessions.
func (t *CUDABlockTable) SharedBytes() int64 {
	t.mu.Lock()
	defer t.mu.Unlock()
	var shared int64
	for _, b := range t.blocks {
		if b.RefCount() > 1 {
			shared += b.Bytes
		}
	}
	return shared
}

// UniqueBytes returns the physical device memory unique to this session.
func (t *CUDABlockTable) UniqueBytes() int64 {
	t.mu.Lock()
	defer t.mu.Unlock()
	var unique int64
	for _, b := range t.blocks {
		if b.RefCount() <= 1 {
			unique += b.Bytes
		}
	}
	return unique
}

// TotalBytes returns the total device memory of all mapped blocks.
func (t *CUDABlockTable) TotalBytes() int64 {
	t.mu.Lock()
	defer t.mu.Unlock()
	var total int64
	for _, b := range t.blocks {
		total += b.Bytes
	}
	return total
}

// Release releases all references held by this block table.
func (t *CUDABlockTable) Release() {
	t.mu.Lock()
	defer t.mu.Unlock()
	for _, b := range t.blocks {
		b.DecRef()
	}
	t.blocks = nil
}
