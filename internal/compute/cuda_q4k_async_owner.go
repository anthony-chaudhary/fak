//go:build cuda && cgo && fak_cuda_q4k_qkv_qualification && fak_cuda_v41_index_select && fak_cuda_q4k_qkv_async

package compute

/*
int fcuda_q4k_qkv_async_finish(void);
*/
import "C"

import (
	"fmt"
	"sync/atomic"
)

// Fixed owner inventory, not a pointer cache. cudaMu protects every access.
// Defaults and ordinary qualification never select this path.
var cudaQ4KAsyncOwner struct {
	owner        *cudaBackend
	scratch      [64]*cudaBuf
	dependencies [64][7]*cudaBuf
	count        int
	bytes        uint64
	failure      int
}

func cudaQ4KAsyncCanRetainLocked(c *cudaBackend, bytes uint64) bool {
	p := &cudaQ4KAsyncOwner
	return c != nil && c == cudaDev && !c.capturing && p.failure == 0 &&
		(p.owner == nil || p.owner == c) && p.count < len(p.scratch) &&
		bytes > 0 && bytes <= (64<<20)-p.bytes
}

func cudaQ4KAsyncRetainLocked(c *cudaBackend, scratch *cudaBuf) {
	if scratch == nil || !cudaQ4KAsyncCanRetainLocked(c, uint64(scratch.n)) {
		panic("Q4 async owner admission changed under cudaMu")
	}
	p := &cudaQ4KAsyncOwner
	p.owner = c
	p.scratch[p.count] = scratch
	p.count++
	p.bytes += uint64(scratch.n)
}

// A failure is sticky: retain both the Go owner references and native storage.
// The existing native checked finish closes allocator/free seams on uncertainty.
func cudaQ4KAsyncDrainLocked(c *cudaBackend) int {
	p := &cudaQ4KAsyncOwner
	if p.failure != 0 {
		return p.failure
	}
	if p.count == 0 {
		return 0
	}
	if p.owner != c || c == nil || c.capturing {
		p.failure = -4
		return p.failure
	}
	status := int(C.fcuda_q4k_qkv_async_finish())
	if status != 0 {
		p.failure = status
		return status
	}
	atomic.AddUint64(&c.fenceGen, 1)
	clear(p.scratch[:])
	clear(p.dependencies[:])
	p.count, p.bytes, p.owner = 0, 0, nil
	return 0
}

func cudaQ4KAsyncRequireDrainLocked(c *cudaBackend) {
	if status := cudaQ4KAsyncDrainLocked(c); status != 0 {
		panic(fmt.Errorf("Q4 async checked owner drain %d; transients retained", status))
	}
}

// Terminal cleanup must preserve the primary failure and all storage.
func cudaQ4KAsyncReleaseBlockedLocked() bool { return cudaQ4KAsyncOwner.failure != 0 }
