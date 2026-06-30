//go:build cuda && nccl

// cuda_collective.go — the Go side of the NCCL device collective (#971): it makes the CUDA
// backend implement compute.CollectiveBackend over the real cross-GPU communicator in
// cuda_nccl.cu, so the expert-parallel forward's coll.AllReduceSum can reduce expert partials
// across distinct GPUs instead of through the host.
//
// Compiled only under -tags cuda,nccl (like cuda.go plus the NCCL opt-in); the default pure-Go
// binary and the plain CUDA build never link it.
//
// SCOPE / HONESTY. This is the SINGLE-PROCESS, MULTI-GPU form: one process holds one ncclComm_t
// per visible device (fcuda_nccl_init), and a collective takes one part per rank with part r
// resident on device r. That is exactly the acceptance witness's shape — a device tensor reduced
// across 2 GPUs matching cpu-ref — and the in-process path a single `fak serve` drives. Caps()
// advertises Collective=true ONLY after fcuda_nccl_init succeeds over >1 device, so a host never
// routes through here before the communicator can actually all-reduce across GPUs.
//
// The fail-closed validation mirrors cpuBackend.collectF32 exactly (ready, F32, owned-by-this-
// backend, equal-length where required) so the device collective rejects the same malformed
// inputs the reference does. The reduction itself runs in NCCL's ring/tree order, which sums in
// a hardware-determined (not strict rank-ascending) order — so the result matches cpu-ref's
// AllReduceSum within reassociation round-off (the Approx rung every device op carries), not
// max|Δ|=0. That is the documented, honest device-vs-reference relationship.

package compute

/*
#cgo LDFLAGS: -lnccl
#include "cuda_backend.h"
#include <stdlib.h>
*/
import "C"

import (
	"fmt"
	"sync"
	"sync/atomic"
	"unsafe"
)

var _ CollectiveBackend = (*cudaBackend)(nil)
var _ RankUploader = (*cudaBackend)(nil)
var _ CollectiveInitializer = (*cudaBackend)(nil)

// cudaNCCLWorld (declared in cuda_collective_state.go so Caps() reads it in the base build too)
// is set here by ensureNCCL once the communicator is up. ncclInitOnce serializes that one-time init.
var ncclInitOnce sync.Mutex

// UploadRank places an F32 host tensor on a specific CUDA device. Generic Upload is device-0;
// BackendCollective uses this optional seam so rank r's part is truly resident on GPU r before
// an NCCL collective sees it.
func (c *cudaBackend) UploadRank(t Tensor, as Dtype, rank int) (Tensor, error) {
	if rank < 0 {
		return Tensor{}, fmt.Errorf("compute: UploadRank got negative rank %d", rank)
	}
	if t.Dtype != F32 || as != F32 {
		return Tensor{}, fmt.Errorf("compute: CUDA UploadRank supports only F32 today (got tensor=%s as=%s)", t.Dtype, as)
	}
	hb, ok := t.buf.(HostBuffer)
	if !ok {
		return Tensor{}, fmt.Errorf("compute: CUDA UploadRank expects host data")
	}
	f := hb.F32()
	cudaMu.Lock()
	defer cudaMu.Unlock()
	buf, err := c.dallocOnDevice(rank, t.Numel()*F32.Bytes())
	if err != nil {
		return Tensor{}, err
	}
	out := makeTensor(c, F32, RowMajor, append([]int(nil), t.Shape...), nil, buf)
	if len(f) > 0 {
		C.fcuda_h2d_on(C.int(rank), buf.ptr, unsafe.Pointer(&f[0]), C.size_t(len(f)*4))
	}
	return out, nil
}

func (c *cudaBackend) dallocOnDevice(device, nbytes int) (*cudaBuf, error) {
	p := C.fcuda_malloc_on(C.int(device), C.size_t(nbytes))
	if p == nil {
		return nil, fmt.Errorf("compute: cudaMalloc on rank/device %d failed for %d bytes", device, nbytes)
	}
	return &cudaBuf{ptr: unsafe.Pointer(p), n: nbytes, device: device}, nil
}

func (c *cudaBackend) devOnDevice(device int, shape []int, dt Dtype) (Tensor, *cudaBuf, error) {
	n := 1
	for _, d := range shape {
		n *= d
	}
	buf, err := c.dallocOnDevice(device, n*dt.Bytes())
	if err != nil {
		return Tensor{}, nil, err
	}
	return makeTensor(c, dt, RowMajor, append([]int(nil), shape...), nil, buf), buf, nil
}

// InitCollective boots the NCCL communicator for serve before the gateway checks
// Caps().Collective. Without this explicit bootstrap, the first collective call would be the
// only place ensureNCCL can run, but the serve gate correctly refuses before that call exists.
func (c *cudaBackend) InitCollective(world int) error {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	return ensureNCCL(world)
}

// ensureNCCL brings up one communicator per device 0..world-1 the first time a collective is
// called (ncclCommInitAll, single-process-multi-GPU). It is idempotent and fails closed: on any
// NCCL/CUDA error cudaNCCLWorld stays 0 so Caps().Collective stays false. Caller holds cudaMu.
func ensureNCCL(world int) error {
	if world < 2 {
		return fmt.Errorf("compute: NCCL collective needs world >= 2 (got %d); a 1-rank collective is the identity and does not need a device communicator", world)
	}
	ncclInitOnce.Lock()
	defer ncclInitOnce.Unlock()
	if int(atomic.LoadInt32(&cudaNCCLWorld)) == world {
		return nil // already up over this many ranks
	}
	if rc := int(C.fcuda_nccl_init(C.int(world))); rc != 0 {
		return fmt.Errorf("compute: fcuda_nccl_init(%d) failed rc=%d (no NCCL communicator; Caps().Collective stays false)", world, rc)
	}
	atomic.StoreInt32(&cudaNCCLWorld, int32(world))
	return nil
}

// collectDevice validates the per-rank device parts against the same fail-closed contract as
// cpuBackend.collectF32 (ready, F32, owned by THIS backend) and returns their device pointers in
// rank order, plus the per-rank element count. requireEqualLen pins the AllReduceSum/ReduceScatter
// equal-length rule; AllGather passes false. A part r is expected resident on device r (the
// single-process-multi-GPU convention); a real all-reduce rejects a cross-backend tensor here,
// the device analog of the host seam's affinity check.
func (c *cudaBackend) collectDevice(parts []Tensor, requireEqualLen bool) ([]unsafe.Pointer, int, error) {
	if len(parts) == 0 {
		return nil, 0, fmt.Errorf("compute: collective got no rank parts")
	}
	ptrs := make([]unsafe.Pointer, len(parts))
	count0 := 0
	for r, p := range parts {
		if p.Backend() != Backend(c) {
			return nil, 0, fmt.Errorf("compute: collective rank %d tensor is owned by a different backend (cross-backend reduction is rejected)", r)
		}
		if p.Dtype != F32 {
			return nil, 0, fmt.Errorf("compute: collective rank %d dtype = %s, want f32", r, p.Dtype)
		}
		if !p.Ready() {
			return nil, 0, fmt.Errorf("compute: collective rank %d tensor is not ready", r)
		}
		db, ok := p.buf.(*cudaBuf)
		if !ok || db.ptr == nil {
			return nil, 0, fmt.Errorf("compute: collective rank %d tensor has no device buffer", r)
		}
		if db.device != r {
			return nil, 0, fmt.Errorf("compute: collective rank %d tensor is resident on CUDA device %d, want device %d", r, db.device, r)
		}
		nElem := p.Numel()
		if r == 0 {
			count0 = nElem
		} else if requireEqualLen && nElem != count0 {
			return nil, 0, fmt.Errorf("compute: AllReduceSum rank %d len = %d, want %d (ragged partials)", r, nElem, count0)
		}
		ptrs[r] = db.ptr
	}
	return ptrs, count0, nil
}

// AllReduceSum reduces the equal-length per-rank device partials with a real NCCL all-reduce-SUM
// (each part r resident on device r), returning a NEW device tensor on device 0 holding the sum.
// It is the device twin of cpuBackend.AllReduceSum: same fail-closed contract, same SUM, but the
// NCCL reduction order makes it Approx (cosine/round-off), not max|Δ|=0, vs the reference.
func (c *cudaBackend) AllReduceSum(parts []Tensor) (Tensor, error) {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	if len(parts) == 1 {
		if _, _, err := c.collectDevice(parts, true); err != nil {
			return Tensor{}, err
		}
		return parts[0], nil
	}
	if err := ensureNCCL(len(parts)); err != nil {
		return Tensor{}, err
	}
	ptrs, count, err := c.collectDevice(parts, true)
	if err != nil {
		return Tensor{}, err
	}
	// NCCL all-reduce is in-place across the parts; every part ends up holding the sum. Run it,
	// then materialize rank 0's result into a fresh tensor the caller owns (the parts are the
	// caller's; we do not return one of them as the result to keep ownership clean).
	cArr := devPtrArray(ptrs)
	defer freeDevPtrArray(cArr)
	if rc := int(C.fcuda_nccl_allreduce_f32(cArr, C.int(len(parts)), C.int(count))); rc != 0 {
		return Tensor{}, fmt.Errorf("compute: fcuda_nccl_allreduce_f32 rc=%d", rc)
	}
	out, _, err := c.devOnDevice(0, []int{count}, F32)
	if err != nil {
		return Tensor{}, err
	}
	C.fcuda_d2d_on(0, out.buf.(*cudaBuf).ptr, ptrs[0], C.size_t(count*4))
	return out, nil
}

// AllGather concatenates the per-rank shards in rank order into a new device tensor on device 0
// (drecv = parts[0]||parts[1]||...). Equal shard sizes only (NCCL all-gather is even-band).
func (c *cudaBackend) AllGather(parts []Tensor) (Tensor, error) {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	if len(parts) == 1 {
		if _, _, err := c.collectDevice(parts, false); err != nil {
			return Tensor{}, err
		}
		return parts[0], nil
	}
	if err := ensureNCCL(len(parts)); err != nil {
		return Tensor{}, err
	}
	ptrs, count, err := c.collectDevice(parts, true) // NCCL all-gather requires equal sendcount
	if err != nil {
		return Tensor{}, err
	}
	n := len(parts)
	// Allocate one recv buffer per rank (n*count each, on each device) so the grouped NCCL call
	// has a destination on every device; return rank 0's gathered buffer.
	recv := make([]unsafe.Pointer, n)
	recvBufs := make([]*cudaBuf, n)
	for r := 0; r < n; r++ {
		_, b, err := c.devOnDevice(r, []int{n * count}, F32)
		if err != nil {
			return Tensor{}, err
		}
		recv[r] = b.ptr
		recvBufs[r] = b
	}
	sArr, rArr := devPtrArray(ptrs), devPtrArray(recv)
	defer freeDevPtrArray(sArr)
	defer freeDevPtrArray(rArr)
	if rc := int(C.fcuda_nccl_allgather_f32(sArr, rArr, C.int(n), C.int(count))); rc != 0 {
		return Tensor{}, fmt.Errorf("compute: fcuda_nccl_allgather_f32 rc=%d", rc)
	}
	return makeTensor(c, F32, RowMajor, []int{n * count}, nil, recvBufs[0]), nil
}

// ReduceScatter reduces the per-rank partials and returns every rank's 1/n band (the dual of
// AllGather: AllReduceSum == AllGather(ReduceScatter)). Equal-length partials, evenly divisible.
func (c *cudaBackend) ReduceScatter(parts []Tensor) ([]Tensor, error) {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	if len(parts) == 1 {
		if _, _, err := c.collectDevice(parts, true); err != nil {
			return nil, err
		}
		return []Tensor{parts[0]}, nil
	}
	if err := ensureNCCL(len(parts)); err != nil {
		return nil, err
	}
	ptrs, count, err := c.collectDevice(parts, true)
	if err != nil {
		return nil, err
	}
	n := len(parts)
	recvCount, err := evenShard("ReduceScatter partial", "reduce-scatter", count, n)
	if err != nil {
		return nil, err
	}
	recv := make([]unsafe.Pointer, n)
	out := make([]Tensor, n)
	for r := 0; r < n; r++ {
		t, b, err := c.devOnDevice(r, []int{recvCount}, F32)
		if err != nil {
			return nil, err
		}
		recv[r] = b.ptr
		out[r] = t
	}
	sArr, rArr := devPtrArray(ptrs), devPtrArray(recv)
	defer freeDevPtrArray(sArr)
	defer freeDevPtrArray(rArr)
	if rc := int(C.fcuda_nccl_reducescatter_f32(sArr, rArr, C.int(n), C.int(recvCount))); rc != 0 {
		return nil, fmt.Errorf("compute: fcuda_nccl_reducescatter_f32 rc=%d", rc)
	}
	return out, nil
}

// AllToAll is the layout-transpose collective EP/TP layout changes need. NCCL has no single
// all-to-all symbol (it is grouped Send/Recv); that device kernel is the documented follow-on,
// so this fails closed rather than silently mis-serving — the routed AllReduceSum/AllGather/
// ReduceScatter path above is the part EP actually exercises today.
func (c *cudaBackend) AllToAll(parts []Tensor) ([]Tensor, error) {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	if len(parts) == 1 {
		if _, _, err := c.collectDevice(parts, true); err != nil {
			return nil, err
		}
		return []Tensor{parts[0]}, nil
	}
	return nil, fmt.Errorf("compute: cuda AllToAll (grouped ncclSend/ncclRecv) is not yet implemented; use AllReduceSum/AllGather/ReduceScatter (#971)")
}

// devPtrArray copies a Go slice of device pointers into a freshly C.malloc'd void** the NCCL ABI
// reads. The C array is heap-allocated (not a Go slice's backing array) so cgo's pointer-passing
// rules are satisfied — a Go slice of unsafe.Pointer cannot be passed as void** directly. The
// caller MUST defer freeDevPtrArray on the returned pointer.
func devPtrArray(ptrs []unsafe.Pointer) *unsafe.Pointer {
	n := len(ptrs)
	arr := C.malloc(C.size_t(n) * C.size_t(unsafe.Sizeof(uintptr(0))))
	slice := (*[1 << 20]unsafe.Pointer)(arr)[:n:n]
	for i, p := range ptrs {
		slice[i] = p
	}
	return (*unsafe.Pointer)(arr)
}

// freeDevPtrArray releases a devPtrArray allocation.
func freeDevPtrArray(arr *unsafe.Pointer) { C.free(unsafe.Pointer(arr)) }
