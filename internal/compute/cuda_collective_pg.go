//go:build cuda && nccl && cgo

// cuda_collective_pg.go — the Go side of the multi-PROCESS NCCL process group (cuda_nccl_pg.cu,
// #971 follow-on). Distinct from cuda_collective.go's CollectiveBackend (single-process,
// ncclCommInitAll, one call taking every rank's Tensor): this file's shape is the natural
// multi-process one — THIS process holds exactly one Tensor (its own rank's) and joins a
// communicator formed out-of-band via NCCL's other bootstrap (ncclGetUniqueId/ncclCommInitRank).
// internal/model/dist_device_collective.go bridges this to the model.Collective seam the
// sharded EP serve consumes; cmd/fak/serve.go wires it as an opt-in upgrade over the existing
// host DistComm reduce.
//
// Compiled only under -tags cuda,nccl, same as cuda_collective.go; the default pure-Go binary
// and the plain CUDA build never link it.
//
// STATUS: unverified on a GPU-free host — builds and runs only on a real CUDA+NCCL toolchain
// (internal/compute/build_cuda.sh test, FAK_CUDA_NCCL=1, 2+ processes on distinct GPUs).

package compute

/*
#cgo LDFLAGS: -lnccl
#include "cuda_backend.h"

// The TP-group lifecycle additions below land in cuda_nccl.cu / cuda_nccl_pg.cu ahead of
// their cuda_backend.h declarations (that header is outside this ticket's fence); the local
// externs keep cgo honest until the ABI moves into the shared header (ticket follow-up).
extern int fcuda_nccl_abort(void);
extern int fcuda_nccl_destroy(void);
extern int fcuda_nccl_pg_send(void *dbuf, int count, int peer);
extern int fcuda_nccl_pg_recv(void *dbuf, int count, int peer);
extern int fcuda_nccl_pg_check(void);
extern int fcuda_nccl_pg_abort(void);
*/
import "C"

import (
	"fmt"
	"unsafe"
)

var _ ProcessGroupBackend = (*cudaBackend)(nil)

// ncclUniqueIDBytes is NCCL_UNIQUE_ID_BYTES (128) — the fixed size of the out-of-band-
// distributed bootstrap ID ProcessGroupUniqueID fills and InitProcessGroup consumes.
const ncclUniqueIDBytes = 128

// ProcessGroupUniqueID mints a fresh NCCL unique ID via fcuda_nccl_pg_get_unique_id.
func (c *cudaBackend) ProcessGroupUniqueID() ([]byte, error) {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	buf := make([]byte, ncclUniqueIDBytes)
	if rc := int(C.fcuda_nccl_pg_get_unique_id(unsafe.Pointer(&buf[0]))); rc != 0 {
		return nil, fmt.Errorf("compute: fcuda_nccl_pg_get_unique_id rc=%d", rc)
	}
	return buf, nil
}

// InitProcessGroup joins this process to the group via fcuda_nccl_pg_init. Fails closed on a
// malformed id rather than passing a short/long buffer into the C ABI.
func (c *cudaBackend) InitProcessGroup(id []byte, world, rank, device int) error {
	if len(id) != ncclUniqueIDBytes {
		return fmt.Errorf("compute: InitProcessGroup id is %d bytes, want %d (NCCL_UNIQUE_ID_BYTES)", len(id), ncclUniqueIDBytes)
	}
	cudaMu.Lock()
	defer cudaMu.Unlock()
	if rc := int(C.fcuda_nccl_pg_init(unsafe.Pointer(&id[0]), C.int(world), C.int(rank), C.int(device))); rc != 0 {
		return fmt.Errorf("compute: fcuda_nccl_pg_init(world=%d,rank=%d,device=%d) rc=%d", world, rank, device, rc)
	}
	return nil
}

// AllReduceSumPG all-reduce-SUMs this process's single device tensor across the process group
// via fcuda_nccl_pg_allreduce_f32, mirroring collectDevice's fail-closed validation (F32, ready,
// owned by this backend) since there is only one part to validate here.
func (c *cudaBackend) AllReduceSumPG(t Tensor) (Tensor, error) {
	if t.Backend() != Backend(c) {
		return Tensor{}, fmt.Errorf("compute: AllReduceSumPG tensor is owned by a different backend (cross-backend reduction is rejected)")
	}
	if t.Dtype != F32 {
		return Tensor{}, fmt.Errorf("compute: AllReduceSumPG dtype = %s, want f32", t.Dtype)
	}
	if !t.Ready() {
		return Tensor{}, fmt.Errorf("compute: AllReduceSumPG tensor is not ready")
	}
	db, ok := t.buf.(*cudaBuf)
	if !ok || db.ptr == nil {
		return Tensor{}, fmt.Errorf("compute: AllReduceSumPG tensor has no device buffer")
	}
	cudaMu.Lock()
	defer cudaMu.Unlock()
	if db.invalidForSubmit() {
		return Tensor{}, fmt.Errorf("compute: AllReduceSumPG tensor is not ready")
	}
	count := t.Numel()
	if rc := int(C.fcuda_nccl_pg_allreduce_f32(db.ptr, C.int(count))); rc != 0 {
		return Tensor{}, fmt.Errorf("compute: fcuda_nccl_pg_allreduce_f32 rc=%d", rc)
	}
	return t, nil
}

// ProcessGroupSend posts this process's single f32 device tensor to rank `peer` over the
// process group via fcuda_nccl_pg_send. Matched counts: the peer's ProcessGroupRecv must
// present a tensor with the SAME element count — the count is derived from the tensor here,
// so a mismatch is refused by the C layer or deadlocks inside NCCL; the closed contract is
// pinned by TestNCCLOpGroupAbortOnceNoDestroy's matched-send-recv subtest.
func (c *cudaBackend) ProcessGroupSend(t Tensor, peer int) error {
	if t.Backend() != Backend(c) {
		return fmt.Errorf("compute: ProcessGroupSend tensor is owned by a different backend (cross-backend exchange is rejected)")
	}
	if t.Dtype != F32 {
		return fmt.Errorf("compute: ProcessGroupSend dtype = %s, want f32", t.Dtype)
	}
	if !t.Ready() {
		return fmt.Errorf("compute: ProcessGroupSend tensor is not ready")
	}
	db, ok := t.buf.(*cudaBuf)
	if !ok || db.ptr == nil {
		return fmt.Errorf("compute: ProcessGroupSend tensor has no device buffer")
	}
	cudaMu.Lock()
	defer cudaMu.Unlock()
	if db.invalidForSubmit() {
		return fmt.Errorf("compute: ProcessGroupSend tensor is not ready")
	}
	if rc := int(C.fcuda_nccl_pg_send(db.ptr, C.int(t.Numel()), C.int(peer))); rc != 0 {
		return fmt.Errorf("compute: fcuda_nccl_pg_send(peer=%d) rc=%d", peer, rc)
	}
	return nil
}

// ProcessGroupRecv receives this process's single f32 device tensor FROM rank `peer` over
// the process group via fcuda_nccl_pg_recv. Same fail-closed validation and matched-counts
// contract as ProcessGroupSend (the peer's send must carry the same element count).
func (c *cudaBackend) ProcessGroupRecv(t Tensor, peer int) error {
	if t.Backend() != Backend(c) {
		return fmt.Errorf("compute: ProcessGroupRecv tensor is owned by a different backend (cross-backend exchange is rejected)")
	}
	if t.Dtype != F32 {
		return fmt.Errorf("compute: ProcessGroupRecv dtype = %s, want f32", t.Dtype)
	}
	if !t.Ready() {
		return fmt.Errorf("compute: ProcessGroupRecv tensor is not ready")
	}
	db, ok := t.buf.(*cudaBuf)
	if !ok || db.ptr == nil {
		return fmt.Errorf("compute: ProcessGroupRecv tensor has no device buffer")
	}
	cudaMu.Lock()
	defer cudaMu.Unlock()
	if db.invalidForSubmit() {
		return fmt.Errorf("compute: ProcessGroupRecv tensor is not ready")
	}
	if rc := int(C.fcuda_nccl_pg_recv(db.ptr, C.int(t.Numel()), C.int(peer))); rc != 0 {
		return fmt.Errorf("compute: fcuda_nccl_pg_recv(peer=%d) rc=%d", peer, rc)
	}
	return nil
}

// ProcessGroupCheck is the post-sync check (fcuda_nccl_pg_check): it materializes any async
// error the preceding process-group collective left on the default stream BEFORE the caller
// reads results. Call it after a synchronize that consumed a peer-posted buffer; it returns
// nil when the stream is clean.
func (c *cudaBackend) ProcessGroupCheck() error {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	if rc := int(C.fcuda_nccl_pg_check()); rc != 0 {
		return fmt.Errorf("compute: fcuda_nccl_pg_check rc=%d", rc)
	}
	return nil
}

// ProcessGroupAbort aborts the process-group communicator ONCE (fcuda_nccl_pg_abort): NCCL
// gives up the device waits, unblocking any rank stuck in a collective. Subsequent calls
// are no-ops (abort-once), and after an abort DestroyProcessGroup refuses the second
// teardown on the C side (abort-then-destroy is a no-op — no double destroy).
func (c *cudaBackend) ProcessGroupAbort() error {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	if rc := int(C.fcuda_nccl_pg_abort()); rc != 0 {
		return fmt.Errorf("compute: fcuda_nccl_pg_abort rc=%d", rc)
	}
	return nil
}

// DestroyProcessGroup tears down this process's NCCL process-group communicator via
// fcuda_nccl_pg_destroy. Safe to call when no group is active (the C side no-ops). After a
// ProcessGroupAbort the C side REFUSES the ncclCommDestroy — abort-then-destroy is a no-op,
// so this call only resets the group state and never destroys the aborted communicator a
// second time.
func (c *cudaBackend) DestroyProcessGroup() {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	C.fcuda_nccl_pg_destroy()
}
