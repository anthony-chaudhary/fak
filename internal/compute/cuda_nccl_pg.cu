// cuda_nccl_pg.cu — the multi-PROCESS NCCL process-group bootstrap (#971 follow-on).
//
// cuda_nccl.cu is SINGLE-PROCESS, multi-GPU: one process owns every ncclComm_t
// (ncclCommInitAll). A production sharded serve instead runs N SEPARATE OS processes, one per
// GPU (internal/model/serve_ep.go's topology, reducing today over the host DistComm). This
// file is the device-side primitive that lets one such process join a REAL NCCL communicator
// with its peers: rank 0 mints a unique ID (ncclGetUniqueId), the caller distributes it to
// every rank out-of-band (DistComm.BroadcastFromRoot — an existing host TCP primitive, no new
// transport needed for a 128-byte payload), and every rank calls ncclCommInitRank with that ID
// plus its own rank/device. This is NCCL's other, standard multi-process bootstrap — the one
// torchrun/MPI-style distributed training uses — distinct from ncclCommInitAll.
//
// Compiled offline by nvcc into libfakcuda.a alongside cuda_kernels.cu and cuda_nccl.cu only
// when FAK_CUDA_NCCL=1 (build_cuda.sh), then linked by the cgo wrapper under `-tags cuda,nccl`
// with -lnccl. The default `go build ./cmd/fak` excludes all of this, and plain `-tags cuda`
// stays single-device.
//
// DISTINCT STATE FROM cuda_nccl.cu. g_pg_comm/g_pg_rank/g_pg_world are separate globals from
// cuda_nccl.cu's g_comms/g_world — the two collectives are independent (a process could in
// principle hold both a local ncclCommInitAll set AND a cross-process ncclCommInitRank
// communicator) and must never share communicator state.
//
// REDUCTION ORDER / HONESTY — same as cuda_nccl.cu: NCCL's ring/tree all-reduce sums in a
// hardware-determined order, not the strict rank-ascending order the cpu-ref CollectiveBackend
// / DistComm pin, so this is an Approx peer (argmax-exact + cosine), never max|Δ|=0 vs the host
// reduce it replaces.
//
// STATUS: builds ONLY under a real CUDA+NCCL toolchain (`internal/compute/build_cuda.sh test`
// with FAK_CUDA_NCCL=1) — unverified on a GPU-free host. See dist_device_collective.go and
// cmd/fak/serve.go for the Go-side wiring that calls into this file's ABI.

#include "cuda_backend.h"
#include <cuda_runtime.h>
#include <nccl.h>
#include <stdio.h>
#include <string.h>

// NK/CKR are defined in cuda_backend.h (shared with cuda_nccl.cu).

// This process's single communicator to its peers in the group, formed by fcuda_nccl_pg_init.
// g_pg_world==0 means uninitialized (mirrors cuda_nccl.cu's g_world==0 sentinel).
static ncclComm_t g_pg_comm;
static int g_pg_rank = -1;
static int g_pg_world = 0;
// g_pg_aborted latches the first fcuda_nccl_pg_abort (the TP-group abort-once discipline
// adapted from ashhart/TensorFold zig/src/cuda/tp_group.zig@9e8c368b0b4c2c7892a0c9f0f302d7c164016209,
// Apache-2.0; ADAPT — re-expressed, no upstream code copied). Once set, fcuda_nccl_pg_destroy
// REFUSES the ncclCommDestroy (abort-then-destroy is a no-op — no double teardown of an
// already-aborted communicator). The closed contract is pinned on a CPU host by
// nccl_group_lifecycle_test.go (TestNCCLOpGroupAbortOnceNoDestroy).
static int g_pg_aborted = 0;

extern "C" int fcuda_nccl_pg_get_unique_id(void *out_id) {
  if (out_id == nullptr) return -1;
  ncclUniqueId id;
  NK(ncclGetUniqueId(&id));
  memcpy(out_id, &id, sizeof(ncclUniqueId));
  return 0;
}

extern "C" int fcuda_nccl_pg_init(const void *id, int world, int rank, int device) {
  if (id == nullptr) return -1;
  if (world <= 0 || rank < 0 || rank >= world) return -2;
  if (g_pg_world != 0) return -3; // re-init without fcuda_nccl_pg_destroy is a caller error
  CKR(cudaSetDevice(device));
  ncclUniqueId ncclId;
  memcpy(&ncclId, id, sizeof(ncclUniqueId));
  // ncclCommInitRank blocks until every rank in the group has called it with the SAME id —
  // the rendezvous the caller's DistComm.BroadcastFromRoot round makes possible.
  ncclResult_t r = ncclCommInitRank(&g_pg_comm, world, ncclId, rank);
  if (r != ncclSuccess) {
    fprintf(stderr, "fak-nccl-pg: ncclCommInitRank(world=%d,rank=%d) %s\n", world, rank,
            ncclGetErrorString(r));
    return (int)r;
  }
  g_pg_rank = rank;
  g_pg_world = world;
  return 0;
}

extern "C" int fcuda_nccl_pg_allreduce_f32(void *dbuf, int count) {
  if (g_pg_world == 0) return -1;
  if (count <= 0) return -2;
  // One rank, one device, one buffer — no ncclGroupStart/End bracket needed (that bracket in
  // cuda_nccl.cu exists because one process there issues N per-device calls in a single group;
  // here each process issues exactly one call for its own rank).
  NK(ncclAllReduce(dbuf, dbuf, (size_t)count, ncclFloat32, ncclSum, g_pg_comm, /*stream=*/0));
  CKR(cudaStreamSynchronize(0));
  return 0;
}

// fcuda_nccl_pg_send(dbuf, count, peer): post `count` f32 elements at dbuf to rank `peer`
// over g_pg_comm. Matched counts: the peer's matching fcuda_nccl_pg_recv MUST take the SAME
// element count — NCCL pairs posts by (peer, count, type), so a mismatched count deadlocks
// or corrupts the exchange rather than failing clean. Post-sync check: the default stream is
// synchronized before return so a failed post surfaces HERE, not later in the reader.
// Returns 0 on success, -1 when no group is active, -2 for a non-positive count, -3 for a
// peer outside [0, world), else the NCCL/CUDA error code.
extern "C" int fcuda_nccl_pg_send(void *dbuf, int count, int peer) {
  if (g_pg_world == 0) return -1;
  if (count <= 0) return -2;
  if (peer < 0 || peer >= g_pg_world) return -3;
  NK(ncclSend(dbuf, (size_t)count, ncclFloat32, peer, g_pg_comm, /*stream=*/0));
  CKR(cudaStreamSynchronize(0));
  return 0;
}

// fcuda_nccl_pg_recv(dbuf, count, peer): receive `count` f32 elements FROM rank `peer` into
// dbuf. Same matched-counts contract and post-sync check as fcuda_nccl_pg_send; same return
// codes (-1 uninitialized, -2 bad count, -3 peer out of range).
extern "C" int fcuda_nccl_pg_recv(void *dbuf, int count, int peer) {
  if (g_pg_world == 0) return -1;
  if (count <= 0) return -2;
  if (peer < 0 || peer >= g_pg_world) return -3;
  NK(ncclRecv(dbuf, (size_t)count, ncclFloat32, peer, g_pg_comm, /*stream=*/0));
  CKR(cudaStreamSynchronize(0));
  return 0;
}

// fcuda_nccl_pg_check: the post-sync check (TensorFold Group.check analog) — materialize any
// async error the preceding collectives left on the default stream BEFORE the caller reads
// results. Call it after a synchronize that consumed a peer-posted buffer. -1 when no group
// is active, 0 when the stream is clean, the CUDA error code otherwise.
extern "C" int fcuda_nccl_pg_check(void) {
  if (g_pg_world == 0) return -1;
  CKR(cudaStreamSynchronize(0));
  return 0;
}

// fcuda_nccl_pg_abort: abort-ONCE. The first call latches g_pg_aborted and runs
// ncclCommAbort(g_pg_comm) — NCCL gives up the device waits, unblocking any rank stuck in a
// collective. Later calls are no-ops (0) so no path can double-abort. -1 when no group is
// active.
extern "C" int fcuda_nccl_pg_abort(void) {
  if (g_pg_world == 0) return -1;
  if (g_pg_aborted) return 0;
  g_pg_aborted = 1;
  NK(ncclCommAbort(g_pg_comm));
  return 0;
}

// fcuda_nccl_pg_destroy: ordered teardown with the destroy-after-abort refusal. After an
// abort it performs NO ncclCommDestroy — abort-then-destroy is a no-op: the aborted
// communicator is already dead, and tearing it down a second time is the double-destroy
// hazard this lifecycle exists to prevent. Resets the globals either way so a later
// fcuda_nccl_pg_init can form a fresh group. Safe to call when no group is active (no-op).
extern "C" void fcuda_nccl_pg_destroy(void) {
  if (g_pg_world == 0) return;
  if (!g_pg_aborted) ncclCommDestroy(g_pg_comm);
  g_pg_rank = -1;
  g_pg_world = 0;
  g_pg_aborted = 0;
}
