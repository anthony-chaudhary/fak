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

extern "C" void fcuda_nccl_pg_destroy(void) {
  if (g_pg_world == 0) return;
  ncclCommDestroy(g_pg_comm);
  g_pg_rank = -1;
  g_pg_world = 0;
}
