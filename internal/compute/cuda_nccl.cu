// cuda_nccl.cu — the NCCL device-collective seam behind compute.CollectiveBackend (#971).
//
// Compiled offline by nvcc into libfakcuda.a alongside cuda_kernels.cu only when
// FAK_CUDA_NCCL=1, then linked by the cgo wrapper under `-tags cuda,nccl` with -lnccl. The
// default `go build ./cmd/fak` excludes all of this, and plain `-tags cuda` stays single-device.
//
// WHAT THIS IS. A REAL cross-device reduction: NCCL all-reduce / all-gather / reduce-scatter
// over distinct GPUs, the cross-GPU communicator the CollectiveBackend interface needs to be
// honest (a host-staged d2h+sum+h2d would also "work" but is not a device collective and is
// not what a multi-GPU serve runs). This first cut is SINGLE-PROCESS, MULTI-GPU: one process
// holds one ncclComm_t per visible device (the canonical ncclCommInitAll setup), so a single
// `fak serve` drives N GPUs. That is exactly the shape of the acceptance witness — a device
// tensor reduced across 2 GPUs matching the cpu-ref rank-order sum — and the in-process form
// the EP forward's coll.AllReduceSum binds to. Multi-PROCESS NCCL over a rank file (the form a
// production 8-GPU serve uses to dodge the single-cudaMu-stream limit) is the documented
// follow-on; this proves the device communicator end-to-end first.
//
// REDUCTION ORDER / HONESTY. NCCL's ring/tree all-reduce sums in a hardware-determined order,
// not the strict rank-ascending order the cpu-ref CollectiveBackend pins. Float addition is
// not associative, so the device result matches cpu-ref's AllReduceSum within reassociation
// round-off (the same ~1e-6 the model package already documents for any reordered fdot), NOT
// bit-for-bit. The acceptance gate is therefore the Approx rung (argmax-exact + cosine), the
// same class every device op in cuda_kernels.cu already carries — never max|Δ|=0. A future
// in-rank-order reference is a host-staged sum; this file is the performant device path.

#include "cuda_backend.h"
#include <cuda_runtime.h>
#include <nccl.h>
#include <stdio.h>
#include <vector>

#define NK(call) do { ncclResult_t _r = (call); if (_r != ncclSuccess) { \
  fprintf(stderr, "fak-nccl: %s:%d %s\n", __FILE__, __LINE__, ncclGetErrorString(_r)); \
  return (int)_r; } } while (0)

#define CKR(call) do { cudaError_t _e = (call); if (_e != cudaSuccess) { \
  fprintf(stderr, "fak-nccl: %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
  return (int)_e + 1000; } } while (0)

// One communicator per device, all in this process (ncclCommInitAll). g_world is the rank
// count once init succeeds; 0 means uninitialized, which keeps Caps().Collective false on the
// Go side. Single-threaded by the Go-side cudaMu mutex (collectives serialize with every other
// device op), so plain globals are safe.
static std::vector<ncclComm_t> g_comms;
static int g_world = 0;

extern "C" int fcuda_nccl_init(int n) {
  if (n <= 0) return -1;
  if (g_world == n) return 0;  // idempotent: already initialized over this many ranks
  if (g_world != 0) return -2; // re-init with a different world is a caller error (free first)
  int avail = 0;
  if (cudaGetDeviceCount(&avail) != cudaSuccess) return -3;
  if (n > avail) return -4;    // asked for more ranks than the box has GPUs

  std::vector<int> devs(n);
  for (int i = 0; i < n; i++) devs[i] = i;
  g_comms.resize(n);
  // ncclCommInitAll builds all n single-process communicators at once (it internally group-
  // brackets the init), the standard one-process-multi-GPU path. On any failure leave g_world
  // 0 so the Go side never advertises a half-built communicator.
  ncclResult_t r = ncclCommInitAll(g_comms.data(), n, devs.data());
  if (r != ncclSuccess) {
    fprintf(stderr, "fak-nccl: ncclCommInitAll(%d) %s\n", n, ncclGetErrorString(r));
    g_comms.clear();
    return (int)r;
  }
  g_world = n;
  return 0;
}

extern "C" int fcuda_nccl_world(void) { return g_world; }

// fcuda_nccl_allreduce_f32: in-place all-reduce-SUM. dbufs[r] lives on device r. The whole set
// of per-rank ncclAllReduce calls is bracketed by ncclGroupStart/End so NCCL schedules them as
// one collective (required for single-process-multi-GPU — without the group bracket the calls
// would deadlock waiting on each other). Then a per-device sync materializes the result before
// return so the Go-side Read sees finished data.
extern "C" int fcuda_nccl_allreduce_f32(void **dbufs, int n, int count) {
  if (g_world == 0) return -1;
  if (n != g_world) return -2;
  if (count <= 0) return -3;
  NK(ncclGroupStart());
  for (int r = 0; r < n; r++) {
    // Set the device before each grouped collective: in single-process-multi-GPU NCCL the call
    // for rank r must run with device r current, or the exchange silently no-ops (the smoke test
    // returned each rank's own value unsummed until this was added).
    CKR(cudaSetDevice(r));
    NK(ncclAllReduce(dbufs[r], dbufs[r], (size_t)count, ncclFloat32, ncclSum, g_comms[r],
                     /*stream=*/0));
  }
  NK(ncclGroupEnd());
  for (int r = 0; r < n; r++) {
    CKR(cudaSetDevice(r));
    CKR(cudaStreamSynchronize(0));
  }
  return 0;
}

// fcuda_nccl_allgather_f32: dsend[r] (sendcount on device r) -> drecv[r] (n*sendcount on
// device r), rank-ordered. ncclAllGather places rank r's send block at offset r*sendcount in
// every recv buffer, so drecv = send[0]||send[1]||... on every device — the column-parallel
// gather. Equal shard sizes only (NCCL all-gather is even-band by construction).
extern "C" int fcuda_nccl_allgather_f32(void **dsend, void **drecv, int n, int sendcount) {
  if (g_world == 0) return -1;
  if (n != g_world) return -2;
  if (sendcount <= 0) return -3;
  NK(ncclGroupStart());
  for (int r = 0; r < n; r++) {
    CKR(cudaSetDevice(r));
    NK(ncclAllGather(dsend[r], drecv[r], (size_t)sendcount, ncclFloat32, g_comms[r],
                     /*stream=*/0));
  }
  NK(ncclGroupEnd());
  for (int r = 0; r < n; r++) {
    CKR(cudaSetDevice(r));
    CKR(cudaStreamSynchronize(0));
  }
  return 0;
}

// fcuda_nccl_reducescatter_f32: dsend[r] (n*recvcount on device r) reduced-SUM across ranks,
// scattered so drecv[r] (recvcount on device r) holds rank r's 1/n band — the dual of
// all-gather and the third Megatron collective (AllReduceSum == AllGather o ReduceScatter).
extern "C" int fcuda_nccl_reducescatter_f32(void **dsend, void **drecv, int n, int recvcount) {
  if (g_world == 0) return -1;
  if (n != g_world) return -2;
  if (recvcount <= 0) return -3;
  NK(ncclGroupStart());
  for (int r = 0; r < n; r++) {
    CKR(cudaSetDevice(r));
    NK(ncclReduceScatter(dsend[r], drecv[r], (size_t)recvcount, ncclFloat32, ncclSum,
                         g_comms[r], /*stream=*/0));
  }
  NK(ncclGroupEnd());
  for (int r = 0; r < n; r++) {
    CKR(cudaSetDevice(r));
    CKR(cudaStreamSynchronize(0));
  }
  return 0;
}
