package compute

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// This executes the literal device helpers and single-process NCCL host bodies.
// It proves scoped selection and ownership, not CUDA/NCCL compilation, collective
// error recovery, physical pointer affinity, or two-GPU/Blackwell execution.
// fak-test:runtime medium est=1s lane=default
func TestCUDADeviceHelpersRestoreCallerDevice(t *testing.T) {
	t.Parallel()
	var compiler string
	for _, name := range []string{"c++", "g++", "clang++"} {
		if path, err := exec.LookPath(name); err == nil {
			compiler = path
			break
		}
	}
	if compiler == "" {
		t.Skip("host C++ compiler unavailable; CUDA device-restoration witness not run")
	}
	read := func(name string) string {
		t.Helper()
		data, err := os.ReadFile(name)
		if err != nil {
			t.Fatal(err)
		}
		return string(data)
	}
	section := func(source, begin, end string) string {
		t.Helper()
		start := strings.Index(source, begin)
		if start < 0 {
			t.Fatalf("CUDA source section start not found: %s", begin)
		}
		stop := strings.Index(source[start:], end)
		if stop < 0 {
			t.Fatalf("CUDA source section end not found: %s", end)
		}
		return source[start : start+stop]
	}
	kernels, nccl := read("cuda_kernels.cu"), read("cuda_nccl.cu")
	ncclStart := strings.Index(nccl, "// One communicator per device")
	if ncclStart < 0 {
		t.Fatal("NCCL state/function section not found")
	}
	fixture := cudaDeviceMockPrefix +
		section(kernels, `extern "C" void *fcuda_malloc(`, `extern "C" void *fcuda_malloc_managed(`) +
		section(kernels, `extern "C" int fcuda_set_device(`, "// host-transfer witness accessors") +
		nccl[ncclStart:] + cudaDeviceMockMain
	dir := t.TempDir()
	files := map[string]string{
		"cuda_runtime.h":      cudaDeviceMockRuntime,
		"cuda_backend.h":      read("cuda_backend.h"),
		"cuda_device_scope.h": read("cuda_device_scope.h"),
		"device.cpp":          fixture,
	}
	for name, contents := range files {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(contents), 0600); err != nil {
			t.Fatal(err)
		}
	}
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	bin := filepath.Join(dir, "device-probe")
	if output, err := exec.CommandContext(ctx, compiler, "-std=c++11", "-Wall", "-Wextra", "-Werror", "-I", dir, filepath.Join(dir, "device.cpp"), "-o", bin).CombinedOutput(); err != nil {
		t.Fatalf("compile literal CUDA device host bodies: %v\n%s", err, output)
	}
	output, err := exec.CommandContext(ctx, bin).CombinedOutput()
	if err != nil {
		t.Fatalf("CUDA device restoration witness: %v\n%s", err, output)
	}
	if !strings.Contains(string(output), "restore device 0 failed") {
		t.Fatalf("injected restoration failure lost its diagnostic:\n%s", output)
	}
}

const cudaDeviceMockRuntime = `
#ifndef MOCK_CUDA_RUNTIME_H
#define MOCK_CUDA_RUNTIME_H
using cudaError_t = int;
const int cudaSuccess = 0;
int cudaGetDevice(int *);
int cudaSetDevice(int);
const char *cudaGetErrorString(int);
#endif
`

const cudaDeviceMockPrefix = `
#include "cuda_backend.h"
#include "cuda_device_scope.h"
#include <cstdlib>
#include <unordered_map>
#include <vector>
using ncclResult_t = int;
using ncclComm_t = int;
const int ncclSuccess = 0, ncclFloat32 = 0, ncclSum = 0;
const int cudaMemcpyHostToDevice = 1, cudaMemcpyDeviceToHost = 2, cudaMemcpyDeviceToDevice = 3;
static thread_local int current = 0;
static int expected = 1, get_calls, set_calls, work_calls, collective_calls, sync_calls;
static int get_error, fail_set_call, work_error, init_error, group_start_error;
static int fail_collective_call, group_end_error, fail_sync_call;
static int storage[32], allocated;
static std::unordered_map<void *, int> allocation_devices;
static std::unordered_map<size_t, std::vector<void *>> g_pool;
static std::unordered_map<void *, size_t> g_live;
static bool g_capture_open = false;
static size_t g_h2d_bytes = 0, g_host_bytes = 0;
static int g_stream = 17;
static void require(bool ok, const char *why) {
  if (!ok) { fprintf(stderr, "%s\n", why); exit(1); }
}
int cudaGetDevice(int *device) {
  ++get_calls;
  if (get_error) return get_error;
  *device = current; return 0;
}
int cudaSetDevice(int device) {
  if (++set_calls == fail_set_call) return 72;
  current = device; return 0;
}
const char *cudaGetErrorString(int) { return "injected CUDA failure"; }
static int work() {
  require(current == expected, "helper work used the wrong device");
  ++work_calls; return work_error;
}
int cudaMalloc(void **ptr, size_t) {
  if (int error = work()) return error;
  require(allocated < 32, "mock allocation capacity exhausted");
  *ptr = &storage[allocated++]; allocation_devices[*ptr] = current; return 0;
}
int cudaFree(void *) { return work(); }
int cudaMemcpy(void *, const void *, size_t, int) { return work(); }
int cudaMemcpyAsync(void *, const void *, size_t, int, int stream) {
  require(stream == g_stream, "device copy changed the backend stream"); return work();
}
static void q4_coeff_cache_evict(const void *) {
  require(current == expected, "cache eviction used the wrong device");
}
#define CK(call) do { int error = (call); if (error) fprintf(stderr, "mock CUDA error %d\n", error); } while (0)
int cudaGetDeviceCount(int *n) { *n = 3; return 0; }
const char *ncclGetErrorString(int) { return "injected NCCL failure"; }
int ncclCommInitAll(ncclComm_t *comms, int n, const int *) {
  for (int r = 0; r < n; ++r) comms[r] = r;
  current = n - 1; // Treat third-party initialization as a temporary selector too.
  return init_error;
}
int ncclGroupStart() { return group_start_error; }
int ncclGroupEnd() { return group_end_error; }
static int collective(ncclComm_t comm, int stream) {
  require(current == comm && stream == 0, "NCCL rank used wrong device/stream");
  return ++collective_calls == fail_collective_call ? 91 : 0;
}
int ncclAllReduce(const void *, void *, size_t, int, int, ncclComm_t comm, int stream) {
  return collective(comm, stream);
}
int ncclAllGather(const void *, void *, size_t, int, ncclComm_t comm, int stream) {
  return collective(comm, stream);
}
int ncclReduceScatter(const void *, void *, size_t, int, int, ncclComm_t comm, int stream) {
  return collective(comm, stream);
}
int cudaStreamSynchronize(int stream) {
  require(stream == 0 && current == sync_calls, "NCCL sync used wrong device/stream");
  return ++sync_calls == fail_sync_call ? 73 : 0;
}
`

const cudaDeviceMockMain = `
static void reset(int caller, int target = 1) {
  current = caller; expected = target;
  get_calls = set_calls = work_calls = collective_calls = sync_calls = 0;
  get_error = fail_set_call = work_error = init_error = group_start_error = 0;
  fail_collective_call = group_end_error = fail_sync_call = 0;
  allocated = 0; allocation_devices.clear(); g_pool.clear(); g_live.clear();
  g_h2d_bytes = g_host_bytes = 0;
  g_world = 0; g_comms.clear();
}
static void *helper(int op, int target) {
  void *ptr = &storage[31];
  switch (op) {
    case 0: return fcuda_malloc_on(target, 16);
    case 1: fcuda_free_on(target, ptr); break;
    case 2: fcuda_h2d_on(target, ptr, ptr, 16); break;
    case 3: fcuda_d2h_on(target, ptr, ptr, 16); break;
    case 4: fcuda_d2d_on(target, ptr, ptr, 16); break;
  }
  return ptr;
}
static int run_collective(int op) {
  void *ptrs[2] = {&storage[0], &storage[1]};
  if (op == 0) return fcuda_nccl_allreduce_f32(ptrs, 2, 4);
  if (op == 1) return fcuda_nccl_allgather_f32(ptrs, ptrs, 2, 4);
  return fcuda_nccl_reducescatter_f32(ptrs, ptrs, 2, 4);
}
int main() {
  // This cold ordinary allocation is the original contamination regression.
  reset(0);
  require(fcuda_malloc_on(1, 16) != nullptr, "rank allocation failed");
  require(current == 0, "rank allocation contaminated the caller device");
  expected = 0;
  void *ordinary = fcuda_malloc(19);
  require(ordinary && allocation_devices[ordinary] == 0, "ordinary allocation escaped device 0");

  for (int caller : {0, 1, 2}) {
    for (int op = 0; op < 5; ++op) {
      for (int fault = 0; fault < 5; ++fault) {
        reset(caller);
        if (fault == 1) get_error = 71;
        if (fault == 2) fail_set_call = 1;
        if (fault == 3) work_error = 73;
        if (fault == 4) fail_set_call = 2;
        void *ptr = helper(op, 1);
        require(get_calls == 1, "helper did not capture its caller device");
        require(set_calls == (fault == 1 ? 0 : 2), "helper omitted selection/restoration");
        require(work_calls == (fault == 1 || fault == 2 ? 0 : 1), "helper ignored capture/selection failure");
        require(current == (fault == 4 ? 1 : caller), "helper failed to preserve exact caller device");
        if (op == 0) require((ptr != nullptr) == (fault == 0 || fault == 4), "allocation return lost its ownership/error");
        const size_t copied = fault == 0 || fault == 4 ? 16 : 0;
        require(g_h2d_bytes == (op == 2 ? copied : 0), "H2D witness changed on failure");
        require(g_host_bytes == (op == 3 ? copied : 0), "D2H witness changed on failure");
      }
    }
  }
  reset(0);
  fcuda_free_on(1, nullptr);
  require(get_calls == 0 && set_calls == 0 && work_calls == 0, "null free is not a no-op");
  require(fcuda_set_device(2) == 0 && current == 2, "explicit selector became temporary");
  reset(0);
  {
    FcudaDeviceScope outer;
    require(outer.status() == cudaSuccess && fcuda_set_device(1) == 0, "outer scope setup failed");
    expected = 0;
    require(fcuda_malloc_on(0, 16) && current == 1, "nested helper failed to restore outer device");
  }
  require(current == 0, "outer scope failed to restore its caller");

  for (int caller : {0, 2}) {
    for (int fault = 0; fault < 3; ++fault) {
      reset(caller);
      if (fault == 1) get_error = 71;
      if (fault == 2) init_error = 91;
      const int rc = fcuda_nccl_init(2);
      require(rc == (fault == 1 ? 1071 : fault == 2 ? 91 : 0), "NCCL init replaced primary result");
      require(current == caller, "NCCL initialization leaked its internal device");
      require(g_world == (fault == 0 ? 2 : 0), "NCCL init publication changed on failure");
    }
    for (int op = 0; op < 3; ++op) {
      for (int fault = 0; fault < 7; ++fault) {
        reset(caller); g_world = 2; g_comms = {0, 1};
        int want = 0;
        if (fault == 1) { get_error = 71; want = 1071; }
        if (fault == 2) { fail_set_call = 2; want = 1072; }
        if (fault == 3) { group_start_error = 91; want = 91; }
        if (fault == 4) { fail_collective_call = 2; want = 91; }
        if (fault == 5) { group_end_error = 91; want = 91; }
        if (fault == 6) { fail_sync_call = 2; want = 1073; }
        require(run_collective(op) == want, "NCCL collective replaced primary result");
        require(current == caller, "NCCL collective failed to restore caller on exit");
        if (fault == 1) require(set_calls == 0 && collective_calls == 0, "NCCL ignored failed capture");
        if (fault == 2) require(collective_calls == 1, "NCCL submitted a rank after selection failure");
      }
    }
  }
  // Restoration errors are diagnosed, not converted into lost ownership/state.
  reset(0); fail_set_call = 1;
  require(fcuda_nccl_init(2) == 0 && g_world == 2 && current == 1, "restore failure changed NCCL init ownership");
  reset(0); g_world = 2; g_comms = {0, 1}; fail_set_call = 5;
  require(run_collective(0) == 0 && current == 1, "restore failure replaced completed collective result");
  puts("CUDA helper and single-process NCCL device restoration control flow passed");
}
`
