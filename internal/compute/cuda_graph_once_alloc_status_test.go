package compute

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

// Exercise the real create body and allocation-status helper, not device work.
// fak-test:runtime medium est=1s lane=default
func TestCUDAGraphOnceCreateConsumesOnlyCapacityMiss(t *testing.T) {
	compiler, err := exec.LookPath("c++")
	if err != nil {
		t.Skip("host C++ compiler unavailable")
	}
	graph := readCUDASource(t, "cuda/graph_capture_once.cuh")
	begin, end := strings.Index(graph, "struct fak_graph_once {"), strings.Index(graph, "static fak_graph_once *g_graph_once_capture")
	if begin < 0 || end <= begin {
		t.Fatal("graph owner definition missing")
	}
	program := cudaGraphOnceAllocationHarness + "\n" + graph[begin:end] + "\n" +
		strings.TrimPrefix(readCUDASource(t, "cuda/alloc_status.h"), "#pragma once\n") +
		"\nextern \"C\" int fak_graph_once_create(int capacity, void **owner) " +
		cudaFunctionBody(t, graph, "fak_graph_once_create") + cudaGraphOnceAllocationMain
	dir := t.TempDir()
	cpp, binary := filepath.Join(dir, "graph-create.cpp"), filepath.Join(dir, "graph-create")
	if runtime.GOOS == "windows" {
		binary += ".exe"
	}
	if err := os.WriteFile(cpp, []byte(program), 0600); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if out, err := exec.CommandContext(ctx, compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", cpp, "-o", binary).CombinedOutput(); err != nil {
		t.Fatalf("compile real graph create: %v\n%s", err, out)
	}
	if out, err := exec.CommandContext(ctx, binary).CombinedOutput(); err != nil {
		t.Fatalf("real graph create allocation status: %v\n%s", err, out)
	}
}

const cudaGraphOnceAllocationHarness = `
#include <cassert>
#include <climits>
#include <initializer_list>
#include <cstdlib>
using cudaError_t = int;
using cudaGraph_t = void *;
using cudaGraphExec_t = void *;
static constexpr int cudaSuccess = 0, cudaErrorMemoryAllocation = 2;
static bool g_capture_open = false;
static int allocation_status = 0, last_error = 0, reads = 0, allocations = 0;
static bool v41_candidates_stream_uncertain() { return false; }
static cudaError_t cudaGetLastError() { ++reads; int e = last_error; last_error = 0; return e; }
static cudaError_t cudaMalloc(void **pointer, size_t bytes) {
 ++allocations;
 if (allocation_status) { *pointer = nullptr; return last_error = allocation_status; }
 *pointer = std::malloc(bytes); assert(*pointer); return cudaSuccess;
}
`

const cudaGraphOnceAllocationMain = `
int main() {
 for (int status : {cudaErrorMemoryAllocation, 700, cudaSuccess}) {
  allocation_status = status; last_error = reads = allocations = 0;
  void *owner = nullptr;
  int result = fak_graph_once_create(4, &owner);
  auto *state = static_cast<fak_graph_once *>(owner);
  assert(state && allocations == 1 && state->capacity == 4);
  assert(result == (status ? 2 : 0) && state->terminal == (status != 0));
  assert((state->position != nullptr) == (status == 0));
  assert(reads == (status == cudaErrorMemoryAllocation ? 1 : 0));
  assert(last_error == (status == cudaErrorMemoryAllocation ? 0 : status));
  // Fake context teardown only: no claim about real CUDA release/fences.
  std::free(state->position); std::free(state);
 }
 return 0;
}
`
