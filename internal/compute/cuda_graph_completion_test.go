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

// This executes the literal graph completion body with mocked CUDA calls. It
// proves status propagation and cleanup, not CUDA compilation or GPU execution.
// fak-test:runtime medium est=1s lane=default
func TestCUDAGraphCompletionPropagatesSyncFailure(t *testing.T) {
	t.Parallel()
	var compiler string
	for _, name := range []string{"c++", "g++", "clang++"} {
		if path, err := exec.LookPath(name); err == nil {
			compiler = path
			break
		}
	}
	if compiler == "" {
		t.Skip("host C++ compiler unavailable; CUDA graph completion witness not run")
	}
	source, err := os.ReadFile("cuda_kernels.cu")
	if err != nil {
		t.Fatal(err)
	}
	text := string(source)
	start := strings.Index(text, `extern "C" int fcuda_graph_end_launch(`)
	if start < 0 {
		t.Fatal("CUDA graph completion definition not found")
	}
	end := strings.Index(text[start:], "\n// Persistent argmax index buffer")
	if end < 0 {
		t.Fatal("CUDA graph completion section boundary not found")
	}
	fixture := cudaGraphCompletionMockPrefix + text[start:start+end] + cudaGraphCompletionMockMain
	dir := t.TempDir()
	src, bin := filepath.Join(dir, "graph-completion.cpp"), filepath.Join(dir, "graph-completion")
	if err := os.WriteFile(src, []byte(fixture), 0600); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	if output, err := exec.CommandContext(ctx, compiler, "-std=c++11", "-Wall", "-Wextra", "-Werror", src, "-o", bin).CombinedOutput(); err != nil {
		t.Fatalf("compile literal CUDA graph completion host body: %v\n%s", err, output)
	}
	if output, err := exec.CommandContext(ctx, bin).CombinedOutput(); err != nil {
		t.Fatalf("CUDA graph completion witness: %v\n%s", err, output)
	}
}

const cudaGraphCompletionMockPrefix = `
#include <cstdio>
#include <cstdlib>
using cudaError_t = int;
using cudaStream_t = int;
using cudaGraph_t = void *;
using cudaGraphExec_t = void *;
const int cudaSuccess = 0;
struct cudaGraphExecUpdateResultInfo { int result; };
static const cudaStream_t g_stream = 7;
static cudaGraphExec_t g_exec = nullptr;
static bool g_capture_open = true;
static int graph_storage, exec_storage;
static int launch_status, sync_status, sync_calls, graph_destroys;
static void require(bool ok, const char *why) {
  if (!ok) { fprintf(stderr, "%s\n", why); exit(1); }
}
const char *cudaGetErrorString(cudaError_t) { return "injected CUDA failure"; }
cudaError_t cudaGetLastError() { return cudaSuccess; }
cudaError_t cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t *graph) {
  require(stream == g_stream, "capture ended on wrong stream");
  *graph = &graph_storage;
  return cudaSuccess;
}
cudaError_t cudaGraphInstantiate(cudaGraphExec_t *exec, cudaGraph_t graph, unsigned long long) {
  require(graph == &graph_storage, "instantiated wrong graph");
  *exec = &exec_storage;
  return cudaSuccess;
}
cudaError_t cudaGraphExecUpdate(cudaGraphExec_t, cudaGraph_t, cudaGraphExecUpdateResultInfo *info) {
  info->result = 0;
  return cudaSuccess;
}
cudaError_t cudaGraphExecDestroy(cudaGraphExec_t) { return cudaSuccess; }
cudaError_t cudaGraphLaunch(cudaGraphExec_t exec, cudaStream_t stream) {
  require(exec == &exec_storage && stream == g_stream, "launched wrong graph or stream");
  return launch_status;
}
cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
  require(stream == g_stream, "synchronized wrong stream");
  ++sync_calls;
  return sync_status;
}
cudaError_t cudaGraphDestroy(cudaGraph_t graph) {
  require(graph == &graph_storage, "destroyed wrong graph");
  require(sync_calls == 1, "captured graph destroyed before synchronization");
  ++graph_destroys;
  return cudaSuccess;
}
#define CK(call) do { int error = (call); if (error) fprintf(stderr, "mock CUDA error %d\n", error); } while (0)
`

const cudaGraphCompletionMockMain = `
static void run_case(int launch_error, int sync_error, int expected, const char *why) {
  g_exec = nullptr;
  g_capture_open = true;
  launch_status = launch_error;
  sync_status = sync_error;
  sync_calls = graph_destroys = 0;
  const int result = fcuda_graph_end_launch();
  require(result == expected, why);
  require(sync_calls == 1, "graph launch must drain its stream exactly once");
  require(graph_destroys == 1, "captured graph must be destroyed exactly once");
  require(!g_capture_open, "graph completion left capture open");
}
int main() {
  run_case(0, 719, 4, "asynchronous graph completion failure was reported as success");
  run_case(0, 0, 0, "successful graph completion was rejected");
  run_case(700, 0, 3, "graph launch failure status was lost");
  run_case(700, 719, 3, "completion failure replaced the earlier launch failure");
  puts("CUDA graph completion status and cleanup controls passed");
}
`
