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

// This executes the literal ownership, reset, begin, and completion block with
// mocked CUDA calls. It proves host control flow, not CUDA compilation, GPU
// execution, or whether a failed destroy released device resources.
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
	start := strings.Index(text, `static cudaGraphExec_t g_exec = nullptr;`)
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
	for _, scenario := range []string{
		"sync_failure", "success", "launch_failure", "launch_and_sync_failure",
		"reset_success", "update_success", "replace_success", "instantiate_failure", "replace_failure",
		"reset_retained", "reset_consumed", "replace_retained", "replace_consumed",
	} {
		t.Run(scenario, func(t *testing.T) {
			if output, err := exec.CommandContext(ctx, bin, scenario).CombinedOutput(); err != nil {
				t.Fatalf("CUDA graph ownership witness: %v\n%s", err, output)
			}
		})
	}
}

const cudaGraphCompletionMockPrefix = `
#include <cstdio>
#include <cstdlib>
#include <cstring>
using cudaError_t = int;
using cudaStream_t = int;
using cudaGraph_t = void *;
using cudaGraphExec_t = void *;
const int cudaSuccess = 0;
struct cudaGraphExecUpdateResultInfo { int result; };
static const cudaStream_t g_stream = 7;
static bool g_capture_open = false;
static int graph_storage, exec_storage, old_storage, poison_storage;
static int launch_status, sync_status, sync_calls, graph_destroys, launches;
static int destroy_status, destroy_calls, new_destroy_calls, instantiate_status, instantiates, updates, update_status, begins;
static bool consumed_on_error, old_live = true, old_destroy_attempted;
const int cudaStreamCaptureModeGlobal = 0;
static void require(bool ok, const char *why) {
  if (!ok) { fprintf(stderr, "%s\n", why); exit(1); }
}
const char *cudaGetErrorString(cudaError_t) { return "injected CUDA failure"; }
cudaError_t cudaGetLastError() { return cudaSuccess; }
cudaError_t cudaStreamBeginCapture(cudaStream_t stream, int) {
  require(stream == g_stream, "capture began on wrong stream");
  ++begins;
  return cudaSuccess;
}
cudaError_t cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t *graph) {
  require(stream == g_stream, "capture ended on wrong stream");
  *graph = &graph_storage;
  return cudaSuccess;
}
cudaError_t cudaGraphInstantiate(cudaGraphExec_t *exec, cudaGraph_t graph, unsigned long long) {
  require(graph == &graph_storage, "instantiated wrong graph");
  ++instantiates;
  if (instantiate_status) {
    *exec = &poison_storage; // A failed output must never become the published exec.
    return instantiate_status;
  }
  *exec = &exec_storage;
  return cudaSuccess;
}
cudaError_t cudaGraphExecUpdate(cudaGraphExec_t exec, cudaGraph_t graph, cudaGraphExecUpdateResultInfo *info) {
  require(exec == &old_storage && old_live && !old_destroy_attempted, "updated invalid or quarantined exec");
  require(graph == &graph_storage, "updated with wrong graph");
  ++updates;
  info->result = 0;
  return update_status;
}
cudaError_t cudaGraphExecDestroy(cudaGraphExec_t exec) {
  if (exec == &exec_storage) {
    require(new_destroy_calls++ == 0, "retried new executable destruction");
    return cudaSuccess;
  }
  require(exec == &old_storage && old_live && !old_destroy_attempted, "retried or destroyed invalid exec");
  old_destroy_attempted = true;
  ++destroy_calls;
  if (!destroy_status || consumed_on_error) old_live = false;
  return destroy_status;
}
cudaError_t cudaGraphLaunch(cudaGraphExec_t exec, cudaStream_t stream) {
  require((exec == &exec_storage || (exec == &old_storage && old_live && !old_destroy_attempted)) && stream == g_stream,
          "launched wrong or quarantined graph or stream");
  ++launches;
  return launch_status;
}
cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
  require(stream == g_stream, "synchronized wrong stream");
  ++sync_calls;
  return sync_status;
}
cudaError_t cudaGraphDestroy(cudaGraph_t graph) {
  require(graph == &graph_storage, "destroyed wrong graph");
  require(sync_calls == launches, "captured graph destroyed before launch synchronization");
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
int main(int argc, char **argv) {
  require(argc == 2, "missing scenario");
  const char *name = argv[1];
  if (!strcmp(name, "sync_failure")) run_case(0, 719, 4, "async failure reported as success");
  else if (!strcmp(name, "success")) run_case(0, 0, 0, "successful completion rejected");
  else if (!strcmp(name, "launch_failure")) run_case(700, 0, 3, "launch failure lost");
  else if (!strcmp(name, "launch_and_sync_failure")) run_case(700, 719, 3, "sync replaced primary launch failure");
  else if (!strcmp(name, "instantiate_failure")) {
    instantiate_status = 2;
    require(fcuda_graph_begin() == 0, "begin rejected");
    require(fcuda_graph_end_launch() == 2, "instantiate failure lost");
    require(g_exec == nullptr && instantiates == 1 && launches == 0 && sync_calls == 0 && graph_destroys == 1,
            "failed instantiate published candidate or missed cleanup");
  } else {
    g_exec = &old_storage;
    const bool reset = !strncmp(name, "reset_", 6);
    const bool quarantine = strstr(name, "retained") || strstr(name, "consumed");
    consumed_on_error = strstr(name, "consumed") != nullptr;
    destroy_status = quarantine ? 719 : 0;
    update_status = !strcmp(name, "update_success") ? 0 : 1;
    instantiate_status = !strcmp(name, "replace_failure") ? 2 : 0;
    if (reset) {
      fcuda_graph_reset();
      require(destroy_calls == 1, "reset did not destroy once");
    } else {
      require(fcuda_graph_begin() == 0, "initial begin rejected");
      const int expected = quarantine ? 5 : (instantiate_status ? 2 : 0);
      require(fcuda_graph_end_launch() == expected, "replacement status lost");
      require(graph_destroys == 1 && !g_capture_open, "completion missed captured graph cleanup");
      require(updates == 1, "old exec not updated exactly once");
    }
    if (quarantine) {
      require(g_exec == &old_storage, "destroy failure lost quarantined identity");
      require(old_live == !consumed_on_error, "mock destruction outcome mismatch");
      require(destroy_calls == 1 && instantiates == 0 && launches == 0 && sync_calls == 0,
              "destroy failure allowed replacement or launch");
      destroy_status = 0; // Later apparent recovery must not retry the uncertain handle.
      fcuda_graph_reset();
      fcuda_graph_reset();
      require(g_exec == &old_storage && destroy_calls == 1, "reset cleared or retried quarantine");
      const int prior_begins = begins;
      require(fcuda_graph_begin() != 0 && begins == prior_begins && !g_capture_open,
              "quarantine allowed BeginCapture");
      // Exercise end defensively even if a caller ignores begin's failure.
      const int prior_graph_destroys = graph_destroys;
      require(fcuda_graph_end_launch() == 5, "end ignored quarantine");
      require(graph_destroys == prior_graph_destroys + 1 && destroy_calls == 1 &&
              instantiates == 0 && launches == 0 && sync_calls == 0,
              "quarantined end touched executable or missed captured graph cleanup");
    } else if (reset) {
      require(g_exec == nullptr, "successful reset retained exec");
      fcuda_graph_reset();
      require(destroy_calls == 1, "empty reset retried destruction");
      require(fcuda_graph_begin() == 0, "successful reset blocked begin");
      fcuda_graph_abort();
      require(!g_capture_open && graph_destroys == 1, "abort missed cleanup");
    } else {
      require(instantiates == (update_status ? 1 : 0), "unexpected instantiate count");
      require(destroy_calls == (update_status ? 1 : 0), "unexpected destroy count");
      require(launches == (instantiate_status ? 0 : 1) && sync_calls == launches,
              "replacement launch/sync mismatch");
      require(g_exec == (instantiate_status ? nullptr : (update_status ? &exec_storage : &old_storage)),
              "wrong published exec after update/replacement");
      if (update_status && !instantiate_status) {
        fcuda_graph_reset();
        require(g_exec == nullptr && new_destroy_calls == 1, "replacement reset failed");
      }
    }
  }
  puts("CUDA graph ownership and completion control passed");
}
`
