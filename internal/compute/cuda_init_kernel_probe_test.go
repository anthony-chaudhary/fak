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

// This executes the literal host initialization body with mocked CUDA/cuBLAS
// calls. It proves admission, ordering and resource cleanup, not real kernel loading, CUDA syntax,
// native-versus-PTX classification, all-kernel compatibility, or GPU execution.
// fak-test:runtime medium est=1s lane=default
func TestCUDAInitRejectsKernelLoadFailure(t *testing.T) {
	t.Parallel()
	var compiler string
	for _, name := range []string{"c++", "g++", "clang++"} {
		if path, err := exec.LookPath(name); err == nil {
			compiler = path
			break
		}
	}
	if compiler == "" {
		t.Skip("host C++ compiler unavailable; CUDA init control-flow witness not run")
	}
	source, err := os.ReadFile("cuda_kernels.cu")
	if err != nil {
		t.Fatal(err)
	}
	text := string(source)
	start := strings.Index(text, `extern "C" int fcuda_init(`)
	if start < 0 {
		t.Fatal("CUDA initialization definition not found")
	}
	end := strings.Index(text[start:], "\n// fcuda_set_tf32")
	if end < 0 {
		t.Fatal("CUDA initialization section boundary not found")
	}
	fixture := cudaInitProbeMockPrefix + text[start:start+end] + cudaInitProbeMockMain
	dir := t.TempDir()
	src, bin := filepath.Join(dir, "init.cpp"), filepath.Join(dir, "init-probe")
	if err := os.WriteFile(src, []byte(fixture), 0600); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	if output, err := exec.CommandContext(ctx, compiler, "-std=c++11", "-Wall", "-Wextra", "-Werror", src, "-o", bin).CombinedOutput(); err != nil {
		t.Fatalf("compile literal CUDA init host body: %v\n%s", err, output)
	}
	if output, err := exec.CommandContext(ctx, bin).CombinedOutput(); err != nil {
		t.Fatalf("CUDA init admission/cleanup witness: %v\n%s", err, output)
	}
}

const cudaInitProbeMockPrefix = `
#include <cstdio>
#include <cstring>
#include <cstdlib>
using cudaError_t = int;
using cublasStatus_t = int;
using cublasHandle_t = void *;
using cudaStream_t = int;
const int cudaSuccess = 0;
const int CUBLAS_STATUS_SUCCESS = 0;
const int CUBLAS_POINTER_MODE_HOST = 0;
struct cudaDeviceProp { char name[256]; int major; int minor; size_t totalGlobalMem; };
struct cudaFuncAttributes {};
static cublasHandle_t g_blas = nullptr;
static void *g_blas_workspace = nullptr;
static cudaStream_t g_stream = 0;
static int selected_device = -1, property_reads = 0, probes = 0, resources = 0;
static int probe_status = 209;
enum { CREATE_BLAS = 1, POINTER_MODE, CREATE_STREAM, BIND_STREAM, ALLOC_WORKSPACE, BIND_WORKSPACE };
static int fail_at = 0, setup_calls = 0, blas_destroys = 0, stream_destroys = 0, workspace_frees = 0;
static bool live_blas = false, live_stream = false, live_workspace = false;
static int blas_storage, workspace_storage;
static void *bound_workspace = nullptr;
void k_copyrow(float *, const float *, int, int) {}
static void require(bool ok, const char *why) {
  if (!ok) { fprintf(stderr, "failure stage %d: %s\n", fail_at, why); exit(1); }
}
static int setup(int stage) {
  require(stage == ++setup_calls, "resource setup continued after failure or ran out of order");
  return fail_at == stage ? 70 + stage : 0;
}
int cudaGetDeviceCount(int *n) { *n = 1; return 0; }
int cudaSetDevice(int device) { selected_device = device; return 0; }
int cudaGetDeviceProperties(cudaDeviceProp *p, int device) {
  require(selected_device == device && device == 0, "properties queried before selecting device 0");
  ++property_reads;
  strcpy(p->name, "mock Blackwell"); p->major = 10; p->minor = 0; p->totalGlobalMem = 1024;
  return 0;
}
int cudaFuncGetAttributes(cudaFuncAttributes *, void (*kernel)(float *, const float *, int, int)) {
  require(selected_device == 0 && property_reads > 0, "kernel probe lacks selected device context");
  require(kernel == k_copyrow, "kernel probe must reference the real copy-row symbol");
  require(setup_calls == 0 || resources == 3, "kernel probe happened during partial resource setup");
  ++probes; return probe_status;
}
const char *cudaGetErrorString(int) { return "injected CUDA error"; }
int cublasCreate(cublasHandle_t *h) {
  int status = setup(CREATE_BLAS);
  if (!status) { live_blas = true; ++resources; *h = &blas_storage; }
  return status;
}
int cublasSetPointerMode(cublasHandle_t h, int mode) {
  require(live_blas && h == &blas_storage && mode == CUBLAS_POINTER_MODE_HOST, "invalid pointer-mode target");
  return setup(POINTER_MODE);
}
int cudaStreamCreate(cudaStream_t *s) {
  int status = setup(CREATE_STREAM);
  if (!status) { live_stream = true; ++resources; *s = 1; }
  return status;
}
int cublasSetStream(cublasHandle_t h, cudaStream_t s) {
  require(live_blas && live_stream && h == &blas_storage && s == 1, "invalid stream binding");
  int status = setup(BIND_STREAM);
  if (!status) bound_workspace = nullptr; // cuBLAS resets the workspace on every stream bind.
  return status;
}
int cudaMalloc(void **p, size_t n) {
  require(n == (size_t)32 * 1024 * 1024, "unexpected workspace size");
  int status = setup(ALLOC_WORKSPACE);
  if (!status) { live_workspace = true; ++resources; *p = &workspace_storage; }
  return status;
}
int cublasSetWorkspace(cublasHandle_t h, void *p, size_t n) {
  require(live_blas && live_workspace && h == &blas_storage && p == &workspace_storage &&
          n == (size_t)32 * 1024 * 1024, "invalid workspace binding");
  int status = setup(BIND_WORKSPACE);
  if (!status) bound_workspace = p;
  return status;
}
int cublasDestroy(cublasHandle_t h) {
  require(live_blas && h == &blas_storage, "destroyed an unowned cuBLAS handle");
  live_blas = false; --resources; ++blas_destroys; bound_workspace = nullptr; return 0;
}
int cudaFree(void *p) {
  require(!live_blas && live_workspace && p == &workspace_storage, "workspace freed before handle or without ownership");
  live_workspace = false; --resources; ++workspace_frees; return 0;
}
int cudaStreamDestroy(cudaStream_t s) {
  require(!live_blas && live_stream && s == 1, "stream destroyed before handle or without ownership");
  live_stream = false; --resources; ++stream_destroys; return 0;
}
#define CK(call) do { int error = (call); if (error) fprintf(stderr, "mock CUDA error %d\n", error); } while (0)
`

const cudaInitProbeMockMain = `
int main() {
  char name[256] = {};
  int sm = 0; size_t total = 0;
  int result = fcuda_init(name, sizeof(name), &sm, &total);
  require(result != 0, "kernel loader failure was admitted as a usable CUDA backend");
  require(probes == 1 && resources == 0, "loader failure must precede cuBLAS, stream and workspace creation");
  require(g_blas == nullptr && g_stream == 0 && g_blas_workspace == nullptr, "loader rejection created owned resources");
  probe_status = 0;
  for (fail_at = CREATE_BLAS; fail_at <= BIND_WORKSPACE; ++fail_at) {
    setup_calls = blas_destroys = stream_destroys = workspace_frees = 0;
    result = fcuda_init(name, sizeof(name), &sm, &total);
    int expected = fail_at == CREATE_STREAM || fail_at == ALLOC_WORKSPACE ? 6 : 4;
    require(result == expected, "resource setup failure was admitted or lost its init status");
    require(setup_calls == fail_at, "setup did not stop at the first failure");
    require(resources == 0 && g_blas == nullptr && g_stream == 0 && g_blas_workspace == nullptr,
            "failed setup leaked or published partial resources");
    require(blas_destroys == (fail_at > CREATE_BLAS) && stream_destroys == (fail_at > CREATE_STREAM) &&
            workspace_frees == (fail_at > ALLOC_WORKSPACE), "cleanup did not release exactly the acquired resources");
  }
  fail_at = 0;
  setup_calls = blas_destroys = stream_destroys = workspace_frees = 0;
  result = fcuda_init(name, sizeof(name), &sm, &total);
  require(result == 0 && probes == 8, "successful retry must retain kernel admission");
  require(resources == 3, "successful initialization must create cuBLAS, stream and workspace");
  require(g_blas == &blas_storage && g_stream == 1 && g_blas_workspace == &workspace_storage &&
          bound_workspace == g_blas_workspace, "success did not publish fully configured resources");
  require(sm == 100 && total == 1024 && strcmp(name, "mock Blackwell") == 0, "successful initialization lost device metadata");
  result = fcuda_init(nullptr, 0, nullptr, nullptr);
  require(result == 0 && probes == 9 && setup_calls == BIND_WORKSPACE && resources == 3 &&
          bound_workspace == g_blas_workspace, "repeated initialization changed established resources");
  probe_status = 209;
  result = fcuda_init(name, sizeof(name), &sm, &total);
  require(result == 5 && probes == 10 && resources == 3 && g_blas == &blas_storage &&
          g_stream == 1 && g_blas_workspace == &workspace_storage && bound_workspace == g_blas_workspace,
          "loader rejection destroyed previously established resources");
  require(blas_destroys == 0 && stream_destroys == 0 && workspace_frees == 0, "successful resources were cleaned up");
  puts("CUDA init loader admission, resource failure cleanup, retry and idempotence passed");
}
`
