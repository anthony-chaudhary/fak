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
// calls. It proves admission and ordering, not real kernel loading, CUDA syntax,
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
		t.Fatalf("CUDA init loader-admission witness: %v\n%s", err, output)
	}
}

const cudaInitProbeMockPrefix = `
#include <cstdio>
#include <cstring>
#include <cstdlib>
using cudaError_t = int;
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
static int storage;
void k_copyrow(float *, const float *, int, int) {}
static void require(bool ok, const char *why) {
  if (!ok) { fprintf(stderr, "%s\n", why); exit(1); }
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
  require(resources == 0, "kernel probe happened after creating owned resources");
  ++probes; return probe_status;
}
const char *cudaGetErrorString(int) { return "injected loader error"; }
int cublasCreate(cublasHandle_t *h) { ++resources; *h = &storage; return 0; }
int cublasSetPointerMode(cublasHandle_t, int) { return 0; }
int cudaStreamCreate(cudaStream_t *s) { ++resources; *s = 1; return 0; }
int cublasSetStream(cublasHandle_t, cudaStream_t) { return 0; }
int cudaMalloc(void **p, size_t) { ++resources; *p = &storage; return 0; }
int cublasSetWorkspace(cublasHandle_t, void *, size_t) { return 0; }
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
  result = fcuda_init(name, sizeof(name), &sm, &total);
  require(result == 0 && probes == 2, "successful kernel probe must retain normal initialization");
  require(resources == 3, "successful initialization must create cuBLAS, stream and workspace");
  require(sm == 100 && total == 1024 && strcmp(name, "mock Blackwell") == 0, "successful initialization lost device metadata");
  puts("CUDA init loader rejection and success control flow passed");
}
`
