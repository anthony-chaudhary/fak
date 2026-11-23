package compute

import (
	"bytes"
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// Executes the production header's admission and alias helpers on the host with
// stub CUDA queries. It does not compile a CUDA kernel or run a GPU.
// fak-test:runtime fast est=2s lane=default
func TestCUDAQwen35ResidualNormHostContract(t *testing.T) {
	t.Parallel()
	header, err := os.ReadFile("cuda/qwen35_residual_norm.cuh")
	if err != nil {
		t.Fatal(err)
	}
	text := string(bytes.ReplaceAll(header, []byte("\r\n"), []byte("\n")))
	for _, want := range []string{
		"if (rows <= 0 || n <= 0 || rows > INT_MAX / n || g_capture_open) return 0;",
		"if (rows <= 0 || n <= 0 || rows > INT_MAX / n || !(eps > 0.f) || !isfinite(eps)) return -1;",
		"capture != cudaStreamCaptureStatusNone", "if (device != 0) return 0;",
		"static int g_qwen35_residual_norm_max_rows = 0;",
		"attrs.type != cudaMemoryTypeDevice || attrs.device != 0",
		"if (!qwen35_residual_norm_extent(ptrs[i], bytes[i])) return -3;",
		"fcuda_rmsnorm_fused_residual_add_f32(X, Delta, W, X, Y, rows, n, eps);",
	} {
		if !strings.Contains(text, want) {
			t.Errorf("production source lost contract %q", want)
		}
	}
	if strings.Count(text, "cudaGetDeviceProperties(") != 1 {
		t.Error("device limits must be queried at exactly one site")
	}
	launch := strings.Index(text, "fcuda_rmsnorm_fused_residual_add_f32(X,")
	for _, gate := range []string{"qwen35_residual_norm_aliased((uintptr_t)X", "qwen35_residual_norm_extent(ptrs[i]", "fak_qwen35_residual_norm_capability(rows, n);"} {
		if at := strings.Index(text, gate); at < 0 || at > launch {
			t.Errorf("admission %q moved after launch", gate)
		}
	}
	wrapper, err := os.ReadFile("cuda_qwen35_residual_norm.go")
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{
		"if !cudaQwen35ResidualNormEnabled || c.capturing || rows <= 0 {",
		`os.Getenv("FAK_CUDA_QWEN35_RESIDUAL_NORM") == "1"`, "c.qwen35SequenceSyncLocked()",
	} {
		if !strings.Contains(string(wrapper), want) {
			t.Errorf("Go wrapper lost contract %q", want)
		}
	}
	compiler, err := exec.LookPath("c++")
	if err != nil {
		t.Skip("host C++ compiler unavailable; no admission execution witness")
	}
	dir := t.TempDir()
	source := filepath.Join(dir, "contract.cc")
	program := qwen35ResidualNormHostPreamble +
		qwen35PostprocessCFunction(t, text, `extern "C" int fak_qwen35_residual_norm_capability(int rows, int n)`) +
		qwen35PostprocessCFunction(t, text, "static bool qwen35_residual_norm_aliased(") +
		qwen35ResidualNormHostMain
	if err := os.WriteFile(source, []byte(program), 0600); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	binary := filepath.Join(dir, "contract")
	if output, err := exec.CommandContext(ctx, compiler, "-std=c++17", "-O2", source, "-o", binary).CombinedOutput(); err != nil {
		t.Fatalf("production helper compilation: %v\n%s", err, output)
	}
	if output, err := exec.CommandContext(ctx, binary).CombinedOutput(); err != nil {
		t.Fatalf("production helper execution: %v\n%s", err, output)
	}
}

const qwen35ResidualNormHostPreamble = `
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdio>
typedef int cudaError_t;
enum { cudaSuccess = 0 };
enum cudaStreamCaptureStatus { cudaStreamCaptureStatusNone = 0, cudaStreamCaptureStatusActive = 1 };
struct cudaDeviceProp { int maxThreadsPerBlock, maxThreadsDim[3], maxGridSize[3]; size_t sharedMemPerBlock; };
static bool g_capture_open = false;
static void *g_stream = nullptr;
static int t_device = 0, t_property_queries = 0, t_property_error = 0, t_grid = 1 << 20;
static cudaStreamCaptureStatus t_capture = cudaStreamCaptureStatusNone;
static cudaError_t cudaGetDevice(int *d) { *d = t_device; return cudaSuccess; }
static cudaError_t cudaStreamIsCapturing(void *, cudaStreamCaptureStatus *s) { *s = t_capture; return cudaSuccess; }
static cudaError_t cudaGetDeviceProperties(cudaDeviceProp *p, int device) {
  ++t_property_queries;
  if (device != 0) return 101;
  if (t_property_error) return t_property_error;
  *p = cudaDeviceProp{1024, {1024, 1024, 64}, {t_grid, 65535, 65535}, 48 * 1024};
  return cudaSuccess;
}
static int g_qwen35_residual_norm_max_rows = 0;
`

const qwen35ResidualNormHostMain = `
#define REQUIRE(cond) do { if (!(cond)) { std::printf("line %d: %s\n", __LINE__, #cond); return 1; } } while (0)
int main() {
  // A failed query is reported and never cached as a verdict.
  t_property_error = 3;
  REQUIRE(fak_qwen35_residual_norm_capability(1, 8) == -3);
  t_property_error = 0;
  REQUIRE(fak_qwen35_residual_norm_capability(1, 8) == 1);
  REQUIRE(fak_qwen35_residual_norm_capability(2, 5120) == 1);
  REQUIRE(fak_qwen35_residual_norm_capability(t_grid, 1) == 1);
  REQUIRE(fak_qwen35_residual_norm_capability(t_grid + 1, 1) == 0);
  REQUIRE(t_property_queries == 2);
  REQUIRE(fak_qwen35_residual_norm_capability(0, 8) == 0);
  REQUIRE(fak_qwen35_residual_norm_capability(1, 0) == 0);
  REQUIRE(fak_qwen35_residual_norm_capability(INT_MAX, 2) == 0);
  // The per-call refusals survive the cached limits.
  g_capture_open = true;
  REQUIRE(fak_qwen35_residual_norm_capability(1, 8) == 0);
  g_capture_open = false;
  t_capture = cudaStreamCaptureStatusActive;
  REQUIRE(fak_qwen35_residual_norm_capability(1, 8) == 0);
  t_capture = cudaStreamCaptureStatusNone;
  t_device = 1;
  REQUIRE(fak_qwen35_residual_norm_capability(1, 8) == 0);
  t_device = 0;
  REQUIRE(fak_qwen35_residual_norm_capability(1, 8) == 1);
  REQUIRE(t_property_queries == 2);

  // Operands X, Delta, W, Y. Only Delta/W, both read-only, may overlap.
  for (int rows = 1; rows <= 2; ++rows) {
    const size_t w = 64, panel = (size_t)rows * w;
    const uintptr_t base[4] = {0x10000, 0x20000, 0x30000, 0x40000};
    REQUIRE(!qwen35_residual_norm_aliased(base[0], base[1], base[2], base[3], panel, w));
    for (int a = 0; a < 4; ++a) {
      for (int b = a + 1; b < 4; ++b) {
        const size_t lenA = a == 2 ? w : panel, lenB = b == 2 ? w : panel;
        const uintptr_t at[3] = {base[a], base[a] + lenA - 1, base[a] - (lenB - 1)};
        for (int k = 0; k < 3; ++k) {
          uintptr_t p[4] = {base[0], base[1], base[2], base[3]};
          p[b] = at[k];
          const bool refused = qwen35_residual_norm_aliased(p[0], p[1], p[2], p[3], panel, w);
          REQUIRE(refused == !(a == 1 && b == 2));
        }
        uintptr_t p[4] = {base[0], base[1], base[2], base[3]};
        p[b] = base[a] + lenA; // Adjacent is not overlapping.
        REQUIRE(!qwen35_residual_norm_aliased(p[0], p[1], p[2], p[3], panel, w));
      }
    }
  }
  return 0;
}
`
