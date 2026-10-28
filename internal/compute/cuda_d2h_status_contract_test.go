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

// Compile the actual transfer bodies with injected CUDA runtime results. This
// witnesses host status/accounting control flow, not CUDA linkage or execution.
// fak-test:runtime medium est=3s lane=default
func TestCUDAD2HStatusContract(t *testing.T) {
	read := func(path string) string {
		t.Helper()
		b, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		return string(b)
	}
	header := read("cuda_backend.h")
	memory, services := read("cuda/runtime_memory.cuh"), read("cuda/runtime_services.cuh")
	body := func(source, signature string) string {
		t.Helper()
		start := strings.Index(source, signature)
		if start < 0 {
			t.Fatalf("missing production signature %q", signature)
		}
		depth := 0
		for i := start; i < len(source); i++ {
			switch source[i] {
			case '{':
				depth++
			case '}':
				depth--
				if depth == 0 {
					return source[start : i+1]
				}
			}
		}
		t.Fatalf("unclosed production body %q", signature)
		return ""
	}
	definitions := body(services, `extern "C" int fcuda_set_device(`) + "\n" +
		`#include "device_scope.h"` + "\n" +
		body(memory, `extern "C" int fcuda_d2h_checked(`) + "\n" +
		body(memory, `extern "C" void fcuda_d2h(`) + "\n" +
		body(services, `extern "C" int fcuda_d2h_on_checked(`) + "\n" +
		body(services, `extern "C" void fcuda_d2h_on(`)
	if !strings.Contains(body(memory, `extern "C" void fcuda_d2h(`), "(void)fcuda_d2h_checked(h, d, n);") ||
		!strings.Contains(body(services, `extern "C" void fcuda_d2h_on(`), "(void)fcuda_d2h_on_checked(device, h, d, n);") {
		t.Fatal("legacy D2H wrappers must delegate to the checked implementations")
	}
	compiler, err := exec.LookPath("c++")
	if err != nil {
		t.Skip("host C++ compiler unavailable; D2H status behavior remains unverified")
	}
	dir := t.TempDir()
	for name, source := range map[string]string{
		"cuda_backend.h": header, "device_scope.h": read("cuda/device_scope.h"),
		"d2h.cpp": `
#include <cassert>
#include <cstdio>
#include <cstring>
#include "cuda_backend.h"
static int (*checked)(void *, const void *, size_t) = &fcuda_d2h_checked;
static int (*checked_on)(int, void *, const void *, size_t) = &fcuda_d2h_on_checked;
using cudaError_t = int;
static constexpr int cudaSuccess = 0, cudaMemcpyDeviceToHost = 2;
static int selected = 0, expected = 0, get_error = 0, select_error = 0, copy_error = 0;
static int copies = 0, untimed = 0;
static size_t g_host_bytes = 0;
static cudaError_t cudaGetDevice(int *out) {
  if (get_error) return get_error;
  *out = selected;
  return 0;
}
static cudaError_t cudaSetDevice(int device) {
  selected = device;
  return device == 1 ? select_error : 0;
}
static const char *cudaGetErrorString(cudaError_t) { return "injected runtime error"; }
static cudaError_t cudaMemcpy(void *out, const void *in, size_t n, int kind) {
  assert(kind == cudaMemcpyDeviceToHost && selected == expected);
  ++copies;
  if (copy_error) return copy_error;
  memcpy(out, in, n);
  return 0;
}
static void fcuda_ts_untimed_transfer() { ++untimed; }
` + definitions + `
int main() {
  const int input = 42;
  int output = 0;
  const size_t bytes = sizeof(input);
  assert(checked(&output, &input, bytes) == 0);
  assert(output == input && copies == 1 && untimed == 1 && g_host_bytes == bytes);
  copy_error = 7;
  assert(fcuda_d2h_checked(&output, &input, bytes) == 1007);
  assert(copies == 2 && g_host_bytes == bytes);
  copy_error = 0;
  expected = 1;
  assert(checked_on(1, &output, &input, bytes) == 0);
  assert(selected == 0 && copies == 3 && g_host_bytes == 2 * bytes);
  copy_error = 7;
  assert(fcuda_d2h_on_checked(1, &output, &input, bytes) == 1007);
  assert(selected == 0 && copies == 4 && g_host_bytes == 2 * bytes);
  copy_error = 0;
  select_error = 9;
  assert(fcuda_d2h_on_checked(1, &output, &input, bytes) == 1009);
  assert(selected == 0 && copies == 4 && untimed == 4 && g_host_bytes == 2 * bytes);
  select_error = 0;
  get_error = 5;
  assert(fcuda_d2h_on_checked(1, &output, &input, bytes) == 1005);
  assert(selected == 0 && copies == 4 && untimed == 4 && g_host_bytes == 2 * bytes);
  get_error = 0;
  expected = 0;
  fcuda_d2h(&output, &input, bytes);
  expected = 1;
  fcuda_d2h_on(1, &output, &input, bytes);
  assert(selected == 0 && copies == 6 && untimed == 6 && g_host_bytes == 4 * bytes);
  copy_error = 7;
  expected = 0;
  fcuda_d2h(&output, &input, bytes);
  expected = 1;
  fcuda_d2h_on(1, &output, &input, bytes);
  assert(selected == 0 && copies == 8 && untimed == 8 && g_host_bytes == 4 * bytes);
  select_error = 9;
  fcuda_d2h_on(1, &output, &input, bytes);
  assert(selected == 0 && copies == 8 && untimed == 8 && g_host_bytes == 4 * bytes);
}
`,
	} {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(source), 0600); err != nil {
			t.Fatal(err)
		}
	}
	name := "d2h-status"
	if runtime.GOOS == "windows" {
		name += ".exe"
	}
	binary := filepath.Join(dir, name)
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	args := []string{"-std=c++17", "-Wall", "-Wextra", "-Werror", "-I", dir, filepath.Join(dir, "d2h.cpp"), "-o", binary}
	if out, err := exec.CommandContext(ctx, compiler, args...).CombinedOutput(); err != nil {
		t.Fatalf("compile production checked D2H bodies: %v\n%s", err, out)
	}
	if out, err := exec.CommandContext(ctx, binary).CombinedOutput(); err != nil {
		t.Fatalf("production checked D2H contract: %v\n%s", err, out)
	}
}
