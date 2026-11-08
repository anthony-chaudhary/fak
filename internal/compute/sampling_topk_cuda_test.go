package compute

import (
	"context"
	"crypto/sha256"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

// fak-test:runtime fast est=1ms lane=default
func TestSamplingCUDAUnwiredStructuralContract(t *testing.T) {
	b, err := os.ReadFile("sampling_topk.cu")
	if err != nil {
		t.Fatal(err)
	}
	s := string(b)
	for _, fragment := range []string{
		"UNWIRED approximate f32 primitive family",
		"int badScratch[SAMPLING_TOPK_THREADS / 32]",
		"sampling_block_sum_int(bad, s.badScratch)",
		"histB[b] = (b + 1 < SAMPLING_TOPK_BINS) ? histA[b + 1] : 0",
		"s.massSuffix[b] = (b + 1 < SAMPLING_TOPK_BINS) ? s.binMass[b + 1] : 0.f",
		"const float before = (winChunk == 0) ? 0.f : s.chunkEnd[winChunk - 1]",
		"if (before + s.cdf[mid] > goal)",
		"const double target = product > 0.0 ? product : DBL_MIN",
		"vocab > INT_MAX - (SAMPLING_TOPK_THREADS - 1)",
	} {
		if !strings.Contains(s, fragment) {
			t.Errorf("missing structural contract %q", fragment)
		}
	}
	if strings.Count(s, "float* p = probs + (size_t)row * n;") != 3 {
		t.Fatal("top-k, nucleus and draw must each index their own row")
	}
	for _, name := range []string{"cuda_kernels.cu", "cuda_backend.h"} {
		body, err := os.ReadFile(name)
		if err != nil {
			t.Fatal(err)
		}
		if samplingCUDAApproximateWiring(string(body)) {
			t.Fatalf("%s activated the approximate sampler before an exact-law successor", name)
		}
	}
}

// samplingCUDAApproximateWiring fences the actual approximate family, not the
// independent compact candidate API that happens to share its naming prefix.
// Identifier matching is intentionally conservative: even a legacy identifier
// in a comment remains fenced, but a longer independent identifier is not one.
func samplingCUDAApproximateWiring(body string) bool {
	if strings.Contains(body, "sampling_topk.cu") {
		return true
	}
	for _, ident := range strings.FieldsFunc(body, func(r rune) bool {
		return !(r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || r >= '0' && r <= '9' || r == '_')
	}) {
		switch ident {
		case "fcuda_sampling_repetition_penalty_f32", "fcuda_sampling_topk_truncate_f32",
			"fcuda_sampling_nucleus_f32", "fcuda_sampling_draw_f32":
			return true
		}
	}
	return false
}

// fak-test:runtime fast est=1ms lane=default
func TestSamplingCUDAApproximateWiringFence(t *testing.T) {
	for _, symbol := range []string{
		"fcuda_sampling_repetition_penalty_f32",
		"fcuda_sampling_topk_truncate_f32",
		"fcuda_sampling_nucleus_f32",
		"fcuda_sampling_draw_f32",
	} {
		t.Run(symbol, func(t *testing.T) {
			for _, body := range []string{
				`extern "C" int ` + symbol + `(void);`,
				`return ` + symbol + `(p);`,
				`auto entry = &` + symbol + `;`,
			} {
				if !samplingCUDAApproximateWiring(body) {
					t.Fatalf("legacy export escaped the fence: %s", body)
				}
			}
		})
	}
	for _, body := range []string{
		`#include "sampling_topk.cu"`,
		`#include <sampling_topk.cu>`,
		`#include "./sampling_topk.cu"`,
	} {
		if !samplingCUDAApproximateWiring(body) {
			t.Fatalf("direct approximate fragment escaped the fence: %s", body)
		}
	}
	for _, body := range []string{
		`int fcuda_sampling_candidates(const uint32_t *row, int vocab, int k);`,
		`size_t fcuda_sampling_candidates_workspace_bytes(void);`,
		`int fcuda_sampling_candidates_capability(void);`,
		`int fcuda_sampling_candidates_allocation_status(void);`,
		`int fcuda_sampling_candidates_release(void *owned);`,
		`#include "cuda/sampling_candidates.cuh"`,
		`int independent_fcuda_sampling_draw_f32(void);`,
		`int fcuda_sampling_draw_f32_independent(void);`,
		"",
	} {
		if samplingCUDAApproximateWiring(body) {
			t.Fatalf("independent name/header was mistaken for approximate wiring: %s", body)
		}
	}
}

// The explicit owner-controlled opt-in compiles and runs the ACTUAL fragment.
// Default execution does not compile, allocate device memory or discover a GPU.
// This is structural arithmetic correctness, not the #14198 float64 seeded law.
// fak-test:runtime integration est=30s lane=optin
// fak-test:justify Explicit sm90 CUDA compilation and bounded multi-row device regression.
func TestSamplingCUDAUnwiredDevicePrimitives(t *testing.T) {
	if os.Getenv("FAK_CUDA_SAMPLING_PRIMITIVES_REQUIRED") != "1" {
		t.Skip("requires an admitted sm90 compiler/device window and explicit opt-in")
	}
	nvcc, err := exec.LookPath("nvcc")
	if err != nil {
		t.Fatal("required CUDA compiler unavailable: ", err)
	}
	path, err := filepath.Abs("sampling_topk.cu")
	if err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("native fragment sha256=%x; compiler=%s; compile=sm_90", sha256.Sum256(b), nvcc)
	source := samplingCUDADevicePreamble + "\n#include " + strconv.Quote(path) + "\n" + samplingCUDADeviceMain
	dir := t.TempDir()
	cu, binary := filepath.Join(dir, "sampling.cu"), filepath.Join(dir, "sampling")
	if err := os.WriteFile(cu, []byte(source), 0600); err != nil {
		t.Fatal(err)
	}
	build, cancelBuild := context.WithTimeout(t.Context(), 5*time.Minute)
	defer cancelBuild()
	if out, err := exec.CommandContext(build, nvcc, "-std=c++17", "-O2", "-arch=sm_90", "--fmad=false", cu, "-o", binary).CombinedOutput(); err != nil {
		t.Fatalf("required CUDA compilation failed: %v\n%s", err, out)
	}
	run, cancelRun := context.WithTimeout(t.Context(), 30*time.Second)
	defer cancelRun()
	out, err := exec.CommandContext(run, binary).CombinedOutput()
	if err != nil {
		t.Fatalf("required device witness failed: %v\n%s", err, out)
	}
	t.Logf("%s", out)
	if !strings.Contains(string(out), "sampling structural witness PASS") {
		t.Fatal("required device witness did not return its completion marker")
	}
}

const samplingCUDADevicePreamble = `
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
static cudaStream_t g_stream = nullptr;
#define FCUDA_TS(name, grid, ...) do { __VA_ARGS__; } while (0)
`

const samplingCUDADeviceMain = `
static void require(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}
static void checked(cudaError_t error, const char* action) {
  if (error != cudaSuccess) throw std::runtime_error(std::string(action) + ": " + cudaGetErrorString(error));
}
template<class T> static T* upload(const std::vector<T>& host) {
  T* device = nullptr;
  checked(cudaMalloc((void**)&device, host.size() * sizeof(T)), "allocate fixture");
  checked(cudaMemcpy(device, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice), "upload fixture");
  return device;
}
template<class T> static std::vector<T> download(T* device, size_t count) {
  std::vector<T> host(count);
  checked(cudaMemcpy(host.data(), device, count * sizeof(T), cudaMemcpyDeviceToHost), "download witness");
  return host;
}
static float sum(const std::vector<float>& values, int row, int n) {
  float result = 0.f;
  for (int i = 0; i < n; ++i) result += values[row * n + i];
  return result;
}
static std::vector<int> order(const std::vector<float>& values, int row, int n) {
  std::vector<int> ids(n);
  std::iota(ids.begin(), ids.end(), 0);
  std::sort(ids.begin(), ids.end(), [&](int a, int b) {
    const float x = values[row * n + a], y = values[row * n + b];
    return x != y ? x > y : a < b;
  });
  return ids;
}
// These small integer and binary-fraction fixtures have exact f32 partial sums.
// They isolate structural bugs without blessing approximate general-row parity.
static void truncation(bool nucleus, bool narrow) {
  const int n = 2053, rows = 5, k = 1027;
  const double topP = narrow ? std::numeric_limits<double>::denorm_min() : 0.5;
  std::vector<float> input(rows * n, 0.f), mass(rows, 1.f);
  for (int i = 0; i < n; ++i) {
    input[i] = narrow ? 1.f : float(i % 7 + 1);
    input[n + i] = narrow ? 0.f : float((n - i) % 9);
    input[2 * n + i] = 1.f;
  }
  input[n + 1500] = narrow ? 0.25f : 20.f;
  input[n + 12] = narrow ? 0.125f : 3.f;
  input[3 * n + 17] = std::numeric_limits<float>::quiet_NaN();
  input[4 * n + 31] = -1.f;
  for (int r = 0; r < 3; ++r) mass[r] = sum(input, r, n);
  auto expected = input;
  auto expectedMass = mass;
  for (int r = 0; r < 3; ++r) {
    const auto ids = order(input, r, n);
    std::vector<bool> keep(n, false);
    float cumulative = 0.f;
    for (int rank = 0; rank < n; ++rank) {
      if (nucleus) {
        if (rank > 0 && double(cumulative) >= topP * double(mass[r])) break;
      } else if (rank >= k) break;
      keep[ids[rank]] = true;
      cumulative += input[r * n + ids[rank]];
    }
    for (int i = 0; i < n; ++i) if (!keep[i]) expected[r * n + i] = 0.f;
    expectedMass[r] = sum(expected, r, n);
  }
  // Offset all output arrays by one; validate canaries in addition to every row.
  const float guard = -12345.f;
  auto guarded = input;
  guarded.insert(guarded.begin(), guard);
  guarded.push_back(guard);
  std::vector<int> status(rows + 2, -12345);
  auto guardedMass = mass;
  guardedMass.insert(guardedMass.begin(), guard);
  guardedMass.push_back(guard);
  float* p = upload(guarded);
  float* m = upload(guardedMass);
  int* st = upload(status);
  const int rc = nucleus
      ? fcuda_sampling_nucleus_f32(p + 1, st + 1, m + 1, rows, n, topP)
      : fcuda_sampling_topk_truncate_f32(p + 1, st + 1, m + 1, rows, n, k);
  require(rc == 0, "truncation submission failed");
  checked(cudaDeviceSynchronize(), "complete truncation");
  const auto actual = download(p, guarded.size());
  const auto actualMass = download(m, guardedMass.size());
  const auto actualStatus = download(st, status.size());
  require(actual.front() == guard && actual.back() == guard, "probability canary changed");
  require(actualMass.front() == guard && actualMass.back() == guard, "mass canary changed");
  require(actualStatus.front() == -12345 && actualStatus.back() == -12345, "status canary changed");
  require(std::memcmp(actual.data() + 1, expected.data(), expected.size() * sizeof(float)) == 0, "row or kept-set mismatch");
  for (int r = 0; r < rows; ++r) {
    require(actualStatus[r + 1] == (r < 3 ? SAMPLING_STATUS_OK : SAMPLING_STATUS_NON_FINITE), "row status mismatch");
    require(actualMass[r + 1] == expectedMass[r], "post-truncation mass mismatch");
  }
  checked(cudaFree(p), "retire probability fixture");
  checked(cudaFree(m), "retire mass fixture");
  checked(cudaFree(st), "retire status fixture");
}
static void draws() {
  const int n = 2053, rows = 5;
  std::vector<float> input(rows * n, 0.f);
  input[1500] = 1.f; input[2000] = 1.f;
  input[n + 2049] = 5.f;
  input[2 * n + 1024] = 1.f; input[2 * n + 2052] = 1.f;
  input[3 * n + 12] = 2.f; input[3 * n + 2048] = 2.f;
  std::vector<float> coins{0.5f, 0.f, std::nextafter(1.f, 0.f), 1.f, 0.f};
  const std::vector<int> expected{2000, 2049, 2052, 2048, n - 1};
  float* p = upload(input);
  float* c = upload(coins);
  int* token = upload(std::vector<int>(rows + 2, -12345));
  require(fcuda_sampling_draw_f32(p, c, token + 1, rows, n) == 0, "draw submission failed");
  checked(cudaDeviceSynchronize(), "complete draw");
  const auto actual = download(token, size_t(rows + 2));
  require(actual.front() == -12345 && actual.back() == -12345, "token canary changed");
  for (int r = 0; r < rows; ++r) require(actual[r + 1] == expected[r], "strict CDF/chunk/tail mismatch");
  checked(cudaFree(p), "retire draw probabilities");
  checked(cudaFree(c), "retire coins");
  checked(cudaFree(token), "retire tokens");
}
int main() {
  try {
    int count = 0;
    checked(cudaGetDeviceCount(&count), "discover required device");
    require(count > 0, "required device missing");
    cudaDeviceProp device{};
    checked(cudaGetDeviceProperties(&device, 0), "identify device");
    require(device.major == 9 && device.minor == 0, "required sm90 device missing");
    checked(cudaSetDevice(0), "select admitted visible device");
    std::cout << "device=" << device.name << " sm=" << device.major << device.minor << " SMs=" << device.multiProcessorCount << " uuid=";
    const char* hex = "0123456789abcdef";
    for (unsigned char value : device.uuid.bytes) std::cout << hex[value >> 4] << hex[value & 15];
    int driver = 0, runtime = 0;
    checked(cudaDriverGetVersion(&driver), "driver version");
    checked(cudaRuntimeGetVersion(&runtime), "runtime version");
    std::cout << " driver=" << driver << " runtime=" << runtime << '\n';
    truncation(false, false);
    truncation(true, false);
    truncation(true, true);
    draws();
    std::cout << "sampling structural witness PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
`
