package compute

import (
	"context"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// q8_panel_prefill_route_test.go pins the routing decision for P>=128 Q8_0 prefill GEMMs
// (oss-port-cuda-perf-gdn-q8-dequant-retire): the dequant -> fp16 split + 3x cuBLAS arm no
// longer serves these panels. In-contract shapes (block 32, 16-byte aligned codes, bounded
// output groups) run the rowtile panel GEMM, out-of-contract alignment/block shapes stay on
// the plain per-row BitExact GEMM, and only an argument-bounds refusal (P > 65535 or an int
// overflow guard) falls back to the retired dequant arm. TestQ8PanelPrefillRoute proves the
// decision on any host (source contract + compiled production slice), including the GDN
// sequence-spine caller qwen35_gdn_q8_prefill - the live serve path's production gate for the
// measured GDN/attn-qk leaf (wire-q8-panel-prefill-route); TestGDNProjectionParity pins the
// numerics of the new lane against the f32 reference on device, fail-closed without one.

// gdnPrefillShapes mirrors the Qwen3.8 hybrid projection geometry the retired arm served at
// P>=128: GDN in/out projections (nK=16, nV=48, kHd=vHd=128, hidden=5120) and the full-attention
// q/k projections (24 Q / 4 KV heads, head dim 256).
var gdnPrefillShapes = []struct {
	name    string
	out, in int
}{
	{"gdn-in-proj-qkv", 10240, 5120}, // convDim x hidden
	{"gdn-in-proj-z", 6144, 5120},    // valueDim x hidden
	{"gdn-in-proj-ba", 48, 5120},     // nV x hidden
	{"gdn-out-proj", 5120, 6144},     // hidden x valueDim
	{"attn-q-proj", 6144, 5120},      // 24*256 x hidden
	{"attn-k-proj", 1024, 5120},      // 4*256 x hidden
}

// q8TileGroups mirrors kQ8TileWarps*kQ8TileOuts (cuda/quant_matmul.cuh): one output-group
// count per rowtile grid column.
const q8TileGroups = 32

// q8PrefillLane reports which lane fcuda_q8_matmul_f32 serves a P-row Q8_0 GEMM: "decode"
// (per-row GEMM, P<128), "rowtile" (the panel GEMM, in-contract P>=128), "perrow" (P>=128
// outside the rowtile contract: block, alignment or output-group bounds) or "legacy" (the
// argument-bounds refusal that keeps the retired dequant arm as last resort). It mirrors the
// admission chain of fcuda_q8_matmul_f32 / fcuda_q8_matmul_rows_f32 exactly; null-pointer
// refusals have no Go-side analogue because every call site validates its tensors first.
func q8PrefillLane(out, in, p, block int, aligned16 bool) string {
	if p < 128 {
		return "decode"
	}
	if out <= 0 || in <= 0 || p <= 0 || block <= 0 || in%block != 0 ||
		p > 65535 || p > int(math.MaxInt32)/in {
		return "legacy"
	}
	if block != 32 || !aligned16 || (out+q8TileGroups-1)/q8TileGroups > 65535 {
		return "perrow"
	}
	return "rowtile"
}

// gdnSpinePrefillBody extracts the complete qwen35_gdn_q8_prefill definition from the GDN
// sequence spine: the live serve path's production gate for eligible P>=128 Q8_0 GDN panels,
// whose selected arm this file pins as a closed contract.
func gdnSpinePrefillBody(t *testing.T, src string) string {
	t.Helper()
	start := strings.Index(src, "static int qwen35_gdn_q8_prefill(")
	if start < 0 {
		t.Fatal("qwen35_gdn_q8_prefill missing from cuda/qwen35_gdn.cuh")
	}
	open := start + strings.Index(src[start:], "{")
	if open <= start {
		t.Fatal("qwen35_gdn_q8_prefill signature unterminated")
	}
	depth := 0
	for i := open; i < len(src); i++ {
		switch src[i] {
		case '{':
			depth++
		case '}':
			depth--
			if depth == 0 {
				return src[start : i+1]
			}
		}
	}
	t.Fatal("qwen35_gdn_q8_prefill body incomplete")
	return ""
}

// TestQ8PanelPrefillRoute pins the closed routing contract for P>=128 Q8_0 prefill: which arm
// serves the GDN in/out projections and the full-attention q/k projections, and which shapes
// fall back where. Every assertion runs on any host; the compiled witness dominates the cost
// (one host C++ compile of the production slice).
// fak-test:runtime medium est=45s lane=default
func TestQ8PanelPrefillRoute(t *testing.T) {
	t.Run("entry-point-routes-prefill-to-panel-lane", func(t *testing.T) {
		b, err := os.ReadFile(filepath.Join("cuda", "quant_matmul.cuh"))
		if err != nil {
			t.Fatal(err)
		}
		src := string(b)
		start := strings.Index(src, `extern "C" int fcuda_q8_matmul_f32(`)
		end := strings.Index(src[start:], "\n// getScaleMinK4_dev reproduces")
		if start < 0 || end <= 0 {
			t.Fatal("fcuda_q8_matmul_f32 boundaries missing from cuda/quant_matmul.cuh")
		}
		body := src[start : start+end]
		panel := strings.Index(body, "fcuda_q8_matmul_rows_f32(")
		legacy := strings.Index(body, "q8_dequant_sgemm(")
		if panel < 0 {
			t.Fatal("P>=128 arm lost the rowtile panel lane")
		}
		if legacy < 0 {
			t.Fatal("retired dequant arm lost its out-of-contract fallback reference")
		}
		if legacy < panel {
			t.Fatal("dequant arm precedes the panel attempt: P>=128 still served by the retired path")
		}
		if strings.Contains(body, "if (P >= 128) return q8_dequant_sgemm(") {
			t.Fatal("direct P>=128 dequant branch resurrected")
		}
	})

	t.Run("gdn-sequence-spine-delegates-to-routed-entry", func(t *testing.T) {
		b, err := os.ReadFile(filepath.Join("cuda", "qwen35_gdn.cuh"))
		if err != nil {
			t.Fatal(err)
		}
		body := gdnSpinePrefillBody(t, string(b))
		if !strings.Contains(body, "fcuda_q8_matmul_f32(") {
			t.Fatal("GDN sequence spine lost its delegation to the routed Q8 entry")
		}
		if strings.Contains(body, "q8_dequant_sgemm(") {
			t.Fatal("GDN sequence spine still serves the retired dequant arm directly")
		}
	})

	t.Run("scratch-pricing-follows-the-retired-arm", func(t *testing.T) {
		b, err := os.ReadFile("cuda_prompt_panel.go")
		if err != nil {
			t.Fatal(err)
		}
		src := string(b)
		start := strings.Index(src, "func cudaQ8MatMulScratch(")
		end := strings.Index(src[start:], "\nfunc ")
		if start < 0 || end < 0 {
			t.Fatal("cudaQ8MatMulScratch boundaries missing from cuda_prompt_panel.go")
		}
		body := src[start : start+end]
		if !strings.Contains(body, "return cudaQ8RowsScratch(rows, in, block)") {
			t.Fatal("Q8 scratch pricing no longer prices the per-row activation lane")
		}
		if strings.Contains(body, "cudaDequantScratch") {
			t.Fatal("Q8 scratch pricing still prices the retired dequant + fp16 image")
		}
	})

	t.Run("gdn-and-attn-qk-shapes-admit-to-rowtile", func(t *testing.T) {
		for _, sh := range gdnPrefillShapes {
			for _, p := range []int{128, 512} {
				if got := q8PrefillLane(sh.out, sh.in, p, 32, true); got != "rowtile" {
					t.Errorf("%s out=%d in=%d P=%d: lane %q, want the rowtile panel GEMM", sh.name, sh.out, sh.in, p, got)
				}
			}
		}
		sh := gdnPrefillShapes[0]
		if got := q8PrefillLane(sh.out, sh.in, 127, 32, true); got != "decode" {
			t.Errorf("P=127: lane %q, want the unchanged decode/small-panel lane", got)
		}
		if got := q8PrefillLane(sh.out, sh.in, 70000, 32, true); got != "legacy" {
			t.Errorf("P=70000: lane %q, want the out-of-contract legacy fallback", got)
		}
		if got := q8PrefillLane(sh.out, sh.in, 512, 16, true); got != "perrow" {
			t.Errorf("block=16: lane %q, want the plain per-row GEMM", got)
		}
		if got := q8PrefillLane(sh.out, sh.in, 512, 32, false); got != "perrow" {
			t.Errorf("misaligned codes: lane %q, want the plain per-row GEMM", got)
		}
	})

	t.Run("compiled-routing-witness", func(t *testing.T) {
		compiler, err := exec.LookPath("c++")
		if err != nil {
			t.Skip("host C++ compiler unavailable")
		}
		read := func(name string) string {
			t.Helper()
			b, err := os.ReadFile(name)
			if err != nil {
				t.Fatal(err)
			}
			return string(b)
		}
		quant := read(filepath.Join("cuda", "quant_matmul.cuh"))
		start := strings.Index(quant, "static signed char *g_q8_qX = nullptr;")
		end := strings.Index(quant, "// getScaleMinK4_dev reproduces")
		if start < 0 || end <= start {
			t.Fatal("production Q8 ownership boundaries missing")
		}
		constants := ""
		for _, name := range []string{"kQ8TileRows", "kQ8TileOuts", "kQ8TileWarps"} {
			begin := strings.Index(quant, "constexpr int "+name+" = ")
			if begin < 0 {
				t.Fatalf("production tile constant %s missing", name)
			}
			last := strings.Index(quant[begin:], ";")
			if last < 0 {
				t.Fatal("unterminated tile constant")
			}
			constants += quant[begin:begin+last+1] + "\n"
		}
		spine := gdnSpinePrefillBody(t, read(filepath.Join("cuda", "qwen35_gdn.cuh")))
		program := strings.TrimPrefix(read(filepath.Join("cuda", "alloc_ledger.h")), "#pragma once\n") + "\n" +
			q8RouteHarness + "\n" + constants + quant[start:end] + "\n" + spine + "\n" + q8RouteMain
		dir := t.TempDir()
		source, binary := filepath.Join(dir, "route.cpp"), filepath.Join(dir, "route")
		if err := os.WriteFile(source, []byte(program), 0600); err != nil {
			t.Fatal(err)
		}
		compileCtx, compileCancel := context.WithTimeout(context.Background(), 90*time.Second)
		out, err := exec.CommandContext(compileCtx, compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror",
			"-Wno-unused-parameter", "-Wno-unused-variable", source, "-o", binary).CombinedOutput()
		compileCancel()
		if err != nil {
			t.Fatalf("compile exact production Q8 routing slice: %v\n%s", err, out)
		}
		// Each scenario runs the real fcuda_q8_matmul_f32 decision with kernel launches
		// recorded instead of executed; the sequence IS the routing witness.
		want := map[string]string{
			"rowtile":         "k_q8_quant_act k_q8_gemm_rowtile",
			"misaligned":      "k_q8_quant_act k_q8_gemm",
			"block16":         "k_q8_quant_act k_q8_gemm",
			"decode-lane":     "k_q8_quant_act k_q8_gemm",
			"out-of-contract": "q8_dequant_sgemm",
			// Production GDN sequence-spine caller over the Qwen3.8-27B geometry: convDim x hidden
			// in-proj at the measured P=128 cold-prefill panel, hidden x valueDim out-proj, and the
			// argument-bounds refusal that keeps the retired dequant arm reachable as last resort.
			"gdn-in-proj":  "k_q8_quant_act k_q8_gemm_rowtile",
			"gdn-out-proj": "k_q8_quant_act k_q8_gemm_rowtile",
			"gdn-fallback": "q8_dequant_sgemm",
			// Immediate errors are positive CUDA statuses, never argument refusals:
			// failed quantization must not submit a dependent GEMM or legacy fallback.
			"quant-rowtile-error": "k_q8_quant_act",
			"quant-perrow-error":  "k_q8_quant_act",
			"gemm-rowtile-error":  "k_q8_quant_act k_q8_gemm_rowtile",
			"gemm-perrow-error":   "k_q8_quant_act k_q8_gemm",
			"gdn-quant-error":     "k_q8_quant_act",
			"gdn-gemm-error":      "k_q8_quant_act k_q8_gemm_rowtile",
		}
		for scenario, seq := range want {
			t.Run(scenario, func(t *testing.T) {
				runCtx, runCancel := context.WithTimeout(context.Background(), 30*time.Second)
				defer runCancel()
				out, err := exec.CommandContext(runCtx, binary, scenario).CombinedOutput()
				if err != nil {
					t.Fatalf("production Q8 routing %s: %v\n%s", scenario, err, out)
				}
				if got := strings.TrimSpace(string(out)); got != seq {
					t.Fatalf("launch sequence %q, want %q", got, seq)
				}
			})
		}
	})
}

// q8RouteHarness elides CUDA launches to launch-name records and stands in for the retired
// dequant arm, so the production routing slice compiles on any host.
const q8RouteHarness = `
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
using cudaError_t = int;
static constexpr int cudaSuccess = 0, cudaErrorMemoryAllocation = 2;
#define FCUDA_SCRATCH_ALLOC_FAILED 80001
static std::vector<std::string> launched;
static const char *fail_kernel = "";
static cudaError_t launch_error = cudaSuccess;
static constexpr cudaError_t injected_error = 719;
static inline cudaError_t cudaGetLastError() {
  const cudaError_t status = launch_error;
  launch_error = cudaSuccess;
  return status;
}
static void record_launch(const char *name) {
  launched.push_back(name);
  if (std::strcmp(name, fail_kernel) == 0) launch_error = injected_error;
}
template<class T> static cudaError_t cudaMalloc(T **out, size_t bytes) {
  *out = static_cast<T *>(std::malloc(bytes));
  return *out ? cudaSuccess : cudaErrorMemoryAllocation;
}
static cudaError_t cudaFree(void *) { return cudaSuccess; }
#define FCUDA_TS(name, grid, ...) record_launch(name)
// The production translation unit gets this declaration from cuda_backend.h; the host slice
// does not include the header, so carry the flat ABI prototype here.
extern "C" int fcuda_q8_matmul_rows_f32(const int8_t *, const float *, const float *, float *,
                                        int, int, int, int);
static int q8_dequant_sgemm(const int8_t *, const float *, const float *, float *, int, int, int, int) {
  launched.push_back("q8_dequant_sgemm");
  return 0;
}
`

// q8RouteMain drives the production entry point through the five routing cases: the in-contract
// P>=128 panel (rowtile), the misaligned and block-16 fallbacks (per-row), the unchanged P<128
// decode lane, and the P>65535 argument-bounds refusal (legacy dequant fallback). The gdn-*
// cases drive the extracted production GDN sequence-spine caller over the Qwen3.8-27B geometry.
const q8RouteMain = `
static int scenario(int p, int block, int misalign, const char *failure = "") {
  launched.clear();
  fail_kernel = failure;
  alignas(16) static int8_t codes[256] = {};
  static float scale[256] = {};
  const int out = 8, in = 256;
  float *x = new float[(size_t)p * in];
  float *y = new float[(size_t)p * out];
  for (int i = 0; i < p * in; ++i) x[i] = 0.5f;
  const int8_t *c = misalign ? codes + 1 : codes;
  const int rc = fcuda_q8_matmul_f32(c, scale, x, y, out, in, p, block);
  delete[] x;
  delete[] y;
  const int expected = *failure ? injected_error : 0;
  if (rc != expected) {
    std::fprintf(stderr, "rc=%d want=%d\n", rc, expected);
    return 1;
  }
  for (size_t i = 0; i < launched.size(); ++i)
    std::fprintf(stdout, "%s%s", i ? " " : "", launched[i].c_str());
  std::fprintf(stdout, "\n");
  return 0;
}
static int spineScenario(int out, int in, int p, int block, const char *failure = "") {
  launched.clear();
  fail_kernel = failure;
  alignas(16) static int8_t codes[256] = {};
  static float scale[256] = {};
  // The argument-bounds refusal reads no operand, so only in-bounds panels allocate; the
  // out-of-contract case costs nothing and still proves the last-resort arm stays reachable.
  float *x = nullptr, *y = nullptr;
  if (p <= 65535 && p <= INT_MAX / in) {
    x = new float[(size_t)p * in];
    y = new float[(size_t)p * out];
    for (int i = 0; i < p * in; ++i) x[i] = 0.5f;
  }
  const int rc = qwen35_gdn_q8_prefill(codes, scale, x, y, out, in, p, block);
  delete[] x;
  delete[] y;
  const int expected = *failure ? injected_error : 0;
  if (rc != expected) {
    std::fprintf(stderr, "rc=%d want=%d\n", rc, expected);
    return 1;
  }
  for (size_t i = 0; i < launched.size(); ++i)
    std::fprintf(stdout, "%s%s", i ? " " : "", launched[i].c_str());
  std::fprintf(stdout, "\n");
  return 0;
}
int main(int argc, char **argv) {
  if (argc != 2) return 2;
  const char *s = argv[1];
  if (!std::strcmp(s, "rowtile")) return scenario(200, 32, 0);
  if (!std::strcmp(s, "misaligned")) return scenario(200, 32, 1);
  if (!std::strcmp(s, "block16")) return scenario(200, 16, 0);
  if (!std::strcmp(s, "decode-lane")) return scenario(64, 32, 0);
  if (!std::strcmp(s, "out-of-contract")) return scenario(70000, 32, 0);
  if (!std::strcmp(s, "gdn-in-proj")) return spineScenario(10240, 5120, 128, 32);
  if (!std::strcmp(s, "gdn-out-proj")) return spineScenario(5120, 6144, 512, 32);
  if (!std::strcmp(s, "gdn-fallback")) return spineScenario(10240, 5120, 70000, 32);
  if (!std::strcmp(s, "quant-rowtile-error")) return scenario(200, 32, 0, "k_q8_quant_act");
  if (!std::strcmp(s, "quant-perrow-error")) return scenario(64, 32, 0, "k_q8_quant_act");
  if (!std::strcmp(s, "gemm-rowtile-error")) return scenario(200, 32, 0, "k_q8_gemm_rowtile");
  if (!std::strcmp(s, "gemm-perrow-error")) return scenario(200, 32, 1, "k_q8_gemm");
  if (!std::strcmp(s, "gdn-quant-error")) return spineScenario(10240, 5120, 128, 32, "k_q8_quant_act");
  if (!std::strcmp(s, "gdn-gemm-error")) return spineScenario(5120, 6144, 512, 32, "k_q8_gemm_rowtile");
  return 2;
}
`

// q8RouteSeedVec fills n values with seeded ~[-0.5,0.5) scaled by scale (the lcg/randVec
// helpers are untagged, so this stays available on every build).
func q8RouteSeedVec(g *lcg, n int, scale float32) []float32 {
	v := randVec(g, n)
	for i := range v {
		v[i] *= scale
	}
	return v
}

// q8RouteResident puts host data onto a backend uniformly: cpuref.Upload is identity over the
// host slice; cuda.Upload H2D-copies to VRAM.
func q8RouteResident(be Backend, shape []int, data []float32) Tensor {
	return be.Upload(NewF32(be, shape, data), F32)
}

// TestGDNProjectionParity pins the numerics of the routed P>=128 Q8_0 lane against the f32
// reference on seeded GDN/attn-class fixtures. The retired dequant-compensated arm was an
// Approx peer of this same f32 reference at the recorded cudaQ8CosineMin floor, so holding the
// new lane to a tighter sentinel here holds it to the retired path's numeric class; the lane is
// additionally BitExact per row against the P=1 decode GEMV (TestCUDAQ8PanelGEMMBitExactVsDecodeGEMV).
// Without a CUDA device the test fails closed under FAK_CUDA_REQUIRE_DEVICE=1 and skips otherwise.
// fak-test:runtime fast est=4s lane=default
func TestGDNProjectionParity(t *testing.T) {
	be, ok := Lookup("cuda")
	if !ok {
		if os.Getenv("FAK_CUDA_REQUIRE_DEVICE") == "1" {
			t.Fatal("FAK_CUDA_REQUIRE_DEVICE=1 but the cuda backend is not registered")
		}
		t.Skip("cuda backend not registered (build without -tags cuda or no device)")
	}
	if !be.Caps().UploadDtype {
		t.Skip("cuda backend does not advertise Q8_0 narrowing at H2D")
	}
	ref := Default()
	var seed lcg = 0x7d1e
	g := &seed
	// Scaled GDN/attn projection widths: every contract property the routing reads (in%32==0,
	// block 32, tail tiles, wide-K out-proj class) at unit-test memory budgets.
	shapes := []struct{ out, in int }{{512, 512}, {1024, 512}, {48, 512}, {512, 640}}
	for _, sh := range shapes {
		w := q8RouteSeedVec(g, sh.out*sh.in, 0.5)
		for _, P := range []int{128, 129, 512} {
			t.Run(fmt.Sprintf("%dx%d/P%d", sh.out, sh.in, P), func(t *testing.T) {
				x := q8RouteSeedVec(g, P*sh.in, 1.0)
				yRef := ref.Read(ref.BatchedMatMul(q8RouteResident(ref, []int{sh.out, sh.in}, w),
					q8RouteResident(ref, []int{P, sh.in}, x), P))
				wQ8 := be.Upload(NewF32(be, []int{sh.out, sh.in}, w), Q8_0)
				devX := q8RouteResident(be, []int{P, sh.in}, x)
				yCu := be.Read(be.BatchedMatMul(wQ8, devX, P))
				be.Free(devX)
				be.Free(wQ8)
				if len(yRef) != P*sh.out || len(yCu) != P*sh.out {
					t.Fatalf("shape ref=%d cu=%d want %d", len(yRef), len(yCu), P*sh.out)
				}
				c := cosine(yRef, yCu)
				if c < 0.9995 {
					t.Fatalf("cosine %.8f < 0.9995 sentinel (tighter than the recorded 0.999 Q8 floor)", c)
				}
				maxWant := 0.0
				for _, v := range yRef {
					if a := math.Abs(float64(v)); a > maxWant {
						maxWant = a
					}
				}
				delta := maxAbsDelta(yRef, yCu)
				if maxWant > 0 && delta > 0.02*maxWant {
					t.Fatalf("maxAbsDelta %.3e exceeds 2%% of max|want| %.3e", delta, maxWant)
				}
				t.Logf("GDN-class %dx%d P=%d: cosine=%.8f maxAbs=%.2e [SW-VERIFIED until HW receipt]",
					sh.out, sh.in, P, c, delta)
			})
		}
	}
}
