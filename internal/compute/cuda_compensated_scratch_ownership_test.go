package compute

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

// Compile the production prelaunch allocation blocks with a fake allocator.
// This proves ownership/fail-fast behavior, not CUDA execution or VRAM savings.
// fak-test:runtime medium est=1s lane=default
func TestCUDACompensatedScratchFailureOwnership(t *testing.T) {
	compiler, err := exec.LookPath("c++")
	if err != nil {
		t.Skip("host C++ compiler unavailable")
	}
	read := func(path string) string {
		t.Helper()
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		return string(data)
	}
	gdn, quant := read("cuda/qwen35_gdn.cuh"), read("cuda/quant_matmul.cuh")
	// Include the exact production transaction when present. Keeping extraction
	// compatible with the old inline blocks makes the pre-fix failure executable.
	helper := ""
	if begin := strings.Index(gdn, "#define FCUDA_SCRATCH_RETIRE_FAILED_BASE "); begin >= 0 {
		end := strings.Index(gdn[begin:], "// Same three ordered GEMMs")
		if end < 0 {
			t.Fatal("compensated scratch transaction boundary missing")
		}
		helper = gdn[begin : begin+end]
	}
	program := cudaCompensatedScratchHarness + helper
	for i, tc := range []struct {
		source, function string
	}{
		{gdn, "static int fp16_compensated_matmul("},
		{gdn, "static int q8_dequant_sgemm("},
		{quant, "extern \"C\" int fcuda_q4k_matmul_f32("},
		{quant, "extern \"C\" int fcuda_q2k_matmul_f32("},
	} {
		begin := strings.Index(tc.source, tc.function)
		if begin < 0 {
			t.Fatalf("production function missing: %s", tc.function)
		}
		body := tc.source[begin:]
		begin = strings.Index(body, "size_t wn =")
		if begin < 0 {
			t.Fatalf("allocation block missing: %s", tc.function)
		}
		body = body[begin:]
		end := strings.Index(body, "FCUDA_TS(")
		if end < 0 {
			t.Fatalf("first launch boundary missing: %s", tc.function)
		}
		// The success suffix is also copied, rather than modeling pool release.
		release := "fcuda_free(wHi); fcuda_free(wLo); fcuda_free(xHi); fcuda_free(xLo);"
		if !strings.Contains(body[end:], release) {
			t.Fatalf("production success release changed: %s", tc.function)
		}
		program += fmt.Sprintf("\nstatic int route%d() {\n", i) +
			"const int out=320, in=256, tokens=128, rows=320, cols=256, P=128;\n" +
			body[:end] + "\n++launches;\n" + release + "\nreturn 0;\n}\n"
	}
	program += cudaCompensatedScratchMain
	dir := t.TempDir()
	cpp, binary := filepath.Join(dir, "scratch.cpp"), filepath.Join(dir, "scratch")
	if runtime.GOOS == "windows" {
		binary += ".exe"
	}
	if err := os.WriteFile(cpp, []byte(program), 0600); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if out, err := exec.CommandContext(ctx, compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", "-Wno-unused-variable", cpp, "-o", binary).CombinedOutput(); err != nil {
		t.Fatalf("compile production scratch allocation blocks: %v\n%s", err, out)
	}
	if out, err := exec.CommandContext(ctx, binary).CombinedOutput(); err != nil {
		t.Fatalf("production scratch ownership: %v\n%s", err, out)
	}
	// The CUDA-tagged Go adapter cannot execute on this host; pin the decoder
	// separately so the raw CUDA code is not mislabeled as an allocation miss.
	mapping := read("cuda_prompt_panel.go")
	for _, want := range []string{
		"cudaScratchRetireFailedBase = 83000",
		"rc > cudaScratchRetireFailedBase && rc < cudaScratchRetireFailedBase+1000",
		`Site: "gemm-scratch-retirement", Msg: "scratch retirement failed after allocation miss", Code: rc - cudaScratchRetireFailedBase, Class: MemoryScratchpad`,
	} {
		if !strings.Contains(mapping, want) {
			t.Fatalf("production scratch-retirement decoder missing %q", want)
		}
	}
	gdnMapping := read("cuda_specialized.go")
	for _, want := range []string{
		"status > cudaScratchRetireFailedBase && status < cudaScratchRetireFailedBase+1000",
		`kernelErr.Stage = "scratch-retirement"`,
		"kernelErr.Code = status - cudaScratchRetireFailedBase",
	} {
		if !strings.Contains(gdnMapping, want) {
			t.Fatalf("production GDN scratch-retirement decoder missing %q", want)
		}
	}
}

const cudaCompensatedScratchHarness = `
#include <cstddef>
#include <cstdio>
using __half = unsigned short;
#define FCUDA_SPLIT_SCRATCH_ALLOC_FAILED 80002
static __half storage[4];
static int attempts, miss, discarded, pooled, launches, cleanup_failure;
static bool live[4], held[4];
static size_t allocated_bytes[4];
static void reset(int failure, int cleanup) {
  attempts=discarded=pooled=launches=0; miss=failure; cleanup_failure=cleanup;
  for (int i=0; i<4; ++i) { live[i]=held[i]=false; allocated_bytes[i]=0; }
}
static void *scratch_malloc(size_t bytes) {
  const int i=attempts++;
  if (attempts == miss) return nullptr;
  if (i >= 4) return nullptr;
  live[i]=true; allocated_bytes[i]=bytes;
  return &storage[i];
}
static void fcuda_free(void *p) {
  if (!p) return;
  const int i=(__half *)p-storage;
  if (i<0 || i>=4 || !live[i] || held[i]) { std::puts("invalid pool release"); return; }
  live[i]=false; ++pooled;
}
[[maybe_unused]] static int fcuda_discard_unsubmitted(void *p, size_t bytes) {
  const int i=(__half *)p-storage;
  if (i<0 || i>=4 || !live[i] || bytes!=allocated_bytes[i]) return 701;
  ++discarded;
  if (discarded == cleanup_failure) { held[i]=true; return 700; }
  live[i]=false;
  return 0;
}
`

const cudaCompensatedScratchMain = `
int main() {
  int (*routes[])() = {route0, route1, route2, route3};
  for (int route=0; route<4; ++route) {
    for (int fail=1; fail<=4; ++fail) {
      reset(fail, 0);
      const int rc=routes[route]();
      if (rc!=FCUDA_SPLIT_SCRATCH_ALLOC_FAILED || attempts!=fail ||
          discarded!=fail-1 || pooled!=0 || launches!=0) {
        std::printf("route=%d miss=%d rc=%d attempts=%d retired=%d pooled=%d launches=%d\n",
                    route,fail,rc,attempts,discarded,pooled,launches);
        return 1;
      }
      for (int i=0; i<4; ++i) if (live[i]) return 2;
    }
    reset(0, 0);
    if (routes[route]()!=0 || attempts!=4 || discarded!=0 || pooled!=4 || launches!=1) return 3;
    // A retirement fault must not be hidden as OOM or make its pointer reusable;
    // the other acquired pointers must still get their own cleanup attempt.
    reset(4, 1);
    if (routes[route]()!=83700 || attempts!=4 || discarded!=3 || pooled!=0 || launches!=0 ||
        !live[0] || !held[0] || live[1] || live[2] || live[3]) return 4;
  }
  std::puts("four production allocation routes: fail-fast, exact retirement, success pooling, cleanup-fault hold");
}
`
