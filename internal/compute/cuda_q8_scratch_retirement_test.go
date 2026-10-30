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

// Execute the actual persistent Q8 growth code and ledger with fake CUDA calls.
// Kernel launches are counted, not executed. This is a host ownership witness.
// fak-test:runtime medium est=1s lane=default
func TestCUDAQ8ScratchRetirementOwnership(t *testing.T) {
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
	quant := read("cuda/quant_matmul.cuh")
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
	program := strings.TrimPrefix(read("cuda/alloc_ledger.h"), "#pragma once\n") + "\n" +
		cudaQ8RetirementHarness + "\n" + constants + quant[start:end] + "\n" + cudaQ8RetirementMain
	dir := t.TempDir()
	source, binary := filepath.Join(dir, "q8.cpp"), filepath.Join(dir, "q8")
	if err := os.WriteFile(source, []byte(program), 0600); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	// Kernel launch elision leaves rowsPerTile unused; retain all other warnings as errors.
	if out, err := exec.CommandContext(ctx, compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", "-Wno-unused-parameter", "-Wno-unused-variable", source, "-o", binary).CombinedOutput(); err != nil {
		t.Fatalf("compile exact production Q8 ownership: %v\n%s", err, out)
	}
	for _, scenario := range []string{"code-free", "scale-free", "success", "code-alloc", "scale-alloc"} {
		t.Run(scenario, func(t *testing.T) {
			if out, err := exec.CommandContext(ctx, binary, scenario).CombinedOutput(); err != nil {
				t.Fatalf("production Q8 %s: %v\n%s", scenario, err, out)
			}
		})
	}
}

// The CUDA-tagged adapter is not executed by this host test. Pin its distinct
// retirement decoding separately, without changing the older 83000 contract.
// fak-test:runtime fast est=1ms lane=default
func TestCUDAQ8ScratchRetirementDecoderContract(t *testing.T) {
	b, err := os.ReadFile("cuda_prompt_panel.go")
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{
		"cudaQ8ScratchRetireFailedBase = 92000",
		"rc > cudaQ8ScratchRetireFailedBase && rc < cudaQ8ScratchRetireFailedBase+1000",
		`return &CUDAOpError{Op: op, Site: "q8-scratch-retirement", Msg: "persistent Q8 scratch retirement failed", Code: rc - cudaQ8ScratchRetireFailedBase, Class: MemoryScratchpad`,
		"cudaScratchRetireFailedBase = 83000",
	} {
		if !strings.Contains(string(b), want) {
			t.Fatalf("retirement decoder missing %q", want)
		}
	}
}

const cudaQ8RetirementHarness = `
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
using cudaError_t = int;
static constexpr int cudaSuccess=0, cudaErrorMemoryAllocation=2;
#define FCUDA_SCRATCH_ALLOC_FAILED 80001
static int mallocs=0, frees=0, launches=0, free_fail_at=0, alloc_fail_at=0;
static std::unordered_map<void *, size_t> physical;
template<class T> static cudaError_t cudaMalloc(T **out, size_t bytes) {
  ++mallocs;
  if (mallocs==alloc_fail_at) { *out=nullptr; return cudaErrorMemoryAllocation; }
  *out=static_cast<T *>(std::malloc(bytes));
  if (!*out) return cudaErrorMemoryAllocation;
  physical[*out]=bytes;
  return cudaSuccess;
}
static cudaError_t cudaFree(void *ptr) {
  ++frees;
  if (frees==free_fail_at) return 700;
  auto it=physical.find(ptr);
  if (it==physical.end()) return 1;
  physical.erase(it); std::free(ptr); return cudaSuccess;
}
// Discard only the CUDA launch expression. All ownership statements are exact.
#define FCUDA_TS(name, grid, ...) (++launches)
static int q8_dequant_sgemm(const int8_t *, const float *, const float *, float *, int, int, int, int) {
  ++launches; return 0;
}
`

const cudaQ8RetirementMain = `
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr,"Q8 ownership line=%d mallocs=%d frees=%d launches=%d physical=%zu tags=%zu\n",__LINE__,mallocs,frees,launches,physical.size(),g_alloc_tags.size()); return 1; } } while(0)
static bool ledger_matches_physical() {
  int64_t cells[FAK_ALLOC_CELLS*5]={}, unknown=0;
  if (fcuda_alloc_ledger_read(cells, FAK_ALLOC_CELLS, &unknown)!=FAK_ALLOC_CELLS || unknown) return false;
  int64_t bytes=0;
  for (const auto &owner:physical) {
    auto tag=g_alloc_tags.find(owner.first);
    if (tag==g_alloc_tags.end() || tag->second.cls!=FAK_ALLOC_ACTIVATIONS_SCRATCH ||
        tag->second.kind!=FAK_ALLOC_DEVICE_LOCAL || tag->second.bytes!=owner.second) return false;
    bytes+=owner.second;
  }
  const int cell=(FAK_ALLOC_ACTIVATIONS_SCRATCH*FAK_ALLOC_KIND_COUNT+FAK_ALLOC_DEVICE_LOCAL)*5;
  return g_alloc_tags.size()==physical.size() && cells[cell]==bytes && cells[cell+2]==(int64_t)physical.size();
}
static int call_rows(bool tiled, int rows) {
  alignas(16) static int8_t codes[32]={}; static float x[32]={}, y[32]={}, scale[1]={};
  if (tiled) return fcuda_q8_matmul_rows_f32(codes,scale,x,y,8,32,rows,32);
  return fcuda_q8_matmul_f32(codes,scale,x,y,8,32,rows,32);
}
int main(int argc,char **argv) {
  CHECK(argc==2);
  CHECK(call_rows(false,1)==0);
  CHECK(mallocs==2 && frees==0 && launches==2 && physical.size()==2 && ledger_matches_physical());
  void *old_codes=g_q8_qX, *old_scale=g_q8_xScale;
  const int old_code_cap=g_q8_qX_cap, old_scale_cap=g_q8_xScale_cap;
  launches=0;
  if (std::strcmp(argv[1],"code-free")==0 || std::strcmp(argv[1],"scale-free")==0) {
    const bool scale=std::strcmp(argv[1],"scale-free")==0;
    free_fail_at=scale?2:1;
    CHECK(call_rows(scale,2)==92700);
    CHECK(launches==0 && frees==free_fail_at && mallocs==(scale?3:2));
    CHECK(physical.size()==2 && ledger_matches_physical());
    if (scale) {
      CHECK(g_q8_xScale==old_scale && g_q8_xScale_cap==old_scale_cap && g_q8_qX_cap==64);
    } else {
      CHECK(g_q8_qX==old_codes && g_q8_qX_cap==old_code_cap && g_q8_xScale==old_scale);
    }
    // A smaller call must not reuse an owner after an uncertain retirement.
    const int before_mallocs=mallocs, before_frees=frees;
    CHECK(call_rows(false,1)==92700 && call_rows(true,1)==92700);
    CHECK(mallocs==before_mallocs && frees==before_frees && launches==0 && ledger_matches_physical());
  } else if (std::strcmp(argv[1],"success")==0) {
    CHECK(call_rows(true,2)==0);
    CHECK(mallocs==4 && frees==2 && launches==2 && physical.size()==2 && ledger_matches_physical());
    CHECK(g_q8_qX_cap==64 && g_q8_xScale_cap==2);
    int64_t cells[FAK_ALLOC_CELLS*5]={},unknown=0;
    fcuda_alloc_ledger_read(cells,FAK_ALLOC_CELLS,&unknown);
    const int cell=(FAK_ALLOC_ACTIVATIONS_SCRATCH*FAK_ALLOC_KIND_COUNT+FAK_ALLOC_DEVICE_LOCAL)*5;
    CHECK(cells[cell]==72 && cells[cell+2]==2 && cells[cell+3]==4 && cells[cell+4]==2);
  } else {
    const bool scale=std::strcmp(argv[1],"scale-alloc")==0;
    CHECK(scale || std::strcmp(argv[1],"code-alloc")==0);
    alloc_fail_at=scale?4:3;
    CHECK(call_rows(scale,2)==FCUDA_SCRATCH_ALLOC_FAILED);
    CHECK(launches==0 && frees==(scale?2:1) && mallocs==alloc_fail_at);
    CHECK(physical.size()==1 && ledger_matches_physical());
    if (scale) CHECK(g_q8_xScale==nullptr && g_q8_xScale_cap==0 && g_q8_qX_cap==64);
    else CHECK(g_q8_qX==nullptr && g_q8_qX_cap==0 && g_q8_xScale==old_scale);
    CHECK(call_rows(false,1)==0 && ledger_matches_physical());
  }
  // Fake process teardown only; production has no quarantine reset API.
  for (const auto &owner:physical) std::free(owner.first);
  return 0;
}
`
