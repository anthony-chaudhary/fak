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

// fak-test:runtime medium est=10s lane=default
// fak-test:justify Compiles and executes the shared MLA scatter helpers against an independent host reference.
func TestCUDAMLAConcatCacheInsertMatchesHost(t *testing.T) {
	read := func(path string) string {
		t.Helper()
		b, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		return string(b)
	}
	header := read("cuda/paged_kv_write_map.h")
	kernel := read("cuda/mla_cache_insert_vllm.cuh")

	// Port contract: pinned upstream citation, kept kernel bodies, write-map-driven
	// extern C adapters, shared scatter helpers called by every kernel.
	for _, want := range []string{
		"ae065918640a198a56f6f17eb45e251c3229fe41",
		"Apache-2.0",
		"concat_and_cache_mla_kernel",
		"concat_and_cache_mla_grouped_kernel",
		"concat_and_cache_ds_mla_kernel",
		"__global__ void mla_concat_cache_pages_kernel(",
		"__global__ void mla_concat_cache_grouped_pages_kernel(",
		"__global__ void mla_concat_cache_ds_pages_kernel(",
		`extern "C" int fcuda_mla_concat_cache_f32(`,
		`extern "C" int fcuda_mla_concat_cache_grouped_f32(`,
		`extern "C" int fcuda_mla_concat_cache_ds_f32(`,
		"fak_vllm_pa::mla_kv_entry_base",
		"fak_vllm_pa::mla_kv_entry_source",
		"fak_vllm_pa::mla_kv_write_ds_thread",
	} {
		if !strings.Contains(kernel, want) {
			t.Errorf("missing port contract %q", want)
		}
	}
	// Only the f32 lane is served: the quantized-cache arms stay absent.
	for _, banned := range []string{"fp8", "nvfp4", "scaled_convert", "Fp8KVCacheDataType", "float8"} {
		if strings.Contains(strings.ToLower(kernel), banned) {
			t.Errorf("dropped dtype arm %q present in the port", banned)
		}
	}
	// ds family contract: 512-wide latent, rope 0 or 64, reserved-tail entry width.
	if !strings.Contains(kernel, "if (kv_lora_rank != 512 || (pe_dim != 0 && pe_dim != 64)) return 2;") {
		t.Error("ds adapter lost the closed 512/(0|64) geometry refusal")
	}
	if !strings.Contains(kernel, "block_offset, tokens, kv_lora_rank + 64, block_size") {
		t.Error("ds adapter must admit against the reserved 512+64 entry width")
	}
	for _, want := range []string{
		"kMlaMaxLayers", "MlaLayerBases", "mla_kv_write_map_valid",
		"mla_kv_entry_base", "mla_kv_entry_source", "mla_kv_write_ds_thread",
		"ae065918640a198a56f6f17eb45e251c3229fe41",
	} {
		if !strings.Contains(header, want) {
			t.Errorf("write-map header missing shared MLA seam %q", want)
		}
	}

	// Launch-status capture contract: every adapter captures the launch status
	// inside FCUDA_TS, before profiler end can change last-error.
	adapters := []struct{ name, tag string }{
		{"fcuda_mla_concat_cache_f32", `FCUDA_TS("fak_mla_concat_cache", tokens,`},
		{"fcuda_mla_concat_cache_grouped_f32", `FCUDA_TS("fak_mla_concat_cache_grouped", tokens,`},
		{"fcuda_mla_concat_cache_ds_f32", `FCUDA_TS("fak_mla_concat_cache_ds", tokens,`},
	}
	for _, a := range adapters {
		marker := `extern "C" int ` + a.name + `(`
		start := strings.Index(kernel, marker)
		if start < 0 {
			t.Fatalf("%s adapter source missing", a.name)
		}
		// The adapter is a top-level function: its body ends at the next
		// column-zero closing brace.
		end := strings.Index(kernel[start:], "\n}")
		if end < 0 {
			t.Fatalf("%s adapter body unterminated", a.name)
		}
		normal := strings.Join(strings.Fields(kernel[start:start+end]), " ")
		if strings.Count(normal, "cudaGetLastError()") != 1 ||
			!strings.Contains(normal, "cudaError_t launch_status = cudaSuccess; "+a.tag) ||
			!strings.Contains(normal, "launch_status = cudaGetLastError());") ||
			!strings.HasSuffix(normal, "return launch_status == cudaSuccess ? 0 : 4;") {
			t.Fatalf("%s must capture the launch status inside FCUDA_TS before returning", a.name)
		}
	}

	// Host regression: compile the EXACT production header and execute its shared
	// scatter helpers against an independent reference of the host mlaProject+write
	// semantics. GPU compilation, device execution and performance stay unrun.
	compiler, err := exec.LookPath("c++")
	if err != nil {
		t.Fatal("C++14 compiler required for the MLA concat-and-cache host witness: ", err)
	}
	wd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	cpp, binary := filepath.Join(dir, "mla_cache_insert.cpp"), filepath.Join(dir, "mla_cache_insert")
	if err := os.WriteFile(cpp, []byte(mlaConcatCacheHostMain), 0600); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(t.Context(), 60*time.Second)
	defer cancel()
	args := []string{"-std=c++14", "-O1", "-Wall", "-Wextra", "-Werror",
		"-I", filepath.Join(wd, "cuda"), cpp, "-o", binary}
	if out, err := exec.CommandContext(ctx, compiler, args...).CombinedOutput(); err != nil {
		t.Fatalf("compile MLA host regression against the production header: %v\n%s", err, out)
	}
	out, err := exec.CommandContext(ctx, binary).CombinedOutput()
	if err != nil {
		t.Fatalf("MLA host regression: %v\n%s", err, out)
	}
	const sentinel = "PASS MLA concat-and-cache host regression: single fresh page, page-crossing run, grouped/ds remap vs host reference; CUDA/device/performance unrun"
	if strings.TrimSpace(string(out)) != sentinel {
		t.Fatalf("incomplete host witness: %s", out)
	}
}

const mlaConcatCacheHostMain = `
// Host qualification of the exact MLA scatter helpers used by the CUDA kernels,
// not GPU execution. Includes the production paged_kv_write_map.h so the
// regression executes the same code the kernels run.
#include "paged_kv_write_map.h"
#include <cassert>
#include <cstdio>
#include <cstring>
#include <vector>
#ifdef NDEBUG
#error "MLA address regression requires active assertions"
#endif

using namespace fak_vllm_pa;

static std::vector<float> row_data(int n, int seed) {
  std::vector<float> out(n);
  for (int i = 0; i < n; ++i) out[i] = float((i * 37 + seed * 19) % 211 - 105) * 0.001f;
  return out;
}

static void fill_map(PagedKVWriteMap& pages, const int* blocks, int count, int first_offset) {
  pages = PagedKVWriteMap();
  pages.first_offset = first_offset;
  for (int i = 0; i < count; ++i) pages.blocks[i] = blocks[i];
}

// Independent reference: host mlaProject+write semantics. Each logical token t in
// [0, tokens) owns the row [latent | rope] that model.mlaProject emits, and the
// generic paged write places it at page blocks[t/block_size], in-page offset
// t%block_size, entry width rank+pe. Every other cell keeps the sentinel.
static void reference_fill(std::vector<float>& arena, int capacity, const int* blocks,
                           int first_offset, int tokens, int rank, int pe,
                           const float* kv_c, const float* k_pe) {
  const int entry = rank + pe;
  const int block_size = 16;
  assert(int(arena.size()) == capacity * block_size * entry);
  std::fill(arena.begin(), arena.end(), -1.0f);
  for (int t = 0; t < tokens; ++t) {
    const int pos = first_offset + t;
    const int page = blocks[pos / block_size];
    const int off = pos % block_size;
    float* dst = arena.data() + int64_t(page) * block_size * entry + int64_t(off) * entry;
    for (int e = 0; e < rank; ++e) dst[e] = kv_c[int64_t(t) * rank + e];
    for (int e = 0; e < pe; ++e) dst[rank + e] = k_pe[int64_t(t) * pe + e];
  }
}

// Executes the shared scatter primitives exactly as the CUDA kernels do: entry
// base once per token, source-selected element stores.
static void helper_scatter(std::vector<float>& arena, const int* blocks, int count,
                           int first_offset, int tokens, int rank, int pe,
                           const float* kv_c, const float* k_pe) {
  PagedKVWriteMap pages;
  fill_map(pages, blocks, count, first_offset);
  const int entry = rank + pe;
  for (int t = 0; t < tokens; ++t) {
    const int64_t base = mla_kv_entry_base(pages, t, entry, 16);
    for (int e = 0; e < entry; ++e)
      arena[base + e] = mla_kv_entry_source(kv_c, k_pe, t, e, rank, pe);
  }
}

static void check_admission() {
  int good[3] = {1, 2, 3};
  assert(mla_kv_write_map_valid(good, 3, 10, 0, 48, 576, 16));
  assert(!mla_kv_write_map_valid(nullptr, 3, 10, 0, 48, 576, 16));
  assert(!mla_kv_write_map_valid(good, 0, 10, 0, 48, 576, 16));
  int big[65];
  for (int i = 0; i < 65; ++i) big[i] = i;
  assert(!mla_kv_write_map_valid(big, 65, 100, 0, 1040, 576, 16));
  assert(!mla_kv_write_map_valid(good, 3, 0, 0, 48, 576, 16));
  assert(!mla_kv_write_map_valid(good, 3, 10, 16, 48, 576, 16));
  assert(!mla_kv_write_map_valid(good, 3, 10, -1, 48, 576, 16));
  assert(!mla_kv_write_map_valid(good, 3, 10, 0, 0, 576, 16));
  assert(!mla_kv_write_map_valid(good, 3, 10, 0, 48, 0, 16));
  assert(!mla_kv_write_map_valid(good, 3, 10, 0, 48, 576, 32));
  assert(!mla_kv_write_map_valid(good, 3, 10, 0, 49, 576, 16));
  int dup[2] = {5, 5};
  assert(!mla_kv_write_map_valid(dup, 2, 10, 0, 32, 576, 16));
  int oob[1] = {10};
  assert(!mla_kv_write_map_valid(oob, 1, 10, 0, 16, 576, 16));
  int neg[1] = {-1};
  assert(!mla_kv_write_map_valid(neg, 1, 10, 0, 16, 576, 16));
}

static void check_single_fresh_page() {
  // (a) single-token insert into a fresh page equals host mlaProject+write.
  const int rank = 512, pe = 64, capacity = 8;
  const int blocks[] = {3};
  assert(mla_kv_write_map_valid(blocks, 1, capacity, 0, 1, rank + pe, 16));
  std::vector<float> kv_c = row_data(rank, 1), k_pe = row_data(pe, 2);
  std::vector<float> want(capacity * 16 * (rank + pe)), got(want.size());
  reference_fill(want, capacity, blocks, 0, 1, rank, pe, kv_c.data(), k_pe.data());
  std::fill(got.begin(), got.end(), -1.0f);
  helper_scatter(got, blocks, 1, 0, 1, rank, pe, kv_c.data(), k_pe.data());
  assert(std::memcmp(got.data(), want.data(), want.size() * sizeof(float)) == 0);
}

static void check_multi_token_page_crossing() {
  // (b) multi-token run crossing page boundaries through the write map.
  const int rank = 512, pe = 64, capacity = 12;
  const int blocks[] = {7, 2, 11, 4};
  const int first_offset = 14, tokens = 35;
  assert(mla_kv_write_map_valid(blocks, 4, capacity, first_offset, tokens, rank + pe, 16));
  std::vector<float> kv_c = row_data(tokens * rank, 3), k_pe = row_data(tokens * pe, 4);
  std::vector<float> want(capacity * 16 * (rank + pe)), got(want.size());
  reference_fill(want, capacity, blocks, first_offset, tokens, rank, pe, kv_c.data(), k_pe.data());
  std::fill(got.begin(), got.end(), -1.0f);
  helper_scatter(got, blocks, 4, first_offset, tokens, rank, pe, kv_c.data(), k_pe.data());
  assert(std::memcmp(got.data(), want.data(), want.size() * sizeof(float)) == 0);
  // A second geometry (narrower latent, wider rope) over its own page map.
  const int rank2 = 256, pe2 = 128;
  const int blocks2[] = {0, 5, 1};
  const int first2 = 15, tokens2 = 20;
  assert(mla_kv_write_map_valid(blocks2, 3, capacity, first2, tokens2, rank2 + pe2, 16));
  std::vector<float> c2 = row_data(tokens2 * rank2, 5), p2 = row_data(tokens2 * pe2, 6);
  std::vector<float> want2(capacity * 16 * (rank2 + pe2)), got2(want2.size());
  reference_fill(want2, capacity, blocks2, first2, tokens2, rank2, pe2, c2.data(), p2.data());
  std::fill(got2.begin(), got2.end(), -1.0f);
  helper_scatter(got2, blocks2, 3, first2, tokens2, rank2, pe2, c2.data(), p2.data());
  assert(std::memcmp(got2.data(), want2.data(), want2.size() * sizeof(float)) == 0);
}

static void check_grouped_remap() {
  // (c) grouped variant: layer-strided sources, one shared write map, per-layer
  // arena bases - each layer's remap equals the host reference for its slice.
  const int layers = 2, rank = 512, pe = 64, capacity = 10;
  const int blocks[] = {6, 1, 9};
  const int first_offset = 8, tokens = 32;
  assert(mla_kv_write_map_valid(blocks, 3, capacity, first_offset, tokens, rank + pe, 16));
  std::vector<float> kv_c = row_data(layers * tokens * rank, 7),
                     k_pe = row_data(layers * tokens * pe, 8);
  std::vector<std::vector<float>> arenas(
      layers, std::vector<float>(capacity * 16 * (rank + pe), -1.0f));
  // The by-value layer pointer array is the exact kernel argument shape.
  MlaLayerBases bases = {};
  for (int l = 0; l < layers; ++l) bases.layers[l] = arenas[l].data();
  PagedKVWriteMap pages;
  fill_map(pages, blocks, 3, first_offset);
  const int entry = rank + pe;
  for (int l = 0; l < layers; ++l) {
    const float* c = kv_c.data() + int64_t(l) * tokens * rank;
    const float* p = k_pe.data() + int64_t(l) * tokens * pe;
    float* cache = bases.layers[l];
    for (int t = 0; t < tokens; ++t) {
      const int64_t base = mla_kv_entry_base(pages, t, entry, 16);
      for (int e = 0; e < entry; ++e)
        cache[base + e] = mla_kv_entry_source(c, p, t, e, rank, pe);
    }
    std::vector<float> want(capacity * 16 * (rank + pe));
    reference_fill(want, capacity, blocks, first_offset, tokens, rank, pe, c, p);
    assert(std::memcmp(cache, want.data(), want.size() * sizeof(float)) == 0);
  }
}

static void check_ds_remap() {
  // (c) ds variant: fixed 96-thread partition, reserved 64-wide rope tail.
  const int rank = 512, capacity = 9;
  const int blocks[] = {4, 8, 0};
  const int first_offset = 10, tokens = 26;
  assert(mla_kv_write_map_valid(blocks, 3, capacity, first_offset, tokens, rank + 64, 16));
  std::vector<float> kv_c = row_data(tokens * rank, 9), k_pe = row_data(tokens * 64, 10);
  std::vector<float> arena(capacity * 16 * (rank + 64)), want(arena.size());
  std::fill(arena.begin(), arena.end(), -1.0f);
  PagedKVWriteMap pages;
  fill_map(pages, blocks, 3, first_offset);
  for (int t = 0; t < tokens; ++t)
    for (int th = 0; th < 96; ++th)
      mla_kv_write_ds_thread(pages, t, th, rank, 64, 16, kv_c.data(), k_pe.data(), arena.data());
  reference_fill(want, capacity, blocks, first_offset, tokens, rank, 64, kv_c.data(),
                  k_pe.data());
  assert(std::memcmp(arena.data(), want.data(), want.size() * sizeof(float)) == 0);
  // pe_dim = 0: the reserved rope tail is zeroed, the entry stays 512 + 64 wide.
  std::fill(arena.begin(), arena.end(), -1.0f);
  for (int t = 0; t < tokens; ++t)
    for (int th = 0; th < 96; ++th)
      mla_kv_write_ds_thread(pages, t, th, rank, 0, 16, kv_c.data(), nullptr, arena.data());
  std::fill(want.begin(), want.end(), -1.0f);
  for (int t = 0; t < tokens; ++t) {
    const int pos = first_offset + t;
    const int page = blocks[pos / 16], off = pos % 16;
    float* dst = want.data() + int64_t(page) * 16 * (rank + 64) + int64_t(off) * (rank + 64);
    for (int e = 0; e < rank; ++e) dst[e] = kv_c[int64_t(t) * rank + e];
    for (int e = 0; e < 64; ++e) dst[rank + e] = 0.0f;
  }
  assert(std::memcmp(arena.data(), want.data(), want.size() * sizeof(float)) == 0);
}

int main() {
  check_admission();
  check_single_fresh_page();
  check_multi_token_page_crossing();
  check_grouped_remap();
  check_ds_remap();
  std::puts("PASS MLA concat-and-cache host regression: single fresh page, page-crossing run, "
            "grouped/ds remap vs host reference; CUDA/device/performance unrun");
  return 0;
}
`
