// Programmatic Dependent Launch (PDL) for the Inkling-2 resident graphs (sm_90+).
//
// PDL lets a kernel launch while its predecessor in the captured graph is still running,
// so launch latency and block scheduling overlap the predecessor's tail. It changes only
// when blocks start, never what they compute: every PDL-aware kernel calls
// ink2_pdl_enter() as its first statement, which waits for the predecessor grid to finish
// and flush its writes before the body reads or writes anything, then lets the successor
// launch.
//
// The capture stays an ordinary <<<>>> capture. ink2_pdl_rewrite() turns each captured
// kernel->kernel edge whose downstream kernel is PDL-aware into a programmatic edge, which
// is what cudaLaunchAttributeProgrammaticStreamSerialization would have recorded. A
// downstream kernel is PDL-aware when its symbol carries one of the k_ink2_, k_ink2v_ or
// k_ink2d_ prefixes; TestInkling2PDLKernelsWaitFirst holds every kernel with those prefixes
// to the ink2_pdl_enter() contract. Edges into any other kernel, and edges touching
// memcpy or memset nodes, stay full serialization.
//
// Waiting first, in every block, before any early return, is what keeps the chain
// transitive: a kernel cannot finish before its predecessor has, so a successor that waits
// on it also waits on everything earlier.
//
// FAK_INKLING2_PDL=1 enables the rewrite. It stays off by default because on sm_120 it
// measured slower: triggering dependents at entry lets their blocks sit resident on SMs
// in griddepcontrol.wait, taking occupancy from the predecessor still running.
// FAK_INKLING2_PDL_TRACE=1 prints the per-graph edge census to stderr.
#pragma once

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <unordered_map>
#include <vector>

__device__ __forceinline__ void ink2_pdl_enter(void) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}

static int g_ink2_pdl_mode = -1;

static bool ink2_pdl_enabled(void) {
  if (g_ink2_pdl_mode < 0) {
    const char *v = getenv("FAK_INKLING2_PDL");
    g_ink2_pdl_mode = (v && v[0] == '1' && v[1] == 0) ? 1 : 0;
  }
  return g_ink2_pdl_mode == 1;
}

// Test hook: 1 enables the rewrite for graphs captured from now on, 0 disables it;
// returns the previous mode. An already-instantiated graph keeps the edges it was built with.
extern "C" int fcuda_ink2_pdl_select(int mode) {
  const int prev = ink2_pdl_enabled() ? 1 : 0;
  g_ink2_pdl_mode = mode ? 1 : 0;
  return prev;
}

// Matches the kernel's own name in an Itanium-mangled symbol (_Z<len><name>...), the same
// prefix rule the lint enforces, so a parameter type or a longer name cannot match.
static bool ink2_pdl_symbol_aware(const char *sym) {
  if (!sym || sym[0] != '_' || sym[1] != 'Z') return false;
  const char *p = sym + 2;
  if (*p < '0' || *p > '9') return false;
  while (*p >= '0' && *p <= '9') ++p;
  return strncmp(p, "k_ink2_", 7) == 0 || strncmp(p, "k_ink2v_", 8) == 0 || strncmp(p, "k_ink2d_", 8) == 0;
}

// The wait compiles out below sm_90, so the loaded code (not the device) decides: a
// compute_8x PTX JIT-compiled for a newer GPU has no wait and must stay serialized.
static bool ink2_pdl_code_ok(const void *func) {
  cudaFuncAttributes a = {};
  return cudaFuncGetAttributes(&a, func) == cudaSuccess && a.ptxVersion >= 90 && a.binaryVersion >= 90;
}

#if CUDART_VERSION >= 13000
#define INK2_PDL_GET_EDGES cudaGraphGetEdges
#define INK2_PDL_REMOVE_DEPS cudaGraphRemoveDependencies
#define INK2_PDL_ADD_DEPS cudaGraphAddDependencies
#elif CUDART_VERSION >= 12030
#define INK2_PDL_GET_EDGES cudaGraphGetEdges_v2
#define INK2_PDL_REMOVE_DEPS cudaGraphRemoveDependencies_v2
#define INK2_PDL_ADD_DEPS cudaGraphAddDependencies_v2
#endif

// Rewrites the eligible edges of a freshly captured graph in place; a no-op unless PDL is
// enabled on an sm_90+ device. Returns a cudaError_t.
static int ink2_pdl_rewrite(cudaGraph_t graph) {
  if (!graph || !ink2_pdl_enabled()) return (int)cudaSuccess;
#if defined(INK2_PDL_GET_EDGES)
  size_t n = 0;
  cudaError_t e = INK2_PDL_GET_EDGES(graph, nullptr, nullptr, nullptr, &n);
  if (e != cudaSuccess) return (int)e;
  std::vector<cudaGraphNode_t> from(n), to(n);
  std::vector<cudaGraphEdgeData> data(n);
  if (n) {
    e = INK2_PDL_GET_EDGES(graph, from.data(), to.data(), data.data(), &n);
    if (e != cudaSuccess) return (int)e;
  }
  std::unordered_map<cudaGraphNode_t, int> aware;  // 1 aware kernel, 0 other kernel, -1 not a kernel
  std::unordered_map<std::string, int> foreign;  // serialized kernel symbol -> node count
  auto classify = [&](cudaGraphNode_t node) -> int {
    auto it = aware.find(node);
    if (it != aware.end()) return it->second;
    int v = -1;
    cudaGraphNodeType t;
    if (cudaGraphNodeGetType(node, &t) == cudaSuccess && t == cudaGraphNodeTypeKernel) {
      cudaKernelNodeParams p = {};
      const char *name = nullptr;
      if (cudaGraphKernelNodeGetParams(node, &p) == cudaSuccess && p.func &&
          cudaFuncGetName(&name, p.func) == cudaSuccess && ink2_pdl_symbol_aware(name) && ink2_pdl_code_ok(p.func)) {
        v = 1;
      } else {
        v = 0;
        ++foreign[name ? name : "?"];
      }
    }
    aware.emplace(node, v);
    return v;
  };
  size_t converted = 0, kernelEdges = 0;
  for (size_t i = 0; i < n; ++i) {
    if (data[i].type != cudaGraphDependencyTypeDefault || data[i].from_port != 0) continue;
    if (classify(from[i]) < 0) continue;
    const int downstream = classify(to[i]);
    if (downstream < 0) continue;
    ++kernelEdges;
    if (downstream != 1) continue;
    e = INK2_PDL_REMOVE_DEPS(graph, &from[i], &to[i], &data[i], 1);
    if (e != cudaSuccess) return (int)e;
    cudaGraphEdgeData pe = {};
    pe.from_port = cudaGraphKernelNodePortProgrammatic;
    pe.type = cudaGraphDependencyTypeProgrammatic;
    e = INK2_PDL_ADD_DEPS(graph, &from[i], &to[i], &pe, 1);
    if (e != cudaSuccess) return (int)e;
    ++converted;
  }
  const char *tr = getenv("FAK_INKLING2_PDL_TRACE");
  if (tr && tr[0] == '1') {
    size_t kernels = 0, awareKernels = 0;
    for (const auto &kv : aware) {
      if (kv.second >= 0) ++kernels;
      if (kv.second == 1) ++awareKernels;
    }
    fprintf(stderr, "fak-cuda ink2-pdl edges=%zu kernel_edges=%zu programmatic=%zu kernels=%zu aware=%zu\n", n,
            kernelEdges, converted, kernels, awareKernels);
    for (const auto &kv : foreign)
      fprintf(stderr, "fak-cuda ink2-pdl serialized-kernel nodes=%d %s\n", kv.second, kv.first.c_str());
  }
  return (int)cudaSuccess;
#else
  return (int)cudaSuccess;
#endif
}
