package compute

import (
	"context"
	"errors"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

// TestGraphFitAccounting witnesses that CUDA-graph fixed KV preallocation is charged into
// the pre-launch MemoryPlan so fits that would exceed device VRAM fail pre-launch with a
// typed *FitError (Verdict: FitTooBig) while keeping CUDA-graph decode strictly opt-in.
// This is a fit-contract and AST-extracted runtime-logic witness, not a hardware serve.
// fak-test:runtime medium est=2s lane=default
func TestGraphFitAccounting(t *testing.T) {
	rawGraphCUDA, err := os.ReadFile("graph_cuda.go")
	if err != nil {
		t.Fatal(err)
	}
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "graph_cuda.go", rawGraphCUDA, 0)
	if err != nil {
		t.Fatal(err)
	}

	wanted := map[string]bool{
		"EnableCUDAGraph":            false,
		"CUDAGraphReplays":           false,
		"SetCUDAGraphKVCapacity":     false,
		"CUDAGraphEnabled":           false,
		"CUDAGraphKVCapacity":        false,
		"CUDAGraphKVPreallocDemand":  false,
		"graphKVAlreadyChargedBytes": false,
		"withCUDAGraphKVPreallocFor": false,
		"WithCUDAGraphKVPrealloc":    false,
		"RefuseCUDAGraphFitIfTooBig": false,
	}
	var extracted strings.Builder
	var setCapBody string
	for _, decl := range file.Decls {
		fn, ok := decl.(*ast.FuncDecl)
		if !ok || fn.Recv != nil {
			continue
		}
		name := fn.Name.Name
		if name == "SetCUDAGraphKVCapacity" && fn.Body != nil {
			setCapBody = string(rawGraphCUDA[fset.Position(fn.Body.Pos()).Offset:fset.Position(fn.Body.End()).Offset])
		}
		if _, need := wanted[name]; !need {
			continue
		}
		if wanted[name] {
			t.Fatalf("duplicate graph_cuda.go declaration %s", name)
		}
		wanted[name] = true
		extracted.Write(rawGraphCUDA[fset.Position(fn.Pos()).Offset:fset.Position(fn.End()).Offset])
		extracted.WriteByte('\n')
	}
	for name, seen := range wanted {
		if !seen {
			t.Fatalf("missing graph_cuda.go declaration %s", name)
		}
	}
	if strings.Contains(setCapBody, "graphEnabled") {
		t.Fatal("SetCUDAGraphKVCapacity must never flip graphEnabled; CUDA graph decode must remain opt-in")
	}

	t.Run("production-graph-cuda-ast-and-behavior", func(t *testing.T) {
		rawGraphKV, err := os.ReadFile("graph_kv.go")
		if err != nil {
			t.Fatal(err)
		}
		kvFset := token.NewFileSet()
		kvFile, err := parser.ParseFile(kvFset, "graph_kv.go", rawGraphKV, 0)
		if err != nil {
			t.Fatal(err)
		}
		var kvExtracted strings.Builder
		for _, decl := range kvFile.Decls {
			fn, ok := decl.(*ast.FuncDecl)
			if !ok || fn.Recv != nil {
				continue
			}
			if fn.Name.Name == "GraphKVPreallocBytes" || fn.Name.Name == "GraphKVPreallocFor" {
				kvExtracted.Write(rawGraphKV[kvFset.Position(fn.Pos()).Offset:kvFset.Position(fn.End()).Offset])
				kvExtracted.WriteByte('\n')
			}
		}

		program := graphFitAccountingHarness + "\n" + kvExtracted.String() + "\n" + extracted.String() + "\n" + graphFitAccountingMain
		dir := t.TempDir()
		goFile := filepath.Join(dir, "main.go")
		binary := filepath.Join(dir, "graphfit")
		if runtime.GOOS == "windows" {
			binary += ".exe"
		}
		if err := os.WriteFile(goFile, []byte(program), 0600); err != nil {
			t.Fatal(err)
		}
		ctx, cancel := context.WithTimeout(t.Context(), 60*time.Second)
		defer cancel()
		build := exec.CommandContext(ctx, filepath.Join(runtime.GOROOT(), "bin", "go"), "build", "-o", binary, goFile)
		build.Env = append(os.Environ(), "CGO_ENABLED=0", "GOTOOLCHAIN=local", "GOFLAGS=")
		build.WaitDelay = time.Second
		if out, err := build.CombinedOutput(); err != nil {
			t.Fatalf("compile extracted graph fit accounting: %v\n%s", err, out)
		}
		run := exec.CommandContext(ctx, binary)
		run.WaitDelay = time.Second
		if out, err := run.CombinedOutput(); err != nil {
			t.Fatalf("graph fit accounting witness failed: %v\n%s", err, out)
		}
	})

	t.Run("live-compute-plan-integration", func(t *testing.T) {
		// Qwen3.8-27B full-attention geometry: 16 full-attn layers, 4 KV heads, headDim 256 -> 192 KiB/pos F32.
		cfg := KVConfig{
			NumLayers:    16,
			NumKVHeads:   4,
			HeadDim:      256,
			Precision:    KVPrecisionF32,
			GraphCapture: true,
		}
		const graphCapPos = 32768 // 6 GiB fixed graph prealloc at 192 KiB/pos
		const promptPos = 512     // 96 MiB prompt KV
		wantGraphKV := GraphKVPreallocBytes(cfg, graphCapPos)
		if wantGraphKV != 6<<30 {
			t.Fatalf("GraphKVPreallocBytes = %d, want 6 GiB (%d)", wantGraphKV, int64(6<<30))
		}
		// 24 GiB device: 19 GiB weights + 96 MiB prompt KV fits before graph KV charge, refuses after.
		dev := capDevice{total: 24 << 30, free: 24 << 30, known: true}
		base := MemoryPlan{
			{Class: MemoryWeights, Bytes: 19 << 30, Detail: "resident-weights", Scope: MemoryScopeDevice},
		}
		base = append(base, EstimateKVStoreMemoryPlan(cfg, promptPos)...)
		if err := RefuseMemoryPlanIfTooBig(dev, base, 0); err != nil {
			t.Fatalf("base prompt plan without graph prealloc should admit: %v", err)
		}
		// Opt-in disabled keeps prealloc zero.
		off := GraphKVPreallocFor(false, cfg, graphCapPos, 24<<30)
		if off.Preallocate {
			t.Fatalf("GraphKVPreallocFor(graphOn=false) preallocated: %+v", off)
		}
		// Opt-in enabled delta above already-charged hal-kv-store pushes total over 24 GiB.
		delta := wantGraphKV - EstimateKVStoreBytes(cfg, promptPos)
		charged := append(append(MemoryPlan(nil), base...), MemoryDemand{
			Class:  MemoryKVCache,
			Bytes:  delta,
			Detail: "cuda-graph-kv-prealloc",
			Scope:  MemoryScopeDevice,
			DType:  cfg.Precision.StorageLabel(),
		})
		err := RefuseMemoryPlanIfTooBig(dev, charged, 0)
		var fe *FitError
		if !errors.As(err, &fe) || fe.Verdict != FitTooBig || fe.Scope != MemoryScopeDevice {
			t.Fatalf("graph-charged refusal = %T (%v), want *FitError(FitTooBig, device)", err, err)
		}
		if fe.Want != (19<<30)+wantGraphKV || fe.Avail != 24<<30 {
			t.Fatalf("FitError sizing Want=%d Avail=%d, want Want=%d Avail=%d", fe.Want, fe.Avail, (19<<30)+wantGraphKV, int64(24<<30))
		}
		// F16 KV cuts prealloc to 8 B/elem (4 GiB for 32768 pos), fitting inside 19 + 4 = 23 <= 24 GiB.
		cfgF16 := cfg
		cfgF16.Precision = KVPrecisionF16
		f16Bytes := GraphKVPreallocBytes(cfgF16, graphCapPos)
		if f16Bytes != 4<<30 {
			t.Fatalf("F16 GraphKVPreallocBytes = %d, want 4 GiB", f16Bytes)
		}
		f16Plan := MemoryPlan{
			{Class: MemoryWeights, Bytes: 19 << 30, Detail: "resident-weights", Scope: MemoryScopeDevice},
			{Class: MemoryKVCache, Bytes: f16Bytes, Detail: "cuda-graph-kv-prealloc", Scope: MemoryScopeDevice},
		}
		if err := RefuseMemoryPlanIfTooBig(dev, f16Plan, 0); err != nil {
			t.Fatalf("F16 graph prealloc plan (23 GiB <= 24 GiB) refused: %v", err)
		}
	})
}

const graphFitAccountingHarness = `package main

import (
	"errors"
	"fmt"
	"math"
	"os"
	"sync/atomic"
)

type Backend interface{}

const FreeUnknown int64 = -1
const GraphKVDefaultPositions = 1024

const (
	GraphKVPreallocated  = "preallocated"
	GraphKVGraphOff      = "graph-off"
	GraphKVNotEligible   = "not-capture-eligible"
	GraphKVDoesNotFit    = "prealloc-exceeds-free-device-memory"
	GraphKVEmptyGeometry = "empty-kv-geometry"
)

type KVPrecision string

const (
	KVPrecisionF32  KVPrecision = "f32"
	KVPrecisionF16  KVPrecision = "f16"
	KVPrecisionBF16 KVPrecision = "bf16"
)

func (p KVPrecision) StorageLabel() string {
	if p == "" {
		return "f32"
	}
	return string(p)
}

type KVConfig struct {
	NumLayers    int
	NumKVHeads   int
	HeadDim      int
	Precision    KVPrecision
	GraphCapture bool
}

type GraphKVPrealloc struct {
	Preallocate bool
	Positions   int
	Bytes       int64
	Reason      string
}

type MemoryClass string

const (
	MemoryWeights MemoryClass = "weights"
	MemoryKVCache MemoryClass = "kv_cache"
)

type MemoryScope string

const (
	MemoryScopeDevice MemoryScope = "device"
	MemoryScopeHost   MemoryScope = "host"
)

type MemoryDemand struct {
	Class       MemoryClass
	Bytes       int64
	Detail      string
	Scope       MemoryScope
	DType       string
	EpochDigest string
}

type MemoryPlan []MemoryDemand

type FitVerdict int

const (
	FitOK FitVerdict = iota
	FitTooBig
	FitUnknown
)

type FitError struct {
	Verdict     FitVerdict
	Want        int64
	Avail       int64
	Demands     MemoryPlan
	Scope       MemoryScope
	EpochDigest string
}

func (e *FitError) Error() string {
	return fmt.Sprintf("FitTooBig want=%d avail=%d scope=%s", e.Want, e.Avail, e.Scope)
}

type mockCapDevice struct {
	total int64
	free  int64
	known bool
}

func saturatingAddInt64(vals ...int64) int64 {
	var sum int64
	for _, v := range vals {
		if v <= 0 {
			continue
		}
		if sum > math.MaxInt64-v {
			return math.MaxInt64
		}
		sum += v
	}
	return sum
}

func saturatingMulInt64(vals ...int64) int64 {
	prod := int64(1)
	for _, v := range vals {
		if v <= 0 {
			return 0
		}
		if prod > math.MaxInt64/v {
			return math.MaxInt64
		}
		prod *= v
	}
	return prod
}

func RefuseMemoryPlanIfTooBig(b Backend, plan MemoryPlan, headroom float64) error {
	dev, ok := b.(mockCapDevice)
	if !ok || !dev.known {
		return nil
	}
	var want int64
	for _, d := range plan {
		if d.Bytes <= 0 {
			continue
		}
		if d.Scope == "" || d.Scope == MemoryScopeDevice {
			want = saturatingAddInt64(want, d.Bytes)
		}
	}
	avail := dev.free
	if avail < 0 {
		avail = dev.total
	}
	if headroom > 0 && headroom < 1 {
		avail = int64(float64(avail) * (1 - headroom))
	}
	if want > avail {
		return &FitError{Verdict: FitTooBig, Want: want, Avail: avail, Demands: append(MemoryPlan(nil), plan...), Scope: MemoryScopeDevice}
	}
	return nil
}

var (
	graphEnabled     bool
	cudaKVMaxPos     = GraphKVDefaultPositions
	cudaGraphReplays atomic.Uint64
)

func joinDeferredBackends() {}
`

const graphFitAccountingMain = `
func assert(cond bool, msg string) {
	if !cond {
		fmt.Fprintln(os.Stderr, "FAIL:", msg)
		os.Exit(1)
	}
}

func main() {
	cfg := KVConfig{NumLayers: 16, NumKVHeads: 4, HeadDim: 256, Precision: KVPrecisionF32, GraphCapture: true}
	dev := mockCapDevice{total: 24 << 30, free: 24 << 30, known: true}
	base := MemoryPlan{
		{Class: MemoryWeights, Bytes: 19 << 30, Detail: "resident-weights", Scope: MemoryScopeDevice},
		{Class: MemoryKVCache, Bytes: 96 << 20, Detail: "hal-kv-store", Scope: MemoryScopeDevice},
	}

	// 1. Graph decode is strictly opt-in: off by default even after raising KV capacity.
	assert(!CUDAGraphEnabled(), "graphEnabled must start false (opt-in)")
	assert(CUDAGraphKVCapacity() == GraphKVDefaultPositions, "initial graph KV capacity must be default 1024")
	SetCUDAGraphKVCapacity(32768)
	assert(CUDAGraphKVCapacity() == 32768, "SetCUDAGraphKVCapacity must raise capacity to 32768")
	assert(!CUDAGraphEnabled(), "SetCUDAGraphKVCapacity alone must NOT turn on graph decode")
	planOff := WithCUDAGraphKVPrealloc(base, cfg)
	assert(len(planOff) == len(base), "graph-off plan must not add graph prealloc demand")
	assert(RefuseCUDAGraphFitIfTooBig(dev, base, cfg, 0) == nil, "graph-off base plan must admit")

	// 2. Enabling graph decode keeps ineligible sessions uncharged.
	EnableCUDAGraph()
	assert(CUDAGraphEnabled(), "EnableCUDAGraph must opt in")
	ineligible := cfg
	ineligible.GraphCapture = false
	planIneligible := WithCUDAGraphKVPrealloc(base, ineligible)
	assert(len(planIneligible) == len(base), "non-GraphCapture session must not be charged graph prealloc")
	assert(RefuseCUDAGraphFitIfTooBig(dev, base, ineligible, 0) == nil, "non-GraphCapture plan must admit")

	// 3. Eligible capturing session charges delta (6 GiB - 96 MiB) and refuses pre-launch with FitTooBig.
	charged := WithCUDAGraphKVPrealloc(base, cfg)
	assert(len(charged) == len(base)+1, "graph prealloc must append exactly one cuda-graph-kv-prealloc demand")
	added := charged[len(charged)-1]
	assert(added.Class == MemoryKVCache && added.Scope == MemoryScopeDevice && added.Detail == "cuda-graph-kv-prealloc", "unexpected graph KV demand metadata")
	assert(added.Bytes == (6<<30)-(96<<20), fmt.Sprintf("delta=%d want=%d", added.Bytes, int64((6<<30)-(96<<20))))

	err := RefuseCUDAGraphFitIfTooBig(dev, base, cfg, 0)
	var fe *FitError
	assert(errors.As(err, &fe), "ooming graph fit must return *FitError pre-launch")
	assert(fe.Verdict == FitTooBig && fe.Scope == MemoryScopeDevice, "FitError must report FitTooBig on device")
	assert(fe.Want == (19<<30)+(6<<30) && fe.Avail == 24<<30, "FitError Want/Avail mismatch")

	// 4. Double application is idempotent (no double-charging of cuda-graph-kv-prealloc).
	chargedTwice := WithCUDAGraphKVPrealloc(charged, cfg)
	assert(len(chargedTwice) == len(charged), "WithCUDAGraphKVPrealloc must be idempotent")

	// 5. GDN recurrent/conv state and host-scoped caches must NOT reduce device graph KV prealloc.
	gdnBase := MemoryPlan{
		{Class: MemoryWeights, Bytes: 19 << 30, Detail: "resident-weights", Scope: MemoryScopeDevice},
		{Class: MemoryKVCache, Bytes: 2 << 30, Detail: "qwen35-gdn-recurrent-conv-state", Scope: MemoryScopeDevice},
		{Class: MemoryKVCache, Bytes: 4 << 30, Detail: "kv-host-cache", Scope: MemoryScopeHost},
	}
	gdnCharged := WithCUDAGraphKVPrealloc(gdnBase, cfg)
	assert(gdnCharged[len(gdnCharged)-1].Bytes == 6<<30, "GDN conv state and host cache must not reduce graph KV prealloc")

	// 6. Smaller weights (17 GiB + 6 GiB = 23 GiB <= 24 GiB) admit cleanly; unknown capacity fails open.
	fitBase := MemoryPlan{
		{Class: MemoryWeights, Bytes: 17 << 30, Detail: "resident-weights", Scope: MemoryScopeDevice},
		{Class: MemoryKVCache, Bytes: 96 << 20, Detail: "hal-kv-store", Scope: MemoryScopeDevice},
	}
	assert(RefuseCUDAGraphFitIfTooBig(dev, fitBase, cfg, 0) == nil, "fitting graph plan must admit")
	assert(RefuseCUDAGraphFitIfTooBig(mockCapDevice{known: false}, base, cfg, 0) == nil, "unknown capacity must fail open")
	_ = CUDAGraphReplays()
}
`

// TestDevicePoolWatermarkTrim compiles and runs the production runtime_memory.cuh idle-pool
// alloc, free, large-trim, and watermark-trim bodies against a deterministic host allocator
// and ledger shim. Verifies proactive TrimLarge(watermark) keeps pool retention within watermark
// between requests without needing an OOM/refusal trigger while guarding open captures/graph spares.
// fak-test:runtime medium est=2s lane=default
func TestDevicePoolWatermarkTrim(t *testing.T) {
	compiler, err := exec.LookPath("c++")
	if err != nil {
		t.Skip("host C++ compiler unavailable")
	}
	memory := readCUDASource(t, "cuda/runtime_memory.cuh")
	program := poolWatermarkTrimHarness + "\n" + readCUDASource(t, "cuda/v41_compressor_pending.cuh") +
		"\nstatic bool fcuda_pool_may_release(void) " + cudaFunctionBody(t, memory, "fcuda_pool_may_release") +
		"\nstatic void fcuda_pool_evict_to_cap(void) " + cudaFunctionBody(t, memory, "fcuda_pool_evict_to_cap") +
		"\nstatic void fcuda_pool_put(void *d, size_t bytes) " + cudaFunctionBody(t, memory, "fcuda_pool_put") +
		"\nextern \"C\" void fcuda_pool_set_cap(size_t cap_bytes) " + cudaFunctionBody(t, memory, "fcuda_pool_set_cap") +
		"\nextern \"C\" void fcuda_pool_cap_stats(size_t *cap, size_t *bytes, size_t *peak, unsigned long long *hits, unsigned long long *misses, unsigned long long *evictions) " + cudaFunctionBody(t, memory, "fcuda_pool_cap_stats") +
		"\nextern \"C\" void fcuda_pool_trim_to_watermark(size_t watermark_bytes) " + cudaFunctionBody(t, memory, "fcuda_pool_trim_to_watermark") +
		"\nextern \"C\" int fcuda_pool_trim_to_watermark_checked(size_t watermark_bytes) " + cudaFunctionBody(t, memory, "fcuda_pool_trim_to_watermark_checked") +
		"\nextern \"C\" void fcuda_trim_pool_large(size_t max_keep_bytes) " + cudaFunctionBody(t, memory, "fcuda_trim_pool_large") +
		"\nextern \"C\" int fcuda_trim_pool_large_checked(size_t max_keep_bytes) " + cudaFunctionBody(t, memory, "fcuda_trim_pool_large_checked") +
		"\nextern \"C\" void *fcuda_malloc_tagged(size_t bytes, int cls, int site) " + cudaFunctionBody(t, memory, "fcuda_malloc_tagged") +
		"\nextern \"C\" void fcuda_free(void *d) " + cudaFunctionBody(t, memory, "fcuda_free") +
		poolWatermarkTrimMain

	dir := t.TempDir()
	cpp := filepath.Join(dir, "pool_watermark.cpp")
	binary := filepath.Join(dir, "pool_watermark")
	if runtime.GOOS == "windows" {
		binary += ".exe"
	}
	if err := os.WriteFile(cpp, []byte(program), 0600); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(t.Context(), 60*time.Second)
	defer cancel()
	if out, err := exec.CommandContext(ctx, compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", cpp, "-o", binary).CombinedOutput(); err != nil {
		t.Fatalf("compile production pool watermark trim: %v\n%s", err, out)
	}
	if out, err := exec.CommandContext(ctx, binary).CombinedOutput(); err != nil {
		t.Fatalf("production pool watermark trim failed: %v\n%s", err, out)
	}
}

const poolWatermarkTrimHarness = `
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <unordered_map>
#include <vector>
#define FAK_ALLOC_IDLE_POOL 7
#define FAK_ALLOC_DEVICE_LOCAL 0
#define FAK_ALLOC_SITE(name) 1
#define CK(value) do { (void)(value); } while (0)
using cudaError_t = int;
static constexpr int cudaSuccess = 0;
static constexpr int cudaErrorNotReady = 600;
static const char *cudaGetErrorString(cudaError_t) { return "fake"; }
static bool g_capture_open = false, g_v41_router_uncertain = false, g_v41_mmid_uncertain = false, g_stream_uncertain = false;
static int g_cuda_cleanup_status = 0, free_status = 0;
static std::unordered_map<void *, size_t> g_live, g_managed_live, g_panel_live;
static std::unordered_map<size_t, std::vector<void *>> g_pool, g_panel_pool;
static size_t g_panel_live_bytes = 0, g_panel_cached_bytes = 0;
static size_t g_pool_bytes = 0, g_pool_peak_bytes = 0;
static size_t g_pool_cap_bytes = (size_t)512 << 20;
static unsigned long long g_pool_clock = 0, g_pool_hits = 0, g_pool_misses = 0, g_pool_evictions = 0;
static std::unordered_map<size_t, unsigned long long> g_pool_touch;
static bool g_pool_graph_deep = false;
struct LedgerRow { int cls; size_t bytes; };
static std::unordered_map<void *, LedgerRow> ledger;
static std::set<void *> device;
static std::vector<void *> freed;
static bool v41_candidates_stream_uncertain(void) { return g_stream_uncertain || g_cuda_cleanup_status != 0; }
static cudaError_t fcuda_alloc_status(cudaError_t e) { return e; }
static cudaError_t cudaMalloc(void **d, size_t bytes) {
  *d = std::malloc(bytes);
  device.insert(*d);
  return cudaSuccess;
}
static cudaError_t cudaFree(void *d) {
  if (free_status != 0) return free_status;
  device.erase(d);
  freed.push_back(d);
  std::free(d);
  return cudaSuccess;
}
static void fak_alloc_track(void *d, int cls, int, int, size_t bytes) { ledger[d] = {cls, bytes}; }
static void fak_alloc_untrack(void *d) { ledger.erase(d); }
static void fak_alloc_retag(void *d, int cls, int) { ledger[d].cls = cls; }
static void q4_coeff_cache_evict(const void *) {}
static void require(bool ok, const char *message) {
  if (!ok) { std::printf("FAIL: %s (pool=%zu cap=%zu evictions=%llu status=%d)\n", message, g_pool_bytes, g_pool_cap_bytes, g_pool_evictions, g_cuda_cleanup_status); std::exit(1); }
}
static size_t idle_ledger_bytes() {
  size_t n = 0;
  for (auto &kv : ledger) if (kv.second.cls == FAK_ALLOC_IDLE_POOL) n += kv.second.bytes;
  return n;
}
static size_t bucket_bytes() {
  size_t n = 0;
  for (auto &kv : g_pool) n += kv.first * kv.second.size();
  return n;
}
static bool pooled(void *p) {
  for (auto &kv : g_pool) for (void *q : kv.second) if (q == p) return true;
  return false;
}
static void ledger_consistent(const char *where) {
  require(g_pool_bytes == bucket_bytes(), where);
  require(idle_ledger_bytes() == g_pool_bytes, where);
}
static void reset_pool(size_t cap) {
  for (auto &kv : g_pool) for (void *p : kv.second) { device.erase(p); std::free(p); }
  g_pool.clear(); g_pool_touch.clear(); ledger.clear(); freed.clear(); g_live.clear();
  g_pool_bytes = g_pool_peak_bytes = 0; g_pool_hits = g_pool_misses = g_pool_evictions = 0;
  g_pool_cap_bytes = cap; g_capture_open = g_pool_graph_deep = g_stream_uncertain = false;
  g_cuda_cleanup_status = free_status = 0;
}
`

const poolWatermarkTrimMain = `
int main() {
  // 1. Proactive watermark trim between N requests without any OOM or refusal trigger.
  // Each request uses a distinct prefill transient <= max_keep; Trim(watermark) keeps
  // idle pool bounded <= watermark after every request.
  reset_pool(4096);
  const size_t watermark = 600;
  const size_t sizes[6] = {300, 400, 500, 350, 450, 250};
  for (int i = 0; i < 6; ++i) {
    void *buf = fcuda_malloc_tagged(sizes[i], 3, 1);
    fcuda_free(buf);
    int st = fcuda_pool_trim_to_watermark_checked(watermark);
    require(st == 0, "between-request watermark trim returned nonzero status");
    ledger_consistent("between-request watermark trim");
    require(g_pool_bytes <= watermark, "pool retained more than watermark between requests");
  }
  require(g_pool_evictions >= 4, "proactive watermark trim must have evicted older request transients");
  size_t st_cap = 0, st_bytes = 0, st_peak = 0;
  unsigned long long st_hits = 0, st_misses = 0, st_evict = 0;
  fcuda_pool_cap_stats(&st_cap, &st_bytes, &st_peak, &st_hits, &st_misses, &st_evict);
  require(st_bytes == g_pool_bytes && st_evict == g_pool_evictions && st_evict > 0, "cap_stats did not reflect watermark evictions");

  // 2. Steady-state reuse under watermark: repeated same-size decode buffers hit the pool with zero extra evictions.
  reset_pool(4096);
  void *steady = fcuda_malloc_tagged(256, 3, 1);
  fcuda_free(steady);
  unsigned long long evict_before = g_pool_evictions;
  require(fcuda_pool_trim_to_watermark_checked(512) == 0, "under-watermark trim failed");
  require(g_pool_evictions == evict_before && pooled(steady), "under-watermark buffer should not be evicted");
  void *reused = fcuda_malloc_tagged(256, 3, 1);
  require(reused == steady && g_pool_hits == 1, "steady-state small buffer failed to reuse after trim");
  fcuda_free(reused);

  // 3. LRU eviction order in watermark trim: touching bucket 100 protects it while oldest 200 leaves first.
  reset_pool(4096);
  void *a = fcuda_malloc_tagged(100, 3, 1);
  void *b = fcuda_malloc_tagged(200, 3, 1);
  void *c = fcuda_malloc_tagged(300, 3, 1);
  fcuda_free(a); fcuda_free(b); fcuda_free(c); // touches: 100 < 200 < 300
  void *a2 = fcuda_malloc_tagged(100, 3, 1);
  fcuda_free(a2); // touches: 200 < 300 < 100
  require(fcuda_pool_trim_to_watermark_checked(400) == 0, "LRU watermark trim failed");
  ledger_consistent("LRU watermark trim");
  require(g_pool_bytes == 400 && !pooled(b) && pooled(a2) && pooled(c), "LRU watermark trim evicted wrong bucket");

  // 4. Open capture defers both fcuda_trim_pool_large_checked and watermark trim; post-capture
  // fcuda_trim_pool_large_checked enforces pool cap even for buckets <= max_keep_bytes.
  reset_pool(1000);
  void *p1 = fcuda_malloc_tagged(350, 3, 1);
  void *p2 = fcuda_malloc_tagged(450, 3, 1);
  void *p3 = fcuda_malloc_tagged(500, 3, 1);
  g_capture_open = true;
  fcuda_free(p1); fcuda_free(p2); fcuda_free(p3); // 1300 > 1000, all <= max_keep=1024
  size_t frees_during_capture = freed.size();
  require(fcuda_trim_pool_large_checked(1024) == 0, "open capture large trim should be inert");
  require(fcuda_pool_trim_to_watermark_checked(0) == 0, "open capture watermark trim should be inert");
  require(freed.size() == frees_during_capture && g_pool_bytes == 1300, "trim released buffers while capture was open");
  g_capture_open = false;
  require(fcuda_trim_pool_large_checked(1024) == 0, "post-capture large trim failed");
  ledger_consistent("post-capture large trim enforces watermark");
  require(g_pool_bytes == 950 && !pooled(p1) && pooled(p2) && pooled(p3), "post-capture trim did not enforce watermark cap on <=max_keep buckets");

  // 5. Graph-deep prewarm protects spares during watermark trim, while explicit max_keep=0 flushes all.
  reset_pool(300);
  void *g1 = fcuda_malloc_tagged(200, 3, 1);
  void *g2 = fcuda_malloc_tagged(200, 3, 1);
  g_pool_graph_deep = true;
  fcuda_free(g1); fcuda_free(g2);
  size_t frees_deep = freed.size();
  require(fcuda_trim_pool_large_checked(512) == 0 && freed.size() == frees_deep && g_pool_bytes == 400, "graph prewarm spares evicted by watermark trim");
  require(fcuda_trim_pool_large_checked(0) == 0 && g_pool_bytes == 0, "explicit TrimLarge(0) failed to flush pool");

  // 6. Failed cudaFree retains pointer + ledger entry and surfaces cleanup error status cleanly.
  reset_pool(4096);
  void *bad = fcuda_malloc_tagged(512, 3, 1);
  fcuda_free(bad);
  free_status = 700;
  int err_st = fcuda_pool_trim_to_watermark_checked(0);
  require(err_st == 700 && g_cuda_cleanup_status == 700, "failed watermark trim did not return cleanup status");
  require(pooled(bad) && g_pool_bytes == 512, "failed watermark trim dropped pointer from pool");
  ledger_consistent("failed watermark trim keeps ledger");
  free_status = 0;
  require(fcuda_pool_trim_to_watermark_checked(0) == 700 && pooled(bad), "latched cleanup fault must refuse later trims");
  return 0;
}
`
