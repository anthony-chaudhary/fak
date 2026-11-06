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

// Executes the production MTP kernel, not a duplicate CPU implementation. A
// bounded cooperative SIMT model supplies barriers, shuffles and synchronous
// copy only; it cannot establish CUDA compilation, asynchronous-copy ordering
// or GPU speed. Unlike the contiguous-panel witness this harness uses portable
// OS threads, so it runs on any host with a C++14 compiler.
// fak-test:runtime medium est=15s lane=default
// fak-test:justify Compiles and executes one exact-source C++14 SIMT arithmetic/layout witness.
func TestCUDAGDNMTPVLLMPostConvHost(t *testing.T) {
	read := func(path string) string {
		t.Helper()
		b, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		return string(b)
	}
	mtp := read("cuda/gdn_mtp_decode_vllm.cuh")
	base := read("cuda/gdn_decode_vllm.cuh")
	const mtpCut = "template <int Ratio>\nstatic int launch_mtp("
	end := strings.Index(mtp, mtpCut)
	if end < 0 || strings.Count(mtp, mtpCut) != 1 {
		t.Fatal("production MTP kernel/exact eligibility boundary missing")
	}
	const baseCut = "template <int Ratio>\nstatic int launch("
	bend := strings.Index(base, baseCut)
	if bend < 0 || strings.Count(base, baseCut) != 1 {
		t.Fatal("production contiguous-panel kernel boundary missing")
	}
	const wantPass = "exact GDN MTP source, FP64 oracle, accepted-token replay and bounded dispatch; CUDA/device/performance unrun"
	dir := t.TempDir()
	mtpOut := gdnMTPCompileRun(t, dir, "gdn_mtp", cudaGDNMTPHostPreamble+mtp[:end]+"\n}\n"+cudaGDNMTPHostMain)
	baseOut := gdnMTPCompileRun(t, dir, "gdn_base", cudaGDNMTPHostPreamble+base[:bend]+"\n}\n"+cudaGDNBaseDigestMain)
	if got := gdnMTPLine(t, mtpOut, "PASS "); got != wantPass {
		t.Fatalf("incomplete host witness: PASS %s", got)
	}
	// All-8-accepted panel must equal the existing contiguous-panel path.
	if a, b := gdnMTPLine(t, mtpOut, "DIGEST-A "), gdnMTPLine(t, baseOut, "DIGEST-B "); a != b {
		t.Fatalf("all-accepted MTP panel diverges from the contiguous-panel path: %s vs %s", a, b)
	}
}

// Pins the MTP form's selection gating: env opt-in, graph-capture exclusion,
// shape eligibility and the accepted-count bound, executed as closed boolean
// expectations against the production gate plus textual caller wiring.
// fak-test:runtime fast est=3s lane=default
func TestCUDAGDNMTPVLLMContract(t *testing.T) {
	read := func(path string) string {
		t.Helper()
		b, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		return string(b)
	}
	mtp := read("cuda/gdn_mtp_decode_vllm.cuh")
	for _, want := range []string{
		"ae065918640a198a56f6f17eb45e251c3229fe41",
		"SPDX-License-Identifier: Apache-2.0",
		"csrc/gdn/fused_gdn_decode_kernel.cu",
		"fused_gdn_decode_post_conv_mtp.py",
		"cp.async.cg.shared.global", "cp.async.wait_all", "static_assert(",
		"k_qwen35_gdn_mtp_post_conv_vllm", "fak_gdn_vllm_mtp",
	} {
		if !strings.Contains(mtp, want) {
			t.Errorf("missing MTP port contract %q", want)
		}
	}
	caller := read("cuda/qwen35_gdn.cuh")
	if !strings.Contains(caller, `#include "gdn_mtp_decode_vllm.cuh"`) {
		t.Fatal("MTP port is absent from the CUDA translation unit")
	}
	body := cudaFunctionBody(t, caller, "fcuda_qwen35_gdn_mtp_post_conv_f32")
	start := strings.Index(body, `fak_gdn_vllm_mtp::eligible_mtp(getenv("FAK_CUDA_GDN_POST_CONV")`)
	if start < 0 {
		t.Fatal("MTP entry does not select through the extended gate")
	}
	arm := strings.Join(strings.Fields(body[start:]), " ")
	if !strings.Contains(arm, "g_capture_open, accepted)") ||
		!strings.Contains(arm, "return -1;") ||
		!strings.Contains(arm, "status = fak_gdn_vllm_mtp::dispatch_mtp(dConvOut, dA, dB, dALog, dDtBias,") ||
		!strings.Contains(arm, "if (status != 0) return qwen35_gdn_drain_after_error(status);") {
		t.Fatal("MTP entry lost capture admission, the decline path or checked candidate completion")
	}
	const gateCut = "template <int ValueHeadsPerKeyHead>\n__global__ __launch_bounds__(kThreads, 1) void k_qwen35_gdn_mtp_post_conv_vllm("
	gend := strings.Index(mtp, gateCut)
	if gend < 0 || strings.Count(mtp, gateCut) != 1 {
		t.Fatal("production MTP eligibility boundary missing")
	}
	out := gdnMTPCompileRun(t, t.TempDir(), "gdn_mtp_gate",
		cudaGDNMTPHostPreamble+mtp[:gend]+"\n}\n"+cudaGDNMTPGateMain)
	if strings.TrimSpace(out) != "PASS MTP selection gate booleans; CUDA/device/performance unrun" {
		t.Fatalf("incomplete MTP gate witness: %s", out)
	}
}

func gdnMTPCompiler(t *testing.T) string {
	t.Helper()
	for _, name := range []string{"c++", "g++"} {
		if p, err := exec.LookPath(name); err == nil {
			return p
		}
	}
	t.Fatal("C++14 compiler required for GDN MTP exact-source host witness")
	return ""
}

func gdnMTPBinaryPath(dir, name string) string {
	p := filepath.Join(dir, name)
	if runtime.GOOS == "windows" {
		p += ".exe"
	}
	return p
}

func gdnMTPCompileRun(t *testing.T, dir, name, source string) string {
	t.Helper()
	compiler := gdnMTPCompiler(t)
	cpp := filepath.Join(dir, name+".cpp")
	binary := gdnMTPBinaryPath(dir, name)
	if err := os.WriteFile(cpp, []byte(source), 0600); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(t.Context(), 90*time.Second)
	defer cancel()
	if out, err := exec.CommandContext(ctx, compiler, "-std=c++14", "-O1", cpp, "-o", binary).CombinedOutput(); err != nil {
		t.Fatalf("compile %s GDN source: %v\n%s", name, err, out)
	}
	out, err := exec.CommandContext(ctx, binary).CombinedOutput()
	if err != nil {
		t.Fatalf("run %s GDN source: %v\n%s", name, err, out)
	}
	return string(out)
}

func gdnMTPLine(t *testing.T, out, prefix string) string {
	t.Helper()
	for _, line := range strings.Split(out, "\n") {
		if rest, ok := strings.CutPrefix(line, prefix); ok {
			return strings.TrimSpace(rest)
		}
	}
	t.Fatalf("%q line missing from witness output:\n%s", prefix, out)
	return ""
}

const cudaGDNMTPHostPreamble = `
#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>
#if defined(_WIN32)
#include <windows.h>
#else
#include <pthread.h>
#include <sched.h>
#endif
#define __global__
#define __device__
#define __forceinline__ inline
#define __shared__ static
// A GNU attribute stays legal where the CUDA attribute sits (mid
// decl-specifiers); alignas would not.
#define __align__(n) __attribute__((aligned(n)))
#define __launch_bounds__(...)
#define __expf expf
struct dim3 { unsigned x,y,z; dim3(unsigned x=1,unsigned y=1,unsigned z=1):x(x),y(y),z(z){} };
struct alignas(16) float4 { float x,y,z,w; };
static float4 make_float4(float x,float y,float z,float w) { return {x,y,z,w}; }
// Newer libm headers declare rsqrtf as a real symbol, so keep the host
// implementation under its own name and map the kernel's intrinsic call sites
// onto it; the <cmath> include above stays unmacro'd.
static float host_rsqrtf(float x) { return 1.0f/std::sqrt(x); }
#define rsqrtf host_rsqrtf
static dim3 blockIdx;
// Per-thread id lives in thread-local storage behind the kernel's global
// threadIdx name: a plain global raced across the 256 real OS threads and
// corrupted warp indices.
namespace simt_tls {
thread_local dim3 tls_dim;
inline dim3& ref() { return tls_dim; }
}  // namespace simt_tls
#define threadIdx (simt_tls::ref())

namespace simt {
constexpr int kThreads = 256;
void yield_cpu();

// Generational barrier: all participants arrive; the last one advances the
// generation; each waiter leaves once the generation moves past its own.
class Barrier {
 public:
  explicit Barrier(int total) : total_(total), arrived_(0), generation_(0) { init_lock(); }
  Barrier(const Barrier&) = delete;
  Barrier& operator=(const Barrier&) = delete;
#if !defined(_WIN32)
  ~Barrier() { pthread_mutex_destroy(&mu_); }
#endif
  void reset() { lock(); arrived_.store(0); generation_.store(0); unlock(); }
  void wait() {
    lock();
    const long gen = generation_.load(std::memory_order_relaxed);
    const bool last = (arrived_.fetch_add(1) + 1 == total_);
    if (last) {
      arrived_.store(0, std::memory_order_relaxed);
      generation_.fetch_add(1, std::memory_order_release);
    }
    unlock();
    if (!last) {
      while (generation_.load(std::memory_order_acquire) == gen) yield_cpu();
    }
  }
  bool quiescent() const { return arrived_.load(std::memory_order_relaxed) == 0; }
 private:
  void init_lock() {
#if defined(_WIN32)
    InitializeCriticalSection(&cs_);
#else
    pthread_mutex_init(&mu_, nullptr);
#endif
  }
  void lock() {
#if defined(_WIN32)
    EnterCriticalSection(&cs_);
#else
    pthread_mutex_lock(&mu_);
#endif
  }
  void unlock() {
#if defined(_WIN32)
    LeaveCriticalSection(&cs_);
#else
    pthread_mutex_unlock(&mu_);
#endif
  }
  const int total_;
  std::atomic<long> arrived_;
  std::atomic<long> generation_;
#if defined(_WIN32)
  CRITICAL_SECTION cs_;
#else
  pthread_mutex_t mu_;
#endif
};

void yield_cpu() {
#if defined(_WIN32)
  SwitchToThread();
#else
  sched_yield();
#endif
}

static Barrier block_barrier(256);
// Named instances: a non-copyable barrier cannot be array-brace-initialized.
static Barrier warp_a(32), warp_b(32), warp_c(32), warp_d(32);
static Barrier warp_e(32), warp_f(32), warp_g(32), warp_h(32);
static Barrier* const warp_barriers[8] = {
    &warp_a, &warp_b, &warp_c, &warp_d, &warp_e, &warp_f, &warp_g, &warp_h};
static float shuffle_values[kThreads];
static std::function<void()> kernel_entry;

void syncthreads_impl() { block_barrier.wait(); }
float shfl_xor_impl(float value, int offset) {
  assert(offset > 0 && offset < 32);
  const int tid = threadIdx.x;
  shuffle_values[tid] = value;
  warp_barriers[tid / 32]->wait();
  const float other = shuffle_values[tid ^ offset];
  warp_barriers[tid / 32]->wait();
  return other;
}

struct ThreadArg { int tid; };
#if defined(_WIN32)
static DWORD WINAPI thread_main(LPVOID arg) {
  ThreadArg* a = static_cast<ThreadArg*>(arg);
  threadIdx = dim3(static_cast<unsigned>(a->tid));
  kernel_entry();
  delete a;
  return 0;
}
#else
static void* thread_main(void* arg) {
  ThreadArg* a = static_cast<ThreadArg*>(arg);
  threadIdx = dim3(static_cast<unsigned>(a->tid));
  kernel_entry();
  delete a;
  return nullptr;
}
#endif

// One block of 256 real OS threads; barriers and shuffles supply the SIMT
// coupling. Blocks run sequentially and are joined before the next launch.
void run_block() {
  block_barrier.reset();
  for (int w = 0; w < 8; ++w) warp_barriers[w]->reset();
#if defined(_WIN32)
  HANDLE handles[kThreads];
  for (int i = 0; i < kThreads; ++i)
    handles[i] = CreateThread(nullptr, 0, thread_main, new ThreadArg{i}, 0, nullptr);
  for (int i = 0; i < kThreads; ++i) {
    if (!handles[i]) std::abort();
    WaitForSingleObject(handles[i], INFINITE);
    CloseHandle(handles[i]);
  }
#else
  pthread_t ids[kThreads];
  for (int i = 0; i < kThreads; ++i)
    if (pthread_create(&ids[i], nullptr, thread_main, new ThreadArg{i})) std::abort();
  for (int i = 0; i < kThreads; ++i) pthread_join(ids[i], nullptr);
#endif
  assert(block_barrier.quiescent());
  for (int w = 0; w < 8; ++w) assert(warp_barriers[w]->quiescent());
}
}  // namespace simt

static void __syncthreads() { simt::syncthreads_impl(); }
static float __shfl_xor_sync(unsigned mask, float value, int offset) {
  assert(mask == 0xffffffffu);
  return simt::shfl_xor_impl(value, offset);
}

namespace sha256ns {
const uint32_t K[64] = {
    0x428a2b97, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
    0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
    0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
    0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380914,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
    0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990634, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
    0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84384ec7, 0x8c52e39b,
    0xd3f4d93b, 0x20849d00, 0xfd469501, 0x06c29c72};
struct Hash {
  uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                   0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  uint64_t len = 0;
  uint8_t buf[64];
  size_t buflen = 0;
  static uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
  void transform(const uint8_t* p) {
    uint32_t w[64];
    for (int i = 0; i < 16; ++i)
      w[i] = (uint32_t(p[i * 4]) << 24) | (uint32_t(p[i * 4 + 1]) << 16) |
             (uint32_t(p[i * 4 + 2]) << 8) | uint32_t(p[i * 4 + 3]);
    for (int i = 16; i < 64; ++i) {
      const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
      const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
    for (int i = 0; i < 64; ++i) {
      const uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
      const uint32_t ch = (e & f) ^ (~e & g);
      const uint32_t t1 = hh + S1 + ch + K[i] + w[i];
      const uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
      const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
      const uint32_t t2 = S0 + maj;
      hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d;
    h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
  }
  void update(const uint8_t* p, size_t n) {
    len += n;
    while (n > 0) {
      const size_t take = std::min(n, size_t(64 - buflen));
      std::memcpy(buf + buflen, p, take);
      buflen += take; p += take; n -= take;
      if (buflen == 64) { transform(buf); buflen = 0; }
    }
  }
  std::string hex() {
    uint8_t bits[8];
    for (int i = 0; i < 8; ++i) bits[i] = uint8_t((len >> (56 - 8 * i)) & 0xff);
    const uint8_t pad = 0x80;
    update(&pad, 1);
    const uint8_t zero = 0;
    while (buflen != 56) update(&zero, 1);
    update(bits, 8);
    char out[65];
    for (int i = 0; i < 8; ++i)
      std::snprintf(out + 8 * i, 9, "%06x%02x", h[i] >> 8, h[i] & 0xff);
    return std::string(out, 64);
  }
};
}  // namespace sha256ns

static std::string digest_range(const float* p, size_t n) {
  sha256ns::Hash dg;
  dg.update(reinterpret_cast<const uint8_t*>(p), n * sizeof(float));
  return dg.hex();
}

static std::vector<float> data(int n, int seed, float scale) {
  std::vector<float> out(n);
  for (int i = 0; i < n; ++i) out[i] = float((i * 37 + seed * 19) % 211 - 105) * scale;
  return out;
}

static void close_range(const char* name, const float* got, const double* want, size_t n) {
  double scale = 0;
  for (size_t i = 0; i < n; ++i) scale = std::max(scale, std::abs(want[i]));
  for (size_t i = 0; i < n; ++i)
    if (!std::isfinite(got[i]) || std::abs(double(got[i]) - want[i]) > 1e-5 * (1 + scale)) {
      std::fprintf(stderr, "%s[%zu]: %.9g != %.12g\n", name, i, got[i], want[i]);
      std::abort();
    }
}

static void expect_sentinel(const char* name, const float* got, size_t n, float sentinel) {
  for (size_t i = 0; i < n; ++i)
    if (got[i] != sentinel) {
      std::fprintf(stderr, "%s[%zu]: sentinel moved to %.9g\n", name, i, got[i]);
      std::abort();
    }
}
`

const cudaGDNMTPHostMain = `
struct MtpFixture {
  int tokens, nk, nv, width, slots;
  std::vector<float> mixed, a, b, alog, bias, z, norm, state, out;
  std::vector<int> snap;
  MtpFixture(int t, int k, int ratio, int slotCount, const std::vector<int>& snapIn)
      : tokens(t), nk(k), nv(k * ratio), width((2 * k + nv) * 128), slots(slotCount),
        mixed(data(t * width, 1, .003f)), a(data(t * nv, 2, .01f)), b(data(t * nv, 3, .01f)),
        alog(data(nv, 4, .004f)), bias(data(nv, 5, .002f)), z(data(t * nv * 128, 6, .008f)),
        norm(data(128, 7, .001f)), state(size_t(slots) * nv * 128 * 128, 123456.0f),
        out(size_t(t) * nv * 128, -54321.0f), snap(snapIn) {
    for (float& x : norm) x += 1;
    // Slot 0 holds the checkpoint carried state. Signed index: an unsigned
    // i would underflow "% 191 - 95" into a huge value for residues below 95.
    for (long i = 0; i < (long)nv * 128 * 128; ++i)
      state[i] = float(((i * 31 + 7) % 191 - 95) * 0.001f);
    // Exercise both softplus branches and distinct head gates.
    if (t * nv > 1) a[1] = 22.0f;
  }
};

struct RefResult {
  std::vector<double> state;  // [slots][nv][128][128]
  std::vector<double> out;    // [tokens][nv][128]
};

// Independent FP64 recurrence over accepted positions only. initState is
// the pool captured before the kernel launch: the kernel rewrites slot 0
// with the post-panel state, so the oracle must start from the checkpoint.
static RefResult reference(const MtpFixture& f, const float* initState) {
  RefResult r;
  r.state.assign(size_t(f.slots) * f.nv * 128 * 128, 0.0);
  r.out.assign(size_t(f.tokens) * f.nv * 128, 0.0);
  for (int h = 0; h < f.nv; ++h) {
    const int kh = h / (f.nv / f.nk);
    double carry[128][128];
    for (int i = 0; i < 128; ++i)
      for (int v = 0; v < 128; ++v)
        carry[i][v] = initState[size_t(h) * 128 * 128 + i * 128 + v];
    for (int t = 0; t < f.tokens; ++t) {
      if (f.snap[t] == 0) continue;
      double q[128], k[128], qss = 0, kss = 0;
      for (int i = 0; i < 128; ++i) {
        q[i] = f.mixed[t * f.width + kh * 128 + i];
        k[i] = f.mixed[t * f.width + f.nk * 128 + kh * 128 + i];
        qss += q[i] * q[i]; kss += k[i] * k[i];
      }
      for (int i = 0; i < 128; ++i) {
        q[i] /= std::sqrt(qss + 1e-6) * std::sqrt(128.0);
        k[i] /= std::sqrt(kss + 1e-6);
      }
      const double gate = double(f.a[t * f.nv + h]) + f.bias[h];
      const double decay =
          std::exp(-std::exp(double(f.alog[h])) * (gate > 20 ? gate : std::log1p(std::exp(gate))));
      const double beta = 1 / (1 + std::exp(-double(f.b[t * f.nv + h])));
      double raw[128], ss = 0;
      for (int v = 0; v < 128; ++v) {
        double hk = 0, hq = 0;
        for (int i = 0; i < 128; ++i) {
          double sd = carry[i][v] * decay;
          carry[i][v] = sd;
          hk += sd * k[i];
        }
        const double delta =
            (f.mixed[t * f.width + 2 * f.nk * 128 + h * 128 + v] - hk) * beta;
        for (int i = 0; i < 128; ++i) {
          carry[i][v] += k[i] * delta;
          hq += carry[i][v] * q[i];
        }
        raw[v] = hq; ss += hq * hq;
      }
      const double inv = 1 / std::sqrt(ss / 128 + 1e-6);
      for (int v = 0; v < 128; ++v) {
        const int o = (t * f.nv + h) * 128 + v;
        const double z = f.z[o];
        r.out[o] = raw[v] * inv * f.norm[v] * z / (1 + std::exp(-z));
      }
      for (int i = 0; i < 128; ++i)
        for (int v = 0; v < 128; ++v)
          r.state[(size_t(f.snap[t]) * f.nv + h) * 128 * 128 + i * 128 + v] = carry[i][v];
    }
    for (int i = 0; i < 128; ++i)
      for (int v = 0; v < 128; ++v)
        r.state[size_t(h) * 128 * 128 + i * 128 + v] = carry[i][v];
  }
  return r;
}

template <int Ratio>
static void run_mtp(MtpFixture& f) {
  simt::kernel_entry = [&] {
    fak_gdn_vllm_mtp::k_qwen35_gdn_mtp_post_conv_vllm<Ratio>(
        f.mixed.data(), f.a.data(), f.b.data(), f.alog.data(), f.bias.data(),
        f.state.data(), f.z.data(), f.norm.data(), f.out.data(),
        f.tokens, f.nk, f.nv, f.width, 1e-6f, f.snap.data());
  };
  for (int h = 0; h < f.nv; ++h) { blockIdx = dim3(h); simt::run_block(); }
}

int main() {
  // (a) All-8-accepted panel equals the existing contiguous-panel path.
  {
    MtpFixture f(8, 1, 1, 9, {1, 2, 3, 4, 5, 6, 7, 8});
    assert(reinterpret_cast<uintptr_t>(f.state.data()) % 16 == 0);
    std::vector<float> pre(f.state.begin(), f.state.end());
    run_mtp<1>(f);
    RefResult r = reference(f, pre.data());
    close_range("A carried state", f.state.data(), r.state.data(), 128 * 128);
    for (int s = 1; s <= 8; ++s)
      close_range("A snapshot", f.state.data() + size_t(s) * 128 * 128,
                  r.state.data() + size_t(s) * 128 * 128, 128 * 128);
    close_range("A outputs", f.out.data(), r.out.data(), 8 * 128);
    sha256ns::Hash dg;
    dg.update(reinterpret_cast<const uint8_t*>(f.state.data()), 128 * 128 * sizeof(float));
    dg.update(reinterpret_cast<const uint8_t*>(f.out.data()), 8 * 128 * sizeof(float));
    std::printf("DIGEST-A %s\n", dg.hex().c_str());
  }
  // (b) Acceptance pattern {0,2,5} of 8: rejected slots stay untouched.
  {
    MtpFixture f(8, 1, 1, 9, {1, 0, 3, 0, 0, 6, 0, 0});
    std::vector<float> pre(f.state.begin(), f.state.end());
    run_mtp<1>(f);
    RefResult r = reference(f, pre.data());
    close_range("B carried state", f.state.data(), r.state.data(), 128 * 128);
    for (int s : {1, 3, 6})
      close_range("B snapshot", f.state.data() + size_t(s) * 128 * 128,
                  r.state.data() + size_t(s) * 128 * 128, 128 * 128);
    for (int s : {2, 4, 5, 7, 8})
      expect_sentinel("B rejected slot", f.state.data() + size_t(s) * 128 * 128,
                      128 * 128, 123456.0f);
    for (int t : {0, 2, 5})
      close_range("B output", f.out.data() + size_t(t) * 128,
                  r.out.data() + size_t(t) * 128, 128);
    for (int t : {1, 3, 4, 6, 7})
      expect_sentinel("B rejected output", f.out.data() + size_t(t) * 128, 128, -54321.0f);
  }
  // (c) Single accepted token after a reject run replays from the checkpoint.
  {
    MtpFixture f(4, 1, 1, 5, {0, 0, 0, 1});
    std::vector<float> pre(f.state.begin(), f.state.end());
    run_mtp<1>(f);
    RefResult r = reference(f, pre.data());
    close_range("C carried state", f.state.data(), r.state.data(), 128 * 128);
    close_range("C snapshot", f.state.data() + 128 * 128,
                r.state.data() + 128 * 128, 128 * 128);
    for (int s : {2, 3, 4})
      expect_sentinel("C untouched slot", f.state.data() + size_t(s) * 128 * 128,
                      128 * 128, 123456.0f);
    close_range("C output", f.out.data() + 3 * 128, r.out.data() + 3 * 128, 128);
    expect_sentinel("C rejected outputs", f.out.data(), 3 * 128, -54321.0f);
  }
  // Zero accepted: no-op launch touches neither state nor outputs.
  {
    MtpFixture f(8, 1, 1, 9, {0, 0, 0, 0, 0, 0, 0, 0});
    std::vector<float> checkpoint(f.state.begin(), f.state.begin() + 128 * 128);
    run_mtp<1>(f);
    for (size_t i = 0; i < 128 * 128; ++i)
      if (f.state[i] != checkpoint[i]) {
        std::fprintf(stderr, "D checkpoint[%zu] moved: %.9g -> %.9g\n", i, checkpoint[i], f.state[i]);
        std::abort();
      }
    for (int s = 1; s <= 8; ++s)
      expect_sentinel("D untouched slot", f.state.data() + size_t(s) * 128 * 128,
                      128 * 128, 123456.0f);
    expect_sentinel("D outputs", f.out.data(), 8 * 128, -54321.0f);
  }
  // Multi-head-ratio replay with a middle rejection.
  {
    MtpFixture f(3, 2, 4, 4, {1, 0, 3});
    std::vector<float> pre(f.state.begin(), f.state.end());
    run_mtp<4>(f);
    RefResult r = reference(f, pre.data());
    close_range("E carried state", f.state.data(), r.state.data(), 8 * 128 * 128);
    for (int s : {1, 3})
      close_range("E snapshot", f.state.data() + size_t(s) * 8 * 128 * 128,
                  r.state.data() + size_t(s) * 8 * 128 * 128, 8 * 128 * 128);
    expect_sentinel("E rejected slot", f.state.data() + 2 * 8 * 128 * 128,
                    8 * 128 * 128, 123456.0f);
    for (int t : {0, 2})
      close_range("E output", f.out.data() + size_t(t) * 8 * 128,
                  r.out.data() + size_t(t) * 8 * 128, 8 * 128);
    expect_sentinel("E rejected output", f.out.data() + 8 * 128, 8 * 128, -54321.0f);
  }
  std::puts("PASS exact GDN MTP source, FP64 oracle, accepted-token replay and bounded dispatch; CUDA/device/performance unrun");
  return 0;
}
`

const cudaGDNBaseDigestMain = `
// Runs the existing contiguous-panel kernel over the same all-8 fixture as
// the MTP witness scenario (a) and prints the carried-state/output digest.
struct BaseFixture {
  int tokens = 8, nk = 1, nv = 1, width = (2 + 1) * 128;
  std::vector<float> mixed, a, b, alog, bias, z, norm, state, out;
  BaseFixture()
      : mixed(data(tokens * width, 1, .003f)), a(data(tokens * nv, 2, .01f)),
        b(data(tokens * nv, 3, .01f)), alog(data(nv, 4, .004f)),
        bias(data(nv, 5, .002f)), z(data(tokens * nv * 128, 6, .008f)),
        norm(data(128, 7, .001f)), state(nv * 128 * 128, 0.0f),
        out(tokens * nv * 128, 0.0f) {
    for (float& x : norm) x += 1;
    for (long i = 0; i < (long)state.size(); ++i)
      state[i] = float(((i * 31 + 7) % 191 - 95) * 0.001f);
    a[1] = 22.0f;
  }
};

int main() {
  BaseFixture f;
  assert(reinterpret_cast<uintptr_t>(f.state.data()) % 16 == 0);
  simt::kernel_entry = [&] {
    fak_gdn_vllm::k_qwen35_gdn_post_conv_vllm<1>(
        f.mixed.data(), f.a.data(), f.b.data(), f.alog.data(), f.bias.data(),
        f.state.data(), f.z.data(), f.norm.data(), f.out.data(),
        f.tokens, f.nk, f.nv, f.width, 1e-6f);
  };
  for (int h = 0; h < f.nv; ++h) { blockIdx = dim3(h); simt::run_block(); }
  sha256ns::Hash dg;
  dg.update(reinterpret_cast<const uint8_t*>(f.state.data()), 128 * 128 * sizeof(float));
  dg.update(reinterpret_cast<const uint8_t*>(f.out.data()), 8 * 128 * sizeof(float));
  std::printf("DIGEST-B %s\n", dg.hex().c_str());
  return 0;
}
`

const cudaGDNMTPGateMain = `
int main() {
  using fak_gdn_vllm_mtp::eligible_mtp;
  // Env opt-in.
  assert(eligible_mtp("vllm", 8, 1, 1, 128, 128, false, 8));
  assert(!eligible_mtp(nullptr, 1, 1, 1, 128, 128, false, 1));
  assert(!eligible_mtp("", 1, 1, 1, 128, 128, false, 1));
  assert(!eligible_mtp("vllmx", 1, 1, 1, 128, 128, false, 1));
  // An open graph capture refuses MTP selection.
  assert(!eligible_mtp("vllm", 1, 1, 1, 128, 128, true, 1));
  // Token counts outside 1..8 refuse.
  for (int t : {0, 9}) assert(!eligible_mtp("vllm", t, 1, 1, 128, 128, false, 0));
  // Head ratios beyond 1,2,3,4,8 refuse; eligible ratios admit.
  for (int r : {5, 6, 7}) assert(!eligible_mtp("vllm", 1, 1, r, 128, 128, false, 1));
  for (int r : {1, 2, 3, 4, 8}) assert(eligible_mtp("vllm", 1, 2, 2 * r, 128, 128, false, 1));
  // Dimension and divisibility refusals.
  assert(!eligible_mtp("vllm", 1, 1, 1, 127, 128, false, 1));
  assert(!eligible_mtp("vllm", 1, 1, 1, 128, 129, false, 1));
  assert(!eligible_mtp("vllm", 1, 0, 1, 128, 128, false, 1));
  assert(!eligible_mtp("vllm", 1, 2, 3, 128, 128, false, 1));
  // Accepted-count bound: 0..tokens admits, outside refuses.
  assert(eligible_mtp("vllm", 8, 1, 1, 128, 128, false, 0));
  assert(!eligible_mtp("vllm", 8, 1, 1, 128, 128, false, -1));
  assert(!eligible_mtp("vllm", 8, 1, 1, 128, 128, false, 9));
  assert(!eligible_mtp("vllm", 3, 1, 1, 128, 128, false, 4));
  std::puts("PASS MTP selection gate booleans; CUDA/device/performance unrun");
  return 0;
}
`
