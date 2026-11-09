package compute

import (
	"context"
	"crypto/sha256"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// Runs the real kernel/dispatcher against a bounded host SIMT shuffle/barrier
// model and the retained old source. Host finite rounding and special-value
// inputs are checked; nonfinite casts, NVCC, CUDA numerics and speed are not.
// fak-test:runtime medium est=3s lane=default
// fak-test:justify Compiles one source-extracted host C++14 reduction/dispatch witness.
func TestCUDAQ8ActivationWarpReduction(t *testing.T) {
	compiler, err := exec.LookPath("c++")
	if err != nil {
		t.Skip("host C++ compiler unavailable; Q8 warp witness unrun")
	}
	b, err := os.ReadFile("cuda/quant_matmul.cuh")
	if err != nil {
		t.Fatal(err)
	}
	source := string(b)
	t.Logf("production source sha256=%x", sha256.Sum256(b))
	extract := func(text, signature string) string {
		t.Helper()
		if strings.Count(text, signature) != 1 {
			t.Fatalf("expected one definition %q", signature)
		}
		start := strings.Index(text, signature)
		open := strings.Index(text[start:], "{")
		if open < 0 {
			t.Fatalf("missing body %q", signature)
		}
		depth := 0
		for end := start + open; end < len(text); end++ {
			switch text[end] {
			case '{':
				depth++
			case '}':
				depth--
				if depth == 0 {
					return text[start : end+1]
				}
			}
		}
		t.Fatalf("unterminated %q", signature)
		return ""
	}
	caller := func(text string) string {
		t.Helper()
		start := strings.Index(text, "static signed char *g_q8_qX = nullptr;")
		end := strings.Index(text, "// q8_rows_matmul")
		if start < 0 || end <= start {
			t.Fatal("real Q8 scratch/caller boundaries missing")
		}
		return text[start:end]
	}
	round := extract(source, "__device__ signed char q8round_dev(")
	round = strings.Replace(round, "q8round_dev(", "q8round_real(", 1)
	oldCaller, newCaller := cudaQ8WarpBaselineCaller, caller(source)
	replaceLaunch := func(text, threads string) string {
		t.Helper()
		old := "k_q8_quant_act<<<dim3(nblk, P), " + threads + ", 0, g_stream>>>(dX, g_q8_qX, g_q8_xScale, P, in, block)"
		if strings.Count(text, old) != 1 {
			t.Fatal("real Q8 launch changed or uncovered")
		}
		text = strings.Replace(text, old, "host_launch(k_q8_quant_act, dim3(nblk, P), "+threads+", g_stream, dX, g_q8_qX, g_q8_xScale, P, in, block)", 1)
		if strings.Contains(text, "<<<") {
			t.Fatal("uncovered launch")
		}
		return text
	}
	oldCaller = replaceLaunch(oldCaller, "64")
	newCaller = replaceLaunch(newCaller, "quant_threads")
	if !strings.Contains(newCaller, "const int quant_threads = block == 32 ? 32 : 64;") {
		t.Fatal("real block32 dispatch missing")
	}
	kernel := extract(source, "__global__ void k_q8_quant_act(")
	helper := extract(source, "static __device__ __forceinline__ float q8_activation_warp_max(")
	fixture := cudaQ8WarpPreamble + "\nnamespace baseline {\n" + round + cudaQ8WarpRoundWrapper + cudaQ8WarpBaselineKernel + oldCaller + "\n}\n" +
		"namespace candidate {\n" + round + cudaQ8WarpRoundWrapper + helper + kernel + newCaller + "\n}\n" + cudaQ8WarpMain
	t.Logf("extracted fixture sha256=%x", sha256.Sum256([]byte(fixture)))
	dir := t.TempDir()
	cpp, binary := filepath.Join(dir, "q8_warp.cpp"), filepath.Join(dir, "q8_warp")
	if err := os.WriteFile(cpp, []byte(fixture), 0600); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
	defer cancel()
	if out, err := exec.CommandContext(ctx, compiler, "-std=c++14", "-O0", "-pthread", cpp, "-o", binary).CombinedOutput(); err != nil {
		t.Fatalf("compile actual Q8 source: %v\n%s", err, out)
	}
	run, cancelRun := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancelRun()
	out, err := exec.CommandContext(run, binary).CombinedOutput()
	if err != nil {
		t.Fatalf("actual Q8 reduction/dispatch: %v\n%s", err, out)
	}
	const want = "PASS actual Q8 block32 warp max/dispatch and retained shared fallback; CUDA/device/performance unrun\n"
	if strings.ReplaceAll(string(out), "\r\n", "\n") != want {
		t.Fatalf("incomplete witness: %s", out)
	}
	t.Log(strings.TrimSpace(string(out)))
}

const cudaQ8WarpPreamble = `
#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#define __global__
#define __device__
#define __forceinline__ inline
#define __shared__ static
struct dim3 { unsigned x,y,z; dim3(unsigned x=1,unsigned y=1,unsigned z=1):x(x),y(y),z(z){} };
static dim3 blockIdx,blockDim;
static thread_local dim3 threadIdx;
static thread_local int round_call;
static int round_base;
static std::vector<uint32_t> rounding_inputs;
static std::vector<unsigned char> rounding_seen;
static uint32_t bits(float x) { uint32_t u; std::memcpy(&u,&x,sizeof(u)); return u; }
static float frombits(uint32_t u) { float x; std::memcpy(&x,&u,sizeof(x)); return x; }
struct Barrier {
  explicit Barrier(int n):total(n){}
  std::mutex mu; std::condition_variable cv;
  int total,arrived=0,generation=0;
  void wait() {
    std::unique_lock<std::mutex> lock(mu); const int old=generation;
    if (++arrived==total) { arrived=0; ++generation; lock.unlock(); cv.notify_all(); }
    else cv.wait(lock,[&]{return generation!=old;});
  }
};
static std::unique_ptr<Barrier> barrier;
static float shuffle_values[64];
static std::atomic<int> block_barriers{0},shuffles{0};
static int launched_threads=0,launches=0,timing_begin=0,timing_end=0,mallocs=0,frees=0,tracks=0;
static void __syncthreads() { ++block_barriers; barrier->wait(); }
static float __shfl_xor_sync(unsigned mask,float value,int offset,int width) {
  assert(mask==0xffffffffu && width==32 && blockDim.x==32 && offset>0 && offset<32);
  ++shuffles; shuffle_values[threadIdx.x]=value; barrier->wait();
  const float other=shuffle_values[threadIdx.x^offset]; barrier->wait(); return other;
}
static int g_stream=19;
using cudaError_t=int;
static constexpr int cudaSuccess=0,FCUDA_SCRATCH_ALLOC_FAILED=80001;
static inline cudaError_t cudaGetLastError() { return cudaSuccess; }
static constexpr int FAK_ALLOC_ACTIVATIONS_SCRATCH=1,FAK_ALLOC_DEVICE_LOCAL=2;
#define FAK_ALLOC_SITE(x) (x)
template<class T> static cudaError_t cudaMalloc(T **p,size_t n) { ++mallocs; *p=static_cast<T *>(std::malloc(n)); return *p?0:2; }
static cudaError_t cudaFree(void *p) { ++frees; std::free(p); return 0; }
static void fak_alloc_untrack(void *) {}
static void fak_alloc_track(void *,int,int,const char *,size_t) { ++tracks; }
#define FCUDA_TS(name, grid, ...) do { assert(std::strcmp(name,"k_q8_quant_act")==0); ++timing_begin; __VA_ARGS__; ++timing_end; } while(0)
template<class Kernel> static void host_launch(Kernel kernel,dim3 grid,int threads,int stream,
    const float *x,signed char *q,float *scales,int P,int in,int block) {
  assert(stream==g_stream && (threads==32 || threads==64)); ++launches;
  for (unsigned row=0;row<grid.y;++row) for (unsigned col=0;col<grid.x;++col) {
    blockIdx=dim3(col,row); blockDim=dim3(threads); round_base=(int)row*in+(int)col*block;
    barrier.reset(new Barrier(threads)); launched_threads+=threads;
    std::vector<std::thread> workers;
    for (int lane=0;lane<threads;++lane) workers.emplace_back([&,lane] {
      threadIdx=dim3(lane); round_call=0; kernel(x,q,scales,P,in,block);
    });
    for (auto &worker:workers) worker.join();
  }
}
static signed char record_round(float x,signed char (*native_round)(float)) {
  const int i=round_base+(int)threadIdx.x+round_call++*(int)blockDim.x;
  assert(i>=0 && i<(int)rounding_inputs.size()); rounding_inputs[i]=bits(x); rounding_seen[i]=1;
  // C++ casts of NaN/Inf are not a CUDA oracle. Retain their exact input bits
  // and use a deterministic fake result only for those non-finite inputs.
  return std::isfinite(x)?native_round(x):(signed char)17;
}
struct Result {
  std::vector<signed char> codes; std::vector<uint32_t> scales,inputs;
  std::vector<unsigned char> seen;
  int threads,barriers,shuffle_count;
};
static void reset_observation(int count) {
  rounding_inputs.assign(count,0); rounding_seen.assign(count,0);
  block_barriers=0; shuffles=0; launched_threads=0; launches=0;
  timing_begin=0; timing_end=0; mallocs=0; frees=0; tracks=0;
}

`

const cudaQ8WarpRoundWrapper = `
static signed char q8round_dev(float x) { return record_round(x,q8round_real); }

`

const cudaQ8WarpBaselineKernel = `
__global__ void k_q8_quant_act(const float *X, signed char *qX, float *xScale,
                               int P, int in, int block) {
  int b = blockIdx.x, t = blockIdx.y;
  int nblk = in / block;
  if (t >= P || b >= nblk) return;
  const float *xb = X + (size_t)t * in + (size_t)b * block;
  __shared__ float red[64];
  float a = 0.f;
  for (int i = threadIdx.x; i < block; i += blockDim.x) a = fmaxf(a, fabsf(xb[i]));
  red[threadIdx.x] = a;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
    __syncthreads();
  }
  float d = red[0] / 127.f;
  if (threadIdx.x == 0) xScale[(size_t)t * nblk + b] = d;
  float inv = d > 0.f ? 1.f / d : 0.f;
  signed char *qb = qX + (size_t)t * in + (size_t)b * block;
  for (int i = threadIdx.x; i < block; i += blockDim.x)
    qb[i] = d > 0.f ? q8round_dev(xb[i] * inv) : (signed char)0;
}
`

const cudaQ8WarpBaselineCaller = `
static signed char *g_q8_qX = nullptr;
static int g_q8_qX_cap = 0; // bytes
static float *g_q8_xScale = nullptr;
static int g_q8_xScale_cap = 0; // floats
// A failed retirement leaves ownership uncertain. Preserve its pointer and
// ledger entry, and refuse all later uses (including smaller shapes) until exit.
#define FCUDA_Q8_SCRATCH_RETIRE_FAILED_BASE 92000
static int g_q8_scratch_retire_status = 0;
// q8_quant_rows grows the activation scratch, then quantizes every row of X on its own. Returns 0
// or an allocation/retirement failure before any launch.
static int q8_quant_rows(const float *dX, int in, int P, int block) {
#ifdef FAK_CUDA_V41_CHECKED_STREAM
  if (v41_candidates_stream_uncertain()) return 90003;
#endif
  if (g_q8_scratch_retire_status) return g_q8_scratch_retire_status;
  int nblk = in / block;
  int needQX = P * in;          // int8 activation codes
  int needScale = P * nblk;     // per-block act scales (floats)
  if (needQX > g_q8_qX_cap) {
    if (g_q8_qX) {
      const cudaError_t retired = cudaFree(g_q8_qX);
      if (retired != cudaSuccess) {
        g_q8_scratch_retire_status = FCUDA_Q8_SCRATCH_RETIRE_FAILED_BASE + (int)retired;
        return g_q8_scratch_retire_status;
      }
      fak_alloc_untrack(g_q8_qX);
    }
    g_q8_qX = nullptr;
    g_q8_qX_cap = 0;
    if (cudaMalloc(&g_q8_qX, (size_t)needQX) != cudaSuccess || !g_q8_qX) {
      g_q8_qX = nullptr;
      return FCUDA_SCRATCH_ALLOC_FAILED;
    }
    fak_alloc_track(g_q8_qX, FAK_ALLOC_ACTIVATIONS_SCRATCH, FAK_ALLOC_DEVICE_LOCAL,
                    FAK_ALLOC_SITE("q8-activation-codes"), (size_t)needQX);
    g_q8_qX_cap = needQX;
  }
  if (needScale > g_q8_xScale_cap) {
    if (g_q8_xScale) {
      const cudaError_t retired = cudaFree(g_q8_xScale);
      if (retired != cudaSuccess) {
        g_q8_scratch_retire_status = FCUDA_Q8_SCRATCH_RETIRE_FAILED_BASE + (int)retired;
        return g_q8_scratch_retire_status;
      }
      fak_alloc_untrack(g_q8_xScale);
    }
    g_q8_xScale = nullptr;
    g_q8_xScale_cap = 0;
    if (cudaMalloc(&g_q8_xScale, (size_t)needScale * sizeof(float)) != cudaSuccess || !g_q8_xScale) {
      g_q8_xScale = nullptr;
      return FCUDA_SCRATCH_ALLOC_FAILED;
    }
    fak_alloc_track(g_q8_xScale, FAK_ALLOC_ACTIVATIONS_SCRATCH, FAK_ALLOC_DEVICE_LOCAL,
                    FAK_ALLOC_SITE("q8-activation-scales"), (size_t)needScale * sizeof(float));
    g_q8_xScale_cap = needScale;
  }
  FCUDA_TS("k_q8_quant_act", dim3(nblk, P), k_q8_quant_act<<<dim3(nblk, P), 64, 0, g_stream>>>(dX, g_q8_qX, g_q8_xScale, P, in, block));
  return 0;
}

`

const cudaQ8WarpMain = `
static Result exercise(bool old,const std::vector<float> &x,int in,int P,int block,int forced_threads=0) {
  reset_observation((int)x.size());
  int status;
  if (!forced_threads) status=old?baseline::q8_quant_rows(x.data(),in,P,block):candidate::q8_quant_rows(x.data(),in,P,block);
  else {
    // Actual shared fallback must still handle an unexpected 64-thread block32 launch.
    assert(!old && forced_threads==64 && block==32);
    candidate::g_q8_qX=(signed char *)std::malloc(x.size());
    candidate::g_q8_xScale=(float *)std::malloc((size_t)P*(in/block)*sizeof(float));
    host_launch(candidate::k_q8_quant_act,dim3(in/block,P),forced_threads,g_stream,
      x.data(),candidate::g_q8_qX,candidate::g_q8_xScale,P,in,block); status=0;
  }
  assert(status==0 && launches==1);
  if (!forced_threads) assert(mallocs==2 && frees==0 && tracks==2 && timing_begin==1 && timing_end==1);
  signed char *q=old?baseline::g_q8_qX:candidate::g_q8_qX;
  float *d=old?baseline::g_q8_xScale:candidate::g_q8_xScale;
  Result r; r.codes.assign(q,q+x.size());
  for (int i=0;i<P*(in/block);++i) r.scales.push_back(bits(d[i]));
  r.inputs=rounding_inputs; r.seen=rounding_seen; r.threads=launched_threads;
  r.barriers=block_barriers; r.shuffle_count=shuffles;
  std::free(q); std::free(d);
  if (old) { baseline::g_q8_qX=nullptr; baseline::g_q8_xScale=nullptr; baseline::g_q8_qX_cap=baseline::g_q8_xScale_cap=0; }
  else { candidate::g_q8_qX=nullptr; candidate::g_q8_xScale=nullptr; candidate::g_q8_qX_cap=candidate::g_q8_xScale_cap=0; }
  return r;
}
static void compare(const std::vector<float> &x,int in,int P,int block,bool all_neutral=false) {
  const Result a=exercise(true,x,in,P,block), b=exercise(false,x,in,P,block);
  assert(a.scales==b.scales && a.codes==b.codes && a.inputs==b.inputs && a.seen==b.seen);
  const int groups=P*(in/block),threads=block==32?32:64;
  assert(a.threads==groups*64 && a.barriers==groups*64*7 && a.shuffle_count==0);
  assert(b.threads==groups*threads && b.barriers==(block==32?0:groups*64*7));
  assert(b.shuffle_count==(block==32?groups*32*5:0));
  if (all_neutral) {
    assert(std::all_of(b.scales.begin(),b.scales.end(),[](uint32_t v){return v==0;}));
    assert(std::all_of(b.codes.begin(),b.codes.end(),[](signed char v){return v==0;}));
    assert(std::all_of(b.seen.begin(),b.seen.end(),[](unsigned char v){return v==0;}));
  }
  if (block==32 && P==1) {
    const Result fallback=exercise(false,x,in,P,block,64);
    assert(a.scales==fallback.scales && a.codes==fallback.codes && a.inputs==fallback.inputs && a.seen==fallback.seen);
    assert(fallback.threads==groups*64 && fallback.barriers==groups*64*7 && fallback.shuffle_count==0);
  }
}
int main() {
  std::vector<float> finite(2*96);
  for (size_t i=0;i<finite.size();++i) finite[i]=(float)((int)(i*13%65)-32)*0.25f;
  compare(finite,96,2,32);
  std::vector<float> zero(32,0.0f); zero[3]=-0.0f; compare(zero,32,1,32,true);
  std::vector<float> nan(32,frombits(0x7fc12345)); compare(nan,32,1,32,true);
  std::vector<float> special(32,1.0f);
  special[0]=frombits(0x7fc12345); special[1]=frombits(0x7f800000);
  special[2]=frombits(0xff800000); special[3]=-0.0f; special[4]=frombits(1);
  compare(special,32,1,32);
  std::vector<float> small(32); for(int i=0;i<32;++i) small[i]=(float)(i-16)/8.0f;
  compare(small,32,1,16);
  std::vector<float> wide(64); for(int i=0;i<64;++i) wide[i]=(float)(i-32)/16.0f;
  compare(wide,64,1,64);
  std::puts("PASS actual Q8 block32 warp max/dispatch and retained shared fallback; CUDA/device/performance unrun");
}

`
