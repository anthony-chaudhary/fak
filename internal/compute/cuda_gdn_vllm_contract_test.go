package compute

import (
	"context"
	"crypto/sha256"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

// fak-test:runtime fast est=1ms lane=default
func TestCUDAGDNVLLMPostConvContract(t *testing.T) {
	read := func(path string) string {
		t.Helper()
		b, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		return string(b)
	}
	kernel := read("cuda/gdn_decode_vllm.cuh")
	caller := read("cuda/qwen35_gdn.cuh")
	for _, want := range []string{
		"fa3b9d1377219247060d3c8b49eae199c9e7e154", "SPDX-License-Identifier: Apache-2.0",
		"cp.async.cg.shared.global", "cp.async.wait_all", "static_assert(",
		"tile[value][key] = canonical[key][value]", "head_state + key * kDim + chunk * kChunkV + value",
		"status = qwen35_gdn_launch_status(5));", "k_qwen35_gdn_post_conv_vllm",
	} {
		if !strings.Contains(kernel, want) {
			t.Errorf("missing port contract %q", want)
		}
	}
	if !strings.Contains(caller, `#include "gdn_decode_vllm.cuh"`) {
		t.Fatal("port is absent from the CUDA translation unit")
	}
	for _, entry := range []string{"fcuda_qwen35_gdn_decode_f32", "fcuda_qwen35_gdn_sequence_f32"} {
		body := cudaFunctionBody(t, caller, entry)
		start := strings.Index(body, `fak_gdn_vllm::eligible(getenv("FAK_CUDA_GDN_POST_CONV")`)
		if start < 0 {
			t.Fatalf("%s does not select the port", entry)
		}
		arm := body[start:]
		end := strings.Index(arm, "} else {")
		if end < 0 {
			t.Fatalf("%s lost the existing native fallback", entry)
		}
		arm = strings.Join(strings.Fields(arm[:end]), " ")
		if !strings.Contains(arm, "kHd, vHd, g_capture_open)") ||
			!strings.Contains(arm, "status = fak_gdn_vllm::dispatch(dConvOut, dA, dB, dALog, dDtBias,") ||
			!strings.Contains(arm, "rmsEps); if (status != 0) return qwen35_gdn_drain_after_error(status);") {
			t.Fatalf("%s lost capture admission or checked candidate completion", entry)
		}
	}
	if !strings.Contains(read("cuda_kernels.go"), "C.fcuda_qwen35_gdn_sequence_f32(") ||
		!strings.Contains(read("cuda_specialized.go"), "C.fcuda_qwen35_gdn_decode_f32(") {
		t.Fatal("normal model/decode entry does not reach the port's whole-operation seam")
	}
	notice := read("../../third_party/vllm-gdn-decode/NOTICE")
	if !strings.Contains(notice, "fa3b9d1377219247060d3c8b49eae199c9e7e154") ||
		!strings.Contains(read("../../third_party/vllm-gdn-decode/LICENSE"), "Apache License") {
		t.Fatal("pinned upstream license or adaptation notice missing")
	}
}

// Executes the production kernel, not a duplicate CPU implementation. A bounded
// cooperative SIMT model supplies barriers, shuffles and synchronous copy only;
// it cannot establish CUDA compilation, asynchronous-copy ordering or GPU speed.
// fak-test:runtime medium est=5s lane=default
// fak-test:justify Compiles and executes one exact-source C++14 SIMT arithmetic/layout witness.
func TestCUDAGDNVLLMPostConvHost(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("exact-source cooperative SIMT witness requires Linux ucontext")
	}
	compiler, err := exec.LookPath("c++")
	if err != nil {
		t.Fatal("C++14 compiler required for GDN exact-source host witness: ", err)
	}
	b, err := os.ReadFile("cuda/gdn_decode_vllm.cuh")
	if err != nil {
		t.Fatal(err)
	}
	source := string(b)
	end := strings.Index(source, "template <int Ratio>\nstatic int launch(")
	if end < 0 || strings.Count(source, "template <int Ratio>\nstatic int launch(") != 1 {
		t.Fatal("production kernel/exact eligibility boundary missing")
	}
	fixture := cudaGDNVLLMHostPreamble + source[:end] + "\n}\n" + cudaGDNVLLMHostMain
	t.Logf("production source sha256=%x", sha256.Sum256(b))
	dir := t.TempDir()
	cpp, binary := filepath.Join(dir, "gdn_vllm.cpp"), filepath.Join(dir, "gdn_vllm")
	if err := os.WriteFile(cpp, []byte(fixture), 0600); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
	defer cancel()
	if out, err := exec.CommandContext(ctx, compiler, "-std=c++14", "-O1", cpp, "-o", binary).CombinedOutput(); err != nil {
		t.Fatalf("compile actual GDN source: %v\n%s", err, out)
	}
	if out, err := exec.CommandContext(ctx, binary).CombinedOutput(); err != nil {
		t.Fatalf("actual GDN source arithmetic/layout: %v\n%s", err, out)
	} else if strings.TrimSpace(string(out)) != "PASS exact GDN source, FP64 oracle, carried state and bounded dispatch; CUDA/device/performance unrun" {
		t.Fatalf("incomplete host witness: %s", out)
	}
}

const cudaGDNVLLMHostPreamble = `
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <vector>
#include <ucontext.h>
#define __global__
#define __device__
#define __forceinline__ inline
#define __shared__ static
#define __align__(n) __attribute__((aligned(n)))
#define __launch_bounds__(...)
#define __expf expf
struct dim3 { unsigned x,y,z; dim3(unsigned x=1,unsigned y=1,unsigned z=1):x(x),y(y),z(z){} };
struct alignas(16) float4 { float x,y,z,w; };
static float4 make_float4(float x,float y,float z,float w) { return {x,y,z,w}; }
static float rsqrtf(float x) { return 1.0f/std::sqrt(x); }
static dim3 blockIdx,threadIdx;
struct Barrier;
static ucontext_t scheduler,contexts[256];
static std::vector<char> stacks[256];
static Barrier *waiting[256];
static bool finished[256];
static int current;
static std::function<void()> kernel_entry;
struct Barrier {
  int total,arrived=0;
  explicit Barrier(int n):total(n){}
  void wait() {
    waiting[current]=this;
    if (++arrived==total) {
      arrived=0;
      for(int i=0;i<256;++i) if(waiting[i]==this) waiting[i]=nullptr;
    }
    if(swapcontext(&contexts[current],&scheduler)) std::abort();
  }
};
static Barrier block_barrier(256);
static Barrier warp_barriers[8]={Barrier(32),Barrier(32),Barrier(32),Barrier(32),Barrier(32),Barrier(32),Barrier(32),Barrier(32)};
static float shuffle_values[256];
static void __syncthreads() { block_barrier.wait(); }
static float __shfl_xor_sync(unsigned mask,float value,int offset) {
  assert(mask==0xffffffffu && offset>0 && offset<32);
  const int tid=threadIdx.x;
  shuffle_values[tid]=value; warp_barriers[tid/32].wait();
  const float other=shuffle_values[tid^offset]; warp_barriers[tid/32].wait();
  return other;
}
static void fiber() { kernel_entry(); finished[current]=true; }
static void run_block() {
  for(int i=0;i<256;++i) {
    waiting[i]=nullptr; finished[i]=false;
    stacks[i].resize(32*1024);
    if(getcontext(&contexts[i])) std::abort();
    contexts[i].uc_stack.ss_sp=stacks[i].data();
    contexts[i].uc_stack.ss_size=stacks[i].size();
    contexts[i].uc_link=&scheduler;
    makecontext(&contexts[i],fiber,0);
  }
  for(;;) {
    bool live=false,progress=false;
    for(int i=0;i<256;++i) if(!finished[i]) {
      live=true;
      if(waiting[i]) continue;
      progress=true; current=i; threadIdx=dim3(i);
      if(swapcontext(&scheduler,&contexts[i])) std::abort();
    }
    if(!live) break;
    assert(progress && "kernel barrier deadlock");
  }
  assert(block_barrier.arrived==0);
  for(auto &barrier:warp_barriers) assert(barrier.arrived==0);
}
`

const cudaGDNVLLMHostMain = `
static std::vector<float> data(int n,int seed,float scale) {
  std::vector<float> out(n);
  for(int i=0;i<n;++i) out[i]=float((i*37+seed*19)%211-105)*scale;
  return out;
}
struct Fixture {
  int tokens,nk,nv,width;
  std::vector<float> mixed,a,b,alog,bias,z,norm,state,out;
  Fixture(int t,int k,int ratio):tokens(t),nk(k),nv(k*ratio),width((2*k+nv)*128),
      mixed(data(t*width,1,.003f)),a(data(t*nv,2,.01f)),b(data(t*nv,3,.01f)),
      alog(data(nv,4,.004f)),bias(data(nv,5,.002f)),z(data(t*nv*128,6,.008f)),
      norm(data(128,7,.001f)),state(data(nv*128*128+8,8,.001f)),
      out(t*nv*128+8,NAN) {
    for(float &x:norm) x+=1;
    for(int i=0;i<4;++i) state[i]=state[state.size()-1-i]=out[i]=out[out.size()-1-i]=123456.0f;
    // Exercise both softplus branches and distinct head gates.
    if(t*nv>1) a[1]=22.0f;
  }
};
static void reference(Fixture &f,std::vector<double> &state,std::vector<double> &out) {
  state.assign(f.state.begin()+4,f.state.end()-4); out.resize(f.tokens*f.nv*128);
  for(int t=0;t<f.tokens;++t) for(int h=0;h<f.nv;++h) {
    const int kh=h/(f.nv/f.nk);
    double q[128],k[128],qss=0,kss=0;
    for(int i=0;i<128;++i) {
      q[i]=f.mixed[t*f.width+kh*128+i]; k[i]=f.mixed[t*f.width+f.nk*128+kh*128+i];
      qss+=q[i]*q[i]; kss+=k[i]*k[i];
    }
    for(int i=0;i<128;++i) { q[i]/=std::sqrt(qss+1e-6)*std::sqrt(128.0); k[i]/=std::sqrt(kss+1e-6); }
    const double gate=double(f.a[t*f.nv+h])+f.bias[h];
    const double decay=std::exp(-std::exp(double(f.alog[h]))*(gate>20?gate:std::log1p(std::exp(gate))));
    const double beta=1/(1+std::exp(-double(f.b[t*f.nv+h])));
    double raw[128],ss=0;
    for(int v=0;v<128;++v) {
      double hk=0,hq=0;
      for(int i=0;i<128;++i) { double &s=state[(h*128+i)*128+v]; s*=decay; hk+=s*k[i]; }
      const double delta=(f.mixed[t*f.width+2*f.nk*128+h*128+v]-hk)*beta;
      for(int i=0;i<128;++i) { double &s=state[(h*128+i)*128+v]; s+=k[i]*delta; hq+=s*q[i]; }
      raw[v]=hq; ss+=hq*hq;
    }
    const double inv=1/std::sqrt(ss/128+1e-6);
    for(int v=0;v<128;++v) {
      const int o=(t*f.nv+h)*128+v; const double z=f.z[o];
      out[o]=raw[v]*inv*f.norm[v]*z/(1+std::exp(-z));
    }
  }
}
template<int Ratio> static void run(Fixture &f,int start,int count) {
  kernel_entry=[&] {
    fak_gdn_vllm::k_qwen35_gdn_post_conv_vllm<Ratio>(f.mixed.data()+start*f.width,
      f.a.data()+start*f.nv,f.b.data()+start*f.nv,f.alog.data(),f.bias.data(),
      f.state.data()+4,f.z.data()+start*f.nv*128,f.norm.data(),
      f.out.data()+4+start*f.nv*128,count,f.nk,f.nv,f.width,1e-6f);
  };
  for(int h=0;h<f.nv;++h) { blockIdx=dim3(h); run_block(); }
}
static void close(const char*name,const float*got,const std::vector<double>&want) {
  double scale=0; for(double v:want) scale=std::max(scale,std::abs(v));
  for(size_t i=0;i<want.size();++i) if(!std::isfinite(got[i]) || std::abs(double(got[i])-want[i])>1e-5*(1+scale)) {
    std::fprintf(stderr,"%s[%zu]: %.9g != %.12g\n",name,i,got[i],want[i]); std::abort();
  }
}
template<int Ratio> static void check(int tokens,int nk,bool split) {
  Fixture f(tokens,nk,Ratio);
  std::vector<double> state,out; reference(f,state,out);
  if(split) { run<Ratio>(f,0,1); run<Ratio>(f,1,tokens-1); }
  else run<Ratio>(f,0,tokens);
  close("canonical state",f.state.data()+4,state); close("token outputs",f.out.data()+4,out);
  for(int i=0;i<4;++i) assert(f.state[i]==123456 && f.state[f.state.size()-1-i]==123456 && f.out[i]==123456 && f.out[f.out.size()-1-i]==123456);
}
int main() {
  using fak_gdn_vllm::eligible;
  for(int ratio:{1,2,3,4,8}) for(int t=1;t<=8;++t) assert(eligible("vllm",t,2,2*ratio,128,128,false));
  assert(!eligible(nullptr,1,1,1,128,128,false)); assert(!eligible("",1,1,1,128,128,false));
  assert(!eligible("vllmx",1,1,1,128,128,false)); assert(!eligible("vllm",1,1,1,128,128,true));
  for(int t:{0,9}) assert(!eligible("vllm",t,1,1,128,128,false));
  for(int ratio:{5,6,7}) assert(!eligible("vllm",1,1,ratio,128,128,false));
  assert(!eligible("vllm",1,0,1,128,128,false)); assert(!eligible("vllm",1,2,3,128,128,false));
  assert(!eligible("vllm",1,1,1,127,128,false)); assert(!eligible("vllm",1,1,1,128,129,false));
  check<1>(1,1,false); check<2>(2,2,false); check<3>(8,1,false);
  check<4>(4,1,false); check<8>(1,1,false); check<3>(8,1,true);
  std::puts("PASS exact GDN source, FP64 oracle, carried state and bounded dispatch; CUDA/device/performance unrun");
}
`
