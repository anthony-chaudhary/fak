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

// The actual producer, rounding kernels, three-GEMM helper and scratch borrower
// execute against a scalar host cuBLAS substitute. This is NOT CUDA compilation,
// CUDA arithmetic, asynchronous execution, full model parity or a timing witness.
// fak-test:runtime medium est=2s lane=default
func TestCUDAGDNSharedSplitSource(t *testing.T) {
	compiler, err := exec.LookPath("c++")
	if err != nil {
		t.Skip("host C++ compiler unavailable; shared-split witness unrun")
	}
	read := func(path string) string {
		t.Helper()
		b, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		return string(b)
	}
	gdn, group := read("cuda/qwen35_gdn.cuh"), read("cuda/gdn_shared_split.cuh")
	function := func(source, signature string) string {
		t.Helper()
		if strings.Count(source, signature) != 1 {
			t.Fatalf("ambiguous/missing %s", signature)
		}
		start := strings.Index(source, signature)
		open := start + strings.Index(source[start:], "{")
		depth := 0
		for i := open; i < len(source); i++ {
			switch source[i] {
			case '{':
				depth++
			case '}':
				depth--
				if depth == 0 {
					return source[start : i+1]
				}
			}
		}
		t.Fatalf("incomplete %s", signature)
		return ""
	}
	replace := func(source, old, next string) string {
		t.Helper()
		if strings.Count(source, old) != 1 {
			t.Fatalf("launch/source seam changed: %s", old)
		}
		return strings.Replace(source, old, next, 1)
	}
	legacy := function(gdn, "static int q8_dequant_sgemm(")
	legacy = replace(legacy, `FCUDA_TS("k_q8_dequant_split", (wn + 255) / 256, k_q8_dequant_split<<<(wn + 255) / 256, 256, 0, g_stream>>>(
      (const signed char *)codes, scales, wHi, wLo, wn, cols, block));`, `launch_weight((const signed char *)codes, scales, wHi, wLo, wn, cols, block);`)
	legacy = replace(legacy, `FCUDA_TS("k_fp16_split", (xn + 255) / 256, k_fp16_split<<<(xn + 255) / 256, 256, 0, g_stream>>>(X, xHi, xLo, xn));`, `launch_activation(X, xHi, xLo, xn);`)
	group = replace(group, `FCUDA_TS("k_fp16_split", (xn + 255) / 256, k_fp16_split<<<(xn + 255) / 256, 256, 0, g_stream>>>(x, xHi, xLo, xn));`, `launch_activation(x, xHi, xLo, xn);`)
	group = replace(group, `FCUDA_TS("k_q8_dequant_split", (wn + 255) / 256, k_q8_dequant_split<<<(wn + 255) / 256, 256, 0, g_stream>>>(
        (const signed char *)codes[i], scales[i], wHi, wLo, wn, cols, 32));`, `launch_weight((const signed char *)codes[i], scales[i], wHi, wLo, wn, cols, 32);`)
	if strings.Contains(legacy+group, "<<<") {
		t.Fatal("uncovered native launch")
	}
	// Bind the actual caller's all-Q8 gate, exact operand/row order, MMQ precedence
	// and existing drain. Extract the complete branch, then execute it below too.
	caller := function(gdn, "if (tokens >= 128 && inQKVQ8 == 1 && inZQ8 == 1 && inBQ8 == 1 && inAQ8 == 1)")
	if !strings.Contains(caller, "getenv(\"FAK_CUDA_Q8_SEQUENCE_PREFILL\")") ||
		!strings.Contains(caller, "if (dq != 0) return qwen35_gdn_drain_after_error(dq);") {
		t.Fatal("selected producer must retain MMQ gate and caller error drain")
	}
	caller = strings.ReplaceAll(caller, "getenv(\"FAK_CUDA_Q8_SEQUENCE_PREFILL\")", "test_sequence_arm")
	if !strings.Contains(gdn, "#include \"gdn_shared_split.cuh\"") {
		t.Fatal("producer is not included")
	}
	program := cudaGDNSharedSplitPreamble
	start, end := strings.Index(gdn, "enum class SequenceSplitPhase"), strings.Index(gdn, "static int sequence_split_status(")
	if start < 0 || end <= start {
		t.Fatal("actual sequence owner declaration missing")
	}
	program += gdn[start:end]
	program += function(gdn, "struct FP16CompensatedScratch") + ";\n"
	for _, sig := range []string{"static int fp16_compensated_owned_scratch(", "static int fp16_compensated_scratch(", "static int fp16_compensated_release(", "static int fp16_compensated_gemm(", "__global__ void k_fp16_split(", "__global__ void k_q8_dequant_split("} {
		program += "\n" + function(gdn, sig) + "\n"
	}
	program += cudaGDNSharedSplitLaunches + legacy + "\n" + group + "\n"
	program += cudaGDNSharedSplitCallerPrefix + caller + "\nreturn 0;\n}\n"
	program += cudaGDNSharedSplitMain
	t.Logf("gdn=%x helper=%x fixture=%x", sha256.Sum256([]byte(gdn)), sha256.Sum256([]byte(group)), sha256.Sum256([]byte(program)))
	dir := t.TempDir()
	cpp, binary := filepath.Join(dir, "shared.cpp"), filepath.Join(dir, "shared.exe")
	if err := os.WriteFile(cpp, []byte(program), 0600); err != nil {
		t.Fatal(err)
	}
	for _, mmq := range []bool{false, true} {
		ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
		args := []string{"-std=c++17", "-O0", "-ffp-contract=off", cpp, "-o", binary}
		if mmq {
			args = append([]string{"-DFAK_CUDA_Q8_MMQ_PREFILL"}, args...)
		}
		out, err := exec.CommandContext(ctx, compiler, args...).CombinedOutput()
		cancel()
		if err != nil {
			t.Fatalf("compile exact shared producer mmq=%v: %v\n%s", mmq, err, out)
		}
		run, stop := context.WithTimeout(t.Context(), 10*time.Second)
		out, err = exec.CommandContext(run, binary).CombinedOutput()
		stop()
		if err != nil || strings.TrimSpace(string(out)) != "PASS shared split source; CUDA/device unrun" {
			t.Fatalf("shared producer mmq=%v: %v\n%s", mmq, err, out)
		}
	}
}

const cudaGDNSharedSplitPreamble = `
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <map>
#include <vector>
using __half=uint16_t;
using cudaError_t=int;
using cudaStreamCaptureStatus=int;
using cublasStatus_t=int;
constexpr int cudaSuccess=0, cudaErrorNotReady=34, cudaStreamCaptureStatusNone=0;
constexpr int CUBLAS_STATUS_SUCCESS=0, CUBLAS_OP_T=1, CUBLAS_OP_N=0;
constexpr int CUDA_R_16F=2, CUDA_R_32F=4, CUBLAS_COMPUTE_32F=4, CUBLAS_GEMM_DEFAULT_TENSOR_OP=0;
constexpr int FCUDA_SPLIT_SCRATCH_ALLOC_FAILED=80002, FCUDA_SCRATCH_RETIRE_FAILED_BASE=83000, FCUDA_COMPENSATED_GEMM_FAILED_BASE=93000;
static int g_stream=1,g_blas=0,g_cuda_cleanup_status=0;
static bool g_capture_open=false,uncertain=false;
#define FAK_CUDA_V41_CHECKED_STREAM 1
static bool v41_candidates_stream_uncertain() { return uncertain; }
static int last_error=0,capture_error=0,runtime_capture=0,queries=0;
static int cudaGetLastError() { int e=last_error; last_error=0; return e; }
static int cudaStreamIsCapturing(int,cudaStreamCaptureStatus *s) { ++queries; *s=runtime_capture; return capture_error; }
static int allocations=0,releases=0,miss=0,release_error=0;
static std::map<void*,size_t> owners;
static void *scratch_malloc(size_t bytes) { if (++allocations==miss) return nullptr; void *p=std::malloc(bytes); if(p) owners[p]=bytes; return p; }
static void fcuda_free(void *p) { ++releases; if(owners.erase(p)) std::free(p); }
static int fcuda_discard_unsubmitted(void *p,size_t) { fcuda_free(p); return release_error; }
// Independent IEEE binary16 conversion; the extracted production split kernels
// select the values and rounding boundaries. No CUDA intrinsic is executed here.
static __half __float2half_rn(float f) {
  uint32_t u; std::memcpy(&u,&f,4); uint32_t sign=(u>>16)&0x8000, a=u&0x7fffffff;
  if(a>=0x7f800000) return (__half)(sign|0x7c00|(a>0x7f800000?0x200:0));
  int e=(int)(a>>23)-127; uint32_t m=(a&0x7fffff)|0x800000;
  if(e>15) return (__half)(sign|0x7c00);
  if(e<-25) return (__half)sign;
  unsigned shift=e<-14?(unsigned)(-e-1):13;
  uint32_t q=m>>shift, r=m&((1u<<shift)-1), half=1u<<(shift-1);
  q+=(r>half || (r==half && (q&1)));
  if(e<-14) return (__half)(sign|q);
  return (__half)(sign|((uint32_t)(e+14)<<10)+q);
}
static float __half2float(__half h) {
  uint32_t sign=(uint32_t)(h&0x8000)<<16, exp=(h>>10)&31, frac=h&1023, u;
  if(exp==31) u=sign|0x7f800000|(frac<<13);
  else if(exp) u=sign|((exp+112)<<23)|(frac<<13);
  else if(!frac) u=sign;
  else { int e=-14; while(!(frac&1024)){frac<<=1;--e;} u=sign|((uint32_t)(e+127)<<23)|((frac&1023)<<13); }
  float f; std::memcpy(&f,&u,4); return f;
}
static float __fmul_rn(float a,float b) { volatile float x=a*b; return x; }
static float __fsub_rn(float a,float b) { volatile float x=a-b; return x; }
#define __global__
struct dim3 { size_t x,y,z; dim3(size_t a=1,size_t b=1,size_t c=1):x(a),y(b),z(c){} };
static dim3 blockIdx,blockDim(256),threadIdx;
static int gemms=0,gemm_fail=0,splits=0,weights=0,launches=0,launch_fail=0;
static std::vector<int> order;
#define FCUDA_TS(name,grid,...) do { (void)(name); (void)(grid); __VA_ARGS__; } while(0)
static cublasStatus_t cublasGemmEx(int,int,int,int rows,int tokens,int cols,const float *alpha,
    const void *wv,int,int,const void *xv,int,int,const float *beta,void *yv,int,int,int,int) {
  order.push_back(3); if(++gemms==gemm_fail) return 13;
  const __half *w=(const __half*)wv,*x=(const __half*)xv; float *y=(float*)yv;
  for(int t=0;t<tokens;++t) for(int r=0;r<rows;++r) {
    float acc=0; for(int k=0;k<cols;++k) acc=acc+__half2float(w[(size_t)r*cols+k])*__half2float(x[(size_t)t*cols+k]);
    y[(size_t)t*rows+r]=*alpha*acc+(*beta==0?0:*beta*y[(size_t)t*rows+r]);
  }
  return 0;
}
`

const cudaGDNSharedSplitLaunches = `
static void launch_activation(const float *x,__half *hi,__half *lo,size_t n) {
  ++splits; order.push_back(1); if(++launches==launch_fail) { last_error=9; return; }
  for(size_t i=0;i<n;++i) { blockIdx.x=i/256;threadIdx.x=i%256;k_fp16_split(x,hi,lo,n); }
}
static void launch_weight(const signed char *w,const float *s,__half *hi,__half *lo,size_t n,int cols,int block) {
  ++weights; order.push_back(2); if(++launches==launch_fail) { last_error=9; return; }
  for(size_t i=0;i<n;++i) { blockIdx.x=i/256;threadIdx.x=i%256;k_q8_dequant_split(w,s,hi,lo,n,cols,block); }
}
`

const cudaGDNSharedSplitCallerPrefix = `
static int fallback_calls=0,drains=0;
static const char *test_sequence_arm=nullptr;
static int qwen35_gdn_q8_prefill(const void *w,const float *s,const float *x,float *y,int rows,int cols,int tokens,int block) {
  ++fallback_calls; return q8_dequant_sgemm(w,s,x,y,rows,cols,tokens,block);
}
static int qwen35_gdn_drain_after_error(int status) { ++drains; return status; }
static int call_sequence(const char *projectionMode,const void *const codes[4],const float *const scales[4],
    const float *dX,float *const outputs[4],int hidden,int convDim,int valueDim,int nV,int tokens,
    int inQKVQ8=1,int inZQ8=1,int inBQ8=1,int inAQ8=1) {
  if(nV<=0 || valueDim<=0 || valueDim%nV!=0) return -1;
  const int vHd=valueDim/nV; // Bind the production valueDim=nV*vHd identity.
  const void *dInQKV=codes[0],*dInZ=codes[1],*dInB=codes[2],*dInA=codes[3];
  const float *dInQKVScale=scales[0],*dInZScale=scales[1],*dInBScale=scales[2],*dInAScale=scales[3];
  float *dMixed=outputs[0],*dZ=outputs[1],*dB=outputs[2],*dA=outputs[3];
`

const cudaGDNSharedSplitMain = `
#define CHECK(c) do { if(!(c)){std::fprintf(stderr,"shared split line %d\n",__LINE__);return 1;} } while(0)
static void reset() {
  for(auto p:owners) std::free(p.first); owners.clear(); g_sequence_split={};
  allocations=releases=miss=release_error=queries=last_error=capture_error=runtime_capture=0;
  gemms=gemm_fail=splits=weights=launches=launch_fail=fallback_calls=drains=0;
  g_cuda_cleanup_status=0;g_capture_open=uncertain=false;test_sequence_arm=nullptr;order.clear();
}
int main() {
  constexpr int cols=32,tokens=128; const int rows[4]={8,4,2,2};
  std::vector<float> x(tokens*cols); std::vector<signed char> w[4]; std::vector<float> s[4],y[4],reference[4];
  const void *codes[4];const float *scales[4];float *out[4];
  for(size_t i=0;i<x.size();++i) x[i]=(float)((int)(i%131)-65)*0.031257f;
  for(int j=0;j<4;++j){w[j].resize(rows[j]*cols);s[j].resize(rows[j]);y[j].resize(tokens*rows[j]);
    for(size_t i=0;i<w[j].size();++i)w[j][i]=(signed char)((i*19+j*31)%255-127);
    for(size_t i=0;i<s[j].size();++i)s[j][i]=(i%2?-1.f:1.f)*(0.00113f+(float)i*0.00017f);
    codes[j]=w[j].data();scales[j]=s[j].data();out[j]=y[j].data();}
  // Conversion boundary checks independent of the producer comparison.
  CHECK(__float2half_rn(1.f)==0x3c00 && __float2half_rn(-0.f)==0x8000);
  CHECK(__float2half_rn(65504.f)==0x7bff && __float2half_rn(std::ldexp(1.f,-24))==1);
  CHECK(__float2half_rn(1.f+std::ldexp(1.f,-11))==0x3c00);
  CHECK(__float2half_rn(1.f+3*std::ldexp(1.f,-11))==0x3c02);
  auto call=[&](const char *mode){return call_sequence(mode,codes,scales,x.data(),out,cols,rows[0],rows[1],rows[2],tokens);};
  reset();CHECK(call("")==0 && splits==4 && weights==4 && gemms==12 && fallback_calls==4);
  for(int i=0;i<4;++i)reference[i]=y[i];
  // Owned and existing sequence-loan execution both use the actual production
  // acquire/release bodies. The unchanged owner-retirement suite is complementary.
  for(bool loan:{false,true}) {
    reset();
    if(loan){g_sequence_split.phase=SequenceSplitPhase::Active;g_sequence_split.token=7;
      for(int i=0;i<4;++i){size_t n=i<2?rows[0]*cols:tokens*cols;g_sequence_split.ptr[i]=scratch_malloc(n*2);g_sequence_split.bytes[i]=n*2;}}
    int before=allocations;
    CHECK(call("shared-split")==0 && splits==1 && weights==4 && gemms==12 && fallback_calls==0);
    CHECK(allocations-before==(loan?0:4) && releases==(loan?0:4) && !g_sequence_split.borrowed);
    std::vector<int> want={1};for(int i=0;i<4;++i)want.insert(want.end(),{2,3,3,3});CHECK(order==want);
    for(int i=0;i<4;++i)CHECK(std::memcmp(y[i].data(),reference[i].data(),y[i].size()*sizeof(float))==0);
  }
  auto loan_start=[&](){g_sequence_split.phase=SequenceSplitPhase::Active;g_sequence_split.token=9;
    for(int i=0;i<4;++i){size_t n=i<2?rows[0]*cols:tokens*cols;g_sequence_split.ptr[i]=scratch_malloc(n*2);g_sequence_split.bytes[i]=n*2;}};
  for(bool loan:{false,true}) {
    for(int fail=1;fail<=5;++fail){reset();if(loan)loan_start();launch_fail=fail;
      CHECK(call("shared-split")==20009 && launches==fail && drains==1 && fallback_calls==0 && releases==(loan?0:4));
      CHECK(gemms==(fail<=2?0:3*(fail-2)) && !g_sequence_split.borrowed);}
    for(int fail=1;fail<=12;++fail){reset();if(loan)loan_start();gemm_fail=fail;
      CHECK(call("shared-split")==93013 && gemms==fail && drains==1 && fallback_calls==0 && releases==(loan?0:4));
      CHECK(weights==(fail+2)/3 && splits==1 && !g_sequence_split.borrowed);}
  }
  for(int fail=1;fail<=4;++fail){reset();miss=fail;
    CHECK(call("shared-split")==80002 && launches==0 && releases==fail-1 && drains==1);}
  reset();miss=3;release_error=700;CHECK(call("shared-split")==83700 && launches==0 && releases==2);
  reset();runtime_capture=1;CHECK(call("shared-split")==-1 && allocations==0 && launches==0 && drains==1);
  reset();capture_error=700;CHECK(call("shared-split")==20700 && allocations==0 && launches==0);
  reset();last_error=719;CHECK(call("shared-split")==20719 && queries==0 && allocations==0);
  reset();uncertain=true;CHECK(call("shared-split")==70034 && queries==0 && allocations==0);
  reset();g_cuda_cleanup_status=719;CHECK(call("shared-split")==70719 && queries==0 && allocations==0);
  reset();g_sequence_split.phase=SequenceSplitPhase::Quarantined;CHECK(call("shared-split")==-1 && queries==0);
  reset();g_capture_open=true;CHECK(call("shared-split")==0 && fallback_calls==4 && splits==4);
  reset();CHECK(call("shared-splitx")==0 && fallback_calls==4 && splits==4);
  reset();CHECK(call_sequence("shared-split",codes,scales,x.data(),out,cols,8,4,2,127)==0 && launches==0);
  reset();CHECK(call_sequence("shared-split",codes,scales,x.data(),out,cols,8,4,2,tokens,1,1,0,1)==0 && launches==0);
  reset();test_sequence_arm="mmq";CHECK(call("shared-split")==0);
#ifdef FAK_CUDA_Q8_MMQ_PREFILL
  CHECK(fallback_calls==4 && splits==4);
#else
  CHECK(fallback_calls==0 && splits==1);
#endif
  using namespace fak_gdn_shared_split;
  CHECK(!selected(nullptr,nullptr,false,128) && !selected("shared-split",nullptr,false,127));
  CHECK(!selected("shared-split",nullptr,false,65536) && !selected("shared-split",nullptr,true,128));
  // Validate aliases and bounded arithmetic without allocating large tensors.
  reset();float *save=out[1];out[1]=out[0]+1;CHECK(call("shared-split")==-1 && allocations==0);out[1]=save;
  reset();save=out[0];out[0]=x.data();CHECK(call("shared-split")==-1 && allocations==0);out[0]=save;
  reset();const float *ss=scales[0];scales[0]=(const float*)((const char*)ss+1);CHECK(call("shared-split")==-1 && allocations==0);scales[0]=ss;
  reset();const void *cw=codes[0];codes[0]=(const void*)(UINTPTR_MAX-8);CHECK(call("shared-split")==-1 && allocations==0);codes[0]=cw;
  span reads[9],writes[4];size_t wn=0,xn=0;int bad[4]={INT_MAX,4,2,2};
  CHECK(!ranges(codes,scales,x.data(),out,bad,32,128,reads,writes,&wn,&xn));
  CHECK(!ranges(codes,scales,x.data(),out,rows,INT_MAX-31,128,reads,writes,&wn,&xn));
  reset();g_sequence_split.phase=SequenceSplitPhase::Active;g_sequence_split.token=4;
  CHECK(call("shared-split")==-1 && allocations==0 && launches==0 && !g_sequence_split.borrowed);
  reset();g_sequence_split.phase=SequenceSplitPhase::Active;g_sequence_split.token=4;
  for(int i=0;i<4;++i){size_t n=i<2?rows[0]*cols:tokens*cols;g_sequence_split.ptr[i]=scratch_malloc(n*2);g_sequence_split.bytes[i]=n*2;}
  void *p=g_sequence_split.ptr[2];g_sequence_split.ptr[2]=x.data();
  CHECK(call("shared-split")==-1 && launches==0 && !g_sequence_split.borrowed);g_sequence_split.ptr[2]=p;
  reset();std::puts("PASS shared split source; CUDA/device unrun");return 0;
}
`
