package compute

import (
	"context"
	"flag"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

var kvFP8QwenIngressHost = flag.Bool("fak-kv-fp8-qwen-ingress-host", false, "run source-bound Qwen ingress host controls; not CUDA arithmetic")

// fak-test:runtime fast est=10ms lane=default
func TestCUDAKVFP8QwenIngressSourceContract(t *testing.T) {
	src := readCUDASource(t, "cuda/kv_fp8_qwen_ingress.cuh")
	for _, part := range []string{"s->heads!=24", "s->rotary!=64", "FAK_ALLOC_WEIGHTS", "FAK_ALLOC_DEVICE_LOCAL",
		"s->work_bytes[i]!=outs[i]", "disjoint(s->work[i],outs[i],s->input[j],ins[j])",
		"disjoint(s->work[i],outs[i],s->work[j],outs[j])", "s->tokens>INT_MAX/(24*256)"} {
		if !strings.Contains(src, part) {
			t.Fatalf("missing exact admission %q", part)
		}
	}
	for _, name := range []string{"fcuda_kv_fp8_qwen_validate_owned", "fcuda_kv_fp8_qwen_norm_owned", "fcuda_kv_fp8_qwen_rope_owned"} {
		body := cudaFunctionBody(t, src, name)
		if strings.Index(body, "admit_qwen(") < 0 || strings.Index(body, "valid(s)") < strings.Index(body, "admit_qwen(") {
			t.Fatal("effect before native gate")
		}
	}
	norm := cudaFunctionBody(t, src, "fcuda_kv_fp8_qwen_norm_owned")
	for _, part := range []string{"fcuda_rmsnorm_f32(s->input[0],s->input[3],s->work[0],s->tokens*24,256,s->epsilon)",
		"fcuda_rmsnorm_f32(s->input[1],s->input[4],s->work[1],s->tokens*4,256,s->epsilon)", "cudaGetLastError()", "scan(s->work[i]"} {
		if !strings.Contains(norm, part) {
			t.Fatal("wrong block RMSNorm binding", part)
		}
	}
	for _, part := range []string{"if (g_ts_on) return refusal(FCUDA_KV_FP8_QWEN_TIMING_UNSUPPORTED,phase)",
		"if (v41_candidates_stream_uncertain())", "g_cuda_cleanup_status ? g_cuda_cleanup_status",
		"r=producer_result((int)cudaGetLastError(),r.phase); if (r.status) return r;"} {
		if !strings.Contains(src, part) {
			t.Fatal("missing terminal/timing gate", part)
		}
	}
	if strings.Count(norm, "r=producer_result((int)cudaGetLastError(),r.phase)") != 2 {
		t.Fatal("each RMSNorm must recheck terminal")
	}
	if strings.Contains(src, "fcuda_rmsnorm_dispatched") || strings.Contains(src, "vllm_qk_f32(") {
		t.Fatal("unreviewed arithmetic path")
	}
	producer := readCUDASource(t, "cuda/transformer_ops.cuh")
	for _, part := range []string{"k_rmsnorm<<<rows, 256, 0, g_stream>>>", "pow(theta, -(2.0 * j) / rotaryDim)",
		"(double)(startPos + token) * freq", "a * cs - b * sn : b * cs + a * sn"} {
		if !strings.Contains(producer, part) {
			t.Fatal("bound producer drift", part)
		}
	}
	adapter := readCUDASource(t, "cuda_kv_fp8_qwen_ingress_qualification.go")
	for _, part := range []string{"tensor.buf.(*cudaBuf)", "tensor.Quant != nil", "b.n != bytes[i]",
		"b.graphOncePinned", "b.invalidForSubmit()", "b.class != MemoryWeights", "root: sources", "C.fcuda_kv_fp8_qwen_admit_owned()"} {
		if !strings.Contains(adapter, part) {
			t.Fatal("metadata preflight drift", part)
		}
	}
	for _, bad := range []string{"makeTensor(", "devTr(", "cudaMu.Lock(", "cudaBufForSubmit("} {
		if strings.Contains(adapter, bad) {
			t.Fatal("unexpected generic effect", bad)
		}
	}
}

// This compiles the actual native ownership wrappers with fake norm/RoPE and
// transport. It cannot establish CUDA arithmetic, asynchrony or H200 validity.
// fak-test:runtime medium est=1s lane=optin
func TestCUDAKVFP8QwenIngressHostControl(t *testing.T) {
	if !*kvFP8QwenIngressHost {
		t.Skip("explicit source-bound host control not selected; no verification credit")
	}
	compiler, err := exec.LookPath("c++")
	if err != nil {
		t.Fatal("selected host witness requires C++")
	}
	read := func(path string) string { return readCUDASource(t, path) }
	strip := func(s string) string {
		for _, inc := range []string{"#include \"cuda_kv_fp8_lifecycle.h\"", "#include \"../cuda_kv_fp8_lifecycle.h\"",
			"#include \"kv_fp8.cuh\"", "#include \"../cuda_kv_fp8_append.h\"", "#include \"kv_fp8_lifecycle.cuh\"",
			"#include \"../cuda_kv_fp8_qwen_ingress.h\""} {
			s = strings.ReplaceAll(s, inc, "")
		}
		return s
	}
	header := read("cuda_kv_fp8_lifecycle.h") + read("cuda_kv_fp8_rotary.h") +
		strip(read("cuda_kv_fp8_append.h")) + strip(read("cuda_kv_fp8_qwen_ingress.h"))
	i := strings.Index(kvFP8LifecycleMain, "int main()")
	if i < 0 {
		t.Fatal("lifecycle fixture drift")
	}
	program := header + kvFP8LifecycleHarness + kvFP8AppendNativeStubs + kvFP8QwenIngressStubs +
		strip(read("cuda/kv_fp8_lifecycle.cuh")) + strip(read("cuda/kv_fp8_append.cuh")) +
		strip(read("cuda/kv_fp8_qwen_ingress.cuh")) + kvFP8LifecycleMain[:i] + kvFP8QwenIngressMain
	dir := t.TempDir()
	cpp, bin := filepath.Join(dir, "qwen-ingress.cpp"), filepath.Join(dir, "qwen-ingress")
	if runtime.GOOS == "windows" {
		bin += ".exe"
	}
	if err := os.WriteFile(cpp, []byte(program), 0600); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if out, err := exec.CommandContext(ctx, compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", cpp, "-o", bin).CombinedOutput(); err != nil {
		t.Fatalf("compile actual wrappers: %v\n%s", err, out)
	}
	if out, err := exec.CommandContext(ctx, bin).CombinedOutput(); err != nil {
		t.Fatalf("actual wrappers: %v\n%s", err, out)
	}
}

const kvFP8QwenIngressStubs = `
#include <cmath>
using std::isfinite;
enum { FAK_ALLOC_WEIGHTS=0 };
static int norm_calls,rope_calls,norm_fail,rope_fail,norm_latch,rope_latch;
static bool g_ts_on;
static cudaError_t cudaGetLastError(){int e=pending_error;pending_error=0;return e;}
static void fcuda_rmsnorm_f32(const float *x,const float *w,float *y,int rows,int n,float eps){
 ++norm_calls;assert(n==256 && eps==0.000001f);
 for(int i=0;i<rows*n;i++)y[i]=x[i]+w[i%n];
 if(norm_latch==norm_calls)g_cuda_cleanup_status=45;
 if(norm_fail==norm_calls)pending_error=17;
}
static int fak_qwen35_partial_rope_panel_f32(const float *q,const float *k,float *qo,float *ko,
 int tokens,int first,int nq,int nk,int hd,int rotary,double theta){
 ++rope_calls;if(rope_latch)g_cuda_cleanup_status=46;assert(tokens==1 && first==4 && nq==24 && nk==4 && hd==256 && rotary==64 && theta==10000.0);
 if(rope_fail)return rope_fail;
 memcpy(qo,q,(size_t)tokens*nq*hd*4);memcpy(ko,k,(size_t)tokens*nk*hd*4);return 0;
}
`

const kvFP8QwenIngressMain = `
static fcuda_kv_fp8_qwen_stage prepare_qwen(){
 reset();validate_calls=quantize_calls=validate_mask=quantize_failure=0;
 norm_calls=rope_calls=norm_fail=rope_fail=norm_latch=rope_latch=0;g_ts_on=false;
 fcuda_kv_fp8_qwen_stage s={};
 size_t in[5]={24*256*4,4*256*4,4*256*4,256*4,256*4},out[4]={in[0],in[1],in[0],in[1]};
 for(int i=0;i<5;i++){
  float *p=(float *)allocation(in[i],FCUDA_KV_FP8_OWNER_WORKSPACE);
  if(i>=3)g_alloc_tags[p].cls=FAK_ALLOC_WEIGHTS;
  for(size_t j=0;j<in[i]/4;j++)p[j]=1.0f;
  s.input[i]=p;s.input_bytes[i]=in[i];
 }
 for(int i=0;i<4;i++){s.work[i]=(float *)allocation(out[i],FCUDA_KV_FP8_OWNER_WORKSPACE);s.work_bytes[i]=out[i];}
 s.status=status_allocation();*s.status=0;s.tokens=1;s.first=4;s.heads=24;s.rotary=64;
 s.producer=FCUDA_KV_FP8_QWEN_PARTIAL_PANEL_V1;s.epsilon=0.000001f;s.theta=10000;
 return s;
}
int main(){
 auto s=prepare_qwen();
 auto r=fcuda_kv_fp8_qwen_admit_owned();assert(!r.status && !validate_calls && !norm_calls && !rope_calls);
 r=fcuda_kv_fp8_qwen_validate_owned(&s);assert(!r.status && !r.completed && validate_calls==5 && !norm_calls);
 r=fcuda_kv_fp8_finish_owned(s.status,1);assert(r.completed && r.numerical_checked && !r.numerical);
 r=fcuda_kv_fp8_qwen_norm_owned(&s);assert(!r.status && norm_calls==2 && validate_calls==7 && !rope_calls);
 r=fcuda_kv_fp8_finish_owned(s.status,1);assert(r.completed && !r.numerical);
 r=fcuda_kv_fp8_qwen_rope_owned(&s);assert(!r.status && rope_calls==1 && validate_calls==9 && s.work[2][6143]==2.0f && s.work[3][1023]==2.0f);
 r=fcuda_kv_fp8_finish_owned(s.status,1);assert(r.completed && !r.numerical && !free_calls);
 for(int mode=0;mode<14;mode++){
  s=prepare_qwen();
  if(mode==0)g_managed_live[(void *)s.input[3]]=s.input_bytes[3];
  if(mode==1)g_alloc_tags[s.input[3]].cls=FAK_ALLOC_ACTIVATIONS_SCRATCH;
  if(mode==2)g_panel_live[(void *)s.input[0]]=s.input_bytes[0];
  if(mode==3)s.input_bytes[4]-=4;
  if(mode==4)s.work_bytes[0]-=4;
  if(mode==5)s.work[2]=s.work[0];
  if(mode==6)s.work[0]=(float *)s.input[0];
  if(mode==7)s.first=INT_MAX;
  if(mode==8)s.tokens=INT_MAX;
  if(mode==9)s.heads=8;
  if(mode==10)s.rotary=256;
  if(mode==11)s.theta=INFINITY;
  if(mode==12)s.epsilon=NAN;
  if(mode==13)s.producer=0;
  r=fcuda_kv_fp8_qwen_validate_owned(&s);
  assert(r.status && !norm_calls && !rope_calls && !validate_calls && !free_calls);
 }
 s=prepare_qwen();g_ts_on=true;int allocations_before=alloc_calls;
 r=fcuda_kv_fp8_qwen_admit_owned();assert(r.status==FCUDA_KV_FP8_QWEN_TIMING_UNSUPPORTED && !r.terminal);
 r=fcuda_kv_fp8_qwen_validate_owned(&s);assert(r.status && !validate_calls);
 r=fcuda_kv_fp8_qwen_norm_owned(&s);assert(r.status && !norm_calls);
 r=fcuda_kv_fp8_qwen_rope_owned(&s);assert(r.status && !rope_calls && alloc_calls==allocations_before);
 g_cuda_cleanup_status=61;r=fcuda_kv_fp8_qwen_admit_owned();assert(r.status==61 && r.terminal);
 s=prepare_qwen();g_capture_open=true;r=fcuda_kv_fp8_qwen_validate_owned(&s);assert(r.status && !validate_calls);
 s=prepare_qwen();pending_error=23;r=fcuda_kv_fp8_qwen_norm_owned(&s);assert(r.status==23 && r.terminal && pending_error==23 && !norm_calls);
 for(int failed=1;failed<=2;failed++){
  s=prepare_qwen();norm_fail=failed;r=fcuda_kv_fp8_qwen_norm_owned(&s);
  assert(r.status==17 && r.terminal && norm_calls==failed && !rope_calls && !validate_calls && !free_calls);
 }
 for(int failed=1;failed<=2;failed++){
  s=prepare_qwen();norm_latch=failed;r=fcuda_kv_fp8_qwen_norm_owned(&s);
  assert(r.status==45 && r.terminal && norm_calls==failed && !validate_calls && !rope_calls && !free_calls);
 }
 s=prepare_qwen();norm_latch=norm_fail=1;r=fcuda_kv_fp8_qwen_norm_owned(&s);
 assert(r.status==45 && r.terminal && norm_calls==1 && !validate_calls);
 s=prepare_qwen();rope_latch=1;r=fcuda_kv_fp8_qwen_rope_owned(&s);
 assert(r.status==46 && r.terminal && rope_calls==1 && !validate_calls && !free_calls);
 s=prepare_qwen();rope_fail=31;r=fcuda_kv_fp8_qwen_rope_owned(&s);
 assert(r.status==31 && r.terminal && rope_calls==1 && !validate_calls && !free_calls);
 s=prepare_qwen();validate_mask=8;r=fcuda_kv_fp8_qwen_validate_owned(&s);assert(!r.status);
 r=fcuda_kv_fp8_finish_owned(s.status,1);assert(r.completed && r.numerical_checked && r.numerical==8 && !norm_calls);
 reset();return 0;
}
`
