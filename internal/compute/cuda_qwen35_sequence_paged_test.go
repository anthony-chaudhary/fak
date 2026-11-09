package compute

import (
	"context"
	"crypto/sha256"
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

// Actual ordinary entry, paged allocator/Clone, transaction and native page
// layout/retirement bodies run with an explicit host CUDA/linear-algebra model.
// This is required host evidence, never nvcc, H200 or whole-model parity.
// fak-test:runtime medium est=8s lane=default
func TestCUDAQwenPagedSequenceOwnership(t *testing.T) {
	read := func(path string) string {
		b, e := os.ReadFile(path)
		if e != nil {
			t.Fatal(e)
		}
		return string(b)
	}
	section := func(s, a, b string) string {
		i, j := strings.Index(s, a), strings.Index(s, b)
		if i < 0 || j <= i || strings.Count(s, a) != 1 || strings.Count(s, b) != 1 {
			t.Fatalf("ambiguous section %s", a)
		}
		return s[i:j]
	}
	wanted := map[string]bool{}
	selectNames := func(names ...string) {
		for _, n := range names {
			wanted[n] = false
		}
	}
	selectNames("cudaKVPagePool", "cudaPagedPoolKey", "cudaPagedPoolsMu", "cudaPagedKV", "cudaPagedKVCopyFailure", "*cudaPagedKVCopyFailure.Error", "*cudaKVPagePool.retainCopyOwnerLocked", "*cudaKVPagePool.stopCopyLocked", "*cudaKVPagePool.copyErrorLocked", "*cudaPagedKV.requireCopyHealthyLocked", "*cudaBackend.pagedPool", "*cudaBackend.newPagedKV", "*cudaPagedKV.Len", "*cudaPagedKV.stride", "*cudaPagedKV.Clone", "*cudaPagedKV.releaseLocked", "*cudaPagedKV.Free", "*cudaBackend.NewKV")
	selectNames("Qwen35SequenceState", "Qwen35SequenceLayer", "Qwen35SequencePrefillRequest", "Qwen35SequencePrefillResult", "Qwen35SequenceTransferCounters", "Qwen35SequenceFeatureRows", "Qwen35SequencePrefixReplay", "Qwen35SequenceError", "*Qwen35SequenceError.Error", "*Qwen35SequenceError.Unwrap", "qwen35SequenceEmbeddingContract")
	selectNames("qwen35SequenceFailure", "validateQwen35SequenceGeometry", "qwen35SequenceSameShape", "qwen35SequenceMatrixBytes", "*cudaBackend.validateQwen35SequenceTensor", "*cudaBackend.validateQwen35SequenceRequestLocked", "*cudaBackend.validateQwen35SequenceStoreRequestLocked", "*cudaBackend.Qwen35SequencePrefill", "*cudaBackend.qwen35SequenceReleaseSinceLocked", "*cudaBackend.qwen35SequencePoisonStates", "*cudaBackend.Qwen35SequenceRawHiddenPath")
	var body strings.Builder
	body.WriteString(cudaPagedSequenceGoPreamble)
	paths := []string{"cuda_qwen35_sequence_paged.go", "cuda_paged_kv_blocks.go", "cuda_paged_kv.go", "cuda_kv.go", "qwen35_sequence_contract.go", "cuda_qwen35_sequence.go"}
	for _, path := range paths {
		source := read(path)
		fs := token.NewFileSet()
		f, e := parser.ParseFile(fs, path, source, 0)
		if e != nil {
			t.Fatal(e)
		}
		emit := func(name string, n ast.Node, prefix string, all bool) {
			if !all {
				seen, ok := wanted[name]
				if !ok {
					return
				}
				if seen {
					t.Fatalf("duplicate %s", name)
				}
				wanted[name] = true
			}
			text := source[fs.Position(n.Pos()).Offset:fs.Position(n.End()).Offset]
			body.WriteString(prefix + text + "\n")
			t.Logf("%s/%s %x", path, name, sha256.Sum256([]byte(text)))
		}
		all := path == "cuda_qwen35_sequence_paged.go" || path == "cuda_paged_kv_blocks.go"
		for _, decl := range f.Decls {
			switch n := decl.(type) {
			case *ast.FuncDecl:
				name := n.Name.Name
				if n.Recv != nil {
					r := n.Recv.List[0].Type
					star := ""
					if p, ok := r.(*ast.StarExpr); ok {
						star = "*"
						r = p.X
					}
					name = star + r.(*ast.Ident).Name + "." + name
				}
				emit(name, n, "", all)
			case *ast.GenDecl:
				if n.Tok == token.IMPORT {
					continue
				}
				if path == "cuda_paged_kv.go" && n.Tok == token.VAR && strings.Contains(source[fs.Position(n.Pos()).Offset:fs.Position(n.End()).Offset], "cudaPagedPoolsMu") {
					emit("cudaPagedPoolsMu", n, "", false)
					continue
				}
				if all {
					emit("declaration", n, "", true)
					continue
				}
				for _, s := range n.Specs {
					if typ, ok := s.(*ast.TypeSpec); ok {
						emit(typ.Name.Name, typ, "type ", false)
					}
				}
			}
		}
	}
	for name, seen := range wanted {
		if !seen {
			t.Fatalf("missing actual declaration %s", name)
		}
	}
	// Keep the shared fixture constants unchanged: the independent decode
	// witness also uses them. This extension qualifies only sequence prefill.
	replace := func(s, old, next string) string {
		if strings.Count(s, old) != 1 {
			t.Fatalf("ambiguous FFN fixture seam %q", old)
		}
		return strings.Replace(s, old, next, 1)
	}
	shims := replace(cudaPagedSequenceGoShims, `if e==nil&&site=="attention-v"`, `if e==nil&&(site=="ffn-gate"||site=="ffn-up"){a:=unsafe.Slice((*float32)(b.ptr),t.Numel());for i:=range a{a[i]=float32(i%13-6)/4}};if e==nil&&site=="ffn-down"{must(C.test_ffn_down(c.cf(x),C.int(x.Numel()))==0,"FFN down lost live activated gate")};if e==nil&&site=="attention-v"`)
	body.WriteString(shims)
	body.WriteString(replace(cudaPagedSequenceGoControls, "func main(){", `
func checkFFNGateReuse(){
 must(C.test_swiglu_alias()==0,"source-extracted SwiGLU alias/tail control")
 for _,tokens:=range []int{1,2,17}{for _,allLogits:=range []bool{false,true}{
  c,k:=reset();r:=reqFor(c,k,0,tokens);r.NeedAllLogits=allLogits
  result,e:=c.Qwen35SequencePrefill(r)
  must(e==nil&&result.Tokens==tokens,"FFN gate reuse actual entry failed")
  must(C.test_counter(11)==8&&C.test_counter(12)==0&&C.test_counter(13)==0,"FFN gate reuse still allocates or writes a separate activation")
  must(C.test_counter(14)==8&&C.test_counter(15)==8,"FFN gate lifetime/down consumer mismatch")
  must(cudaQwen35PagedOwners==nil&&C.test_counter(7)==0,"FFN reuse changed owner settlement")
 }}
 fmt.Println("actual prefill FFN gate alias, down-consumer and retirement controls PASS")
}
func main(){
 checkFFNGateReuse()`))
	kv := read("cuda/kv_precision.cuh")
	nativeShims := replace(cudaPagedSequenceNativeShims,
		`extern "C" void fcuda_swiglu_f32(const float*x,const float*y,float*z,int n){for(int i=0;i<n;++i)z[i]=x[i]*y[i]/(1+expf(-x[i]));}`,
		`extern "C" void fcuda_swiglu_f32(const float*x,const float*y,float*z,int n){++ffn_calls;if(z!=x)++ffn_separate;ffn_gate=x;ffn_out=z;ffn_n=n;run_swiglu(x,y,z,n);}`)
	nativeShims = replace(nativeShims, `++allocs;if(mode==1`, `if(!strcmp(name,"ffn-swiglu"))++ffn_allocs;++allocs;if(mode==1`)
	nativeShims = replace(nativeShims, `extern "C" void test_reset(){`, `extern "C" void test_reset(){ffn_calls=ffn_separate=ffn_allocs=ffn_down=ffn_retired=0;ffn_gate=ffn_out=nullptr;ffn_n=0;`)
	nativeShims = replace(nativeShims, `case 10:return pool_faults;`, `case 10:return pool_faults;case 11:return ffn_calls;case 12:return ffn_separate;case 13:return ffn_allocs;case 14:return ffn_down;case 15:return ffn_retired;`)
	nativeShims = replace(nativeShims, `if(!g_cuda_cleanup_status){++retires;g_live.erase(p);}`, `if(!g_cuda_cleanup_status){if(p==ffn_gate){++ffn_retired;ffn_gate=nullptr;}++retires;g_live.erase(p);}`)
	// Execute the real pointwise kernel body with host block/thread indices.
	// This checks alias indexing and identical host arithmetic, not CUDA math.
	swiglu := section(read("cuda/transformer_ops.cuh"), "__global__ void k_swiglu(", "extern \"C\" void fcuda_swiglu_f32(")
	native := cudaPagedSequenceNativePreamble +
		swiglu + cudaPagedSequenceFFNAliasControls +
		section(kv, "static int kv_clone_copy_status(", "// Snapshot-only completion handling.") +
		section(kv, "static int kv_snapshot_status(", "extern \"C\" int fcuda_kv_snapshot_load_checked(") +
		section(read("cuda/paged_attention_vllm.cuh"), "__global__ void reshape_and_cache_run_kernel(", "// Current vLLM reshape_and_cache_kernel") +
		section(read("cuda/paged_attention_vllm.cuh"), "__global__ void paged_gather_rows_kernel(", "__global__ void paged_store_int_kernel(") +
		section(kv, "// BEGIN checked detached paged-sequence owner (#13881).", "// END checked detached paged-sequence owner (#13881).") + nativeShims
	dir := t.TempDir()
	header := cudaPagedSequenceFixtureHeader + "\n#ifdef __cplusplus\nextern \"C\" {\n#endif\nint test_swiglu_alias(void);\nint test_ffn_down(const float*,int);\n#ifdef __cplusplus\n}\n#endif\n"
	for name, data := range map[string]string{"main.go": body.String(), "fixture.h": header, "fixture.cc": native, "cuda_kv_precision.h": read("cuda_kv_precision.h")} {
		if e := os.WriteFile(filepath.Join(dir, name), []byte(data), 0600); e != nil {
			t.Fatal(e)
		}
	}
	t.Logf("actual entry/owner host program %x; native body fixture %x", sha256.Sum256([]byte(body.String())), sha256.Sum256([]byte(native)))
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	command := exec.CommandContext(ctx, filepath.Join(runtime.GOROOT(), "bin", "go"), "run", ".")
	command.Dir = dir
	command.Env = append(os.Environ(), "CGO_ENABLED=1", "GO111MODULE=off", "GOWORK=off")
	out, e := command.CombinedOutput()
	if e != nil {
		t.Fatalf("actual paged sequence host witness: %v\n%s", e, out)
	}
	if !strings.Contains(string(out), "actual NewKV/Clone/sequence + native write/gather/retirement controls PASS") {
		t.Fatalf("missing complete witness: %s", out)
	}
	t.Log(strings.TrimSpace(string(out)))
}

const cudaPagedSequenceFFNAliasControls = `
static int ffn_calls=0,ffn_separate=0,ffn_allocs=0,ffn_down=0,ffn_retired=0,ffn_n=0;
static const float *ffn_gate=nullptr,*ffn_out=nullptr;
static void run_swiglu(const float*g,const float*u,float*y,int n){
 blockDim=dim3(256);for(int b=0;b<(n+255)/256;++b){blockIdx=dim3(b);for(int t=0;t<256;++t){threadIdx=dim3(t);k_swiglu(g,u,y,n);}}
}
extern "C" int test_ffn_down(const float*p,int n){
 if(p!=ffn_out||n!=ffn_n||g_live.find((void*)p)==g_live.end())return 1;
 ++ffn_down;return 0;
}
extern "C" int test_swiglu_alias(){
 for(int n: {1,255,256,257,1025}){
  std::vector<float>g(n+1),u(n+1),y(n+1,12345.f);
  for(int i=0;i<n;++i){g[i]=(float)(i%37-18)/4;u[i]=(float)(i%29-14)/3;}
  g[n]=u[n]=12345.f;auto original=g,up=u;
  run_swiglu(g.data(),u.data(),y.data(),n);run_swiglu(g.data(),u.data(),g.data(),n);
  if(memcmp(g.data(),y.data(),n*sizeof(float))||memcmp(u.data(),up.data(),(n+1)*sizeof(float)))return 1;
  if(g[n]!=12345.f||u[n]!=12345.f||y[n]!=12345.f)return 2;
  // Explicit exceptional values use the same kernel body in both arms.
  for(float value: {0.f,-0.f,INFINITY,-INFINITY,NAN}){
   float a=value,b=2.f,c=0;run_swiglu(&a,&b,&c,1);run_swiglu(&a,&b,&a,1);
   if(!(std::isnan(a)&&std::isnan(c))&&memcmp(&a,&c,sizeof(float)))return 3;
  }
 }
 return 0;
}
`

const cudaPagedSequenceFixtureHeader = `
#include <stddef.h>
#include <stdlib.h>
#include "cuda_kv_precision.h"
#ifdef __cplusplus
extern "C" {
#endif
enum {FCUDA_SEQUENCE_SPLIT_CONTRACT=1,FCUDA_SEQUENCE_SPLIT_ALLOCATE=2};
typedef struct {int phase,code,allocation_code,allocation_was_oom;size_t failed_bytes;} fcuda_sequence_split_status;
void *test_alloc(size_t,int,const char*);
void test_mode(int,int);
void test_reset(void);
int test_counter(int);
float test_attention_value(int,int);
void test_finish_deferred(void);
void test_control_h2d(size_t);
int test_fence(void);
int test_retire_probe(void);
int fcuda_h2d_checked(void*,const void*,size_t);
void fcuda_d2d(void*,const void*,size_t);
void fcuda_rmsnorm_f32(const float*,const float*,float*,int,int,float);
void fcuda_free(void*);
int fcuda_paged_kv_copy_terminal(void);
int fak_v41_checked_stream_terminal(void);
unsigned long long fcuda_h2dxfer_bytes(void);
unsigned long long fcuda_hostxfer_bytes(void);
int fcuda_paged_kv_write_f32(float*,float*,float*,const float*,const float*,const float*,int,int,int,int,int,int);
int fcuda_paged_kv_gather_f32(float*,const float*,const int*,int,int,int,int,int);
int fak_qwen35_causal_attention_panel_f32(const float*,const float*,const float*,float*,int,int,int,int,int,float);
int fcuda_sequence_split_begin(size_t,size_t,unsigned long long*,fcuda_sequence_split_status*);
int fcuda_sequence_split_end(unsigned long long,fcuda_sequence_split_status*);
void fcuda_qwen35_iq2_sequence_retirement_failed(int);
void fcuda_add_f32(float*,const float*,int);
void fcuda_sigmoid_mul_f32(float*,const float*,int);
void fcuda_swiglu_f32(const float*,const float*,float*,int);
#ifdef __cplusplus
}
#endif
`

const cudaPagedSequenceNativePreamble = `
#include "fixture.h"
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <unordered_map>
#include <vector>
#include <string>
#define __global__
#define __restrict__
struct dim3 {int x,y,z;dim3(int a=1,int b=1,int c=1):x(a),y(b),z(c){}};
static dim3 blockIdx,threadIdx,blockDim;
#define FCUDA_TS(name,grid,body) do {body;} while(0)
typedef int cudaError_t;
const int cudaSuccess=0,cudaErrorInvalidValue=1,cudaErrorInvalidDevice=10,cudaErrorNotReady=34,cudaErrorMemoryAllocation=2;
typedef int cudaStreamCaptureStatus;
const int cudaStreamCaptureStatusNone=0;
static void *g_stream=nullptr;
static bool g_capture_open=false;
static int g_cuda_cleanup_status=0,mode=0,at=0,allocs=0,copies=0,gathers=0,writes=0,retires=0,fences=0,attention=0,pool_attempts=0,pool_faults=0;
static size_t h2d=0;
static std::vector<float> attention_first,attention_last;
static std::unordered_map<void*,size_t> g_live,g_managed_live,g_panel_live;
struct fak_alloc_tag{size_t bytes;int cls,kind,site;};
static std::unordered_map<void*,fak_alloc_tag> g_alloc_tags;
const int FAK_ALLOC_KV_CACHE=3,FAK_ALLOC_DEVICE_LOCAL=0,FAK_ALLOC_SITE_OVERFLOW=0,FAK_ALLOC_SITE_CAP=512;
struct site{char name[128];};static site g_alloc_sites[512];static std::atomic<int> g_alloc_site_n{1};
struct fak_alloc_tags_guard{};
static std::vector<void*> all;
struct pending{void*d;const void*s;size_t n;};static std::vector<pending> queued;
static bool v41_candidates_stream_uncertain(){return g_cuda_cleanup_status!=0;}
static bool kv_snapshot_release_refused(void*){return mode==12;}
static cudaError_t cudaGetDevice(int*d){if(mode==10)return 61;*d=0;return 0;}
static cudaError_t cudaStreamIsCapturing(void*,cudaStreamCaptureStatus*s){if(mode==11)return 62;*s=0;return 0;}
static const int cudaMemcpyDeviceToDevice=1;
static cudaError_t cudaMemcpyAsync(void*d,const void*s,size_t n,int,void*){
 ++copies;if(mode==2&&copies==at)return 52;
 if(mode==3){queued.push_back({d,s,n});return 0;}
 memcpy(d,s,n);if(mode==4&&copies==at)g_cuda_cleanup_status=54;return 0;
}
// Pool fault selection is independent of generic transient frees counted in retires.
static void fcuda_pool_put(void*p,size_t){++retires;++pool_attempts;if(mode==7&&pool_attempts==at){++pool_faults;g_cuda_cleanup_status=57;return;} (void)p;}
`

const cudaPagedSequenceNativeShims = `
extern "C" void *test_alloc(size_t n,int cls,const char*name){
 ++allocs;if(mode==1&&allocs==at)return nullptr;
 void*p=calloc(1,n);if(!p)return nullptr;all.push_back(p);g_live[p]=n;
 int s=1;for(;s<g_alloc_site_n.load();++s)if(!strcmp(g_alloc_sites[s].name,name))break;
 if(s==g_alloc_site_n.load()){if(s>=512)abort();strncpy(g_alloc_sites[s].name,name,127);g_alloc_site_n.store(s+1);}g_alloc_tags[p]={n,cls,0,s};return p;
}
extern "C" void test_mode(int m,int a){mode=m;at=a;}
extern "C" void test_reset(){for(void*p:all)free(p);all.clear();queued.clear();attention_first.clear();attention_last.clear();g_live.clear();g_managed_live.clear();g_panel_live.clear();g_alloc_tags.clear();g_alloc_site_n.store(1);g_cuda_cleanup_status=mode=at=allocs=copies=gathers=writes=retires=fences=attention=pool_attempts=pool_faults=0;h2d=0;}
extern "C" int test_counter(int n){switch(n){case 0:return allocs;case 1:return copies;case 2:return writes;case 3:return gathers;case 4:return retires;case 5:return fences;case 6:return attention;case 7:return g_cuda_cleanup_status;case 8:return (int)queued.size();case 9:return pool_attempts;case 10:return pool_faults;default:return (int)g_live.size();}}
extern "C" float test_attention_value(int call,int last){return last?attention_last.at(call):attention_first.at(call);}
extern "C" void test_finish_deferred(){for(auto p:queued)memcpy(p.d,p.s,p.n);queued.clear();}
extern "C" int test_fence(){++fences;if(mode==3)return 53;test_finish_deferred();return g_cuda_cleanup_status;}
extern "C" void test_control_h2d(size_t n){h2d+=n;}
extern "C" int fcuda_h2d_checked(void*d,const void*s,size_t n){if(mode==5)return 55;memcpy(d,s,n);h2d+=n;return 0;}
extern "C" void fcuda_d2d(void*d,const void*s,size_t n){memcpy(d,s,n);}
// Explicit identity norm model, matching the existing full-panel Go shim.
extern "C" void fcuda_rmsnorm_f32(const float*x,const float*,float*y,int rows,int n,float){memcpy(y,x,(size_t)rows*n*sizeof(float));}
extern "C" void fcuda_free(void*p){if(!g_cuda_cleanup_status){++retires;g_live.erase(p);}}
extern "C" int fcuda_paged_kv_copy_terminal(){return g_cuda_cleanup_status;}
extern "C" int fak_v41_checked_stream_terminal(){return g_cuda_cleanup_status;}
extern "C" unsigned long long fcuda_h2dxfer_bytes(){return h2d;}
extern "C" unsigned long long fcuda_hostxfer_bytes(){return 0;}
extern "C" int fcuda_paged_kv_write_f32(float*k,float*v,float*r,const float*ks,const float*vs,const float*rs,int block,int off,int tokens,int nh,int hd,int bs){
 ++writes;if(mode==6&&writes==at)return 56;
 blockDim=dim3(256);for(int t=0;t<tokens;++t){blockIdx=dim3(t);for(int lane=0;lane<256;++lane){threadIdx=dim3(lane);reshape_and_cache_run_kernel(ks,vs,rs,k,v,r,block,off,nh,hd,bs,4);}}return 0;
}
extern "C" int fcuda_paged_kv_gather_f32(float*d,const float*s,const int*table,int tokens,int nh,int hd,int bs,int kl){
 ++gathers;if(mode==8&&gathers==at)return 58;
 blockDim=dim3(256);for(int t=0;t<tokens;++t){blockIdx=dim3(t);for(int lane=0;lane<256;++lane){threadIdx=dim3(lane);paged_gather_rows_kernel(d,s,table,tokens,nh,hd,bs,4,kl);}}return 0;
}
// Explicit independent scalar attention oracle, not the CUDA attention kernel.
extern "C" int fak_qwen35_causal_attention_panel_f32(const float*q,const float*k,const float*v,float*out,int tokens,int prefix,int nh,int nk,int hd,float scale){
 ++attention;for(int t=0;t<tokens;++t)for(int h=0;h<nh;++h){int kh=h/(nh/nk),n=prefix+t+1;std::vector<double>s(n);double m=-1e300,sum=0;for(int j=0;j<n;++j){double d=0;for(int x=0;x<hd;++x)d+=q[(t*nh+h)*hd+x]*k[(j*nk+kh)*hd+x];s[j]=d*scale;if(s[j]>m)m=s[j];}for(double &x:s){x=exp(x-m);sum+=x;}for(int x=0;x<hd;++x){double d=0;for(int j=0;j<n;++j)d+=s[j]*v[(j*nk+kh)*hd+x];out[(t*nh+h)*hd+x]=(float)(d/sum);}}attention_first.push_back(out[0]);attention_last.push_back(out[(tokens-1)*nh*hd]);return 0;
}
extern "C" int fcuda_sequence_split_begin(size_t,size_t,unsigned long long*t,fcuda_sequence_split_status*s){*t=1;*s={};return 0;}
extern "C" int fcuda_sequence_split_end(unsigned long long,fcuda_sequence_split_status*s){*s={};if(mode==9){s->code=59;return 59;}return test_fence();}
extern "C" void fcuda_qwen35_iq2_sequence_retirement_failed(int n){if(n&&!g_cuda_cleanup_status)g_cuda_cleanup_status=n;}
extern "C" void fcuda_add_f32(float*x,const float*y,int n){for(int i=0;i<n;++i)x[i]+=y[i];}
extern "C" void fcuda_sigmoid_mul_f32(float*x,const float*y,int n){for(int i=0;i<n;++i)x[i]/=(1+expf(-y[i]));}
extern "C" void fcuda_swiglu_f32(const float*x,const float*y,float*z,int n){for(int i=0;i<n;++i)z[i]=x[i]*y[i]/(1+expf(-x[i]));}
extern "C" int test_retire_probe(){
 test_reset();void*p=test_alloc(64,3,"qwen35-paged-table");int d=-1;
 if(fcuda_paged_kv_owner_checked(p,63)==0||fcuda_paged_kv_owner_checked((char*)p+4,60)==0)return 1;
 g_alloc_tags[p].cls=0;if(fcuda_paged_kv_owner_checked(p,64)==0)return 2;g_alloc_tags[p].cls=3;
 test_mode(10,0);if(fcuda_paged_kv_owner_checked(p,64)!=61||g_cuda_cleanup_status!=61)return 3;
 test_reset();p=test_alloc(64,3,"qwen35-paged-table");test_mode(7,1);
 if(fcuda_paged_kv_release_checked(p,64,&d)!=57||d!=FCUDA_PAGED_KV_RETIRED||g_live.count(p)||pool_attempts!=1||pool_faults!=1)return 4;
 if(fcuda_paged_kv_release_checked(p,64,&d)!=57||d!=FCUDA_PAGED_KV_RETAINED||retires!=1||pool_attempts!=1||pool_faults!=1)return 5;
 test_reset();return 0;
}
`

const cudaPagedSequenceGoPreamble = `package main
/*
#cgo CXXFLAGS: -std=c++17
#cgo LDFLAGS: -lstdc++ -lm
#include "fixture.h"
*/
import "C"
import("errors";"fmt";"math";"os";"runtime";"strings";"sync";"sync/atomic";"time";"unsafe")
type Backend interface{}
type Buffer interface{}
type Dtype int
const(F32 Dtype=iota;F16;Q8_0;Q4_K;Q5_K;Q6_K;Q2_K;Q3_K;IQ4_XS;Q2_0;IQ2_XXS;IQ2_XS)
func(d Dtype)Bytes()int{if d==F32{return 4};if d==F16{return 2};return 1}
type Layout int
const RowMajor Layout=0
type QuantSpec struct{Block int}
type Tensor struct{Dtype Dtype;Layout Layout;Shape []int;Quant *QuantSpec;buf Buffer;be Backend}
func(t Tensor)Buf()Buffer{return t.buf}
func(t Tensor)Backend()Backend{return t.be}
func(t Tensor)Numel()int{n:=1;for _,d:=range t.Shape{n*=d};return n}
type MemoryClass int
const(MemoryKVCache MemoryClass=3;MemoryScratchpad MemoryClass=5)
type cudaBuf struct{ptr,scales unsafe.Pointer;n,device,scalesN int;managed bool;class MemoryClass;invalid uint32}
func(b *cudaBuf)invalidStateError(string)error{if b.invalid!=0{return errors.New("invalid")};return nil}
type KVConfig struct{NumLayers,NumKVHeads,HeadDim int;Precision int}
const KVPrecisionF32=0
type KVStore interface{Len()int}
func cloneKVConfig(k KVConfig)KVConfig{return k}
type dslice struct{ptr unsafe.Pointer;len,cap int;dtype Dtype}
type cudaKV struct{be *cudaBackend;cfg KVConfig;K,Kraw,V []dslice;pos []int}
func(k *cudaKV)Len()int{return len(k.pos)}
func(k *cudaKV)stride()int{return k.cfg.NumKVHeads*k.cfg.HeadDim}
type cudaPagedKVSnapshotCopy struct{}
var cudaPagedKVCopyQualification bool
var cudaMu sync.Mutex
var cudaKernelProfileOn bool
var cudaDev *cudaBackend
var qwen35SequenceIQ2Qualification,cudaQ56KMMQQualification bool
const(qwen35GDNMaxCInt=2147483647;qwen35GDNPanelMaxKeyHeadDim=256;qwen35GDNPanelKeyHeadReason="bounded";q8DeviceBlock=32;q3kSuperBlock=110;iq4xsSuperBlock=136)
const(Qwen35SequencePrefillPath="qwen35-hybrid-sequence-prefill-v1";Qwen35SequenceRawHiddenPath="raw")
const(Qwen35DenseMainLayers=64;Qwen35DenseHidden=5120;Qwen35DenseIntermediate=17408;Qwen35DenseQueryHeads=24;Qwen35DenseKVHeads=4;Qwen35DenseHeadDim=256;Qwen35DenseGDNGroups=16;Qwen35DenseGDNRank=48;Qwen35DenseGDNState=128;Qwen35DenseGDNInner=6144;Qwen35DenseGDNConv=4)
const(cudaQwen35IQ2SequenceOp="iq2";cudaQ56KMMQOp="q56";DeviceFaultExecution=1)
type CUDAOpError struct{Op,Site,Msg string;Code int;Class MemoryClass}
func(e *CUDAOpError)Error()string{return fmt.Sprintf("%s/%s %d",e.Op,e.Site,e.Code)}
type latch struct{bad bool}
func(l *latch)Observe(int,string,int){l.bad=true}
func(l *latch)ObserveError(error,string){l.bad=true}
func(l *latch)Admit(string)error{if l.bad{return errors.New("stopped")};return nil}
func AdmitDevice(c *cudaBackend,_ string)error{return c.faultLatch.Admit("")}
type cudaBackend struct{capturing,promptPanel bool;graphOnce interface{};faultLatch *latch;transient []*cudaBuf;fenceGen uint64;gdn int;panicAlloc,exitAlloc bool;cleanupMode int}
type locks struct{}
func(locks)observe(time.Duration,time.Duration){}
var cudaQwen35SequenceLocks locks
var splitSelected bool
func CUDAQ4KPrefillArmInUse()(int,error){return 0,nil}
type splitPlan struct{weightHalfBytes,activationHalfBytes int}
func planCUDASequenceSplitScratch(Qwen35SequencePrefillRequest,int,error)(splitPlan,error){if splitSelected{return splitPlan{1,1},nil};return splitPlan{},nil}
func cudaSequenceSplitStatusError(s C.fcuda_sequence_split_status)error{return &CUDAOpError{Op:"split",Code:int(s.code)}}
func cudaQwen35IQ2SequenceRetirementFailedLocked(n int){C.fcuda_qwen35_iq2_sequence_retirement_failed(C.int(n))}
`

const cudaPagedSequenceGoShims = `
func qwen35GDNCheckedMul(limit int64,values ...int64)(int64,bool){n:=int64(1);for _,v:=range values{if v<=0||n>limit/v{return 0,false};n*=v};return n,true}
func qwen35GDNShapeBytes(shape []int,bytes int)(int,bool){v:=[]int64{int64(bytes)};for _,x:=range shape{v=append(v,int64(x))};n,ok:=qwen35GDNCheckedMul(int64(^uint(0)>>1),v...);return int(n),ok}
func qwen35GDNSameShape(a,b []int)bool{if len(a)!=len(b){return false};for i:=range a{if a[i]!=b[i]{return false}};return true}
func validateQwen35CausalAttentionPanelGeometry(t,p,h,k,d int)error{if t<1||p<0||h!=24||k!=4||d!=256||int64(t+p)*int64(k)*int64(d)>qwen35GDNMaxCInt{return errors.New("shape")};return nil}
func validateQwen35CausalAttentionPanelScale(s float32)error{if s<=0||math.IsNaN(float64(s))||math.IsInf(float64(s),0){return errors.New("scale")};return nil}
func qwen35SequenceIQ2MatrixBytes(Tensor)(int,bool){return 0,false}
func validateQwen35SequenceIQ2(t Tensor,_ int,_ bool)error{if t.Dtype!=F32{return errors.New("fixture supports F32")};return nil}
func qwen35GDNPackedInput(Tensor)bool{return false}
func(c *cudaBackend)validateQwen35GDNPackedInput(_ string,t Tensor,_ int)error{if t.Dtype!=F32{return errors.New("fixture supports F32")};return nil}
func(c *cudaBackend)validateQ56KOperation(Qwen35SequencePrefillRequest,int,int)error{return nil}
func(c *cudaBackend)newContiguousKV(cfg KVConfig)*cudaKV{return &cudaKV{be:c,cfg:cfg}}
func(c *cudaBackend)dallocDeviceOnlyClass(n int,cls MemoryClass,site string)(*cudaBuf,error){
 if c.panicAlloc{panic("allocation panic")};if c.exitAlloc{runtime.Goexit()}
 name:=C.CString(site);defer C.free(unsafe.Pointer(name));p:=C.test_alloc(C.size_t(n),C.int(cls),name)
 if p==nil{return nil,errors.New("allocation refused")};return &cudaBuf{ptr:p,n:n,class:cls},nil
}
func(c *cudaBackend)tensor(shape []int,cls MemoryClass,site string)(Tensor,*cudaBuf,error){n,ok:=qwen35GDNShapeBytes(shape,4);if !ok{return Tensor{},nil,errors.New("bytes")};b,e:=c.dallocDeviceOnlyClass(n,cls,site);return Tensor{Dtype:F32,Shape:append([]int(nil),shape...),buf:b,be:c},b,e}
func(c *cudaBackend)qwen35SequenceAllocLocked(shape []int,site string)(Tensor,*cudaBuf,error){t,b,e:=c.tensor(shape,MemoryScratchpad,site);if e==nil{c.transient=append(c.transient,b)};return t,b,e}
func(c *cudaBackend)cf(t Tensor)*C.float{return (*C.float)(t.buf.(*cudaBuf).ptr)}
func offsetF(p unsafe.Pointer,n int)unsafe.Pointer{return unsafe.Add(p,n*4)}
func(c *cudaBackend)qwen35SequenceEmbeddingLocked(_ Tensor,ids []int,h int)(Tensor,*cudaBuf,error){t,b,e:=c.qwen35SequenceAllocLocked([]int{len(ids),h},"embedding");if e==nil{C.test_control_h2d(C.size_t(len(ids)*4))};return t,b,e}
func(c *cudaBackend)qwen35SequenceRMSNormLocked(x,w Tensor,rows int,eps float32,site string)(Tensor,*cudaBuf,error){t,b,e:=c.qwen35SequenceAllocLocked(x.Shape,site);if e==nil{C.fcuda_d2d(b.ptr,x.buf.(*cudaBuf).ptr,C.size_t(b.n))};return t,b,e}
func(c *cudaBackend)qwen35SequenceGDNLocked(x Tensor,_ Qwen35SequenceLayer,s Qwen35SequenceState,_ Qwen35SequencePrefillRequest,tokens,layer int)(Tensor,*cudaBuf,error){c.gdn++;*(*float32)(s.Conv.buf.(*cudaBuf).ptr)+=1;*(*float32)(s.Recurrent.buf.(*cudaBuf).ptr)+=1;return c.qwen35SequenceRMSNormLocked(x,Tensor{},tokens,0,"gdn-standin")}
func(c *cudaBackend)qwen35SequenceResidualNormLocked(x,b,w Tensor,t int,e float32)(Tensor,*cudaBuf,bool,error){return Tensor{},nil,false,nil}
func(c *cudaBackend)qwen35SequenceMatMulLocked(w,x Tensor,tokens int,site string)(Tensor,*cudaBuf,error){t,b,e:=c.qwen35SequenceAllocLocked([]int{tokens,w.Shape[0]},site);if e==nil&&site=="attention-v"{for i:=range unsafe.Slice((*float32)(b.ptr),t.Numel()){unsafe.Slice((*float32)(b.ptr),t.Numel())[i]=float32(i%1024+1)/128}};return t,b,e}
func(c *cudaBackend)qwen35SequenceSplitQGLocked(_ Tensor,t,h,d int)(Tensor,*cudaBuf,Tensor,*cudaBuf,error){q,qb,e:=c.qwen35SequenceAllocLocked([]int{t,h*d},"query");if e!=nil{return q,qb,Tensor{},nil,e};g,gb,e:=c.qwen35SequenceAllocLocked([]int{t,h*d},"gate");return q,qb,g,gb,e}
func(c *cudaBackend)qwen35SequenceAttentionPostprocessLocked(q,k,v,qw,kw Tensor,kv *cudaKV,l,t,s,r int,e float32,theta float64)(Tensor,*cudaBuf,bool,error){if kv==nil{panic("paged route reached contiguous fusion")};return Tensor{},nil,false,nil}
func(c *cudaBackend)qwen35SequenceRoPELocked(q,k Tensor,t,s,qh,kh,hd,rd int,theta float64)(Tensor,*cudaBuf,Tensor,*cudaBuf,error){a,ab,e:=c.qwen35SequenceRMSNormLocked(q,Tensor{},t,0,"rope-q-standin");if e!=nil{return a,ab,Tensor{},nil,e};b,bb,e:=c.qwen35SequenceRMSNormLocked(k,Tensor{},t,0,"rope-k-standin");return a,ab,b,bb,e}
func(c *cudaBackend)qwen35SequenceReserveKVLocked(*cudaKV,int,int)error{panic("unexpected contiguous reserve")}
func(c *cudaBackend)qwen35SequenceAppendKVLocked(*cudaKV,int,Tensor,Tensor,Tensor,int,int)error{panic("unexpected contiguous append")}
func(c *cudaBackend)qwen35SequenceAttentionLocked(Tensor,*cudaKV,int,int,int,int,float32)(Tensor,*cudaBuf,error){panic("unexpected contiguous attention")}
func(c *cudaBackend)qwen35SequenceSyncLocked()error{n:=int(C.test_fence());if n!=0{C.fcuda_kv_snapshot_failed(C.int(n));c.faultLatch.Observe(1,"fence",n);return &CUDAOpError{Op:"fence",Code:n}};atomic.AddUint64(&c.fenceGen,1);return nil}
func(c *cudaBackend)releaseTransientBuffers(list []*cudaBuf){
 if c.cleanupMode!=0&&C.test_counter(5)>=2{m:=c.cleanupMode;c.cleanupMode=0;switch m{case 1:panic("final cleanup panic");case 2:runtime.Goexit();case 3:C.fcuda_kv_snapshot_failed(63)}}
 if C.fak_v41_checked_stream_terminal()!=0{return};for _,b:=range list{if b!=nil&&b.ptr!=nil{C.fcuda_free(b.ptr);b.ptr=nil}};keep:=c.transient[:0];for _,b:=range c.transient{if b.ptr!=nil{keep=append(keep,b)}};c.transient=keep}
func reqFor(c *cudaBackend,k *cudaPagedKV,start,tokens int)Qwen35SequencePrefillRequest{
 makeT:=func(cls MemoryClass,shape ...int)Tensor{t,_,e:=c.tensor(shape,cls,"fixture-weight");if e!=nil{panic(e)};return t}
 r:=Qwen35SequencePrefillRequest{Path:Qwen35SequencePrefillPath,TokenIDs:make([]int,tokens),StartPos:start,Hidden:4,Intermediate:4,NumHeads:24,NumKVHeads:4,HeadDim:256,RotaryDim:64,NumKeyHeads:1,NumValueHeads:1,KeyHeadDim:8,ValueHeadDim:8,ConvKernel:3,RMSNormEpsilon:1e-6,KV:k,NeedLogits:true}
 r.TokenEmbedding=makeT(0,2,4);r.OutputNorm=makeT(0,4);r.Output=makeT(0,2,4)
 r.Layers=make([]Qwen35SequenceLayer,8);r.States=make([]Qwen35SequenceState,8);r.RoPEThetaForLayer=make([]float64,8)
 for i:=range r.Layers{l:=&r.Layers[i];l.Linear=(i+1)%4!=0;r.RoPEThetaForLayer[i]=1e6;l.InputNorm=makeT(0,4);l.PostNorm=makeT(0,4);l.Gate=makeT(0,4,4);l.Up=makeT(0,4,4);l.Down=makeT(0,4,4)
  if l.Linear{l.GDNInQKV=makeT(0,24,4);l.GDNInZ=makeT(0,8,4);l.GDNInA=makeT(0,1,4);l.GDNInB=makeT(0,1,4);l.GDNConv=makeT(0,24*3);l.GDNALog=makeT(0,1);l.GDNDTBias=makeT(0,1);l.GDNNorm=makeT(0,8);l.GDNOut=makeT(0,4,8);r.States[i]=Qwen35SequenceState{makeT(MemoryKVCache,2,24),makeT(MemoryKVCache,1,8,8)}}else{l.Q=makeT(0,12288,4);l.K=makeT(0,1024,4);l.V=makeT(0,1024,4);l.O=makeT(0,4,6144);l.QNorm=makeT(0,256);l.KNorm=makeT(0,256)}
 }
 return r
}
// Each reset destroys a synchronous fake-C world, not a real CUDA context.
// Archive failed Go owners first: their deliberately retained table pins must
// remain reachable through subprocess exit. Never execute archived operations,
// Unpin uncertain transfers, or report this as successful native retirement.
type fixtureQuarantinedOwner struct{owner *cudaQwen35PagedSequence;table *int32;pinned bool;length int;lens []int;terminal int32}
var fixtureQuarantine []fixtureQuarantinedOwner
func checkFixtureQuarantine()int{
 if len(fixtureQuarantine)>0{runtime.GC()}
 pins:=0
 for _,q:=range fixtureQuarantine{p:=q.owner;var table *int32;if len(p.table)>0{table=&p.table[0]}
  must(p.started&&!p.completed&&p.pinned==q.pinned&&table==q.table,"archived owner/table pin changed after GC")
  must(len(p.k.pos)==q.length&&len(p.k.lens)==len(q.lens)&&p.k.pool.copyFailure.Load()==q.terminal&&q.terminal!=0,"archived terminal/frontier changed after GC")
  for i,n:=range q.lens{must(p.k.lens[i]==n,"archived layer frontier changed after GC")}
  if p.pinned{pins++}
 }
 return pins
}
func resetFixtureWorld()int{
 for p:=cudaQwen35PagedOwners;p!=nil;p=p.next{var table *int32;if len(p.table)>0{table=&p.table[0]};fixtureQuarantine=append(fixtureQuarantine,fixtureQuarantinedOwner{p,table,p.pinned,len(p.k.pos),append([]int(nil),p.k.lens...),p.k.pool.copyFailure.Load()})}
 // All fake work is inline or represented by the queued vector that reset
 // discards. The archived Go owners are never reused after native world reset.
 C.test_reset();cudaQwen35PagedOwners=nil;cudaPagedPools=map[cudaPagedPoolKey]*cudaKVPagePool{};cudaDev=nil
 return checkFixtureQuarantine()
}
func reset()(*cudaBackend,*cudaPagedKV){resetFixtureWorld();splitSelected=false;cudaKernelProfileOn=false;qwen35SequenceIQ2Qualification=false;cudaQ56KMMQQualification=false;os.Setenv("FAK_CUDA_PAGED_KV","1");c:=&cudaBackend{faultLatch:&latch{}};cudaDev=c;return c,c.NewKV(KVConfig{NumLayers:2,NumKVHeads:4,HeadDim:256}).(*cudaPagedKV)}
func must(v bool,s string){if !v{panic(s)}}
func rows(c *cudaBackend,n,width,seed int)Tensor{t,b,e:=c.tensor([]int{n,width},MemoryScratchpad,"fixture-rows");if e!=nil{panic(e)};a:=unsafe.Slice((*float32)(b.ptr),n*width);for i:=range a{a[i]=float32(seed+i%width)};return t}
func preparePrefix(c *cudaBackend,k *cudaPagedKV,n int){r:=reqFor(c,k,0,n);p:=&cudaQwen35PagedSequence{c:c,k:k};must(p.validateHostLocked(r)==nil,"prefix validation");must(p.prepareLocked()==nil,"prefix reserve");for l:=0;l<2;l++{x:=rows(c,n,1024,l+1);must(p.appendLocked(l,x,x,x)==nil,"prefix write")};must(c.qwen35SequenceSyncLocked()==nil,"prefix fence");p.drained=true;p.executing=false;must(p.completeLocked()==nil,"prefix commit")}
`

const cudaPagedSequenceGoControls = `
func checkHeadlessPrefix(){
 c,k:=reset();r:=reqFor(c,k,0,2);r.Output=Tensor{};r.NeedLogits=false
 result,e:=c.Qwen35SequencePrefill(r)
 must(e==nil&&result.Tokens==2&&result.LastHidden.Buf()!=nil&&result.Logits.Buf()==nil&&result.LogitsRows.Buf()==nil,"actual headless paged prefix")
 must(k.Len()==2&&k.lens[0]==2&&k.lens[1]==2&&c.gdn==6&&C.test_counter(6)==2,"headless prefix did not execute full stateful stack")
 must(cudaQwen35PagedOwners==nil&&C.test_counter(7)==0,"headless prefix did not retire successfully")
 // Continue the same recurrent tensors and KV with a real requested head.
 next:=reqFor(c,k,2,1);next.States=r.States;next.Layers=r.Layers;next.TokenEmbedding=r.TokenEmbedding;next.OutputNorm=r.OutputNorm
 continued,e:=c.Qwen35SequencePrefill(next)
 must(e==nil&&continued.Tokens==1&&continued.Logits.Buf()!=nil&&k.Len()==3&&k.lens[0]==3&&k.lens[1]==3,"logits continuation after headless prefix")
 for i,l:=range r.Layers{if l.Linear{must(*(*float32)(r.States[i].Conv.buf.(*cudaBuf).ptr)==2&&*(*float32)(r.States[i].Recurrent.buf.(*cudaBuf).ptr)==2,"headless continuation replaced or lost recurrent state")}}
 must(cudaQwen35PagedOwners==nil&&C.test_counter(7)==0,"headless continuation did not retire successfully")
 for _,mode:=range []string{"last-logits","all-logits","supplied-malformed-head"}{
  c,k=reset();r=reqFor(c,k,0,2);r.NeedLogits=false
  if mode=="supplied-malformed-head"{r.Output.Shape=[]int{1,8}}else{r.Output=Tensor{};r.NeedLogits=mode=="last-logits";r.NeedAllLogits=mode=="all-logits"}
  allocBefore,copyBefore,writeBefore,gatherBefore:=C.test_counter(0),C.test_counter(1),C.test_counter(2),C.test_counter(3)
  got,e:=c.Qwen35SequencePrefill(r)
  must(e!=nil&&strings.Contains(e.Error(),"output")&&got.Tokens==0,"required or supplied head escaped validation: "+mode)
  must(C.test_counter(0)==allocBefore&&C.test_counter(1)==copyBefore&&C.test_counter(2)==writeBefore&&C.test_counter(3)==gatherBefore&&c.gdn==0,"head refusal occurred after effects: "+mode)
  must(k.Len()==0&&k.lens[0]==0&&k.lens[1]==0&&cudaQwen35PagedOwners==nil,"head refusal changed KV ownership: "+mode)
  for i,l:=range r.Layers{if l.Linear{must(*(*float32)(r.States[i].Conv.buf.(*cudaBuf).ptr)==0&&*(*float32)(r.States[i].Recurrent.buf.(*cudaBuf).ptr)==0,"head refusal changed recurrent state: "+mode)}}
 }
 fmt.Println("headless prefix, logits continuation and pre-effect head refusals PASS")
}
func main(){
 must(C.test_retire_probe()==0,"actual native provenance/disposition controls")
 checkHeadlessPrefix()
 c,k:=reset();r:=reqFor(c,k,0,18);before:=uint64(C.fcuda_h2dxfer_bytes());result,e:=c.Qwen35SequencePrefill(r)
 must(e==nil&&result.Tokens==18&&k.Len()==18&&k.lens[0]==18&&k.lens[1]==18,"actual initial sequence")
 must(c.gdn==6&&C.test_counter(6)==2,"actual full layer consumer")
 for l:=0;l<2;l++{must(float32(C.test_attention_value(C.int(l),0))==1.0/128&&float32(C.test_attention_value(C.int(l),1))==1.0/128,"initial layer V layout/attention consumer")}
 must(result.Transfers.H2DBytes==80&&uint64(C.fcuda_h2dxfer_bytes())-before==80&&result.Transfers.ActivationH2DBytes==0&&result.Transfers.ActivationD2HBytes==0,"actual control-only transfer accounting")
 must(cudaQwen35PagedOwners==nil&&C.test_counter(7)==0,"success owner retirement")
 oldK:=k.pool.K[0];clone:=k.Clone().(*cudaPagedKV);oldPage:=k.table[1]
 r=reqFor(c,clone,18,2);_,e=c.Qwen35SequencePrefill(r)
 must(e==nil&&clone.Len()==20&&k.Len()==18&&clone.table[1]!=oldPage&&k.table[1]==oldPage&&k.pool.K[0]==oldK,"actual partial-tail Clone COW")
 // A larger existing prefix drives real shared-pool growth with a live clone.
 c,k=reset();preparePrefix(c,k,1023);clone=k.Clone().(*cudaPagedKV);oldK=k.pool.K[0];oldTail:=k.table[len(k.table)-1]
 r=reqFor(c,clone,1023,2);_,e=c.Qwen35SequencePrefill(r)
 must(e==nil&&k.Len()==1023&&clone.Len()==1025&&k.pool.K[0]!=oldK&&k.table[len(k.table)-1]==oldTail&&clone.table[len(k.table)-1]!=oldTail,"shared arena growth and COW")
 for l:=0;l<2;l++{first:=float64(float32((1023*float64(l+1)+1.0/128)/1024));last:=float64(float32((1023*float64(l+1)+2.0/128)/1025));must(math.Abs(float64(C.test_attention_value(C.int(l),0))-first)<1e-6&&math.Abs(float64(C.test_attention_value(C.int(l),1))-last)<1e-6,"grown COW actual V consumer/prefix visibility")}
 // Read untouched sibling data through the real gather kernel after growth.
 out,ob,_:=c.tensor([]int{1023,1024},MemoryScratchpad,"oracle");_ = out
 must(C.fcuda_paged_kv_snapshot_gather_checked((*C.float)(ob.ptr),(*C.float)(k.pool.K[0]),(*C.int)(k.dTable),1023,4,256,16,1)==0,"sibling gather")
 must(*(*float32)(ob.ptr)==1&&unsafe.Slice((*float32)(ob.ptr),1023*1024)[1022*1024+7]==8,"growth preserved every sibling page")
 // Pure malformed/profile/capture checks do not allocate or touch GDN state.
 for _,mode:=range []string{"profile","capture","frontier","alias"}{c,k=reset();r=reqFor(c,k,0,2);p:=&cudaQwen35PagedSequence{c:c,k:k};switch mode{case "profile":cudaKernelProfileOn=true;case "capture":c.capturing=true;case "frontier":k.lens[1]=1;case "alias":preparePrefix(c,k,1);r=reqFor(c,k,1,2);r.Output.buf=&cudaBuf{ptr:k.pool.K[0],n:32}}
  before:=C.test_counter(0);must(p.validateHostLocked(r)!=nil,"preflight "+mode);must(C.test_counter(0)==before&&!p.started,"preflight has effects")}
 // A separate primary allocation cannot hide a scale side-channel alias.
 for _,dt:=range []Dtype{Q8_0,Q2_0}{c,k=reset();preparePrefix(c,k,1);r=reqFor(c,k,1,2);w,b,_:=c.tensor([]int{2,32},MemoryScratchpad,"scale-probe");w.Dtype=dt;w.Quant=&QuantSpec{Block:32};b.scales=unsafe.Add(k.pool.K[0],4);b.scalesN=8;r.Output=w;p:=&cudaQwen35PagedSequence{c:c,k:k};before:=C.test_counter(0);e:=p.validateHostLocked(r);must(e!=nil&&strings.Contains(e.Error(),"operand scale aliases")&&C.test_counter(0)==before&&!p.started,"scale-only alias must refuse without effects")}
 for _,shape:=range []string{"nil-positive","nonnull-zero","negative","short"}{c,k=reset();r=reqFor(c,k,0,2);w,b,_:=c.tensor([]int{2,32},MemoryScratchpad,"scale-bound-probe");_,scale,_:=c.tensor([]int{2},MemoryScratchpad,"separate-scales");w.Dtype=Q8_0;w.Quant=&QuantSpec{Block:32};b.scales=scale.ptr;b.scalesN=8;switch shape{case "nil-positive":b.scales=nil;case "nonnull-zero":b.scalesN=0;case "negative":b.scalesN=-1;case "short":b.scalesN=4};r.Output=w;p:=&cudaQwen35PagedSequence{c:c,k:k};before:=C.test_counter(0);must(p.validateHostLocked(r)!=nil&&C.test_counter(0)==before&&!p.started,"invalid scale extent accepted: "+shape)}
 // Final cleanup still belongs to the actual entry transaction, before commit.
 for _,mode:=range []int{1,2,3}{c,k=reset();r=reqFor(c,k,0,2);c.cleanupMode=mode
  if mode==3{got,e:=c.Qwen35SequencePrefill(r);must(e!=nil&&got.Tokens==0&&C.test_counter(7)==63,"cleanup latch published result")}else{done:=make(chan interface{},1);returned:=false;go func(){defer func(){done<-recover()}();_,_=c.Qwen35SequencePrefill(r);returned=true}();select{case recovered:=<-done:must(!returned,"abnormal final cleanup returned");if mode==1{must(recovered=="final cleanup panic","cleanup panic changed")}else{must(recovered==nil,"cleanup Goexit became panic")};case <-time.After(2*time.Second):panic("final cleanup hung")}}
  must(k.Len()==0&&k.lens[0]==0&&k.lens[1]==0&&cudaQwen35PagedOwners!=nil&&cudaQwen35PagedOwners.k==k&&k.pool.copyFailure.Load()!=0,"cleanup lost retained transaction or published KV")
 }
 // Existing batch-facing validation still refuses paged stores before effects.
 c,k=reset();r=reqFor(c,k,0,1);beforeA:=C.test_counter(0);_,e=c.validateQwen35SequenceRequestLocked(r);must(e!=nil&&C.test_counter(0)==beforeA,"batch contract changed")
 // Host allocation/launch errors and accepted-but-unfinished copies cannot
 // publish positions, free exact old roots, or permit direct Clone/Free retry.
 for _,mode:=range []int{1,2,3,4,5,6,7,8,9,10,11,12}{
  c,k=reset();preparePrefix(c,k,14);clone=k.Clone().(*cudaPagedKV);r=reqFor(c,clone,14,3)
  oldTable:=append([]int32(nil),clone.table...);oldArena:=clone.pool.K[0]
  target:=1;switch mode{case 1:target=int(C.test_counter(0))+1;case 2,4:target=int(C.test_counter(1))+1;case 6:target=int(C.test_counter(2))+1;case 7:target=int(C.test_counter(9))+1;case 8:target=int(C.test_counter(3))+1}
  if mode==9{splitSelected=true};C.test_mode(C.int(mode),C.int(target));_,e=c.Qwen35SequencePrefill(r)
  if mode==7{must(int(C.test_counter(9))==target&&C.test_counter(10)==1&&C.test_counter(7)==57,fmt.Sprintf("pool fault not injected exactly once: target=%d attempts=%d hits=%d terminal=%d",target,C.test_counter(9),C.test_counter(10),C.test_counter(7)));fmt.Printf("checked pool-retirement fault hit: attempts=%d target=%d hits=%d terminal=%d\n",C.test_counter(9),target,C.test_counter(10),C.test_counter(7))}
  must(e!=nil&&clone.Len()==14&&clone.lens[0]==14&&clone.lens[1]==14,fmt.Sprintf("failure publishes frontier: mode=%d error=%v len=%d lens=%v terminal=%d",mode,e,clone.Len(),clone.lens,C.test_counter(7)))
  must(cudaQwen35PagedOwners!=nil&&cudaQwen35PagedOwners.k==clone&&clone.pool.copyFailure.Load()!=0,"failure lost actual owner")
  retireBefore:=C.test_counter(4);func(){defer func(){must(recover()!=nil,"terminal Clone accepted")}();clone.Clone()}();func(){defer func(){must(recover()!=nil,"terminal Free accepted")}();clone.Free()}()
  must(C.test_counter(4)==retireBefore&&k.Len()==14&&k.pool.K[0]!=nil,"terminal path released source")
  if mode==2||mode==3||mode==4{must(clone.table[0]==oldTable[0]&&clone.pool.K[0]==oldArena,"failed prep published reservation")}
  if mode==3{must(C.test_counter(8)>0,"deferred copy was not accepted");C.test_finish_deferred();must(C.test_counter(7)!=0&&clone.Len()==14,"late completion upgraded failure")}
  if mode==3||mode==5{must(cudaQwen35PagedOwners.pinned&&!cudaQwen35PagedOwners.completed&&len(cudaQwen35PagedOwners.table)>0,"uncertain table transfer lost its retained pin")}
 }
 // Runtime unwinds execute the real preparation defer; normal-return checks
 // distinguish Goexit from success and preserve retained operation identity.
 for _,exit:=range []bool{false,true}{c,k=reset();r=reqFor(c,k,0,2);p:=&cudaQwen35PagedSequence{c:c,k:k};must(p.validateHostLocked(r)==nil,"unwind validation");c.panicAlloc=!exit;c.exitAlloc=exit;done:=make(chan interface{},1);returned:=false
  go func(){defer func(){done<-recover()}();_ = p.prepareLocked();returned=true}()
  select{case recovered:=<-done:must(!returned,"abnormal preparation returned");if exit{must(recovered==nil,"Goexit became panic")}else{must(recovered=="allocation panic","panic changed")};case <-time.After(2*time.Second):panic("unwind hung")}
  must(cudaQwen35PagedOwners==p&&p.k==k&&k.pool.copyFailure.Load()!=0&&k.Len()==0,"unwind lost roots")
 }
 must(resetFixtureWorld()==2,"expected both failed table-transfer pins retained through final GC")
 fmt.Println("terminal owner archive retained both table pins through GC; no unpin or rehabilitation")
 fmt.Println("actual NewKV/Clone/sequence + native write/gather/retirement controls PASS")
}
`
