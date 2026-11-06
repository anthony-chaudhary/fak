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

// fak-test:runtime slow est=90s lane=default
// Required host-only cgo witness: the real Go issue/cleanup calls the real
// native uploader and Free in one process. Only CUDA/device allocation and
// event operations are replaced. No CUDA compiler, GPU or speed claim follows.
func TestCUDAAsyncUploadFailureOwnership(t *testing.T) {
	read := func(path string) string {
		b, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		return string(b)
	}
	section := func(src, start, end string) string {
		a, b := strings.Index(src, start), strings.Index(src, end)
		if a < 0 || b <= a || strings.Count(src, start) != 1 || strings.Count(src, end) != 1 {
			t.Fatalf("missing/ambiguous source section %s", start)
		}
		return src[a:b]
	}
	// Intact uploader prefix, including the real status encoder and every
	// fallback/fence branch. The unrelated checked refill arm is not mirrored.
	native := section(read("cuda/async_upload.cuh"), "#define FCUDA_ASYNC_SLOTS", "// BEGIN checked expert-ring refill (#14199).")
	free := section(read("cuda/runtime_memory.cuh"), "extern \"C\" void fcuda_free(void *d) {", "// Only for a caller-owned, unpublished reservation.")
	program := cudaAsyncFailureNativePreamble + native + free + cudaAsyncFailureNativeControls
	t.Logf("actual uploader sha256=%x; actual Free sha256=%x; native fixture sha256=%x", sha256.Sum256([]byte(native)), sha256.Sum256([]byte(free)), sha256.Sum256([]byte(program)))
	wanted := map[string]bool{
		"Dtype": false, "F32": false, "Dtype.Bytes": false, "Layout": false, "RowMajor": false,
		"QuantSpec": false, "Buffer": false, "HostBuffer": false, "Tensor": false, "Tensor.Numel": false, "makeTensor": false,
		"hostBuf": false, "*hostBuf.Ready": false, "*hostBuf.F32": false, "*hostBuf.I8": false, "NewF32": false,
		"cudaBuf": false, "*cudaBuf.Ready": false, "*cudaBuf.invalidForSubmit": false,
		"makeF32TensorLike": false, "NewQ4K": false, "q4kSuper": false,
		"cudaUploadFence": false, "cudaLandedFence": false, "cudaAsyncPending": false,
		"*cudaUploadFence.Done": false, "*cudaUploadFence.Wait": false, "*cudaUploadFence.pollLocked": false,
		"awaitAsyncUploadLocked": false, "*cudaBackend.uploadF32Async": false,
		"*cudaBackend.uploadRawKQuantAsync": false, "*cudaBackend.issueAsyncLocked": false,
		"*cudaBackend.uploadRawKQuantClassWithOperations": false, "*cudaBackend.devRawKQuantWithAllocator": false,
		"CUDAOpError": false, "*CUDAOpError.Error": false,
	}
	var body strings.Builder
	body.WriteString(cudaAsyncFailureGoPreamble)
	for _, path := range []string{"compute.go", "cpuref.go", "cuda_backend_state.go", "backend_helpers.go", "quant_q4k.go", "cuda_async_upload.go", "cuda_raw_k_offload.go", "cuda_tensor_allocation.go", "cuda_errors.go"} {
		src := read(path)
		fs := token.NewFileSet()
		f, err := parser.ParseFile(fs, path, src, 0)
		if err != nil {
			t.Fatal(err)
		}
		emit := func(name string, node ast.Node, prefix string) {
			seen, ok := wanted[name]
			if !ok {
				return
			}
			if seen {
				t.Fatalf("duplicate declaration %s", name)
			}
			wanted[name] = true
			text := src[fs.Position(node.Pos()).Offset:fs.Position(node.End()).Offset]
			t.Logf("%s/%s sha256=%x", path, name, sha256.Sum256([]byte(text)))
			body.WriteString(prefix + text + "\n")
		}
		for _, decl := range f.Decls {
			switch n := decl.(type) {
			case *ast.GenDecl:
				if n.Tok == token.CONST {
					first, ok := n.Specs[0].(*ast.ValueSpec)
					if ok && len(first.Names) > 0 && (first.Names[0].Name == "F32" || first.Names[0].Name == "RowMajor") {
						emit(first.Names[0].Name, n, "")
						continue
					}
				}
				for _, spec := range n.Specs {
					switch s := spec.(type) {
					case *ast.TypeSpec:
						emit(s.Name.Name, s, "type ")
					case *ast.ValueSpec:
						if len(s.Names) == 1 {
							emit(s.Names[0].Name, s, n.Tok.String()+" ")
						}
					}
				}
			case *ast.FuncDecl:
				key := n.Name.Name
				if n.Recv != nil {
					r := n.Recv.List[0].Type
					prefix := ""
					if p, ok := r.(*ast.StarExpr); ok {
						prefix = "*"
						r = p.X
					}
					id, ok := r.(*ast.Ident)
					if !ok {
						continue
					}
					key = prefix + id.Name + "." + key
				}
				emit(key, n, "")
			}
		}
	}
	for key, seen := range wanted {
		if !seen {
			t.Fatalf("missing production declaration %s", key)
		}
	}
	body.WriteString(cudaAsyncFailureGoControls)
	goProgram := body.String()
	t.Logf("derived Go cgo fixture sha256=%x", sha256.Sum256([]byte(goProgram)))
	// Fail, never skip, if the required host C/C++ toolchain is absent.
	for _, name := range []string{"cc", "c++"} {
		if _, err := exec.LookPath(name); err != nil {
			t.Fatalf("required host compiler %s: %v", name, err)
		}
	}
	dir := t.TempDir()
	for name, text := range map[string]string{"fixture.h": cudaAsyncFailureHeader, "fixture.cc": program, "fixture.go": goProgram} {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(text), 0600); err != nil {
			t.Fatal(err)
		}
	}
	goExe := filepath.Join(runtime.GOROOT(), "bin", "go")
	if runtime.GOOS == "windows" {
		goExe += ".exe"
	}
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, goExe, "run", ".")
	cmd.Dir = dir
	cmd.Env = append(os.Environ(), "GOWORK=off", "GO111MODULE=off", "CGO_ENABLED=1", "CC=cc", "CXX=c++")
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("actual-source async upload controls: %v\n%s", err, out)
	} else {
		t.Logf("%s", out)
	}
}

const cudaAsyncFailureHeader = `
#include <stdint.h>
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
int fcuda_upload_async(void*,const void*,size_t,int*,uint64_t*);
int fcuda_upload_async_fence(int,uint64_t,int);
void fcuda_free(void*);
void fcuda_h2d(void*,const void*,size_t);
void* test_device(size_t);
void test_reset(int,int);
int test_stat(int);
void test_complete(void);
void test_terminal_probes(void*,const void*,size_t);
void test_prior_terminal(void*,const void*);
void test_unissued_cases(void*,const void*);
void test_release(void*);
#ifdef __cplusplus
}
#endif
`

const cudaAsyncFailureNativePreamble = `
#include "fixture.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>
using cudaError_t=int; using cudaStream_t=void*; using cudaEvent_t=void*;
enum {cudaSuccess=0,cudaErrorNotReady=34,cudaStreamNonBlocking=1,cudaEventDisableTiming=2,cudaHostAllocPortable=3,cudaMemcpyHostToDevice=4};
static int g_cuda_cleanup_status=0;
static bool g_capture_open=false,g_v41_router_uncertain=false,g_v41_mmid_uncertain=false;
static cudaStream_t g_stream=(void*)1;
static size_t g_h2d_bytes=0,g_panel_live_bytes=0,g_panel_cached_bytes=0;
static std::unordered_map<void*,size_t> g_live,g_managed_live,g_panel_live,pool;
static std::unordered_map<size_t,std::vector<void*>> g_panel_pool;
static std::unordered_set<void*> hosts;
static int mode=0,fail_drain=0,api=0,copies=0,drains=0,h2d=0,freed=0,queries=0,hostfrees=0,marks=0;
static uintptr_t event_id=10;
static void *delayed_d=nullptr; static const void *delayed_h=nullptr; static size_t delayed_n=0;
static void require(bool ok,const char*what){if(!ok){std::fprintf(stderr,"CONTROL: %s\n",what);std::abort();}}
static bool v41c_pending_contains(void*){return false;}
static void q4_coeff_cache_evict(void*){}
static void fak_alloc_untrack(void*){}
static void fak_alloc_retag(void*,int,int){}
enum{FAK_ALLOC_IDLE_POOL=0};
#define FAK_ALLOC_SITE(s) 0
static void fcuda_pool_put(void*d,size_t n){freed++;pool[d]=n;}
static cudaError_t cudaFree(void*d){freed++;std::free(d);return 0;}
static cudaError_t cudaGetLastError(){api++;return 0;}
static const char* cudaGetErrorString(int){return "injected";}
static cudaError_t cudaStreamCreateWithFlags(void**p,unsigned){api++;*p=(void*)2;return 0;}
static cudaError_t cudaEventCreateWithFlags(void**p,unsigned){api++;*p=(void*)++event_id;return 0;}
static cudaError_t cudaHostAlloc(void**p,size_t n,unsigned){api++;if(mode==5)return 75;*p=std::malloc(n);require(*p,"host malloc");hosts.insert(*p);return 0;}
static cudaError_t cudaFreeHost(void*p){api++;hostfrees++;require(hosts.erase(p)==1,"host free ownership");std::free(p);return 0;}
static cudaError_t cudaEventRecord(void*,cudaStream_t s){api++;if(s==g_stream&&mode==3)return 73;if(s!=g_stream&&mode==2)return 72;return 0;}
static cudaError_t cudaStreamWaitEvent(cudaStream_t,void*,unsigned){api++;return mode==4?74:0;}
static void complete(){
 if(delayed_d){require(g_live.count(delayed_d)&&hosts.count(const_cast<void*>(delayed_h)),"DMA operands retired early");std::memcpy(delayed_d,delayed_h,delayed_n);delayed_d=nullptr;delayed_h=nullptr;delayed_n=0;}
}
static cudaError_t cudaMemcpyAsync(void*d,const void*h,size_t n,int,cudaStream_t){api++;copies++;require(!delayed_d,"overlapping fixture DMA");delayed_d=d;delayed_h=h;delayed_n=n;return mode==1?71:0;}
static cudaError_t cudaStreamSynchronize(cudaStream_t s){api++;drains++;require(s==(void*)2,"drained wrong stream");if(fail_drain)return 81;complete();return 0;}
static cudaError_t cudaEventQuery(cudaEvent_t){api++;queries++;return delayed_d?cudaErrorNotReady:cudaSuccess;}
static cudaError_t cudaEventSynchronize(cudaEvent_t){api++;complete();return 0;}
static void fcuda_ts_untimed_transfer(){marks++;}
extern "C" void fcuda_h2d(void*d,const void*h,size_t n){api++;h2d++;require(n<=144,"unexpected fallback extent");std::memcpy(d,h,n);}
`

const cudaAsyncFailureNativeControls = `
extern "C" void* test_device(size_t n){void*p=std::malloc(n?n:1);require(p,"device malloc");std::memset(p,0x55,n);g_live[p]=n;return p;}
extern "C" void test_reset(int m,int fail){
 require(!delayed_d,"test teardown abandoned DMA");
 for(void*p:hosts)std::free(p);hosts.clear();
 for(auto p:g_live)std::free(p.first);g_live.clear();
 for(auto p:pool)std::free(p.first);pool.clear();
 for(auto&s:g_async_slots)s={};g_copy_stream=nullptr;g_copy_order=nullptr;
 g_async_issued=g_async_fallback=g_async_pinned_bytes=g_h2d_bytes=0;
 g_cuda_cleanup_status=0;g_capture_open=false;mode=m;fail_drain=fail;
 api=copies=drains=h2d=freed=queries=hostfrees=marks=0;
}
extern "C" int test_stat(int i){switch(i){
 case 0:return g_cuda_cleanup_status;case 1:return copies;case 2:return drains;
 case 3:return h2d;case 4:return freed;case 5:return (int)g_live.size();
 case 6:return (int)hosts.size();case 7:return (int)g_async_issued;case 8:return (int)g_async_fallback;
 case 9:return api;case 10:return delayed_d!=nullptr;case 11:return (int)g_h2d_bytes;
 case 12:return hostfrees;default:std::abort();}}
extern "C" void test_complete(){complete();}
extern "C" void test_release(void*p){fcuda_free(p);}
extern "C" void test_terminal_probes(void*d,const void*h,size_t n){
 require(g_cuda_cleanup_status!=0,"missing terminal latch");
 int terminal=g_cuda_cleanup_status,before=api,oldfree=freed,oldhostfree=hostfrees;
 uint64_t issued=g_async_issued,fallback=g_async_fallback;size_t pinned=g_async_pinned_bytes;
 void*root=g_async_slots[0].host;uint64_t generation=g_async_slots[0].gen;bool inflight=g_async_slots[0].inflight;
 for(size_t bytes:{n,(size_t)0,FCUDA_ASYNC_SLOT_CAP+1}){
  int slot=7;uint64_t gen=999;
  require(fcuda_upload_async(d,h,bytes,&slot,&gen)==1000+terminal&&slot==-1&&gen==0,"terminal issue fell through");
 }
 g_capture_open=true;int slot=7;uint64_t gen=999;
 require(fcuda_upload_async(d,h,n,&slot,&gen)==1000+terminal,"terminal capture fallback");g_capture_open=false;
 require(fcuda_upload_async_fence(0,0,0)==1000+terminal&&fcuda_upload_async_fence(-1,0,1)==1000+terminal,"terminal fence claimed completion");
 fcuda_free(d);
 require(api==before&&freed==oldfree&&hostfrees==oldhostfree&&g_cuda_cleanup_status==terminal,"terminal retry had side effects");
 require(root==g_async_slots[0].host&&generation==g_async_slots[0].gen&&inflight==g_async_slots[0].inflight&&pinned==g_async_pinned_bytes&&issued==g_async_issued&&fallback==g_async_fallback,"terminal retry changed slot/accounting");
}
extern "C" void test_prior_terminal(void*d,const void*h){g_cuda_cleanup_status=91;test_terminal_probes(d,h,16);require(api==0,"prior terminal entered CUDA");}
extern "C" void test_unissued_cases(void*d,const void*h){
 int s;uint64_t g;
 for(size_t n:{(size_t)0,FCUDA_ASYNC_SLOT_CAP+1})require(fcuda_upload_async(d,h,n,&s,&g)==1&&s==-1&&g==0,"unissued size fallback changed");
 g_capture_open=true;require(fcuda_upload_async(d,h,16,&s,&g)==1,"capture fallback changed");g_capture_open=false;
 require(api==0&&g_async_fallback==3,"unissued fallback entered CUDA");
 // An exhausted pool must query existing fences, never grow it or issue DMA.
 require(fcuda_async_init()==0,"fixture init");for(auto &slot:g_async_slots)slot.inflight=true;
 // Query shim reports NotReady whenever a real delayed operand pair exists.
 void*host=nullptr;require(cudaHostAlloc(&host,16,cudaHostAllocPortable)==0,"fixture host");
 delayed_d=d;delayed_h=host;delayed_n=16;std::memcpy(host,h,16);
 int before=api;require(fcuda_upload_async(d,h,16,&s,&g)==1&&copies==0&&drains==0&&queries==FCUDA_ASYNC_SLOTS&&api==before+FCUDA_ASYNC_SLOTS,"exhaustion fallback changed");
 complete();for(auto &slot:g_async_slots)slot.inflight=false;
}
`

const cudaAsyncFailureGoPreamble = `package main
/*
#cgo CXXFLAGS: -std=c++17
#include "fixture.h"
*/
import "C"
import("fmt";"strconv";"strings";"sync";"sync/atomic";"unsafe")
type Backend=any
type MemoryClass string
const(MemoryUnknown MemoryClass="";MemoryOffload MemoryClass="offload")
type Fence interface{Done()bool;Wait()}
type cudaBackend struct{fenceGen uint64;last *cudaBuf;allocation unsafe.Pointer}
var cudaMu sync.Mutex
func(c *cudaBackend)dallocClass(n int,class MemoryClass,_ string)*cudaBuf{
 p:=C.test_device(C.size_t(n));c.last=&cudaBuf{ptr:p,n:n,class:class};c.allocation=p;return c.last
}
func(c *cudaBackend)uploadRawKQuant(Tensor,HostBuffer)Tensor{panic("entered immutable upload")}
// Q4_K never enters the separate IQ metadata branch; refill opt-in is outside
// this legacy async-upload witness and is covered by its own real-source test.
func(c *cudaBackend)OffloadRefillEnabled()bool{return false}
func cudaOffloadRefillBytes(Tensor)(int,bool){panic("unexpected IQ metadata branch")}
`

const cudaAsyncFailureGoControls = `
func require(ok bool,s string){if !ok{panic(s)}}
func stat(i int)int{return int(C.test_stat(C.int(i)))}
func caught(f func())(r any){defer func(){r=recover()}();f();return nil}
func fixture(raw bool)(*cudaBackend,Tensor,func()(Tensor,Fence),[]byte){
 c:=&cudaBackend{}
 var src Tensor;var bytes []byte
 if raw{bytes=make([]byte,144);for i:=range bytes{bytes[i]=byte(i+3)};src=NewQ4K(nil,[]int{1,256},bytes)}else{
  values:=[]float32{1,2,3,4};src=NewF32(nil,[]int{2,2},values);bytes=append([]byte(nil),unsafe.Slice((*byte)(unsafe.Pointer(&values[0])),16)...)
 }
 call:=func()(Tensor,Fence){if raw{return c.uploadRawKQuantAsync(src,src.buf.(HostBuffer),MemoryOffload,"fault")};return c.uploadF32Async(src,src.buf.(HostBuffer),MemoryOffload,"fault")}
 return c,src,call,bytes
}
func same(p unsafe.Pointer,b []byte)bool{a:=unsafe.Slice((*byte)(p),len(b));for i:=range b{if a[i]!=b[i]{return false}};return true}
func reset(mode,drain int){C.test_reset(C.int(mode),C.int(drain));cudaAsyncPending=map[unsafe.Pointer]*cudaUploadFence{}}
func main(){
 for _,raw:=range []bool{false,true}{
  for _,mode:=range []int{1,2}{for _,failedDrain:=range []int{0,1}{
   reset(mode,failedDrain);c,_,call,bytes:=fixture(raw);var out Tensor;var fence Fence
   r:=caught(func(){out,fence=call()});e,ok:=r.(*CUDAOpError)
   require(ok&&e.Code==70+mode&&e.Op=="UploadAsync"&&e.Site=="upload-async","original operation failure lost")
   require(out.buf==nil&&fence==nil&&c.last.ptr==nil&&len(cudaAsyncPending)==0,"failed upload published")
   require(stat(1)==1&&stat(2)==1&&stat(3)==0&&stat(7)==0&&stat(8)==0&&stat(11)==0,"attempt/drain/accounting mismatch")
   require(stat(6)==1&&stat(12)==0,"pinned source root lost")
   if failedDrain==0{
    require(stat(0)==0&&stat(4)==1&&stat(5)==0&&stat(10)==0&&same(c.allocation,bytes),"checked drain did not precede native Free")
   }else{
    require(stat(0)==81&&stat(4)==0&&stat(5)==1&&stat(10)==1,"failed drain did not retain native allocation")
    require(!same(c.allocation,bytes)&&*(*byte)(c.allocation)==0x55,"fixture DMA completed too early")
    C.test_terminal_probes(c.allocation,unsafe.Pointer(&bytes[0]),C.size_t(len(bytes)))
    // Source remains usable after the Go caller has detached its handle.
    C.test_complete();require(same(c.allocation,bytes)&&stat(0)==81&&stat(4)==0,"late DMA lost root or upgraded status")
    C.test_terminal_probes(c.allocation,unsafe.Pointer(&bytes[0]),C.size_t(len(bytes)))
   }
  }}
  for _,mode:=range []int{3,4}{
   reset(mode,1);c,_,call,_:=fixture(raw);r:=caught(func(){call()});e,ok:=r.(*CUDAOpError)
   require(ok&&e.Code==70+mode&&c.last.ptr==nil&&len(cudaAsyncPending)==0,"pre-copy failure publication")
   require(stat(1)==0&&stat(2)==0&&stat(0)==0&&stat(4)==1&&stat(5)==0&&stat(3)==0,"unsubmitted failure changed cleanup")
  }
  reset(0,0);c,_,call,bytes:=fixture(raw);out,fence:=call()
  require(out.buf==c.last&&c.last.ptr==c.allocation&&fence!=cudaLandedFence&&len(cudaAsyncPending)==1,"successful async publication")
  require(!fence.Done()&&stat(2)==0&&stat(7)==1&&stat(3)==0&&!same(c.allocation,bytes),"ordinary overlap changed")
  fence.Wait();require(fence.Done()&&len(cudaAsyncPending)==0&&same(c.allocation,bytes)&&stat(11)==len(bytes),"ordinary fence/payload")
  C.test_release(c.allocation);require(stat(4)==1&&stat(5)==0,"completed Free")
  reset(5,0);c,_,call,bytes=fixture(raw);out,fence=call()
  require(out.buf==c.last&&fence==cudaLandedFence&&len(cudaAsyncPending)==0&&same(c.allocation,bytes),"pinned-allocation fallback changed")
  require(stat(3)==1&&stat(1)==0&&stat(2)==0&&stat(8)==1,"fallback accidentally retried async")
  C.test_release(c.allocation)
 }
 reset(0,0);c,_,_,bytes:=fixture(false);p:=c.dallocClass(16,MemoryOffload,"prior").ptr
 C.test_prior_terminal(p,unsafe.Pointer(&bytes[0]));require(stat(0)==91&&stat(4)==0&&stat(5)==1,"prior terminal lost")
 reset(0,0);c,_,_,bytes=fixture(false);p=c.dallocClass(16,MemoryOffload,"fallback").ptr
 C.test_unissued_cases(p,unsafe.Pointer(&bytes[0]));require(stat(0)==0&&stat(8)==4,"genuine fallback counter")
 reset(0,0)
 fmt.Println("PASS: real Go -> native async issue/drain/Free; F32 and raw Q4; deferred DMA; terminal no-effects; success and unissued fallback")
}
`
