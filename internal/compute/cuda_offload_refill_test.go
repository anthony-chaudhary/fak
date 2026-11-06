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

// fak-test:runtime slow est=80s lane=default
// Required, deadline-bounded SOFTWARE witness. Runs the intact production
// native fragment against explicit CUDA/ledger fault shims and the intact Go
// ownership core against host allocations. It neither launches CUDA nor proves
// model quality, device completion, native link availability or performance.
func TestCUDAOffloadRefillSource(t *testing.T) {
	read := func(path string) string {
		b, e := os.ReadFile(path)
		if e != nil {
			t.Fatal(e)
		}
		return string(b)
	}
	section := func(src, start, end string) string {
		a := strings.Index(src, start)
		b := strings.Index(src, end)
		if a < 0 || b <= a || strings.Count(src, start) != 1 || strings.Count(src, end) != 1 {
			t.Fatalf("missing/ambiguous source section %s", start)
		}
		text := src[a+len(start) : b]
		t.Logf("%s sha256=%x", start, sha256.Sum256([]byte(text)))
		return text
	}
	goSource := read("cuda_offload_refill.go")
	compact := strings.Join(strings.Fields(goSource), "")
	for _, want := range []string{
		"os.Getenv(\"FAK_CUDA_EXPERT_RING_REFILL\")==\"1\"",
		"func(c*cudaBackend)CanRefillOffload(destination,sourceTensor,bytesint64)bool{cudaMu.Lock()defercudaMu.Unlock()",
		"func(c*cudaBackend)RefillOffload(destination,sourceTensor,bytesint64)(Tensor,OwnedTransferResult){cudaMu.Lock()defercudaMu.Unlock()",
		"C.fcuda_offload_refill_probe(", "C.fcuda_offload_refill_copy(", "C.fcuda_offload_refill_stop(",
	} {
		if !strings.Contains(compact, want) {
			t.Fatalf("missing real provider boundary %s", want)
		}
	}
	if strings.Contains(goSource, "cudaForwardExecutionMu.Lock") {
		t.Fatal("provider re-acquires enclosing domain")
	}
	native := section(read("cuda/async_upload.cuh"), "// BEGIN checked expert-ring refill (#14199).", "// END checked expert-ring refill.")
	core := section(goSource, "// BEGIN offload refill owner core.", "// END offload refill owner core.")
	t.Run("native-retained-copy", func(t *testing.T) {
		compiler := ""
		for _, name := range []string{"c++", "g++", "clang++"} {
			if p, e := exec.LookPath(name); e == nil {
				compiler = p
				break
			}
		}
		if compiler == "" {
			t.Fatal("required host C++ compiler unavailable")
		}
		dir := t.TempDir()
		src := filepath.Join(dir, "refill.cc")
		exe := filepath.Join(dir, "refill")
		if runtime.GOOS == "windows" {
			exe += ".exe"
		}
		program := cudaOffloadRefillNativeFixture + native + cudaOffloadRefillNativeControls
		t.Logf("derived native fixture sha256=%x", sha256.Sum256([]byte(program)))
		if e := os.WriteFile(src, []byte(program), 0600); e != nil {
			t.Fatal(e)
		}
		ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
		defer cancel()
		if out, e := exec.CommandContext(ctx, compiler, "-std=c++17", "-O0", src, "-o", exe).CombinedOutput(); e != nil {
			t.Fatalf("host compile: %v\n%s", e, out)
		}
		if out, e := exec.CommandContext(ctx, exe).CombinedOutput(); e != nil {
			t.Fatalf("native ownership controls: %v\n%s", e, out)
		}
	})
	t.Run("go-owner", func(t *testing.T) {
		wanted := map[string]bool{
			"QuantSpec": false, "Buffer": false, "HostBuffer": false, "Tensor": false, "makeTensor": false, "Tensor.Backend": false,
			"hostBuf": false, "*hostBuf.Ready": false, "*hostBuf.F32": false, "*hostBuf.I8": false,
			"cudaBuf": false, "*cudaBuf.Ready": false, "*cudaBuf.invalidForSubmit": false, "cudaRawUploadIsVerbatim": false,
			"OwnedTransferState": false, "OwnedTransferResult": false, "OwnedTransferUnknown": false, "ErrOwnedTransferUnknown": false,
			"*cudaBackend.OffloadRefillEnabled": false,
			"NewRawIQ":                          false, "NewIQ2XXS": false, "newRawKQuant": false, "RawIQBlockBytes": false,
			"q4kSuper": false, "rawIQSuper": false, "iq2xxsSuper": false, "iq2xxsSuperBlock": false,
			"iq2xsSuperBlock": false, "iq1mSuperBlock": false, "iq4nlBlockBytes": false,
			"iq4nlSuperBlock": false, "iq4xsSuperBlock": false,
			"*cudaBackend.devRawKQuantWithAllocator":          false,
			"*cudaBackend.uploadRawKQuantClassWithOperations": false,
		}
		// Registry is a fixture boundary, but its selected block associations
		// must match production, not only the test's nominal dtype names.
		registry := strings.Join(strings.Fields(read("quant_rawiq.go")), "")
		for _, pair := range []string{"IQ2_XXS:{blockBytes:iq2xxsSuperBlock,", "IQ2_XS:{blockBytes:iq2xsSuperBlock,", "IQ1_M:{blockBytes:iq1mSuperBlock,", "IQ4_NL:{blockBytes:iq4nlSuperBlock,", "IQ4_XS:{blockBytes:iq4xsSuperBlock,"} {
			if !strings.Contains(registry, pair) {
				t.Fatalf("raw IQ registry changed: %s", pair)
			}
		}
		var program strings.Builder
		program.WriteString(cudaOffloadRefillGoFixture)
		for _, path := range []string{"compute.go", "cpuref.go", "cuda_backend_state.go", "cuda_device_ops.go", "owned_transfer.go", "cuda_offload_refill.go", "quant_rawiq.go", "quant_q4k.go", "quant_iq2_xxs.go", "cuda_tensor_allocation.go", "cuda_raw_k_offload.go"} {
			src := read(path)
			fs := token.NewFileSet()
			f, e := parser.ParseFile(fs, path, src, 0)
			if e != nil {
				t.Fatal(e)
			}
			emit := func(name string, n ast.Node, prefix string) {
				seen, ok := wanted[name]
				if !ok {
					return
				}
				if seen {
					t.Fatalf("duplicate %s", name)
				}
				wanted[name] = true
				text := src[fs.Position(n.Pos()).Offset:fs.Position(n.End()).Offset]
				t.Logf("%s/%s sha256=%x", path, name, sha256.Sum256([]byte(text)))
				program.WriteString(prefix + text + "\n")
			}
			for _, d := range f.Decls {
				switch n := d.(type) {
				case *ast.GenDecl:
					if n.Tok == token.CONST {
						for _, s := range n.Specs {
							if v, ok := s.(*ast.ValueSpec); ok && len(v.Names) > 0 && v.Names[0].Name == "OwnedTransferUnknown" {
								emit("OwnedTransferUnknown", n, "")
							} else if v, ok := s.(*ast.ValueSpec); ok && len(v.Names) == 1 && len(v.Values) == 1 {
								emit(v.Names[0].Name, v, "const ")
							}
						}
					} else {
						for _, s := range n.Specs {
							switch v := s.(type) {
							case *ast.TypeSpec:
								emit(v.Name.Name, v, "type ")
							case *ast.ValueSpec:
								if n.Tok == token.VAR && len(v.Names) == 1 {
									emit(v.Names[0].Name, v, "var ")
								}
							}
						}
					}
				case *ast.FuncDecl:
					key := n.Name.Name
					if n.Recv != nil {
						r := n.Recv.List[0].Type
						prefix := ""
						if p, ok := r.(*ast.StarExpr); ok {
							r = p.X
							prefix = "*"
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
		program.WriteString(core)
		program.WriteString(cudaOffloadRefillGoControls)
		text := program.String()
		t.Logf("derived Go fixture sha256=%x", sha256.Sum256([]byte(text)))
		dir := t.TempDir()
		if e := os.WriteFile(filepath.Join(dir, "owner_test.go"), []byte(text), 0600); e != nil {
			t.Fatal(e)
		}
		exe := filepath.Join(runtime.GOROOT(), "bin", "go")
		if runtime.GOOS == "windows" {
			exe += ".exe"
		}
		ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
		defer cancel()
		cmd := exec.CommandContext(ctx, exe, "test", "-run", "^TestOwner$", "-count=1", "-timeout", "45s", ".")
		cmd.Dir = dir
		cmd.Env = append(os.Environ(), "GOWORK=off", "GO111MODULE=off")
		if out, e := cmd.CombinedOutput(); e != nil {
			t.Fatalf("Go owner controls: %v\n%s", e, out)
		}
	})
}

const cudaOffloadRefillNativeFixture = `
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <limits>
using cudaStream_t = void *;
using cudaError_t = int;
enum {cudaSuccess=0,cudaHostAllocPortable=1,cudaMemcpyHostToDevice=2,cudaMemoryTypeDevice=3};
enum cudaStreamCaptureStatus {cudaStreamCaptureStatusNone,cudaStreamCaptureStatusActive};
struct cudaPointerAttributes {int type,device;void *devicePointer;};
enum {FAK_ALLOC_WEIGHTS=0,FAK_ALLOC_DEVICE_LOCAL=0,FAK_ALLOC_PINNED_HOST=2};
struct fak_alloc_tag{int cls,kind;size_t bytes;};
struct fak_alloc_tags_guard{};
#define FAK_ALLOC_SITE(s) 1
static std::unordered_map<const void*,size_t> g_live,g_managed_live,g_panel_live;
static std::unordered_map<const void*,fak_alloc_tag> g_alloc_tags;
static std::unordered_set<void*> hosts;
static cudaStream_t g_stream=(void*)1;
static bool g_capture_open=false,g_pool_graph_deep=false,pending=false,peer=false,g_ts_on=false;
static int g_cuda_cleanup_status=0;
static size_t g_h2d_bytes=0;
static int mode=0,api=0,allocs=0,copies=0,syncs=0,frees=0,marks=0;
static void *delayedD=nullptr; static const void *delayedH=nullptr; static size_t delayedN=0;
static bool v41_candidates_stream_uncertain(){return g_cuda_cleanup_status||peer;}
static bool v41c_pending_contains(void*){return pending;}
#define FCUDA_TS(name, grid, ...) do { marks++; __VA_ARGS__; } while(0)
static cudaError_t cudaGetDevice(int*p){api++;*p=mode==1?1:0;return 0;}
static cudaError_t cudaPointerGetAttributes(cudaPointerAttributes*p,void*d){api++;p->type=cudaMemoryTypeDevice;p->device=0;p->devicePointer=d;return mode==2?72:0;}
static cudaError_t cudaPeekAtLastError(){api++;return mode==3?73:0;}
static cudaError_t cudaStreamIsCapturing(cudaStream_t,cudaStreamCaptureStatus*p){api++;*p=mode==4?cudaStreamCaptureStatusActive:cudaStreamCaptureStatusNone;return 0;}
static cudaError_t cudaHostAlloc(void**p,size_t n,unsigned){api++;allocs++;if(mode==5)return 75;*p=std::malloc(n);hosts.insert(*p);return 0;}
static void fak_alloc_track(const void*p,int c,int k,int,size_t n){if(mode==6)throw std::runtime_error("track");g_alloc_tags[p]={c,k,n};}
static cudaError_t cudaMemcpyAsync(void*d,const void*h,size_t n,int,cudaStream_t){api++;copies++;delayedD=d;delayedH=h;delayedN=n;return mode==7?77:0;}
static cudaError_t cudaStreamSynchronize(cudaStream_t){api++;syncs++;if(mode==8)return 78;if(delayedD)std::memcpy(delayedD,delayedH,delayedN);delayedD=nullptr;if(mode==9)peer=true;return 0;}
static cudaError_t cudaFreeHost(void*p){api++;frees++;if(mode==10)return 80;hosts.erase(p);std::free(p);return 0;}
static bool fak_alloc_untrack(const void*p){if(mode==11)throw std::runtime_error("untrack");if(mode==12)return false;return g_alloc_tags.erase(p)==1;}
static void require(bool ok,const char*s){if(!ok){std::fprintf(stderr,"%s\n",s);std::exit(1);}}
`

const cudaOffloadRefillNativeControls = `
static unsigned char dst[16],src[16];
static void reset(int m=0){
  // Fixture-only independent cases. A retained DMA is completed before fixture
  // teardown; production deliberately has no reset/recovery path.
  if(delayedD){std::memcpy(delayedD,delayedH,delayedN);delayedD=nullptr;}
  for(void*p:hosts)std::free(p);hosts.clear();
  g_offload_refill_hold={};g_live.clear();g_managed_live.clear();g_panel_live.clear();g_alloc_tags.clear();
  g_live[dst]=16;g_alloc_tags[dst]={FAK_ALLOC_WEIGHTS,FAK_ALLOC_DEVICE_LOCAL,16};
  g_stream=(void*)1;g_capture_open=g_pool_graph_deep=pending=peer=g_ts_on=false;g_cuda_cleanup_status=0;
  g_h2d_bytes=0;mode=m;api=allocs=copies=syncs=frees=marks=0;std::memset(dst,0x55,16);
  for(int i=0;i<16;i++)src[i]=(unsigned char)(i*11);
}
int main(){
  reset();require(fcuda_offload_refill_probe(dst,src,16)==1&&api==0&&allocs==0,"pure positive");
  require(fcuda_offload_refill_copy(dst,src,16)==0,"success");
  require(!std::memcmp(dst,src,16)&&g_live[dst]==16&&g_alloc_tags.count(dst)==1,"same allocation/payload");
  require(g_h2d_bytes==16&&copies==1&&syncs==1&&frees==1&&hosts.empty()&&!g_offload_refill_hold.device&&!g_offload_refill_hold.host,"success retirement/accounting");
  for(int edge=0;edge<13;edge++){
    reset();size_t n=16;void*d=dst;const void*h=src;
    switch(edge){
     case 0:g_capture_open=true;break;case 1:g_pool_graph_deep=true;break;
     case 2:pending=true;break;case 3:peer=true;break;case 4:g_live[dst]=15;break;
     case 5:g_managed_live[dst]=16;break;case 6:g_panel_live[dst]=16;break;
     case 7:g_alloc_tags[dst].kind=FAK_ALLOC_PINNED_HOST;break;
     case 8:g_h2d_bytes=std::numeric_limits<size_t>::max()-15;break;
     case 9:h=dst+1;break;case 10:d=(void*)(UINTPTR_MAX-8);break;
     case 11:n=FCUDA_OFFLOAD_REFILL_CAP+1;break;
     case 12:g_ts_on=true;break;
    }
    const size_t before=g_h2d_bytes;
    require(!fcuda_offload_refill_probe(d,h,n)&&api==0&&allocs==0&&!g_offload_refill_hold.device&&g_h2d_bytes==before,"negative probe had effects");
  }
  reset();g_h2d_bytes=std::numeric_limits<size_t>::max()-16;
  require(fcuda_offload_refill_copy(dst,src,16)==0&&g_h2d_bytes==std::numeric_limits<size_t>::max(),"exact counter boundary");
  for(int m=1;m<=12;m++){
    reset(m);int st=fcuda_offload_refill_copy(dst,src,16);
    require(st!=0&&g_cuda_cleanup_status==st&&g_offload_refill_hold.device==dst,"failure owner/first cause");
    int before=api;require(!fcuda_offload_refill_probe(dst,src,16),"terminal probe");
    fcuda_offload_refill_stop(dst);require(g_cuda_cleanup_status==st,"stop replaced first cause");
    require(fcuda_offload_refill_copy(dst,src,16)==st&&api==before,"terminal retry had effects");
    if(m>=6&&m<=10)require(g_offload_refill_hold.host&&hosts.count(g_offload_refill_hold.host),"lost live pinned root");
    if(m==11||m==12)require(!g_offload_refill_hold.host&&hosts.empty()&&frees==1,"physical retirement misrepresented");
    if(m==8){
      require(dst[0]==0x55&&delayedD==dst&&delayedH==g_offload_refill_hold.host&&frees==0,"failed fence freed/not retained DMA");
      std::memcpy(delayedD,delayedH,delayedN);delayedD=nullptr;
      require(!std::memcmp(dst,src,16)&&g_cuda_cleanup_status==st&&g_offload_refill_hold.device==dst,"late DMA incorrectly upgraded result");
    }
    if(m<8)require(g_h2d_bytes==0,"rejected copy counted");
    if(m>=8)require(g_h2d_bytes==16,"accepted copy not counted");
  }
  reset();g_cuda_cleanup_status=991;int before=api;
  require(fcuda_offload_refill_copy(dst,src,16)==991&&api==before,"prior cause lost");
  reset();return 0;
}
`

const cudaOffloadRefillGoFixture = `package compute
import("errors";"fmt";"os";"runtime";"slices";"sync/atomic";"testing";"time";"unsafe")
type Backend=any
type MemoryClass string
const MemoryOffload MemoryClass="offload"
const MemoryWeights MemoryClass="weights"
type Dtype int
const(F32 Dtype=iota;Q2_K;Q3_K;Q4_K;Q5_K;Q6_K;IQ2_XXS;IQ2_XS;IQ1_M;IQ4_NL;IQ4_XS;Q8_0)
type Layout int
const RowMajor Layout=0
type health bool
func(h health)Refusing()bool{return bool(h)}
type DeviceFaultLatch struct{bad bool}
func(f *DeviceFaultLatch)Health()health{return health(f!=nil&&f.bad)}
func(f *DeviceFaultLatch)Observe(_ string,_ string,_ int){if f!=nil{f.bad=true}}
const DeviceFaultExecution="execution"
type cudaBackend struct{graphOnce any;capturing bool;faultLatch *DeviceFaultLatch;transient []*cudaBuf;fenceGen uint64}
type cudaUploadFence struct{}
type ucKey struct{n int}
var cudaDev *cudaBackend
var graphEnabled bool
var cudaAsyncPending=map[unsafe.Pointer]*cudaUploadFence{}
var uploadCache=map[ucKey]Tensor{}
// Raw-IQ registry is an explicit fixture boundary; the production byte/metadata
// validator and whitelist are copied intact below, not reimplemented here.
type rawIQFormat struct{blockBytes int}
var rawIQFormats=map[Dtype]rawIQFormat{IQ2_XXS:{iq2xxsSuperBlock},IQ2_XS:{iq2xsSuperBlock},IQ1_M:{iq1mSuperBlock},IQ4_NL:{iq4nlSuperBlock},IQ4_XS:{iq4xsSuperBlock}}
func(d Dtype)String()string{return fmt.Sprintf("%d",int(d))}
`

const cudaOffloadRefillGoControls = `
func refillFixture()(c *cudaBackend,dst,src Tensor,b *cudaBuf){
 cudaOffloadRefillActive=nil;graphEnabled=false;cudaAsyncPending=map[unsafe.Pointer]*cudaUploadFence{};uploadCache=map[ucKey]Tensor{}
 c=&cudaBackend{faultLatch:&DeviceFaultLatch{}};cudaDev=c
 raw:=make([]int8,144);for i:=range raw{raw[i]=int8(i*7)}
 storage:=make([]int8,144)
 b=&cudaBuf{ptr:unsafe.Pointer(&storage[0]),n:144,class:MemoryOffload}
 q:=&QuantSpec{Block:256,Axis:2,Bits:4}
 dst=makeTensor(c,Q4_K,RowMajor,[]int{1,256},q,b)
 v:=*q;src=makeTensor(nil,Q4_K,RowMajor,[]int{1,256},&v,&hostBuf{i8:raw})
 return
}
func refillIQFixture(t *testing.T,dt Dtype,block int,dedicated bool,shape []int)(c *cudaBackend,dst,src Tensor,b *cudaBuf){
 t.Helper();c,_,_,_=refillFixture()
 n:=shape[0]*(shape[1]/256)*block;raw:=make([]byte,n);victimRaw:=make([]byte,n)
 for i:=range raw{raw[i]=byte(i*7+31);victimRaw[i]=raw[i]^0xff}
 makeHost:=func(data []byte)Tensor{if dedicated{return NewIQ2XXS(nil,shape,data)};return NewRawIQ(nil,dt,shape,data)}
 src=makeHost(raw);victim:=makeHost(victimRaw)
 allocs,copies,releases:=0,0,0
 dst=c.uploadRawKQuantClassWithOperations(victim,victim.buf.(HostBuffer),MemoryOffload,
  func(Tensor,HostBuffer)Tensor{t.Fatal("offload entered immutable cache");return Tensor{}},
  func(size int)*cudaBuf{
   allocs++;if size!=n{t.Fatal("initial offload extent")}
   storage:=make([]int8,size);b=&cudaBuf{ptr:unsafe.Pointer(&storage[0]),n:size,class:MemoryOffload};return b
  },
  func(p unsafe.Pointer,data []int8){copies++;if p!=b.ptr||len(data)!=n{t.Fatal("initial offload copy")};copy(unsafe.Slice((*int8)(p),n),data)},
  func(unsafe.Pointer){releases++})
 if allocs!=1||copies!=1||releases!=0||dst.buf!=b||dst.Quant==victim.Quant||!slices.Equal(unsafe.Slice((*int8)(b.ptr),n),victim.buf.(*hostBuf).i8){t.Fatal("actual initial offload ownership")}
 return
}
func TestOwner(t *testing.T){
 t.Setenv("FAK_CUDA_EXPERT_RING_REFILL","")
 c,d,s,b:=refillFixture()
 if c.OffloadRefillEnabled(){t.Fatal("default on")}
 t.Setenv("FAK_CUDA_EXPERT_RING_REFILL","1")
 if !c.OffloadRefillEnabled()||(&cudaBackend{}).OffloadRefillEnabled(){t.Fatal("opt in/owner")}
 p:=b.ptr;copies,stops:=0,0
 out,r:=c.offloadRefillWithOperations(d,s,144,true,func(x,y unsafe.Pointer,n int)bool{return x==p&&y!=nil&&n==144},
 func(x,y unsafe.Pointer,n int)int{
  copies++;a:=cudaOffloadRefillActive
  if a==nil||a.source.buf!=s.buf||a.destination.buf!=d.buf||a.replacement.buf==d.buf||a.pointer!=p||b.ptr!=nil||!b.invalidForSubmit(){t.Fatal("pre-copy identity/root transition")}
  copy(unsafe.Slice((*byte)(x),n),unsafe.Slice((*byte)(y),n));return 0
 },func(unsafe.Pointer){stops++})
 fresh:=out.buf.(*cudaBuf)
 if r.State!=OwnedTransferComplete||r.Err!=nil||fresh==b||fresh.ptr!=p||!fresh.Ready()||cudaOffloadRefillActive!=nil||copies!=1||stops!=0||b.Ready(){t.Fatal("success ownership")}
 if !slices.Equal(unsafe.Slice((*int8)(fresh.ptr),144),s.buf.(*hostBuf).i8){t.Fatal("payload")}
 s.Shape[0]=2;s.Quant.Bits=2
 if out.Shape[0]!=1||out.Quant.Bits!=4{t.Fatal("borrowed descriptor")}
 // Every pure negative is checked without invoking native operations or
 // changing roots/identity. Native range/ledger checks are a separate witness.
 for _,name:=range []string{"managed","cached","alias-cache","pending","graph","graph-owner","graph-pin","transient","scales","charge","host-key","async","invalid","foreign","side-scale","extent","overflow","dtype","alias","host-type"}{
  c,d,s,b=refillFixture();p=b.ptr
  switch name{
  case "managed":b.managed=true
  case "cached":b.hostKeep=s.buf.(HostBuffer)
  case "alias-cache":uploadCache[ucKey{1}]=makeTensor(c,Q4_K,RowMajor,d.Shape,d.Quant,&cudaBuf{ptr:p})
  case "pending":cudaAsyncPending[p]=&cudaUploadFence{}
  case "graph":graphEnabled=true
  case "graph-owner":c.graphOnce=1
  case "graph-pin":b.graphOncePinned=true
  case "transient":c.transient=[]*cudaBuf{{ptr:p}}
  case "scales":b.scales=p
  case "charge":b.budgetedWeightBytes=1
  case "host-key":b.host=1
  case "async":b.be=c
  case "invalid":b.invalid=1
  case "foreign":d.be=&cudaBackend{}
  case "side-scale":s.Quant.Scale=[]float32{1}
  case "extent":s.buf.(*hostBuf).i8=s.buf.(*hostBuf).i8[:143]
  case "overflow":d.Shape=[]int{int(^uint(0)>>1),256};s.Shape=slices.Clone(d.Shape)
  case "dtype":s.Dtype=Q8_0
  case "alias":s.buf=&hostBuf{i8:unsafe.Slice((*int8)(p),144)}
  case "host-type":s.buf=b
  }
  before:=b.invalid
  if _,_,ok:=c.offloadRefillOperands(d,s,144);ok||b.ptr!=p||b.invalid!=before||cudaOffloadRefillActive!=nil{t.Fatalf("pure refusal %s",name)}
 }
 for _,g:=range []struct{dt Dtype;n int;bits uint8}{{Q2_K,84,2},{Q3_K,110,3},{Q4_K,144,4},{Q5_K,176,5},{Q6_K,210,6},{IQ4_NL,144,4},{IQ4_XS,136,4}}{
  c,d,s,b=refillFixture();_ = c
  d.Dtype=g.dt;d.Quant.Bits=g.bits
  if n,ok:=cudaOffloadRefillBytes(d);!ok||n!=g.n{t.Fatalf("raw geometry %v",g)}
 }
 for _,dt:=range []Dtype{IQ2_XXS,IQ2_XS,IQ1_M,IQ4_NL,IQ4_XS}{
  n,_:=RawIQBlockBytes(dt);src:=NewRawIQ(nil,dt,[]int{1,256},make([]byte,n))
  if src.Quant.Bits!=4{t.Fatal("constructor contract changed")}
  got,ok:=cudaOffloadRefillBytes(src)
  if !ok||got!=n{t.Fatalf("actual IQ constructor envelope %v",dt)}
 }
 // Constructor-derived IQ cases use the actual owner transition, not just
 // the geometry helper. Distinct allocations model host/native storage only.
 for _,g:=range []struct{dt Dtype;block int;dedicated bool}{{IQ2_XXS,66,false},{IQ2_XXS,66,true},{IQ2_XS,74,false},{IQ1_M,56,false}}{
  // The real allocator retains its incumbent nominal Bits on opt-out.
  t.Setenv("FAK_CUDA_EXPERT_RING_REFILL","")
  c,d,s,b=refillIQFixture(t,g.dt,g.block,g.dedicated,[]int{1,256})
  legacy:=uint8(2);if g.dt==IQ1_M{legacy=1}
  if d.Quant.Bits!=legacy||c.OffloadRefillEnabled(){t.Fatal("opt-out descriptor changed")}
  t.Setenv("FAK_CUDA_EXPERT_RING_REFILL","1")
  // Invalid source metadata keeps the incumbent upload descriptor; it never
  // becomes a refillable descriptor merely because the selector is enabled.
  badSource:=s;badQuant:=*s.Quant;badQuant.Bits=7;badSource.Quant=&badQuant
  allocated,copied:=0,0
  legacyOut:=c.uploadRawKQuantClassWithOperations(badSource,badSource.buf.(HostBuffer),MemoryOffload,
   func(Tensor,HostBuffer)Tensor{t.Fatal("invalid metadata entered weights");return Tensor{}},
   func(n int)*cudaBuf{allocated++;storage:=make([]int8,n);return &cudaBuf{ptr:unsafe.Pointer(&storage[0]),n:n,class:MemoryOffload}},
   func(p unsafe.Pointer,data []int8){copied++;copy(unsafe.Slice((*int8)(p),len(data)),data)},
   func(unsafe.Pointer){t.Fatal("legacy upload unexpectedly retired")})
  if allocated!=1||copied!=1||legacyOut.Quant.Bits!=legacy{t.Fatal("unvalidated source descriptor preserved")}
  for _,shape:=range [][]int{{1,256},{3,512}}{
   c,d,s,b=refillIQFixture(t,g.dt,g.block,g.dedicated,shape);p=b.ptr
   n:=shape[0]*(shape[1]/256)*g.block;bits:=uint8(4);if g.dedicated{bits=2}
   if s.Quant.Bits!=bits||d.Quant.Bits!=bits{t.Fatal("actual upload constructor metadata")}
   if slices.Equal(unsafe.Slice((*int8)(p),n),s.buf.(*hostBuf).i8){t.Fatal("overwrite control has identical old bytes")}
   // Non-offload classes still delegate immediately to the old weight route.
   weights:=0
   cached:=c.uploadRawKQuantClassWithOperations(s,s.buf.(HostBuffer),MemoryWeights,
    func(x Tensor,h HostBuffer)Tensor{weights++;if x.buf!=s.buf||h!=s.buf{t.Fatal("weight input changed")};return d},
    func(int)*cudaBuf{t.Fatal("weight allocation bypass");return nil},
    func(unsafe.Pointer,[]int8){t.Fatal("weight copy bypass")},func(unsafe.Pointer){t.Fatal("weight release bypass")})
   if weights!=1||cached.buf!=b{t.Fatal("immutable delegation changed")}
   copied,stopped,probed:=0,0,0;sourceBytes:=slices.Clone(s.buf.(*hostBuf).i8)
   out,r:=c.offloadRefillWithOperations(d,s,int64(n),true,
    func(x,y unsafe.Pointer,z int)bool{probed++;return x==p&&y==unsafe.Pointer(&s.buf.(*hostBuf).i8[0])&&z==n},
    func(x,y unsafe.Pointer,z int)int{
     copied++;a:=cudaOffloadRefillActive
     if a==nil||a.source.buf!=s.buf||a.destination.buf!=b||a.replacement.buf==b||b.ptr!=nil||!b.invalidForSubmit()||x!=p||z!=n{t.Fatal("IQ copy transition")}
     copy(unsafe.Slice((*int8)(x),z),unsafe.Slice((*int8)(y),z));return 0
    },func(unsafe.Pointer){stopped++})
   fresh,ok:=out.buf.(*cudaBuf)
   if !ok||r.State!=OwnedTransferComplete||r.Err!=nil||fresh==b||fresh.ptr!=p||!fresh.Ready()||b.Ready()||cudaOffloadRefillActive!=nil||copied!=1||probed!=1||stopped!=0{t.Fatal("IQ copy ownership")}
   if !slices.Equal(unsafe.Slice((*int8)(p),n),sourceBytes)||!slices.Equal(s.buf.(*hostBuf).i8,sourceBytes){t.Fatal("IQ payload/source changed")}
   if out.Dtype!=g.dt||out.Quant==s.Quant||out.Quant==d.Quant||out.Quant.Bits!=bits||out.Quant.Block!=256||out.Quant.Axis!=2||out.Quant.Symmetric||out.Quant.StaticAct||len(out.Quant.Scale)!=0||len(out.Quant.ZeroPoint)!=0{t.Fatal("IQ metadata identity")}
   s.Shape[0]++;s.Quant.Bits=7
   if out.Shape[0]!=shape[0]||out.Quant.Bits!=bits{t.Fatal("IQ descriptor alias")}
  }
  for _,bad:=range []string{"bits","block","axis","shape","host-extent","device-extent","limit"}{
   c,d,s,b=refillIQFixture(t,g.dt,g.block,g.dedicated,[]int{1,256});p=b.ptr
   switch bad{
   case "bits":s.Quant.Bits=1;if g.dt==IQ2_XS{s.Quant.Bits=2};if g.dt==IQ2_XXS{s.Quant.Bits=3}
   case "block":s.Quant.Block=32
   case "axis":s.Quant.Axis=1
   case "shape":s.Shape[1]=257;d.Shape[1]=257
   case "host-extent":s.buf.(*hostBuf).i8=s.buf.(*hostBuf).i8[:g.block-1]
   case "device-extent":b.n--
   case "limit":d.Shape=[]int{(64<<20)/g.block+1,256};s.Shape=slices.Clone(d.Shape)
   }
   if _,_,ok:=c.offloadRefillOperands(d,s,int64(g.block));ok||b.ptr!=p||b.invalidForSubmit()||cudaOffloadRefillActive!=nil{t.Fatalf("IQ pure refusal %v/%s",g,bad)}
  }
  if g.dt==IQ2_XXS{
   c,d,s,b=refillIQFixture(t,g.dt,g.block,g.dedicated,[]int{1,256});p=b.ptr
   s.Quant.Bits=6-d.Quant.Bits // each descriptor is valid alone; the pair is not.
   if _,ok:=cudaOffloadRefillBytes(d);!ok{t.Fatal("XXS destination control")}
   if _,ok:=cudaOffloadRefillBytes(s);!ok{t.Fatal("XXS source control")}
   if _,_,ok:=c.offloadRefillOperands(d,s,int64(g.block));ok||b.ptr!=p||b.invalidForSubmit()||cudaOffloadRefillActive!=nil{t.Fatal("cross-descriptor XXS reuse")}
  }
  c,d,s,b=refillIQFixture(t,g.dt,g.block,g.dedicated,[]int{1,256});p=b.ptr
  copied,stopped:=0,0
  _,r:=c.offloadRefillWithOperations(d,s,int64(g.block),true,
   func(unsafe.Pointer,unsafe.Pointer,int)bool{return true},
   func(unsafe.Pointer,unsafe.Pointer,int)int{copied++;return 77},
   func(x unsafe.Pointer){stopped++;if x!=p{t.Fatal("IQ stop pointer")}})
  a:=cudaOffloadRefillActive
  if a==nil||a.source.buf!=s.buf||a.destination.buf!=b||a.pointer!=p||a.failure==nil||b.ptr!=nil||!b.invalidForSubmit()||r.State!=OwnedTransferUnknown||r.Err!=a.failure||copied!=1||stopped!=1{t.Fatal("IQ failure retention")}
  fresh,ok:=a.replacement.buf.(*cudaBuf)
  if !ok||fresh==b||fresh.ptr!=p||!fresh.invalidForSubmit(){t.Fatal("IQ replacement retention")}
  _,retry:=c.offloadRefillWithOperations(d,s,int64(g.block),true,nil,
   func(unsafe.Pointer,unsafe.Pointer,int)int{copied++;return 0},func(unsafe.Pointer){stopped++})
  if retry.State!=OwnedTransferUnknown||copied!=1||cudaOffloadRefillActive!=a{t.Fatal("IQ failure retried/replaced roots")}
 }
 c,d,s,b=refillFixture();d.Dtype=F32;d.Quant=nil;d.Shape=[]int{2,3}
 if n,ok:=cudaOffloadRefillBytes(d);!ok||n!=24{t.Fatal("f32 geometry")}
 // Selected failures are terminal, with no second copy or successful reuse.
 for _,kind:=range []string{"refused","status","panic","reentry"}{
  c,d,s,b=refillFixture();p=b.ptr;copies,stops=0,0
  primary:=errors.New("original")
  var recovered any
  func(){
   defer func(){recovered=recover()}()
   _,r=c.offloadRefillWithOperations(d,s,144,true,
    func(unsafe.Pointer,unsafe.Pointer,int)bool{return kind!="refused"},
    func(unsafe.Pointer,unsafe.Pointer,int)int{
     copies++;if kind=="panic"{panic(primary)}
     if kind=="reentry"{c.offloadRefillWithOperations(d,s,144,true,nil,nil,func(unsafe.Pointer){stops++});return 0}
     return 77
    },func(unsafe.Pointer){stops++})
  }()
  a:=cudaOffloadRefillActive
  if a==nil||a.source.buf!=s.buf||a.destination.buf!=d.buf||a.failure==nil||!b.invalidForSubmit()||stops==0{t.Fatalf("failed retention %s",kind)}
  if kind=="panic"&&recovered!=primary{t.Fatal("panic replaced")}
  if kind!="panic"&&(recovered!=nil||r.State!=OwnedTransferUnknown||r.Err==nil){t.Fatal("failed receipt")}
  old:=copies
  c.offloadRefillWithOperations(d,s,144,true,nil,func(unsafe.Pointer,unsafe.Pointer,int)int{copies++;return 0},func(unsafe.Pointer){stops++})
  if copies!=old||cudaOffloadRefillActive!=a{t.Fatal("retry/root replacement")}
 }
 c,d,s,b=refillFixture();p=b.ptr;stops=0
 done:=make(chan struct{});normal:=false;var recovered any
 go func(){defer close(done);defer func(){recovered=recover()}()
  c.offloadRefillWithOperations(d,s,144,true,func(unsafe.Pointer,unsafe.Pointer,int)bool{return true},
   func(unsafe.Pointer,unsafe.Pointer,int)int{runtime.Goexit();return 0},
   func(x unsafe.Pointer){stops++;if x!=p{panic("wrong retained allocation")}})
  normal=true
 }()
 select{case <-done:case <-time.After(3*time.Second):t.Fatal("Goexit hung")}
 if normal||recovered!=nil||stops!=1||cudaOffloadRefillActive==nil||cudaOffloadRefillActive.pointer!=p||cudaOffloadRefillActive.source.buf!=s.buf||b.ptr!=nil||!b.invalidForSubmit(){t.Fatal("Goexit retention/propagation")}
}
`
