package compute

import (
	"context"
	"flag"
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

var cudaKVCloneCopyHost = flag.Bool("fak-cuda-kv-clone-copy-host", false, "run source-bound checked Clone submission and retained-owner witnesses")

// These source assertions bind status propagation and allocation-free rooting;
// they are not execution or copy-completion evidence.
// fak-test:runtime fast est=10ms lane=default
func TestCUDAKVCloneCopySourceContract(t *testing.T) {
	native := readCUDASource(t, "cuda/kv_precision.cuh")
	normalize := func(s string) string { return strings.Join(strings.Fields(s), " ") }
	if got := normalize(cudaFunctionBody(t, native, "kv_clone_copy_status")); got != "{ if (status != 0 && g_cuda_cleanup_status == 0) g_cuda_cleanup_status = status; return status; }" {
		t.Fatal("Clone copy lost first-status preservation")
	}
	want := "{ if (v41_candidates_stream_uncertain()) return kv_clone_copy_status(g_cuda_cleanup_status != 0 ? g_cuda_cleanup_status : (int)cudaErrorNotReady); cudaError_t status = cudaSuccess; FCUDA_TS(\"d2d_copy\", dim3(0, 0), status = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, g_stream)); return kv_clone_copy_status((int)status); }"
	if got := normalize(cudaFunctionBody(t, native, "fcuda_kv_clone_d2d_checked")); got != want {
		t.Fatal("Clone copy must preserve direct submission status and add no explicit fence/retry/free")
	}
	if !strings.Contains(readCUDASource(t, "cuda_kv_precision.h"), "int fcuda_kv_clone_d2d_checked(void *dst, const void *src, size_t bytes);") {
		t.Fatal("missing checked Clone ABI")
	}
	raw := readCUDASource(t, "cuda_kv.go")
	set := token.NewFileSet()
	file, err := parser.ParseFile(set, "cuda_kv.go", raw, 0)
	if err != nil {
		t.Fatal(err)
	}
	var helper, public string
	var helperNode *ast.FuncDecl
	for _, decl := range file.Decls {
		fn, ok := decl.(*ast.FuncDecl)
		if !ok || fn.Recv == nil || fn.Body == nil {
			continue
		}
		ptr, ok := fn.Recv.List[0].Type.(*ast.StarExpr)
		if !ok {
			continue
		}
		owner, ok := ptr.X.(*ast.Ident)
		if !ok || owner.Name != "cudaKV" {
			continue
		}
		body := normalize(raw[set.Position(fn.Body.Pos()).Offset:set.Position(fn.Body.End()).Offset])
		switch fn.Name.Name {
		case "Clone":
			public = body
		case "cloneWithOperations":
			helper = body
			helperNode = fn
		}
	}
	if !strings.Contains(public, "return int(C.fcuda_kv_clone_d2d_checked(dst, src, C.size_t(bytes)))") || strings.Contains(public, "C.fcuda_d2d(") {
		t.Fatal("public Clone must return the direct checked-copy status")
	}
	guard := "if published || n.releaseRetained { return }"
	if !strings.Contains(helper, guard) {
		t.Fatal("rollback must leave the retained destination untouched")
	}
	if helperNode == nil {
		t.Fatal("missing Clone helper")
	}
	text := func(node ast.Node) string {
		return normalize(raw[set.Position(node.Pos()).Offset:set.Position(node.End()).Offset])
	}
	matched := 0
	ast.Inspect(helperNode.Body, func(node ast.Node) bool {
		block, ok := node.(*ast.BlockStmt)
		if !ok {
			return true
		}
		for i, stmt := range block.List {
			check, ok := stmt.(*ast.IfStmt)
			if !ok || check.Init == nil || text(check.Init) != "status := copyDevice(np, src.ptr, src.len*src.dtype.Bytes())" {
				continue
			}
			matched++
			if i == 0 || text(block.List[i-1]) != "dst.ptr, dst.len, dst.cap = np, src.len, src.len" || text(check.Cond) != "status != 0" || check.Else != nil {
				t.Fatal("copy must follow exact row installation and test its returned status")
			}
			want := []string{
				"k.retainReleaseOwnerLocked()", "n.retainReleaseOwnerLocked()",
				"panic(&CUDAOpError{Op: \"Clone\", Site: site, Code: status, Class: MemoryKVCache, Msg: \"device copy submission failed; owners retained\"})",
			}
			if len(check.Body.List) != len(want) {
				t.Fatal("copy failure must root owners before constructing an error")
			}
			for j, expected := range want {
				if text(check.Body.List[j]) != expected {
					t.Fatalf("copy failure statement %d must be %s", j, expected)
				}
			}
		}
		return true
	})
	if matched != 1 {
		t.Fatal("missing or duplicate checked-copy failure branch")
	}
}

// The native adapter, latch and timestamp macro are actual source. CUDA and
// timestamp functions are mocked. Timestamp begin may already drain, and end
// may issue an event after copy failure: this checks no ADDITIONAL fence.
// fak-test:runtime slow est=240s lane=optin
func TestCUDAKVCloneCopyNativeHostContract(t *testing.T) {
	if !*cudaKVCloneCopyHost {
		t.Skip("set -fak-cuda-kv-clone-copy-host for checked-copy witnesses")
	}
	compiler, err := exec.LookPath("c++")
	if err != nil {
		t.Skip("host C++ compiler unavailable")
	}
	native := readCUDASource(t, "cuda/kv_precision.cuh")
	profile := readCUDASource(t, "cuda/kernel_profile.cuh")
	start := strings.Index(profile, "#define FCUDA_TS(")
	if start < 0 {
		t.Fatal("missing timestamp macro")
	}
	end := strings.Index(profile[start:], "\n\n")
	if end < 0 {
		t.Fatal("unterminated timestamp macro")
	}
	program := cudaKVCloneCopyNativeFixture + profile[start:start+end] + "\n" +
		"static bool v41_candidates_stream_uncertain(void) " + cudaFunctionBody(t, readCUDASource(t, "cuda_kernels.cu"), "v41_candidates_stream_uncertain") + "\n" +
		"static int kv_clone_copy_status(int status) " + cudaFunctionBody(t, native, "kv_clone_copy_status") + "\n" +
		"extern \"C\" int fcuda_kv_clone_d2d_checked(void *dst, const void *src, size_t bytes) " + cudaFunctionBody(t, native, "fcuda_kv_clone_d2d_checked") + "\n" + cudaKVCloneCopyNativeMain
	for _, variant := range []struct{ name, before, after, mode, failure string }{
		{name: "candidate", mode: "all"},
		{"latch-removed", "if (status != 0 && g_cuda_cleanup_status == 0) g_cuda_cleanup_status = status;", "", "failure", "native first error latch"},
		{"status-overwritten", "return kv_clone_copy_status((int)status);", "return kv_clone_copy_status(0);", "failure", "native direct copy status"},
		{"guard-removed", "if (v41_candidates_stream_uncertain())\n    return kv_clone_copy_status(g_cuda_cleanup_status != 0 ? g_cuda_cleanup_status : (int)cudaErrorNotReady);", "", "stopped", "native stopped entry"},
	} {
		t.Run(variant.name, func(t *testing.T) {
			source := program
			if variant.before != "" {
				if strings.Count(source, variant.before) != 1 {
					t.Fatal("native mutation boundary changed")
				}
				source = strings.Replace(source, variant.before, variant.after, 1)
			}
			dir := t.TempDir()
			cpp, binary := filepath.Join(dir, "copy.cpp"), filepath.Join(dir, "copy")
			if runtime.GOOS == "windows" {
				binary += ".exe"
			}
			if err := os.WriteFile(cpp, []byte(source), 0600); err != nil {
				t.Fatal(err)
			}
			ctx, cancel := context.WithTimeout(t.Context(), 55*time.Second)
			defer cancel()
			build := exec.CommandContext(ctx, compiler, "-std=c++17", "-O0", cpp, "-o", binary)
			build.WaitDelay = time.Second
			if output, err := build.CombinedOutput(); err != nil {
				t.Fatalf("compile actual copy adapter: %v\n%s", err, output)
			}
			run := exec.CommandContext(ctx, binary, variant.mode)
			run.WaitDelay = time.Second
			output, err := run.CombinedOutput()
			if variant.name == "candidate" {
				if err != nil {
					t.Fatalf("native copy contract: %v\n%s", err, output)
				}
			} else if err == nil || !strings.Contains(string(output), "FAIL: "+variant.failure) {
				t.Fatalf("native control must fail its named assertion: %v\n%s", err, output)
			}
		})
	}
}

// Actual Go Clone, retention and later Free/admission behavior with fake native
// status, allocation and copy boundaries. No CUDA/cgo/GC completion claim.
// fak-test:runtime slow est=300s lane=optin
func TestCUDAKVCloneCopyOwnerHostContract(t *testing.T) {
	if !*cudaKVCloneCopyHost {
		t.Skip("set -fak-cuda-kv-clone-copy-host for retained-owner witnesses")
	}
	program := cudaKVCloneCopyOwnerProgram(t)
	for _, variant := range []struct{ name, before, after, failure string }{
		{name: "candidate"},
		{"source-root-removed", "\t\t\tk.retainReleaseOwnerLocked()\n", "", "source owner retained"},
		{"destination-root-removed", "\t\t\tn.retainReleaseOwnerLocked()\n", "", "retained destination cleanup ran"},
		{"retained-cleanup-enabled", "if published || n.releaseRetained {", "if published {", "retained destination cleanup ran"},
		{"status-ignored", "if status := copyDevice(np, src.ptr, src.len*src.dtype.Bytes()); status != 0 {", "if status := copyDevice(np, src.ptr, src.len*src.dtype.Bytes()); false {", "copy failure published a result"},
	} {
		t.Run(variant.name, func(t *testing.T) {
			source := program
			if variant.before != "" {
				if strings.Count(source, variant.before) != 1 {
					t.Fatal("owner mutation boundary changed")
				}
				source = strings.Replace(source, variant.before, variant.after, 1)
			}
			dir := t.TempDir()
			goFile, binary := filepath.Join(dir, "main.go"), filepath.Join(dir, "copy")
			if runtime.GOOS == "windows" {
				binary += ".exe"
			}
			if err := os.WriteFile(goFile, []byte(source), 0600); err != nil {
				t.Fatal(err)
			}
			ctx, cancel := context.WithTimeout(t.Context(), 55*time.Second)
			defer cancel()
			build := exec.CommandContext(ctx, filepath.Join(runtime.GOROOT(), "bin", "go"), "build", "-o", binary, goFile)
			build.Env = append(os.Environ(), "CGO_ENABLED=0", "GOTOOLCHAIN=local", "GOFLAGS=")
			build.WaitDelay = time.Second
			if output, err := build.CombinedOutput(); err != nil {
				t.Fatalf("compile actual Clone owners: %v\n%s", err, output)
			}
			mode := "failure"
			if variant.name == "candidate" {
				mode = "all"
			} else if variant.name == "status-ignored" {
				mode = "last-failure"
			}
			run := exec.CommandContext(ctx, binary, mode)
			run.WaitDelay = time.Second
			output, err := run.CombinedOutput()
			if variant.name == "candidate" {
				if err != nil {
					t.Fatalf("copy owner contract: %v\n%s", err, output)
				}
			} else if err == nil || !strings.Contains(string(output), "FAIL: "+variant.failure) {
				t.Fatalf("owner control must fail its named assertion: %v\n%s", err, output)
			}
		})
	}
}

func cudaKVCloneCopyOwnerProgram(t *testing.T) string {
	t.Helper()
	wanted := map[string]bool{
		"dslice": false, "cudaKV": false, "cudaKVSnapshotCopy": false, "cudaKVReleaseOwners": false,
		"*cudaKV.Clone": false, "*cudaKV.cloneWithOperations": false, "*cudaKV.sharedStopErrorLocked": false,
		"*cudaKV.retainReleaseOwnerLocked": false, "*cudaKV.Free": false,
		"cudaQ56KMMVQReleaseBlocked": false, "itoaC": false,
		"releaseKVDeviceSlices": false, "releaseDeviceSlice": false, "clearDeviceSliceMetadata": false,
		"BackendError": false, "*BackendError.Error": false, "CUDAOpError": false, "*CUDAOpError.Error": false,
	}
	var program strings.Builder
	program.WriteString(cudaKVCloneCopyOwnerFixture)
	for _, path := range []string{"cuda_kv.go", "cuda.go", "backend_helpers.go", "vulkan_errors.go", "cuda_errors.go"} {
		raw := readCUDASource(t, path)
		set := token.NewFileSet()
		file, err := parser.ParseFile(set, path, raw, 0)
		if err != nil {
			t.Fatal(err)
		}
		emit := func(key string, node ast.Node, prefix string) {
			seen, ok := wanted[key]
			if !ok {
				return
			}
			if seen {
				t.Fatalf("duplicate owner declaration %s", key)
			}
			wanted[key] = true
			program.WriteString(prefix)
			program.WriteString(raw[set.Position(node.Pos()).Offset:set.Position(node.End()).Offset])
			program.WriteByte('\n')
		}
		for _, decl := range file.Decls {
			switch node := decl.(type) {
			case *ast.GenDecl:
				for _, spec := range node.Specs {
					switch spec := spec.(type) {
					case *ast.TypeSpec:
						emit(spec.Name.Name, spec, "type ")
					case *ast.ValueSpec:
						if len(spec.Names) == 1 {
							emit(spec.Names[0].Name, spec, node.Tok.String()+" ")
						}
					}
				}
			case *ast.FuncDecl:
				key := node.Name.Name
				if node.Recv != nil {
					ptr, ok := node.Recv.List[0].Type.(*ast.StarExpr)
					if !ok {
						continue
					}
					owner, ok := ptr.X.(*ast.Ident)
					if !ok {
						continue
					}
					key = "*" + owner.Name + "." + key
				}
				emit(key, node, "")
			}
		}
	}
	for key, seen := range wanted {
		if !seen {
			t.Fatalf("missing owner declaration %s", key)
		}
	}
	body := program.String()
	if strings.Count(body, "C.size_t") != 1 {
		t.Fatal("review copy scalar alias")
	}
	return strings.ReplaceAll(body, "C.size_t", "uint64") + cudaKVCloneCopyOwnerMain
}

const cudaKVCloneCopyNativeFixture = `
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
using cudaError_t=int;
static const int cudaSuccess=0,cudaErrorNotReady=34,cudaMemcpyDeviceToDevice=3;
static int g_cuda_cleanup_status=0,g_stream=7,copy_status=0,copies=0,begins=0,ends=0,profile_drains=0,last_error=0;
static bool g_v41_router_uncertain=false,g_v41_mmid_uncertain=false,profile_drain=false;
static void *last_dst=nullptr;static const void *last_src=nullptr;static size_t last_bytes=0;static int last_stream=0,last_kind=0;
struct dim3{unsigned x,y,z;explicit dim3(unsigned vx=1,unsigned vy=1,unsigned vz=1):x(vx),y(vy),z(vz){}};
static void require(bool ok,const char *message){if(!ok){std::fprintf(stderr,"FAIL: %s\n",message);std::exit(1);}}
static int fcuda_ts_begin(const char *name,dim3 grid){require(std::strcmp(name,"d2d_copy")==0&&grid.x==0&&grid.y==0,"native timing label/grid");++begins;if(profile_drain)++profile_drains;return 0;}
static void fcuda_ts_end(int slot){require(slot==0,"native timing slot");++ends;last_error=999;}
static cudaError_t cudaMemcpyAsync(void *dst,const void *src,size_t bytes,int kind,int stream){++copies;last_dst=dst;last_src=src;last_bytes=bytes;last_kind=kind;last_stream=stream;return copy_status;}
`

const cudaKVCloneCopyNativeMain = `
static void reset(){g_cuda_cleanup_status=copy_status=copies=begins=ends=profile_drains=last_error=0;g_v41_router_uncertain=g_v41_mmid_uncertain=profile_drain=false;last_dst=nullptr;last_src=nullptr;last_bytes=0;last_stream=last_kind=0;}
static void failure(){reset();copy_status=700;profile_drain=true;int a=1,b=0;const int rc=fcuda_kv_clone_d2d_checked(&b,&a,sizeof(a));require(rc==700,"native direct copy status");require(g_cuda_cleanup_status==700,"native first error latch");require(copies==1&&begins==1&&ends==1&&profile_drains==1&&last_error==999,"native profiling boundary");require(last_dst==&b&&last_src==&a&&last_bytes==sizeof(a)&&last_kind==cudaMemcpyDeviceToDevice&&last_stream==g_stream,"native copy arguments");require(kv_clone_copy_status(701)==701&&g_cuda_cleanup_status==700,"native first error overwritten");}
static void stopped(){reset();g_cuda_cleanup_status=41;int a=1,b=0;const int rc=fcuda_kv_clone_d2d_checked(&b,&a,sizeof(a));require(rc==41&&g_cuda_cleanup_status==41&&copies==0&&begins==0&&ends==0,"native stopped entry");}
int main(int argc,char **argv){require(argc==2,"choose native mode");if(std::strcmp(argv[1],"failure")==0){failure();return 0;}if(std::strcmp(argv[1],"stopped")==0){stopped();return 0;}require(std::strcmp(argv[1],"all")==0,"unknown native mode");failure();stopped();reset();g_v41_router_uncertain=true;int a=1,b=0;require(fcuda_kv_clone_d2d_checked(&b,&a,sizeof(a))==cudaErrorNotReady&&g_cuda_cleanup_status==cudaErrorNotReady&&copies==0&&begins==0&&ends==0,"native peer stop");reset();require(fcuda_kv_clone_d2d_checked(&b,&a,sizeof(a))==0&&g_cuda_cleanup_status==0&&copies==1&&begins==1&&ends==1&&profile_drains==0,"native healthy enqueue");return 0;}
`

const cudaKVCloneCopyOwnerFixture = `package main
import("fmt";"os";"reflect";"runtime";"strconv";"strings";"sync";"unsafe")
type Dtype int
const(F32 Dtype=iota;F16;BF16)
func(d Dtype)Bytes()int{if d==F32{return 4};return 2}
type KVConfig struct{NumLayers,NumKVHeads,HeadDim int;Precision string;GraphCapture bool}
type KVStore interface{}
type cudaBackend struct{}
type cudaBuf struct{ptr unsafe.Pointer}
type VulkanErrorClass string
type MemoryClass string
const(MemoryUnknown MemoryClass="unknown";MemoryKVCache MemoryClass="kv_cache")
var cudaMu sync.Mutex
var cleanupStatus,failAt,copies,releases,queries,invalidations int
type allocation struct{ptr unsafe.Pointer;n int;site string}
var allocations []allocation
var backing [][]byte
func require(ok bool,msg string){if !ok{fmt.Fprintln(os.Stderr,"FAIL:",msg);os.Exit(1)}}
func requireLock(){if cudaMu.TryLock(){cudaMu.Unlock();require(false,"native boundary without mutex")}}
func(*cudaBackend)dallocKV(n int,site string)*cudaBuf{requireLock();if cleanupStatus!=0{panic("allocation after copy stop")};b:=make([]byte,n);backing=append(backing,b);ptr:=unsafe.Pointer(&b[0]);allocations=append(allocations,allocation{ptr,n,site});return &cudaBuf{ptr}}
func(*cudaBackend)graphOnceInvalidateKVLocked(*cudaKV){invalidations++}
var C=struct{fcuda_kv_clone_d2d_checked func(unsafe.Pointer,unsafe.Pointer,uint64)int;fcuda_free func(unsafe.Pointer);fak_v41_checked_stream_terminal func()int}{
 func(dst,src unsafe.Pointer,n uint64)int{requireLock();copies++;if copies==failAt{if cleanupStatus==0{cleanupStatus=700};return 700};copy(unsafe.Slice((*byte)(dst),int(n)),unsafe.Slice((*byte)(src),int(n)));return 0},
 func(unsafe.Pointer){requireLock();releases++},
 func()int{requireLock();queries++;if cleanupStatus!=0{return 1};return 0},
}
func recoverCall(f func())(r any){defer func(){r=recover()}();f();return}
func saved(k *cudaKV)*cudaKV{v:=*k;v.K=append([]dslice(nil),k.K...);v.Kraw=append([]dslice(nil),k.Kraw...);v.V=append([]dslice(nil),k.V...);v.pos=append([]int(nil),k.pos...);return &v}
func makeSource(dtype Dtype)*cudaKV{
 k:=&cudaKV{be:&cudaBackend{},cfg:KVConfig{NumLayers:2,NumKVHeads:1,HeadDim:2,Precision:fmt.Sprint(dtype)},maxPos:7,K:make([]dslice,2),Kraw:make([]dslice,2),V:make([]dslice,2),pos:[]int{3,8}}
 for p,rows:=range [][]dslice{k.K,k.Kraw,k.V}{for i:=range rows{d:=dtype;if p==1{d=F32};n:=i+2;b:=make([]byte,(n+3)*d.Bytes());for j:=range b{b[j]=byte(1+30*p+7*i+j)};backing=append(backing,b);rows[i]=dslice{ptr:unsafe.Pointer(&b[0]),len:n,cap:n+3,dtype:d}}}
 return k
}
func payloads(k *cudaKV)[][]byte{var out [][]byte;for _,rows:=range [][]dslice{k.K,k.Kraw,k.V}{for _,d:=range rows{out=append(out,append([]byte(nil),unsafe.Slice((*byte)(d.ptr),d.cap*d.dtype.Bytes())...))}};return out}
func orderedRows(k *cudaKV)[]dslice{var out []dslice;for l:=range k.K{out=append(out,k.K[l],k.Kraw[l],k.V[l])};return out}
func reset(){cleanupStatus=0;failAt=0;copies=0;releases=0;queries=0;invalidations=0;allocations=nil;cudaKVReleaseOwners=nil}
`

const cudaKVCloneCopyOwnerMain = `
func failure(dtype Dtype,at int){
 reset();failAt=at;tail:=&cudaKV{releaseRetained:true};cudaKVReleaseOwners=tail;source:=makeSource(dtype);before:=saved(source);bytesBefore:=payloads(source);var out KVStore
 r:=recoverCall(func(){out=source.Clone()});e,ok:=r.(*CUDAOpError)
 require(out==nil&&ok,"copy failure published a result")
 require(e.Op=="Clone"&&e.Code==700&&e.Class==MemoryKVCache&&e.Site==allocations[at-1].site&&strings.Contains(e.Msg,"owners retained"),"copy error identity")
 require(releases==0,"retained destination cleanup ran")
 require(copies==at&&len(allocations)==at&&queries==1&&cleanupStatus==700,"copy work continued after failure")
 require(source.releaseRetained&&source.releaseNext==tail,"source owner retained")
 n:=cudaKVReleaseOwners;require(n!=nil&&n!=source&&n!=tail&&n.releaseRetained&&n.releaseNext==source,"destination owner retained")
 expectedSource:=*before;expectedSource.releaseRetained=true;expectedSource.releaseNext=tail;require(reflect.DeepEqual(source,&expectedSource)&&reflect.DeepEqual(payloads(source),bytesBefore),"source data/metadata changed")
 expected:=&cudaKV{be:source.be,cfg:source.cfg,K:make([]dslice,2),Kraw:make([]dslice,2),V:make([]dslice,2),pos:append([]int(nil),source.pos...),releaseRetained:true,releaseNext:source}
 original:=orderedRows(source);wantPointers:=map[unsafe.Pointer]bool{}
 for i,a:=range allocations{src:=original[i];require(a.n==src.len*src.dtype.Bytes()&&a.ptr!=src.ptr&&!wantPointers[a.ptr],"allocation identity/extent");wantPointers[a.ptr]=true;d:=dslice{ptr:a.ptr,len:src.len,cap:src.len,dtype:src.dtype};switch i%3{case 0:expected.K[i/3]=d;case 1:expected.Kraw[i/3]=d;case 2:expected.V[i/3]=d}}
 require(reflect.DeepEqual(n,expected),"destination row metadata changed")
 gotPointers:=map[unsafe.Pointer]bool{};for _,d:=range orderedRows(n){if d.ptr!=nil{require(!gotPointers[d.ptr],"duplicate destination pointer");gotPointers[d.ptr]=true}}
 require(reflect.DeepEqual(gotPointers,wantPointers),"exact distinct destination pointer set")
 seen:=map[*cudaKV]bool{};count:=0;for owner:=cudaKVReleaseOwners;owner!=nil;owner=owner.releaseNext{require(!seen[owner],"retention cycle");seen[owner]=true;count++};require(count==3&&seen[tail]&&tail.releaseNext==nil,"retention tail changed")
 require(cudaMu.TryLock(),"failed Clone left mutex locked");cudaMu.Unlock()
 // Even if the fake native status clears, retained owners stay inert at these
 // existing Free/Clone entry points. This is not all-consumer coverage.
 cleanupStatus=0;queries=0;sourceBefore,nBefore:=saved(source),saved(n);source.Free();n.Free()
 for _,owner:=range []*cudaKV{source,n}{r:=recoverCall(func(){owner.Clone()});err,ok:=r.(*BackendError);require(ok&&err.Class=="cuda-terminal-stream"&&err.Site=="kv-clone","retained Clone admission")}
 require(queries==0&&releases==0&&invalidations==0&&copies==at&&len(allocations)==at&&reflect.DeepEqual(source,sourceBefore)&&reflect.DeepEqual(n,nBefore)&&cudaKVReleaseOwners==n,"later entry changed retained owners")
 require(cudaMu.TryLock(),"later entry left mutex locked");cudaMu.Unlock()
}
func healthy(){reset();tail:=&cudaKV{releaseRetained:true};cudaKVReleaseOwners=tail;source:=makeSource(F32);before:=saved(source);out:=source.Clone().(*cudaKV);require(copies==6&&len(allocations)==6&&releases==0&&queries==1&&cudaKVReleaseOwners==tail&&!out.releaseRetained&&reflect.DeepEqual(source,before),"healthy clone contract");for i,d:=range orderedRows(out){src:=orderedRows(source)[i];require(d.ptr!=src.ptr&&d.len==src.len&&d.cap==src.len&&d.dtype==src.dtype&&reflect.DeepEqual(unsafe.Slice((*byte)(d.ptr),d.len*d.dtype.Bytes()),unsafe.Slice((*byte)(src.ptr),src.len*src.dtype.Bytes())),"healthy clone payload")};out.pos[0]=99;require(source.pos[0]==3,"healthy position alias")}
func main(){require(len(os.Args)==2,"choose owner mode");if os.Args[1]=="failure"{failure(F32,4)}else if os.Args[1]=="last-failure"{failure(F32,6)}else{require(os.Args[1]=="all","unknown owner mode");for _,dtype:=range []Dtype{F32,F16,BF16}{failure(dtype,1);failure(dtype,4);failure(dtype,6)};healthy()};runtime.KeepAlive(backing)}
`
