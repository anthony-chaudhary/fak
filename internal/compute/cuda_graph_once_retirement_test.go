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

var graphOnceRetirementHost = flag.Bool("fak-cuda-graph-once-retirement-host", false, "run source-derived graph arena retirement host controls; no GPU qualification")

func graphOnceRetirementMethods(t *testing.T) map[string]string {
	t.Helper()
	source := readCUDASource(t, "cuda_graph_once_qualification.go")
	fs := token.NewFileSet()
	file, err := parser.ParseFile(fs, "cuda_graph_once_qualification.go", source, 0)
	if err != nil {
		t.Fatal(err)
	}
	out := map[string]string{}
	for _, decl := range file.Decls {
		if f, ok := decl.(*ast.FuncDecl); ok && f.Body != nil {
			out[f.Name.Name] = source[fs.Position(f.Pos()).Offset:fs.Position(f.End()).Offset]
		}
	}
	return out
}

// No release status, free-list transfer, or earlier successful free can certify
// a different arena member. These guards bind the actual native and Go paths.
// fak-test:runtime fast est=10ms lane=default
func TestCUDAGraphOnceRetirementContract(t *testing.T) {
	graph := readCUDASource(t, "cuda/graph_capture_once.cuh")
	native := cudaFunctionBody(t, graph, "fak_graph_once_arena_release_checked")
	ordered := func(body string, tokens ...string) {
		t.Helper()
		at := 0
		for _, token := range tokens {
			index := strings.Index(body[at:], token)
			if index < 0 {
				t.Fatalf("missing/reordered retirement step %q", token)
			}
			at += index + len(token)
		}
	}
	ordered(native, "if (g_cuda_cleanup_status)", "kv_snapshot_release_refused(expected)", "cudaPeekAtLastError()", "cudaStreamIsCapturing", "g_live.find(expected)", "g_managed_live.find(expected)", "g_panel_live.find(expected)", "!= 1", "fak_alloc_tags_guard guard", "tag.bytes != bytes", "status = cudaFree(expected)", "if (status != cudaSuccess) return fak_graph_once_arena_failed", "fak_alloc_untrack(expected)", "g_live.erase(ordinary)", "g_managed_live.erase(managed)", "g_panel_live_bytes -= bytes", "return 0")
	for _, token := range []string{"fcuda_free(", "fcuda_pool_put(", "g_panel_pool[", "q4_coeff_cache_evict(", "cudaGetLastError(", "cudaStreamSynchronize("} {
		if strings.Contains(native, token) {
			t.Fatalf("arena release gained unchecked work %q", token)
		}
	}
	failed := cudaFunctionBody(t, graph, "fak_graph_once_arena_failed")
	ordered(failed, "if (!status)", "if (!g_cuda_cleanup_status)", "g_cuda_cleanup_status = status", "return g_cuda_cleanup_status")
	methods := graphOnceRetirementMethods(t)
	body := methods["graphOnceReleaseLocked"]
	ordered(body, "c.faultLatch.Admit", "s.invalidated = true", "graphOnceArenaBufferLegal", "C.fak_graph_once_release(&s.owner)", "C.fak_graph_once_arena_release_checked", "b.ptr = nil", "b.graphOncePinned = false", "cached.buf == b", "delete(uploadCache, key)", "b.hostKeep = nil", "c.graphOnce = nil", "cudaGraphOnceOwner = nil")
	if strings.Contains(body, "C.fcuda_free(") {
		t.Fatal("unchecked free returned to graph arena")
	}
	ordered(methods["GraphOnceBegin"], "input.Dtype != F32", "!graphOnceArenaBufferLegal(b, true)", "C.fcuda_graph_prewarm", "C.fak_graph_once_begin")
	root := readCUDASource(t, "cuda_kernels.cu")
	ordered(root, "#include \"cuda/kv_precision.cuh\"", "#include \"cuda/graph_capture_once.cuh\"")
	// kv_snapshot_release_refused itself is bound to generic Free's full refusal
	// inventory by TestCUDAKVSnapshotReleaseSourceContract, not reimplemented here.
}

// Exact production functions with CUDA/free boundaries replaced. The native
// half uses actual ledger, shared-stop and peer-refusal definitions. The Go half
// extracts actual retirement methods, never a second modeled algorithm.
// fak-test:runtime slow est=120s lane=optin
func TestCUDAGraphOnceRetirementHost(t *testing.T) {
	if !*graphOnceRetirementHost {
		t.Skip("explicit host retirement witness not selected")
	}
	graph := readCUDASource(t, "cuda/graph_capture_once.cuh")
	native := readCUDASource(t, "cuda/kv_precision.cuh")
	root := readCUDASource(t, "cuda_kernels.cu")
	t.Run("native", func(t *testing.T) {
		compiler, err := exec.LookPath("c++")
		if err != nil {
			t.Fatal(err)
		}
		program := strings.TrimPrefix(readCUDASource(t, "cuda/alloc_ledger.h"), "#pragma once\n") + cudaKVSnapshotReleaseHarness + graphOnceRetirementNativeShim +
			"\nstatic bool v41_candidates_stream_uncertain(void) " + cudaFunctionBody(t, root, "v41_candidates_stream_uncertain") +
			readCUDASource(t, "cuda/v41_compressor_pending.cuh") +
			"\nstatic bool kv_snapshot_release_refused(void *d) " + cudaFunctionBody(t, native, "kv_snapshot_release_refused") +
			"\nextern \"C\" int fak_graph_once_arena_failed(int status) " + cudaFunctionBody(t, graph, "fak_graph_once_arena_failed") +
			"\nextern \"C\" int fak_graph_once_arena_release_checked(void *expected, size_t bytes, int f32_input) " + cudaFunctionBody(t, graph, "fak_graph_once_arena_release_checked")
		stop := strings.Index(cudaKVSnapshotReleaseMain, "static int release(")
		if stop < 0 {
			t.Fatal("common fake owner helpers moved")
		}
		program += cudaKVSnapshotReleaseMain[:stop] + graphOnceRetirementNativeMain
		dir := t.TempDir()
		cpp := filepath.Join(dir, "retire.cpp")
		if err := os.WriteFile(cpp, []byte(program), 0600); err != nil {
			t.Fatal(err)
		}
		for _, selected := range []bool{false, true} {
			binary := filepath.Join(dir, "retire")
			if runtime.GOOS == "windows" {
				binary += ".exe"
			}
			args := []string{"-std=c++17", "-O0", cpp, "-o", binary}
			if selected {
				args = append([]string{"-DSNAPSHOT_TEST_SELECTED=1"}, args...)
			}
			ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
			cmd := exec.CommandContext(ctx, compiler, args...)
			cmd.WaitDelay = time.Second
			if output, err := cmd.CombinedOutput(); err != nil {
				cancel()
				t.Fatalf("compile retirement: %v\n%s", err, output)
			}
			cmd = exec.CommandContext(ctx, binary)
			cmd.WaitDelay = time.Second
			output, err := cmd.CombinedOutput()
			cancel()
			if err != nil {
				t.Fatalf("retirement: %v\n%s", err, output)
			}
		}
	})
	t.Run("go-owner", func(t *testing.T) {
		goExe, err := exec.LookPath("go")
		if err != nil {
			t.Fatal(err)
		}
		methods := graphOnceRetirementMethods(t)
		program := graphOnceRetirementGoShim
		replace := strings.NewReplacer("C.fak_graph_once_arena_release_checked", "fakeArenaRelease", "C.fak_graph_once_arena_failed", "fakeFailed", "C.fak_graph_once_release", "fakeGraphRelease", "C.int", "int", "C.size_t", "uintptr")
		for _, name := range []string{"graphOnceCheck", "graphOnceArenaBufferLegal", "graphOnceReleaseLocked"} {
			if methods[name] == "" {
				t.Fatalf("missing production method %s", name)
			}
			program += "\n" + replace.Replace(methods[name]) + "\n"
		}
		program += graphOnceRetirementGoMain
		dir := t.TempDir()
		path := filepath.Join(dir, "retire.go")
		if err := os.WriteFile(path, []byte(program), 0600); err != nil {
			t.Fatal(err)
		}
		ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
		defer cancel()
		cmd := exec.CommandContext(ctx, goExe, "run", "-p=1", "-trimpath", "-buildvcs=false", path)
		cmd.Dir = dir
		cmd.WaitDelay = time.Second
		for _, entry := range os.Environ() {
			key, _, _ := strings.Cut(entry, "=")
			switch key {
			case "GOWORK", "GOENV", "GOTOOLCHAIN", "CGO_ENABLED", "GOMAXPROCS", "GOFLAGS":
				continue
			}
			cmd.Env = append(cmd.Env, entry)
		}
		cmd.Env = append(cmd.Env, "GOWORK=off", "GOENV=off", "GOTOOLCHAIN=local", "CGO_ENABLED=0", "GOMAXPROCS=1", "GOFLAGS=")
		if output, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("source-derived Go owner: %v\n%s", err, output)
		}
	})
}

const graphOnceRetirementNativeShim = `
static void *g_graph_once_capture=nullptr;
using cudaStreamCaptureStatus=int;
static constexpr int cudaStreamCaptureStatusNone=0;
static int capture_query_status=0, actual_capture=0, prior_error=0;
static cudaError_t cudaPeekAtLastError(){return prior_error;}
static cudaError_t cudaStreamIsCapturing(int,cudaStreamCaptureStatus *out){*out=actual_capture;return capture_query_status;}
`
const graphOnceRetirementNativeMain = `
static void setup(void *p,int type=0){reset();g_graph_once_capture=nullptr;capture_query_status=actual_capture=prior_error=0;own(p,16,type,"transient");}
int main(){
 int storage[2]={};void *p=&storage[0];
 for(int type=0;type<3;++type){setup(p,type);require(fak_graph_once_arena_release_checked(p,16,0)==0&&frees==1&&g_live.empty()&&g_managed_live.empty()&&g_panel_live.empty()&&g_alloc_tags.empty()&&g_panel_live_bytes==0&&g_pool.empty()&&g_panel_pool.empty()&&syncs==0,"exact physical retirement");}
 for(int mode=0;mode<13;++mode){
  setup(p);
  if(mode==0)g_live[p]=17;
  if(mode==1)g_managed_live[p]=16;
  if(mode==2)g_alloc_tags.erase(p);
  if(mode==3)g_alloc_tags[p].bytes=17;
  if(mode==4)g_alloc_tags[p].kind=FAK_ALLOC_MANAGED;
  if(mode==5)g_alloc_tags[p].cls=FAK_ALLOC_Q4_COEFF_CACHE;
  if(mode==6)g_alloc_tags[p].cls=FAK_ALLOC_WEIGHTS;
  if(mode==7)g_capture_open=true;
  if(mode==8)g_graph_once_capture=p;
  if(mode==9)actual_capture=1;
  if(mode==10)capture_query_status=719;
  if(mode==11)prior_error=700;
  if(mode==12)g_v41_router_uncertain=true;
  int rc=fak_graph_once_arena_release_checked(p,16,0);
  require(rc!=0&&frees==0&&g_live.count(p)&&syncs==0&&g_cuda_cleanup_status==rc,"refusal lost root or touched allocator");
  require(fak_graph_once_arena_failed(888)==rc,"first failure overwritten");
 }
 setup(p);g_alloc_tags[p].cls=FAK_ALLOC_WEIGHTS;require(fak_graph_once_arena_release_checked(p,16,1)==0&&frees==1,"F32 input weight permitted");
 for(int type=0;type<3;++type){setup(p,type);free_status=719;require(fak_graph_once_arena_release_checked(p,16,0)==719&&frees==1&&g_alloc_tags.count(p)&&g_live.size()+g_managed_live.size()+g_panel_live.size()==1,"failed physical free lost exact owner");free_status=0;require(fak_graph_once_arena_release_checked(p,16,0)==719&&frees==1,"uncertain free retried");}
 setup(p,2);g_panel_live_bytes=15;require(fak_graph_once_arena_release_checked(p,16,0)!=0&&frees==0&&g_panel_live.count(p),"panel underflow");
#ifdef SNAPSHOT_TEST_SELECTED
 setup(p);g_v41_rope_quarantine[0]=p;require(fak_graph_once_arena_release_checked(p,16,0)!=0&&frees==0,"peer quarantine omitted");
#endif
 return 0;
}
`
const graphOnceRetirementGoShim = `package main
import("fmt";"unsafe")
type MemoryClass int
const(MemoryScratchpad MemoryClass=iota;MemoryActivation;MemoryWeights)
const F32=1
const DeviceFaultExecution=1
type cudaBuf struct{ptr unsafe.Pointer;n,device,scalesN int;scales unsafe.Pointer;budgetedWeightBytes int64;managedWeight,managed,graphOncePinned bool;class MemoryClass;host uintptr;hostDt,hostLo int;hostKeep any}
type Tensor struct{buf any}
type ucKey struct{host uintptr;dt,lo int}
var uploadCache=map[ucKey]Tensor{}
type cudaGraphOnceState struct{owner unsafe.Pointer;capturing,invalidated bool;buffers []*cudaBuf;input Tensor}
type latch struct{code int}
func(l *latch)Observe(_ int,_ string,code int){if l.code==0 {l.code=code}}
func(l *latch)Admit(_ string)error{if l.code!=0{return fmt.Errorf("stopped %d",l.code)};return nil}
type cudaBackend struct{graphOnce *cudaGraphOnceState;faultLatch *latch}
var cudaGraphOnceOwner *cudaBackend
var calls,graphCalls,failAt,graphStatus,stopped int
func fakeFailed(code int)int{if stopped==0 {if code==0 {code=1};stopped=code};return stopped}
func fakeGraphRelease(owner *unsafe.Pointer)int{graphCalls++;if graphStatus!=0{return graphStatus};*owner=nil;return 0}
func fakeArenaRelease(_ unsafe.Pointer,_ uintptr,_ int)int{if stopped!=0{return stopped};calls++;if calls==failAt{return fakeFailed(719)};return 0}
func require(ok bool,why string){if !ok{panic(why)}}
func caught(f func())(value any){defer func(){value=recover()}();f();return nil}
func setup()(*cudaBackend,[]*cudaBuf){
 calls=0;graphCalls=0;failAt=0;graphStatus=0;stopped=0;uploadCache=map[ucKey]Tensor{}
 buffers:=make([]*cudaBuf,3)
 for i:=range buffers {buffers[i]=&cudaBuf{ptr:unsafe.Pointer(new(int)),n:16,class:MemoryScratchpad,graphOncePinned:true,hostKeep:new(int)}}
 buffers[0].class=MemoryWeights;buffers[0].host=1;buffers[0].hostDt=F32;uploadCache[ucKey{1,F32,0}]=Tensor{buffers[0]}
 c:=&cudaBackend{graphOnce:&cudaGraphOnceState{owner:unsafe.Pointer(new(int)),buffers:buffers,input:Tensor{buffers[0]}},faultLatch:&latch{}}
 cudaGraphOnceOwner=c;return c,buffers
}
`
const graphOnceRetirementGoMain = `
func main(){
 c,b:=setup();c.graphOnceReleaseLocked();require(c.graphOnce==nil&&cudaGraphOnceOwner==nil&&calls==3&&graphCalls==1&&len(uploadCache)==0,"healthy owner retirement")
 for _,x:=range b{require(x.ptr==nil&&!x.graphOncePinned&&x.hostKeep==nil,"healthy buffer not cleared")}
 for failure:=1;failure<=3;failure++{
  c,b=setup();s:=c.graphOnce;keep:=b[failure-1].hostKeep;failAt=failure
  require(caught(c.graphOnceReleaseLocked)!=nil,"selected failure returned normally")
  require(c.graphOnce==s&&cudaGraphOnceOwner==c&&s.invalidated&&s.owner==nil&&calls==failure&&c.faultLatch.code==719,"partial retirement lost graph root")
  for i,x:=range b {if i<failure-1{require(x.ptr==nil&&!x.graphOncePinned&&x.hostKeep==nil,"retired owner retained as retryable")}else{require(x.ptr!=nil&&x.graphOncePinned&&x.hostKeep!=nil,"unretired owner lost")}}
  require(b[failure-1].hostKeep==keep,"exact host root lost")
  attempts,g:=calls,graphCalls;require(caught(c.graphOnceReleaseLocked)!=nil&&calls==attempts&&graphCalls==g,"caught failure retried cleanup")
  if failure==1 {require(uploadCache[ucKey{1,F32,0}].buf==b[0],"failed input lost cache owner")}
 }
 c,b=setup();graphStatus=700;require(caught(c.graphOnceReleaseLocked)!=nil&&calls==0&&c.graphOnce.owner!=nil&&b[0].graphOncePinned,"failed graph retirement unpinned arena")
 c,b=setup();b[1].scales=unsafe.Pointer(new(int));require(caught(c.graphOnceReleaseLocked)!=nil&&calls==0&&graphCalls==0&&c.graphOnce!=nil,"malformed later member partially retired")
 c,b=setup();foreign:=&cudaBuf{};uploadCache[ucKey{1,F32,0}]=Tensor{foreign};c.graphOnceReleaseLocked();require(uploadCache[ucKey{1,F32,0}].buf==foreign,"retiring stale input evicted foreign cache owner")
 c,b=setup();c.graphOnce.capturing=true;require(caught(c.graphOnceReleaseLocked)!=nil&&calls==0&&graphCalls==0&&b[0].graphOncePinned,"capture cleanup reached native")
 c,b=setup();cudaGraphOnceOwner=&cudaBackend{};require(caught(c.graphOnceReleaseLocked)!=nil&&calls==0&&graphCalls==0,"foreign owner retired")
 c,b=setup();require(graphOnceArenaBufferLegal(b[0],true)&&!graphOnceArenaBufferLegal(b[0],false),"F32 input role lost")
}
`
