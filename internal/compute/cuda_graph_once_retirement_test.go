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
	ordered(methods["GraphOnceBegin"], "graphOnceBindKVLocked(c, pos)", "C.fcuda_graph_prewarm", "c.graphOnce = s", "C.fak_graph_once_create", "C.fak_graph_once_begin")
	root := readCUDASource(t, "cuda_kernels.cu")
	ordered(root, "#include \"cuda/kv_precision.cuh\"", "#include \"cuda/graph_capture_once.cuh\"")
	state := readCUDASource(t, "cuda_graph_once_state.go")
	for _, token := range []string{"dependencyInvalidated", "hostKeep", "b.bornGen == d.bornGen", "b.scales == d.scales", "b.be == d.be", "b.graphOncePinned == d.pinned", "d.arenaIndex != i", "len(s.dependencies) != len(s.buffers)+len(s.borrowed)"} {
		if !strings.Contains(state, token) {
			t.Fatalf("missing immutable dependency contract %q", token)
		}
	}
	ordered(methods["graphOnceRequireDependenciesLocked"], "s.graphOnceDependenciesValid()", "s.dependencyInvalidated = true", "C.fak_graph_once_arena_failed(-75092)")
	ordered(methods["GraphOnceSave"], "graphOnceRequireDependenciesLocked()", "s.dependencies[b]", "s.output = output")
	ordered(methods["GraphOnceEnd"], "graphOnceRequireDependenciesLocked()", "s.capturing = false", "C.fak_graph_once_end")
	ordered(methods["GraphOnceReplay"], "graphOnceRequireDependenciesLocked()", "C.fak_graph_once_copy_input", "C.fak_graph_once_replay")
	ordered(methods["graphOnceReleaseLocked"], "graphOnceRequireDependenciesLocked()", "C.fak_graph_once_release", "C.fak_graph_once_arena_release_checked")
	// Both default and qualification archives export the batch definition. Keep
	// its declaration in the shared ABI root enrolled by tools/cuda_abi_parity.py,
	// not a Go-local preamble that the parity collector does not enroll.
	header := readCUDASource(t, "cuda_backend.h")
	compact := strings.NewReplacer(" ", "", "\t", "", "\r", "", "\n", "").Replace(header)
	prototype := "intfcuda_alloc_reservations_match(constvoid*const*pointers,constsize_t*bytes,constuint64_t*ids,intcount);"
	if strings.Count(compact, prototype) != 1 {
		t.Fatal("batch reservation ABI must be declared exactly once in the shared header")
	}
	preamble, _, ok := strings.Cut(state, "import \"C\"")
	if !ok || !strings.Contains(preamble, "#include \"cuda_backend.h\"") || strings.Contains(preamble, "fcuda_alloc_reservations_match") {
		t.Fatal("graph state must obtain the batch ABI from the shared header")
	}
	ledger := readCUDASource(t, "cuda/alloc_ledger.h")
	nativeBatch := cudaFunctionBody(t, ledger, "fcuda_alloc_reservations_match")
	ordered(nativeBatch, "count <= 0", "SIZE_MAX / sizeof(*pointers)", "fak_alloc_tags_guard guard", "if (!pointers[i] || !bytes[i] || !ids[i])", "tag.reservation != ids[i]", "FAK_ALLOC_IDLE_POOL", "FAK_ALLOC_PINNED_HOST", "return 1;", "#else", "return 0;")
	if strings.Count(state, "C.fcuda_alloc_reservations_match(") != 1 {
		t.Fatal("one batch validation call required")
	}
	ordered(state, "valid := C.fcuda_alloc_reservations_match(", "runtime.KeepAlive(s)", "return valid")
	for _, token := range []string{"reservationSlotsMatch", "d.reservationIndex > len(s.reservationPointers)-count", "len(s.reservationPointers) > 2147483647", "len(s.reservationBytes) != len(s.reservationPointers)", "expectedTickets != len(s.reservationPointers)"} {
		if !strings.Contains(state, token) {
			t.Fatalf("missing ticket bound %q", token)
		}
	}
	for _, token := range []string{"kvBindingAttempted", "kvBound", "d.cap > maxFloats", "d.len != pos*w", "graphOnceKVSlotsValid", "live.len > live.cap", "s.kvTickets[at]", "C.size_t(saved.cap*4)"} {
		if !strings.Contains(state, token) {
			t.Fatalf("missing complete KV reservation contract %q", token)
		}
	}
	ordered(state, "s.kvBindingAttempted = true", "id := uint64(C.fcuda_alloc_reservation_id(d.ptr, C.size_t(bytes)))", "if id == 0", "s.kvBound = true")
	ordered(methods["graphOnceSameKV"], "graphOnceKVSlotsValid()", "pos != s.next", "row.len != pos*w")
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
				args = append([]string{"-DSNAPSHOT_TEST_SELECTED=1", "-DFAK_CUDA_CAPTURE_ONCE_QUALIFICATION=1"}, args...)
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
		state := readCUDASource(t, "cuda_graph_once_state.go")
		start := strings.Index(state, "var cudaGraphOnceOwner *cudaBackend")
		if start < 0 {
			t.Fatal("actual graph state declarations missing")
		}

		replace := strings.NewReplacer("C.fak_graph_once_arena_release_checked", "fakeArenaRelease", "C.fak_graph_once_arena_failed", "fakeFailed", "C.fak_graph_once_release", "fakeGraphRelease", "C.fcuda_alloc_reservation_id", "fakeReservationID", "C.fcuda_alloc_reservations_match", "fakeReservationBatch", "C.fcuda_graph_prewarm", "fakePrewarm", "C.fak_graph_once_create", "fakeCreate", "C.fak_graph_once_begin", "fakeBegin", "C.fak_graph_once_end", "fakeEnd", "C.fak_graph_once_copy_input", "fakeCopyInput", "C.fak_graph_once_replay", "fakeReplay", "C.uint64_t", "uint64", "C.int", "int", "C.size_t", "uintptr")
		program += "\n" + replace.Replace(state[start:]) + "\n"
		for _, name := range []string{"graphOnceCheck", "graphOnceArenaBufferLegal", "graphOnceRequireDependenciesLocked", "graphOnceReleaseLocked", "graphOnceSameKV", "GraphOnceBegin", "GraphOnceEnd", "GraphOnceSave", "GraphOnceReplay"} {
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

static void reservation_controls(){
 int a=0,b=0;void *pointers[2]={&a,&b};size_t bytes[2]={sizeof(a),sizeof(b)};uint64_t ids[2]={1,1};
 reset();own(&a,sizeof(a),0,"transient");own(&b,sizeof(b),0,"transient");
#ifdef FAK_CUDA_CAPTURE_ONCE_QUALIFICATION
 ids[0]=fcuda_alloc_reservation_id(&a,sizeof(a));ids[1]=fcuda_alloc_reservation_id(&b,sizeof(b));
 require(ids[0]&&ids[1]&&fcuda_alloc_reservations_match(pointers,bytes,ids,2)==1,"actual native batch positive");
 uint64_t first=ids[1];fak_alloc_retag(&b,FAK_ALLOC_ACTIVATIONS_SCRATCH,FAK_ALLOC_SITE("transient"));
 require(fcuda_alloc_reservations_match(pointers,bytes,ids,2)==0,"actual native batch missed retag");
 ids[1]=fcuda_alloc_reservation_id(&b,sizeof(b));require(ids[1]!=first&&fcuda_alloc_reservations_match(pointers,bytes,ids,2)==1,"fresh owner batch positive");
 require(fak_alloc_untrack(&b),"fake release metadata");fak_alloc_track(&b,FAK_ALLOC_ACTIVATIONS_SCRATCH,FAK_ALLOC_DEVICE_LOCAL,FAK_ALLOC_SITE("transient"),sizeof(b));
 require(fcuda_alloc_reservations_match(pointers,bytes,ids,2)==0,"actual same-address reuse accepted");ids[1]=fcuda_alloc_reservation_id(&b,sizeof(b));
 uint64_t saved=ids[1];ids[1]=0;require(fcuda_alloc_reservations_match(pointers,bytes,ids,2)==0,"zero ticket accepted");ids[1]=saved;
 bytes[1]--;require(fcuda_alloc_reservations_match(pointers,bytes,ids,2)==0,"partial scale extent accepted");bytes[1]++;
 pointers[1]=(char*)&b+1;require(fcuda_alloc_reservations_match(pointers,bytes,ids,2)==0,"interior base accepted");pointers[1]=&b;
 fak_alloc_retag(&b,FAK_ALLOC_IDLE_POOL,FAK_ALLOC_SITE("device-pool"));ids[1]=g_alloc_tags[&b].reservation;
 require(fcuda_alloc_reservations_match(pointers,bytes,ids,2)==0,"idle pool accepted with current generation");
#else
 require(fcuda_alloc_reservations_match(pointers,bytes,ids,2)==0,"default batch profile accepted tickets");
#endif
 require(fcuda_alloc_reservations_match(nullptr,bytes,ids,2)==0&&fcuda_alloc_reservations_match(pointers,nullptr,ids,2)==0&&fcuda_alloc_reservations_match(pointers,bytes,nullptr,2)==0&&fcuda_alloc_reservations_match(pointers,bytes,ids,0)==0&&fcuda_alloc_reservations_match(pointers,bytes,ids,-1)==0,"malformed array admission");
 require(frees==0&&syncs==0&&g_cuda_cleanup_status==0,"observational ticket query changed runtime state");
 reset();
}
int main(){
 reservation_controls();
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
import("fmt";"runtime";"unsafe";"sync";"sync/atomic")
type MemoryClass int
const(MemoryScratchpad MemoryClass=iota;MemoryActivation;MemoryWeights)
const F32=1
const DeviceFaultExecution=1
type cudaBuf struct{be *cudaBackend;bornGen uint64;invalid bool;ptr unsafe.Pointer;n,device,scalesN int;scales unsafe.Pointer;budgetedWeightBytes int64;managedWeight,managed,graphOncePinned bool;class MemoryClass;host uintptr;hostDt,hostLo int;hostKeep any}
func(b *cudaBuf)invalidForSubmit()bool{return b!=nil&&b.invalid}
type Tensor struct{buf any;Dtype Dtype}
func(t Tensor)Backend()*cudaBackend{return t.buf.(*cudaBuf).be}
func(t Tensor)Numel()int{return t.buf.(*cudaBuf).n/4}
type KVStore interface{}
type ucKey struct{host uintptr;dt,lo int}
var uploadCache=map[ucKey]Tensor{}
const KVPrecisionF32=0
type KVConfig struct{NumLayers,NumKVHeads,HeadDim,Precision int}
type cudaKVSnapshotCopy struct{}
type cudaKV struct{be *cudaBackend;cfg KVConfig;maxPos int;K,Kraw,V []dslice;pos []int;releaseRetained bool;snapshotCopy *cudaKVSnapshotCopy}
func(k *cudaKV)stride()int{return k.cfg.NumKVHeads*k.cfg.HeadDim}
type dslice struct{ptr unsafe.Pointer;len,cap int;dtype Dtype}
type Dtype = int
type Layout = int
type HostBuffer = any
type latch struct{code int}
func(l *latch)Observe(_ int,_ string,code int){if l.code==0 {l.code=code}}
func(l *latch)Admit(_ string)error{if l.code!=0{return fmt.Errorf("stopped %d",l.code)};return nil}
type cudaBackend struct{graphOnce *cudaGraphOnceState;faultLatch *latch;fenceGen uint64;capturing bool}
func(c *cudaBackend)cudaBufForSubmit(t Tensor)*cudaBuf{b:=t.buf.(*cudaBuf);c.graphOnceObserveLocked(b);return b}
var cudaMu sync.Mutex
var graphEnabled=true
var cudaKernelProfileOn bool
var cudaCaptureOnceMode=struct{enabled bool}{true}
const graphPrewarmDepth=1
func cudaAsyncGraphOnceAllowed()bool{return true}
var cudaGraphOnceWitnessCaptures,cudaGraphOnceWitnessReplays uint64
var cudaGraphReplays atomic.Int64
var prewarms,creates,begins,ends,copies,replays,launchStatus int
func fakePrewarm(int){prewarms++}
func fakeCreate(_ int,p *unsafe.Pointer)int{creates++;*p=unsafe.Pointer(new(int));return 0}
func fakeBegin(unsafe.Pointer,int)int{begins++;return 0}
func fakeEnd(unsafe.Pointer)int{ends++;return launchStatus}
func fakeCopyInput(unsafe.Pointer,unsafe.Pointer,unsafe.Pointer,uintptr)int{copies++;return 0}
func fakeReplay(unsafe.Pointer,int)int{replays++;return launchStatus}

type fakeReservation struct{bytes uintptr;id uint64}
var fakeReservations map[unsafe.Pointer]fakeReservation
var nextReservation uint64
var ticketQueries,ticketBatches int
var ticketSupported bool
func fakeTrack(p unsafe.Pointer,n uintptr){nextReservation++;fakeReservations[p]=fakeReservation{n,nextReservation}}
func fakeTrackBuffer(b *cudaBuf){fakeTrack(b.ptr,uintptr(b.n));if b.scales!=nil{fakeTrack(b.scales,uintptr(b.scalesN))}}
func fakeReservationID(p unsafe.Pointer,n uintptr)uint64{ticketQueries++;r,ok:=fakeReservations[p];if !ticketSupported||!ok||n==0||r.bytes!=n{return 0};return r.id}
func fakeReservationBatch(p *unsafe.Pointer,n *uintptr,ids *uint64,count int)int{
 ticketBatches++;if !ticketSupported||count<=0||p==nil||n==nil||ids==nil{return 0}
 pointers,bytes,want:=unsafe.Slice(p,count),unsafe.Slice(n,count),unsafe.Slice(ids,count)
 for i,q:=range pointers{got,ok:=fakeReservations[q];if !ok||want[i]==0||got.id!=want[i]||got.bytes!=bytes[i]{return 0}}
 return 1
}
var calls,graphCalls,failAt,graphStatus,stopped int
func fakeFailed(code int)int{if stopped==0 {if code==0 {code=1};stopped=code};return stopped}
func fakeGraphRelease(owner *unsafe.Pointer)int{graphCalls++;if graphStatus!=0{return graphStatus};*owner=nil;return 0}
func fakeArenaRelease(p unsafe.Pointer,_ uintptr,_ int)int{if stopped!=0{return stopped};calls++;if calls==failAt{return fakeFailed(719)};delete(fakeReservations,p);return 0}
func require(ok bool,why string){if !ok{panic(why)}}
func caught(f func())(value any){defer func(){value=recover()}();f();return nil}
func resetFixture(){
 fakeReservations=map[unsafe.Pointer]fakeReservation{};nextReservation=0;ticketQueries=0;ticketBatches=0;ticketSupported=true
 calls=0;graphCalls=0;failAt=0;graphStatus=0;stopped=0;uploadCache=map[ucKey]Tensor{}
 cudaGraphOnceOwner=nil;prewarms=0;creates=0;begins=0;ends=0;copies=0;replays=0;launchStatus=0
 cudaGraphOnceWitnessCaptures=0;cudaGraphOnceWitnessReplays=0;cudaGraphReplays.Store(0)
}
func fixtureKV(c *cudaBackend)*cudaKV{
 k:=&cudaKV{be:c,cfg:KVConfig{2,1,2,KVPrecisionF32},maxPos:4,K:make([]dslice,2),Kraw:make([]dslice,2),V:make([]dslice,2)}
 for _,rows:=range [3][]dslice{k.K,k.Kraw,k.V}{for i:=range rows{storage:=new([8]float32);rows[i]=dslice{ptr:unsafe.Pointer(&storage[0]),cap:8,dtype:F32};fakeTrack(rows[i].ptr,32)}}
 return k
}
func fixtureInput(c *cudaBackend)Tensor{
 b:=&cudaBuf{be:c,ptr:unsafe.Pointer(new([4]float32)),n:16,class:MemoryWeights,hostKeep:new(int)};fakeTrackBuffer(b);return Tensor{buf:b,Dtype:F32}
}
func setup()(*cudaBackend,[]*cudaBuf){
 resetFixture();c:=&cudaBackend{faultLatch:&latch{}}
 buffers:=make([]*cudaBuf,3)
 for i:=range buffers {buffers[i]=&cudaBuf{be:c,ptr:unsafe.Pointer(new(int)),n:16,class:MemoryScratchpad,hostKeep:new(int)}}
 buffers[0].class=MemoryWeights;buffers[0].host=1;buffers[0].hostDt=F32;uploadCache[ucKey{1,F32,0}]=Tensor{buf:buffers[0],Dtype:F32}
 c.graphOnce=&cudaGraphOnceState{kv:fixtureKV(c),owner:unsafe.Pointer(new(int)),capturing:true,input:Tensor{buf:buffers[0],Dtype:F32},next:1}
 require(c.graphOnce.graphOnceBindKVLocked(c,0),"fixture complete KV binding")
 cudaGraphOnceOwner=c;for _,b:=range buffers{fakeTrackBuffer(b);c.graphOncePinLocked(b)};c.graphOnce.capturing=false;return c,buffers
}
func beginFixture()(*cudaBackend,*cudaKV,Tensor){resetFixture();c:=&cudaBackend{faultLatch:&latch{}};return c,fixtureKV(c),fixtureInput(c)}
func captureFixture()(*cudaBackend,*cudaKV,Tensor){
 c,k,input:=beginFixture();require(c.GraphOnceBegin(input,k,0),"actual Begin refused healthy capacity")
 // Replace only the recording producer boundary: all captured layers have
 // appended one row in Go, but no native kernel ran in this host fixture.
 for _,rows:=range [3][]dslice{k.K,k.Kraw,k.V}{for i:=range rows{rows[i].len=k.stride()}}
 k.pos=append(k.pos,0)
 out:=&cudaBuf{be:c,ptr:unsafe.Pointer(new([4]float32)),n:16,class:MemoryActivation};fakeTrackBuffer(out);c.graphOncePinLocked(out)
 c.GraphOnceEnd();output:=Tensor{buf:out,Dtype:F32};c.GraphOnceSave(output)
 return c,k,output
}

`
const graphOnceRetirementGoMain = `
func main(){
 kvReservationControls()
 reservationControls()
 dependencyControls()
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
 c,b=setup();foreign:=&cudaBuf{};uploadCache[ucKey{1,F32,0}]=Tensor{buf:foreign,Dtype:F32};c.graphOnceReleaseLocked();require(uploadCache[ucKey{1,F32,0}].buf==foreign,"retiring stale input evicted foreign cache owner")
 c,b=setup();c.graphOnce.capturing=true;require(caught(c.graphOnceReleaseLocked)!=nil&&calls==0&&graphCalls==0&&b[0].graphOncePinned,"capture cleanup reached native")
 c,b=setup();cudaGraphOnceOwner=&cudaBackend{};require(caught(c.graphOnceReleaseLocked)!=nil&&calls==0&&graphCalls==0,"foreign owner retired")
 c,b=setup();require(graphOnceArenaBufferLegal(b[0],true)&&!graphOnceArenaBufferLegal(b[0],false),"F32 input role lost")
}

func dependencyControls(){
 mutations:=[]func(*cudaBuf){
  func(b *cudaBuf){b.ptr=unsafe.Pointer(new(int))},func(b *cudaBuf){b.n++},func(b *cudaBuf){b.device++},
  func(b *cudaBuf){b.scales=unsafe.Pointer(new(int));b.scalesN=4},func(b *cudaBuf){b.be=&cudaBackend{}},
  func(b *cudaBuf){b.bornGen++},func(b *cudaBuf){b.managed=!b.managed},func(b *cudaBuf){b.class=MemoryActivation},
  func(b *cudaBuf){b.host++},func(b *cudaBuf){b.hostDt++},func(b *cudaBuf){b.hostLo++},
  func(b *cudaBuf){b.budgetedWeightBytes++},func(b *cudaBuf){b.managedWeight=true},func(b *cudaBuf){b.graphOncePinned=!b.graphOncePinned},func(b *cudaBuf){b.invalid=true},
 }
 for _,borrowed:=range []bool{false,true}{for _,mutate:=range mutations{
  c,owned:=setup();s:=c.graphOnce;b:=owned[1]
  if borrowed {b=&cudaBuf{ptr:unsafe.Pointer(new(int)),n:16,class:MemoryWeights,hostKeep:new(int)};fakeTrackBuffer(b);s.capturing=true;c.graphOnceObserveLocked(b);s.capturing=false}
  original:=s.dependencies[b];mutate(b)
  require(!s.graphOnceDependenciesValid(),"allocation mutation accepted")
  require(caught(c.graphOnceRequireDependenciesLocked)!=nil&&stopped==-75092&&s.dependencyInvalidated&&s.invalidated&&calls==0&&graphCalls==0,"mutation did not stop before effects")
  require(s.dependencies[b].ptr==original.ptr&&s.dependencies[b].hostKeep==original.hostKeep&&c.graphOnce==s&&cudaGraphOnceOwner==c,"mutation lost original roots")
  require(caught(c.graphOnceReleaseLocked)!=nil&&calls==0&&graphCalls==0,"mutation retired a rebound pointer")
 }}
 c,b:=setup();s:=c.graphOnce;s.capturing=true
 borrowed:=&cudaBuf{ptr:unsafe.Pointer(new(int)),n:16,class:MemoryWeights,scales:unsafe.Pointer(new(int)),scalesN:4}
 fakeTrackBuffer(borrowed);c.graphOnceObserveLocked(borrowed);original:=s.dependencies[borrowed];borrowed.scalesN=8;c.graphOnceObserveLocked(borrowed);borrowed.scalesN=4;c.graphOnceObserveLocked(borrowed)
 require(s.dependencies[borrowed].scalesN==original.scalesN&&s.dependencyInvalidated&&!s.graphOnceDependenciesValid(),"re-observation laundered changed metadata")
 c,b=setup();s=c.graphOnce;original=s.dependencies[b[1]];s.capturing=true;b[1].n++;c.graphOnceObserveLocked(b[1]);b[1].n=original.n
 require(s.dependencyInvalidated&&!s.graphOnceDependenciesValid(),"owned re-observation missed change")
 c,b=setup();s=c.graphOnce;delete(s.dependencies,b[1]);require(!s.graphOnceDependenciesValid(),"missing receipt accepted")
 c,b=setup();s=c.graphOnce;s.buffers=append(s.buffers,b[1]);require(!s.graphOnceDependenciesValid(),"duplicate arena entry accepted")
 c,b=setup();s=c.graphOnce;s.input.buf=b[1];require(!s.graphOnceDependenciesValid(),"input rebound to another arena member")
 c,b=setup();s=c.graphOnce;s.buffers[1],s.buffers[2]=s.buffers[2],s.buffers[1];require(!s.graphOnceDependenciesValid(),"arena ordinal rebound accepted")
 c,b=setup();s=c.graphOnce;s.buffers[2]=b[1];require(!s.graphOnceDependenciesValid(),"same-count duplicate arena member accepted")
 c,b=setup();s=c.graphOnce;s.borrowed=map[*cudaBuf]struct{}{b[1]:{}};require(!s.graphOnceDependenciesValid(),"owned/borrowed role collision accepted")
 c,b=setup();s=c.graphOnce;root:=s.dependencies[b[1]].hostKeep;b[1].hostKeep=new(int);c.fenceGen++
 require(s.graphOnceDependenciesValid()&&s.dependencies[b[1]].hostKeep==root,"host root not retained or unrelated completion falsely invalidates receipt")
 c,b=setup();s=c.graphOnce;s.capturing=true
 borrowed=&cudaBuf{ptr:unsafe.Pointer(new(int)),n:16,class:MemoryWeights,scales:unsafe.Pointer(new(int)),scalesN:4,be:c,bornGen:7}
 fakeTrackBuffer(borrowed);c.graphOnceObserveLocked(borrowed);c.graphOnceObserveLocked(borrowed);s.capturing=false;c.fenceGen+=3
 require(s.graphOnceDependenciesValid(),"unchanged Q8 borrowed receipt rejected")
}

func addBorrowed(c *cudaBackend,scales bool)*cudaBuf{
 b:=&cudaBuf{ptr:unsafe.Pointer(new(int)),n:16,class:MemoryWeights,hostKeep:new(int)}
 if scales{b.scales=unsafe.Pointer(new(int));b.scalesN=4}
 fakeTrackBuffer(b);s:=c.graphOnce;s.capturing=true;c.graphOnceObserveLocked(b);s.capturing=false;return b
}
func reservationControls(){
 c,_:=setup();s:=c.graphOnce;for i:=0;i<64;i++{addBorrowed(c,i%2==0)}
 queries,batches:=ticketQueries,ticketBatches
 require(s.graphOnceDependenciesValid()&&ticketQueries==queries&&ticketBatches==batches+1,"replay validation must use one batch and no reacquisition")
 for _,scale:=range []bool{false,true}{
  c,_=setup();s=c.graphOnce;b:=addBorrowed(c,true);original:=s.dependencies[b]
  p,n:=b.ptr,uintptr(b.n);if scale{p,n=b.scales,uintptr(b.scalesN)}
  delete(fakeReservations,p);fakeTrack(p,n) // same pointer and full extent, new native owner
  require(!s.graphOnceDependenciesValid(),"same-address reservation ABA accepted")
  require(caught(c.graphOnceRequireDependenciesLocked)!=nil&&stopped==-75092&&calls==0&&graphCalls==0,"stale ticket did not stop before effects")
  require(s.dependencies[b].reservation==original.reservation&&s.dependencies[b].scaleReservation==original.scaleReservation&&s.dependencies[b].hostKeep==original.hostKeep,"stale native ticket silently refreshed")
 }
 c,owned:=setup();s=c.graphOnce;fakeTrack(owned[1].ptr,uintptr(owned[1].n))
 require(caught(c.graphOnceReleaseLocked)!=nil&&calls==0&&graphCalls==0&&owned[1].graphOncePinned,"owned reservation changed before retirement")
 c,_=setup();s=c.graphOnce;ticketSupported=false;b:=addBorrowed(c,true);ticketSupported=true
 require(s.dependencies[b].reservation==0&&s.dependencyInvalidated&&!s.graphOnceDependenciesValid(),"zero ticket later promoted after native support changed")
 c,_=setup();s=c.graphOnce;b=addBorrowed(c,true);fakeReservations[b.scales]=fakeReservation{bytes:8,id:s.dependencies[b].scaleReservation}
 require(!s.graphOnceDependenciesValid(),"native scale extent drift accepted")
 for mode:=0;mode<6;mode++{
  c,owned=setup();s=c.graphOnce;before:=ticketBatches
  switch mode{
  case 0:s.reservationIDs=s.reservationIDs[:len(s.reservationIDs)-1]
  case 1:s.reservationPointers[s.dependencies[owned[0]].reservationIndex]=unsafe.Pointer(new(int))
  case 2:s.reservationIDs[s.dependencies[owned[0]].reservationIndex]=0
  case 3:r:=s.dependencies[owned[1]];r.reservationIndex=int(^uint(0)>>1);s.dependencies[owned[1]]=r
  case 4:s.reservationBytes[s.dependencies[owned[0]].reservationIndex]++
  case 5:fakeTrack(owned[0].ptr,uintptr(owned[0].n));s.reservationIDs[s.dependencies[owned[0]].reservationIndex]=fakeReservations[owned[0].ptr].id
  }
  require(!s.graphOnceDependenciesValid()&&ticketBatches==before,"malformed ticket array crossed cgo or refreshed immutable receipt")
 }
 c,_=setup();s=c.graphOnce;ticketSupported=false
 require(!s.graphOnceDependenciesValid(),"default native profile accepted a saved nonzero ticket")
}
func kvReservationControls(){
 // Mathematical metadata boundary only: these fake registry extents do not
 // allocate a multi-gigabyte buffer or claim a native allocator observation.
 boundC,boundK,_:=beginFixture();limit:=int(^uint(0)>>1)/4;if limit>2147483647{limit=2147483647}
 for _,rows:=range [3][]dslice{boundK.K,boundK.Kraw,boundK.V}{for i:=range rows{rows[i].cap=limit;fakeTrack(rows[i].ptr,uintptr(limit*4))}}
 boundary:=&cudaGraphOnceState{kv:boundK};require(boundary.graphOnceBindKVLocked(boundC,0),"exact signed/host byte capacity boundary refused")
 slots,ok:=boundary.graphOnceKVSlotsValid();require(ok&&slots==6,"boundary slots not retained")
 boundK.K[0].cap=limit+1;queriesBefore:=ticketQueries
 tooLarge:=&cudaGraphOnceState{kv:boundK};require(!tooLarge.graphOnceBindKVLocked(boundC,0)&&ticketQueries==queriesBefore,"above-bound capacity queried native identity")
 boundC,boundK,_=beginFixture();boundK.maxPos=2147483646;boundK.cfg.NumKVHeads=2;boundK.cfg.HeadDim=1
 queriesBefore=ticketQueries;tooLarge=&cudaGraphOnceState{kv:boundK};require(!tooLarge.graphOnceBindKVLocked(boundC,0)&&ticketQueries==queriesBefore,"position-times-width bound applied after identity acquisition")
 c,k,output:=captureFixture();s:=c.graphOnce;ids:=append([]uint64(nil),s.kvTickets...);queries:=ticketQueries
 for pos:=1;pos<4;pos++{
  input:=fixtureInput(c);batches:=ticketBatches
  got,used:=c.GraphOnceReplay(input,k,pos)
  require(used&&got.buf==output.buf&&s.next==pos+1&&len(k.pos)==pos+1&&ticketQueries==queries&&ticketBatches==batches+1,"actual replay changed ticket identity or failed bookkeeping")
  for _,rows:=range [3][]dslice{k.K,k.Kraw,k.V}{for _,row:=range rows{require(row.len==(pos+1)*k.stride()&&row.cap==8,"replay changed capacity instead of logical rows")}}
  for i,id:=range ids{require(s.kvTickets[i]==id&&uint64(s.reservationIDs[i])==id,"append refreshed frozen KV ticket")}
 }
 require(prewarms==1&&creates==1&&begins==1&&ends==1&&replays==3&&cudaGraphOnceWitnessReplays==3&&c.fenceGen==4,"actual capture/replay boundary counters wrong")
 _,used:=c.GraphOnceReplay(fixtureInput(c),k,4)
 require(!used&&c.graphOnce==nil&&graphCalls==1&&calls==2,"full store did not retire graph before refusing further replay")
 for _,rows:=range [3][]dslice{k.K,k.Kraw,k.V}{for _,row:=range rows{_,live:=fakeReservations[row.ptr];require(live&&row.len==8,"arena retirement freed or rewrote borrowed KV")}}
 for plane:=0;plane<3;plane++{
  c,k,_=captureFixture();s=c.graphOnce;row:=[3]dslice{k.K[1],k.Kraw[1],k.V[1]}[plane];slot:=3+plane;id:=s.kvTickets[slot];oldNext,oldRows:=s.next,len(k.pos);fence:=c.fenceGen
  fakeTrack(row.ptr,uintptr(row.cap*4)) // same exact base/capacity, distinct native reservation
  require(caught(func(){c.GraphOnceReplay(fixtureInput(c),k,1)})!=nil,"KV ABA replay returned normally")
  require(copies==0&&replays==0&&calls==0&&graphCalls==0&&c.fenceGen==fence&&s.next==oldNext&&len(k.pos)==oldRows,"KV refusal occurred after effects")
  require(stopped==-75092&&s.kvTickets[slot]==id&&c.graphOnce==s&&s.kv==k&&cudaGraphOnceOwner==c,"KV refusal lost owner or refreshed identity")
 }
 for _,boundary:=range []string{"end","save","release"}{
  c,k,output=captureFixture();s=c.graphOnce;row:=k.V[1];fakeTrack(row.ptr,uintptr(row.cap*4));beforeEnd:=ends
  if boundary=="end" {s.capturing=true;c.capturing=true;require(caught(c.GraphOnceEnd)!=nil,"stale KV End returned normally")}else if boundary=="save" {require(caught(func(){c.GraphOnceSave(output)})!=nil,"stale KV Save returned normally")}else{require(caught(c.graphOnceReleaseLocked)!=nil,"stale KV release returned normally")}
  require(ends==beforeEnd&&calls==0&&graphCalls==0&&c.graphOnce==s&&s.kv==k,"stale KV boundary crossed native effect")
 }
 mutations:=[]func(*cudaKV){
  func(k *cudaKV){k.Kraw=k.Kraw[:1]},func(k *cudaKV){k.V=k.V[:1]},func(k *cudaKV){k.cfg.NumLayers=1},
  func(k *cudaKV){k.K[1].ptr=nil},func(k *cudaKV){k.Kraw[1].dtype=F32+1},func(k *cudaKV){k.V[1].cap=7},
  func(k *cudaKV){k.K[0].cap=int(^uint(0)>>1)/4+1},func(k *cudaKV){k.K[0].len=-1},
  func(k *cudaKV){k.cfg.NumKVHeads=int(^uint(0)>>1);k.cfg.HeadDim=2},func(k *cudaKV){k.cfg.Precision=1},
  func(k *cudaKV){k.Kraw[1]=k.K[0]},func(k *cudaKV){k.releaseRetained=true},func(k *cudaKV){k.snapshotCopy=&cudaKVSnapshotCopy{}},
 }
 for _,mutate:=range mutations{
  var input Tensor;c,k,input=beginFixture();mutate(k)
  require(!c.GraphOnceBegin(input,k,0)&&c.graphOnce==nil&&cudaGraphOnceOwner==nil&&prewarms==0&&creates==0&&begins==0,"invalid KV crossed graph creation/prewarm")
 }
 c,k,_=beginFixture();local:=&cudaGraphOnceState{kv:k};last:=k.V[1];delete(fakeReservations,last.ptr)
 require(!local.graphOnceBindKVLocked(c,0)&&!local.kvBound,"missing capacity ticket admitted")
 fakeTrack(last.ptr,uintptr(last.cap*4));beforeQueries:=ticketQueries
 require(!local.graphOnceBindKVLocked(c,0)&&ticketQueries==beforeQueries&&!local.kvBound,"failed observation refreshed KV tickets")
 c,k,_=captureFixture();s=c.graphOnce;k.cfg.NumKVHeads=2;k.cfg.HeadDim=1
 require(!s.graphOnceDependenciesValid(),"equal-stride geometry rebind accepted")
 c,k,_=captureFixture();s=c.graphOnce;k.Kraw=k.Kraw[:1]
 require(!s.graphOnceDependenciesValid(),"short plane indexed before inventory refusal")
 c,k,_=captureFixture();s=c.graphOnce;k.V[1].len=k.V[1].cap+1
 require(!s.graphOnceDependenciesValid(),"logical row beyond retained capacity admitted")
 c,k,_=captureFixture();s=c.graphOnce;row:=k.K[0];slot:=0;id:=s.kvTickets[slot];fakeReservations[row.ptr]=fakeReservation{bytes:uintptr(row.len*4),id:id}
 require(!s.graphOnceDependenciesValid(),"logical row byte count substituted for full allocation capacity")
}

`
