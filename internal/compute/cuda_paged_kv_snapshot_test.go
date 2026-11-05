package compute

import (
	"context"
	"crypto/sha256"
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

// Default-lane bindings complement, but do not replace, the executed host test.
// fak-test:runtime fast est=10ms lane=default
func TestCUDAPagedKVSnapshotSourceContract(t *testing.T) {
	read := func(path string) string {
		b, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		return string(b)
	}
	source := read("cuda_paged_kv_snapshot.go")
	for _, want := range []string{"k.snapshotCopy = holder", "holder.tablePin.Pin(&k.table[0])", "holder.hostPin.Pin(&rows[0])", "C.fcuda_paged_kv_snapshot_gather_checked", "C.fcuda_kv_snapshot_d2h_checked", "k.pool.retainCopyOwnerLocked(k)"} {
		if !strings.Contains(source, want) {
			t.Fatalf("missing ownership boundary %s", want)
		}
	}
	if strings.Contains(source, "recover()") || strings.Contains(source, "C.fcuda_d2h(") {
		t.Fatal("unwind or checked copy regressed")
	}
	if strings.Index(source, "defer func()") > strings.Index(source, "holder.scratch = k.pool.be.dallocKV") {
		t.Fatal("ownership defer registered after allocation")
	}
	pool := strings.Join(strings.Fields(read("cuda_paged_kv.go")), " ")
	if !strings.Contains(pool, "k.snapshotCopy == nil && len(k.table) == 0") {
		t.Fatal("snapshot-only owner excluded from retention")
	}
}

// Actual snapshot native adapters and header, with explicit fake CUDA edges.
// fak-test:runtime slow est=5s lane=optin
func TestCUDAPagedKVSnapshotNativeContract(t *testing.T) {
	if !*cudaPagedSnapshotHost {
		t.Skip("set -fak-cuda-paged-kv-snapshot-host")
	}
	compiler, err := exec.LookPath("g++")
	if err != nil {
		t.Skip("g++ unavailable")
	}
	b, err := os.ReadFile("cuda/kv_precision.cuh")
	if err != nil {
		t.Fatal(err)
	}
	s := string(b)
	start, end := strings.Index(s, "static int kv_snapshot_status"), strings.Index(s, "__global__ void k_kv_attention_typed")
	if start < 0 || end <= start {
		t.Fatal("snapshot adapter boundaries missing")
	}
	header, err := os.ReadFile("cuda_kv_precision.h")
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "cuda_kv_precision.h"), header, 0600); err != nil {
		t.Fatal(err)
	}
	program := pagedSnapshotNativePrefix + s[start:end] + pagedSnapshotNativeSuffix
	if err := os.WriteFile(filepath.Join(dir, "probe.cpp"), []byte(program), 0600); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	binary := filepath.Join(dir, "probe")
	if out, err := exec.CommandContext(ctx, compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", filepath.Join(dir, "probe.cpp"), "-o", binary).CombinedOutput(); err != nil {
		t.Fatalf("%v\n%s", err, out)
	}
	if out, err := exec.CommandContext(ctx, binary).CombinedOutput(); err != nil {
		t.Fatalf("%v\n%s", err, out)
	} else {
		t.Logf("%s", out)
	}
}

const pagedSnapshotNativePrefix = `
#include <cassert>
#include <cstdio>
#include "cuda_kv_precision.h"
static int g_cuda_cleanup_status=0,gatherStatus=0,copyStatus=0,loadStatus=0,gathers=0,copies=0,loads=0;
static const int cudaErrorNotReady=34;
static bool uncertain=false;
static bool v41_candidates_stream_uncertain(){return uncertain||g_cuda_cleanup_status!=0;}
extern "C" int fcuda_paged_kv_gather_f32(float*,const float*,const int*,int,int,int,int,int){gathers++;return gatherStatus;}
extern "C" int fcuda_d2h_checked(void*,const void*,size_t){copies++;return copyStatus;}
extern "C" int fcuda_kv_load_typed(float*,const void*,int,int){loads++;return loadStatus;}
`
const pagedSnapshotNativeSuffix = `
int main(){
assert(fcuda_paged_kv_snapshot_gather_checked(nullptr,nullptr,nullptr,1,1,32,16,1)==0&&gathers==1);
gatherStatus=4;assert(fcuda_paged_kv_snapshot_gather_checked(nullptr,nullptr,nullptr,1,1,32,16,1)==4&&g_cuda_cleanup_status==4);
assert(fcuda_kv_snapshot_d2h_checked(nullptr,nullptr,4)==4&&copies==0);
fcuda_kv_snapshot_failed(1057);assert(g_cuda_cleanup_status==4);
g_cuda_cleanup_status=0;copyStatus=1057;assert(fcuda_kv_snapshot_d2h_checked(nullptr,nullptr,4)==1057&&g_cuda_cleanup_status==1057&&copies==1);
g_cuda_cleanup_status=0;fcuda_kv_snapshot_failed(0);assert(g_cuda_cleanup_status==-75707);
g_cuda_cleanup_status=0;uncertain=true;int old=gathers;assert(fcuda_paged_kv_snapshot_gather_checked(nullptr,nullptr,nullptr,1,1,32,16,1)==34&&gathers==old&&g_cuda_cleanup_status==34);
puts("7 snapshot native protocol controls PASS; fake CUDA only");}
`

// Host-only protocol witness; this flag never requests a CUDA device run.
var cudaPagedSnapshotHost = flag.Bool("fak-cuda-paged-kv-snapshot-host", false, "run the exact-source host ownership/orchestration witness")

// Executes intact production pool/owner methods and the real page refcounter.
// CUDA status, backend admission, allocation and transfers are explicit host
// boundaries. It is not native compilation, fault injection, completion or
// numerical/performance evidence. Child compilation is opt-in and deadline-bound.
// fak-test:runtime slow est=60s lane=optin
func TestCUDAPagedKVSnapshotHostContract(t *testing.T) {
	if !*cudaPagedSnapshotHost {
		t.Skip("set -fak-cuda-paged-kv-snapshot-host for the CPU contract witness")
	}
	var program strings.Builder
	program.WriteString(cudaPagedSnapshotHostFixture)
	wanted := map[string]bool{
		"cudaPagedKVSnapshotCopy":     false,
		"*cudaPagedKV.SnapshotToHost": false, "*cudaPagedKV.snapshotToHostLocked": false, "*cudaPagedKV.snapshotFailureLocked": false,
		"*cudaPagedKV.syncDeviceTableLocked": false, "KVHostSnapshot": false, "KVHostSnapshot.Validate": false,
		"CUDAOpError": false, "*CUDAOpError.Error": false,
		"cudaKVPagePool": false, "cudaPagedPoolKey": false, "cudaPagedKV": false,
		"cudaPagedKVCopyFailure": false, "pagedKVBlocks": false,
		"*cudaBackend.pagedPool": false, "*cudaBackend.newPagedKV": false,
		"*cudaBackend.restorePagedKVFromHost":   false,
		"*cudaPagedKVCopyFailure.Error":         false,
		"*cudaKVPagePool.retainCopyOwnerLocked": false,
		"*cudaKVPagePool.stopCopyLocked":        false, "*cudaKVPagePool.copyErrorLocked": false,
		"*cudaKVPagePool.allocLocked": false, "*cudaKVPagePool.growLocked": false,
		"*cudaKVPagePool.copyBlockLocked":       false,
		"*cudaPagedKV.requireCopyHealthyLocked": false, "*cudaPagedKV.writablePageLocked": false,
		"*cudaPagedKV.Clone": false, "*cudaPagedKV.Free": false, "*cudaPagedKV.releaseLocked": false,
		"*pagedKVBlocks.grow": false, "*pagedKVBlocks.alloc": false, "*pagedKVBlocks.check": false,
		"*pagedKVBlocks.retain": false, "*pagedKVBlocks.release": false, "*pagedKVBlocks.shared": false,
		"pagedKVGrowBlocks": false,
	}
	for _, path := range []string{"cuda_paged_kv.go", "cuda_paged_kv_blocks.go", "cuda_paged_kv_snapshot.go", "kvhost.go", "cuda_errors.go"} {
		source, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		fs := token.NewFileSet()
		file, err := parser.ParseFile(fs, path, source, 0)
		if err != nil {
			t.Fatal(err)
		}
		emit := func(key string, node ast.Node, prefix string) {
			seen, ok := wanted[key]
			if !ok {
				return
			}
			if seen {
				t.Fatalf("duplicate real declaration %s", key)
			}
			wanted[key] = true
			body := source[fs.Position(node.Pos()).Offset:fs.Position(node.End()).Offset]
			t.Logf("%s/%s SHA256=%x", path, key, sha256.Sum256(body))
			program.WriteString(prefix)
			program.Write(body)
			program.WriteByte('\n')
		}
		for _, decl := range file.Decls {
			switch node := decl.(type) {
			case *ast.GenDecl:
				if node.Tok == token.TYPE {
					for _, spec := range node.Specs {
						if typ, ok := spec.(*ast.TypeSpec); ok {
							emit(typ.Name.Name, typ, "type ")
						}
					}
				}
			case *ast.FuncDecl:
				key := node.Name.Name
				if node.Recv != nil {
					r := node.Recv.List[0].Type
					prefix := ""
					if ptr, ok := r.(*ast.StarExpr); ok {
						prefix, r = "*", ptr.X
					}
					id, ok := r.(*ast.Ident)
					if !ok {
						continue
					}
					key = prefix + id.Name + "." + key
				}
				emit(key, node, "")
			}
		}
	}
	for key, seen := range wanted {
		if !seen {
			t.Fatalf("missing real declaration %s", key)
		}
	}
	dir := t.TempDir()
	probe := filepath.Join(dir, "paged_copy_test.go")
	if err := os.WriteFile(probe, []byte(strings.ReplaceAll(strings.ReplaceAll(program.String(), "C.float", "float32"), "(*C.int)", "(*int32)")), 0600); err != nil {
		t.Fatal(err)
	}
	goExe := filepath.Join(runtime.GOROOT(), "bin", "go")
	if runtime.GOOS == "windows" {
		goExe += ".exe"
	}
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, goExe, "test", "-p=1", "-trimpath", "-buildvcs=false", "-count=1", "-v", "-timeout=45s", "-run", "^TestPagedSnapshotHost$", probe)
	cmd.Dir = dir
	for _, entry := range os.Environ() {
		key, _, _ := strings.Cut(entry, "=")
		switch key {
		case "GOWORK", "GOENV", "GOTOOLCHAIN", "CGO_ENABLED", "GOMAXPROCS", "GOFLAGS", "GODEBUG":
			continue
		}
		cmd.Env = append(cmd.Env, entry)
	}
	// Force the legacy nil-panic semantics in this isolated host child so its
	// propagation control does not depend on the invoking shell's GODEBUG.
	cmd.Env = append(cmd.Env, "GOWORK=off", "GOENV=off", "GOTOOLCHAIN=local", "CGO_ENABLED=0", "GOMAXPROCS=1", "GOFLAGS=", "GODEBUG=panicnil=1")
	output, err := cmd.CombinedOutput()
	t.Logf("exact-source host witness:\n%s", output)
	if err != nil {
		t.Fatal(err)
	}
}

const cudaPagedSnapshotHostFixture = `package compute

import (
	"fmt"
	"reflect"
	"runtime"
	"weak"
	"errors"
	"strings"
	"strconv"
	"sync"
	"sync/atomic"
	"testing"
	"unsafe"
)

const CUDAPagedKVBlockTokens = 16

type KVConfig struct{ NumLayers, NumKVHeads, HeadDim int }
type MemoryClass string
const MemoryUnknown MemoryClass = ""
const DeviceFaultExecution = 1
type snapshotFaultLatch struct{code int}
func(l *snapshotFaultLatch)Observe(_ int,_ string,code int){if l!=nil{l.code=code}}
func H2DCensusRecord(string,int64){}


func (k *cudaPagedKV) loadFromHostLocked(KVHostSnapshot) {
	panic("unexpected healthy host restoration in failure-only fixture")
}

type KVStore interface {
	Clone() KVStore
	Free()
}
type cudaBackend struct{ fault bool; faultLatch *snapshotFaultLatch }
type cudaBuf struct{ ptr unsafe.Pointer }

func cloneKVConfig(c KVConfig) KVConfig { return c }
func itoaC(v int) string                { return strconv.Itoa(v) }
func(c *cudaBackend)dallocKV(n int,_ string)*cudaBuf{p:=&cudaBuf{ptr:unsafe.Pointer(&make([]byte,n)[0])};snapshotAllocations=append(snapshotAllocations,p);return p}
func AdmitDevice(c *cudaBackend, _ string) error {
	if c.fault {
		return fmt.Errorf("host device-fault boundary")
	}
	return nil
}

var cudaMu sync.Mutex
var cudaPagedPoolsMu sync.Mutex
var cudaPagedPools = map[cudaPagedPoolKey]*cudaKVPagePool{}
var cudaPagedKVCopyQualification bool
var copyStatus int32
var terminalStatus int32
var copyCalls, d2dCalls, freeCalls, terminalQueries int
var expectedPool *cudaKVPagePool
var C = struct {
	int                            func(any) int32
	size_t                         func(int) uintptr
	fcuda_paged_kv_copy_terminal   func() int32
	fcuda_paged_kv_copy_blocks_f32 func(unsafe.Pointer, int32, int32, int32, int32, int32) int32
fcuda_kv_snapshot_failed func(int32)
 fcuda_paged_kv_snapshot_gather_checked func(*float32,*float32,*int32,int32,int32,int32,int32,int32)int32
 fcuda_kv_snapshot_d2h_checked func(unsafe.Pointer,unsafe.Pointer,uintptr)int32
 fcuda_h2d func(unsafe.Pointer,unsafe.Pointer,uintptr)
 fcuda_paged_store_int func(*int32,int32)int32
 fcuda_d2d func(unsafe.Pointer, unsafe.Pointer, uintptr)
	fcuda_free                     func(unsafe.Pointer)
	fcuda_discard_unsubmitted      func(unsafe.Pointer, uintptr) int32
}{
	int: func(v any) int32 {
		switch x := v.(type) {
		case int:
			return int32(x)
		case int32:
			return x
		default:
			panic("unexpected C integer argument")
		}
	},
	size_t:                       func(v int) uintptr { return uintptr(v) },
	fcuda_paged_kv_copy_terminal: func() int32 { terminalQueries++; return terminalStatus },
	fcuda_paged_kv_copy_blocks_f32: func(raw unsafe.Pointer, planes, capacity, floats, dst, src int32) int32 {
		p := expectedPool
		if planes != int32(3*p.layers) || capacity != int32(p.capBlocks) ||
			floats != int32(p.blockFloats) || src == dst || src < 0 || dst < 0 ||
			int(src) >= p.capBlocks || int(dst) >= p.capBlocks {
			panic("incorrect actual copy arguments")
		}
		got := unsafe.Slice((*unsafe.Pointer)(raw), int(planes))
		want := append(append(append([]unsafe.Pointer(nil), p.K...), p.Kraw...), p.V...)
		if !reflect.DeepEqual(got, want) {
			panic("missing/reordered plane arena")
		}
		copyCalls++
		return copyStatus
	},
fcuda_kv_snapshot_failed:func(status int32){if terminalStatus==0{terminalStatus=status}},
 fcuda_paged_kv_snapshot_gather_checked:func(dst,src *float32,table *int32,n,heads,hd,block,key int32)int32{
 snapshotGathers++;if snapshotGathers==snapshotFailGather{terminalStatus=4;return 4};for i:=range unsafe.Slice(dst,int(n*heads*hd)){unsafe.Slice(dst,int(n*heads*hd))[i]=float32(i+snapshotGathers)};return 0},
 fcuda_kv_snapshot_d2h_checked:func(dst,src unsafe.Pointer,n uintptr)int32{
 snapshotCopies++;if snapshotCopies==snapshotFailCopy{terminalStatus=1057;return 1057};copy(unsafe.Slice((*byte)(dst),int(n)),unsafe.Slice((*byte)(src),int(n)));return 0},
 fcuda_h2d:func(unsafe.Pointer,unsafe.Pointer,uintptr){switch snapshotPrepFailure{case "panic":panic("table prep");case "panicnil":panic(nil);case "goexit":runtime.Goexit()}},
 fcuda_paged_store_int:func(*int32,int32)int32{return 0},
 fcuda_d2d: func(unsafe.Pointer, unsafe.Pointer, uintptr) { d2dCalls++ },
	fcuda_free:                func(unsafe.Pointer) { freeCalls++ },
	fcuda_discard_unsubmitted: func(unsafe.Pointer, uintptr) int32 { panic("unexpected arena retirement") },
}

func setup(layers int, enabled bool) (*cudaPagedKV, *cudaPagedKV) {
	cudaPagedKVCopyQualification, copyStatus, terminalStatus = enabled, 0, 0
	copyCalls, d2dCalls, freeCalls, terminalQueries = 0, 0, 0, 0
	cudaPagedPools = map[cudaPagedPoolKey]*cudaKVPagePool{}
	c := &cudaBackend{}
	cfg := KVConfig{NumLayers: layers, NumKVHeads: 1, HeadDim: 32}
	a := c.newPagedKV(cfg)
	p := a.pool
	p.capBlocks = 4
	p.blocks.grow(4)
	// Distinct host-owned storage is only a pointer fixture: no device load/store
	// is executed. Real allocator growth remains in the extracted source.
	for _, plane := range [][]unsafe.Pointer{p.K, p.Kraw, p.V} {
		for i := range plane {
			plane[i] = unsafe.Pointer(new([8192]byte))
		}
	}
	a.table = []int32{p.allocLocked()}
	a.lens[0], a.pos = 7, []int{0, 1, 2, 3, 4, 5, 6}
	a.dirty = false
	for _, ptr := range []*unsafe.Pointer{&a.dTable, &a.dSeqLen, &a.scratch, &a.gK, &a.gV} {
		*ptr = unsafe.Pointer(new(int))
	}
	a.dTableCap, a.scratchF, a.gKF, a.gVF, a.seqLenDev = 4, 19, 13, 17, 7
	b := a.Clone().(*cudaPagedKV)
	expectedPool = p
	return a, b
}

func terminal(t *testing.T, f func()) {
	t.Helper()
	defer func() {
		r := recover()
		if _, ok := r.(*cudaPagedKVCopyFailure); !ok {
			t.Fatalf("want typed terminal refusal, got %T %v", r, r)
		}
	}()
	f()
}

func stable(k *cudaPagedKV) string {
	p := k.pool
	return fmt.Sprintf("%v/%v/%v/%v/%v/%d/%d/%d/%p/%p/%p/%p/%p/%d/%d/%d/%d/%d/%v",
		k.table, k.lens, k.pos, p.blocks.refs, p.blocks.free, p.blocks.live, p.capBlocks, p.copySubmissions,
		k.dTable, k.dSeqLen, k.scratch, k.gK, k.gV,
		k.dTableCap, k.scratchF, k.gKF, k.gVF, k.seqLenDev, k.dirty)
}

func retained(t *testing.T, p *cudaKVPagePool, want ...*cudaPagedKV) {
	t.Helper()
	seen := map[*cudaPagedKV]bool{}
	for k := p.copyOwners; k != nil; k = k.copyNext {
		if seen[k] {
			t.Fatal("duplicate/cyclic retained owner")
		}
		seen[k] = true
		if !k.copyRetained {
			t.Fatal("retained owner missing marker")
		}
	}
	if len(seen) != len(want) {
		t.Fatalf("retained %d owners, want %d", len(seen), len(want))
	}
	for _, k := range want {
		if !seen[k] {
			t.Fatal("lost store/private-buffer owner")
		}
	}
}


var snapshotAllocations []*cudaBuf
var snapshotGathers,snapshotCopies,snapshotFailGather,snapshotFailCopy int
var snapshotPrepFailure string
var snapshotRetainedPools []*cudaKVPagePool
func snapshotSetup()(*cudaPagedKV,*cudaPagedKV){
 if expectedPool!=nil&&expectedPool.copyOwners!=nil{snapshotRetainedPools=append(snapshotRetainedPools,expectedPool)}
 snapshotAllocations=nil;snapshotGathers,snapshotCopies,snapshotFailGather,snapshotFailCopy=0,0,0,0;snapshotPrepFailure=""
 a,b:=setup(2,false);a.lens[1]=7;b.lens[1]=7;a.pool.be.faultLatch=&snapshotFaultLatch{};return a,b
}
func TestPagedSnapshotHost(t *testing.T){
 for _,gather:=range []bool{false,true}{for _,at:=range []int{1,3,6}{t.Run(fmt.Sprintf("gather=%v/at=%d",gather,at),func(t *testing.T){
  a,b:=snapshotSetup();before:=stable(a)
  status,site:=1057,"paged-kv-snapshot-d2h";if gather{snapshotFailGather=at;status=4;site="paged-kv-snapshot-gather"}else{snapshotFailCopy=at}
  out,err:=a.SnapshotToHost();var typed *CUDAOpError
  if !errors.As(err,&typed)||typed.Code!=status||typed.Site!=site||len(out.K)!=0||terminalStatus!=int32(status)||a.pool.copyFailure.Load()!=int32(status)||a.pool.be.faultLatch.code!=status{t.Fatalf("lost error/publication: %v",err)}
  wantCopies:=at;if gather{wantCopies--};if snapshotGathers!=at||snapshotCopies!=wantCopies||freeCalls!=0||stable(a)!=before{t.Fatal("continued, freed scratch or changed source")}
  h:=a.snapshotCopy;if h==nil||h.scratch!=snapshotAllocations[0]||len(h.host)!=224||!a.copyRetained{t.Fatal("exact snapshot owner missing")}
  terminal(t,func(){a.Free()});terminal(t,func(){b.Free()});terminal(t,func(){cudaMu.Lock();defer cudaMu.Unlock();a.pool.allocLocked()})
  if stable(a)!=before||freeCalls!=0||a.snapshotCopy!=h{t.Fatal("Free/reallocation changed retained source")}
  wk,wh:=weak.Make(a),weak.Make(h);p:=a.pool;a,h=nil,nil;runtime.GC()
  if wk.Value()==nil||wh.Value()==nil||wk.Value().snapshotCopy!=wh.Value()||p.copyOwners==nil{t.Fatal("detached source/holder not strongly retained")}
 })}}
 for _,mode:=range []string{"panic","panicnil","goexit"}{t.Run("table-"+mode,func(t *testing.T){
  a,b:=snapshotSetup();a.dirty=true;snapshotPrepFailure=mode;done:=make(chan struct{});returned:=false
  go func(){defer close(done);defer func(){_ = recover()}();a.SnapshotToHost();returned=true}();<-done
  if returned||snapshotGathers!=0||snapshotCopies!=0||terminalStatus!=-75707||a.pool.copyFailure.Load()!=-75707||a.snapshotCopy==nil||a.snapshotCopy.scratch!=snapshotAllocations[0]||!a.copyRetained||freeCalls!=0{t.Fatal("table unwind lost owner/terminal")}
  terminal(t,func(){a.Free()});terminal(t,func(){b.Free()});runtime.GC()
 })}
 t.Run("holder-only",func(t *testing.T){a,_:=snapshotSetup();k:=&cudaPagedKV{pool:a.pool,snapshotCopy:&cudaPagedKVSnapshotCopy{}};cudaMu.Lock();a.pool.retainCopyOwnerLocked(k);cudaMu.Unlock();if !k.copyRetained||a.pool.copyOwners!=k{t.Fatal("holder-only source not rooted")}})
 t.Run("healthy",func(t *testing.T){a,_:=snapshotSetup();before:=stable(a);out,err:=a.SnapshotToHost();if err!=nil||out.Validate()!=nil||snapshotGathers!=6||snapshotCopies!=6||freeCalls!=1||a.snapshotCopy!=nil||a.copyRetained||stable(a)!=before{t.Fatalf("healthy changed: %v",err)};runtime.GC()})
 t.Run("empty",func(t *testing.T){a,_:=snapshotSetup();a.pos=nil;a.lens=[]int{0,0};out,err:=a.SnapshotToHost();if err!=nil||out.Validate()!=nil||snapshotGathers!=0||snapshotCopies!=0||len(snapshotAllocations)!=0||a.snapshotCopy!=nil{t.Fatalf("empty changed: %v",err)}})
 for _,invalid:=range []string{"heads","width","tokens"}{t.Run("invalid-"+invalid,func(t *testing.T){a,_:=snapshotSetup();switch invalid{case "heads":a.cfg.NumKVHeads=int(^uint(0)>>1);case "width":a.cfg.HeadDim=0;case "tokens":n:=int64(2147483648);a.lens[0]=int(n)};out,err:=a.SnapshotToHost();if err==nil||len(out.K)!=0||len(snapshotAllocations)!=0||snapshotGathers!=0||snapshotCopies!=0||a.snapshotCopy!=nil{t.Fatal("invalid geometry reached effects")}})}
}
`
