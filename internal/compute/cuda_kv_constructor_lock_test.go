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

var cudaKVConstructorLockHost = flag.Bool("fak-cuda-kv-constructor-lock-host", false, "run actual contiguous constructor bodies against host lock witnesses")

// fak-test:runtime fast est=10ms lane=default
func TestCUDAKVConstructorLockBoundary(t *testing.T) {
	raw, err := os.ReadFile("cuda_kv.go")
	if err != nil {
		t.Fatal(err)
	}
	set := token.NewFileSet()
	file, err := parser.ParseFile(set, "cuda_kv.go", raw, 0)
	if err != nil {
		t.Fatal(err)
	}
	bodies := map[string]string{}
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
		if !ok || owner.Name != "cudaBackend" {
			continue
		}
		bodies[fn.Name.Name] = string(raw[set.Position(fn.Body.Pos()).Offset:set.Position(fn.Body.End()).Offset])
	}
	wrapper := strings.Join(strings.Fields(bodies["newContiguousKV"]), " ")
	if wrapper != "{ cudaMu.Lock() defer cudaMu.Unlock() k, err := c.newContiguousKVLocked(cfg) if err != nil { panic(err) } return k }" {
		t.Fatal("contiguous constructor must lock exactly once around the locked implementation")
	}
	locked := bodies["newContiguousKVLocked"]
	if locked == "" || strings.Contains(locked, "cudaMu.Lock()") || strings.Contains(locked, "cudaMu.Unlock()") || strings.Contains(locked, ".Free()") {
		t.Fatal("locked constructor missing or recursively acquires/releases the mutex")
	}
	if !strings.Contains(bodies["NewKV"], "return c.newContiguousKV(cfg)") || !strings.Contains(bodies["NewKV"], "return c.newPagedKV(cfg)") {
		t.Fatal("public constructor dispatch changed")
	}
	restore := bodies["restoreContiguousKVFromHost"]
	lock := strings.Index(restore, "cudaMu.Lock()")
	unlock := strings.Index(restore, "defer cudaMu.Unlock()")
	call := strings.Index(restore, "k, err := c.newContiguousKVLocked(cloneKVConfig(state.Config))")
	if lock < 0 || unlock <= lock || call <= unlock || strings.Count(restore, "cudaMu.Lock()") != 1 || strings.Count(restore, "cudaMu.Unlock()") != 1 || strings.Contains(restore, "c.newContiguousKV(") {
		t.Fatal("restore must call the locked constructor under its existing mutex")
	}
}

// Actual constructors and preallocation rollback, with native operations and
// graph-fit selection mocked. The Locked case models the source-pinned restore
// call site; it does not execute the complete restore method or prove native
// ownership retirement, CUDA concurrency, or copy completion.
// fak-test:runtime slow est=120s lane=optin
func TestCUDAKVConstructorLockHostContract(t *testing.T) {
	if !*cudaKVConstructorLockHost {
		t.Skip("set -fak-cuda-kv-constructor-lock-host for the host lock witness")
	}
	wanted := map[string]bool{
		"*cudaBackend.NewKV": false, "*cudaBackend.newContiguousKV": false,
		"*cudaBackend.newContiguousKVLocked": false,
		"dslice":                             false, "cudaKV": false, "cudaKVSnapshotCopy": false,
		"*cudaKV.preallocateGraphWithOperations": false,
		"itoaC":                                  false, "releaseKVDeviceSlices": false,
	}
	var program strings.Builder
	program.WriteString(cudaKVConstructorLockFixture)
	for _, path := range []string{"cuda_kv.go", "backend_helpers.go"} {
		raw, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
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
				t.Fatalf("duplicate constructor declaration %s", key)
			}
			wanted[key] = true
			program.WriteString(prefix)
			program.Write(raw[set.Position(node.Pos()).Offset:set.Position(node.End()).Offset])
			program.WriteByte('\n')
		}
		for _, decl := range file.Decls {
			switch node := decl.(type) {
			case *ast.GenDecl:
				for _, spec := range node.Specs {
					if spec, ok := spec.(*ast.TypeSpec); ok {
						emit(spec.Name.Name, spec, "type ")
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
			t.Fatalf("missing constructor declaration %s", key)
		}
	}
	body := program.String()
	if strings.Count(body, "C.size_t") != 1 {
		t.Fatal("review constructor scalar type boundary")
	}
	body = strings.ReplaceAll(body, "C.size_t", "uint64") + cudaKVConstructorLockMain
	guard := "\tcudaMu.Lock()\n\tdefer cudaMu.Unlock()\n\tk, err := c.newContiguousKVLocked(cfg)"
	if strings.Count(body, guard) != 1 {
		t.Fatal("constructor lock negative-control boundary changed")
	}
	for _, variant := range []string{"candidate", "wrapper-lock-removed"} {
		t.Run(variant, func(t *testing.T) {
			source := body
			if variant != "candidate" {
				source = strings.Replace(source, guard, "\tk, err := c.newContiguousKVLocked(cfg)", 1)
			}
			dir := t.TempDir()
			goFile, binary := filepath.Join(dir, "main.go"), filepath.Join(dir, "probe")
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
				t.Fatalf("compile actual constructor: %v\n%s", err, output)
			}
			modes := []string{"public", "direct", "already-locked", "panic"}
			if variant != "candidate" {
				modes = []string{"public"}
			}
			for _, mode := range modes {
				run := exec.CommandContext(ctx, binary, mode)
				run.WaitDelay = time.Second
				output, err := run.CombinedOutput()
				if variant == "candidate" {
					if err != nil {
						t.Fatalf("constructor %s: %v\n%s", mode, err, output)
					}
				} else if err == nil || !strings.Contains(string(output), "native allocation reached without cudaMu") {
					t.Fatalf("removed-lock control must fail the lock assertion, not compile/timeout: %v\n%s", err, output)
				}
			}
		})
	}
}

const cudaKVConstructorLockFixture = `package main
import("fmt";"os";"runtime";"sync";"sync/atomic";"unsafe")
type Dtype int
const(F32 Dtype=iota;F16;BF16)
func(d Dtype)Bytes()int{if d==F32{return 4};return 2}
type KVConfig struct{NumLayers,NumKVHeads,HeadDim int;Precision string;GraphCapture bool}
const(KVPrecisionF32="f32";KVPrecisionF16="f16";KVPrecisionBF16="bf16";FreeUnknown int64=-1;GraphKVDoesNotFit="no-fit")
type KVStore interface{}
type cudaBackend struct{}
type cudaBuf struct{ptr unsafe.Pointer}
var cudaMu sync.Mutex
var graphEnabled=true
var cudaKVMaxPos=1
var graphKVFitWarned atomic.Bool
var allocations,unlockedAllocations,queries,releases,unlockedReleases,failAt int
var storage []*float32
var primary=new(int)
func require(ok bool,msg string){if !ok{fmt.Fprintln(os.Stderr,"FAIL:",msg);os.Exit(1)}}
func CUDAPagedKVEnabled()bool{return false}
func CUDAPagedKVEligible(KVConfig)bool{return false}
func(*cudaBackend)newPagedKV(KVConfig)KVStore{panic("paged dispatch outside fixture")}
// Admission is healthy in this lock-only witness; terminal behavior has its own witness.
func(*cudaKV)sharedStopErrorLocked(string)error{return nil}
func(*cudaBackend)dallocKV(int,string)*cudaBuf{allocations++;if cudaMu.TryLock(){unlockedAllocations++;cudaMu.Unlock()};if allocations==failAt{panic(primary)};v:=new(float32);storage=append(storage,v);return &cudaBuf{unsafe.Pointer(v)}}
func cudaKVElements(int){}
func uint64ToCapInt64(x uint64)int64{return int64(x)}
type preallocResult struct{Preallocate bool;Positions int;Bytes int64;Reason string}
func GraphKVPreallocFor(bool,KVConfig,int,int64)preallocResult{return preallocResult{Preallocate:true,Positions:1,Bytes:12}}
var C=struct{fcuda_mem_info func(*uint64,*uint64)int;fcuda_free func(unsafe.Pointer)}{func(f,t *uint64)int{queries++;*f=1024;*t=2048;return 0},func(unsafe.Pointer){releases++;if cudaMu.TryLock(){unlockedReleases++;cudaMu.Unlock()}}}
`

const cudaKVConstructorLockMain = `
func main(){
 require(len(os.Args)==2,"choose constructor mode")
 c:=&cudaBackend{};cfg:=KVConfig{NumLayers:1,NumKVHeads:1,HeadDim:1,Precision:KVPrecisionF32,GraphCapture:true}
 var recovered any
 switch os.Args[1]{
 case "public":c.NewKV(cfg)
 case "direct":c.newContiguousKV(cfg)
 case "already-locked":cudaMu.Lock();c.newContiguousKVLocked(cfg);cudaMu.Unlock()
 case "panic":failAt=2;func(){defer func(){recovered=recover()}();c.NewKV(cfg)}()
 default:require(false,"unknown case")
 }
 require(unlockedAllocations==0,"native allocation reached without cudaMu")
 require(unlockedReleases==0,"native rollback reached without cudaMu")
 require(queries==1,"graph preallocation query not reached")
 if failAt==0{require(allocations==3&&releases==0,"healthy constructor allocation contract changed")}else{require(recovered==primary&&allocations==2&&releases==1,"panic identity or rollback attempt changed")}
 require(cudaMu.TryLock(),"constructor left cudaMu locked");cudaMu.Unlock()
 runtime.KeepAlive(storage)
}
`
