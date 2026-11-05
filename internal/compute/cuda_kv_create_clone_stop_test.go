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

var cudaKVCreateCloneHost = flag.Bool("fak-cuda-kv-create-clone-host", false, "run actual contiguous KV construction and Clone against host terminal-entry witnesses")

// The constructor keeps its precision/extent validation before admission. The
// check precedes device allocation and every graph/native action. Clone admits
// under its existing mutex before publishing or copying an unpublished owner.
// fak-test:runtime fast est=10ms lane=default
func TestCUDAKVCreateCloneEntryGuards(t *testing.T) {
	raw, err := os.ReadFile("cuda_kv.go")
	if err != nil {
		t.Fatal(err)
	}
	set := token.NewFileSet()
	file, err := parser.ParseFile(set, "cuda_kv.go", raw, 0)
	if err != nil {
		t.Fatal(err)
	}
	wanted := map[string][]string{
		"*cudaKV.Clone": {
			"cudaMu.Lock()", "defer cudaMu.Unlock()",
			"if err := k.sharedStopErrorLocked(\"kv-clone\"); err != nil { panic(err) }",
		},
		"*cudaBackend.newContiguousKVLocked": {
			"dtype := F32",
			"switch cfg.Precision { case KVPrecisionF32: case KVPrecisionF16: dtype = F16 case KVPrecisionBF16: dtype = BF16 default: panic(fmt.Sprintf(\"cuda: unrealized KV precision %s\", cfg.Precision)) }",
			"if dtype != F32 && (cfg.NumKVHeads <= 0 || cfg.HeadDim <= 0 || cfg.NumKVHeads > 2147483647/cfg.HeadDim) { panic(\"cuda: narrowed KV row exceeds native element bound\") }",
			"k := &cudaKV{be: c, cfg: cfg}",
			"k.K = make([]dslice, cfg.NumLayers)",
			"k.Kraw = make([]dslice, cfg.NumLayers)",
			"k.V = make([]dslice, cfg.NumLayers)",
			"for l := range k.K { k.K[l].dtype = dtype k.V[l].dtype = dtype }",
			"if err := k.sharedStopErrorLocked(\"kv-create\"); err != nil { return nil, err }",
			"free := FreeUnknown",
		},
		"*cudaBackend.restoreContiguousKVFromHost": {
			"if err := state.Validate(); err != nil { return nil, err }",
			"cudaMu.Lock()", "defer cudaMu.Unlock()",
			"k, err := c.newContiguousKVLocked(cloneKVConfig(state.Config))",
			"if err != nil { return nil, err }",
		},
	}
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
		if !ok {
			continue
		}
		key := "*" + owner.Name + "." + fn.Name.Name
		prefix, ok := wanted[key]
		if !ok {
			continue
		}
		if len(fn.Body.List) < len(prefix) {
			t.Fatalf("missing admission prefix: %s", key)
		}
		for i, want := range prefix {
			stmt := fn.Body.List[i]
			got := strings.Join(strings.Fields(string(raw[set.Position(stmt.Pos()).Offset:set.Position(stmt.End()).Offset])), " ")
			if got != want {
				t.Fatalf("%s statement %d = %q, want %q", key, i, got, want)
			}
		}
		delete(wanted, key)
	}
	if len(wanted) != 0 {
		t.Fatalf("missing methods: %v", wanted)
	}
}

// Execute the actual entry points, helper and ownership-copy bodies against
// fake native boundaries. These witnesses certify neither native retirement
// nor asynchronous copy completion. Restore exercises its actual entry and
// healthy F32 host boundary; narrowed conversion and failed rollback are not
// qualified by these admission checks.
// fak-test:runtime slow est=240s lane=optin
func TestCUDAKVCreateCloneHostContract(t *testing.T) {
	if !*cudaKVCreateCloneHost {
		t.Skip("set -fak-cuda-kv-create-clone-host for terminal-entry witnesses")
	}
	wanted := map[string]bool{
		"*cudaBackend.NewKV": false, "*cudaBackend.newContiguousKV": false,
		"*cudaBackend.newContiguousKVLocked":     false,
		"*cudaKV.preallocateGraphWithOperations": false, "*cudaKV.Clone": false,
		"*cudaKV.cloneWithOperations": false, "*cudaKV.sharedStopErrorLocked": false,
		"dslice": false, "cudaKV": false, "cudaKVSnapshotCopy": false,
		"itoaC": false, "releaseKVDeviceSlices": false,
		"cudaQ56KMMVQReleaseBlocked": false, "BackendError": false, "*BackendError.Error": false,
		"*cudaBackend.RestoreKVFromHost": false, "*cudaBackend.restoreContiguousKVFromHost": false,
		"KVHostSnapshot": false, "KVHostSnapshot.Validate": false,
		"restoreCUDAKVAllocation": false, "cudaKVCheck": false,
	}
	var program strings.Builder
	program.WriteString(cudaKVCreateCloneFixture)
	for _, path := range []string{"cuda_kv.go", "cuda.go", "backend_helpers.go", "vulkan_errors.go", "cuda_kv_restore.go", "kvhost.go"} {
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
				t.Fatalf("duplicate declaration %s", key)
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
					switch recv := node.Recv.List[0].Type.(type) {
					case *ast.StarExpr:
						owner, ok := recv.X.(*ast.Ident)
						if !ok {
							continue
						}
						key = "*" + owner.Name + "." + key
					case *ast.Ident:
						key = recv.Name + "." + key
					default:
						continue
					}
				}
				emit(key, node, "")
			}
		}
	}
	for key, seen := range wanted {
		if !seen {
			t.Fatalf("missing actual declaration %s", key)
		}
	}
	body := program.String()
	if strings.Count(body, "C.size_t") != 4 || strings.Count(body, "C.int") != 3 || strings.Count(body, "C.float") != 1 {
		t.Fatal("review cgo scalar boundaries")
	}
	body = strings.NewReplacer("C.size_t", "uint64", "C.int", "int", "C.float", "float32").Replace(body) + cudaKVCreateCloneMain
	for _, variant := range []string{"candidate", "create-guard-removed", "clone-guard-removed", "restore-error-ignored"} {
		t.Run(variant, func(t *testing.T) {
			source := body
			mode := "all"
			if variant != "candidate" {
				mode = strings.TrimSuffix(variant, "-guard-removed")
				action := "panic(err)"
				if mode == "create" {
					action = "return nil, err"
				}
				guard := "\tif err := k.sharedStopErrorLocked(\"kv-" + mode + "\"); err != nil {\n\t\t" + action + "\n\t}\n"
				if variant == "restore-error-ignored" {
					mode = "restore"
					guard = "\tif err != nil {\n\t\treturn nil, err\n\t}\n"
				}
				if strings.Count(source, guard) != 1 {
					t.Fatal("negative-control admission boundary changed")
				}
				source = strings.Replace(source, guard, "", 1)
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
				t.Fatalf("compile actual KV entry points: %v\n%s", err, output)
			}
			run := exec.CommandContext(ctx, binary, mode)
			run.WaitDelay = time.Second
			output, err := run.CombinedOutput()
			if variant == "candidate" {
				if err != nil {
					t.Fatalf("entry contract: %v\n%s", err, output)
				}
			} else if err == nil || !strings.Contains(string(output), "FAIL: "+mode+" terminal admission") {
				t.Fatalf("removed-guard control must fail its admission assertion: %v\n%s", err, output)
			}
		})
	}
}

const cudaKVCreateCloneFixture = `package main
import("errors";"fmt";"os";"reflect";"runtime";"strings";"sync";"sync/atomic";"unsafe")
type Dtype int
const(F32 Dtype=iota;F16;BF16)
func(d Dtype)Bytes()int{if d==F32{return 4};return 2}
type KVConfig struct{NumLayers,NumKVHeads,HeadDim int;Precision string;GraphCapture bool}
const(KVPrecisionF32="f32";KVPrecisionF16="f16";KVPrecisionBF16="bf16";FreeUnknown int64=-1;GraphKVDoesNotFit="no-fit")
type KVStore interface{}
type cudaBackend struct{}
type cudaBuf struct{ptr unsafe.Pointer}
type VulkanErrorClass string
var cudaMu sync.Mutex
var graphEnabled=true
var cudaKVMaxPos=2
var graphKVFitWarned atomic.Bool
var sharedStop bool
var allocations,copies,releases,queries,memQueries,uploads int
var storage [][]byte
var sites []string
var copySources []unsafe.Pointer
var copySizes []int
func require(ok bool,msg string){if !ok{fmt.Fprintln(os.Stderr,"FAIL:",msg);os.Exit(1)}}
func requireLock(){if cudaMu.TryLock(){cudaMu.Unlock();require(false,"native boundary without cudaMu")}}
func CUDAPagedKVEnabled()bool{return false}
func CUDAPagedKVEligible(KVConfig)bool{return false}
func(*cudaBackend)newPagedKV(KVConfig)KVStore{panic("paged dispatch outside fixture")}
func(*cudaBackend)restorePagedKVFromHost(KVHostSnapshot)(KVStore,error){panic("paged restore outside fixture")}
func cloneKVConfig(c KVConfig)KVConfig{return c}
const MemoryScratchpad="scratch"
func(*cudaBackend)dallocClass(int,string,string)*cudaBuf{panic("narrowed restore conversion outside fixture")}
func(*cudaBackend)dallocKV(n int,site string)*cudaBuf{requireLock();allocations++;sites=append(sites,site);b:=make([]byte,n);storage=append(storage,b);return &cudaBuf{unsafe.Pointer(&b[0])}}
func cudaKVElements(n int){if n<0||uint64(n)>2147483647{panic("cuda: narrowed KV exceeds native element bound")}}
func uint64ToCapInt64(x uint64)int64{return int64(x)}
type preallocResult struct{Preallocate bool;Positions int;Bytes int64;Reason string}
func GraphKVPreallocFor(enabled bool,cfg KVConfig,n int,free int64)preallocResult{return preallocResult{Preallocate:enabled&&cfg.GraphCapture,Positions:n,Bytes:48}}
var C=struct{
 fcuda_mem_info func(*uint64,*uint64)int
 fcuda_free func(unsafe.Pointer)
 fcuda_d2d func(unsafe.Pointer,unsafe.Pointer,uint64)
 fak_v41_checked_stream_terminal func()int
 fcuda_h2d func(unsafe.Pointer,unsafe.Pointer,uint64)
 fcuda_kv_store_typed func(unsafe.Pointer,*float32,int,int,int)int
}{
 func(f,t *uint64)int{requireLock();memQueries++;*f=1024;*t=2048;return 0},
 func(unsafe.Pointer){requireLock();releases++},
 func(dst,src unsafe.Pointer,n uint64){requireLock();copies++;copySources=append(copySources,src);copySizes=append(copySizes,int(n));copy(unsafe.Slice((*byte)(dst),int(n)),unsafe.Slice((*byte)(src),int(n)))},
 func()int{requireLock();queries++;if sharedStop{return 1};return 0},
 func(dst,src unsafe.Pointer,n uint64){requireLock();uploads++;copy(unsafe.Slice((*byte)(dst),int(n)),unsafe.Slice((*byte)(src),int(n)))},
 func(unsafe.Pointer,*float32,int,int,int)int{panic("narrowed restore conversion outside fixture")},
}
func recoverCall(f func())(r any){defer func(){r=recover()}();f();return}
func reset(){allocations=0;copies=0;releases=0;queries=0;memQueries=0;uploads=0;sites=nil;copySources=nil;copySizes=nil;sharedStop=false}
func unlocked(){require(cudaMu.TryLock(),"entry left cudaMu locked");cudaMu.Unlock()}
func stopped(r any,site string)bool{e,ok:=r.(*BackendError);return ok&&e.Backend=="cuda"&&e.Class=="cuda-terminal-stream"&&e.Site==site}
func config(dtype Dtype)KVConfig{p:=KVPrecisionF32;if dtype==F16{p=KVPrecisionF16};if dtype==BF16{p=KVPrecisionBF16};return KVConfig{NumLayers:2,NumKVHeads:1,HeadDim:1,Precision:p,GraphCapture:true}}
func owner(dtype Dtype)*cudaKV{
 k:=&cudaKV{be:&cudaBackend{},cfg:config(dtype),maxPos:7,K:make([]dslice,2),Kraw:make([]dslice,2),V:make([]dslice,2),pos:[]int{3,8}}
 for plane,rows:=range [][]dslice{k.K,k.Kraw,k.V}{for i:=range rows{d:=dtype;if plane==1{d=F32};n:=i+1;if plane==1&&i==1{rows[i]=dslice{dtype:d};continue};b:=make([]byte,(n+3)*d.Bytes());for j:=range b{b[j]=byte(13+plane*30+i*7+j)};storage=append(storage,b);rows[i]=dslice{ptr:unsafe.Pointer(&b[0]),len:n,cap:n+3,dtype:d}}}
 return k
}
func saved(k *cudaKV)*cudaKV{v:=*k;v.K=append([]dslice(nil),k.K...);v.Kraw=append([]dslice(nil),k.Kraw...);v.V=append([]dslice(nil),k.V...);v.pos=append([]int(nil),k.pos...);return &v}
`

const cudaKVCreateCloneMain = `
func checkCreateStop(dtype Dtype,graph bool,locked bool){
 reset();sharedStop=true;graphEnabled=graph;c:=&cudaBackend{};cfg:=config(dtype)
 var out KVStore
 r:=recoverCall(func(){if locked{cudaMu.Lock();defer cudaMu.Unlock();k,err:=c.newContiguousKVLocked(cfg);if err!=nil{panic(err)};out=k}else{out=c.NewKV(cfg)}})
 require(stopped(r,"kv-create")&&out==nil,"create terminal admission")
 require(queries==1&&memQueries==0&&allocations==0&&copies==0&&releases==0&&uploads==0,"create touched device after terminal admission")
 unlocked()
}
func checkCloneStop(dtype Dtype,retained bool){
 reset();sharedStop=!retained;k:=owner(dtype);k.releaseRetained=retained;k.releaseNext=&cudaKV{};before:=saved(k)
 var out KVStore;r:=recoverCall(func(){out=k.Clone()})
 require(stopped(r,"kv-clone")&&out==nil,"clone terminal admission")
 q:=1;if retained{q=0};require(queries==q&&memQueries==0&&allocations==0&&copies==0&&releases==0&&uploads==0,"clone touched device after terminal admission")
 require(reflect.DeepEqual(k,before),"clone changed stopped source metadata or ownership")
 unlocked()
}
func hostState(dtype Dtype)KVHostSnapshot{return KVHostSnapshot{Config:config(dtype),Pos:[]int{4},K:[][]float32{{1},{2}},KRaw:[][]float32{{3},{4}},V:[][]float32{{5},{6}}}}
func checkRestoreStop(dtype Dtype,graph bool){
 reset();sharedStop=true;graphEnabled=graph;c:=&cudaBackend{};state:=hostState(dtype)
 var out KVStore;var err error;r:=recoverCall(func(){out,err=c.RestoreKVFromHost(state)})
 require(r==nil&&out==nil&&stopped(err,"kv-create"),"restore terminal admission")
 require(queries==1&&memQueries==0&&allocations==0&&copies==0&&releases==0&&uploads==0,"restore touched device after terminal admission")
 require(reflect.DeepEqual(state,hostState(dtype)),"restore changed stopped input");unlocked()
}
func checkRestoreHealthy(){
 reset();graphEnabled=false;c:=&cudaBackend{};state:=hostState(F32);out,err:=c.RestoreKVFromHost(state);require(err==nil&&out!=nil,"healthy restore error")
 require(queries==1&&memQueries==0&&allocations==6&&uploads==6&&copies==0&&releases==0,"healthy restore operations")
 k:=out.(*cudaKV);require(k.cfg==state.Config&&reflect.DeepEqual(k.pos,state.Pos),"healthy restore metadata")
 for l:=range k.K{for p,d:=range []dslice{k.K[l],k.Kraw[l],k.V[l]}{src:=[][]float32{state.K[l],state.KRaw[l],state.V[l]}[p];require(d.dtype==F32&&d.len==1&&d.cap==1&&reflect.DeepEqual(unsafe.Slice((*float32)(d.ptr),d.len),src),"host mock restore payload")}}
 k.pos[0]=99;require(state.Pos[0]==4,"restore position alias");unlocked()
}
func checkHealthy(dtype Dtype){
 reset();graphEnabled=true;c:=&cudaBackend{};cfg:=config(dtype);k:=c.NewKV(cfg).(*cudaKV)
 require(queries==1&&memQueries==1&&allocations==6&&copies==0&&releases==0,"healthy constructor operation contract")
 require(k.cfg==cfg&&k.maxPos==2&&len(k.pos)==0,"healthy constructor geometry")
 for l:=range k.K{for p,d:=range []dslice{k.K[l],k.Kraw[l],k.V[l]}{expected:=dtype;if p==1{expected=F32};require(d.ptr!=nil&&d.len==0&&d.cap==2&&d.dtype==expected,"healthy constructor row ownership")}}
 unlocked();reset();k=owner(dtype);before:=saved(k);n:=k.Clone().(*cudaKV)
 require(queries==1&&allocations==5&&copies==5&&releases==0,"healthy Clone operation contract")
 require(reflect.DeepEqual(k,before)&&n.be==k.be&&n.cfg==k.cfg&&n.maxPos==0&&reflect.DeepEqual(n.pos,k.pos),"healthy Clone source/geometry contract")
 index:=0;seen:=map[unsafe.Pointer]bool{}
 for l:=range k.K{for p,pair:=range [][2]dslice{{k.K[l],n.K[l]},{k.Kraw[l],n.Kraw[l]},{k.V[l],n.V[l]}}{src,dst:=pair[0],pair[1];require(src.dtype==dst.dtype,"Clone dtype");if src.len==0{require(dst.ptr==nil&&dst.len==0&&dst.cap==0,"Clone empty row");continue};require(dst.ptr!=src.ptr&&!seen[dst.ptr]&&dst.len==src.len&&dst.cap==src.len,"Clone exact distinct ownership");seen[dst.ptr]=true;nb:=src.len*src.dtype.Bytes();require(copySources[index]==src.ptr&&copySizes[index]==nb,"Clone copy order/extent");prefix:=[]string{"kv-key-clone layer ","kv-pre-rope-key-clone layer ","kv-value-clone layer "}[p];require(sites[index]==prefix+itoaC(l),"Clone allocation site/order");require(reflect.DeepEqual(unsafe.Slice((*byte)(dst.ptr),nb),unsafe.Slice((*byte)(src.ptr),nb)),"host mock Clone payload");index++}}
 require(index==5&&len(seen)==5,"Clone distinct pointer set")
 n.pos[0]=77;require(k.pos[0]==3,"Clone position alias");unlocked()
}
func main(){
 require(len(os.Args)==2,"choose mode")
 switch os.Args[1]{case "create":checkCreateStop(F32,true,false);case "clone":checkCloneStop(F32,false);case "restore":checkRestoreStop(F32,true);case "all":
 for _,d:=range []Dtype{F32,F16,BF16}{for _,graph:=range []bool{false,true}{checkCreateStop(d,graph,false);checkCreateStop(d,graph,true);checkRestoreStop(d,graph)};checkCloneStop(d,false);checkCloneStop(d,true);checkHealthy(d)}
 checkRestoreHealthy()
 reset();sharedStop=true;c:=&cudaBackend{};cfg:=config(F32);cfg.Precision="invalid";r:=recoverCall(func(){c.NewKV(cfg)});require(r=="cuda: unrealized KV precision invalid"&&queries==0,"precision validation precedence");unlocked()
 cfg=config(F16);cfg.HeadDim=0;r=recoverCall(func(){c.NewKV(cfg)});require(r=="cuda: narrowed KV row exceeds native element bound"&&queries==0,"extent validation precedence");unlocked()
 cfg=config(F32);cfg.NumLayers=-1;r=recoverCall(func(){c.NewKV(cfg)});require(r!=nil&&queries==0,"negative layer validation precedence");unlocked()
 state:=hostState(F32);state.Config.NumLayers=-1;var out KVStore;var err error;r=recoverCall(func(){out,err=c.RestoreKVFromHost(state)});require(r==nil&&out==nil&&err!=nil&&err.Error()=="compute: invalid negative KV host snapshot geometry"&&queries==0,"restore validation precedence");unlocked()
 state=hostState(F32);state.Config.Precision="invalid";r=recoverCall(func(){c.RestoreKVFromHost(state)});require(r=="cuda: unrealized KV precision invalid"&&queries==0,"restore precision precedence");unlocked()
 state=hostState(F16);state.Config.HeadDim=0;state.K=make([][]float32,2);state.KRaw=make([][]float32,2);state.V=make([][]float32,2);r=recoverCall(func(){c.RestoreKVFromHost(state)});require(r=="cuda: narrowed KV row exceeds native element bound"&&queries==0,"restore narrowed extent precedence");unlocked()
 default:require(false,"unknown mode")}
 runtime.KeepAlive(storage)
}
`
