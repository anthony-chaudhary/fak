//go:build fak_cuda_read_terminal_host_test

package compute

import (
	"context"
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

// Execute exact Read/guard/readiness functions and the entire real Go fault
// latch with host boundary fakes. C is a fixture selector, not cgo. This does not
// qualify CUDA linking, native terminal aggregation, or device behavior.
// fak-test:runtime medium est=4s lane=optin
func TestCUDAReadTerminalExactHost(t *testing.T) {
	extract := func(path string, names ...string) string {
		t.Helper()
		source, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		fs := token.NewFileSet()
		file, err := parser.ParseFile(fs, path, source, 0)
		if err != nil {
			t.Fatal(err)
		}
		var out strings.Builder
		for _, name := range names {
			found := 0
			for _, decl := range file.Decls {
				if fn, ok := decl.(*ast.FuncDecl); ok && fn.Name.Name == name {
					out.Write(source[fs.Position(fn.Pos()).Offset:fs.Position(fn.End()).Offset])
					out.WriteByte('\n')
					found++
				}
			}
			if found != 1 {
				t.Fatalf("expected one %s in %s, got %d", name, path, found)
			}
		}
		return out.String()
	}
	read := extract("cuda.go", "Read", "readLocked")
	const call = "\tcudaV41CheckedSubmit()\n"
	if strings.Count(read, call) != 1 {
		t.Fatal("actual Read must contain exactly one shared terminal guard")
	}
	functions := extract("cuda_v41_checked_stream.go", "cudaV41CheckedSubmit") +
		extract("cuda_backend_state.go", "invalidForSubmit", "invalidStateError", "Ready", "DeviceFaultLatch")
	latch, err := os.ReadFile("devicefault.go")
	if err != nil {
		t.Fatal(err)
	}
	tool := filepath.Join(runtime.GOROOT(), "bin", "go")
	if runtime.GOOS == "windows" {
		tool += ".exe"
	}
	for _, removed := range []bool{false, true} {
		dir := t.TempDir()
		body := read
		if removed {
			body = strings.Replace(body, call, "", 1)
		}
		for name, data := range map[string]string{
			"go.mod":          "module cuda-read-terminal-witness\n\ngo 1.26.0\n",
			"actual.go":       "package compute\nimport (\"sync/atomic\"; \"unsafe\")\n" + body + functions,
			"devicefault.go":  string(latch),
			"witness_test.go": cudaReadTerminalHostFixture,
		} {
			if err := os.WriteFile(filepath.Join(dir, name), []byte(data), 0600); err != nil {
				t.Fatal(err)
			}
		}
		ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
		cmd := exec.CommandContext(ctx, tool, "test", "-count=1", "-run=^TestReadTerminalWitness$", ".")
		cmd.Dir = dir
		cmd.Env = append(os.Environ(), "GOWORK=off", "GOTOOLCHAIN=local", "GOFLAGS=", "CGO_ENABLED=0")
		output, runErr := cmd.CombinedOutput()
		cancel()
		if !removed && runErr != nil {
			t.Fatalf("exact source host witness: %v\n%s", runErr, output)
		}
		if removed && (runErr == nil || !strings.Contains(string(output), "terminal Read was admitted")) {
			t.Fatalf("removed-guard control must fail by the behavioral assertion, not compilation: %v\n%s", runErr, output)
		}
	}
}

const cudaReadTerminalHostFixture = `package compute
import (
 "reflect"
 "sync"
 "testing"
 "unsafe"
)
// Host-only surrounding types and API boundaries; actual functions and latch
// are emitted separately without rewriting their bodies.
type Backend interface{}
type Dtype int
const (F32 Dtype=iota; F16; BF16)
type Tensor struct { buf any; Dtype Dtype; elements int }
func (t Tensor) Numel() int { return t.elements }
type hostBuf struct { f32 []float32 }
type cudaBackend struct { fenceGen uint64; faultLatch *DeviceFaultLatch }
type cudaBuf struct { ptr unsafe.Pointer; device int; be *cudaBackend; bornGen uint64; invalid uint32 }
var cudaMu sync.Mutex
type BackendError struct { Backend,Site,Class,Message string }
func (e *BackendError) Error() string { return e.Message }
type CUDAOpError struct { Op,Site,Msg string; Code int }
func (e *CUDAOpError) Error() string { return e.Msg }
type Qwen35GDNKernelError struct { Stage string; Code int }
func (*Qwen35GDNKernelError) Error() string { return "fixture kernel error" }
type Qwen35GDNInvalidStateError struct { Operand string }
func (*Qwen35GDNInvalidStateError) Error() string { return "fixture invalid buffer" }
type DeviceAllocError struct { Site string }
func (*DeviceAllocError) Error() string { return "fixture allocation error" }
var nativeStatus, queries, submits, begins, ends, copies, allocs, converts, frees int
var events []string
var C = struct { fak_v41_checked_stream_terminal func() int }{
 func() int {
  if cudaMu.TryLock() { cudaMu.Unlock(); panic("terminal query outside cudaMu") }
  queries++; events=append(events,"terminal"); return nativeStatus
 },
}
func cudaV41CompressorSubmitLocked(*cudaBuf) { submits++; events=append(events,"submit") }
func cudaV41CompressorReadBeginLocked(*cudaBackend) bool { begins++; events=append(events,"begin"); return false }
func cudaV41CompressorReadEndLocked(*cudaBackend,bool,bool) { ends++; events=append(events,"end") }
const cudaQ56KMMQQualification=false
func cudaQ56KMMQRetirementFailedLocked(int) { panic("unexpected qualification-only boundary") }
var cudaReadNative struct {
 copy func(int,unsafe.Pointer,unsafe.Pointer,uintptr) int
 allocConversion func(*cudaBackend,int) unsafe.Pointer
 convert func(unsafe.Pointer,unsafe.Pointer,int,Dtype)
 freeConversion func(unsafe.Pointer)
}
func TestReadTerminalWitness(t *testing.T) {
 for _, tc := range []struct {
  name string; terminal, host, empty, invalid, nilLatch, reconstruct, poisoned bool; dtype Dtype; device int
 }{
  {name:"terminal-nil-latch",terminal:true,nilLatch:true},
  {name:"terminal-reporting-latch",terminal:true},
  {name:"terminal-poisoned-latch",terminal:true,poisoned:true},
  {name:"terminal-reconstructed-latch",terminal:true,reconstruct:true},
  {name:"terminal-device1",terminal:true,device:1},
  {name:"terminal-f16",terminal:true,dtype:F16},
  {name:"terminal-bf16",terminal:true,dtype:BF16},
  {name:"terminal-empty",terminal:true,empty:true},
  {name:"terminal-host-view",terminal:true,host:true},
  {name:"terminal-invalid-precedence",terminal:true,invalid:true},
  {name:"healthy-f32"}, {name:"healthy-device1",device:1},
  {name:"healthy-f16",dtype:F16}, {name:"healthy-bf16",dtype:BF16},
  {name:"healthy-empty",empty:true}, {name:"healthy-host-view",host:true},
 } {
  t.Run(tc.name,func(t *testing.T) {
   nativeStatus=0; if tc.terminal { nativeStatus=711 }
   queries,submits,begins,ends,copies,allocs,converts,frees=0,0,0,0,0,0,0,0; events=nil
   c:=&cudaBackend{fenceGen:41}
   if !tc.nilLatch { c.faultLatch=NewDeviceFaultLatch("cuda",3) }
   if tc.poisoned { c.faultLatch.Observe(DeviceFaultExecution,"older-go-fault",1700) }
   if tc.reconstruct {
    c.faultLatch.Observe(DeviceFaultExecution,"prior-native-stop",711)
    if err:=c.faultLatch.Reconstruct(func()error{return nil},func()error{return nil});err!=nil { t.Fatal(err) }
    if c.faultLatch.Snapshot().Health!=DeviceHealthy { t.Fatal("real latch reconstruction did not reopen Go admission") }
   }
   storage:=[]float32{3,4}; scratch:=[]float32{0,0}
   db:=&cudaBuf{ptr:unsafe.Pointer(&storage[0]),device:tc.device,be:c,bornGen:41}
   if tc.invalid { db.invalid=1 }
   tensor:=Tensor{buf:db,Dtype:tc.dtype,elements:2}
   if tc.empty { tensor.elements=0 }
   if tc.host { tensor.buf=&hostBuf{f32:storage} }
   before:=*db; latchBefore:=c.faultLatch.Snapshot()
   cudaReadNative.copy=func(device int,dst,src unsafe.Pointer,n uintptr)int {
    copies++; events=append(events,"copy")
    if device!=tc.device || n!=8 { panic("copy domain changed") }
    want:=unsafe.Pointer(&storage[0]);if tc.dtype!=F32 {want=unsafe.Pointer(&scratch[0])}
    if src!=want {panic("copy source changed")}
    out:=unsafe.Slice((*float32)(dst),2);out[0]=11;out[1]=22;return 0
   }
   cudaReadNative.allocConversion=func(owner *cudaBackend,n int)unsafe.Pointer {allocs++;if owner!=c||n!=8{panic("conversion allocation changed")};return unsafe.Pointer(&scratch[0])}
   cudaReadNative.convert=func(dst,src unsafe.Pointer,n int,dt Dtype) {converts++;if dst!=unsafe.Pointer(&scratch[0])||src!=db.ptr||n!=2||dt!=tc.dtype{panic("conversion changed")}}
   cudaReadNative.freeConversion=func(p unsafe.Pointer) {frees++;if p!=unsafe.Pointer(&scratch[0]){panic("conversion free changed")}}
   var out []float32;var failure any
   func(){defer func(){failure=recover()}();out=c.Read(tensor)}()
   stopped:=tc.terminal&&!tc.host&&!tc.invalid
   if stopped {
    err,ok:=failure.(*BackendError)
    if !ok||err.Class!="cuda-terminal-stream"||err.Site!="cudaBufForSubmit"||out!=nil {t.Fatalf("terminal Read was admitted or lost existing typed refusal: out=%v panic=%#v",out,failure)}
    if queries!=1||submits!=0||begins!=0||ends!=0||copies!=0||allocs!=0||converts!=0||frees!=0 {t.Fatal("terminal Read reached a producer, copy or cleanup boundary")}
   } else if tc.invalid {
    if _,ok:=failure.(*Qwen35GDNInvalidStateError);!ok||queries!=0||copies!=0 {t.Fatal("invalid-buffer precedence changed")}
   } else {
    if failure!=nil {t.Fatal(failure)}
    if tc.host {
     if queries!=0||submits!=0||begins!=0||ends!=0||copies!=0||len(out)!=2||&out[0]!=&storage[0] {t.Fatal("host-only view changed")}
    } else {
     if queries!=1||submits!=1||begins!=1||ends!=1||len(events)==0||events[0]!="terminal" {t.Fatal("healthy admission ordering changed")}
     wantCopies:=1;if tc.empty{wantCopies=0}
     if copies!=wantCopies {t.Fatal("healthy copy count changed")}
     if tc.empty {if len(out)!=0{t.Fatal("empty result changed")}} else if !reflect.DeepEqual(out,[]float32{11,22}) {t.Fatal("healthy result changed")}
    }
   }
   wantGen:=uint64(41);if !stopped&&!tc.host&&!tc.empty&&!tc.invalid&&tc.device==0{wantGen++}
   if c.fenceGen!=wantGen||db.Ready()!=(wantGen>41&&!tc.invalid) {t.Fatal("unjustified readiness promotion")}
   wantConversions:=0;if !stopped&&!tc.host&&!tc.empty&&!tc.invalid&&tc.dtype!=F32{wantConversions=1}
   if allocs!=wantConversions||converts!=wantConversions||frees!=wantConversions{t.Fatal("conversion lifetime changed")}
   if *db!=before||!reflect.DeepEqual(storage,[]float32{3,4})||!reflect.DeepEqual(latchBefore,c.faultLatch.Snapshot()){t.Fatal("read changed borrower or Go fault state")}
   wantStatus:=0;if tc.terminal{wantStatus=711};if nativeStatus!=wantStatus{t.Fatal("read cleared native terminal")}
  })
 }
}
`
