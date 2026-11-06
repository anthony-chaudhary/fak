package compute

import (
	"bytes"
	"context"
	"go/ast"
	"go/format"
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

func assertCUDAUploadLockedBoundary(t *testing.T) {
	t.Helper()
	f, err := parser.ParseFile(token.NewFileSet(), "cuda.go", nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	for _, d := range f.Decls {
		fn, ok := d.(*ast.FuncDecl)
		if !ok || fn.Recv == nil || fn.Name.Name != "uploadClass" {
			continue
		}
		var body bytes.Buffer
		if err := format.Node(&body, token.NewFileSet(), fn.Body); err != nil {
			t.Fatal(err)
		}
		const want = "{\n\trequireCUDAUploadLayout(\"cuda Upload\", t, as, class, site)\n\tcudaMu.Lock()\n\tdefer cudaMu.Unlock()\n\treturn c.uploadClassLocked(t, as, class, site)\n}"
		if body.String() != want {
			t.Fatalf("uploadClass layout/lock/forward boundary changed: %s", body.String())
		}
		return
	}
	t.Fatal("missing public uploadClass wrapper")
}

// Executes production owner Go, native binding and native owner C++ against
// injected CUDA/allocator/dispatch primitives. No CUDA device or full backend.
// fak-test:runtime medium est=3s lane=default
func TestCUDAPrefillOwnedInputHostContract(t *testing.T) {
	assertCUDAUploadLockedBoundary(t)
	read := func(name string) string {
		t.Helper()
		b, e := os.ReadFile(name)
		if e != nil {
			t.Fatal(e)
		}
		return string(b)
	}
	fs := token.NewFileSet()
	raw := read("cuda_backend_state.go")
	f, e := parser.ParseFile(fs, "cuda_backend_state.go", raw, 0)
	if e != nil {
		t.Fatal(e)
	}
	var buffer string
	for _, d := range f.Decls {
		g, ok := d.(*ast.GenDecl)
		if !ok {
			continue
		}
		for _, s := range g.Specs {
			typ, ok := s.(*ast.TypeSpec)
			if ok && typ.Name.Name == "cudaBuf" {
				buffer = "type " + raw[fs.Position(typ.Pos()).Offset:fs.Position(typ.End()).Offset]
			}
		}
	}
	if buffer == "" {
		t.Fatal("production cudaBuf absent")
	}
	dir := t.TempDir()
	files := map[string]string{
		"go.mod":                       "module prefillownerwitness\ngo 1.26\n",
		"owner.go":                     strings.Replace(read("cuda_prefill_input.go"), "//go:build cuda && cgo\n", "", 1),
		"cuda_prefill_input_native.go": strings.Replace(read("cuda_prefill_input_native.go"), "//go:build cuda && cgo\n", "", 1),
		"cuda_backend.h":               read("cuda_backend.h"),
		"prefill_input_owner.cuh":      read("cuda/prefill_input_owner.cuh"),
		"owner_test.go":                strings.Replace(cudaPrefillOwnerGoFixture, "__CUDA_BUF_DECL__", buffer, 1),
		"native_binding_test.go":       cudaPrefillOwnerBindingFixture,
		"mock_native.cc":               cudaPrefillOwnerNativePrefix,
	}
	for name, body := range files {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0600); err != nil {
			t.Fatal(err)
		}
	}
	compiler, err := exec.LookPath("c++")
	if err != nil {
		t.Fatal("host C++ compiler required for ownership witness")
	}
	cpp := filepath.Join(dir, "native-witness.cpp")
	binary := filepath.Join(dir, "native-witness")
	if runtime.GOOS == "windows" {
		binary += ".exe"
	}
	if err := os.WriteFile(cpp, []byte(cudaPrefillOwnerNativeFixture), 0600); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", cpp, "-o", binary)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("compile native owner fixture: %v\n%s", err, out)
	}
	if out, err := exec.CommandContext(ctx, binary).CombinedOutput(); err != nil {
		t.Fatalf("native owner fixture: %v\n%s", err, out)
	}
	// The standalone main is not part of the cgo package.
	if err := os.Remove(cpp); err != nil {
		t.Fatal(err)
	}
	goBin := filepath.Join(runtime.GOROOT(), "bin", "go")
	cmd = exec.CommandContext(ctx, goBin, "test", "-timeout=20s", "-count=1", "-v", ".")
	cmd.Dir = dir
	cmd.Env = append(os.Environ(), "GOWORK=off", "CGO_ENABLED=1", "GOMAXPROCS=2")
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("actual owner/cgo fixture: %v\n%s", err, out)
	} else {
		t.Log(string(out))
	}
}

const cudaPrefillOwnerGoFixture = `package compute

import (
	"errors"
	"runtime"
	"sync"
	"testing"
	"unsafe"
)

const (
	F32              = 1
	RowMajor         = 0
	MemoryActivation = 5
	MemoryWeights    = 0
)

type MemoryClass=int
type Dtype=int
type Layout=int
type HostBuffer interface{}
__CUDA_BUF_DECL__
type Tensor struct {
	host          bool
	Dtype, Layout int
	Shape         []int
	buf           *cudaBuf
	values        []float32
}

func (t Tensor) Numel() int {
	n := 1
	for _, d := range t.Shape {
		n *= d
	}
	return n
}

type cudaBackend struct {
	capturing, promptPanel bool
	graphOnce              *int
}

var cudaGraphOnceOwner *cudaBackend
var cudaMu sync.Mutex

func AdmitDevice(*cudaBackend, string) error { return nil }
func makeTensor(_ *cudaBackend, dt, layout int, shape []int, _ any, b *cudaBuf) Tensor {
	return Tensor{Dtype: dt, Layout: layout, Shape: shape, buf: b}
}

type CUDAOpError struct {
	Op, Site, Msg string
	Code, Class   int
}

func (e *CUDAOpError) Error() string { return e.Msg }

var mode string
var projectAt, readAt, projects, reads, acquires, retires, failures int
var retirementFail, retirementPanic, notificationPanic bool
var uploads, admissionStatus int

func (c *cudaBackend) Host(t Tensor) ([]float32, bool)         { return t.values, t.host }
func requireCUDAUploadLayout(string, Tensor, int, int, string) {}
func (c *cudaBackend) uploadClassLocked(t Tensor, _ int, _ int, _ string) Tensor {
	assertHeld()
	uploads++
	t.host = false
	return t
}

var copied []float32
var primary = errors.New("original projection failure")

func assertHeld() {
	if cudaMu.TryLock() {
		cudaMu.Unlock()
		panic("critical section unlocked")
	}
}
func (c *cudaBackend) batchedMatMulLocked(w, x Tensor, p int) Tensor {
	assertHeld()
	projects++
	if mode == "project" && projects == projectAt {
		panic(primary)
	}
	if x.buf != cudaPrefillInputOwner.buffer || x.buf.ptr == nil {
		panic("owner not held")
	}
	return Tensor{values: append([]float32(nil), unsafe.Slice((*float32)(x.buf.ptr),x.Numel())...)}
}
func (c *cudaBackend) readLocked(t Tensor) []float32 {
	assertHeld()
	reads++
	if mode == "read" && reads == readAt {
		panic(&CUDAOpError{Op: "Read", Site: "read-d2h", Code: 1719, Msg: "original read failure"})
	}
	return t.values
}
func reset() {
	// Fake runtime has no outstanding GPU work; test teardown may unpin safely.
	cudaPrefillInputOwner.pin.Unpin()
	cudaPrefillInputOwner.pin = runtime.Pinner{}
	cudaPrefillInputOwner.buffer = nil
	cudaPrefillInputOwner.host = nil
	cudaPrefillInputOwner.terminal = nil
	retirementFail = false
	retirementPanic = false
	notificationPanic = false
	uploads = 0
	admissionStatus = 0
	cudaGraphOnceOwner = nil
	mode = ""
	projects = 0
	reads = 0
	acquires = 0
	retires = 0
	failures = 0
	copied = nil
	cudaPrefillInputNative = cudaPrefillInputOperations{
		admit: func() int { assertHeld(); return admissionStatus },
		acquire: func(host []float32, panel bool, b *cudaBuf) int {
			assertHeld()
			acquires++
			if mode == "allocation" {
				return -76002
			}
			copied = append([]float32(nil), host...)
			b.ptr = unsafe.Pointer(&copied[0])
			if mode == "upload" {
				return 700
			}
			return 0
		},
		retire: func(b *cudaBuf) int {
			assertHeld()
			retires++
			if retirementFail || mode == "retire" || mode == "project-retire" {
				return 719
			}
			if retirementPanic || mode == "retire-panic" {
				panic("retirement panic")
			}
			return 0
		},
		failed: func(status int) {
			assertHeld()
			failures++
			if notificationPanic {
				panic("notification panic")
			}
			if status != 1719 {
				panic("wrong failure status")
			}
		},
	}
}
func input(host []float32) Tensor {
	return Tensor{Dtype: F32, Layout: RowMajor, Shape: []int{1, len(host)}}
}

// fak-test:runtime fast est=20ms lane=default
func TestActualOwnerFreshInputAndRefusals(t *testing.T) {
	reset()
	defer reset()
	c := &cudaBackend{}
	h := []float32{1, 2}
	for i := 0; i < 100; i++ {
		h[0] = float32(i)
		q, k, v, e := c.prefillHostProjections(input(h), h, Tensor{}, Tensor{}, Tensor{}, 1)
		if e != nil || q[0] != h[0] || k[0] != h[0] || v[0] != h[0] {
			t.Fatalf("fresh bytes lost: %v", e)
		}
		if cudaPrefillInputOwner.buffer != nil || cudaPrefillInputOwner.host != nil {
			t.Fatal("owner not retired")
		}
	}
	for _, kind := range []string{"capture", "receiver-graph", "global-graph"} {
		reset()
		if kind == "capture" {
			c.capturing = true
		}
		if kind == "receiver-graph" {
			c.graphOnce = new(int)
		}
		if kind == "global-graph" {
			cudaGraphOnceOwner = c
		}
		_, _, _, e := c.prefillHostProjections(input(h), h, Tensor{}, Tensor{}, Tensor{}, 1)
		if e == nil || acquires != 0 {
			t.Fatal("graph scope allocated")
		}
		c.capturing = false
		c.graphOnce = nil
		cudaGraphOnceOwner = nil
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestActualOwnerTerminalBound(t *testing.T) {
	for _, m := range []string{"upload", "retire", "retire-panic", "read"} {
		for at := 1; at <= 3; at++ {
			reset()
			mode = m
			readAt = at
			c := &cudaBackend{}
			h := []float32{1}
			var caught any
			func() {
				defer func() { caught = recover() }()
				_, _, _, e := c.prefillHostProjections(input(h), h, Tensor{}, Tensor{}, Tensor{}, 1)
				if m != "read" && e == nil {
					t.Fatal("failure lost")
				}
			}()
			if m == "read" {
				e, ok := caught.(*CUDAOpError)
				if !ok || e.Code != 1719 || failures != 1 || retires != 0 {
					t.Fatalf("read failure lost/retried: %v", caught)
				}
			} else if caught != nil {
				t.Fatalf("unexpected panic %v", caught)
			}
			if cudaPrefillInputOwner.buffer == nil || cudaPrefillInputOwner.buffer.ptr == nil || cudaPrefillInputOwner.host == nil || cudaPrefillInputOwner.terminal == nil {
				t.Fatal("strong owner lost")
			}
			h[0] = 99
			if cudaPrefillInputOwner.host[0] != 1 { t.Fatal("retained host source aliases caller mutation") }
			before := acquires
			for i := 0; i < 1000; i++ {
				_, _, _, e := c.prefillHostProjections(input(h), h, Tensor{}, Tensor{}, Tensor{}, 1)
				if e == nil || acquires != before {
					t.Fatal("failure admitted another allocation")
				}
			}
		}
	}
	reset()
}

// fak-test:runtime fast est=20ms lane=default
func TestActualOwnerProjectionPanicAndAllocationFailure(t *testing.T) {
	defer reset()
	h := []float32{1}
	c := &cudaBackend{}
	for at := 1; at <= 3; at++ {
		reset()
		mode = "project"
		projectAt = at
		var caught any
		func() {
			defer func() { caught = recover() }()
			c.prefillHostProjections(input(h), h, Tensor{}, Tensor{}, Tensor{}, 1)
		}()
		if caught != primary || retires != 1 || cudaPrefillInputOwner.buffer != nil {
			t.Fatal("projection panic changed or owner lost")
		}
	}
	reset()
	mode = "allocation"
	_, _, _, e := c.prefillHostProjections(input(h), h, Tensor{}, Tensor{}, Tensor{}, 1)
	if e == nil || retires != 0 || cudaPrefillInputOwner.buffer != nil {
		t.Fatal("allocation failure retained nonexistent storage")
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestActualOwnerPreservesPrimaryDuringCleanupFailures(t *testing.T) {
	defer reset()
	h := []float32{1}
	c := &cudaBackend{}
	for _, panicCleanup := range []bool{false, true} {
		reset()
		mode = "project"
		projectAt = 2
		retirementFail = !panicCleanup
		retirementPanic = panicCleanup
		var caught any
		func() {
			defer func() { caught = recover() }()
			c.prefillHostProjections(input(h), h, Tensor{}, Tensor{}, Tensor{}, 1)
		}()
		if caught != primary || cudaPrefillInputOwner.buffer == nil || cudaPrefillInputOwner.terminal == nil {
			t.Fatal("primary or retained owner lost")
		}
	}
	reset()
	mode = "read"
	readAt = 2
	notificationPanic = true
	var caught any
	func() {
		defer func() { caught = recover() }()
		c.prefillHostProjections(input(h), h, Tensor{}, Tensor{}, Tensor{}, 1)
	}()
	e, ok := caught.(*CUDAOpError)
	if !ok || e.Code != 1719 || cudaPrefillInputOwner.buffer == nil {
		t.Fatal("read primary lost to notification panic")
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestActualOwnerSerializesConcurrentCallers(t *testing.T) {
	reset()
	defer reset()
	var wg sync.WaitGroup
	for worker := 0; worker < 32; worker++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			c := &cudaBackend{}
			h := []float32{float32(id)}
			for call := 0; call < 30; call++ {
				q, _, _, err := c.prefillHostProjections(input(h), h, Tensor{}, Tensor{}, Tensor{}, 1)
				if err != nil || q[0] != h[0] {
					t.Errorf("interleaved input %d: %v", id, err)
					return
				}
			}
		}(worker)
	}
	wg.Wait()
	if acquires != 960 || retires != 960 || cudaPrefillInputOwner.buffer != nil {
		t.Fatal("cross-call owner failure")
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestActualOwnerAdmissionBeforeWeightUpload(t *testing.T) {
	reset()
	defer reset()
	c := &cudaBackend{}
	h := []float32{1}
	w := Tensor{Dtype: F32, Layout: RowMajor, Shape: []int{1, 1}, host: true, values: h}
	admissionStatus = 34
	_, _, _, err := c.prefillHostProjections(input(h), h, w, w, w, 1)
	if err == nil || uploads != 0 || acquires != 0 {
		t.Fatal("native refusal uploaded a weight")
	}
	admissionStatus = 0
	_, _, _, err = c.prefillHostProjections(input(h), h, w, w, w, 1)
	if err != nil || uploads != 3 || acquires != 1 {
		t.Fatalf("admitted host weights missing: %v", err)
	}
}
`

const cudaPrefillOwnerBindingFixture = `package compute
import("testing";"runtime")
var originalNative=cudaPrefillInputNative
// fak-test:runtime fast est=10ms lane=default
func TestActualCgoOwnerBinding(t *testing.T){
 reset();defer reset();cudaPrefillInputNative=originalNative
 c:=&cudaBackend{};host:=[]float32{1,2}
 for i:=0;i<10;i++{host[0]=float32(i);q,_,_,err:=c.prefillHostProjections(input(host),host,Tensor{},Tensor{},Tensor{},1);if err!=nil||q[0]!=host[0]||cudaPrefillInputOwner.buffer!=nil{t.Fatalf("actual owner/native ABI: %v",err)}}
 b:=&cudaBuf{n:8,hostKeep:host};var pin runtime.Pinner;pin.Pin(&host[0]);defer pin.Unpin()
 if rc:=originalNative.acquire(host,false,b);rc!=0||b.ptr==nil||b.managed{t.Fatalf("native ABI acquire: %d",rc)}
 originalNative.failed(700)
 if rc:=originalNative.retire(b);rc!=700||b.ptr==nil{t.Fatalf("native failed owner: %d",rc)}
 next:=&cudaBuf{n:8};if rc:=originalNative.acquire(host,false,next);rc!=700||next.ptr!=nil{t.Fatalf("native terminal admission: %d",rc)}
 // Native fake runtime has no pending GPU work; test teardown may unpin.
}
`

const cudaPrefillOwnerNativePrefix = `#include <cassert>
#include <cstddef>
#include <cstring>
#include <unordered_map>
using cudaError_t=int;
constexpr int cudaSuccess=0,cudaErrorInvalidValue=1,cudaErrorInitializationError=3,cudaErrorInvalidDevice=10,cudaErrorNotReady=34;
constexpr int FAK_ALLOC_PREFILL_SCRATCH=6,FAK_ALLOC_ACTIVATIONS_SCRATCH=5;
#define FAK_ALLOC_SITE(x) 1
static void *g_stream=reinterpret_cast<void *>(1);
static bool g_capture_open=false,uncertain=false,pending=false,ordinary_fail=false,all_fail=false;
static int g_cuda_cleanup_status=0,device=0,device_status=0,copy_status=0,drain_status=0,free_status=0;
static int allocs=0,copies=0,drains=0,frees=0,untracks=0;
static size_t g_panel_live_bytes=0;
static std::unordered_map<void *,size_t>g_live,g_managed_live,g_panel_live;
static char storage[64];
static bool v41_candidates_stream_uncertain(){return uncertain;}
static bool v41c_pending_contains(void *){return pending;}
static int cudaGetDevice(int *out){*out=device;return device_status;}
static void *fcuda_malloc_tagged(size_t bytes,int,int){++allocs;if(ordinary_fail||all_fail)return nullptr;g_live[storage]=bytes;return storage;}
static void *fcuda_malloc_managed_tagged(size_t bytes,int,int){++allocs;if(all_fail)return nullptr;g_managed_live[storage]=bytes;return storage;}
static void *fcuda_malloc_panel_tagged(size_t bytes,int,int){++allocs;if(all_fail)return nullptr;g_panel_live[storage]=bytes;g_panel_live_bytes+=bytes;return storage;}
static int fcuda_h2d_checked(void *dst,const void *host,size_t bytes){++copies;if(copy_status)return copy_status;memcpy(dst,host,bytes);return 0;}
static int cudaStreamSynchronize(void *){++drains;return drain_status;}
static int cudaFree(void *){++frees;return free_status;}
static void fak_alloc_untrack(void *){++untracks;}
#include "prefill_input_owner.cuh"
`

const cudaPrefillOwnerNativeFixture = `#include <cassert>
#include <cstddef>
#include <cstring>
#include <unordered_map>
using cudaError_t=int;
constexpr int cudaSuccess=0,cudaErrorInvalidValue=1,cudaErrorInitializationError=3,cudaErrorInvalidDevice=10,cudaErrorNotReady=34;
constexpr int FAK_ALLOC_PREFILL_SCRATCH=6,FAK_ALLOC_ACTIVATIONS_SCRATCH=5;
#define FAK_ALLOC_SITE(x) 1
static void *g_stream=reinterpret_cast<void *>(1);
static bool g_capture_open=false,uncertain=false,pending=false,ordinary_fail=false,all_fail=false;
static int g_cuda_cleanup_status=0,device=0,device_status=0,copy_status=0,drain_status=0,free_status=0;
static int allocs=0,copies=0,drains=0,frees=0,untracks=0;
static size_t g_panel_live_bytes=0;
static std::unordered_map<void *,size_t>g_live,g_managed_live,g_panel_live;
static char storage[64];
static bool v41_candidates_stream_uncertain(){return uncertain;}
static bool v41c_pending_contains(void *){return pending;}
static int cudaGetDevice(int *out){*out=device;return device_status;}
static void *fcuda_malloc_tagged(size_t bytes,int,int){++allocs;if(ordinary_fail||all_fail)return nullptr;g_live[storage]=bytes;return storage;}
static void *fcuda_malloc_managed_tagged(size_t bytes,int,int){++allocs;if(all_fail)return nullptr;g_managed_live[storage]=bytes;return storage;}
static void *fcuda_malloc_panel_tagged(size_t bytes,int,int){++allocs;if(all_fail)return nullptr;g_panel_live[storage]=bytes;g_panel_live_bytes+=bytes;return storage;}
static int fcuda_h2d_checked(void *dst,const void *host,size_t bytes){++copies;if(copy_status)return copy_status;memcpy(dst,host,bytes);return 0;}
static int cudaStreamSynchronize(void *){++drains;return drain_status;}
static int cudaFree(void *){++frees;return free_status;}
static void fak_alloc_untrack(void *){++untracks;}
#include "prefill_input_owner.cuh"
static void reset(){
 g_prefill_input=nullptr;g_prefill_input_bytes=0;g_prefill_input_failure=0;
 g_stream=reinterpret_cast<void *>(1);g_capture_open=uncertain=pending=ordinary_fail=all_fail=false;
 g_cuda_cleanup_status=device=device_status=copy_status=drain_status=free_status=0;
 allocs=copies=drains=frees=untracks=0;g_panel_live_bytes=0;g_live.clear();g_managed_live.clear();g_panel_live.clear();
}
int main(){
 int host=1;
 for(int variant=0;variant<3;++variant){
  reset();ordinary_fail=variant==1;
  for(int cycle=0;cycle<100;++cycle){
   void *owned=nullptr;int managed=-1;host=cycle;
   assert(fcuda_prefill_input_acquire(&host,sizeof(host),variant==2,&owned,&managed)==0);
   assert(owned==storage&&g_prefill_input==owned&&managed==(variant==1));
   assert(memcmp(storage,&host,sizeof(host))==0);
   int before=allocs;void *other=nullptr;int m;
   assert(fcuda_prefill_input_acquire(&host,sizeof(host),variant==2,&other,&m)!=0&&other==nullptr&&allocs==before);
   assert(fcuda_prefill_input_retire(reinterpret_cast<void *>(2),sizeof(host))!=0);
   assert(g_prefill_input==owned&&frees==cycle);
   assert(fcuda_prefill_input_retire(owned,sizeof(host))==0);
   assert(g_prefill_input==nullptr&&g_live.empty()&&g_managed_live.empty()&&g_panel_live.empty()&&g_panel_live_bytes==0);
  }
  assert(copies==100&&drains==100&&frees==100&&untracks==100);
  for(int fail=0;fail<5;++fail){
   reset();ordinary_fail=variant==1;void *owned=nullptr;int managed;
   if(fail==0)copy_status=700;
   int acquired=fcuda_prefill_input_acquire(&host,sizeof(host),variant==2,&owned,&managed);
   assert(owned==storage);
   if(fail==0)assert(acquired==700);else assert(acquired==0);
   if(fail==1)drain_status=719;
   if(fail==2)free_status=700;
   if(fail==3)fcuda_prefill_input_failed(1719);
   if(fail==4)g_stream=nullptr;
   int first=fcuda_prefill_input_retire(owned,sizeof(host));assert(first!=0);
   assert(g_prefill_input==owned&&untracks==0&&g_cuda_cleanup_status!=0);
   int oldallocs=allocs,olddrains=drains,oldfrees=frees;
   copy_status=drain_status=free_status=0;g_stream=reinterpret_cast<void *>(1);
   for(int repeat=0;repeat<1000;++repeat){
    void *p=nullptr;int m;
    assert(fcuda_prefill_input_acquire(&host,sizeof(host),variant==2,&p,&m)!=0);
    assert(p==nullptr&&allocs==oldallocs);
    assert(fcuda_prefill_input_retire(owned,sizeof(host))==first);
   }
   assert(drains==olddrains&&frees==oldfrees&&untracks==0);
  }
 }
 for(int admission=0;admission<5;++admission){
  reset();if(admission==0)g_capture_open=true;if(admission==1)uncertain=true;if(admission==2)g_stream=nullptr;if(admission==3)device=1;if(admission==4)all_fail=true;
  void *p=nullptr;int m;assert(fcuda_prefill_input_acquire(&host,sizeof(host),0,&p,&m)!=0);
  assert(p==nullptr&&g_prefill_input==nullptr&&copies==0&&frees==0);
  if(admission!=4)assert(allocs==0);
 }
}
`

// Follow the actual upload source split; keep each existing semantic assertion.
func cudaUploadContractBodies(t *testing.T) map[string]string {
	t.Helper()
	bodies := map[string]string{}
	for _, name := range []string{"cuda.go", "cuda_upload.go"} {
		raw, err := os.ReadFile(name)
		if err != nil {
			t.Fatal(err)
		}
		fs := token.NewFileSet()
		f, err := parser.ParseFile(fs, name, raw, 0)
		if err != nil {
			t.Fatal(err)
		}
		for _, d := range f.Decls {
			fn, ok := d.(*ast.FuncDecl)
			if ok && fn.Body != nil {
				bodies[fn.Name.Name] = string(raw[fs.Position(fn.Body.Pos()).Offset:fs.Position(fn.Body.End()).Offset])
			}
		}
	}
	return bodies
}

// fak-test:runtime fast est=10ms lane=default
func TestCUDAPrefillOwnedInputRouteContract(t *testing.T) {
	f, err := parser.ParseFile(token.NewFileSet(), "cuda.go", nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	var route *ast.IfStmt
	for _, d := range f.Decls {
		fn, ok := d.(*ast.FuncDecl)
		if !ok || fn.Name.Name != "PrefillBatch" {
			continue
		}
		ast.Inspect(fn.Body, func(n ast.Node) bool {
			if branch, ok := n.(*ast.IfStmt); ok {
				if id, ok := branch.Cond.(*ast.Ident); ok && id.Name == "isHostX" {
					route = branch
				}
			}
			return true
		})
	}
	if route == nil || route.Else == nil {
		t.Fatal("missing distinct host/device prefill routes")
	}
	calls := func(n ast.Node) map[string]int {
		counts := map[string]int{}
		ast.Inspect(n, func(n ast.Node) bool {
			call, ok := n.(*ast.CallExpr)
			if !ok {
				return true
			}
			sel, ok := call.Fun.(*ast.SelectorExpr)
			if !ok {
				return true
			}
			if id, ok := sel.X.(*ast.Ident); ok && id.Name == "c" {
				counts[sel.Sel.Name]++
			}
			return true
		})
		return counts
	}
	host, device := calls(route.Body), calls(route.Else)
	if host["prefillHostProjections"] != 1 || host["Upload"] != 0 || host["BatchedMatMul"] != 0 || host["Read"] != 0 {
		t.Fatal("host input escaped owned projection route")
	}
	if device["Upload"] != 3 || device["BatchedMatMul"] != 3 || device["Read"] != 3 {
		t.Fatal("device input route changed")
	}
}
