package compute

import (
	"context"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// Literal native transfer and Go retirement bodies, with host-only API stubs.
// This does not establish CUDA/cgo compilation or hardware qualification.
// fak-test:runtime medium est=2s lane=default
func TestCUDAF32UploadFailurePublication(t *testing.T) {
	t.Run("go-ownership-and-publication", func(t *testing.T) {
		raw, err := os.ReadFile("cuda.go")
		if err != nil {
			t.Fatal(err)
		}
		fs := token.NewFileSet()
		file, err := parser.ParseFile(fs, "cuda.go", raw, 0)
		if err != nil {
			t.Fatal(err)
		}
		functions := map[string]string{}
		for _, decl := range file.Decls {
			if fn, ok := decl.(*ast.FuncDecl); ok {
				functions[fn.Name.Name] = string(raw[fs.Position(fn.Pos()).Offset:fs.Position(fn.End()).Offset])
			}
		}
		body := functions["uploadClass"]
		if strings.Count(body, "C.fak_cuda_upload_h2d(") != 2 || strings.Count(body, "c.checkF32Upload(buf, status, site,") != 2 || strings.Contains(body, "C.fcuda_h2d(") {
			t.Error("both F32 consumers must check native status")
		}
		if !strings.Contains(string(raw), "int fak_cuda_upload_h2d(void *d, const void *h, size_t n);") {
			t.Error("missing private checked ABI declaration")
		}
		check := strings.LastIndex(body, "c.checkF32Upload(")
		if check < 0 || check > strings.Index(body, "c.accountImmutableWeightUpload(") || check > strings.Index(body, "uploadCache[ucKey{hp, F32, t.Layout}] = out") {
			t.Error("weight publication must follow checked transfer")
		}
		classStart := strings.Index(body, "if class != MemoryWeights")
		classEnd := strings.Index(body, "store := F32")
		classBody := body[classStart:classEnd]
		if !strings.Contains(classBody, "return finishF32Upload(out, f, func(values []float32)") || !strings.Contains(classBody, "c.checkF32Upload(buf, status, site,") {
			t.Error("classed upload must check inside nonempty copy callback")
		}
		helper := functions["checkF32Upload"]
		if helper == "" {
			t.Fatal("missing failed-upload ownership boundary")
		}
		for _, bad := range []string{"cudaMu.Lock", ".Free(", "uploadCache", "accountImmutableWeightUpload"} {
			if strings.Contains(helper, bad) {
				t.Fatalf("retirement contains unsafe %s", bad)
			}
		}
		dir := t.TempDir()
		path := filepath.Join(dir, "ownership.go")
		if err := os.WriteFile(path, []byte(cudaUploadGoPrefix+helper+cudaUploadGoMain), 0600); err != nil {
			t.Fatal(err)
		}
		ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
		defer cancel()
		if out, err := exec.CommandContext(ctx, "go", "run", path).CombinedOutput(); err != nil {
			t.Fatalf("literal Go ownership: %v\n%s", err, out)
		}
	})
	t.Run("native-status-and-counter", func(t *testing.T) {
		compiler, err := exec.LookPath("c++")
		if err != nil {
			t.Skip("host C++ compiler unavailable")
		}
		raw, err := os.ReadFile("cuda_kernels.cu")
		if err != nil {
			t.Fatal(err)
		}
		source := string(raw)
		start := strings.Index(source, `extern "C" int fak_cuda_upload_h2d(`)
		checked := start >= 0
		if !checked {
			start = strings.Index(source, `extern "C" void fcuda_h2d(`)
		}
		end := strings.Index(source, `extern "C" void fcuda_d2h(`)
		if start < 0 || end < start {
			t.Fatal("missing native transfer boundary")
		}
		body := source[start:end]
		if !checked {
			body += "\nstatic int fak_cuda_upload_h2d(void *d,const void *h,size_t n) { fcuda_h2d(d,h,n); return 0; }\n"
		}
		dir := t.TempDir()
		cpp, bin := filepath.Join(dir, "transfer.cpp"), filepath.Join(dir, "transfer")
		if err := os.WriteFile(cpp, []byte(cudaUploadNativePrefix+body+cudaUploadNativeMain), 0600); err != nil {
			t.Fatal(err)
		}
		ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		defer cancel()
		if out, err := exec.CommandContext(ctx, compiler, "-std=c++11", "-Wall", "-Wextra", "-Werror", cpp, "-o", bin).CombinedOutput(); err != nil {
			t.Fatalf("native host compile: %v\n%s", err, out)
		}
		if out, err := exec.CommandContext(ctx, bin).CombinedOutput(); err != nil {
			t.Fatalf("native transfer: %v\n%s", err, out)
		}
	})
}

const cudaUploadGoPrefix = `package main
import "unsafe"
type cudaBuf struct { ptr unsafe.Pointer; budgetedWeightBytes int64; managedWeight bool; class string }
type cudaBackend struct { dlUsed int64; managedN int; faultLatch *latch }
type latch struct { observed int }
func (l *latch) Observe(_ int, _ string, status int) { if l != nil { l.observed = status } }
const DeviceFaultUnknown = 0
type CUDAOpError struct { Op, Site, Msg, Class string; Code int }
func require(ok bool) { if !ok { panic("ownership assertion failed") } }
`

const cudaUploadGoMain = `
func main() {
 for _, status := range []int{0, 77} { for _, withLatch := range []bool{false,true} { for _, managed := range []bool{false,true} { for _, charged := range []bool{false,true} { for _, cleanupPanic := range []bool{false,true} {
  storage := new(byte); ptr := unsafe.Pointer(storage)
  dst := &cudaBuf{ptr:ptr, class:"weight", managedWeight:managed}
  c := &cudaBackend{dlUsed:31, managedN:2}
  if charged { dst.budgetedWeightBytes=16; c.dlUsed+=16 }
  if managed { c.managedN++ }
  if withLatch { c.faultLatch=&latch{} }
  beforeUsed, beforeManaged := c.dlUsed, c.managedN
  releases, published := 0, false
  var caught any
  func() { defer func(){ caught=recover() }()
   c.checkF32Upload(dst,status,"test-upload",func(p unsafe.Pointer) {
    releases++; require(p==ptr && dst.ptr==nil && dst.budgetedWeightBytes==0 && !dst.managedWeight && c.dlUsed==31 && c.managedN==2)
    if cleanupPanic { panic("secondary release failure") }
   })
   published=true
  }()
  if status==0 { require(caught==nil && published && releases==0 && dst.ptr==ptr && c.dlUsed==beforeUsed && c.managedN==beforeManaged) } else {
   err,ok := caught.(*CUDAOpError)
   require(ok && err.Op=="Upload" && err.Site=="test-upload" && err.Code==77 && err.Class=="weight" && !published && releases==1 && dst.ptr==nil && c.dlUsed==31 && c.managedN==2)
  }
  if withLatch { require(c.faultLatch.observed==status) }
 } } } } }
}
`

const cudaUploadNativePrefix = `
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
using cudaError_t = int;
enum { cudaSuccess=0, cudaMemcpyHostToDevice=1 };
static size_t g_h2d_bytes=0;
static int fail=0, calls=0;
static void require(bool ok) { if(!ok) { fprintf(stderr,"transfer assertion failed\n"); exit(1); } }
static int cudaMemcpy(void *d,const void *h,size_t n,int kind) { ++calls; require(kind==cudaMemcpyHostToDevice); if(fail) return fail; memcpy(d,h,n); return 0; }
static const char *cudaGetErrorString(int) { return "injected transfer failure"; }
`

const cudaUploadNativeMain = `
int main() {
 int source=9, destination=-7;
 fail=77;
 require(fak_cuda_upload_h2d(&destination,&source,sizeof(source))==77);
 require(destination==-7 && g_h2d_bytes==0 && calls==1);
 fail=0;
 require(fak_cuda_upload_h2d(&destination,&source,sizeof(source))==0);
 require(destination==9 && g_h2d_bytes==sizeof(source) && calls==2);
 destination=-7; fail=77; fcuda_h2d(&destination,&source,sizeof(source));
 require(destination==-7 && g_h2d_bytes==sizeof(source) && calls==3);
 fail=0; fcuda_h2d(&destination,&source,sizeof(source));
 require(destination==9 && g_h2d_bytes==2*sizeof(source) && calls==4);
}
`
