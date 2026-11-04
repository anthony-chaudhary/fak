package compute

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

// Compile the actual transfer bodies with injected CUDA runtime results. This
// witnesses host status/accounting control flow, not CUDA linkage or execution.
// fak-test:runtime medium est=3s lane=default
func TestCUDAH2DStatusContract(t *testing.T) {
	read := func(path string) string {
		t.Helper()
		b, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		return string(b)
	}
	header := read("cuda_backend.h")
	memory := read("cuda/runtime_memory.cuh")
	body := func(source, signature string) string {
		t.Helper()
		start := strings.Index(source, signature)
		if start < 0 {
			t.Fatalf("missing production signature %q", signature)
		}
		depth := 0
		for i := start; i < len(source); i++ {
			switch source[i] {
			case '{':
				depth++
			case '}':
				depth--
				if depth == 0 {
					return source[start : i+1]
				}
			}
		}
		t.Fatalf("unclosed production body %q", signature)
		return ""
	}
	definitions := body(memory, `extern "C" int fcuda_h2d_checked(`) + "\n" +
		body(memory, `extern "C" void fcuda_h2d(`)
	compiler, err := exec.LookPath("c++")
	if err != nil {
		t.Fatal("host C++ compiler required for checked H2D regression")
	}
	dir := t.TempDir()
	for name, source := range map[string]string{
		"cuda_backend.h": header,
		"h2d.cpp": `#include <initializer_list>
#include <cstdio>
#include <cstddef>
#include <cstdlib>
#include "cuda_backend.h"
static int (*checked)(void *, const void *, size_t) = &fcuda_h2d_checked;
static void (*legacy)(void *, const void *, size_t) = &fcuda_h2d;
using cudaError_t=int;
constexpr int cudaSuccess=0;
constexpr int cudaMemcpyHostToDevice=1;
static size_t g_h2d_bytes=0;
static int copy_status=0, copies=0, timestamps=0;
static void *expected_dst=nullptr;
static const void *expected_src=nullptr;
static size_t expected_bytes=0;
void fcuda_ts_untimed_transfer(){++timestamps;}
const char *cudaGetErrorString(int){return "injected failure";}
int cudaMemcpy(void *d,const void *h,size_t n,int direction){
 if(d!=expected_dst||h!=expected_src||n!=expected_bytes||direction!=cudaMemcpyHostToDevice) std::abort();
 ++copies;return copy_status;
}
void require(bool condition){if(!condition)std::abort();}
` + definitions + `
int main(){
 char dst[16]={},src[16]={};expected_dst=dst;expected_src=src;
 for(int status: {0,1,700,719}){
  for(size_t n: {size_t(0),size_t(16)}){
   copy_status=status;expected_bytes=n;size_t before=g_h2d_bytes;int count=copies,ts=timestamps;
   require(checked(dst,src,n)==status);
   require(copies==count+1&&timestamps==ts+1);
   require(g_h2d_bytes==before+(status==0?n:0));
   before=g_h2d_bytes;count=copies;ts=timestamps;
   legacy(dst,src,n);
   require(copies==count+1&&timestamps==ts+1);
   require(g_h2d_bytes==before+(status==0?n:0));
  }
 }
 std::puts("PASS checked H2D exact status, arguments, copy/timestamp counts, success-only bytes, legacy wrapper");
}
`,
	} {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(source), 0600); err != nil {
			t.Fatal(err)
		}
	}
	name := "h2d-status"
	if runtime.GOOS == "windows" {
		name += ".exe"
	}
	binary := filepath.Join(dir, name)
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	args := []string{"-std=c++17", "-Wall", "-Wextra", "-Werror", "-I", dir, filepath.Join(dir, "h2d.cpp"), "-o", binary}
	if out, err := exec.CommandContext(ctx, compiler, args...).CombinedOutput(); err != nil {
		t.Fatalf("compile production checked H2D bodies: %v\n%s", err, out)
	}
	if out, err := exec.CommandContext(ctx, binary).CombinedOutput(); err != nil {
		t.Fatalf("production checked H2D contract: %v\n%s", err, out)
	}
}
