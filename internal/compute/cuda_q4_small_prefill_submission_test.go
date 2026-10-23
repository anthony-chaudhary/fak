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

// Literal host submission bodies, with only CUDA launches stubbed. This checks
// immediate failures, not CUDA compilation, device completion, or numeric quality.
// fak-test:runtime medium est=1s lane=default
func TestCUDAQ4SmallPrefillSubmission(t *testing.T) {
	t.Parallel()
	t.Run("typed-consumers", func(t *testing.T) {
		for _, target := range []struct{ file, method string }{{"cuda.go", "MatMul"}, {"cuda.go", "BatchedMatMul"}, {"cuda_kernels.go", "qwen35SequenceMatMulLocked"}} {
			raw, err := os.ReadFile(target.file)
			if err != nil {
				t.Fatal(err)
			}
			fs := token.NewFileSet()
			parsed, err := parser.ParseFile(fs, target.file, raw, 0)
			if err != nil {
				t.Fatal(err)
			}
			text := func(n ast.Node) string { return string(raw[fs.Position(n.Pos()).Offset:fs.Position(n.End()).Offset]) }
			var body *ast.BlockStmt
			for _, d := range parsed.Decls {
				if f, ok := d.(*ast.FuncDecl); ok && f.Recv != nil && f.Name.Name == target.method {
					body = f.Body
				}
			}
			if body == nil {
				t.Fatal("missing consumer", target.method)
			}
			for _, kind := range []string{"q4k"} {
				if strings.Count(text(body), "status = int(C.fak_cuda_"+kind+"_matmul_f32(") != 1 {
					t.Errorf("%s must consume checked %s status", target.method, kind)
				}
				if strings.Contains(text(body), "C.fcuda_"+kind+"_matmul_f32(") {
					t.Error("unchecked quantized consumer", target.method)
				}
			}
			var failure *ast.IfStmt
			for _, s := range body.List {
				if branch, ok := s.(*ast.IfStmt); ok && text(branch.Cond) == "status != 0" {
					failure = branch
				}
			}
			if failure == nil || failure.Else != nil {
				t.Fatal("missing unconditional status refusal")
			}
			if !strings.Contains(text(failure), "c.faultLatch.Observe(DeviceFaultUnknown,") {
				t.Error("missing fault latch")
			}
			last := text(failure.Body.List[len(failure.Body.List)-1])
			if !strings.Contains(last, "Code: status") || (target.method != "qwen35SequenceMatMulLocked" && !strings.HasPrefix(last, "panic(&CUDAOpError{")) || (target.method == "qwen35SequenceMatMulLocked" && !strings.HasPrefix(last, "return Tensor{}, nil, &Qwen35SequenceError{")) {
				t.Error("missing typed refusal")
			}
			if strings.Index(text(body), "if status != 0") > strings.Index(text(body), "return y") || strings.Contains(text(body), "fenceGen") {
				t.Error("failed submission published")
			}
		}
	})
	t.Run("literal-native-failures", func(t *testing.T) {
		compiler, err := exec.LookPath("c++")
		if err != nil {
			t.Skip("host C++ compiler unavailable")
		}
		raw, err := os.ReadFile("cuda_kernels.cu")
		if err != nil {
			t.Fatal(err)
		}
		source := string(raw)
		start := strings.Index(source, "struct q4_coeff_cache_entry {")
		if start < 0 {
			t.Fatal("missing Q4 coefficient cache")
		}
		end := strings.Index(source[start:], "// k_q2_0_gemm:")
		if end < 0 {
			t.Fatal("missing Q4 boundary")
		}
		body := source[start : start+end]
		for _, replacement := range [][2]string{
			{"k_q4k_coeffs<<<(n + 255) / 256, 256, 0, g_stream>>>", "mock_coeffs"},
			{"k_q4k_dequant_transient<<<((size_t)out * in + 255) / 256, 256, 0, g_stream>>>", "mock_dequant"},
			{"k_q4k_gemm<<<dim3((out + 1) / 2, P), 256, 0, g_stream>>>", "mock_gemm"},
			{"k_q4k_gemm_panel<token_tile><<<dim3(out, (P + token_tile - 1) / token_tile), 32, 0, g_stream>>>", "mock_panel<token_tile>"},
		} {
			if strings.Count(body, replacement[0]) != 1 {
				t.Fatal("launch changed; update faithful substitution")
			}
			body = strings.Replace(body, replacement[0], replacement[1], 1)
		}
		dir := t.TempDir()
		cpp, bin := filepath.Join(dir, "q4k.cpp"), filepath.Join(dir, "q4k")
		if err := os.WriteFile(cpp, []byte(cudaQ4SmallPrefillPrefix+body+cudaQ4SmallPrefillMain), 0600); err != nil {
			t.Fatal(err)
		}
		ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		defer cancel()
		if out, err := exec.CommandContext(ctx, compiler, "-std=c++11", "-Wall", "-Wextra", "-Werror", cpp, "-o", bin).CombinedOutput(); err != nil {
			t.Fatalf("compile literal Q4 body: %v\n%s", err, out)
		}
		if out, err := exec.CommandContext(ctx, bin).CombinedOutput(); err != nil {
			t.Fatalf("Q4 submission: %v\n%s", err, out)
		}
	})
}

const cudaQ4SmallPrefillPrefix = `
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <initializer_list>
using cudaError_t = int;
struct float2 { float x, y; };
enum { cudaSuccess=0 };
static int launches, checks, failAt, allocs, frees, live;
static void *owned;
static void require(bool ok, const char *why) { if (!ok) { fprintf(stderr,"%s\n",why); exit(1); } }
static int cudaGetDevice(int *device) { *device=0; return 0; }
template<typename T> static int cudaMalloc(T **p, size_t n) { require(!owned,"duplicate coefficient allocation"); *p=(T*)malloc(n); require(*p!=nullptr,"host allocation"); owned=*p; ++allocs; ++live; return 0; }
static int cudaFree(void *p) { require(p && p==owned,"foreign or duplicate free"); free(p); owned=nullptr; ++frees; --live; return 0; }
#define CK(expr) do { int e=(expr); if(e) fprintf(stderr,"CUDA error %d\n",e); } while(0)
static inline int cudaGetLastError() { ++checks; return launches==failAt ? 77 : 0; }
static void *fcuda_malloc(size_t) { require(false,"large prefill allocation entered"); return nullptr; }
static void fcuda_free(void *) { require(false,"large prefill free entered"); }
static int fp16_compensated_matmul_checked(const float *,const float *,float *,int,int,int) { require(false,"large prefill GEMM entered"); return 0; }
static void mock_dequant(const unsigned char *,float *,int,int) { require(false,"large prefill dequant entered"); }
static void mock_coeffs(const uint8_t *q,float2 *coeff,int out,int in) { require(q && coeff==owned && out==2 && in==256,"coefficient operands changed"); require(launches==0 && checks==0,"coefficient ordering"); ++launches; }
static void mock_gemm(const unsigned char *q,const float2 *coeff,const float *x,float *y,int out,int in,int p) { require(q && coeff==owned && x && y && out==2 && in==256 && p>0 && p<4,"GEMV operands changed"); require(checks==launches,"GEMV submitted before coefficient status"); ++launches; }
template<int Tile> static void mock_panel(const unsigned char *q,const float *x,float *y,int out,int in,int p) { require(Tile==4 && q && x && y && out==2 && in==256 && p>=4 && p<128,"panel operands changed"); require(!owned && launches==0,"panel uses coefficient cache"); ++launches; }
`
const cudaQ4SmallPrefillMain = `
int main() {
 const uint8_t q[288]={}; const float x[32512]={}; float y[254]={};
 for(int p:{1,3}) {
  // A rejected coefficient launch must not poison the cache or submit GEMV.
  launches=checks=0; failAt=1;
  int a=allocs, f=frees;
  require(fak_cuda_q4k_matmul_f32(q,x,y,2,256,p)==20077,"coefficient launch failure returned success");
  require(launches==1 && checks==1 && g_q4_coeff_cache_n==0 && live==0 && allocs==a+1 && frees==f+1,"failed coefficient cached, leaked, or consumed");
  // Retry must rebuild; a later GEMV failure may retain the valid coefficient.
  launches=checks=0; failAt=2;
  require(fak_cuda_q4k_matmul_f32(q,x,y,2,256,p)==20077,"cold GEMV launch failure returned success");
  require(launches==2 && checks==2 && g_q4_coeff_cache_n==1 && live==1 && allocs==a+2 && frees==f+1,"retry did not rebuild valid coefficient");
  for(int fault=0;fault<=1;++fault) {
   launches=checks=0; failAt=fault;
   require(fak_cuda_q4k_matmul_f32(q,x,y,2,256,p)==(fault?20077:0),"warm GEMV status lost");
   require(launches==1 && checks==1 && allocs==a+2 && frees==f+1 && g_q4_coeff_cache_n==1,"warm coefficient cache changed");
  }
  q4_coeff_cache_evict(q);
  require(live==0 && g_q4_coeff_cache_n==0,"eviction ownership changed");
  launches=checks=0; failAt=0;
  require(fak_cuda_q4k_matmul_f32(q,x,y,2,256,p)==0 && launches==2 && checks==2,"cold success changed");
  q4_coeff_cache_evict(q);
 }
 for(int p:{4,127}) for(int fault=0;fault<=1;++fault) {
  launches=checks=0; failAt=fault; int a=allocs;
  require(fak_cuda_q4k_matmul_f32(q,x,y,2,256,p)==(fault?20077:0),"panel launch status lost");
  require(launches==1 && checks==1 && allocs==a && live==0 && g_q4_coeff_cache_n==0,"panel cache or launch changed");
 }
}
`
