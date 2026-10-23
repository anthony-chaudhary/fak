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
func TestCUDAQuantizedPrefillSubmission(t *testing.T) {
	t.Parallel()
	t.Run("typed-consumers", func(t *testing.T) {
		for _, target := range []struct{ file, method string }{{"cuda.go", "BatchedMatMul"}, {"cuda_kernels.go", "qwen35SequenceMatMulLocked"}} {
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
			for _, kind := range []string{"q8", "q4k"} {
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
			if !strings.Contains(last, "Code: status") || (target.method == "BatchedMatMul" && !strings.HasPrefix(last, "panic(&CUDAOpError{")) || (target.method != "BatchedMatMul" && !strings.HasPrefix(last, "return Tensor{}, nil, &Qwen35SequenceError{")) {
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
		section := func(start, end string) string {
			at := strings.Index(source, start)
			if at < 0 {
				t.Fatal("missing", start)
			}
			stop := strings.Index(source[at:], end)
			if stop < 0 {
				t.Fatal("missing", end)
			}
			return source[at : at+stop]
		}
		checked := strings.Contains(source, `extern "C" int fak_cuda_q8_matmul_f32(`)
		q8, q4 := `extern "C" void fcuda_q8_matmul_f32(`, `extern "C" void fcuda_q4k_matmul_f32(`
		tail := "}\n"
		helperType, helperSuffix := "void", ""
		if checked {
			q8, q4, tail = `extern "C" int fak_cuda_q8_matmul_f32(`, `extern "C" int fak_cuda_q4k_matmul_f32(`, "  return 0;\n}\n"
			helperType, helperSuffix = "int", "_checked"
		}
		// Only the assigned P>=128 paths are executed; small-P implementations
		// stay outside this witness and retain their public void compatibility.
		body := section("static "+helperType+" fp16_compensated_matmul"+helperSuffix+"(", "__global__ void k_q8_dequant_transient") +
			section("static "+helperType+" q8_dequant_sgemm"+helperSuffix+"(", "__global__ void k_qwen35_gdn_fused_in_proj") +
			section(q8, "  int nblk = in / block;") + tail + section(q4, "  if (P < 4)") + tail
		for _, replacement := range [][2]string{
			{"k_fp16_split<<<(wn + 255) / 256, 256, 0, g_stream>>>(W, wHi, wLo, wn);", "mock_split(W, wHi, wLo, wn);"},
			{"k_fp16_split<<<(xn + 255) / 256, 256, 0, g_stream>>>(X, xHi, xLo, xn);", "mock_split(X, xHi, xLo, xn);"},
			{"k_q8_dequant_transient<<<((size_t)rows * cols + 255) / 256, 256, 0, g_stream>>>(\n      (const signed char *)codes, scales, wide, rows, cols, block);", "mock_dequant(codes, scales, wide, rows, cols, block);"},
			{"k_q4k_dequant_transient<<<((size_t)out * in + 255) / 256, 256, 0, g_stream>>>(\n        (const unsigned char *)dQ4K, wide, out, in);", "mock_dequant(dQ4K, dX, wide, out, in, 32);"},
		} {
			if strings.Count(body, replacement[0]) != 1 {
				t.Fatal("launch changed; update faithful substitution", replacement[0])
			}
			body = strings.Replace(body, replacement[0], replacement[1], 1)
		}
		prefix := cudaQuantizedPrefillPrefix
		if checked {
			prefix += "\n#define CHECKED_PREFILL 1\n"
		}
		dir := t.TempDir()
		cpp, bin := filepath.Join(dir, "prefill.cpp"), filepath.Join(dir, "prefill")
		if err := os.WriteFile(cpp, []byte(prefix+body+cudaQuantizedPrefillMain), 0600); err != nil {
			t.Fatal(err)
		}
		ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		defer cancel()
		if out, err := exec.CommandContext(ctx, compiler, "-std=c++11", "-Wall", "-Wextra", "-Werror", cpp, "-o", bin).CombinedOutput(); err != nil {
			t.Fatalf("compile literal host paths: %v\n%s", err, out)
		}
		if out, err := exec.CommandContext(ctx, bin).CombinedOutput(); err != nil {
			t.Fatalf("quantized prefill submission: %v\n%s", err, out)
		}
	})
}

const cudaQuantizedPrefillPrefix = `
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
using __half = unsigned short;
using cudaError_t = int;
using cublasStatus_t = int;
enum { cudaSuccess=0, CUBLAS_STATUS_SUCCESS=0, CUBLAS_OP_T=1, CUBLAS_OP_N=2, CUDA_R_16F=3, CUDA_R_32F=4, CUBLAS_COMPUTE_32F=5, CUBLAS_GEMM_DEFAULT_TENSOR_OP=6, g_blas=7 };
static int allocs, launches, gemms, live, failAlloc, failLaunch, failGemm;
static void *owned[5];
static void require(bool ok, const char *why) { if (!ok) { fprintf(stderr, "%s\n", why); exit(1); } }
static void *fcuda_malloc(size_t n) { ++allocs; if (allocs==failAlloc) return nullptr; void *p=malloc(n); require(p!=nullptr,"host allocation"); owned[allocs-1]=p; ++live; return p; }
static void fcuda_free(void *p) { require(p!=nullptr,"freed unacquired scratch"); bool found=false; for(void *&q:owned) if(q==p) { q=nullptr; found=true; } require(found,"double or foreign free"); free(p); --live; }
static void mock_split(const float *p, __half *hi, __half *lo, size_t n) { require(p && hi && lo && n,"invalid split scratch"); ++launches; }
static void mock_dequant(const void *p, const float *scale, float *wide, int out, int in, int block) { require(p && scale && wide && out==2 && in==256 && block==32,"invalid dequant scratch"); ++launches; }
static inline int cudaGetLastError() { return launches==failLaunch ? 77 : 0; }
static int cublasGemmEx(int handle,int opA,int opB,int out,int p,int in,const float *alpha,const void *a,int atype,int lda,const void *x,int xtype,int ldb,const float *beta,float *y,int ytype,int ldc,int compute,int algo) {
 ++gemms;
 require(a==owned[gemms==2?2:1] && x==owned[gemms==3?4:3],"compensation operands changed");
 require(handle==g_blas && opA==CUBLAS_OP_T && opB==CUBLAS_OP_N && out==2 && p==128 && in==256 && *alpha==1 && *beta==(gemms==1?0:1) && a && x && atype==CUDA_R_16F && xtype==CUDA_R_16F && ytype==CUDA_R_32F && lda==256 && ldb==256 && ldc==2 && compute==CUBLAS_COMPUTE_32F && algo==CUBLAS_GEMM_DEFAULT_TENSOR_OP,"GEMM recipe changed");
 if(gemms==failGemm) return 13;
 for(int i=0;i<256;++i) y[i]=*beta*y[i]+9;
 return 0;
}
`

const cudaQuantizedPrefillMain = `
#ifndef CHECKED_PREFILL
// Old Go callers consume no status from the void ABI.
static int fak_cuda_q8_matmul_f32(const int8_t *w,const float *s,const float *x,float *y,int out,int in,int p,int b) { fcuda_q8_matmul_f32(w,s,x,y,out,in,p,b); return 0; }
static int fak_cuda_q4k_matmul_f32(const uint8_t *w,const float *x,float *y,int out,int in,int p) { fcuda_q4k_matmul_f32(w,x,y,out,in,p); return 0; }
#endif
int main() {
 const int8_t w[512]={}; const float x[32768]={}, scales[16]={}; float y[256];
 for(int q4=0;q4<2;++q4) {
  // First case witnesses the original ignored primary cuBLAS failure.
  for(int fault=0;fault<12;++fault) {
   allocs=launches=gemms=live=0; failAlloc=failLaunch=failGemm=0;
   if(fault==0) failGemm=1;
   if(fault>=1 && fault<=5) failAlloc=fault;
   if(fault>=6 && fault<=8) failLaunch=fault-5;
   if(fault==9 || fault==10) failGemm=fault-7;
   for(float &v:y) v=-7;
   int status=q4 ? fak_cuda_q4k_matmul_f32((const uint8_t *)w,x,y,2,256,128) : fak_cuda_q8_matmul_f32(w,scales,x,y,2,256,128,32);
   require(status==(failAlloc?-1:failLaunch?20077:failGemm?10013:0),"failed submission returned usable output");
   require(live==0,"scratch leaked");
   require(gemms==(failAlloc || failLaunch ? 0:failGemm?failGemm:3),"GEMM submitted after failure");
   require(launches==(failAlloc ? (failAlloc==1?0:1):failLaunch?failLaunch:3),"launch submitted after failure");
   if(!status) for(float v:y) require(v==27,"success accumulation changed");
  }
 }
}
`
