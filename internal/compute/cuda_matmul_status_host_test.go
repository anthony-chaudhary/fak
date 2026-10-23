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

// Executes the literal cuBLAS host bodies with deterministic API stubs and only
// the conversion launch syntax replaced. Go AST checks pin all three consumers.
// This is not CUDA/cgo compilation, asynchronous completion, or a GPU witness.
// fak-test:runtime medium est=1s lane=default
func TestCUDAMatMulCheckedStatusBoundary(t *testing.T) {
	t.Parallel()
	t.Run("go-refusal-before-result", func(t *testing.T) {
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
			for _, decl := range parsed.Decls {
				if fn, ok := decl.(*ast.FuncDecl); ok && fn.Recv != nil && fn.Name.Name == target.method {
					body = fn.Body
				}
			}
			if body == nil {
				t.Fatalf("missing %s", target.method)
			}
			source := text(body)
			if strings.Count(source, "status = int(C.fak_cuda_matmul_f32(") != 1 || strings.Count(source, "status = int(C.fak_cuda_matmul_f16(") != 1 || strings.Contains(source, "C.fcuda_matmul_") {
				t.Errorf("%s must consume checked F32/F16 status", target.method)
			}
			if strings.Contains(source, "fenceGen") {
				t.Errorf("%s must not publish an asynchronous completion fence", target.method)
			}
			var failure *ast.IfStmt
			for _, stmt := range body.List {
				if branch, ok := stmt.(*ast.IfStmt); ok && text(branch.Cond) == "status != 0" {
					failure = branch
				}
			}
			if failure == nil || failure.Else != nil {
				t.Errorf("%s missing unconditional status refusal", target.method)
				continue
			}
			if !strings.Contains(text(failure), "c.faultLatch.Observe(DeviceFaultUnknown,") {
				t.Errorf("%s failure must poison latch", target.method)
			}
			last := text(failure.Body.List[len(failure.Body.List)-1])
			if target.method == "qwen35SequenceMatMulLocked" {
				if !strings.HasPrefix(last, "return Tensor{}, nil, &Qwen35SequenceError{") || !strings.Contains(last, "Code: status") {
					t.Error("sequence must return no tensor and a typed status cause")
				}
			} else if !strings.HasPrefix(last, "panic(&CUDAOpError{") || !strings.Contains(last, "Code: status") {
				t.Errorf("%s must panic typed even with a nil latch", target.method)
			}
			if strings.Index(source, "if status != 0") > strings.Index(source, "return y") {
				t.Errorf("%s publishes failed result", target.method)
			}
		}
	})
	t.Run("literal-native-failure", func(t *testing.T) {
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
				t.Fatalf("missing %s", start)
			}
			stop := strings.Index(source[at:], end)
			if stop < 0 {
				t.Fatalf("missing %s", end)
			}
			return source[at : at+stop]
		}
		body := section("// y[P,out] = x[P,in]", "// ---- fp16 compute path") + section("// fcuda_matmul_f16: Y[P,out]", "// ---- native quantized device GEMM")
		launch := "k_f32_to_f16<<<(P * in + 255) / 256, 256, 0, g_stream>>>(dX, dXh, P * in);"
		if strings.Count(body, launch) != 1 {
			t.Fatal("conversion launch changed; update faithful host substitution")
		}
		body = strings.Replace(body, launch, "mock_convert(dX, dXh, P * in);", 1)
		prefix := cudaMatMulStatusPrefix
		if strings.Contains(body, `extern "C" int fak_cuda_matmul_f32(`) {
			prefix += "\n#define CHECKED_MATMUL 1\n"
		}
		dir := t.TempDir()
		cpp, bin := filepath.Join(dir, "matmul.cpp"), filepath.Join(dir, "matmul")
		if err := os.WriteFile(cpp, []byte(prefix+body+cudaMatMulStatusMain), 0600); err != nil {
			t.Fatal(err)
		}
		ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		defer cancel()
		if out, err := exec.CommandContext(ctx, compiler, "-std=c++11", "-Wall", "-Wextra", "-Werror", cpp, "-o", bin).CombinedOutput(); err != nil {
			t.Fatalf("compile literal matmul host bodies: %v\n%s", err, out)
		}
		if out, err := exec.CommandContext(ctx, bin).CombinedOutput(); err != nil {
			t.Fatalf("matmul failure propagation: %v\n%s", err, out)
		}
	})
}

const cudaMatMulStatusPrefix = `
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
using __half = unsigned short;
using cublasStatus_t = int;
using cudaError_t = int;
enum { CUBLAS_STATUS_SUCCESS=0, CUBLAS_OP_T=1, CUBLAS_OP_N=2, CUDA_R_16F=3, CUDA_R_32F=4, CUBLAS_COMPUTE_32F=5, CUBLAS_GEMM_DEFAULT_TENSOR_OP=6, cudaSuccess=0, g_blas=7, g_stream=8 };
static int fail, gemms, converts, frees, allocated;
static __half scratch[6];
static void require(bool ok, const char *why) { if (!ok) { fprintf(stderr, "%s\n", why); exit(1); } }
static inline void *fcuda_malloc(size_t bytes) { require(bytes==sizeof(scratch), "scratch size changed"); if (fail==1) return nullptr; ++allocated; return scratch; }
static inline void fcuda_free(void *p) { require(p==scratch && allocated==1, "scratch ownership lost"); ++frees; --allocated; }
static inline void mock_convert(const float *x, __half *h, int n) { require(x && h==scratch && n==6, "conversion submitted invalid scratch"); ++converts; }
static inline int cudaGetLastError() { return fail==2 ? 77 : 0; }
static inline int cudaMemsetAsync(void *out, int value, size_t bytes, int stream) { require(stream==g_stream, "wrong stream"); memset(out,value,bytes); return 0; }
static inline int cublasSgemm(int handle,int opA,int opB,int out,int p,int in,const float *alpha,const float *a,int lda,const float *x,int ldb,const float *beta,float *y,int ldc) {
 require(handle==g_blas && opA==CUBLAS_OP_T && opB==CUBLAS_OP_N && out==2 && p==2 && in==3 && *alpha==1 && *beta==0 && a && x && lda==3 && ldb==3 && ldc==2, "F32 GEMM recipe changed"); ++gemms;
 if(fail==3) { return 13; }
 for(int i=0;i<4;++i) y[i]=9;
 return 0;
}
static inline int cublasGemmEx(int handle,int opA,int opB,int out,int p,int in,const float *alpha,const void *a,int atype,int lda,const void *x,int xtype,int ldb,const float *beta,float *y,int ytype,int ldc,int compute,int algo) {
 require(handle==g_blas && (opA==CUBLAS_OP_T || opA==CUBLAS_OP_N) && opB==CUBLAS_OP_N && out==2 && p==2 && in==3 && *alpha==1 && *beta==0 && a && x==scratch && atype==CUDA_R_16F && xtype==CUDA_R_16F && ytype==CUDA_R_32F && lda==(opA==CUBLAS_OP_T?3:2) && ldb==3 && ldc==2 && compute==CUBLAS_COMPUTE_32F && algo==CUBLAS_GEMM_DEFAULT_TENSOR_OP, "F16 GEMM recipe changed"); ++gemms;
 if(fail==3) { return 13; }
 for(int i=0;i<4;++i) y[i]=9;
 return 0;
}
`

const cudaMatMulStatusMain = `
#ifndef CHECKED_MATMUL
// Model the old Go call sites: void native APIs have no failure to consume.
static int fak_cuda_matmul_f32(const float *w,const float *x,float *y,int out,int in,int p) { fcuda_matmul_f32(w,x,y,out,in,p); return 0; }
static int fak_cuda_matmul_f16(const void *w,const float *x,float *y,int out,int in,int p,int col) { fcuda_matmul_f16(w,x,y,out,in,p,col); return 0; }
#endif
int main() {
 const float w[6]={}, x[6]={}; float y[4];
 // Exercise both old failing cuBLAS branches before reporting behavioral RED.
 fail=3; int f32=fak_cuda_matmul_f32(w,x,y,2,3,2);
 int f16=fak_cuda_matmul_f16(w,x,y,2,3,2,0);
 require(f32==10013 && f16==10013, "cuBLAS failure was returned as a usable output");
 for(int half=0;half<=1;++half) for(int col=0;col<=1;++col) for(int fault=0;fault<=3;++fault) {
  if(!half && (fault==1 || fault==2)) continue;
  fail=fault; gemms=converts=frees=allocated=0; for(float &v:y) v=-7;
  int status=half ? fak_cuda_matmul_f16(w,x,y,2,3,2,col) : fak_cuda_matmul_f32(w,x,y,2,3,2);
  int expected= fault==1 ? -1 : fault==2 ? 20077 : fault==3 ? 10013 : 0;
  require(status==expected, "lost stage-specific error");
  require(gemms==(fault==1 || fault==2 ? 0:1), "submitted GEMM after failure");
  require(converts==(half && fault!=1 ? 1:0) && frees==converts && allocated==0, "scratch leaked or failed allocation was submitted");
  for(float v:y) require(v==(fault ? -7:9), "failed operation fabricated output or success changed");
 }
 fail=0; fcuda_matmul_f32(w,x,y,2,3,2); fcuda_matmul_f16(w,x,y,2,3,2,1);
 require(y[0]==9, "public void ABI success changed");
}
`
