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
func TestCUDAQ8SmallPrefillSubmission(t *testing.T) {
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
			for _, kind := range []string{"q8"} {
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
		start := strings.Index(source, "static signed char *g_q8_qX = nullptr;")
		if start < 0 {
			t.Fatal("missing Q8 scratch")
		}
		end := strings.Index(source[start:], "// getScaleMinK4_dev")
		if end < 0 {
			t.Fatal("missing Q8 boundary")
		}
		body := source[start : start+end]
		for _, replacement := range [][2]string{
			{"k_q8_quant_act<<<dim3(nblk, P), 64, 0, g_stream>>>", "mock_quant"},
			{"k_q8_gemm<<<dim3((out + 7) / 8, P), 256, 0, g_stream>>>", "mock_gemm"},
		} {
			if strings.Count(body, replacement[0]) != 1 {
				t.Fatal("launch changed; update faithful substitution")
			}
			body = strings.Replace(body, replacement[0], replacement[1], 1)
		}
		dir := t.TempDir()
		cpp, bin := filepath.Join(dir, "q8.cpp"), filepath.Join(dir, "q8")
		if err := os.WriteFile(cpp, []byte(cudaQ8SmallPrefillPrefix+body+cudaQ8SmallPrefillMain), 0600); err != nil {
			t.Fatal(err)
		}
		ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		defer cancel()
		if out, err := exec.CommandContext(ctx, compiler, "-std=c++11", "-Wall", "-Wextra", "-Werror", cpp, "-o", bin).CombinedOutput(); err != nil {
			t.Fatalf("compile literal Q8 body: %v\n%s", err, out)
		}
		if out, err := exec.CommandContext(ctx, bin).CombinedOutput(); err != nil {
			t.Fatalf("Q8 submission: %v\n%s", err, out)
		}
	})
}

const cudaQ8SmallPrefillPrefix = `
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
using cudaError_t = int;
enum { cudaSuccess=0 };
static int launches, checks, failAt;
static void require(bool ok, const char *why) { if (!ok) { fprintf(stderr,"%s\n",why); exit(1); } }
template<typename T> static int cudaMalloc(T **p, size_t n) { *p=(T*)malloc(n); require(*p!=nullptr,"host allocation"); return 0; }
static int cudaFree(void *p) { free(p); return 0; }
#define CK(expr) require((expr)==0,"allocation failed")
static inline int cudaGetLastError() { ++checks; return launches==failAt ? 77 : 0; }
static int q8_dequant_sgemm_checked(const void *,const float *,const float *,float *,int,int,int,int) { require(false,"large prefill entered"); return 0; }
static void mock_quant(const float *x,signed char *q,float *s,int p,int in,int block) { require(x && q && s && p>0 && p<128 && in==32 && block==32,"quant operands changed"); require(launches==0,"quant ordering"); ++launches; }
static void mock_gemm(const signed char *w,const float *ws,const signed char *q,const float *s,float *y,int out,int in,int p,int block) { require(w && ws && q && s && y && out==2 && in==32 && p>0 && p<128 && block==32,"GEMM operands changed"); require(launches==1,"GEMM ordering"); ++launches; }
`
const cudaQ8SmallPrefillMain = `
int main() {
 const int8_t w[64]={}; const float x[4064]={}, scales[2]={}; float y[254]={};
 const int ps[]={1,2,127};
 for(int p:ps) for(int fault=1;fault<=3;++fault) {
  launches=checks=0; failAt=fault==3?0:fault;
  int status=fak_cuda_q8_matmul_f32(w,scales,x,y,2,32,p,32);
  require(status==(failAt?20077:0),"Q8 failed launch returned usable status");
  require(launches==(failAt==1?1:2),"Q8 GEMM submitted after quant failure");
  require(checks==launches,"Q8 launch status not consumed immediately");
 }
 cudaFree(g_q8_qX); cudaFree(g_q8_xScale);
}
`
