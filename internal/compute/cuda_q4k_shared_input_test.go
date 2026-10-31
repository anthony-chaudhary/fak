package compute

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// This executes production admission/control flow with recording launch stubs.
// It is not a CUDA compile, numerical-parity, allocator-lifetime or device test.
// fak-test:runtime medium est=1s lane=default
// fak-test:justify Compiles the actual candidate submission source with host launch stubs.
func TestCUDAQ4KSharedInputSubmission(t *testing.T) {
	compiler, err := exec.LookPath("c++")
	if err != nil {
		t.Skip("C++17 compiler unavailable")
	}
	raw, err := os.ReadFile("cuda/q4k_shared_input.cuh")
	if err != nil {
		t.Fatal(err)
	}
	source := string(raw)
	for _, pair := range [][2]string{
		{`FCUDA_TS("quantize_q8_1", qgrid, quantize_q8_1<<<qgrid, 256, 0, g_stream>>>(x, quantized, in));`, `mock_quantize(x, quantized);`},
		{`FCUDA_TS("k_q4k_mmvq_gemv", grid, mul_mat_vec_q<1><<<grid, block, 0, g_stream>>>(weights[i], quantized, outputs[i], in, widths[i]));`, `mock_project(quantized, outputs[i], widths[i]);`},
	} {
		if strings.Count(source, pair[0]) != 1 {
			t.Fatalf("actual launch changed: %s", pair[0])
		}
		source = strings.Replace(source, pair[0], pair[1], 1)
	}
	for _, forbidden := range []string{"<<<", "cudaMalloc", "cudaFree", "cudaStreamSynchronize", "cudaDeviceSynchronize"} {
		if strings.Contains(source, forbidden) {
			t.Fatalf("uncovered launch/allocation/fence: %s", forbidden)
		}
	}
	dir := t.TempDir()
	cpp, binary := filepath.Join(dir, "shared.cpp"), filepath.Join(dir, "shared.exe")
	if err := os.WriteFile(cpp, []byte(q4kSharedInputPreamble+source+q4kSharedInputMain), 0600); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(t.Context(), 20*time.Second)
	defer cancel()
	if output, err := exec.CommandContext(ctx, compiler, "-std=c++17", "-O0", cpp, "-o", binary).CombinedOutput(); err != nil {
		t.Fatalf("host compile: %v\n%s", err, output)
	}
	if output, err := exec.CommandContext(ctx, binary).CombinedOutput(); err != nil {
		t.Fatalf("submission control flow: %v\n%s", err, output)
	}
}

const q4kSharedInputPreamble = `
#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#define FAK_CUDA_V41_CHECKED_STREAM
#define FCUDA_Q4K_MMVQ_REFUSED 80006
namespace fak_q4k_mmvq {
constexpr int QK_K=256, QK8_1=32, warp_size=32;
struct alignas(4) block_q8_1 { unsigned char bytes[36]; };
struct alignas(4) block_q4_K { unsigned char bytes[144]; };
static int calc_nwarps(int) { return 4; }
}
struct dim3 { dim3(unsigned, unsigned=1, unsigned=1) {} };
using cudaError_t = int;
constexpr int cudaSuccess=0, cudaErrorUnknown=999;
static bool g_capture_open=false, uncertain=false;
static unsigned long long g_q4k_mmvq_launches=0;
static int calls=0, failAt=0, lastError=0, quantizations=0;
static bool v41_candidates_stream_uncertain() { return uncertain; }
static int cudaGetLastError() { int e=lastError; lastError=0; return e; }
static void launch() { if (++calls==failAt) lastError=7; }
static void mock_quantize(const float *x, fak_q4k_mmvq::block_q8_1 *q) {
    ++quantizations;
    q[0].bytes[0]=(unsigned char)x[0];
    launch();
}
static void mock_project(const fak_q4k_mmvq::block_q8_1 *q, float *y, int out) {
    for(int i=0;i<out;++i) y[i]=q[0].bytes[0];
    launch();
}
`

const q4kSharedInputMain = `
int main() {
    alignas(4) uint8_t w[3*144]={};
    alignas(4) unsigned char scratch[8*36]={};
    float x[256]={}, q[1]={}, k[2]={}, v[3]={};
    auto run=[&]() { return fcuda_q4k_qkv_shared_input_f32(w,w,w,x,q,k,v,1,2,3,256,scratch,sizeof(scratch)); };
    x[0]=3; assert(run()==0 && calls==4 && quantizations==1 && v[2]==3);
    // Same pointer, new contents: no persistent activation identity/cache.
    x[0]=9; calls=0; assert(run()==0 && calls==4 && quantizations==2 && q[0]==9 && k[1]==9);
    calls=0; g_capture_open=true; assert(run()==80006 && calls==0); g_capture_open=false;
    uncertain=true; assert(run()==70999 && calls==0); uncertain=false;
    lastError=8; assert(run()==70008 && calls==0);
    // Quantization and each of three projections fail: never launch subsequent work.
    for(int n=1;n<=4;++n) { calls=0; failAt=n; assert(run()==70007 && calls==n); }
    failAt=0; calls=0;
    assert(fcuda_q4k_qkv_shared_input_f32(w,w,w,x,x,k,v,1,2,3,256,scratch,sizeof(scratch))==80006 && calls==0);
    assert(fcuda_q4k_qkv_shared_input_f32(w,w,w,x,q,v+1,v,1,2,3,256,scratch,sizeof(scratch))==80006 && calls==0);
    assert(fcuda_q4k_qkv_shared_input_f32(w,w,w,x,q,k,v,1,2,3,256,x,sizeof(x))==80006 && calls==0);
    assert(fcuda_q4k_qkv_shared_input_f32(w,w,w,x,(float*)w,k,v,1,2,3,256,scratch,sizeof(scratch))==80006 && calls==0);
    assert(fcuda_q4k_qkv_shared_input_f32(w,w,w,x,q,k,v,1,2,3,256,scratch,sizeof(scratch)-1)==80006 && calls==0);
    assert(fcuda_q4k_qkv_shared_input_f32(w,w,w,x,q,k,v,1,2,3,255,scratch,sizeof(scratch))==80006 && calls==0);
    assert(fcuda_q4k_qkv_shared_input_f32(w,w,w,x,q,k,v,1,0,3,256,scratch,sizeof(scratch))==80006 && calls==0);
    assert(fcuda_q4k_qkv_shared_input_f32(w,w,w,x,q,k,v,1,2,3,256,scratch+1,sizeof(scratch)-1)==80006 && calls==0);
    return 0;
}
`
