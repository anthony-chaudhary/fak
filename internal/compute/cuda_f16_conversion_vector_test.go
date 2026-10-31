package compute

import (
	"context"
	"crypto/sha256"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// Execute the actual conversion kernels/dispatcher and both native consumers with
// host CUDA coordinates and recording allocation/cuBLAS stubs. The half-conversion
// stub records input bits, not CUDA rounding: device numerics, NVCC and speed remain
// separate witnesses. Special values must reach the unchanged intrinsic verbatim.
// fak-test:runtime medium est=2s lane=default
// fak-test:justify Compiles one bounded source-extracted C++14 indexing/dispatch witness.
func TestCUDAF16ContiguousConversionVector(t *testing.T) {
	compiler, err := exec.LookPath("c++")
	if err != nil {
		t.Skip("C++14 compiler unavailable; F16 conversion host witness not run")
	}
	b, err := os.ReadFile("cuda/quant_matmul.cuh")
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("production cuda/quant_matmul.cuh sha256=%x", sha256.Sum256(b))
	source := string(b)
	extract := func(signature string) string {
		if strings.Count(source, signature) != 1 {
			t.Fatalf("expected one production definition %q", signature)
		}
		start := strings.Index(source, signature)
		open := strings.Index(source[start:], "{")
		if open < 0 {
			t.Fatalf("missing body for %q", signature)
		}
		depth := 0
		for end := start + open; end < len(source); end++ {
			switch source[end] {
			case '{':
				depth++
			case '}':
				depth--
				if depth == 0 {
					return source[start : end+1]
				}
			}
		}
		t.Fatalf("unterminated definition %q", signature)
		return ""
	}
	actual := strings.Join([]string{
		extract("template <typename T> struct alignas(sizeof(T)*4) fak_f16_cvt_vec4 {") + ";",
		extract("__global__ void k_f32_to_f16("),
		extract("__global__ void k_f32_to_f16_vec4("),
		extract("static void launch_f32_to_f16("),
		extract(`extern "C" void fcuda_f32_to_f16(`),
		extract(`extern "C" void fcuda_matmul_f16(`),
	}, "\n")
	for _, replacement := range [][2]string{
		{`k_f32_to_f16_vec4<<<(n4 + 255) / 256, 256, 0, g_stream>>>(src, dst, n4)`, `run_vec4((n4 + 255) / 256, 256, g_stream, src, dst, n4)`},
		{`k_f32_to_f16<<<(n + 255) / 256, 256, 0, g_stream>>>(src, dst, n)`, `run_scalar((n + 255) / 256, 256, g_stream, src, dst, n)`},
	} {
		if strings.Count(actual, replacement[0]) != 1 {
			t.Fatalf("actual launch changed: %s", replacement[0])
		}
		actual = strings.Replace(actual, replacement[0], replacement[1], 1)
	}
	if strings.Contains(actual, "<<<") || strings.Count(actual, "launch_f32_to_f16(") != 3 {
		t.Fatal("both real consumers must use the shared dispatcher, with no uncovered launch")
	}
	fixture := cudaF16VectorPreamble + actual + cudaF16VectorWitness
	t.Logf("extracted fixture sha256=%x", sha256.Sum256([]byte(fixture)))
	dir := t.TempDir()
	cpp, binary := filepath.Join(dir, "f16_vector.cpp"), filepath.Join(dir, "f16_vector.exe")
	if err := os.WriteFile(cpp, []byte(fixture), 0o600); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(t.Context(), 20*time.Second)
	defer cancel()
	if out, err := exec.CommandContext(ctx, compiler, "-std=c++14", "-O0", cpp, "-o", binary).CombinedOutput(); err != nil {
		t.Fatalf("compile actual-source F16 witness: %v\n%s", err, out)
	}
	run, cancelRun := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancelRun()
	out, err := exec.CommandContext(run, binary).CombinedOutput()
	if err != nil {
		t.Fatalf("F16 conversion source witness: %v\n%s", err, out)
	}
	if string(out) != "PASS actual F16 conversion dispatch/indexing at upload and HGEMM; CUDA rounding/device unrun\n" {
		t.Fatalf("incomplete F16 witness: %s", out)
	}
	t.Log(strings.TrimSpace(string(out)))
}

const cudaF16VectorPreamble = `
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <vector>
#define __global__
using __half = uint16_t;
struct dim3 { unsigned x, y, z; dim3(unsigned x=1, unsigned y=1, unsigned z=1):x(x),y(y),z(z){} };
static dim3 blockIdx, blockDim, threadIdx;
static int starts, ends, scalar_calls, vector_calls, blocks_seen, mallocs, frees, gemms;
static int expected_n, expected_out, expected_p, expected_in, expected_col;
static const void *expected_weight;
static void *expected_output;
static std::vector<uint32_t> converted;
static constexpr int g_stream = 17, g_blas = 19;
static constexpr uint16_t sentinel = 0xa55a;
alignas(16) static __half scratch[1100];
static uint32_t bits(float x) { uint32_t u; std::memcpy(&u, &x, 4); return u; }
static uint16_t encoded(uint32_t u) { return uint16_t(u ^ (u >> 16) ^ 0x5a5a); }
static __half __float2half(float x) { const auto u=bits(x); converted.push_back(u); return encoded(u); }
#define FCUDA_TS(label, grid, ...) do { \
    const dim3 recorded_grid(grid); (void)recorded_grid; \
    assert(std::strcmp(label, "k_f32_to_f16")==0 || std::strcmp(label, "cublas_gemm_ex")==0); \
    ++starts; __VA_ARGS__; ++ends; \
} while (0)
static void run_scalar(int, int, int, const float *, __half *, int);
static void run_vec4(int, int, int, const float *, __half *, int);
static void *fcuda_malloc(size_t bytes) {
    assert(bytes == size_t(expected_n)*sizeof(__half) && bytes <= sizeof(scratch));
    ++mallocs; std::fill(std::begin(scratch), std::end(scratch), sentinel); return scratch;
}
static void fcuda_free(void *p) { assert(p==scratch); ++frees; }
using cublasStatus_t = int;
enum { CUBLAS_STATUS_SUCCESS, CUBLAS_OP_N, CUBLAS_OP_T, CUDA_R_16F, CUDA_R_32F,
       CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP };
static cublasStatus_t cublasGemmEx(int handle, int opA, int opB, int m, int n, int k,
        const float *alpha, const void *A, int at, int lda, const void *B, int bt, int ldb,
        const float *beta, void *C, int ct, int ldc, int compute, int algorithm) {
    assert(handle==g_blas && opA==(expected_col ? CUBLAS_OP_N : CUBLAS_OP_T) && opB==CUBLAS_OP_N);
    assert(m==expected_out && n==expected_p && k==expected_in && *alpha==1 && *beta==0);
    assert(A==expected_weight && B==scratch && C==expected_output && at==CUDA_R_16F && bt==CUDA_R_16F && ct==CUDA_R_32F);
    assert(lda==(expected_col ? expected_out : expected_in) && ldb==expected_in && ldc==expected_out);
    assert(compute==CUBLAS_COMPUTE_32F && algorithm==CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    assert(mallocs==1 && frees==0 && int(converted.size())==expected_n);
    for (int i=0;i<expected_n;++i) assert(scratch[i]==encoded(converted[i]));
    for (size_t i=expected_n;i<1100;++i) assert(scratch[i]==sentinel);
    ++gemms; return CUBLAS_STATUS_SUCCESS;
}
`

const cudaF16VectorWitness = `
static_assert(sizeof(fak_f16_cvt_vec4<float>)==16 && alignof(fak_f16_cvt_vec4<float>)==16, "source vec4 geometry");
static_assert(sizeof(fak_f16_cvt_vec4<__half>)==8 && alignof(fak_f16_cvt_vec4<__half>)==8, "destination vec4 geometry");
static void run_scalar(int blocks, int threads, int stream, const float *src, __half *dst, int n) {
    assert(threads==256 && stream==g_stream); ++scalar_calls; blocks_seen=blocks;
    blockDim=dim3(threads);
    for(int b=0;b<blocks;++b) for(int t=0;t<threads;++t) { blockIdx.x=b; threadIdx.x=t; k_f32_to_f16(src,dst,n); }
}
static void run_vec4(int blocks, int threads, int stream, const float *src, __half *dst, int n4) {
    assert(threads==256 && stream==g_stream); ++vector_calls; blocks_seen=blocks;
    blockDim=dim3(threads);
    for(int b=0;b<blocks;++b) for(int t=0;t<threads;++t) { blockIdx.x=b; threadIdx.x=t; k_f32_to_f16_vec4(src,dst,n4); }
}
static void reset() { starts=ends=scalar_calls=vector_calls=blocks_seen=mallocs=frees=gemms=0; converted.clear(); }
static void fill(float *src, int n) {
    // Include signed zeros, infinities, both-sign NaN payloads, a subnormal,
    // half-rounding-boundary inputs and ordinary finite values. No arithmetic
    // model here is a substitute for CUDA's actual conversion intrinsic.
    const uint32_t values[]={0,0x80000000,0x7f800000,0xff800000,0x7fc12345,0xffc54321,
                             1,0x33800000,0x3f801000,0xbf801000,0x477fe000,0xc0200000};
    for(int i=0;i<n;++i) std::memcpy(src+i,values+i%12,4);
}
static void check_inputs(const float *src, int n) {
    assert(int(converted.size())==n);
    for(int i=0;i<n;++i) assert(converted[i]==bits(src[i]));
}
static void upload_case(int n, int src_offset, int dst_offset) {
    alignas(16) float input[1100]; alignas(16) __half output[1100];
    fill(input,1100); std::fill(std::begin(output),std::end(output),sentinel);
    const float *src=input+4+src_offset; __half *dst=output+4+dst_offset;
    reset(); fcuda_f32_to_f16(dst,src,n); check_inputs(src,n);
    const bool vector=n>0 && n%4==0 && src_offset%4==0 && dst_offset%4==0;
    assert(vector_calls==int(vector) && scalar_calls==int(!vector));
    assert(blocks_seen==((vector ? n/4 : n)+255)/256 && starts==1 && ends==1);
    assert(mallocs==0 && frees==0 && gemms==0);
    for(int i=0;i<1100;++i) {
        const int j=i-(4+dst_offset);
        assert(output[i]==(j>=0 && j<n ? encoded(bits(src[j])) : sentinel));
    }
}
static void matmul_case(int p, int in, int col) {
    alignas(16) float input[1100], output[200]; alignas(16) __half weight[200];
    fill(input,1100); reset();
    expected_n=p*in; expected_out=11; expected_p=p; expected_in=in; expected_col=col;
    expected_weight=weight; expected_output=output;
    fcuda_matmul_f16(weight,input,output,expected_out,in,p,col); check_inputs(input,expected_n);
    const bool vector=expected_n%4==0;
    assert(vector_calls==int(vector) && scalar_calls==int(!vector));
    assert(blocks_seen==((vector ? expected_n/4 : expected_n)+255)/256);
    assert(mallocs==1 && frees==1 && gemms==1 && starts==2 && ends==2);
}
int main() {
    for(int n: {0,1,3,4,7,8,255,256,257,1024,1028}) upload_case(n,0,0);
    for(int offset: {1,2,3}) { upload_case(1024,offset,0); upload_case(1024,0,offset); }
    upload_case(1028,1,3);
    for(int col: {0,1}) { matmul_case(17,16,col); matmul_case(3,5,col); }
    std::puts("PASS actual F16 conversion dispatch/indexing at upload and HGEMM; CUDA rounding/device unrun");
}
`
