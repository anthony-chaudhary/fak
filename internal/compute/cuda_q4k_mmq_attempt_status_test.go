package compute

import (
	"context"
	"crypto/sha256"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
	"time"
)

// Compile the actual three-layer submission path, not a duplicate status policy.
// This host witness covers immediate status/control flow, not CUDA syntax/ABI,
// kernel math, allocator/device lifetime, graph capture or physical GPU execution.
// fak-test:runtime medium est=1s lane=default
// fak-test:justify Compiles extracted MMQ submission functions with bounded host C++17 recording stubs.
func TestCUDAQ4KMMQAttemptStatus(t *testing.T) {
	compiler, err := exec.LookPath("c++")
	if err != nil {
		t.Skip("C++17 compiler unavailable; MMQ attempted-status host witness not run")
	}
	read := func(path string) string {
		b, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("production %s sha256=%x", path, sha256.Sum256(b))
		return string(b)
	}
	mmq := read("cuda/q4k_mmq.cuh")
	quant := read("cuda/quant_matmul.cuh")
	extract := func(source, signature string) string {
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
		t.Fatalf("unterminated production definition %q", signature)
		return ""
	}
	declaration := func(source, pattern string) string {
		matches := regexp.MustCompile(pattern).FindAllString(source, -1)
		if len(matches) != 1 {
			t.Fatalf("expected one production declaration %q, got %d", pattern, len(matches))
		}
		return matches[0]
	}
	var definitions []string
	for _, name := range []string{"warp_size", "QK_K", "QK8_1", "QI8_1", "MATRIX_ROW_PADDING", "QK8_1_MMQ", "MMQ_TILE_NE_K", "nthreads", "I", "nwarps", "qk", "sram_stride", "CUDA_QUANTIZE_BLOCK_SIZE_MMQ"} {
		definitions = append(definitions, declaration(mmq, `(?m)^constexpr int `+name+` = [^;]+;`))
	}
	definitions = append(definitions,
		declaration(mmq, `(?m)^#define FAK_MMQ_PAD[^\n]+`),
		extract(mmq, "struct block_q8_1_mmq {")+";",
		extract(mmq, "struct device_info {")+";",
		declaration(mmq, `(?m)^static device_info g_device_info[^;]+;`),
		extract(mmq, "static uint3 init_fastdiv_values("),
		extract(mmq, "static constexpr __host__ __device__ bool config_has_J("),
		extract(mmq, "static size_t nbytes_shared("))
	launch := extract(mmq, "template <int J, bool fallback>\nstatic int launch_mul_mat_q(")
	dispatch := extract(mmq, "template <bool fallback>\nstatic int mul_mat_q_switch_J(")
	wrapper := extract(mmq, "static int fcuda_q4k_mmq_matmul(")
	outer := extract(quant, `extern "C" int fcuda_q4k_matmul_f32(`)
	replaceLaunch := func(body, launch, replacement string) string {
		if strings.Count(body, launch) != 1 {
			t.Fatalf("production launch changed; update the exact-source witness: %s", launch)
		}
		return strings.Replace(body, launch, replacement, 1)
	}
	launch = replaceLaunch(launch, `FCUDA_TS("mul_mat_q", block_nums_stream_k, mul_mat_q<J, fallback><<<block_nums_stream_k, block_dims, nbytes, stream>>>
        (x, y, dst, tmp_fixup, blocks_per_ne00_fd, nrows_x, ncols_dst, stride_row_x, ncols_dst, nrows_x, ntx_fd));`, `mock_launch("main");`)
	launch = replaceLaunch(launch, `FCUDA_TS("mul_mat_q_stream_k_fixup", block_nums_fixup, mul_mat_q_stream_k_fixup<J, fallback><<<block_nums_fixup, block_dims_fixup, 0, stream>>>
            (dst, tmp_fixup, blocks_per_ne00_fd, nrows_x, ncols_dst, nrows_x, ntx_fd));`, `mock_launch("fixup");`)
	wrapper = replaceLaunch(wrapper, `FCUDA_TS("quantize_mmq_q8_1_ds4", num_blocks, quantize_mmq_q8_1_ds4<<<num_blocks, block_size, 0, g_stream>>>(dX, y, in, in, in_padded, P));`, `mock_launch("quantize");`)
	outer = replaceLaunch(outer, `FCUDA_TS("k_q4k_dequant_split", (wn + 255) / 256, k_q4k_dequant_split<<<(wn + 255) / 256, 256, 0, g_stream>>>(
        (const unsigned char *)dQ4K, wHi, wLo, out, in));`, `mock_launch("dequant");`)
	outer = replaceLaunch(outer, `FCUDA_TS("k_fp16_split", (xn + 255) / 256, k_fp16_split<<<(xn + 255) / 256, 256, 0, g_stream>>>(dX, xHi, xLo, xn));`, `mock_launch("split");`)
	outer = replaceLaunch(outer, `FCUDA_TS("k_q4k_gemm", dim3((out + 1) / 2, P), k_q4k_gemm<<<dim3((out + 1) / 2, P), 256, 0, g_stream>>>(
        (const unsigned char *)dQ4K, coeff, dX, dY, out, in, P));`, `mock_launch("decode");`)
	outer = replaceLaunch(outer, `FCUDA_TS("k_q4k_gemm_panel", dim3(out, (P + token_tile - 1) / token_tile), k_q4k_gemm_panel<token_tile><<<dim3(out, (P + token_tile - 1) / token_tile), 32, 0, g_stream>>>(
      (const unsigned char *)dQ4K, dX, dY, out, in, P));`, `mock_launch("panel");`)
	actual := strings.Join(definitions, "\n") + "\n" + launch + "\n" + dispatch
	for _, unexpected := range []string{"<<<", "FCUDA_TS(", "cudaStreamSynchronize", "cudaDeviceSynchronize"} {
		if strings.Contains(actual+wrapper+outer, unexpected) {
			t.Fatalf("uncovered launch or synchronization in submission path: %s", unexpected)
		}
	}
	globals := strings.Join([]string{
		declaration(mmq, `(?m)^#define FCUDA_Q4K_MMQ_REFUSED[^\n]+`),
		declaration(mmq, `(?m)^static unsigned long long g_q4k_mmq_launches[^;]+;`),
		declaration(quant, `(?m)^#define FCUDA_Q4K_PREFILL_ARM_REFUSED[^\n]+`),
		declaration(quant, `(?m)^#define FCUDA_Q4K_MMQ_MIN_COLS[^\n]+`),
		declaration(quant, `(?m)^static int g_q4k_prefill_arm[^;]+;`),
		declaration(quant, `(?m)^static int g_q4k_decode_arm[^;]+;`),
	}, "\n")
	fixture := q4kMMQStatusPreamble + "\nnamespace fak_q4k_mmq {\n" + actual + "\n}\n" + globals + "\n" + wrapper + "\n" + outer + "\n" + q4kMMQStatusMain
	t.Logf("extracted host fixture sha256=%x", sha256.Sum256([]byte(fixture)))
	dir := t.TempDir()
	cpp, binary := filepath.Join(dir, "q4k_mmq_status.cpp"), filepath.Join(dir, "q4k_mmq_status.exe")
	if err := os.WriteFile(cpp, []byte(fixture), 0o600); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(t.Context(), 20*time.Second)
	defer cancel()
	if out, err := exec.CommandContext(ctx, compiler, "-std=c++17", "-O0", cpp, "-o", binary).CombinedOutput(); err != nil {
		t.Fatalf("compile actual-source host fixture: %v\n%s", err, out)
	}
	run, cancelRun := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancelRun()
	out, err := exec.CommandContext(run, binary).CombinedOutput()
	if err != nil {
		t.Fatalf("MMQ attempted status: %v\n%s", err, out)
	}
	if string(out) != "PASS 25 actual-source MMQ attempted-status cases; CUDA/device unrun\n" {
		t.Fatalf("incomplete host witness: %s", out)
	}
	t.Log(strings.TrimSpace(string(out)))
}

const q4kMMQStatusPreamble = `
#include <cstdint>
#include <cstddef>
#include <climits>
#include <cstdio>
#include <cstring>
#define __host__
#define __device__
using cudaError_t=int;
using cudaStream_t=int;
using __half=uint16_t;
struct half2 { uint16_t x,y; };
struct float2 { float x,y; };
struct dim3 { unsigned x,y,z; dim3(unsigned a=1,unsigned b=1,unsigned c=1):x(a),y(b),z(c){} };
struct uint3 { unsigned x,y,z; };
static uint3 make_uint3(unsigned x,unsigned y,unsigned z){return {x,y,z};}
constexpr int cudaSuccess=0,FCUDA_SCRATCH_ALLOC_FAILED=92000;
enum {cudaDevAttrComputeCapabilityMajor,cudaDevAttrComputeCapabilityMinor,cudaDevAttrMultiProcessorCount,cudaDevAttrMaxSharedMemoryPerBlockOptin,cudaFuncAttributeMaxDynamicSharedMemorySize};
struct cudaFuncAttributes { int ptxVersion; };
static int g_stream=0;
static int device_error,device_id,attribute_error,attribute_at,function_error,set_attribute_error;
static int major_version,compiled_arch,sm_count,shared_bytes;
static int quantize_error,main_error,fixup_error,cleanup_error,last_error;
static int quantize_calls,main_calls,fixup_calls,panel_calls,other_calls,checks;
static int allocation_calls,fail_allocation,free_calls,device_calls,attribute_calls;
alignas(16) static unsigned char storage[64];
static int cudaGetDevice(int *p){++device_calls;*p=device_id;return device_error;}
static int cudaDeviceGetAttribute(int *p,int attribute,int){
 ++attribute_calls;
 if(attribute==attribute_at && attribute_error)return attribute_error;
 switch(attribute){case cudaDevAttrComputeCapabilityMajor:*p=major_version;break;
 case cudaDevAttrComputeCapabilityMinor:*p=0;break;
 case cudaDevAttrMultiProcessorCount:*p=sm_count;break;default:*p=shared_bytes;}
 return 0;
}
template<class T> static int cudaFuncGetAttributes(cudaFuncAttributes *p,T){p->ptxVersion=compiled_arch;return function_error;}
template<class T> static int cudaFuncSetAttribute(T,int,int){return set_attribute_error;}
static int cudaGetLastError(){++checks;int result=last_error;last_error=0;return result;}
static void *scratch_malloc(size_t){++allocation_calls;return allocation_calls==fail_allocation?nullptr:storage;}
static void fcuda_free(void*){++free_calls;last_error=cleanup_error;}
static void mock_launch(const char *name){
 if(!std::strcmp(name,"quantize")){++quantize_calls;last_error=quantize_error;}
 else if(!std::strcmp(name,"main")){++main_calls;last_error=main_error;}
 else if(!std::strcmp(name,"fixup")){++fixup_calls;last_error=fixup_error;}
 else if(!std::strcmp(name,"panel")){++panel_calls;last_error=0;}
 else {++other_calls;last_error=0;}
}
namespace fak_q4k_mmq { template<int J,bool fallback> static void mul_mat_q(){} }
struct FP16CompensatedScratch { __half *ptr[4]; };
static int fp16_compensated_scratch(size_t,size_t,FP16CompensatedScratch*){return 77;}
static void fp16_compensated_gemm(__half*,__half*,__half*,__half*,float*,int,int,int){}
static int fp16_compensated_release(FP16CompensatedScratch*){return 0;}
static int fcuda_q4k_fp8lt_matmul(const uint8_t*,const float*,float*,int,int,int){return 78;}
static int fcuda_q4k_mmvq_matmul(const uint8_t*,const float*,float*,int,int,int){return 79;}
static float2 *q4_coeffs_for(const uint8_t*,int,int){return reinterpret_cast<float2*>(storage);}
`

const q4kMMQStatusMain = `
#define REQUIRE(x) do {if(!(x)){std::fprintf(stderr,"case %s line %d: %s\n",name,__LINE__,#x);return 1;}}while(0)
static const char *name;
static int cases;
static void reset(const char *label){
 name=label;++cases;
 device_error=device_id=attribute_error=function_error=set_attribute_error=0;attribute_at=-1;
 major_version=8;compiled_arch=80;sm_count=80;shared_bytes=100000;
 quantize_error=main_error=fixup_error=cleanup_error=last_error=0;
 quantize_calls=main_calls=fixup_calls=panel_calls=other_calls=checks=0;
 allocation_calls=fail_allocation=free_calls=device_calls=attribute_calls=0;
 std::memset(fak_q4k_mmq::g_device_info,0,sizeof(fak_q4k_mmq::g_device_info));
 g_q4k_prefill_arm=1;g_q4k_decode_arm=1;g_q4k_mmq_launches=0;
}
static bool no_work(){return !quantize_calls&&!main_calls&&!fixup_calls&&!panel_calls&&!other_calls;}
int main(){
 alignas(16) uint8_t w[16]={};alignas(16) float x[16]={},y[16]={};
 auto call=[&](int out=128,int in=256,int p=8){return fcuda_q4k_matmul_f32(w,x,y,out,in,p);};
 reset("device-api");device_error=17;
 REQUIRE(call()==17);REQUIRE(no_work()&&allocation_calls==0);
 for(int attribute=0;attribute<4;++attribute){
  reset("attribute-api");attribute_at=attribute;attribute_error=21+attribute;
  REQUIRE(call()==attribute_error);REQUIRE(no_work()&&allocation_calls==0);
 }
 reset("function-api");function_error=27;
 REQUIRE(call()==27);REQUIRE(no_work()&&allocation_calls==0);
 reset("old-device-decline");major_version=7;
 REQUIRE(call()==0);REQUIRE(panel_calls==1&&!quantize_calls&&!allocation_calls);
 reset("old-image-decline");compiled_arch=75;
 REQUIRE(call()==0);REQUIRE(panel_calls==1&&!quantize_calls&&!allocation_calls);
 reset("shape-decline");
 REQUIRE(call(128,255)==0);REQUIRE(panel_calls==1&&!device_calls&&!quantize_calls);
 reset("activation-scratch");fail_allocation=1;
 REQUIRE(call()==FCUDA_SCRATCH_ALLOC_FAILED);REQUIRE(no_work()&&allocation_calls==1);
 reset("quantize-failure");quantize_error=31;cleanup_error=91;
 REQUIRE(call()==31);REQUIRE(quantize_calls==1&&!main_calls&&!fixup_calls&&!panel_calls&&free_calls==1&&!g_q4k_mmq_launches);
 reset("late-function-attribute");set_attribute_error=32;cleanup_error=92;
 REQUIRE(call()==32);REQUIRE(quantize_calls==1&&!main_calls&&!fixup_calls&&!panel_calls&&free_calls==1);
 reset("late-kbc-overflow");
 REQUIRE(call(32768,1<<30)==-1);REQUIRE(quantize_calls==1&&!main_calls&&!fixup_calls&&!panel_calls&&free_calls==1);
 reset("late-no-J");shared_bytes=1;
 REQUIRE(call()==-1);REQUIRE(quantize_calls==1&&!main_calls&&!fixup_calls&&!panel_calls&&free_calls==1);
 reset("fixup-scratch");fail_allocation=2;
 REQUIRE(call()==FCUDA_SCRATCH_ALLOC_FAILED);REQUIRE(quantize_calls==1&&!main_calls&&!fixup_calls&&!panel_calls&&free_calls==1);
 reset("main-failure");main_error=33;cleanup_error=93;
 REQUIRE(call()==33);REQUIRE(quantize_calls==1&&main_calls==1&&!fixup_calls&&!panel_calls&&free_calls==2&&!g_q4k_mmq_launches);
 reset("fixup-failure");fixup_error=34;cleanup_error=94;
 REQUIRE(call()==34);REQUIRE(quantize_calls==1&&main_calls==1&&fixup_calls==1&&!panel_calls&&free_calls==2&&!g_q4k_mmq_launches);
 reset("P128-fixup-cleanup-failure");cleanup_error=95;
 REQUIRE(call(128,256,128)==95);REQUIRE(quantize_calls==1&&main_calls==1&&fixup_calls==1&&!panel_calls&&free_calls==2&&!g_q4k_mmq_launches);
 reset("success-fixup");
 REQUIRE(call()==0);REQUIRE(quantize_calls==1&&main_calls==1&&fixup_calls==1&&!panel_calls&&g_q4k_mmq_launches==1);
 reset("success-no-fixup");sm_count=1;
 REQUIRE(call()==0);REQUIRE(quantize_calls==1&&main_calls==1&&!fixup_calls&&!panel_calls&&g_q4k_mmq_launches==1);
 reset("below-threshold");
 REQUIRE(call(128,256,7)==0);REQUIRE(panel_calls==1&&!device_calls&&!quantize_calls);
 reset("default-small-panel");g_q4k_prefill_arm=0;
 REQUIRE(call()==0);REQUIRE(panel_calls==1&&!device_calls&&!quantize_calls);
 reset("P128-MMQ-error");quantize_error=35;
 REQUIRE(call(128,256,128)==35);REQUIRE(quantize_calls==1&&!main_calls&&!fixup_calls&&!panel_calls);
 reset("P128-default");g_q4k_prefill_arm=0;
 REQUIRE(call(128,256,128)==77);REQUIRE(no_work()&&!device_calls);
 reset("P128-fp8");g_q4k_prefill_arm=2;
 REQUIRE(call(128,256,128)==78);REQUIRE(no_work()&&!device_calls);
 REQUIRE(cases==25);
 std::printf("PASS %d actual-source MMQ attempted-status cases; CUDA/device unrun\n",cases);
}
`
