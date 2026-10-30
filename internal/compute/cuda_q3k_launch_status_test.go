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

// Compile the actual submission functions with recording CUDA launch stubs, not
// a duplicate implementation. This covers immediate status/control flow only:
// no CUDA syntax/ABI, kernel math, allocator, graph-capture or device witness.
// fak-test:runtime medium est=1s lane=default
// fak-test:justify Compiles two extracted submission functions with a host C++17 compiler.
func TestCUDAQ3KImmediateLaunchStatus(t *testing.T) {
	compiler, err := exec.LookPath("c++")
	if err != nil {
		t.Skip("C++17 compiler unavailable; immediate-launch host witness not run")
	}
	raw, err := os.ReadFile("cuda/q3k_mmvq.cuh")
	if err != nil {
		t.Fatal(err)
	}
	source := string(raw)
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
		t.Fatalf("unterminated production definition %q", signature)
		return ""
	}
	// The old void signature remains extractable for an unchanged-test red/green
	// run against the real parent source; do not synthesize a broken function.
	signature := "static int fcuda_q3k_mmvq_launch("
	if !strings.Contains(source, signature) {
		signature = "static void fcuda_q3k_mmvq_launch("
	}
	helper := extract(signature)
	wrapper := extract(`extern "C" int fcuda_q3k_matmul_f32(`)
	replaceLaunch := func(body, launch, replacement string) string {
		if strings.Count(body, launch) != 1 {
			t.Fatalf("production launch changed; update the exact-source witness: %s", launch)
		}
		return strings.Replace(body, launch, replacement, 1)
	}
	for _, cols := range []string{"1", "2", "3"} {
		launch := `FCUDA_TS("k_q3k_mmvq_gemv", grid, fak_q3k_mmvq::mul_mat_vec_q<` + cols + `><<<grid, block, 0, g_stream>>>(dQ3K, y, dY, in, out));`
		helper = replaceLaunch(helper, launch, `mock_launch("mmvq");`)
	}
	wrapper = replaceLaunch(wrapper, `FCUDA_TS("quantize_q8_1", qgrid, fak_q4k_mmvq::quantize_q8_1<<<qgrid, 256, 0, g_stream>>>(dX, g_q3k_mmvq_y, in));`, `mock_launch("quantize");`)
	wrapper = replaceLaunch(wrapper, `FCUDA_TS("k_q3k_dequant", nb, dequantize_block_q3_K<<<nb, 64, 0, g_stream>>>(dQ3K, w));`, `mock_launch("dequant");`)
	for _, unexpected := range []string{"<<<", "FCUDA_TS(", "cudaStreamSynchronize", "cudaDeviceSynchronize"} {
		if strings.Contains(helper+wrapper, unexpected) {
			t.Fatalf("uncovered launch or synchronization in submission path: %s", unexpected)
		}
	}
	fixture := q3kLaunchStatusPreamble + "\n" + helper + "\n" + wrapper + "\n" + q3kLaunchStatusMain
	t.Logf("production source sha256=%x; extracted host fixture sha256=%x", sha256.Sum256(raw), sha256.Sum256([]byte(fixture)))
	dir := t.TempDir()
	cpp := filepath.Join(dir, "q3k_launch_status.cpp")
	binary := filepath.Join(dir, "q3k_launch_status.exe")
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
		t.Fatalf("immediate-launch status: %v\n%s", err, out)
	}
	if string(out) != "PASS 10 actual-source immediate-launch status cases; CUDA/device unrun\n" {
		t.Fatalf("incomplete host witness: %s", out)
	}
	t.Log(strings.TrimSpace(string(out)))
}

const q3kLaunchStatusPreamble = `
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
using cudaError_t=int;
constexpr int cudaSuccess=0, FCUDA_SCRATCH_ALLOC_FAILED=92000;
struct dim3 { unsigned x,y,z; dim3(unsigned a=1,unsigned b=1,unsigned c=1):x(a),y(b),z(c){} };
namespace fak_q4k_mmvq { struct block_q8_1 { unsigned char bytes[36]; }; }
namespace fak_q3k_mmvq {
using fak_q4k_mmvq::block_q8_1;
constexpr int QK_K=256,QK8_1=32,max_cols=3,warp_size=32;
int calc_rows_per_block(int p){return p==1?1:2;}
int calc_nwarps(int){return 4;}
}
static fak_q4k_mmvq::block_q8_1 storage[64], *g_q3k_mmvq_y=storage;
static size_t g_q3k_mmvq_y_cap=sizeof(storage);
static unsigned long long g_q3k_mmvq_launches=0;
static int g_q4k_decode_arm=1;
static int quantize_calls=0,mmvq_calls=0,checks=0,last_error=0,quantize_error=0,mmvq_error=0;
static void mock_launch(const char *name) {
 if(!std::strcmp(name,"quantize")){++quantize_calls;last_error=quantize_error;}
 if(!std::strcmp(name,"mmvq")){++mmvq_calls;last_error=mmvq_error;}
}
static cudaError_t cudaGetLastError(){++checks;int e=last_error;last_error=0;return e;}
template<class T> int cudaMalloc(T **p,size_t){*p=reinterpret_cast<T*>(storage);return 0;}
static int cudaFree(void*){return 0;}
static int fcuda_alloc_status(int e){return e;}
static void fak_alloc_untrack(void*){}
static void fak_alloc_track(void*,int,int,const char*,size_t){}
#define FAK_ALLOC_ACTIVATIONS_SCRATCH 0
#define FAK_ALLOC_DEVICE_LOCAL 0
#define FAK_ALLOC_SITE(x) x
static float scratch[256];
static void *scratch_malloc(size_t){return scratch;}
static void fcuda_free(void*){}
static int fp16_compensated_matmul(const float*,const float*,float*,int,int,int){return 73;}
static int fcuda_q3k_panel_f32(const uint8_t*,const float*,float*,int,int,int){return 71;}
`

const q3kLaunchStatusMain = `
#define REQUIRE(x) do {if(!(x)){std::fprintf(stderr,"line %d: %s\n",__LINE__,#x);return 1;}}while(0)
static void reset(int q,int m){
 quantize_calls=mmvq_calls=checks=last_error=0;quantize_error=q;mmvq_error=m;
 g_q3k_mmvq_launches=0;g_q4k_decode_arm=1;
}
int main(){
 uint8_t w[110]={};float x[768]={},y[9]={};
 int cases=0;
 for(int p=1;p<=3;++p){
  reset(7,0);
  REQUIRE(fcuda_q3k_matmul_f32(w,x,y,3,256,p)==7);
  REQUIRE(quantize_calls==1 && mmvq_calls==0 && checks==1 && g_q3k_mmvq_launches==0);++cases;
  reset(0,9);
  REQUIRE(fcuda_q3k_matmul_f32(w,x,y,3,256,p)==9);
  REQUIRE(quantize_calls==1 && mmvq_calls==1 && checks==2 && g_q3k_mmvq_launches==0);++cases;
  reset(0,0);
  REQUIRE(fcuda_q3k_matmul_f32(w,x,y,3,256,p)==0);
  REQUIRE(quantize_calls==1 && mmvq_calls==1 && checks==2 && g_q3k_mmvq_launches==1);++cases;
 }
 reset(0,0);g_q4k_decode_arm=0;
 REQUIRE(fcuda_q3k_matmul_f32(w,x,y,3,256,1)==71);
 REQUIRE(quantize_calls==0 && mmvq_calls==0 && checks==0);++cases;
 std::printf("PASS %d actual-source immediate-launch status cases; CUDA/device unrun\n",cases);
}
`
