package compute

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"math"
	"math/rand"
	"sort"

	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/kquantbits"
)

// Compile the actual submission functions with recording CUDA launch stubs, not
// a duplicate implementation. This covers immediate status/control flow only:
// no CUDA syntax/ABI, kernel math, allocator, graph-capture or device witness.
// The P>=128 controls prove the direct weight split uses one common scratch owner.
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
	if strings.Contains(wrapper, "FP16CompensatedScratch scratch;") {
		wrapper = replaceLaunch(wrapper, `FCUDA_TS("k_q3k_dequant", nb, dequantize_block_q3_K<<<nb, 64, 0, g_stream>>>(dQ3K, wHi, wLo));`, `mock_weight_split(dQ3K, wHi, wLo, nb);`)
		wrapper = replaceLaunch(wrapper, `FCUDA_TS("k_fp16_split", (xn + 255) / 256, k_fp16_split<<<(xn + 255) / 256, 256, 0, g_stream>>>(dX, xHi, xLo, xn));`, `mock_input_split(dX, xHi, xLo, xn);`)
	} else {
		// Keep the actual old path executable: it must fail the no-F32-image
		// assertion below, not a missing-source-signature check.
		wrapper = replaceLaunch(wrapper, `FCUDA_TS("k_q3k_dequant", nb, dequantize_block_q3_K<<<nb, 64, 0, g_stream>>>(dQ3K, w));`, `mock_launch("dequant");`)
	}
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
	if strings.ReplaceAll(string(out), "\r\n", "\n") != "PASS 17 actual-source Q3 submission cases; CUDA/device unrun\n" {
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
#include <vector>
using __half=uint16_t;
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
static int old_float_allocs=0, scratch_calls=0, release_calls=0, gemm_calls=0;
static int scratch_error=0, gemm_error=0, release_error=0, bad_binding=0;
static size_t seen_wn=0,seen_xn=0;
static const uint8_t *expected_w=nullptr;
static const float *expected_x=nullptr;
static float *expected_y=nullptr;
static int expected_out=0,expected_in=0,expected_p=0;
static __half planes[4][1];
static std::vector<int> order;
struct FP16CompensatedScratch { __half *ptr[4]={}; };
static int fp16_compensated_scratch(size_t wn,size_t xn,FP16CompensatedScratch *s){
 ++scratch_calls;order.push_back(1);seen_wn=wn;seen_xn=xn;
 if(scratch_error)return scratch_error;
 for(int i=0;i<4;++i)s->ptr[i]=planes[i];return 0;
}
static void mock_weight_split(const uint8_t *w,__half *hi,__half *lo,int nb){
 order.push_back(2);bad_binding|=w!=expected_w||hi!=planes[0]||lo!=planes[1]||nb!=(int)(seen_wn/256);
}
static void mock_input_split(const float *x,__half *hi,__half *lo,size_t n){
 order.push_back(3);bad_binding|=x!=expected_x||hi!=planes[2]||lo!=planes[3]||n!=seen_xn;
}
static int fp16_compensated_gemm(const __half *wh,const __half *wl,const __half *xh,const __half *xl,float *y,int out,int in,int p){
 ++gemm_calls;order.push_back(4);
 bad_binding|=wh!=planes[0]||wl!=planes[1]||xh!=planes[2]||xl!=planes[3]||y!=expected_y||out!=expected_out||in!=expected_in||p!=expected_p;
 return gemm_error;
}
static int fp16_compensated_release(FP16CompensatedScratch *s){
 ++release_calls;order.push_back(5);for(int i=0;i<4;++i)bad_binding|=s->ptr[i]!=planes[i];return release_error;
}
static float scratch[256];
static void *scratch_malloc(size_t){++old_float_allocs;return scratch;}
static void fcuda_free(void*){}
static int fp16_compensated_matmul(const float*,const float*,float*,int,int,int){return 73;}
static int fcuda_q3k_panel_f32(const uint8_t*,const float*,float*,int,int,int){return 71;}
`

const q3kLaunchStatusMain = `
#define REQUIRE(x) do {if(!(x)){std::fprintf(stderr,"line %d: %s\n",__LINE__,#x);return 1;}}while(0)
static void reset(int q,int m){
 quantize_calls=mmvq_calls=checks=last_error=0;quantize_error=q;mmvq_error=m;
 g_q3k_mmvq_launches=0;g_q4k_decode_arm=1;
 old_float_allocs=scratch_calls=release_calls=gemm_calls=0;
 scratch_error=gemm_error=release_error=bad_binding=0;seen_wn=seen_xn=0;order.clear();
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
 reset(0,0);
 REQUIRE(fcuda_q3k_matmul_f32(w,x,y,3,256,127)==71);
 REQUIRE(order.empty() && old_float_allocs==0 && quantize_calls==0 && mmvq_calls==0);++cases;
 expected_w=w;expected_x=x;expected_y=y;expected_out=3;expected_in=256;
 for(int p : {128,129}){
  reset(0,0);expected_p=p;
  const int status=fcuda_q3k_matmul_f32(w,x,y,3,256,p);
  REQUIRE(old_float_allocs==0); // actual old source fails here, not during extraction
  REQUIRE(status==0 && scratch_calls==1 && release_calls==1 && gemm_calls==1 && !bad_binding);
  REQUIRE(seen_wn==768 && seen_xn==(size_t)p*256 && order==std::vector<int>({1,2,3,4,5}));
  REQUIRE(quantize_calls==0 && mmvq_calls==0 && checks==0);++cases;
 }
 reset(0,0);expected_p=128;scratch_error=92001;
 REQUIRE(fcuda_q3k_matmul_f32(w,x,y,3,256,128)==92001);
 REQUIRE(order==std::vector<int>({1}) && release_calls==0 && gemm_calls==0 && old_float_allocs==0);++cases;
 const int errors[3][2]={{93013,0},{0,83007},{93013,83007}};
 for(const auto &e:errors){
  reset(0,0);gemm_error=e[0];release_error=e[1];
  REQUIRE(fcuda_q3k_matmul_f32(w,x,y,3,256,128)==(e[0]?e[0]:e[1]));
  REQUIRE(order==std::vector<int>({1,2,3,4,5}) && !bad_binding && release_calls==1 && gemm_calls==1);++cases;
 }
 std::printf("PASS %d actual-source Q3 submission cases; CUDA/device unrun\n",cases);
}
`

// Execute the actual 64-thread producer body with software CUDA intrinsics.
// The independent Go oracle is the existing scalar Q3 layout decoder plus an
// IEEE binary16 nearest-even search. This is not native FTZ/fast-math evidence.
// fak-test:runtime medium est=2s lane=default
// fak-test:justify Compiles one extracted producer and checks bounded packed blocks on the host.
func TestCUDAQ3KDirectSplitNumerical(t *testing.T) {
	compiler, err := exec.LookPath("c++")
	if err != nil {
		t.Skip("C++17 compiler unavailable; Q3 direct-split host witness not run")
	}
	raw, err := os.ReadFile("cuda/q3k_mmvq.cuh")
	if err != nil {
		t.Fatal(err)
	}
	source := string(raw)
	function := func(signature string) string {
		t.Helper()
		if strings.Count(source, signature) != 1 {
			t.Fatalf("expected one actual source definition: %s", signature)
		}
		start := strings.Index(source, signature)
		open := strings.Index(source[start:], "{")
		if open < 0 {
			t.Fatal("missing actual producer body")
		}
		depth := 0
		for i := start + open; i < len(source); i++ {
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
		t.Fatal("unterminated actual producer body")
		return ""
	}
	block := function("struct block_q3_K {") + ";\n"
	body := function("static __global__ void dequantize_block_q3_K(")
	for _, want := range []string{"__half *__restrict__ hi", "__half *__restrict__ lo", "__fmul_rn", "__fsub_rn"} {
		if !strings.Contains(body, want) {
			t.Fatalf("direct producer contract missing %s", want)
		}
	}
	program := q3kSplitNumericalPreamble + block + body + q3kSplitNumericalMain
	t.Logf("Q3 source sha256=%x; actual-body host fixture sha256=%x", sha256.Sum256(raw), sha256.Sum256([]byte(program)))
	var packed []byte
	appendUniform := func(scale, code int, d uint16) {
		b := make([]byte, q3kSuperBlock)
		for i := 0; i < 32; i++ {
			if code >= 0 {
				b[i] = 255
			}
		}
		for i := 32; i < 96; i++ {
			b[i] = byte(code&3) * 85
		}
		for i := 96; i < 104; i++ {
			b[i] = byte(scale&15) * 17
		}
		for i := 104; i < 108; i++ {
			b[i] = byte(scale>>4) * 85
		}
		binary.LittleEndian.PutUint16(b[108:], d)
		packed = append(packed, b...)
	}
	for scale := 0; scale < 64; scale++ {
		for code := -4; code < 4; code++ {
			appendUniform(scale, code, 0x3555)
		}
	}
	for _, d := range []uint16{0, 0x8000, 1, 0x8001, 0x03ff, 0x0400, 0x3c00, 0xbc00, 0x7bff, 0xfbff, 0x7c00, 0xfc00, 0x7e01, 0xfe01} {
		for _, scale := range []int{0, 31, 32, 63} {
			for code := -4; code < 4; code++ {
				appendUniform(scale, code, d)
			}
		}
	}
	rng := rand.New(rand.NewSource(14217))
	for b := 0; b < 128; b++ {
		block := make([]byte, q3kSuperBlock)
		for i := range block {
			block[i] = byte(rng.Intn(256))
		}
		packed = append(packed, block...)
	}
	dir := t.TempDir()
	cpp, binaryPath := filepath.Join(dir, "q3_split.cpp"), filepath.Join(dir, "q3_split.exe")
	input, output := filepath.Join(dir, "blocks.bin"), filepath.Join(dir, "halves.bin")
	for name, bytes := range map[string][]byte{cpp: []byte(program), input: packed} {
		if err := os.WriteFile(name, bytes, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	ctx, cancel := context.WithTimeout(t.Context(), 20*time.Second)
	defer cancel()
	if out, err := exec.CommandContext(ctx, compiler, "-std=c++17", "-O0", cpp, "-o", binaryPath).CombinedOutput(); err != nil {
		t.Fatalf("compile actual Q3 split producer: %v\n%s", err, out)
	}
	run, cancelRun := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancelRun()
	if out, err := exec.CommandContext(run, binaryPath, input, output).CombinedOutput(); err != nil {
		t.Fatalf("run actual Q3 split producer: %v\n%s", err, out)
	}
	got, err := os.ReadFile(output)
	if err != nil {
		t.Fatal(err)
	}
	blocks := len(packed) / q3kSuperBlock
	if len(got) != blocks*q3kSuper*4 {
		t.Fatalf("incomplete two-plane output: %d bytes", len(got))
	}
	values := make([]float32, q3kSuper)
	for b := 0; b < blocks; b++ {
		q3kDequantSuperBlock(values, packed[b*q3kSuperBlock:(b+1)*q3kSuperBlock])
		for i, value := range values {
			hi := q3kSplitHalfReference(value)
			hiFloat := math.Float32frombits(kquantbits.F16BitsToF32Bits(hi))
			lo := q3kSplitHalfReference(float32(value - hiFloat))
			for plane, want := range []uint16{hi, lo} {
				off := ((plane*blocks+b)*q3kSuper + i) * 2
				bits := binary.LittleEndian.Uint16(got[off:])
				wantNaN := want&0x7c00 == 0x7c00 && want&0x03ff != 0
				gotNaN := bits&0x7c00 == 0x7c00 && bits&0x03ff != 0
				if (wantNaN && !gotNaN) || (!wantNaN && bits != want) {
					t.Fatalf("block=%d element=%d plane=%d value=%g got=%04x want=%04x", b, i, plane, value, bits, want)
				}
			}
		}
	}
	t.Logf("PASS %d packed blocks / %d source-thread outputs, both guarded half planes; CUDA/device unrun", blocks, blocks*q3kSuper)
}

// Search binary16 representable magnitudes, choosing the nearest-even endpoint.
// This deliberately does not share the C++ intrinsic's bit-shift conversion.
func q3kSplitHalfReference(value float32) uint16 {
	sign := uint16(0)
	if math.Signbit(float64(value)) {
		sign = 0x8000
	}
	a := math.Abs(float64(value))
	if math.IsNaN(a) {
		return sign | 0x7e00
	}
	if a >= 65520 {
		return sign | 0x7c00
	}
	asFloat := func(h int) float64 { return float64(math.Float32frombits(kquantbits.F16BitsToF32Bits(uint16(h)))) }
	hi := sort.Search(0x7c00, func(h int) bool { return asFloat(h) >= a })
	if hi == 0 {
		return sign
	}
	if hi == 0x7c00 {
		return sign | 0x7bff
	}
	lo := hi - 1
	dLo, dHi := a-asFloat(lo), asFloat(hi)-a
	if dLo < dHi || (dLo == dHi && lo&1 == 0) {
		return sign | uint16(lo)
	}
	return sign | uint16(hi)
}

const q3kSplitNumericalPreamble = `
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include <limits>
#include <cfenv>
static_assert(sizeof(float)==4 && std::numeric_limits<float>::is_iec559,"IEEE binary32 host required");
#define __global__
using __half=uint16_t;
using half=__half;
constexpr int QK_K=256;
struct Index { unsigned x=0; } blockIdx,threadIdx;
static float __half2float(__half h){
 uint32_t sign=(uint32_t)(h&0x8000)<<16,exp=(h>>10)&31,mant=h&1023,bits;
 if(exp==0){
  if(!mant)bits=sign;
  else{int e=-14;while(!(mant&1024)){mant<<=1;--e;}bits=sign|((uint32_t)(e+127)<<23)|((mant&1023)<<13);}
 }else if(exp==31)bits=sign|0x7f800000u|(mant<<13);
 else bits=sign|((exp+112)<<23)|(mant<<13);
 float f;std::memcpy(&f,&bits,4);return f;
}
static __half __float2half_rn(float f){
 uint32_t bits;std::memcpy(&bits,&f,4);
 uint16_t sign=(uint16_t)((bits>>16)&0x8000);uint32_t exp=(bits>>23)&255,mant=bits&0x7fffff;
 if(exp==255)return sign|0x7c00|(mant?0x0200:0);
 if(exp==0)return sign;
 int e=(int)exp-127;
 if(e>15)return sign|0x7c00;
 if(e<-25)return sign;
 uint32_t significand=mant|0x800000;
 unsigned shift=e<-14?(unsigned)(-e-1):13;
 uint32_t kept=significand>>shift,tail=significand&((1u<<shift)-1),mid=1u<<(shift-1);
 if(tail>mid||(tail==mid&&(kept&1)))++kept;
 if(e<-14)return sign|(uint16_t)kept;
 return sign|(uint16_t)(((e+14)<<10)+kept);
}
static float __fmul_rn(float a,float b){volatile float r=a*b;return r;}
static float __fsub_rn(float a,float b){volatile float r=a-b;return r;}
`

const q3kSplitNumericalMain = `
static_assert(sizeof(block_q3_K)==110,"actual Q3 block layout changed");
int main(int argc,char **argv){
 if(argc!=3 || std::fegetround()!=FE_TONEAREST)return 1;
 FILE *in=std::fopen(argv[1],"rb");if(!in)return 2;
 std::vector<block_q3_K> packed;block_q3_K b;size_t read=0;
 while((read=std::fread(&b,1,sizeof(b),in))==sizeof(b)){
  const unsigned char *raw=reinterpret_cast<const unsigned char*>(&b);
  b.d=(half)((unsigned)raw[108]|((unsigned)raw[109]<<8));
  packed.push_back(b);
 }
 if(read!=0||std::ferror(in)||!std::feof(in)||packed.empty()||packed.size()>2048){std::fclose(in);return 3;}
 std::fclose(in);
 const size_t count=packed.size()*QK_K;
 const __half guard=0x55aa;
 std::vector<__half> hi(count+2,guard),lo(count+2,guard);
 for(blockIdx.x=0;blockIdx.x<packed.size();++blockIdx.x)
  for(threadIdx.x=0;threadIdx.x<64;++threadIdx.x)
   dequantize_block_q3_K(packed.data(),hi.data()+1,lo.data()+1);
 if(hi.front()!=guard||hi.back()!=guard||lo.front()!=guard||lo.back()!=guard)return 4;
 FILE *out=std::fopen(argv[2],"wb");if(!out)return 5;
 // Explicit little-endian serialization avoids a host-endian oracle assumption.
 for(const auto *plane:{&hi,&lo})for(size_t i=1;i<=count;++i){
  unsigned char bytes[2]={(unsigned char)((*plane)[i]&255),(unsigned char)((*plane)[i]>>8)};
  if(std::fwrite(bytes,1,2,out)!=2){std::fclose(out);return 6;}
 }
 return std::fclose(out)==0?0:7;
}
`
