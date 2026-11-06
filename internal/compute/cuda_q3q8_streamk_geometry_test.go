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

// Execute production launcher, partition and fixup source with host coordinates.
// This qualifies launch geometry and contribution order, not CUDA compilation,
// the MMA body, device execution or measured performance.
// fak-test:runtime medium est=4s lane=default
// fak-test:justify Compiles two bounded actual-source C++17 geometry witness.
func TestCUDAQ3Q8StreamKGeometry(t *testing.T) {
	for _, format := range []struct{ name, path, prefix, macro, quant string }{
		{"Q3_K", "cuda/q3k_mmq.cuh", "q3", "Q3", "QK_K"},
		{"Q8_0", "cuda/q8_mmq.cuh", "q8", "Q8", "QK8_0"},
	} {
		t.Run(format.name, func(t *testing.T) {
			compiler, err := exec.LookPath("c++")
			if err != nil {
				t.Skip("C++17 compiler unavailable; Q3_K/Q8_0 Stream-K host witness not run")
			}
			raw, err := os.ReadFile(format.path)
			if err != nil {
				t.Fatal(err)
			}
			t.Logf("production Q3_K/Q8_0 MMQ sha256=%x", sha256.Sum256(raw))
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
				t.Fatalf("unterminated definition %q", signature)
				return ""
			}
			declaration := func(pattern string) string {
				matches := regexp.MustCompile(pattern).FindAllString(source, -1)
				if len(matches) != 1 {
					t.Fatalf("expected one declaration %q, got %d", pattern, len(matches))
				}
				return matches[0]
			}
			var definitions []string
			if format.name == "Q8_0" {
				definitions = append(definitions, declaration(`(?m)^constexpr int QI8_0 = [^;]+;`), extract("struct q8_weight_view {")+";")
			}
			for _, name := range []string{"warp_size", format.quant, "QK8_1", "QI8_1", "MMQ_ITER_K", "QK8_1_MMQ", "MMQ_TILE_NE_K", "nthreads", "I", "nwarps", "qk", "ITER_K", "blocks_per_iter", "sram_stride"} {
				definitions = append(definitions, declaration(`(?m)^constexpr int `+name+` = [^;]+;`))
			}
			definitions = append(definitions,
				declaration(`(?m)^#define FAK_`+format.macro+`_MMQ_PAD[^\n]+`),
				extract("struct block_q8_1_mmq {")+";",
				extract("struct device_info {")+";",
				extract("static uint3 init_fastdiv_values("),
				extract("static __device__ __forceinline__ uint32_t fastdiv("),
				extract("static __device__ __forceinline__ uint32_t fastmodulo("),
				extract("static __device__ __forceinline__ uint2 fast_div_modulo("),
				extract("static size_t nbytes_shared("))
			launch := extract("template <int J, bool fallback>\nstatic int launch_mul_mat_q(")
			for _, replacement := range [][2]string{
				{`FCUDA_TS("q2_mul_mat_q", block_nums_stream_k, mul_mat_q<J, fallback><<<block_nums_stream_k, block_dims, nbytes, stream>>>
        (x, y, dst, tmp_fixup, blocks_per_ne00_fd, nrows_x, ncols_dst, stride_row_x, ncols_dst, nrows_x, ntx_fd));`, `record_launch(false, block_nums_stream_k);`},
				{`FCUDA_TS("q2_mul_mat_q_stream_k_fixup", block_nums_fixup, mul_mat_q_stream_k_fixup<J, fallback><<<block_nums_fixup, block_dims_fixup, 0, stream>>>
            (dst, tmp_fixup, blocks_per_ne00_fd, nrows_x, ncols_dst, nrows_x, ntx_fd));`, `record_launch(true, block_nums_fixup);`},
			} {
				replacement[0] = strings.ReplaceAll(replacement[0], "q2_", format.prefix+"_")
				if strings.Count(launch, replacement[0]) != 1 {
					t.Fatalf("production launch changed: %s", replacement[0])
				}
				launch = strings.Replace(launch, replacement[0], replacement[1], 1)
			}
			start := strings.Index(source, "    int kbc      = int64_t(blockIdx.x)")
			if start < 0 {
				t.Fatal("production partition missing")
			}
			end := strings.Index(source[start:], "    // kb0 == k index")
			if end < 0 {
				t.Fatal("production partition boundary missing")
			}
			interval := `
static std::pair<int,int> interval(int tiles,int kblocks,int grid,int block) {
 const uint3 ntx=init_fastdiv_values(tiles), blocks_per_ne00=init_fastdiv_values(kblocks);
 const int nty=1; blockIdx.x=block; gridDim.x=grid;
` + source[start:start+end] + `
 return {kbc,kbc_stop};
}
`
			fixup := "template <int J, bool fallback>\n" + extract("static __global__ void mul_mat_q_stream_k_fixup(")
			actual := strings.Join(definitions, "\n") + "\n" + launch + "\n" + interval + "\n" + fixup
			if strings.Contains(actual, "<<<") || strings.Contains(actual, "FCUDA_TS(") {
				t.Fatal("uncovered production launch")
			}
			fixture := q3q8StreamKPreamble + "\nnamespace fak_mmq_test {\n" + actual + "\n}\n" + q3q8StreamKMain
			t.Logf("extracted fixture sha256=%x", sha256.Sum256([]byte(fixture)))
			dir := t.TempDir()
			cpp, binary := filepath.Join(dir, "q3q8_streamk.cpp"), filepath.Join(dir, "q3q8_streamk.exe")
			if err := os.WriteFile(cpp, []byte(fixture), 0o600); err != nil {
				t.Fatal(err)
			}
			ctx, cancel := context.WithTimeout(t.Context(), 20*time.Second)
			defer cancel()
			if out, err := exec.CommandContext(ctx, compiler, "-std=c++17", "-O0", cpp, "-o", binary).CombinedOutput(); err != nil {
				t.Fatalf("compile actual-source Q3_K/Q8_0 geometry: %v\n%s", err, out)
			}
			run, stop := context.WithTimeout(t.Context(), 5*time.Second)
			defer stop()
			out, err := exec.CommandContext(run, binary).CombinedOutput()
			if err != nil {
				t.Fatalf("Q3_K/Q8_0 geometry: %v\n%s", err, out)
			}
			const want = "PASS Q3_K/Q8_0 actual-source grid, partition, fixup and overflow; CUDA/device/performance unrun\n"
			if strings.ReplaceAll(string(out), "\r\n", "\n") != want {
				t.Fatalf("incomplete witness: %s", out)
			}
			t.Log(strings.TrimSpace(string(out)))
		})
	}
}

const q3q8StreamKPreamble = `
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <vector>
#include <utility>
#include <limits>
#define __host__
#define __device__
#define __global__
#define __forceinline__
using cudaError_t=int; using cudaStream_t=int; using half=uint16_t;
struct half2 { uint16_t x,y; };
struct dim3 { unsigned x,y,z; dim3(unsigned a=1,unsigned b=1,unsigned c=1):x(a),y(b),z(c){} };
struct uint2 { unsigned x,y; }; struct uint3 { unsigned x,y,z; };
static uint2 make_uint2(unsigned x,unsigned y){return {x,y};}
static uint3 make_uint3(unsigned x,unsigned y,unsigned z){return {x,y,z};}
static uint32_t __umulhi(uint32_t a,uint32_t b){return (uint64_t(a)*b)>>32;}
static dim3 blockIdx,threadIdx,gridDim;
constexpr int cudaSuccess=0,cudaErrorInvalidValue=1,FCUDA_SCRATCH_ALLOC_FAILED=92000,cudaFuncAttributeMaxDynamicSharedMemorySize=0;
static int main_calls,fixup_calls,allocations,frees;
static unsigned main_grid,fixup_grid; static size_t allocated;
static unsigned char storage;
template<class T> static int cudaFuncSetAttribute(T,int,int){return 0;}
static int cudaGetLastError(){return 0;}
static int cudaStreamSynchronize(int){return 0;}
static int v41_checked_stream_error(int base,int code){return base+code;}
static void *scratch_malloc(size_t n){++allocations;allocated=n;return &storage;}
static void fcuda_free(void*){++frees;}
static void record_launch(bool fixup,dim3 grid){if(fixup){++fixup_calls;fixup_grid=grid.x;}else{++main_calls;main_grid=grid.x;}}
namespace fak_mmq_test { template<int J,bool fallback> static void mul_mat_q(){} }
`

const q3q8StreamKMain = `
using namespace fak_mmq_test;
static_assert(ITER_K==256 && (qk==256 || qk==32),"witness covers Q3_K and Q8_0 iteration geometry");
constexpr int J=128;
static std::vector<std::pair<int,int>> ranges(int tiles,int k,int grid){
 std::vector<std::pair<int,int>> result;
 for(int b=0;b<grid;++b){auto r=interval(tiles,k,grid,b);if(r.first!=r.second)result.push_back(r);}
 return result;
}
static std::vector<float> fixed(int tiles,int k,int grid){
 std::vector<float> tmp(size_t(grid)*J*I,std::numeric_limits<float>::quiet_NaN()),dst(size_t(tiles)*J*I,0.0f);
 const float partials[]={1.0e10f,1.0f,-1.0e10f,0.5f,-3.25f};
 for(int b=0;b<grid;++b){
  auto r=interval(tiles,k,grid,b);if(r.first==r.second)continue;
  if(r.second!=r.first+blocks_per_iter)return {};
  float *slot=r.second%k==0?dst.data()+size_t(r.first/k)*J*I:tmp.data()+size_t(b)*J*I;
  for(int z=0;z<J*I;++z)slot[z]=partials[(r.first/blocks_per_iter)%5];
 }
 // Every fixup lane executes its actual source; unused slots remain NaN.
 if(tiles%grid){
  gridDim=dim3(grid,1,1);
  for(int b=0;b<grid;++b)for(int by=0;by<I/warp_size;++by)
   for(int ty=0;ty<(nthreads/2)/warp_size;++ty)for(int tx=0;tx<warp_size;++tx){
    blockIdx=dim3(b,by,0);threadIdx=dim3(tx,ty,0);
    mul_mat_q_stream_k_fixup<J,false>(dst.data(),tmp.data(),init_fastdiv_values(k),I,tiles*J,I,init_fastdiv_values(tiles));
   }
 }
 return dst;
}
static void reset(){main_calls=fixup_calls=allocations=frees=0;main_grid=fixup_grid=0;allocated=0;}
int main(){
 struct Case{int tiles,k,sms;};
 const Case cases[]={{1,1,132},{1,2,132},{3,5,132},{4,16,80},{9,8,80},{10,8,80},{12,8,80},{71,1,80},{72,1,80},{80,1,80},{81,1,80},{1,81,80}};
 for(const Case c:cases){
  reset();device_info info={};info.nsm=c.sms;
  int rc=launch_mul_mat_q<J,false>({},nullptr,nullptr,I,ITER_K*c.k,J*c.tiles,info,0);
  int kblocks=c.k*blocks_per_iter;
  int work=c.tiles*c.k,waves=(c.tiles+c.sms-1)/c.sms;
  int old_grid=100*c.tiles/(c.sms*waves)>=90?c.tiles:c.sms;
  int want=old_grid==c.tiles?c.tiles:(work<c.sms?work:c.sms);
  bool fix=c.tiles%want!=0;
  if(rc||main_calls!=1||main_grid!=(unsigned)want||fixup_calls!=(int)fix||
     allocations!=(int)fix||frees!=(int)fix||(fix&&(fixup_grid!=main_grid||allocated!=size_t(want)*J*I*sizeof(float)))){
   std::fprintf(stderr,"grid/storage mismatch tiles=%d k=%d sms=%d got=%u want=%d rc=%d\n",c.tiles,c.k,c.sms,main_grid,want,rc);return 1;
  }
  if(ranges(c.tiles,kblocks,old_grid)!=ranges(c.tiles,kblocks,want)){std::fputs("nonempty partition changed\n",stderr);return 1;}
  if(want!=old_grid){
   auto before=fixed(c.tiles,kblocks,old_grid),after=fixed(c.tiles,kblocks,want);
   if(before.empty()||before.size()!=after.size()||std::memcmp(before.data(),after.data(),before.size()*sizeof(float))){std::fputs("fixup output/order changed\n",stderr);return 1;}
  }
 }
 reset();device_info info={};info.nsm=132;
 // Exactly 2^30 raw quant blocks for either format. Q8 has only 2^27
 // iterations here, so checking the divided iteration count would miss it.
 if(launch_mul_mat_q<J,false>({},nullptr,nullptr,32768,(1<<22)*qk,128,info,0)!=cudaErrorInvalidValue||main_calls||fixup_calls||allocations){std::fputs("overflow refusal changed\n",stderr);return 1;}
 std::puts("PASS Q3_K/Q8_0 actual-source grid, partition, fixup and overflow; CUDA/device/performance unrun");
}
`
