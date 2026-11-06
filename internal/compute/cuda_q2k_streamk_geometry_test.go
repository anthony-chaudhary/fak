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

// Execute the actual Q2 launcher, selector, partition and fixup source. Only
// CUDA launch syntax is replaced with recording stubs; MMA math is not emulated.
// fak-test:runtime medium est=3s lane=default
// fak-test:justify Compiles bounded extracted C++17 geometry/fixup source with no GPU operations.
func TestCUDAQ2KStreamKGeometry(t *testing.T) {
	compiler, err := exec.LookPath("c++")
	if err != nil {
		t.Skip("C++17 compiler unavailable; Q2 Stream-K host witness unrun")
	}
	b, err := os.ReadFile("cuda/q2k_mmq.cuh")
	if err != nil {
		t.Fatal(err)
	}
	source := string(b)
	t.Logf("production cuda/q2k_mmq.cuh sha256=%x", sha256.Sum256(b))
	extract := func(signature string) string {
		t.Helper()
		if strings.Count(source, signature) != 1 {
			t.Fatalf("expected one actual definition %q", signature)
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
		t.Helper()
		matches := regexp.MustCompile(pattern).FindAllString(source, -1)
		if len(matches) != 1 {
			t.Fatalf("expected one declaration %q, got %d", pattern, len(matches))
		}
		return matches[0]
	}
	var definitions []string
	for _, name := range []string{"warp_size", "QK_K", "QK8_1", "QI8_1", "MMQ_ITER_K", "QK8_1_MMQ", "MMQ_TILE_NE_K", "nthreads", "I", "nwarps", "qk", "ITER_K", "blocks_per_iter", "sram_stride"} {
		definitions = append(definitions, declaration(`(?m)^constexpr int `+name+` = [^;]+;`))
	}
	definitions = append(definitions,
		declaration(`(?m)^#define FAK_Q2_MMQ_PAD[^\n]+`),
		extract("struct block_q8_1_mmq {")+";",
		extract("struct device_info {")+";",
		extract("static uint3 init_fastdiv_values("),
		extract("static __device__ __forceinline__ uint32_t fastdiv("),
		extract("static __device__ __forceinline__ uint32_t fastmodulo("),
		extract("static __device__ __forceinline__ uint2 fast_div_modulo("),
		extract("static constexpr __host__ __device__ bool config_has_J("),
		extract("static size_t nbytes_shared("))
	launch := extract("template <int J, bool fallback>\nstatic int launch_mul_mat_q(")
	replace := func(old, replacement string) {
		t.Helper()
		if strings.Count(launch, old) != 1 {
			t.Fatalf("production launch changed: %s", old)
		}
		launch = strings.Replace(launch, old, replacement, 1)
	}
	replace(`FCUDA_TS("q2_mul_mat_q", block_nums_stream_k, mul_mat_q<J, fallback><<<block_nums_stream_k, block_dims, nbytes, stream>>>
        (x, y, dst, tmp_fixup, blocks_per_ne00_fd, nrows_x, ncols_dst, stride_row_x, ncols_dst, nrows_x, ntx_fd));`,
		`mock_launch(false, block_nums_stream_k, J);`)
	replace(`FCUDA_TS("q2_mul_mat_q_stream_k_fixup", block_nums_fixup, mul_mat_q_stream_k_fixup<J, fallback><<<block_nums_fixup, block_dims_fixup, 0, stream>>>
            (dst, tmp_fixup, blocks_per_ne00_fd, nrows_x, ncols_dst, nrows_x, ntx_fd));`,
		`mock_launch(true, block_nums_fixup, J);`)
	partitionStart := strings.Index(source, "    int kbc      = int64_t(blockIdx.x)")
	if partitionStart < 0 {
		t.Fatal("production partition start changed")
	}
	partitionEnd := strings.Index(source[partitionStart:], "    // kb0 == k index")
	if partitionEnd < 0 {
		t.Fatal("production partition end changed")
	}
	partition := source[partitionStart : partitionStart+partitionEnd]
	interval := `
static std::pair<int,int> interval(int tiles, int kblocks, int grid, int block) {
    const uint3 ntx = init_fastdiv_values(tiles);
    const int nty = 1;
    const uint3 blocks_per_ne00 = init_fastdiv_values(kblocks);
    blockIdx.x = block; gridDim.x = grid;
` + partition + `
    return {kbc, kbc_stop};
}
`
	actual := strings.Join(definitions, "\n") + "\n" + launch + "\n" +
		extract("template <bool fallback>\nstatic int mul_mat_q_switch_J(") + "\n" + interval +
		"\ntemplate <int J, bool fallback>\n" + extract("static __global__ void mul_mat_q_stream_k_fixup(")
	for _, forbidden := range []string{"<<<", "FCUDA_TS(", "cudaDeviceSynchronize"} {
		if strings.Contains(actual, forbidden) {
			t.Fatalf("uncovered device operation: %s", forbidden)
		}
	}
	fixture := q2kStreamKGeometryPreamble + "\nnamespace fak_q2k_mmq {\n" + actual + "\n}\n" + q2kStreamKGeometryMain
	t.Logf("actual-source host fixture sha256=%x", sha256.Sum256([]byte(fixture)))
	dir := t.TempDir()
	cpp, binary := filepath.Join(dir, "q2k_streamk.cpp"), filepath.Join(dir, "q2k_streamk.exe")
	if err := os.WriteFile(cpp, []byte(fixture), 0o600); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
	defer cancel()
	if out, err := exec.CommandContext(ctx, compiler, "-std=c++17", "-O0", cpp, "-o", binary).CombinedOutput(); err != nil {
		t.Fatalf("compile actual-source Q2 geometry: %v\n%s", err, out)
	}
	run, cancelRun := context.WithTimeout(t.Context(), 10*time.Second)
	defer cancelRun()
	out, err := exec.CommandContext(run, binary).CombinedOutput()
	if err != nil {
		t.Fatalf("actual-source Q2 geometry: %v\n%s", err, out)
	}
	const want = "Q2 selected J128 one-unit/132SM: main CTAs 132->1; fixup bytes 8650752->0; timing unmeasured\nPASS Q2 actual-source grid, partition, full fixup, overflow and storage controls; CUDA/device unrun\n"
	if strings.ReplaceAll(string(out), "\r\n", "\n") != want {
		t.Fatalf("incomplete actual-source Q2 witness: %s", out)
	}
	t.Log(strings.TrimSpace(string(out)))
}

const q2kStreamKGeometryPreamble = `
#include <cstdint>
#include <cstddef>
#include <climits>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <utility>
#include <limits>
#define __host__
#define __device__
#define __global__
#define __forceinline__
using cudaError_t=int;
using cudaStream_t=int;
using half=uint16_t;
struct dim3 { unsigned x,y,z; dim3(unsigned a=1,unsigned b=1,unsigned c=1):x(a),y(b),z(c){} };
struct uint2 { unsigned x,y; };
struct uint3 { unsigned x,y,z; };
static uint2 make_uint2(unsigned x,unsigned y){return {x,y};}
static uint3 make_uint3(unsigned x,unsigned y,unsigned z){return {x,y,z};}
static uint32_t __umulhi(uint32_t a,uint32_t b){return (uint64_t(a)*b)>>32;}
static dim3 blockIdx,threadIdx,gridDim;
constexpr int cudaSuccess=0,cudaErrorInvalidValue=1,cudaErrorInvalidConfiguration=9;
constexpr int cudaFuncAttributeMaxDynamicSharedMemorySize=8,FCUDA_SCRATCH_ALLOC_FAILED=92000;
static int main_calls,fixup_calls,allocation_calls,free_calls,sync_calls,checks,attribute_calls;
static int selected_j;
static dim3 main_grid,fixup_grid;
static size_t allocation_bytes;
alignas(16) static unsigned char storage[16];
template<class T> static int cudaFuncSetAttribute(T,int,int){++attribute_calls;return 0;}
static int cudaGetLastError(){++checks;return 0;}
static int cudaStreamSynchronize(cudaStream_t){++sync_calls;return 0;}
static int v41_checked_stream_error(int base,int err){return base+err;}
static void *scratch_malloc(size_t n){++allocation_calls;allocation_bytes=n;return storage;}
static void fcuda_free(void*){++free_calls;}
static void mock_launch(bool fixup,dim3 grid,int j) {
 if(fixup){++fixup_calls;fixup_grid=grid;}else{++main_calls;main_grid=grid;selected_j=j;}
}
namespace fak_q2k_mmq { template<int J,bool fallback> static void mul_mat_q(){} }
static void reset(){
 main_calls=fixup_calls=allocation_calls=free_calls=sync_calls=checks=attribute_calls=0;
 selected_j=0;main_grid=fixup_grid=dim3(0,0,0);allocation_bytes=0;
}
#define CHECK(x) do{if(!(x)){std::fprintf(stderr,"line %d: %s\n",__LINE__,#x);return 1;}}while(0)
`

const q2kStreamKGeometryMain = `
static std::vector<std::pair<int,int>> nonempty(int tiles,int kblocks,int grid) {
 std::vector<std::pair<int,int>> out;
 for(int b=0;b<grid;++b){auto r=fak_q2k_mmq::interval(tiles,kblocks,grid,b);if(r.first!=r.second)out.push_back(r);}
 return out;
}
static float contribution(int unit,int element) {
 const float values[]={1.0e10f,1.0f,-1.0e10f,0.5f,-3.25f};
 return values[(unit+element)%5];
}
// Changed grids have exactly one useful K block per nonempty CTA. Simulate only
// its writeback; run the complete extracted production fixup for every thread.
template<int J> static std::vector<float> fixed(int tiles,int kblocks,int grid) {
 constexpr int I=fak_q2k_mmq::I,guard=8;
 const float canary=1234567.0f;
 std::vector<float> tmp(size_t(grid)*J*I+2*guard,std::numeric_limits<float>::quiet_NaN());
 std::vector<float> dst(size_t(tiles)*J*I+2*guard,0.0f);
 for(int i=0;i<guard;++i){tmp[i]=tmp[tmp.size()-1-i]=canary;dst[i]=dst[dst.size()-1-i]=canary;}
 float *tp=tmp.data()+guard,*dp=dst.data()+guard;
 for(int b=0;b<grid;++b) {
  auto r=fak_q2k_mmq::interval(tiles,kblocks,grid,b);
  if(r.first==r.second)continue;
  if(r.second!=r.first+1)return {};
  float *slot=(r.second%kblocks==0)?dp+size_t(r.first/kblocks)*J*I:tp+size_t(b)*J*I;
  for(int z=0;z<J*I;++z)slot[z]=contribution(r.first,z);
 }
 const auto before=tmp;
 if(tiles%grid) {
  gridDim=dim3(grid,I/32,1);
  for(int b=0;b<grid;++b)for(int by=0;by<I/32;++by)
   for(int ty=0;ty<4;++ty)for(int tx=0;tx<32;++tx) {
    blockIdx=dim3(b,by,0);threadIdx=dim3(tx,ty,0);
    fak_q2k_mmq::mul_mat_q_stream_k_fixup<J,false>(dp,tp,fak_q2k_mmq::init_fastdiv_values(kblocks),I,tiles*J,I,fak_q2k_mmq::init_fastdiv_values(tiles));
   }
 }
 if(std::memcmp(tmp.data(),before.data(),tmp.size()*sizeof(float)))return {};
 for(int i=0;i<guard;++i)if(dst[i]!=canary||dst[dst.size()-1-i]!=canary)return {};
 for(int tile=0;tile<tiles;++tile)for(int z=0;z<J*I;++z) {
  float sum=0.0f;
  for(int k=kblocks-2;k>=0;--k)sum+=contribution(tile*kblocks+k,z);
  float want=contribution((tile+1)*kblocks-1,z);
  if(kblocks>1)want+=sum;
  if(!std::isfinite(dp[size_t(tile)*J*I+z])||std::memcmp(&dp[size_t(tile)*J*I+z],&want,sizeof(float)))return {};
 }
 return dst;
}
template<int J> static int cases() {
 struct C{int tiles,kblocks,sms;};
 const C cases[]={
  {1,1,132},{1,2,132},{3,5,132},{4,16,80},
  {9,8,80},{10,8,80},{12,8,80},
  {71,1,80},{72,1,80},{80,1,80},{81,1,80},
  {1,79,80},{1,80,80},{1,81,80},{1,1,1},
  {32769,32767,132}
 };
 for(const C c:cases) {
  reset();
  fak_q2k_mmq::device_info info={};info.nsm=c.sms;info.smpbo=1<<20;
  const int rc=fak_q2k_mmq::launch_mul_mat_q<J,false>(nullptr,nullptr,nullptr,128,256*c.kblocks,J*c.tiles,info,0);
  const int64_t work=int64_t(c.tiles)*c.kblocks;
  const int waves=(c.tiles+c.sms-1)/c.sms;
  const int old_grid=100LL*c.tiles/(int64_t(c.sms)*waves)>=90?c.tiles:c.sms;
  const int want=old_grid==c.tiles?c.tiles:int(work<c.sms?work:c.sms);
  CHECK(rc==0&&main_calls==1&&selected_j==J&&main_grid.x==unsigned(want)&&main_grid.y==1&&main_grid.z==1);
  const bool fix=c.tiles%want!=0;
  CHECK(fixup_calls==int(fix)&&allocation_calls==int(fix)&&free_calls==int(fix)&&!sync_calls);
  CHECK(attribute_calls==1&&checks==1+int(fix));
  CHECK(allocation_bytes==(fix?size_t(want)*J*128*sizeof(float):0));
  if(fix)CHECK(fixup_grid.x==main_grid.x&&fixup_grid.y==4&&fixup_grid.z==1);
  CHECK(nonempty(c.tiles,c.kblocks,old_grid)==nonempty(c.tiles,c.kblocks,want));
  if(want!=old_grid) {
   CHECK(work<c.sms&&want==work);
   auto before=fixed<J>(c.tiles,c.kblocks,old_grid),after=fixed<J>(c.tiles,c.kblocks,want);
   CHECK(!before.empty()&&before.size()==after.size()&&!std::memcmp(before.data(),after.data(),before.size()*sizeof(float)));
  }
 }
 return 0;
}
int main() {
 static_assert(fak_q2k_mmq::qk==256&&fak_q2k_mmq::ITER_K==256&&fak_q2k_mmq::blocks_per_iter==1,"Q2 one-unit proof must be revisited");
 CHECK(cases<8>()==0&&cases<128>()==0);
 // Invoke the real J selector, not only a fixed-J specialization.
 for(bool fallback:{false,true}) {
  reset();fak_q2k_mmq::device_info info={};info.nsm=132;info.smpbo=1<<20;
  int rc=fallback
   ?fak_q2k_mmq::mul_mat_q_switch_J<true>(nullptr,nullptr,nullptr,127,256,128,info,0)
   :fak_q2k_mmq::mul_mat_q_switch_J<false>(nullptr,nullptr,nullptr,128,256,128,info,0);
  CHECK(rc==0&&selected_j==128&&main_calls==1&&main_grid.x==1&&!fixup_calls&&!allocation_calls&&!free_calls&&!sync_calls);
 }
 // Raw K work must be refused before integer grid narrowing, launch or scratch.
 for(int kb:{32768,32769,131072}) {
  reset();fak_q2k_mmq::device_info info={};info.nsm=132;info.smpbo=1<<20;
  const int rc=fak_q2k_mmq::launch_mul_mat_q<128,false>(nullptr,nullptr,nullptr,128,256*kb,128*32768,info,0);
  CHECK(rc==cudaErrorInvalidValue&&!main_calls&&!fixup_calls&&!allocation_calls&&!free_calls&&!sync_calls&&!checks);
 }
 std::printf("Q2 selected J128 one-unit/132SM: main CTAs 132->1; fixup bytes 8650752->0; timing unmeasured\n");
 std::printf("PASS Q2 actual-source grid, partition, full fixup, overflow and storage controls; CUDA/device unrun\n");
}
`
