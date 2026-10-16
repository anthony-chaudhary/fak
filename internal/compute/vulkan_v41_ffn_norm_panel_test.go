//go:build linux

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

// Actual loan/claim source with synthetic driver transport. This proves no
// allocation call in the selected helper and ownership/order, not GPU behavior.
// fak-test:runtime medium est=2s lane=optin
// fak-test:justify Compiles the bounded native FFN pool loan and fault transitions.
func TestVulkanV41FFNPoolPanel(t *testing.T) {
	if os.Getenv("FAK_TEST_VULKAN_FFN_POOL_PANEL") != "1" {
		t.Skip("explicit native extraction witness")
	}
	compiler, err := exec.LookPath("c++")
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile("vulkan_shim.cpp")
	if err != nil {
		t.Fatal(err)
	}
	source := string(data)
	start := strings.Index(source, "static bool v41NormPanelLive(")
	end := strings.Index(source, "// fvk_v41_d2h for several regions behind one host fence")
	if start < 0 || end <= start {
		t.Fatal("native pool panel definition missing")
	}
	extract := func(signature string) string {
		i := strings.Index(source, signature)
		if i < 0 {
			t.Fatalf("missing %s", signature)
		}
		open := i + strings.Index(source[i:], "{")
		depth := 0
		for end := open; end < len(source); end++ {
			if source[end] == '{' {
				depth++
			}
			if source[end] == '}' {
				depth--
				if depth == 0 {
					return source[i:end+1] + "\n"
				}
			}
		}
		t.Fatalf("unterminated %s", signature)
		return ""
	}
	capacity := extract("size_t stageCapacity(") + extract("Buffer* stagingBuffer(") + extract("bool stepUploadReserve(")
	dir := t.TempDir()
	cpp := filepath.Join(dir, "witness.cpp")
	bin := filepath.Join(dir, "witness")
	if err = os.WriteFile(cpp, []byte(ffnPoolPreamble+capacity+source[start:end]+ffnPoolMain), 0600); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if out, err := exec.CommandContext(ctx, compiler, "-std=c++17", "-O0", "-g0", "-Wall", "-Wextra", cpp, "-o", bin).CombinedOutput(); err != nil {
		t.Fatalf("compile: %v\n%s", err, out)
	}
	if out, err := exec.CommandContext(ctx, bin).CombinedOutput(); err != nil {
		t.Fatalf("witness: %v\n%s", err, out)
	}
}

const ffnPoolPreamble = `
#include <cstdint>
#include <cstddef>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>
#include <unordered_map>
#include <cstdio>
#include <cstdlib>
using VkCommandBuffer=uint64_t;using VkResult=int;
constexpr int VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT=1,VK_MEMORY_PROPERTY_HOST_COHERENT_BIT=2,VK_MEMORY_PROPERTY_HOST_CACHED_BIT=4,VK_BUFFER_USAGE_TRANSFER_SRC_BIT=1,VK_BUFFER_USAGE_TRANSFER_DST_BIT=2,FVK_D2H_STAGING_ALLOCATION_FAILED=-1000;
int g_dev=1;size_t g_maxBufferBytes=0;bool g_debugD2HStagingFailureOnce=false,g_preferCachedTransferStage=false;
constexpr uint64_t VK_NULL_HANDLE=0;
constexpr int VK_SUCCESS=0,VK_ERROR_UNKNOWN=-13;
struct Buffer { uint64_t buf=0,mem=0; size_t bytes=0,allocationBytes=0,weightArenaBlock=0; uint64_t allocationID=0,deviceGeneration=1,nativeCheckoutGeneration=1; bool allocationDeviceLocal=true,weightArenaBound=false,offloadPage=false;void* mapped=nullptr; };
struct Block{uint64_t deviceGeneration=1,mem=0;};
constexpr size_t POOL_BUCKET_CAP=64;
std::vector<Block> g_weightArena;
std::unordered_map<size_t,std::vector<Buffer*>> g_pool;
size_t g_poolCount=0,g_stageCap=0,g_upArenaCap=0,g_upArenaUsed=0;
Buffer *g_stage=nullptr,*g_upArena=nullptr;
void *g_stageMapped=nullptr,*g_upArenaMapped=nullptr;
bool g_ready=true,g_memoryDeviceGenerationReady=true,g_v41SubmissionPendingFailure=false,g_batching=false;
uint64_t g_memoryDeviceGeneration=1;
int g_submissionStatus=0;
struct{void* stage=nullptr;uint64_t cmd=0;}g_restore;
struct{int oneShotH2D=0;}g_dp;
uint64_t g_h2dCount=0,g_h2dBytes=0;
bool g_transferCountersValid=true;
struct Kernel{};Kernel g_kern[1];constexpr int K_RMSNORM=0;
struct VkBufferCopy{size_t a,b,n;};
Buffer* B(const void* p){return (Buffer*)p;}
int calls=0,dispatches=0,frees=0,allocations=0,fault=0;
bool step=false;std::vector<Buffer*> quarantine;
std::vector<float> uploaded;
void need(bool ok,const char* s){if(!ok){std::fprintf(stderr,"%s\n",s);std::abort();}}
// Any allocation reference introduced into the extracted helper trips this trap.
Buffer* allocBuffer(...){++allocations;std::abort();}
void vkUnmapMemory(...) {std::abort();}
int vkMapMemory(...) {std::abort();}
void destroyBuffer(Buffer*){std::abort();}
void batchFlush(){std::abort();}
void batchBegin(){std::abort();}
bool stepUploadReserve(size_t,size_t*);
Buffer* stagingBuffer(size_t,int*);
int vkAllocateMemory(...){++allocations;std::abort();}
void advanceNativeCheckoutGeneration(Buffer*b){++b->nativeCheckoutGeneration;}
bool stepBatchRecording(){return step&&g_batching&&!g_v41SubmissionPendingFailure;}
bool stepRecordUpload(Buffer*,const void*p,size_t n){++calls;if(fault==1){g_v41SubmissionPendingFailure=true;return false;}size_t off=0;need(stepUploadReserve(n,&off),"reserve failed");uploaded.assign((const float*)p,(const float*)p+n/4);g_upArenaUsed=off+n;return true;}
VkCommandBuffer v41BeginCmdChecked(){++calls;if(fault==2){g_v41SubmissionPendingFailure=true;return 0;}return 1;}
void vkCmdCopyBuffer(VkCommandBuffer,uint64_t,uint64_t,int,const VkBufferCopy*r){uploaded.assign((float*)g_stageMapped,(float*)g_stageMapped+r->n/4);}
void dpOneShot(int){}
bool v41EndSubmitWaitChecked(VkCommandBuffer,bool){if(fault==3){g_v41SubmissionPendingFailure=true;return false;}return true;}
bool checkedCounterAdd(uint64_t&v,size_t n){v+=n;return true;}
bool dispatch(Kernel&,Buffer**,const void*,uint32_t,uint32_t,uint32_t,bool checked){need(checked,"unchecked dispatch");++dispatches;if(fault==4){g_v41SubmissionPendingFailure=true;return false;}return true;}
int fvk_v41_d2h(void*host,Buffer*,size_t n){++calls;int st=0;need(stagingBuffer(n,&st)!=nullptr,"unexpected staging miss");if(fault==5){g_v41SubmissionPendingFailure=true;return -4;}need(uploaded.size()==n/4,"upload must precede overwrite");for(size_t i=0;i<n/4;++i)((float*)host)[i]=uploaded[i]+10;if(fault==6){g_batching=false;g_submissionStatus=-13;}return 0;}
void fvk_free(Buffer*b){++frees;if(g_v41SubmissionPendingFailure){quarantine.push_back(b);return;}g_pool[b->bytes].push_back(b);++g_poolCount;}
`
const ffnPoolMain = `
int main(){
 constexpr int R=64,H=128;constexpr size_t N=R*H,BYTES=N*4;
 Buffer x,y,w,stage,arena;std::vector<float>host(N),staging(N),upload(N);int status=0;
 auto reset=[&](bool batching=false){
  x={1,11,BYTES,BYTES,0,1};y={2,12,BYTES,BYTES,0,2};w={3,13,H*4,H*4,0,3};stage={4,14,BYTES,BYTES,0,4};arena={5,15,BYTES,BYTES,0,5};
  g_pool.clear();g_pool[BYTES]={&y,&x};g_poolCount=2;g_stage=&stage;g_stageMapped=staging.data();g_stageCap=BYTES;g_upArena=&arena;g_upArenaMapped=upload.data();g_upArenaCap=BYTES;g_upArenaUsed=0;
  g_ready=g_memoryDeviceGenerationReady=true;g_memoryDeviceGeneration=1;g_v41SubmissionPendingFailure=false;g_submissionStatus=0;step=batching;g_batching=batching;calls=dispatches=frees=allocations=fault=0;quarantine.clear();uploaded.clear();for(size_t i=0;i<N;++i)host[i]=float(i);status=99;
 };
 auto run=[&](){return fvk_v41_try_pool_norm_panel(host.data(),&w,R,H,1e-5f,&status);};
 auto miss=[&](){auto before=host;auto pool=g_pool[BYTES];auto gx=x.nativeCheckoutGeneration,gy=y.nativeCheckoutGeneration;need(run()==0,"expected miss");need(host==before&&g_pool[BYTES]==pool&&x.nativeCheckoutGeneration==gx&&y.nativeCheckoutGeneration==gy&&calls==0&&dispatches==0&&allocations==0&&frees==0,"miss had effects");};
 for(bool batching:{false,true}){reset(batching);need(run()==1&&status==0,"selected success");need(dispatches==1&&frees==2&&allocations==0&&x.nativeCheckoutGeneration==2&&y.nativeCheckoutGeneration==2,"allocation/claim/release count");for(size_t i=0;i<N;++i)need(host[i]==float(i)+10,"same-host input destroyed before staging");}
 reset(true);g_pool.clear();x.bytes=y.bytes=BYTES/2;g_pool[BYTES/2]={&y,&x};need(fvk_v41_try_pool_norm_panel(host.data(),&w,32,H,1e-5f,&status)==1&&allocations==0&&dispatches==1,"32 row fallback-sized pair");
 reset();g_pool[BYTES].pop_back();g_poolCount=1;miss();
 reset();g_pool[BYTES]={&x,&x};miss();
 reset();y.mem=x.mem;miss();
 reset();y.buf=x.buf;miss();
 reset();w.mem=x.mem;miss();
 reset();stage.mem=y.mem;miss();
 reset(true);arena.mem=x.mem;miss();
 reset();x.deviceGeneration=2;miss();
 reset();x.nativeCheckoutGeneration=0;miss();
 reset();x.nativeCheckoutGeneration=std::numeric_limits<uint64_t>::max();miss();
 reset();g_stageCap=BYTES-1;miss();
 reset();g_stageMapped=nullptr;miss();
 reset(true);g_upArenaUsed=1;miss();
 reset(true);g_upArenaUsed=std::numeric_limits<size_t>::max();miss();
 reset(true);g_upArenaMapped=nullptr;miss();
 reset();g_batching=true;miss();
 reset();g_pool[BYTES].insert(g_pool[BYTES].begin(),&x);g_poolCount=3;miss();
 for(int f:{1,2,3,4,5,6}){reset(f==1||f==6);fault=f;need(run()==-1,"selected fault declined/published");need(frees==2&&allocations==0,"fault cleanup count");if(f!=6)need(quarantine.size()==2,"uncertain backing recycled");}
 reset();g_v41SubmissionPendingFailure=true;need(run()==-1&&calls==0,"poison fell back");
 std::puts("pool FFN: success/miss/alias/generation/capacity/transport/resume; zero allocation attempts");
}
`
