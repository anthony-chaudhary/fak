//go:build v41_bf16_boundary_witness && vulkan && (windows || linux) && cgo

package compute

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strings"
	"testing"
	"time"
)

// Exercise the complete, hash-bound native prefix with typed driver interception.
// Synthetic dispatch outputs test transport/ownership, not shader arithmetic.
// Go adapter locking, envelope priority and atomic result publication are separate
// witnesses; this fixture never initializes a device or links a Vulkan loader.
// fak-test:runtime medium est=30s lane=optin
func TestVulkanV41QueryPrefixOrchestration(t *testing.T) {
	if os.Getenv("FAK_TEST_V41_QUERY_PREFIX_ORCHESTRATION") != "1" {
		t.Fatal("explicit query-prefix orchestration opt-in required")
	}
	if os.Getenv("FAK_VULKAN_SPIRV") != "" {
		t.Fatal("orchestration witness must not request Vulkan initialization")
	}
	if _, registered := Lookup("vulkan"); registered {
		t.Fatal("orchestration process unexpectedly registered a Vulkan device")
	}
	// Startup isolation is also required in the execution receipt; this test
	// runs after Go package initialization and cannot undo an earlier attempt.
	sourceDir := os.Getenv("FAK_V41_QUERY_PREFIX_WITNESS_SOURCE_DIR")
	if !filepath.IsAbs(sourceDir) {
		t.Fatal("explicit absolute native source directory required")
	}
	tmp := t.TempDir()
	var sourceText strings.Builder
	for _, file := range []struct{ name, env string }{
		{"vulkan_shim.cpp", "FAK_V41_QUERY_PREFIX_WITNESS_SHIM_SHA256"},
		{"vulkan_backend.h", "FAK_V41_QUERY_PREFIX_WITNESS_HEADER_SHA256"},
	} {
		raw, err := os.ReadFile(filepath.Join(sourceDir, file.name))
		if err != nil {
			t.Fatal(err)
		}
		digest := sha256.Sum256(raw)
		if len(os.Getenv(file.env)) != 64 || hex.EncodeToString(digest[:]) != os.Getenv(file.env) {
			t.Fatalf("%s differs from the reviewed native source receipt", file.name)
		}
		// An exact byte snapshot makes a concurrent checkout edit irrelevant to
		// this compilation. No production body is extracted or rewritten.
		if err := os.WriteFile(filepath.Join(tmp, file.name), raw, 0o600); err != nil {
			t.Fatal(err)
		}
		t.Logf("included %s sha256=%x", file.name, digest)
		sourceText.Write(raw)
		sourceText.WriteByte('\n')
	}
	allowed := map[string]bool{}
	for _, name := range strings.Fields(`vkAllocateDescriptorSets vkUpdateDescriptorSets vkAllocateCommandBuffers vkBeginCommandBuffer vkEndCommandBuffer vkFreeCommandBuffers vkResetFences vkQueueSubmit vkWaitForFences vkCmdBindPipeline vkCmdBindDescriptorSets vkCmdPushConstants vkCmdFillBuffer vkCmdPipelineBarrier vkCmdDispatch vkCmdCopyBuffer vkCreateBuffer vkGetBufferMemoryRequirements vkAllocateMemory vkBindBufferMemory vkMapMemory vkUnmapMemory vkDestroyBuffer vkFreeMemory`) {
		allowed[name] = true
	}
	names := map[string]bool{}
	for _, name := range regexp.MustCompile(`\bvk[A-Z][A-Za-z0-9_]*\b`).FindAllString(sourceText.String(), -1) {
		names[name] = true
	}
	for name := range allowed {
		if !names[name] {
			t.Fatalf("reviewed interception boundary is missing %s", name)
		}
	}
	ordered := make([]string, 0, len(names))
	for name := range names {
		ordered = append(ordered, name)
	}
	sort.Strings(ordered)
	var fixture strings.Builder
	fixture.WriteString(v41QueryPrefixOrchestrationPrefix)
	for _, name := range ordered {
		if allowed[name] {
			fmt.Fprintf(&fixture, "static_assert(std::is_same_v<decltype(&prefix_%s), PFN_%s>);\n#define %s prefix_%s\n", name, name, name, name)
		} else {
			// Official PFN types preserve the real argument/return ABI. Unknown
			// source tokens fail compilation instead of silently escaping.
			fmt.Fprintf(&fixture, "#define %s PrefixAbort<PFN_%s>::call\n", name, name)
		}
	}
	fixture.WriteString("\n#define FAK_V41_BF16_BOUNDARY_WITNESS 1\n#include \"vulkan_shim.cpp\"\n")
	fixture.WriteString(v41QueryPrefixOrchestrationCases)
	raw := []byte(fixture.String())
	path := filepath.Join(tmp, "query-prefix-orchestration.cpp")
	if err := os.WriteFile(path, raw, 0o600); err != nil {
		t.Fatal(err)
	}
	t.Logf("orchestration fixture sha256=%x intercepted=%d allowed=%d", sha256.Sum256(raw), len(ordered), len(allowed))
	compiler, err := exec.LookPath("c++")
	if err != nil {
		t.Fatal("requested witness requires a C++17 compiler: ", err)
	}
	binary := filepath.Join(tmp, "query-prefix-orchestration")
	if runtime.GOOS == "windows" {
		binary += ".exe"
	}
	ctx, cancel := context.WithTimeout(t.Context(), 120*time.Second)
	defer cancel()
	// No Vulkan loader or user-provided linker flags are permitted here.
	args := []string{"-std=c++17", "-O0", "-g0"}
	if include := os.Getenv("FAK_V41_QUERY_PREFIX_WITNESS_VULKAN_INCLUDE"); include != "" {
		if !filepath.IsAbs(include) {
			t.Fatal("Vulkan header include root must be absolute")
		}
		if _, err := os.Stat(filepath.Join(include, "vulkan", "vulkan.h")); err != nil {
			t.Fatal(err)
		}
		args = append(args, "-I", include)
	}
	args = append(args, path, "-o", binary)
	cmd := exec.CommandContext(ctx, compiler, args...)
	cmd.Dir = tmp
	cmd.Env = append(os.Environ(), "FAK_VULKAN_DISPATCH_PROFILE=0", "FAK_VULKAN_TIMESTAMP_PROFILE=0")
	if output, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("compile real-shim fixture: %v\n%s", err, output)
	}
	ctxRun, cancelRun := context.WithTimeout(t.Context(), 10*time.Second)
	defer cancelRun()
	cmd = exec.CommandContext(ctxRun, binary)
	cmd.Dir = tmp
	cmd.Env = append(os.Environ(), "FAK_VULKAN_DISPATCH_PROFILE=0", "FAK_VULKAN_TIMESTAMP_PROFILE=0")
	output, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("real-shim orchestration: %v\n%s", err, output)
	}
	t.Logf("%s", output)
}

const v41QueryPrefixOrchestrationPrefix = `
#include <vulkan/vulkan.h>
#include <array>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <type_traits>
#include <vector>

namespace prefix_fixture {
[[noreturn]] void die(const char* why) { std::fprintf(stderr,"prefix orchestration: %s\n",why); std::abort(); }
void require(bool ok,const char* why) { if (!ok) die(why); }
template<class T> T handle(uintptr_t id) { return (T)id; }
struct Op { int kind; VkBuffer source{}, dest{}, status{}; uint32_t count{}, site{}, step{}; };
struct Command {
    bool begun=false, ended=false, dependency=false;
    int dispatchCount=0, fillCount=0, copyCount=0;
    VkPipeline pipeline{};
    VkPipelineLayout layout{};
    std::vector<VkBuffer> bindings;
    std::vector<uint32_t> push;
    std::set<VkBuffer> filled;
    std::set<VkDescriptorSet> sets;
    std::vector<Op> ops;
};
std::map<VkCommandBuffer,Command> commands;
std::map<VkDescriptorSet,std::vector<VkBuffer>> descriptors;
std::map<VkBuffer,std::vector<uint32_t>> memory;
std::map<std::string,int> calls;
std::map<uint32_t,uint32_t> statusFaults;
std::map<VkBuffer,VkDeviceSize> createdBuffers;
std::set<VkDeviceMemory> allocatedMemory;
std::string failure;
int failureOrdinal=1;
VkResult failureResult=VK_ERROR_OUT_OF_HOST_MEMORY;
uintptr_t nextHandle=1000;
VkCommandBuffer submitted{};
std::array<VkBuffer,15> roles{};
VkBuffer stageHandle{};
std::array<VkPipeline,3> pipelines{};
std::array<VkPipelineLayout,3> layouts{};
std::array<VkDescriptorSetLayout,3> dsls{};
std::vector<uint32_t> stage(16384);
std::vector<float> host, hostBefore;
int hidden=193, rank=65, fills=0, dispatches=0, copies=0, barriers=0;
uint32_t epsilonBits=0;
void descriptorsHeld(); // Defined after including the complete production shim.
VkResult call(const char* api) {
    const int n=++calls[api];
    return failure==api && (failureOrdinal==0 || n==failureOrdinal) ? failureResult : VK_SUCCESS;
}
Command& command(VkCommandBuffer id) {
    auto it=commands.find(id); require(it!=commands.end(),"unknown or twice-released command"); return it->second;
}
std::vector<uint32_t>& words(VkBuffer id) { return id==stageHandle ? stage : memory.at(id); }
uint32_t outputWord(int step) { return 0x3f000000u+uint32_t(step+1)*0x10000u; }
void execute(VkCommandBuffer id) {
    for (const auto& op:command(id).ops) {
        if (op.kind==0) std::fill_n(words(op.dest).begin(),4,UINT32_MAX);
        else if (op.kind==2) {
            require(words(op.source).size()>=op.count && words(op.dest).size()>=op.count,"copy bounds");
            std::copy_n(words(op.source).begin(),op.count,words(op.dest).begin());
        } else if (op.kind==1) {
            // Distinct finite sentinels model writes, never the numerical kernel.
            require(words(op.dest).size()>=op.count,"synthetic output bounds");
            std::fill_n(words(op.dest).begin(),op.count,outputWord(int(op.step)));
            if (op.status) {
                const auto& old=words(op.status);
                require(old==std::vector<uint32_t>(4,UINT32_MAX),"status did not follow an executed fresh fill");
                words(op.status)={0x42463143u,op.site,statusFaults.count(op.site) ? statusFaults.at(op.site) : UINT32_MAX,op.count};
            }
        } else die("unknown recorded operation");
    }
}
} // namespace prefix_fixture

// Every unallowlisted Vulkan API keeps its official PFN signature and aborts.
// This also resolves references from unused helper bodies without a loader.
template<class> struct PrefixAbort;
template<class R,class... A> struct PrefixAbort<R(VKAPI_PTR*)(A...)> {
    static VKAPI_ATTR R VKAPI_CALL call(A...) { prefix_fixture::die("unexpected Vulkan API reached"); }
};
static VKAPI_ATTR VkResult VKAPI_CALL prefix_vkAllocateDescriptorSets(VkDevice,const VkDescriptorSetAllocateInfo* info,VkDescriptorSet* out) {
    using namespace prefix_fixture;
    require(info && out && info->descriptorSetCount==1 && info->pSetLayouts,"descriptor allocation shape");
    require(std::find(dsls.begin(),dsls.end(),info->pSetLayouts[0])!=dsls.end(),"descriptor layout identity");
    const auto r=call("vkAllocateDescriptorSets");
    *out=r==VK_SUCCESS ? handle<VkDescriptorSet>(++nextHandle) : VK_NULL_HANDLE;
    if (r==VK_SUCCESS) descriptors[*out]={};
    return r;
}
static VKAPI_ATTR void VKAPI_CALL prefix_vkUpdateDescriptorSets(VkDevice,uint32_t n,const VkWriteDescriptorSet* w,uint32_t nc,const VkCopyDescriptorSet*) {
    using namespace prefix_fixture; call("vkUpdateDescriptorSets");
    require(n==3 && w && nc==0,"exact three descriptor bindings");
    for (uint32_t i=0;i<n;++i) {
        require(w[i].dstSet==w[0].dstSet && w[i].dstBinding==i && w[i].descriptorCount==1 && w[i].descriptorType==VK_DESCRIPTOR_TYPE_STORAGE_BUFFER && w[i].pBufferInfo,"descriptor role/owner");
        require(w[i].pBufferInfo->offset==0 && w[i].pBufferInfo->range==VK_WHOLE_SIZE,"descriptor byte range");
        auto& b=descriptors.at(w[i].dstSet); b.resize(3); b[i]=w[i].pBufferInfo->buffer;
    }
}
static VKAPI_ATTR VkResult VKAPI_CALL prefix_vkAllocateCommandBuffers(VkDevice,const VkCommandBufferAllocateInfo* info,VkCommandBuffer* out) {
    using namespace prefix_fixture;
    require(info && out && info->commandBufferCount==1 && info->level==VK_COMMAND_BUFFER_LEVEL_PRIMARY,"command allocation shape");
    const auto r=call("vkAllocateCommandBuffers");
    *out=r==VK_SUCCESS ? handle<VkCommandBuffer>(++nextHandle) : VK_NULL_HANDLE;
    if (r==VK_SUCCESS) commands[*out]={};
    return r;
}
static VKAPI_ATTR VkResult VKAPI_CALL prefix_vkBeginCommandBuffer(VkCommandBuffer id,const VkCommandBufferBeginInfo* info) {
    using namespace prefix_fixture;
    require(info && info->flags==VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,"command begin flags");
    const auto r=call("vkBeginCommandBuffer"); if (r==VK_SUCCESS) command(id).begun=true; return r;
}
static VKAPI_ATTR VkResult VKAPI_CALL prefix_vkEndCommandBuffer(VkCommandBuffer id) {
    using namespace prefix_fixture; require(command(id).begun,"end without begin");
    const auto r=call("vkEndCommandBuffer"); if (r==VK_SUCCESS) command(id).ended=true; return r;
}
static VKAPI_ATTR void VKAPI_CALL prefix_vkFreeCommandBuffers(VkDevice,VkCommandPool,uint32_t n,const VkCommandBuffer* ids) {
    using namespace prefix_fixture; call("vkFreeCommandBuffers");
    require(n==1 && ids && commands.erase(ids[0])==1,"exact-once command release");
}
static VKAPI_ATTR VkResult VKAPI_CALL prefix_vkResetFences(VkDevice,uint32_t n,const VkFence*) {
    using namespace prefix_fixture; require(n==1,"fence reset shape"); return call("vkResetFences");
}
static VKAPI_ATTR VkResult VKAPI_CALL prefix_vkQueueSubmit(VkQueue,uint32_t n,const VkSubmitInfo* info,VkFence) {
    using namespace prefix_fixture;
    require(n==1 && info && info->commandBufferCount==1,"single local submit");
    submitted=info->pCommandBuffers[0]; const auto& c=command(submitted);
    require(c.ended,"submit before end");
    if (c.dispatchCount) {
        require(c.dispatchCount==7 && c.fillCount==4 && c.copyCount==1 && c.sets.size()==7,"partial prefix submitted");
        descriptorsHeld();
    } else require(c.copyCount==1 && c.fillCount==0,"terminal read command shape");
    const auto r=call("vkQueueSubmit");
    // An uncertain submit may have written valid-looking bytes. The real
    // checked helper must still return the transport failure and quarantine.
    execute(submitted); return r;
}
static VKAPI_ATTR VkResult VKAPI_CALL prefix_vkWaitForFences(VkDevice,uint32_t n,const VkFence*,VkBool32 all,uint64_t timeout) {
    using namespace prefix_fixture;
    require(n==1 && all==VK_TRUE && timeout==UINT64_MAX && submitted!=VK_NULL_HANDLE,"wait shape");
    if (command(submitted).dispatchCount) descriptorsHeld();
    return call("vkWaitForFences");
}
static VKAPI_ATTR void VKAPI_CALL prefix_vkCmdBindPipeline(VkCommandBuffer id,VkPipelineBindPoint point,VkPipeline pipe) {
    using namespace prefix_fixture; call("vkCmdBindPipeline");
    require(point==VK_PIPELINE_BIND_POINT_COMPUTE,"pipeline bind point"); command(id).pipeline=pipe;
}
static VKAPI_ATTR void VKAPI_CALL prefix_vkCmdBindDescriptorSets(VkCommandBuffer id,VkPipelineBindPoint point,VkPipelineLayout layout,uint32_t first,uint32_t n,const VkDescriptorSet* sets,uint32_t nd,const uint32_t*) {
    using namespace prefix_fixture; call("vkCmdBindDescriptorSets");
    require(point==VK_PIPELINE_BIND_POINT_COMPUTE && first==0 && n==1 && sets && nd==0,"descriptor bind shape");
    auto& c=command(id); c.bindings=descriptors.at(sets[0]); c.layout=layout;
    require(c.sets.insert(sets[0]).second,"descriptor reused before local completion");
}
static VKAPI_ATTR void VKAPI_CALL prefix_vkCmdPushConstants(VkCommandBuffer id,VkPipelineLayout layout,VkShaderStageFlags flags,uint32_t offset,uint32_t size,const void* data) {
    using namespace prefix_fixture; call("vkCmdPushConstants"); auto& c=command(id);
    require(flags==VK_SHADER_STAGE_COMPUTE_BIT && offset==0 && (size==8 || size==12) && data && layout==c.layout,"push shape/layout");
    c.push.resize(size/4); std::memcpy(c.push.data(),data,size);
}
static VKAPI_ATTR void VKAPI_CALL prefix_vkCmdFillBuffer(VkCommandBuffer id,VkBuffer dest,VkDeviceSize offset,VkDeviceSize bytes,uint32_t value) {
    using namespace prefix_fixture; call("vkCmdFillBuffer"); ++fills; auto& c=command(id);
    require(c.begun && c.dispatchCount==0 && c.fillCount<4 && dest==roles[11+c.fillCount] && offset==0 && bytes==16 && value==UINT32_MAX,"fresh ordered four status fills");
    require(c.filled.insert(dest).second,"status owner filled twice"); ++c.fillCount; c.dependency=false;
    c.ops.push_back({0,{},dest,{ },0,0,0});
}
static VKAPI_ATTR void VKAPI_CALL prefix_vkCmdPipelineBarrier(VkCommandBuffer id,VkPipelineStageFlags src,VkPipelineStageFlags dst,VkDependencyFlags flags,uint32_t n,const VkMemoryBarrier* b,uint32_t nb,const VkBufferMemoryBarrier*,uint32_t ni,const VkImageMemoryBarrier*) {
    using namespace prefix_fixture; call("vkCmdPipelineBarrier"); ++barriers;
    const auto stages=VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT|VK_PIPELINE_STAGE_TRANSFER_BIT;
    require(src==stages && dst==stages && flags==0 && n==1 && b && nb==0 && ni==0,"conservative dependency shape");
    require(b[0].srcAccessMask==(VK_ACCESS_SHADER_WRITE_BIT|VK_ACCESS_TRANSFER_WRITE_BIT) && b[0].dstAccessMask==(VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT|VK_ACCESS_TRANSFER_READ_BIT),"dependency access masks");
    command(id).dependency=true;
}
static VKAPI_ATTR void VKAPI_CALL prefix_vkCmdDispatch(VkCommandBuffer id,uint32_t x,uint32_t y,uint32_t z) {
    using namespace prefix_fixture; call("vkCmdDispatch"); ++dispatches; auto& c=command(id);
    const int step=c.dispatchCount;
    require(step<7 && c.begun && c.fillCount==4 && c.copyCount==1 && c.dependency && y==1 && z==1,"prefix sequencing/dependency");
    const int kind[7]={0,1,0,2,0,1,0};
    const int bindings[7][3]={{3,4,11},{4,0,5},{5,6,12},{1,6,7},{7,8,13},{8,2,9},{9,10,14}};
    const uint32_t counts[7]={uint32_t(hidden),uint32_t(hidden),uint32_t(hidden),uint32_t(rank),uint32_t(rank),uint32_t(rank),uint32_t(rank)};
    const uint32_t sites[7]={3,0,4,0,5,0,6};
    require(c.pipeline==pipelines[kind[step]] && c.layout==layouts[kind[step]] && c.bindings.size()==3,"actual prefix kernel identities");
    for (int i=0;i<3;++i) require(c.bindings[i]==roles[bindings[step][i]],"prefix descriptor order");
    if (kind[step]==0) {
        require(c.push==std::vector<uint32_t>{counts[step],sites[step]} && x==std::min((counts[step]+63u)/64u,2u),"checked count/site/grid-stride push");
    } else if (kind[step]==1) {
        require(c.push==std::vector<uint32_t>{1u,counts[step],epsilonBits} && x==1,"existing RMSNorm push/geometry");
    } else {
        require(c.push==std::vector<uint32_t>{uint32_t(rank),uint32_t(hidden),1u} && x==1,"existing F32 matmul push/geometry");
    }
    const VkBuffer out=c.bindings[kind[step]==0 ? 1 : 2];
    c.ops.push_back({1,{},out,kind[step]==0 ? c.bindings[2] : VK_NULL_HANDLE,counts[step],sites[step],uint32_t(step)});
    ++c.dispatchCount; c.dependency=false;
}
static VKAPI_ATTR void VKAPI_CALL prefix_vkCmdCopyBuffer(VkCommandBuffer id,VkBuffer src,VkBuffer dst,uint32_t n,const VkBufferCopy* regions) {
    using namespace prefix_fixture; call("vkCmdCopyBuffer"); ++copies; auto& c=command(id);
    require(c.begun && n==1 && regions && regions[0].srcOffset==0 && regions[0].dstOffset==0 && regions[0].size%4==0,"copy region shape");
    if (src==stageHandle) {
        require(dst==roles[3] && c.copyCount==0 && c.fillCount==0 && c.dispatchCount==0 && regions[0].size==uint64_t(hidden)*4u,"one initial staging copy");
    } else {
        const auto status=std::find(roles.begin()+11,roles.end(),src)!=roles.end();
        const uint64_t size=status ? 16u : src==roles[6] ? uint64_t(hidden)*4u : src==roles[10] ? uint64_t(rank)*4u : 0u;
        require(c.dependency && dst==stageHandle && c.copyCount==0 && c.dispatchCount==0 && size!=0 && regions[0].size==size,"checked terminal status/output read range");
    }
    ++c.copyCount; c.dependency=false;
    c.ops.push_back({2,src,dst,{},uint32_t(regions[0].size/4),0,0});
}

// Cold staging preparation uses the real allocBuffer/map paths. All synthetic
// allocations are individually tracked; unexpected reuse/double-free aborts.
static VKAPI_ATTR VkResult VKAPI_CALL prefix_vkCreateBuffer(VkDevice,const VkBufferCreateInfo* info,const VkAllocationCallbacks*,VkBuffer* out) {
    using namespace prefix_fixture;
    require(info && out && info->size==65536 && info->usage==(VK_BUFFER_USAGE_TRANSFER_SRC_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT),"cold staging allocation shape");
    const auto r=call("vkCreateBuffer"); *out=r==VK_SUCCESS ? handle<VkBuffer>(++nextHandle) : VK_NULL_HANDLE;
    if (r==VK_SUCCESS) { createdBuffers[*out]=info->size; stageHandle=*out; } return r;
}
static VKAPI_ATTR void VKAPI_CALL prefix_vkGetBufferMemoryRequirements(VkDevice,VkBuffer b,VkMemoryRequirements* out) {
    using namespace prefix_fixture; call("vkGetBufferMemoryRequirements"); require(out && createdBuffers.count(b)==1,"memory requirements owner");
    *out={createdBuffers.at(b),4,1};
}
static VKAPI_ATTR VkResult VKAPI_CALL prefix_vkAllocateMemory(VkDevice,const VkMemoryAllocateInfo* info,const VkAllocationCallbacks*,VkDeviceMemory* out) {
    using namespace prefix_fixture; require(info && out && info->allocationSize==65536 && info->memoryTypeIndex==0,"staging allocation type");
    const auto r=call("vkAllocateMemory"); *out=r==VK_SUCCESS ? handle<VkDeviceMemory>(++nextHandle) : VK_NULL_HANDLE;
    if (r==VK_SUCCESS) allocatedMemory.insert(*out); return r;
}
static VKAPI_ATTR VkResult VKAPI_CALL prefix_vkBindBufferMemory(VkDevice,VkBuffer b,VkDeviceMemory m,VkDeviceSize offset) {
    using namespace prefix_fixture; require(createdBuffers.count(b)==1 && allocatedMemory.count(m)==1 && offset==0,"staging bind owners"); return call("vkBindBufferMemory");
}
static VKAPI_ATTR VkResult VKAPI_CALL prefix_vkMapMemory(VkDevice,VkDeviceMemory m,VkDeviceSize offset,VkDeviceSize bytes,VkMemoryMapFlags flags,void** out) {
    using namespace prefix_fixture; require(allocatedMemory.count(m)==1 && offset==0 && bytes==65536 && flags==0 && out,"staging map range");
    const auto r=call("vkMapMemory"); *out=r==VK_SUCCESS ? stage.data() : nullptr; return r;
}
static VKAPI_ATTR void VKAPI_CALL prefix_vkUnmapMemory(VkDevice,VkDeviceMemory m) {
    using namespace prefix_fixture; require(allocatedMemory.count(m)==1,"staging unmap owner"); call("vkUnmapMemory");
}
static VKAPI_ATTR void VKAPI_CALL prefix_vkDestroyBuffer(VkDevice,VkBuffer b,const VkAllocationCallbacks*) {
    using namespace prefix_fixture; call("vkDestroyBuffer"); require(createdBuffers.erase(b)==1,"staging buffer double-free");
}
static VKAPI_ATTR void VKAPI_CALL prefix_vkFreeMemory(VkDevice,VkDeviceMemory m,const VkAllocationCallbacks*) {
    using namespace prefix_fixture; call("vkFreeMemory"); require(allocatedMemory.erase(m)==1,"staging memory double-free");
}
`

const v41QueryPrefixOrchestrationCases = `
namespace prefix_fixture {
std::array<Buffer,15> buffers{};
Buffer staging{};
fvk_v41_query_prefix_buffers args{};
uint32_t failedStage=UINT32_MAX;
constexpr float epsilon=0.00001f;
size_t pooledDescriptors() {
    size_t n=0; for (const auto& bucket:g_descSetPool) n+=bucket.second.size(); return n;
}
void descriptorsHeld() {
    require(pooledDescriptors()==0,"a referenced descriptor was recycled before completion");
}
void reset(int h=193,int r=65) {
    // Synthetic fixture reset AFTER assertions. This is not production recovery
    // or a claim that an uncertain real device can be made quiescent this way.
    require(createdBuffers.empty() && allocatedMemory.empty(),"cold staging owner leaked across cases");
    commands.clear(); descriptors.clear(); memory.clear(); calls.clear(); statusFaults.clear();
    failure.clear(); failureOrdinal=1; failureResult=VK_ERROR_OUT_OF_HOST_MEMORY;
    submitted=VK_NULL_HANDLE; fills=dispatches=copies=barriers=0;
    g_pool.clear(); g_poolCount=0; g_batchFreed.clear(); g_batchSets.clear(); g_descSetPool.clear();
    g_batching=false; g_batchCmd=VK_NULL_HANDLE; g_batchOps=0; g_batchHasV41=false;
    g_v41SubmissionPendingFailure=false; g_submissionStatus=VK_SUCCESS;
    g_restore=RestoreTransaction{}; g_debugD2HStagingFailureOnce=false;
    g_h2dCount=0; g_h2dBytes=0; g_d2hCount=0; g_d2hBytes=0; g_transferCountersValid=true;
    require(!g_dp_on && !g_tsOn,"profiling must be disabled before fixture initialization");
    g_ready=true; g_dev=handle<VkDevice>(1); g_queue=handle<VkQueue>(2);
    g_cmdpool=handle<VkCommandPool>(3); g_submitFence=handle<VkFence>(4); g_descpool=handle<VkDescriptorPool>(5);
    g_maxComputeWorkGroupCountX=2;
    g_maxStorageBufferRange=g_maxMemoryAllocationSize=g_maxBufferBytes=std::numeric_limits<VkDeviceSize>::max();
    g_memprops={}; g_memprops.memoryTypeCount=1; g_memprops.memoryHeapCount=1;
    g_memprops.memoryTypes[0].propertyFlags=VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    g_memprops.memoryTypes[0].heapIndex=0; g_memprops.memoryHeaps[0].size=1u<<20;
    g_weightArena.clear(); g_weightArenaLiveBytes=0;
    hidden=h; rank=r; buffers={}; args={}; staging={}; failedStage=UINT32_MAX;
    for (int i=0;i<15;++i) {
        auto& b=buffers[i]; b.buf=handle<VkBuffer>(10+i); b.mem=handle<VkDeviceMemory>(100+i);
        const uint64_t count=i==1 ? uint64_t(h)*uint64_t(r) : i==2 || (i>=7 && i<=10) ? uint64_t(r) : i>=11 ? 4u : uint64_t(h);
        b.bytes=count*4u; b.props=VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT; roles[i]=b.buf;
        memory[b.buf]=std::vector<uint32_t>(size_t(count),i<3 ? 0x3f800000u : 0xdeadbeefu);
    }
    for (int i=0;i<3;++i) args.weights[i]=&buffers[i];
    args.input=&buffers[3];
    for (int i=0;i<7;++i) args.values[i]=&buffers[i+4];
    for (int i=0;i<4;++i) args.status[i]=&buffers[i+11];
    staging.buf=handle<VkBuffer>(30); staging.mem=handle<VkDeviceMemory>(130); staging.bytes=65536;
    staging.props=VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    stageHandle=staging.buf; std::fill(stage.begin(),stage.end(),0xfeed0001u);
    g_stage=&staging; g_stageMapped=stage.data(); g_stageCap=stage.size()*4;
    const int kinds[3]={K_V41_BF16_CHECKED,K_RMSNORM,K_MATMUL};
    for (int i=0;i<3;++i) {
        auto& k=g_kern[kinds[i]]; k={};
        k.pipe=handle<VkPipeline>(200+i); k.dsl=handle<VkDescriptorSetLayout>(210+i); k.layout=handle<VkPipelineLayout>(220+i);
        k.nbuf=3; k.pcsize=i==0 ? 8 : 12; pipelines[i]=k.pipe; dsls[i]=k.dsl; layouts[i]=k.layout;
    }
    g_have_v41_bf16_checked=1; std::memcpy(&epsilonBits,&epsilon,4);
    host.assign(size_t(h),1.25f); hostBefore=host;
}
void setArg(int i,void* p) {
    if (i<3) args.weights[i]=p; else if (i==3) args.input=p;
    else if (i<11) args.values[i-4]=p; else args.status[i-11]=p;
}
int invoke() { return fvk_v41_query_prefix_f32(host.data(),&args,hidden,rank,epsilon,&failedStage); }
void inputsUnchanged() {
    require(host==hostBefore,"borrowed host row mutated");
    for (int i=0;i<3;++i) require(memory.at(roles[i])==std::vector<uint32_t>(memory.at(roles[i]).size(),0x3f800000u),"borrowed weight bytes mutated");
}
void inputCopyMatchesHost() {
    std::vector<uint32_t> expected(hostBefore.size());
    std::memcpy(expected.data(),hostBefore.data(),hostBefore.size()*sizeof(float));
    require(memory.at(roles[3])==expected,"initial staging copy did not contain the exact supplied host row");
    inputsUnchanged();
}
void mutableBytesUnchanged() {
    for (int i=3;i<15;++i) require(memory.at(roles[i])==std::vector<uint32_t>(memory.at(roles[i]).size(),0xdeadbeefu),"unsubmitted command changed mutable bytes");
}
void releaseAndCheck(bool pending) {
    const auto before=calls; std::set<void*> owners;
    for (int i=3;i<15;++i) { owners.insert(&buffers[i]); fvk_free(&buffers[i]); }
    require(calls==before,"release unexpectedly called Vulkan");
    if (pending) {
        require(g_batchFreed.size()==12 && g_poolCount==0,"pending mutable owners escaped quarantine");
        for (int i=3;i<15;++i) require(g_batchFreed[size_t(i-3)]==&buffers[i],"quarantine owner identity/order");
        require(fvk_malloc(buffers[3].bytes)==nullptr && fvk_malloc(16)==nullptr,"pending owner became reusable");
        require(g_stage && g_stageMapped==stage.data(),"referenced staging owner was lost");
    } else {
        require(g_batchFreed.empty() && g_poolCount==12,"known owners not pooled exactly once");
        if (g_submissionStatus==VK_SUCCESS) {
            for (int i=3;i<15;++i) {
                void* p=fvk_malloc(buffers[i].bytes); require(owners.erase(p)==1,"pool returned wrong/repeated owner");
            }
            require(owners.empty() && g_poolCount==0,"pool reuse conservation");
        }
    }
    inputsUnchanged();
}
void preflightRefused(const char* why) {
    const auto initialStage=stage; require(invoke()!=0 && calls.empty() && commands.empty() && fills==0 && dispatches==0 && stage==initialStage,why);
    inputsUnchanged();
}
void geometryAndAliasCases() {
    reset();
    require(fvk_v41_query_prefix_geometry_supported(193,65)!=0 && fvk_v41_query_prefix_geometry_supported(1,1)!=0,"positive geometry");
    for (const auto dim:std::array<std::array<int,2>,6>{{{0,65},{193,0},{-1,65},{193,-1},{INT32_MAX,2},{46341,46341}}})
        require(fvk_v41_query_prefix_geometry_supported(dim[0],dim[1])==0,"geometry count/product refusal");
    g_maxStorageBufferRange=uint64_t(hidden)*rank*4-1;
    require(!fvk_v41_query_prefix_geometry_supported(hidden,rank),"matrix storage bound missing");
    g_maxStorageBufferRange=uint64_t(hidden)*rank*4;
    require(fvk_v41_query_prefix_geometry_supported(hidden,rank)!=0,"exact matrix storage bound");
    g_maxBufferBytes=g_maxStorageBufferRange-1;
    require(!fvk_v41_query_prefix_geometry_supported(hidden,rank),"matrix allocation bound missing");
    require(calls.empty(),"geometry predicate reached Vulkan");
    for (int i=0;i<15;++i) {
        reset(); setArg(i,nullptr); preflightRefused("null operand owner");
        reset(); buffers[i].bytes-=1; preflightRefused("short operand extent");
        reset(); buffers[i].buf=VK_NULL_HANDLE; preflightRefused("null Vulkan buffer");
        reset(); buffers[i].mem=VK_NULL_HANDLE; preflightRefused("null allocation");
        reset(); buffers[i].memoryOffset=std::numeric_limits<VkDeviceSize>::max(); preflightRefused("operand extent overflow");
    }
    for (int i=3;i<15;++i) for (int j=0;j<i;++j) {
        reset(); buffers[j].bytes=std::max(buffers[j].bytes,buffers[i].bytes); setArg(i,&buffers[j]); preflightRefused("mutable wrapper alias");
        reset(); buffers[i].buf=buffers[j].buf; preflightRefused("mutable Vulkan buffer alias");
        reset(); buffers[i].mem=buffers[j].mem; buffers[i].memoryOffset=buffers[j].bytes-1; preflightRefused("mutable allocation overlap");
    }
    for (int i=0;i<15;++i) {
        reset(); staging.buf=buffers[i].buf; preflightRefused("staging VkBuffer alias");
        reset(); staging.mem=buffers[i].mem; staging.memoryOffset=buffers[i].bytes-1; preflightRefused("staging allocation overlap");
    }
    reset(); staging.memoryOffset=std::numeric_limits<VkDeviceSize>::max(); preflightRefused("staging overflow");
    reset(); VkDeviceSize offset=0;
    for (auto& b:buffers) { b.mem=handle<VkDeviceMemory>(999); b.memoryOffset=offset; offset+=b.bytes; }
    staging.mem=buffers[0].mem; staging.memoryOffset=offset;
    require(invoke()==0 && failedStage==0,"adjacent disjoint same-memory ranges must work");
    reset(); // Read-only weight sharing is legal; only mutable aliases refuse.
    for (int i=1;i<3;++i) buffers[i].mem=buffers[0].mem;
    require(invoke()==0,"read-only shared allocation refused");
    for (int kind:{K_V41_BF16_CHECKED,K_RMSNORM,K_MATMUL}) {
        reset(); g_kern[kind].pipe=VK_NULL_HANDLE; preflightRefused("required pipeline absent");
    }
    reset(); g_have_v41_bf16_checked=0; preflightRefused("checked module absent");
    reset(); g_ready=false; preflightRefused("backend not ready");
    reset(); g_maxComputeWorkGroupCountX=0; preflightRefused("zero workgroup limit");
    reset();
    for (float eps:{0.0f,-1.0f,std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()})
        require(fvk_v41_query_prefix_f32(host.data(),&args,hidden,rank,eps,&failedStage)==2 && calls.empty(),"epsilon refusal");
    require(fvk_v41_query_prefix_f32(nullptr,&args,hidden,rank,epsilon,&failedStage)==2 && calls.empty(),"null host row");
    require(fvk_v41_query_prefix_f32(host.data(),nullptr,hidden,rank,epsilon,&failedStage)==2 && calls.empty(),"null bundle");
    require(fvk_v41_query_prefix_f32(host.data(),&args,hidden,rank,epsilon,nullptr)==2 && calls.empty(),"null stage receiver");
    std::puts("PASS prefix geometry, exact extents and every mutable alias pair");
}
void busyAndLegacyCases() {
    static_assert(FVK_V41_QUERY_PREFIX_BUSY_BATCH==20 && FVK_V41_QUERY_PREFIX_BUSY_RESTORE==21,"busy ABI constants");
    for (int busy=0;busy<3;++busy) {
        reset();
        if (busy==0) g_batching=true; else if (busy==1) g_restore.stage=&staging; else g_restore.cmd=handle<VkCommandBuffer>(777);
        const int expected=busy==0 ? 20 : 21;
        require(fvk_have_v41_query_prefix()!=0 && fvk_v41_query_prefix_state()==expected,"busy changed module availability");
        const auto before=stage;
        require(fvk_v41_query_prefix_f32(nullptr,nullptr,-1,-1,0,&failedStage)==expected && failedStage==0 && calls.empty() && stage==before,"healthy busy must precede payload inspection");
        require(g_submissionStatus==VK_SUCCESS && !g_v41SubmissionPendingFailure,"healthy busy poisoned owner");
        g_submissionStatus=VK_ERROR_DEVICE_LOST;
        require(invoke()==VK_ERROR_DEVICE_LOST && calls.empty(),"sticky must precede busy");
        g_submissionStatus=VK_SUCCESS; g_v41SubmissionPendingFailure=true;
        require(invoke()==VK_ERROR_UNKNOWN && calls.empty(),"pending must precede busy");
        g_v41SubmissionPendingFailure=false; g_batching=false; g_restore=RestoreTransaction{};
        require(invoke()==0,"same healthy context retry after busy");
    }
    reset();
    require(fvk_v41_bf16_checked_f32(&buffers[3],&buffers[4],&buffers[11],hidden,3)==2 && calls.empty(),"old synchronous site domain widened");
    g_batching=true;
    require(fvk_v41_bf16_checked_f32(&buffers[3],&buffers[4],&buffers[11],hidden,1)==2 && calls.empty(),"old synchronous active batch refusal changed");
    std::puts("PASS healthy busy precedence/retry and old checked-site refusal");
}
Buffer* readOwner(int i) { return i<4 ? &buffers[11+i] : i==4 ? &buffers[6] : &buffers[10]; }
std::vector<uint32_t> readExpected(int i) {
    return i<4 ? std::vector<uint32_t>{0x42463143u,uint32_t(i+3),UINT32_MAX,uint32_t(i<2 ? hidden : rank)} : std::vector<uint32_t>(size_t(i==4 ? hidden : rank),outputWord(i==4 ? 2 : 6));
}
void successCases() {
    for (const auto dim:std::array<std::array<int,2>,2>{{{193,65},{64,32}}}) {
        reset(dim[0],dim[1]);
        for (int reuse=0;reuse<2;++reuse) {
            if (reuse) { host.assign(size_t(hidden),-2.5f); hostBefore=host; }
            // Stale SUCCESS records are deliberately indistinguishable by the
            // decoder. The full command must refill all four every invocation.
            for (int i=0;i<4;++i) memory[roles[11+i]]=readExpected(i);
            const int oldFills=fills, oldAllocations=calls["vkAllocateDescriptorSets"];
            require(invoke()==0 && failedStage==0 && fills==oldFills+4,"fresh local prefix completion");
            inputCopyMatchesHost();
            require(commands.empty() && pooledDescriptors()==7 && calls["vkAllocateDescriptorSets"]==(reuse ? oldAllocations : 7),"completed descriptor retention/reuse");
            std::array<std::vector<uint32_t>,6> reads;
            for (int i=0;i<6;++i) {
                reads[i].assign(size_t(readOwner(i)->bytes/4),0xdeadc0deu);
                require(fvk_v41_d2h(reads[i].data(),readOwner(i),size_t(readOwner(i)->bytes))==0 && reads[i]==readExpected(i),"actual status/output D2H bytes");
            }
            for (int i=0;i<6;++i) require(reads[i]==readExpected(i),"later staging reuse changed earlier host copy");
            require(commands.empty() && g_h2dCount==uint64_t(reuse+1) && g_h2dBytes==uint64_t(reuse+1)*hidden*4u && g_d2hCount==uint64_t(reuse+1)*6u && g_d2hBytes==uint64_t(reuse+1)*(64u+uint64_t(hidden+rank)*4u),"bounded successful transfer accounting");
        }
        require(fills==8 && dispatches==14 && copies==14 && barriers==26 && calls["vkQueueSubmit"]==14 && calls["vkFreeCommandBuffers"]==14 && g_transferCountersValid,"two one-plus-six terminal traces");
        releaseAndCheck(false);
    }
    std::puts("PASS seven ordered kernels, fresh fills, barriers, status/output transport and owner reuse");
}
void envelopeRetentionCases() {
    for (uint32_t site=3;site<=6;++site) {
        reset(); const uint32_t count=uint32_t(site<=4 ? hidden : rank);
        statusFaults[site]=count-1;
        require(invoke()==0,"injected envelope prefix completion");
        for (int read=0;read<4;++read) {
            std::array<uint32_t,4> out{};
            require(fvk_v41_d2h(out.data(),readOwner(read),16)==0,"injected envelope transport");
            const uint32_t actualSite=uint32_t(read+3);
            const std::array<uint32_t,4> expected{0x42463143u,actualSite,actualSite==site ? count-1 : UINT32_MAX,uint32_t(read<2 ? hidden : rank)};
            require(out==expected,"later finite stages erased an earlier injected arithmetic record");
        }
        require(memory.at(roles[6])==std::vector<uint32_t>(size_t(hidden),outputWord(2)) && memory.at(roles[10])==std::vector<uint32_t>(size_t(rank),outputWord(6)),"finite later output sentinel setup");
        require(g_d2hCount==4,"envelope-only control accidentally read outputs");
    }
    // These injected records prove native separation/retention, not arithmetic
    // generation or Go aggregation/publication policy.
    std::puts("PASS per-site injected records survive later finite writes");
}
void preparationFailures() {
    reset(); g_debugD2HStagingFailureOnce=true;
    require(invoke()==FVK_D2H_STAGING_ALLOCATION_FAILED && failedStage==1 && calls.empty() && commands.empty(),"explicit staging preparation failure");
    const char* APIs[]={"vkCreateBuffer","vkAllocateMemory","vkBindBufferMemory","vkMapMemory"};
    for (const char* api:APIs) {
        reset(); g_stage=nullptr; g_stageMapped=nullptr; g_stageCap=0;
        failure=api; failureOrdinal=0; failureResult=VK_ERROR_OUT_OF_HOST_MEMORY;
        const int expected=failure=="vkMapMemory" ? int(failureResult) : FVK_D2H_STAGING_ALLOCATION_FAILED;
        require(invoke()==expected && failedStage==1 && calls[api]>0 && calls["vkQueueSubmit"]==0 && calls["vkAllocateCommandBuffers"]==0,"cold preparation failure attribution");
        require(createdBuffers.empty() && allocatedMemory.empty() && g_stage==nullptr && !g_v41SubmissionPendingFailure,"cold preparation owner disposal");
        inputsUnchanged(); mutableBytesUnchanged();
    }
    reset(); g_stage=nullptr; g_stageMapped=nullptr; g_stageCap=0;
    failure="vkMapMemory"; failureResult=VK_ERROR_DEVICE_LOST;
    require(invoke()==VK_ERROR_DEVICE_LOST && failedStage==1 && g_submissionStatus==VK_ERROR_DEVICE_LOST && g_v41SubmissionPendingFailure && commands.empty(),"map device loss classification");
    require(createdBuffers.empty() && allocatedMemory.empty(),"never-submitted mapped owner leaked");
    reset(); g_stage=nullptr; g_stageMapped=nullptr; g_stageCap=0;
    require(invoke()==0 && calls["vkCreateBuffer"]==1 && calls["vkMapMemory"]==1,"cold staging positive control");
    require(g_stage && g_stage!=&staging && createdBuffers.size()==1 && allocatedMemory.size()==1,"actual cold staging owner");
    vkUnmapMemory(g_dev,g_stage->mem); destroyBuffer(g_stage); g_stage=nullptr; g_stageMapped=nullptr; g_stageCap=0;
    require(createdBuffers.empty() && allocatedMemory.empty(),"cold positive owner release");
    std::puts("PASS real cold staging creation/allocation/bind/map faults");
}
void descriptorFailures() {
    for (int ordinal=1;ordinal<=7;++ordinal) for (bool deviceLost:{false,true}) {
        reset(); failure="vkAllocateDescriptorSets"; failureOrdinal=ordinal;
        failureResult=deviceLost ? VK_ERROR_DEVICE_LOST : VK_ERROR_OUT_OF_HOST_MEMORY;
        require(invoke()==failureResult && failedStage==uint32_t(ordinal+1) && g_submissionStatus==failureResult && g_v41SubmissionPendingFailure==deviceLost,"kth descriptor failure identity/stage");
        require(calls["vkQueueSubmit"]==0 && dispatches==ordinal-1 && calls["vkFreeCommandBuffers"]==(deviceLost ? 0 : 1),"partial command submitted or released incorrectly");
        require(commands.size()==(deviceLost ? 1u : 0u) && pooledDescriptors()==(deviceLost ? 0u : size_t(ordinal-1)),"descriptor OOM discard versus device-loss quarantine");
        mutableBytesUnchanged();
        const auto before=calls;
        require(invoke()==failureResult && calls==before,"selected descriptor fault was replayed");
        releaseAndCheck(deviceLost);
    }
    // Missing layout fails in the actual record path rather than allocation.
    for (int step:{0,1,3}) {
        reset(); const int kinds[4]={K_V41_BF16_CHECKED,K_RMSNORM,0,K_MATMUL};
        g_kern[kinds[step]].layout=VK_NULL_HANDLE;
        require(invoke()==4 && failedStage==uint32_t(step+2) && calls["vkQueueSubmit"]==0 && calls["vkFreeCommandBuffers"]==1 && commands.empty() && pooledDescriptors()==size_t(step),"record validation owns earlier descriptors exactly once");
    }
    std::puts("PASS all seven descriptor OOM/device-loss ordinals and partial-record disposal");
}
void commandFailures() {
    const char* APIs[]={"vkAllocateCommandBuffers","vkBeginCommandBuffer","vkEndCommandBuffer","vkResetFences","vkQueueSubmit","vkWaitForFences"};
    for (const char* api:APIs) for (bool negativeWait:{false,true}) {
        if (negativeWait && std::string(api)!="vkWaitForFences") continue;
        reset(); failure=api;
        const bool early=failure=="vkAllocateCommandBuffers" || failure=="vkBeginCommandBuffer";
        const bool pending=failure=="vkQueueSubmit" || failure=="vkWaitForFences";
        failureResult=failure=="vkWaitForFences" ? (negativeWait ? VK_ERROR_DEVICE_LOST : VK_TIMEOUT) : pending ? VK_ERROR_DEVICE_LOST : VK_ERROR_OUT_OF_HOST_MEMORY;
        const auto expected=failureResult>VK_SUCCESS ? VK_ERROR_UNKNOWN : failureResult;
        require(invoke()==expected && failedStage==uint32_t(early ? 1 : 9) && g_submissionStatus==expected && g_v41SubmissionPendingFailure==pending,"command fault attribution");
        require(commands.size()==(pending ? 1u : 0u) && pooledDescriptors()==(early || pending ? 0u : 7u),"command/descriptors terminal ownership");
        require(calls["vkFreeCommandBuffers"]==(pending || failure=="vkAllocateCommandBuffers" ? 0 : 1),"command free once");
        require(calls["vkQueueSubmit"]==(pending ? 1 : 0) && g_h2dCount==0 && g_d2hCount==0,"failed prefix accounting or submit replay");
        if (pending) for (int i=0;i<4;++i) require(memory.at(roles[11+i])==readExpected(i),"hostile valid-looking pending statuses");
        else mutableBytesUnchanged();
        const auto before=calls; require(invoke()==expected && calls==before,"sticky command fault was retried");
        releaseAndCheck(pending);
    }
    std::puts("PASS command allocate/begin/end/reset/submit/wait failures with no retry");
}
void readFailures() {
    const char* APIs[]={"vkAllocateCommandBuffers","vkBeginCommandBuffer","vkEndCommandBuffer","vkResetFences","vkQueueSubmit","vkWaitForFences"};
    for (int read=0;read<6;++read) for (const char* api:APIs) for (bool negativeWait:{false,true}) {
        if (negativeWait && std::string(api)!="vkWaitForFences") continue;
        reset(); require(invoke()==0,"readback prerequisite prefix");
        for (int i=0;i<read;++i) {
            std::vector<uint32_t> earlier(size_t(readOwner(i)->bytes/4));
            require(fvk_v41_d2h(earlier.data(),readOwner(i),size_t(readOwner(i)->bytes))==0 && earlier==readExpected(i),"earlier completed terminal read");
        }
        failure=api; failureOrdinal=read+2;
        const bool pending=failure=="vkQueueSubmit" || failure=="vkWaitForFences";
        failureResult=failure=="vkWaitForFences" ? (negativeWait ? VK_ERROR_DEVICE_LOST : VK_TIMEOUT) : pending ? VK_ERROR_DEVICE_LOST : VK_ERROR_OUT_OF_HOST_MEMORY;
        const auto expected=failureResult>VK_SUCCESS ? VK_ERROR_UNKNOWN : failureResult;
        const auto expectedBytes=readExpected(read); std::copy(expectedBytes.begin(),expectedBytes.end(),stage.begin());
        std::vector<uint32_t> out(size_t(readOwner(read)->bytes/4),0xdeadc0deu); const auto before=out;
        require(fvk_v41_d2h(out.data(),readOwner(read),size_t(readOwner(read)->bytes))==expected && out==before,"failed terminal read published stale bytes");
        require(g_submissionStatus==expected && g_v41SubmissionPendingFailure==pending && g_d2hCount==uint64_t(read) && commands.size()==(pending ? 1u : 0u),"read failure identity/ownership/accounting");
        require(pooledDescriptors()==7,"completed prefix descriptors changed during separate terminal reads");
        const auto oldCalls=calls;
        require(fvk_v41_d2h(out.data(),readOwner(read),size_t(readOwner(read)->bytes))==expected && out==before && calls==oldCalls,"failed terminal read replayed");
        releaseAndCheck(pending);
    }
    for (int read=0;read<6;++read) {
        reset(); require(invoke()==0,"staging-read prerequisite"); g_debugD2HStagingFailureOnce=true;
        std::vector<uint32_t> out(size_t(readOwner(read)->bytes/4),0xdeadc0deu); const auto before=out;
        require(fvk_v41_d2h(out.data(),readOwner(read),size_t(readOwner(read)->bytes))==FVK_D2H_STAGING_ALLOCATION_FAILED && out==before && calls["vkQueueSubmit"]==1 && g_d2hCount==0,"terminal staging preparation refusal");
    }
    std::puts("PASS each of four status and two output read transport faults; failed read bytes withheld");
}
} // namespace prefix_fixture
int main() {
    prefix_fixture::geometryAndAliasCases();
    prefix_fixture::busyAndLegacyCases();
    prefix_fixture::successCases();
    prefix_fixture::envelopeRetentionCases();
    prefix_fixture::preparationFailures();
    prefix_fixture::descriptorFailures();
    prefix_fixture::commandFailures();
    prefix_fixture::readFailures();
    std::puts("PASS real query-prefix full-shim recording, transport and quarantine; numerical/Go-adapter/physical execution unclaimed");
    return 0;
}
`
