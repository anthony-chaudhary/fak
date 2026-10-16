package compute

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// fak-test:runtime medium est=3s lane=default
// fak-test:justify Compiles the production allocator owners with synthetic driver callbacks; no device or Vulkan loader.
func TestVulkanMemoryGuardOwners(t *testing.T) {
	compiler, err := exec.LookPath("c++")
	if err != nil {
		t.Skip("C++ compiler unavailable")
	}
	raw, err := os.ReadFile("vulkan_shim.cpp")
	if err != nil {
		t.Fatal(err)
	}
	source := string(raw)
	extract := func(start string, semicolon bool) string {
		i := strings.Index(source, start)
		if i < 0 {
			t.Fatalf("missing production definition %s", start)
		}
		open := strings.Index(source[i:], "{") + i
		depth := 0
		for end := open; end < len(source); end++ {
			if source[end] == '{' {
				depth++
			}
			if source[end] == '}' {
				depth--
				if depth == 0 {
					if semicolon {
						end++
					}
					return source[i:end+1] + "\n"
				}
			}
		}
		t.Fatalf("unterminated definition %s", start)
		return ""
	}
	// Every OS allocation/free must pass through the same owner; aliases cannot add charges.
	if strings.Count(source, "vkAllocateMemory(g_dev,") != 1 || strings.Count(source, "vkFreeMemory(g_dev,") != 1 || strings.Count(source, "return allocateGuardedMemory(&ai, candidate);") != 2 || strings.Count(source, "freeGuardedMemory(") != 5 {
		t.Fatal("uncovered native memory allocation/free seam")
	}
	init := source[strings.Index(source, "int fvk_init("):]
	if strings.Index(init, "if (!configureMemoryGuard())") > strings.Index(init, "discardIdleBatchCommand();") || !strings.Contains(init, "return 10;") {
		t.Fatal("configuration must refuse before initialization side effects")
	}
	fixture := guardFixturePreamble + extract("struct Buffer {", true) + extract("struct WeightArenaBlock {", true) + guardFixtureGlobals
	for _, definition := range []string{"bool configureMemoryGuard()", "VkResult allocateGuardedMemory(", "void freeGuardedMemory(", "void reportMemoryGuard()", "bool memoryTypeUsesDeviceLocalHeap(", "void trackDeviceAllocation(", "void untrackDeviceAllocation(", "uint64_t nextMemoryAllocationID()", "static void advanceNativeCheckoutGeneration(Buffer* b) {", "void releaseWeightArena()", "void destroyBuffer(", "size_t drainPool()", "bool allocPressure(", "template <typename Allocate>\nVkResult allocateCompatibleMemory(", "uint32_t findDirectWeightMemType(", "VkDeviceSize alignArenaOffset(", "Buffer* allocBuffer(size_t bytes, VkMemoryPropertyFlags props, VkBufferUsageFlags usage,\n                    VkMemoryPropertyFlags preferred) {", "Buffer* allocWeightArenaBuffer(", "void fvk_free(void* d) {", "void* fvk_malloc(size_t bytes) {", "static Buffer* mallocHostVisibleOffloadPage(size_t bytes) {", "void fvk_memory_guard_snapshot("} {
		fixture += extract(definition, false)
	}
	fixture += guardFixtureMain
	dir := t.TempDir()
	cpp := filepath.Join(dir, "guard.cpp")
	binary := filepath.Join(dir, "guard")
	if err := os.WriteFile(cpp, []byte(fixture), 0600); err != nil {
		t.Fatal(err)
	}
	cwd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
	defer cancel()
	if out, err := exec.CommandContext(ctx, compiler, "-std=c++17", "-I", cwd, cpp, "-o", binary).CombinedOutput(); err != nil {
		t.Fatalf("compile: %v\n%s", err, out)
	}
	out, err := exec.CommandContext(ctx, binary).CombinedOutput()
	if err != nil {
		t.Fatalf("allocator owner: %v\n%s", err, out)
	}
	var record struct {
		Scope      string `json:"scope"`
		Limit      uint64 `json:"limit_bytes"`
		Live       uint64 `json:"live_bytes"`
		Configured bool   `json:"configured"`
		Valid      bool   `json:"configuration_valid"`
		Accounting bool   `json:"accounting_valid"`
	}
	prefix := []byte("fak-vulkan-memory-guard: ")
	at := bytes.LastIndex(out, prefix)
	if at < 0 {
		t.Fatal("missing actual startup record")
	}
	payload := bytes.TrimSpace(out[at+len(prefix):])
	if err := json.Unmarshal(payload, &record); err != nil {
		t.Fatalf("actual startup record: %v: %s", err, out)
	}
	if record.Scope != "all_vk_device_memory" || record.Limit != 63 || record.Live != 0 || !record.Configured || !record.Valid || !record.Accounting {
		t.Fatalf("startup attestation differs: %s", out)
	}
}

const guardFixturePreamble = `
#include "vulkan_memory_guard.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <atomic>
#include <unordered_map>
#include <vector>
using VkDeviceSize=uint64_t; using VkDeviceMemory=uintptr_t; using VkBuffer=uintptr_t;
using VkMemoryPropertyFlags=uint32_t; using VkBufferUsageFlags=uint32_t; using VkResult=int;
constexpr int VK_SUCCESS=0,VK_ERROR_OUT_OF_DEVICE_MEMORY=-2,VK_ERROR_OUT_OF_HOST_MEMORY=-1,VK_ERROR_TOO_MANY_OBJECTS=-10,VK_ERROR_FEATURE_NOT_PRESENT=-8;
constexpr uint32_t VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT=1,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT=2,VK_MEMORY_PROPERTY_HOST_COHERENT_BIT=4;
constexpr uint32_t VK_MEMORY_HEAP_DEVICE_LOCAL_BIT=1,VK_MAX_MEMORY_HEAPS=16;
constexpr int VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO=1,VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO=2,VK_SHARING_MODE_EXCLUSIVE=0;
constexpr int VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS=3,VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2=4,VK_STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2=5;
constexpr uint32_t VK_BUFFER_USAGE_STORAGE_BUFFER_BIT=1,STORAGE_USAGE=1;
constexpr uint64_t VK_NULL_HANDLE=0,VK_WHOLE_SIZE=~uint64_t(0),WEIGHT_ARENA_BLOCK_BYTES=256;
struct VkMemoryType {uint32_t propertyFlags,heapIndex;};
struct VkMemoryHeap {uint64_t size;uint32_t flags;};
struct VkPhysicalDeviceMemoryProperties {uint32_t memoryTypeCount,memoryHeapCount;VkMemoryType memoryTypes[32];VkMemoryHeap memoryHeaps[16];};
struct VkMemoryAllocateInfo {int sType; uint64_t allocationSize;uint32_t memoryTypeIndex;};
struct VkBufferCreateInfo {int sType;uint64_t size;uint32_t usage;int sharingMode;};
struct VkMemoryRequirements {uint64_t size,alignment;uint32_t memoryTypeBits;};
struct VkMemoryDedicatedRequirements {int sType;bool requiresDedicatedAllocation;};
struct VkMemoryRequirements2 {int sType;void* pNext;VkMemoryRequirements memoryRequirements;};
struct VkBufferMemoryRequirementsInfo2 {int sType;VkBuffer buffer;};
int calls=0,frees=0,bindStatus=0,osStatus=0;bool dedicated=false;uintptr_t nextHandle=1;
VkPhysicalDeviceMemoryProperties g_memprops{};
VkResult vkAllocateMemory(uintptr_t,const VkMemoryAllocateInfo*,void*,VkDeviceMemory* out) {++calls;*out=nextHandle++;return osStatus;}
void vkFreeMemory(uintptr_t,VkDeviceMemory,void*) {++frees;}
VkResult vkCreateBuffer(uintptr_t,const VkBufferCreateInfo*,void*,VkBuffer* out) {*out=nextHandle++;return 0;}
void vkDestroyBuffer(uintptr_t,VkBuffer,void*) {}
void vkGetBufferMemoryRequirements(uintptr_t,VkBuffer,VkMemoryRequirements* out) {*out={64,64,3};}
void vkGetBufferMemoryRequirements2(uintptr_t,const VkBufferMemoryRequirementsInfo2*,VkMemoryRequirements2* out) {out->memoryRequirements={64,64,3};static_cast<VkMemoryDedicatedRequirements*>(out->pNext)->requiresDedicatedAllocation=dedicated;}
VkResult vkBindBufferMemory(uintptr_t,VkBuffer,VkDeviceMemory,uint64_t) {return bindStatus;}
VkResult vkMapMemory(uintptr_t,VkDeviceMemory,uint64_t,uint64_t,int,void** out) {*out=nullptr;return -5;}
void vkUnmapMemory(uintptr_t,VkDeviceMemory) {}
void clearDescriptorBindingCache() {}
`
const guardFixtureGlobals = `
VulkanMemoryGuard g_memoryGuard;
uintptr_t g_dev=1;
std::vector<WeightArenaBlock> g_weightArena;
std::unordered_map<size_t,std::vector<Buffer*>> g_pool,g_hvOffloadPool;
std::vector<Buffer*> g_batchFreed;
size_t g_poolCount=0;const size_t POOL_BUCKET_CAP=64;
bool g_v41SubmissionPendingFailure=false,g_batching=false,g_umaDirectWeights=false;
uint64_t g_weightArenaReservedBytes=0,g_weightArenaLiveBytes=0,g_weightArenaMemoryAllocations=0,g_weightArenaBufferBindings=0,g_weightArenaPeakReservedBytes=0;
uint64_t g_maxBufferBytes=1024,g_maxMemoryAllocationSize=1024,g_maxStorageBufferRange=1024,g_totalDeviceLocalMemory=1024;
uint64_t g_deviceAllocationLiveBytes=0,g_allocationWindowPeakBytes=0,g_nextMemoryAllocationID=1,g_memoryDeviceGeneration=1;
uint64_t g_offloadWeightReused=0,g_offloadWeightFresh=0,g_offloadHostVisiblePages=0;
bool g_deviceAllocationAccountingValid=true,g_allocationWindowActive=false;
Buffer* B(void* d) {return static_cast<Buffer*>(d);}
void mapHostVisibleWeight(Buffer*) {}
Buffer* allocBuffer(size_t,VkMemoryPropertyFlags,VkBufferUsageFlags,VkMemoryPropertyFlags=0);
`
const guardFixtureMain = `
void reset(const char* cap) {
 assert(g_memoryGuard.live==0 && g_weightArena.empty() && g_poolCount==0 && g_batchFreed.empty());
 g_memoryGuard={};assert(g_memoryGuard.configure(cap));calls=frees=0;osStatus=bindStatus=0;dedicated=false;
 g_memprops={};g_memprops.memoryTypeCount=g_memprops.memoryHeapCount=2;
 g_memprops.memoryTypes[0]={VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,0};
 g_memprops.memoryTypes[1]={VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,1};
 g_memprops.memoryHeaps[0]={1024,VK_MEMORY_HEAP_DEVICE_LOCAL_BIT};g_memprops.memoryHeaps[1]={1024,0};
}
int main() {
 for(const char* bad:{"","0","-1","+1"," 1","1 ","1x","18446744073709551616","9223372036854775808"}) {VulkanMemoryGuard g;assert(!g.configure(bad));assert(!g.reserve(1));assert(g.live==0 && g.refused==1);assert(!g.configure("64"));g_memoryGuard={};assert(!g_memoryGuard.configure(bad));VkMemoryAllocateInfo info{1,1,0};VkDeviceMemory memory=999;calls=0;assert(allocateGuardedMemory(&info,&memory)==VK_ERROR_OUT_OF_DEVICE_MEMORY && memory==0 && calls==0);}
 VulkanMemoryGuard g;assert(g.configure("9223372036854775807"));assert(g.configure("1") && g.limit==INT64_MAX);assert(g.reserve(INT64_MAX));assert(!g.reserve(1));g.release(INT64_MAX);assert(g.live==0);
 VulkanMemoryGuard unlimited;assert(unlimited.configure(nullptr));assert(unlimited.reserve(UINT64_MAX));assert(!unlimited.reserve(1));unlimited.release(UINT64_MAX);unlimited.release(1);assert(!unlimited.accountingValid && !unlimited.reserve(1));
 reset("64");auto a=allocBuffer(1,1,STORAGE_USAGE);assert(a && g_memoryGuard.live==64 && g_memoryGuard.peak==64 && calls==1);
 assert(!allocBuffer(1,2|4,STORAGE_USAGE));assert(calls==1 && g_memoryGuard.refused>0);destroyBuffer(a);assert(g_memoryGuard.live==0 && frees==1);
 reset("64");uint64_t lim,live,peak,refused;int configured,configurationValid,accountingValid;
 fvk_memory_guard_snapshot(&lim,&live,&peak,&refused,&configured,&configurationValid,&accountingValid);
 assert(lim==64 && live==0 && peak==0 && refused==0 && configured && configurationValid && accountingValid);
 osStatus=-2;assert(!allocBuffer(1,1,STORAGE_USAGE));assert(g_memoryGuard.live==0 && g_memoryGuard.peak==0 && calls>0 && frees==0);osStatus=0;
 bindStatus=-5;assert(!allocBuffer(1,1,STORAGE_USAGE));assert(g_memoryGuard.live==0 && frees==1);bindStatus=0;
 reset("64");a=allocBuffer(1,1,STORAGE_USAGE);fvk_free(a);assert(g_memoryGuard.live==64 && g_poolCount==1);auto reused=static_cast<Buffer*>(fvk_malloc(1));assert(reused==a && calls==1 && g_memoryGuard.live==64);fvk_free(reused);drainPool();assert(g_memoryGuard.live==0 && frees==1);
 reset("64");a=allocBuffer(1,2|4,STORAGE_USAGE);a->offloadPage=true;fvk_free(a);assert(g_memoryGuard.live==64 && frees==0);auto page=mallocHostVisibleOffloadPage(1);assert(page==a && calls==1 && g_memoryGuard.live==64);fvk_free(page);drainPool();assert(g_memoryGuard.live==0 && frees==1);
 reset("64");a=allocBuffer(1,1,STORAGE_USAGE);g_batching=true;fvk_free(a);assert(g_memoryGuard.live==64 && frees==0);g_batching=false;g_batchFreed.clear();destroyBuffer(a);assert(g_memoryGuard.live==0);
 reset("64");a=allocBuffer(1,1,STORAGE_USAGE);g_v41SubmissionPendingFailure=true;destroyBuffer(a);assert(g_memoryGuard.live==64 && frees==0);g_v41SubmissionPendingFailure=false;destroyBuffer(a);assert(g_memoryGuard.live==0);
 reset("256");a=allocWeightArenaBuffer(1,256);assert(a && g_memoryGuard.live==256 && calls==1);auto b=allocWeightArenaBuffer(1,256);assert(b && a->mem==b->mem && g_memoryGuard.live==256 && calls==1);destroyBuffer(a);assert(g_memoryGuard.live==256 && frees==0);destroyBuffer(b);assert(g_memoryGuard.live==0 && frees==1);
 reset("256");bindStatus=-5;assert(!allocWeightArenaBuffer(1,256));assert(g_memoryGuard.live==0 && frees==1);
 reset("63");dedicated=true;assert(!allocWeightArenaBuffer(1,256));assert(calls==0 && g_memoryGuard.live==0);
 reset("63");assert(!allocWeightArenaBuffer(1,256));assert(calls==0 && g_memoryGuard.live==0);reportMemoryGuard();
}
`
