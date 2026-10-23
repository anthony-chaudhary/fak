package compute

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// Native capture executes production fvk_matmul_f32, its group selection, and
// record-time verdict functions. The dispatch boundary is a descriptor recorder,
// not a Vulkan implementation. Ticket03/06 separately own full recorder placement
// and window/fence extraction; no loader, GPU, shader execution or parity is claimed.
func testVulkanDecodeCutNativeCapture(t *testing.T) {
	compiler, err := exec.LookPath("c++")
	if err != nil {
		t.Fatal("native capture UNRUN: C++17 compiler required")
	}
	b, err := os.ReadFile("vulkan_shim.cpp")
	if err != nil {
		t.Fatal(err)
	}
	shim := string(b)
	section := func(from, to string) string {
		i := strings.Index(shim, from)
		if i < 0 {
			t.Fatalf("missing %s", from)
		}
		j := strings.Index(shim[i:], to)
		if j < 0 {
			t.Fatalf("missing %s", to)
		}
		return shim[i : i+j]
	}
	var source strings.Builder
	source.WriteString(`#include <unordered_map>
#include <limits>
#include <cstdint>
#include <cstddef>
#include <cassert>
using VkCommandBuffer=int;
struct VkMemoryBarrier { int type; int srcAccessMask=0; int dstAccessMask=0; };
enum {VK_STRUCTURE_TYPE_MEMORY_BARRIER=1,VK_ACCESS_SHADER_WRITE_BIT=2,VK_ACCESS_TRANSFER_WRITE_BIT=4,VK_ACCESS_SHADER_READ_BIT=8,VK_ACCESS_TRANSFER_READ_BIT=16,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT=32,VK_PIPELINE_STAGE_TRANSFER_BIT=64};
struct {int barriers=0,barriersElided=0;}g_dp;
void dp_inc(int& n){++n;}
bool g_tsOn=false,g_batching=true,g_stepBatch=false;
int g_tsBarriersElided=0,g_batchOps=0,captured=0,calls=0,lastP=0;
uint32_t g_maxComputeWorkGroupCountX=65535;
template<class... T>void vkCmdPipelineBarrier(T...){++captured;}
`)
	source.WriteString(section("void recordComputeBarrier(", "// ---- GPU timestamp stage profile"))
	source.WriteString(section("void fvk_batch_hazards_arm(", "int fvk_batch_hazards_armed"))
	source.WriteString(section("void fvk_batch_hazards_set(", "void fvk_batch_hazards_reset"))
	source.WriteString(`
struct Buffer {int id;}; Buffer* B(void* p){return static_cast<Buffer*>(p);}
struct Kernel{}; enum{K_MATMUL}; Kernel g_kern[1];
Buffer *expectedW=nullptr,*expectedX=nullptr,*expectedY=nullptr;
// Capture boundary only. Production dispatch placement remains ticket03's witness.
bool dispatch(Kernel&,Buffer** bufs,void* pc,size_t bytes,uint32_t groups){
 BatchHazardAttempt attempt{g_batchOps};
 assert(bytes==3*sizeof(int));assert(groups>0);
 assert(bufs[0]==expectedW&&bufs[1]==expectedX&&bufs[2]==expectedY);
 lastP=static_cast<int*>(pc)[2];
 if(g_batchOps>0)recordDispatchBarrier(0,g_batchOps);
 ++calls;++g_batchOps;return true;
}
`)
	source.WriteString(section("static uint32_t matmulDecodeGroups(", "void fvk_q8_matmul_2d_f32("))
	source.WriteString("int main(){Buffer w{1},x{2},q{3},wk{4},k{5}; g_batchHazardEpoch=1;\n")
	wa := decodeCutOperand(1, 1, 0, 8, 4)
	xa := decodeCutOperand(2, 2, 0, 4)
	qa := decodeCutOperand(3, 3, 0, 8)
	wka := decodeCutOperand(4, 1, 128, 4, 4)
	ka := decodeCutOperand(5, 4, 0, 4)
	qaccess, ok := vulkanF32DecodeMatMulAccesses(wa, xa, qa, 8, 4)
	if !ok {
		t.Fatal("invalid Q fixture")
	}
	kaccess, ok := vulkanF32DecodeMatMulAccesses(wka, xa, ka, 4, 4)
	if !ok {
		t.Fatal("invalid K fixture")
	}
	var ledger vulkanBatchHazardLedger
	ledger.begin()
	for i, a := range [][3]VulkanBufferAccess{qaccess, kaccess} {
		needs := ledger.recordVerdict(VulkanDispatchAccess{NodeID: i, ShimOrdinal: i, Declared: true, Accesses: a[:]})
		sync := 0
		if needs {
			sync = 1
		}
		weight, out, size := "w", "q", 8
		if i == 1 {
			weight, out, size = "wk", "k", 4
		}
		fmt.Fprintf(&source, "expectedW=&%s;expectedX=&x;expectedY=&%s;fvk_batch_hazards_arm(1);fvk_batch_hazards_set(g_batchOps,%d);fvk_matmul_f32(&%s,&x,&%s,%d,4,1);\n", weight, out, sync, weight, out, size)
	}
	source.WriteString(`
assert(calls==2&&captured==0&&g_batchHazardElisions==1&&lastP==-1);
// No declaration for an intervening op, then a conservative K verdict.
recordDispatchBarrier(0,g_batchOps++);fvk_batch_hazards_set(g_batchOps,1);
expectedW=&wk;expectedY=&k;fvk_matmul_f32(&wk,&x,&k,4,4,1);assert(captured==2);
// Profile/feature-OFF preserves the coarse barrier.
fvk_batch_hazards_arm(0);fvk_matmul_f32(&wk,&x,&k,4,4,1);assert(captured==3);
// Both actual fvk_matmul branches still issue one native dispatch.
g_maxComputeWorkGroupCountX=0;int before=calls;
fvk_matmul_f32(&wk,&x,&k,4,4,1);assert(calls==before+1&&lastP==1);
return 0;}
`)
	dir := t.TempDir()
	cpp := filepath.Join(dir, "decode-cut.cpp")
	binary := filepath.Join(dir, "decode-cut")
	if err := os.WriteFile(cpp, []byte(source.String()), 0600); err != nil {
		t.Fatal(err)
	}
	if out, err := exec.Command(compiler, "-std=c++17", "-O0", cpp, "-o", binary).CombinedOutput(); err != nil {
		t.Fatalf("native capture compile: %v\n%s", err, out)
	}
	if out, err := exec.Command(binary).CombinedOutput(); err != nil {
		t.Fatalf("native capture: %v\n%s", err, out)
	}
}
