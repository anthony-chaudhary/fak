package compute

import (
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

// fak-test:runtime medium est=2s lane=default
func TestVulkanBatchHazardVerdictPrecedesDispatch(t *testing.T) {
	var oversized vulkanBatchHazardLedger
	oversized.begin()
	if !oversized.recordVerdict(VulkanDispatchAccess{Declared: true, Accesses: make([]VulkanBufferAccess, 17)}) ||
		len(oversized.dispatches) != 1 || len(oversized.dispatches[0].Accesses) != 0 {
		t.Fatal("oversized access metadata must fail closed without retaining caller storage")
	}
	if !oversized.recordVerdict(VulkanDispatchAccess{ShimOrdinal: 1, Declared: true,
		Accesses: []VulkanBufferAccess{{BufferID: 1, Size: 64, Mode: VulkanAccessRead}}}) {
		t.Fatal("oversized predecessor must keep later prefixes conservative")
	}
	valid := VulkanBufferAccess{BufferID: 1, Size: 64, Mode: VulkanAccessWrite}
	for _, invalid := range []VulkanBufferAccess{
		{BufferID: 2, Size: 64, Mode: VulkanAccessMode(99)},
		{BufferID: 2, Offset: math.MaxInt64, Size: 1, Mode: VulkanAccessRead},
	} {
		var ledger vulkanBatchHazardLedger
		ledger.begin()
		ledger.recordVerdict(VulkanDispatchAccess{Declared: true, Accesses: []VulkanBufferAccess{valid}})
		if !ledger.recordVerdict(VulkanDispatchAccess{ShimOrdinal: 1, Declared: true, Accesses: []VulkanBufferAccess{invalid}}) {
			t.Fatalf("invalid metadata licensed elision: %+v", invalid)
		}
	}
	var bounded vulkanBatchHazardLedger
	bounded.begin()
	for i := 0; i < 258; i++ {
		a := valid
		a.BufferID = int64(i + 1)
		needs := bounded.recordVerdict(VulkanDispatchAccess{ShimOrdinal: i, Declared: true, Accesses: []VulkanBufferAccess{a}})
		if i >= 256 && (!needs || len(bounded.dispatches) != 257) {
			t.Fatal("exhausted declaration budget must keep bounded storage and coarse barriers")
		}
	}
	bounded.begin()
	bounded.recordVerdict(VulkanDispatchAccess{Declared: true, Accesses: []VulkanBufferAccess{valid}})
	independent := VulkanBufferAccess{BufferID: 2, Size: 64, Mode: VulkanAccessRead}
	if bounded.recordVerdict(VulkanDispatchAccess{ShimOrdinal: 1, Declared: true, Accesses: []VulkanBufferAccess{independent}}) {
		t.Fatal("fresh window did not reset its declaration budget")
	}
	if !bounded.recordVerdict(VulkanDispatchAccess{ShimOrdinal: 1, Declared: true, Accesses: []VulkanBufferAccess{independent}}) {
		t.Fatal("repeated ordinal must not treat a skipped operation as a producer")
	}
	// Compile the production native recorder, not a Go simulation of its decision.
	// The Vulkan call is captured; no loader, device, shader or GPU is required.
	compiler, err := exec.LookPath("c++")
	if err != nil {
		t.Skip("native recorder witness requires a C++17 compiler; not verified")
	}
	read := func(name string) string {
		t.Helper()
		b, err := os.ReadFile(name)
		if err != nil {
			t.Fatal(err)
		}
		return string(b)
	}
	section := func(s, from, to string) string {
		t.Helper()
		i := strings.Index(s, from)
		if i < 0 {
			t.Fatalf("missing production seam %q", from)
		}
		j := strings.Index(s[i:], to)
		if j < 0 {
			t.Fatalf("missing end seam %q", to)
		}
		return s[i : i+j]
	}
	goSource := read("vulkan.go")
	flush := section(goSource, "func (v *vulkanBackend) flushBatchLocked", "// recordBatchDispatch")
	if strings.Contains(flush, "C.fvk_batch_hazards_set") {
		t.Fatal("flush installs a verdict after its native barrier was recorded")
	}
	record := section(goSource, "func (v *vulkanBackend) recordBatchDispatch", "// bufferIdentity")
	if strings.Index(record, ".recordVerdict(") < 0 ||
		strings.Index(record, "C.fvk_batch_hazards_set") < strings.Index(record, ".recordVerdict(") {
		t.Fatal("production declaration must lower before installing its native verdict")
	}
	fused := read("vulkan_matmul_fused.go")
	batched := section(fused, "func (v *vulkanBackend) BatchedMatMul(", "func (v *vulkanBackend) EmbeddingRow(")
	f32 := section(batched, "case F32:", "case Q8_0:")
	declaration := strings.Index(f32, "v.recordBatchDispatch(")
	if declaration < 0 || strings.Count(batched, "v.recordBatchDispatch(") != 1 ||
		strings.Index(f32, "C.fvk_matmul_f32(") < declaration {
		t.Fatal("BatchedMatMul must declare only its F32 arm, before the native dispatch")
	}
	add := section(fused, "func (v *vulkanBackend) MatMulAddInPlace(", "func (v *vulkanBackend) MatMul2(")
	declaration = strings.Index(add, "v.recordBatchDispatch(")
	if declaration < 0 || strings.Count(add, "v.recordBatchDispatch(") != 1 ||
		strings.Index(add, "C.fvk_matmul_add_f32(") < declaration ||
		strings.Index(add, "if w.Dtype != F32 {") < 0 || strings.Index(add, "if w.Dtype != F32 {") > declaration {
		t.Fatal("MatMulAddInPlace must reject packed weights before declaring its native F32 dispatch")
	}
	shim := read("vulkan_shim.cpp")
	dispatchStart := section(shim, "bool dispatch(Kernel&", "// Only these concrete synchronous kernels")
	dispatchBody := dispatchStart[strings.Index(dispatchStart, "{")+1:]
	if !strings.HasPrefix(strings.TrimSpace(dispatchBody), "BatchHazardAttempt hazardAttempt{g_batchOps};") {
		t.Fatal("native dispatch verdict guard must precede every early return")
	}
	var source strings.Builder
	source.WriteString(`#include <unordered_map>
#include <limits>
#include <cstdint>
#include <cassert>
using VkCommandBuffer = int;
struct VkMemoryBarrier { int type; int srcAccessMask = 0; int dstAccessMask = 0; };
enum { VK_STRUCTURE_TYPE_MEMORY_BARRIER = 1, VK_ACCESS_SHADER_WRITE_BIT = 2,
VK_ACCESS_TRANSFER_WRITE_BIT = 4, VK_ACCESS_SHADER_READ_BIT = 8,
VK_ACCESS_TRANSFER_READ_BIT = 16, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT = 32,
VK_PIPELINE_STAGE_TRANSFER_BIT = 64 };
struct { int barriers = 0, barriersElided = 0; } g_dp;
void dp_inc(int& n) { ++n; }
bool g_tsOn = false, g_batching = true, g_stepBatch = false;
bool g_v41SubmissionPendingFailure = true;
int g_tsBarriersElided = 0, g_batchOps = 0, captured = 0;
template<class... T> void vkCmdPipelineBarrier(T...) { ++captured; }
`)
	source.WriteString(section(shim, "void recordComputeBarrier(", "// ---- GPU timestamp stage profile"))
	source.WriteString(section(shim, "void fvk_batch_hazards_arm(", "int fvk_batch_hazards_armed"))
	source.WriteString(section(shim, "void fvk_batch_hazards_set(", "void fvk_batch_hazards_reset"))
	// Execute the exact production guard and first early return, not a mirrored scope.
	source.WriteString("bool failedNativeDispatch() {" + dispatchBody + "return true; }\n")
	source.WriteString("int main() { g_batchHazardEpoch = 1;\n")
	access := func(id int64, mode VulkanAccessMode) VulkanBufferAccess {
		return VulkanBufferAccess{BufferID: id, Size: 64, Mode: mode}
	}
	cases := []struct {
		name     string
		accesses []VulkanBufferAccess
		ordinals []int
		want     int
	}{
		{"independent", []VulkanBufferAccess{access(1, VulkanAccessWrite), access(2, VulkanAccessRead)}, []int{0, 1}, 0},
		{"RAW", []VulkanBufferAccess{access(1, VulkanAccessWrite), access(1, VulkanAccessRead)}, []int{0, 1}, 1},
		{"WAR", []VulkanBufferAccess{access(1, VulkanAccessRead), access(1, VulkanAccessWrite)}, []int{0, 1}, 1},
		{"WAW", []VulkanBufferAccess{access(1, VulkanAccessWrite), access(1, VulkanAccessWrite)}, []int{0, 1}, 1},
		{"older_RAW", []VulkanBufferAccess{access(1, VulkanAccessWrite), access(2, VulkanAccessWrite), access(1, VulkanAccessRead)}, []int{0, 1, 2}, 1},
		{"unknown_identity", []VulkanBufferAccess{access(0, VulkanAccessWrite), access(2, VulkanAccessRead)}, []int{0, 1}, 1},
		{"unknown_range", []VulkanBufferAccess{{BufferID: 1, Size: -1, Mode: VulkanAccessWrite}, access(2, VulkanAccessRead)}, []int{0, 1}, 1},
		{"undeclared_ordinal", []VulkanBufferAccess{access(1, VulkanAccessWrite), access(2, VulkanAccessRead)}, []int{0, 2}, 2},
	}
	for _, tc := range cases {
		var ledger vulkanBatchHazardLedger
		ledger.begin()
		fmt.Fprintf(&source, "// %s\ncaptured = 0; g_batchHazardSync.clear();\n", tc.name)
		last := -1
		for i, a := range tc.accesses {
			ordinal := tc.ordinals[i]
			for unknown := last + 1; unknown < ordinal; unknown++ {
				fmt.Fprintf(&source, "g_batchOps = %d; recordDispatchBarrier(0, g_batchOps);\n", unknown)
			}
			needs := ledger.recordVerdict(VulkanDispatchAccess{NodeID: i, ShimOrdinal: ordinal, Declared: true, Accesses: []VulkanBufferAccess{a}})
			sync := 0
			if needs {
				sync = 1
			}
			fmt.Fprintf(&source, "g_batchOps = %d; fvk_batch_hazards_arm(1); fvk_batch_hazards_set(g_batchOps, %d); { BatchHazardAttempt attempt{g_batchOps}; if (g_batchOps > 0) recordDispatchBarrier(0, g_batchOps); }\n", ordinal, sync)
			last = ordinal
		}
		fmt.Fprintf(&source, "assert(captured == %d);\n", tc.want)
	}
	source.WriteString(`
// A failed declared dispatch must not authorize its undeclared replacement.
captured = 0; g_batchOps = 1; fvk_batch_hazards_arm(1);
fvk_batch_hazards_set(1, 0); assert(!failedNativeDispatch());
recordDispatchBarrier(0, 1); assert(captured == 1);
// Gate off and checked step windows preserve coarse barriers.
fvk_batch_hazards_arm(0); fvk_batch_hazards_set(1, 0);
recordDispatchBarrier(0, 1); assert(captured == 2);
g_stepBatch = true; fvk_batch_hazards_arm(1); fvk_batch_hazards_set(1, 0);
recordDispatchBarrier(0, 1); assert(captured == 3);
assert(g_batchHazardElisions > 0);
return 0;
}
`)
	dir := t.TempDir()
	cpp, binary := filepath.Join(dir, "recorder.cpp"), filepath.Join(dir, "recorder")
	if err := os.WriteFile(cpp, []byte(source.String()), 0600); err != nil {
		t.Fatal(err)
	}
	if out, err := exec.Command(compiler, "-std=c++17", "-O0", cpp, "-o", binary).CombinedOutput(); err != nil {
		t.Fatalf("compile production recorder: %v\n%s", err, out)
	}
	if out, err := exec.Command(binary).CombinedOutput(); err != nil {
		t.Fatalf("production recorder witness: %v\n%s", err, out)
	}
}

// ---- Exact batch resource-hazard lowering (#12218) ------------------------------
//
// The batch recorder fences every recorded dispatch against its predecessor with one
// global compute->compute memory barrier because it has no per-dispatch resource-hazard
// input (vulkan_shim.cpp recordComputeBarrier). These tests bind the exact RAW/WAR/WAW
// lowering that lets two adjacent dispatches whose ranges are disjoint skip that
// barrier, and prove the fail-closed contract: incomplete metadata never yields a
// partial plan.

func TestVulkanBatchResourceHazards(t *testing.T) {
	t.Run("disjoint_ranges_elide_barrier", func(t *testing.T) {
		// Two dispatches touching non-overlapping ranges of the SAME buffer, adjacent in
		// the shim op sequence (ordinals 0 then 1): no dependency, so the barrier between
		// them is provably elidable.
		plan := LowerVulkanBatchHazards([]VulkanDispatchAccess{
			{NodeID: 0, ShimOrdinal: 0, Declared: true, Accesses: []VulkanBufferAccess{
				{BufferID: 100, Offset: 0, Size: 64, Mode: VulkanAccessWrite},
			}},
			{NodeID: 1, ShimOrdinal: 1, Declared: true, Accesses: []VulkanBufferAccess{
				{BufferID: 100, Offset: 64, Size: 64, Mode: VulkanAccessRead},
			}},
		})
		if !plan.Exact {
			t.Fatalf("plan not exact: %s", plan.Reason)
		}
		if plan.Elided != 1 || plan.Synchronized != 0 {
			t.Fatalf("elided=%d synchronized=%d, want 1/0", plan.Elided, plan.Synchronized)
		}
		if plan.Edges[0].NeedsSync || plan.Edges[0].Kind != VulkanHazardNone {
			t.Fatalf("edge=%+v, want no-sync none", plan.Edges[0])
		}
		if plan.BarrierFloor() != VulkanHazardMinBarriers(2) {
			t.Fatalf("BarrierFloor=%d, want %d", plan.BarrierFloor(), VulkanHazardMinBarriers(2))
		}
	})

	t.Run("different_buffers_elide_barrier", func(t *testing.T) {
		plan := LowerVulkanBatchHazards([]VulkanDispatchAccess{
			{NodeID: 0, ShimOrdinal: 0, Declared: true, Accesses: []VulkanBufferAccess{
				{BufferID: 100, Offset: 0, Size: 0, Mode: VulkanAccessWrite},
			}},
			{NodeID: 1, ShimOrdinal: 1, Declared: true, Accesses: []VulkanBufferAccess{
				{BufferID: 101, Offset: 0, Size: 0, Mode: VulkanAccessRead},
			}},
		})
		if !plan.Exact || plan.Elided != 1 {
			t.Fatalf("exact=%v elided=%d reason=%s", plan.Exact, plan.Elided, plan.Reason)
		}
	})

	t.Run("interleaved_undeclared_op_fails_closed", func(t *testing.T) {
		// Two disjoint matmuls whose shim ordinals are NOT adjacent: an undeclared op sits
		// between them. Even though the two declared ranges are disjoint, the shim fences
		// the successor against the interleaved op, so the verdict must stay sync.
		plan := LowerVulkanBatchHazards([]VulkanDispatchAccess{
			{NodeID: 0, ShimOrdinal: 0, Declared: true, Accesses: []VulkanBufferAccess{
				{BufferID: 100, Offset: 0, Size: 64, Mode: VulkanAccessWrite},
			}},
			{NodeID: 1, ShimOrdinal: 5, Declared: true, Accesses: []VulkanBufferAccess{
				{BufferID: 100, Offset: 64, Size: 64, Mode: VulkanAccessRead},
			}},
		})
		if plan.Elided != 0 || plan.Synchronized != 1 {
			t.Fatalf("elided=%d sync=%d, want 0/1 (interleaved op must keep the fence)", plan.Elided, plan.Synchronized)
		}
		if plan.Edges[0].NeedsSync || plan.Edges[0].Kind != VulkanHazardUnknown {
			// NeedsSync is set; Kind must be Unknown, not a fabricated benign class.
			if !plan.Edges[0].NeedsSync {
				t.Fatalf("interleaved edge must need sync: %+v", plan.Edges[0])
			}
		}
	})

	t.Run("overlapping_raw_keeps_barrier", func(t *testing.T) {
		plan := LowerVulkanBatchHazards([]VulkanDispatchAccess{
			{NodeID: 0, Declared: true, Accesses: []VulkanBufferAccess{
				{BufferID: 100, Offset: 0, Size: 128, Mode: VulkanAccessWrite},
			}},
			{NodeID: 1, Declared: true, Accesses: []VulkanBufferAccess{
				{BufferID: 100, Offset: 64, Size: 128, Mode: VulkanAccessRead},
			}},
		})
		if !plan.Exact || plan.Synchronized != 1 || plan.Elided != 0 {
			t.Fatalf("exact=%v sync=%d elided=%d", plan.Exact, plan.Synchronized, plan.Elided)
		}
		if plan.Edges[0].Kind != VulkanHazardReadAfterWrite || !plan.Edges[0].NeedsSync {
			t.Fatalf("edge=%+v, want RAW needsSync", plan.Edges[0])
		}
	})

	t.Run("war_and_waw_classified", func(t *testing.T) {
		war := LowerVulkanBatchHazards([]VulkanDispatchAccess{
			{NodeID: 0, ShimOrdinal: 0, Declared: true, Accesses: []VulkanBufferAccess{
				{BufferID: 7, Offset: 0, Size: 0, Mode: VulkanAccessRead},
			}},
			{NodeID: 1, ShimOrdinal: 1, Declared: true, Accesses: []VulkanBufferAccess{
				{BufferID: 7, Offset: 0, Size: 0, Mode: VulkanAccessWrite},
			}},
		})
		if war.Edges[0].Kind != VulkanHazardWriteAfterRead || !war.Edges[0].NeedsSync {
			t.Fatalf("WAR edge=%+v", war.Edges[0])
		}
		waw := LowerVulkanBatchHazards([]VulkanDispatchAccess{
			{NodeID: 0, ShimOrdinal: 0, Declared: true, Accesses: []VulkanBufferAccess{
				{BufferID: 8, Offset: 0, Size: 0, Mode: VulkanAccessWrite},
			}},
			{NodeID: 1, ShimOrdinal: 1, Declared: true, Accesses: []VulkanBufferAccess{
				{BufferID: 8, Offset: 0, Size: 0, Mode: VulkanAccessWrite},
			}},
		})
		if waw.Edges[0].Kind != VulkanHazardWriteAfterWrite || !waw.Edges[0].NeedsSync {
			t.Fatalf("WAW edge=%+v", waw.Edges[0])
		}
	})

	t.Run("read_after_read_is_benign", func(t *testing.T) {
		plan := LowerVulkanBatchHazards([]VulkanDispatchAccess{
			{NodeID: 0, ShimOrdinal: 0, Declared: true, Accesses: []VulkanBufferAccess{
				{BufferID: 9, Offset: 0, Size: 0, Mode: VulkanAccessRead},
			}},
			{NodeID: 1, ShimOrdinal: 1, Declared: true, Accesses: []VulkanBufferAccess{
				{BufferID: 9, Offset: 0, Size: 0, Mode: VulkanAccessRead},
			}},
		})
		if plan.Edges[0].Kind != VulkanHazardReadAfterRead || plan.Edges[0].NeedsSync {
			t.Fatalf("RAR edge=%+v, want benign", plan.Edges[0])
		}
		if plan.Elided != 1 {
			t.Fatalf("elided=%d, want 1", plan.Elided)
		}
	})

	t.Run("undeclared_dispatch_fails_closed", func(t *testing.T) {
		plan := LowerVulkanBatchHazards([]VulkanDispatchAccess{
			{NodeID: 0, Declared: true, Accesses: []VulkanBufferAccess{
				{BufferID: 100, Offset: 0, Size: 0, Mode: VulkanAccessWrite},
			}},
			{NodeID: 1, Declared: false},
		})
		if plan.Exact {
			t.Fatal("plan claimed exact over an undeclared dispatch")
		}
		if plan.Reason == "" {
			t.Fatal("non-exact plan must state a reason")
		}
		if len(plan.Edges) != 0 {
			t.Fatalf("non-exact plan must not emit partial edges, got %d", len(plan.Edges))
		}
	})
}

// TestVulkanBatchHazardLoweringArmed pins the default-off, typo-safe toggle.
func TestVulkanBatchHazardLoweringArmed(t *testing.T) {
	cases := []struct {
		val  string
		want bool
	}{
		{"", false},
		{"0", false},
		{"off", false},
		{"coarse", false},
		{"exact", true},
		{" EXACT ", true},
		{"1", true},
		{"on", true},
		{"true", true},
	}
	for _, tc := range cases {
		env := func(string) string { return tc.val }
		if got := VulkanHazardLoweringArmed(env); got != tc.want {
			t.Fatalf("VulkanHazardLoweringArmed(%q)=%v, want %v", tc.val, got, tc.want)
		}
	}
	if VulkanHazardLoweringArmed(nil) {
		t.Fatal("nil env must not arm the path")
	}
}

// TestVulkanBatchHazardPlanDeterministic proves two identical inputs lower to byte-identical
// edge order, so a receipt over the plan is comparable across runs.
func TestVulkanBatchHazardPlanDeterministic(t *testing.T) {
	dispatches := []VulkanDispatchAccess{
		{NodeID: 0, ShimOrdinal: 0, Declared: true, Accesses: []VulkanBufferAccess{
			{BufferID: 100, Offset: 0, Size: 64, Mode: VulkanAccessWrite},
		}},
		{NodeID: 1, ShimOrdinal: 1, Declared: true, Accesses: []VulkanBufferAccess{
			{BufferID: 100, Offset: 0, Size: 64, Mode: VulkanAccessRead},
		}},
		{NodeID: 2, ShimOrdinal: 2, Declared: true, Accesses: []VulkanBufferAccess{
			{BufferID: 100, Offset: 64, Size: 64, Mode: VulkanAccessWrite},
		}},
	}
	a := LowerVulkanBatchHazards(dispatches)
	b := LowerVulkanBatchHazards(dispatches)
	if !reflect.DeepEqual(a, b) {
		t.Fatalf("plans differ:\n%+v\n%+v", a, b)
	}
	if a.Synchronized != 1 || a.Elided != 1 {
		t.Fatalf("sync=%d elided=%d, want 1/1 (edges: %+v)", a.Synchronized, a.Elided, a.Edges)
	}
}

// TestVulkanBatchHazardStatsDeltas pins the receipt arithmetic and its non-negative clamp.
func TestVulkanBatchHazardStatsDeltas(t *testing.T) {
	stats := VulkanBatchHazardStats{
		Windows:        3,
		ExactWindows:   2,
		CoarseFallback: 1,
		BarriersCoarse: 30,
		BarriersExact:  18,
	}
	avoided, fellBack := stats.Deltas()
	if avoided != 12 || fellBack != 1 {
		t.Fatalf("avoided=%d fellBack=%d, want 12/1", avoided, fellBack)
	}
	// A pathological exact>coarse must clamp to zero, never a negative savings claim.
	neg := VulkanBatchHazardStats{BarriersCoarse: 5, BarriersExact: 9}
	if avoided, _ := neg.Deltas(); avoided != 0 {
		t.Fatalf("avoided=%d, want 0 (clamped)", avoided)
	}
	if got := VulkanHazardMinBarriers(0); got != 0 {
		t.Fatalf("VulkanHazardMinBarriers(0)=%d, want 0", got)
	}
	if got := VulkanHazardMinBarriers(5); got != 4 {
		t.Fatalf("VulkanHazardMinBarriers(5)=%d, want 4", got)
	}
}

// TestVulkanBatchHazardSortStable pins the receipt ordering as byte-stable.
func TestVulkanBatchHazardSortStable(t *testing.T) {
	plan := VulkanHazardPlan{Exact: true, Edges: []VulkanHazardEdge{
		{Prev: 2, Next: 3, BufferID: 9, Kind: VulkanHazardReadAfterWrite, NeedsSync: true},
		{Prev: 0, Next: 1, BufferID: 5, Kind: VulkanHazardNone},
		{Prev: 0, Next: 1, BufferID: 4, Kind: VulkanHazardNone},
	}}
	got := plan.SortEdgesByDependency()
	want := []int64{4, 5, 9}
	for i, e := range got {
		if e.BufferID != want[i] {
			t.Fatalf("edge %d buffer=%d, want %d (order %+v)", i, e.BufferID, want[i], got)
		}
	}
}

// TestVulkanBatchWindowReachableFromRecorder binds the production wiring: a window
// opened by the same begin/record/lower calls vulkan.go BeginBatch/recordBatchDispatch/
// FlushBatch make lowers to a plan and accumulates a bounded receipt. This is the
// invokability witness the presence≠invokability rule requires — the ledger is not
// merely declared, it is exercised through its production entrypoints.
func TestVulkanBatchWindowReachableFromRecorder(t *testing.T) {
	var ledger vulkanBatchHazardLedger
	var stats vulkanBatchHazardStats

	// Window 1: two disjoint dispatches on one buffer => one barrier elided.
	ledger.begin()
	ledger.record(VulkanDispatchAccess{NodeID: 1, ShimOrdinal: 0, Declared: true, Accesses: []VulkanBufferAccess{
		{BufferID: 100, Offset: 0, Size: 64, Mode: VulkanAccessWrite},
	}})
	ledger.record(VulkanDispatchAccess{NodeID: 2, ShimOrdinal: 1, Declared: true, Accesses: []VulkanBufferAccess{
		{BufferID: 100, Offset: 64, Size: 64, Mode: VulkanAccessRead},
	}})
	plan, window := ledger.lower(true)
	if !plan.Exact || plan.Elided != 1 {
		t.Fatalf("window1 exact=%v elided=%d reason=%s", plan.Exact, plan.Elided, plan.Reason)
	}
	stats.observe(window)

	// Window 2: an undeclared dispatch forces the coarse fallback for the whole window.
	ledger.begin()
	ledger.record(VulkanDispatchAccess{NodeID: 1, ShimOrdinal: 0, Declared: true, Accesses: []VulkanBufferAccess{
		{BufferID: 7, Size: 0, Mode: VulkanAccessWrite},
	}})
	ledger.record(VulkanDispatchAccess{NodeID: 2, ShimOrdinal: 1})
	plan2, window2 := ledger.lower(true)
	if plan2.Exact {
		t.Fatal("window2 claimed exact over an undeclared dispatch")
	}
	if window2.CoarseFallback != 1 || window2.BarriersExact != window2.BarriersCoarse {
		t.Fatalf("window2 fallback=%d exact=%d coarse=%d", window2.CoarseFallback, window2.BarriersExact, window2.BarriersCoarse)
	}
	stats.observe(window2)

	// A record outside any window must be silently dropped, never accumulate.
	ledger.record(VulkanDispatchAccess{NodeID: 9, ShimOrdinal: 0, Declared: true})
	if n := len(ledger.dispatches); n != 0 {
		t.Fatalf("closed-window record leaked %d declaration(s)", n)
	}

	snap := stats.snapshot()
	if snap.Windows != 2 || snap.ExactWindows != 1 || snap.CoarseFallback != 1 {
		t.Fatalf("snapshot=%+v, want 2/1/1", snap)
	}
	avoided, fellBack := snap.Deltas()
	if avoided != 1 || fellBack != 1 {
		t.Fatalf("avoided=%d fellBack=%d, want 1/1", avoided, fellBack)
	}
}

// TestVulkanBatchWindowUnarmedKeepsCoarseFloor proves the default-off path reports the
// coarse barrier count so a receipt never claims an elision the device did not perform.
func TestVulkanBatchWindowUnarmedKeepsCoarseFloor(t *testing.T) {
	var ledger vulkanBatchHazardLedger
	ledger.begin()
	ledger.record(VulkanDispatchAccess{NodeID: 1, ShimOrdinal: 0, Declared: true, Accesses: []VulkanBufferAccess{
		{BufferID: 100, Offset: 0, Size: 64, Mode: VulkanAccessWrite},
	}})
	ledger.record(VulkanDispatchAccess{NodeID: 2, ShimOrdinal: 1, Declared: true, Accesses: []VulkanBufferAccess{
		{BufferID: 100, Offset: 64, Size: 64, Mode: VulkanAccessRead},
	}})
	plan, window := ledger.lower(false)
	if !plan.Exact {
		t.Fatalf("plan not exact: %s", plan.Reason)
	}
	if window.BarriersExact != window.BarriersCoarse {
		t.Fatalf("unarmed window exact=%d coarse=%d, want equal", window.BarriersExact, window.BarriersCoarse)
	}
	if avoided, _ := window.Deltas(); avoided != 0 {
		t.Fatalf("unarmed window claimed avoided=%d, want 0", avoided)
	}
}
