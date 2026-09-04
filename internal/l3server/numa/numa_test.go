package numa

import (
	"testing"
)

// TestPlaceShards_CPUAndMemoryOnly verifies the 3-phase algorithm:
// Phase 1: fill CPU-bearing nodes up to 90% capacity
// Phase 2: overflow to memory-only nodes
// Phase 3: last resort on largest node
func TestPlaceShards_CPUAndMemoryOnly(t *testing.T) {
	// Simulate: node 0 = full (251.6 GB), node 1 = memory-only (1008 GB)
	specs := []NodeSpec{
		{ID: 0, MemGB: 251.6, HasCPUs: true},
		{ID: 1, MemGB: 1008.0, HasCPUs: false},
	}

	// 8 shards at 64 GB each = 512 GB total
	// Node 0 90% threshold: 226.4 GB → max 3 shards (3×64=192 ≤ 226.4)
	// Node 1 90% threshold: 907.2 GB → max 14 shards
	perShard := uint64(64) * (1 << 30)
	assignment, memOnlyUsed := PlaceShards(specs, 8, perShard)

	counts := countNodes(assignment)
	if counts[0] != 3 {
		t.Errorf("expected 3 shards on CPU node 0, got %d (assignment: %v)", counts[0], assignment)
	}
	if counts[1] != 5 {
		t.Errorf("expected 5 shards on memory-only node 1, got %d (assignment: %v)", counts[1], assignment)
	}

	// First 3 shards on CPU node (phase 1), remaining 5 on memory-only (phase 2)
	for i := 0; i < 3; i++ {
		if assignment[i] != 0 {
			t.Errorf("shard %d should be on node 0, got %d", i, assignment[i])
		}
	}
	for i := 3; i < 8; i++ {
		if assignment[i] != 1 {
			t.Errorf("shard %d should be on node 1, got %d", i, assignment[i])
		}
	}

	// Memory-only node 1 should be reported
	if len(memOnlyUsed) != 1 || memOnlyUsed[0] != 1 {
		t.Errorf("expected memOnlyUsed=[1], got %v", memOnlyUsed)
	}
}

// TestPlaceShards_TwoFullNodes verifies round-robin across two CPU nodes.
func TestPlaceShards_TwoFullNodes(t *testing.T) {
	specs := []NodeSpec{
		{ID: 0, MemGB: 256.0, HasCPUs: true},
		{ID: 1, MemGB: 256.0, HasCPUs: true},
	}

	// 6 shards at 64 GB each. 90% of 256 GB = 230.4 GB → max 3 per node.
	perShard := uint64(64) * (1 << 30)
	assignment, memOnlyUsed := PlaceShards(specs, 6, perShard)

	counts := countNodes(assignment)
	if counts[0] != 3 || counts[1] != 3 {
		t.Errorf("expected 3+3, got node0=%d node1=%d (assignment: %v)", counts[0], counts[1], assignment)
	}
	if len(memOnlyUsed) != 0 {
		t.Errorf("expected no memory-only used, got %v", memOnlyUsed)
	}
}

// TestPlaceShards_AllFitOnCPU verifies no overflow when CPU node has capacity.
func TestPlaceShards_AllFitOnCPU(t *testing.T) {
	specs := []NodeSpec{
		{ID: 0, MemGB: 251.6, HasCPUs: true},
		{ID: 1, MemGB: 1008.0, HasCPUs: false},
	}

	// 2 shards at 64 GB each. Node 0 max=3, so 2 fits entirely.
	perShard := uint64(64) * (1 << 30)
	assignment, memOnlyUsed := PlaceShards(specs, 2, perShard)

	counts := countNodes(assignment)
	if counts[0] != 2 {
		t.Errorf("expected 2 shards on node 0, got %d", counts[0])
	}
	if counts[1] != 0 {
		t.Errorf("expected 0 shards on node 1, got %d", counts[1])
	}
	if len(memOnlyUsed) != 0 {
		t.Errorf("expected no memory-only used, got %v", memOnlyUsed)
	}
}

// TestPlaceShards_OverflowPhase3 verifies last-resort overflow to largest node.
func TestPlaceShards_OverflowPhase3(t *testing.T) {
	specs := []NodeSpec{
		{ID: 0, MemGB: 256.0, HasCPUs: true},
		{ID: 1, MemGB: 256.0, HasCPUs: true},
	}

	// 8 shards at 64 GB each. Max 3 per node = 6 via phase 1.
	// Remaining 2 go to phase 3 → largest (same size, picks first = node 0).
	perShard := uint64(64) * (1 << 30)
	assignment, _ := PlaceShards(specs, 8, perShard)

	counts := countNodes(assignment)
	total := counts[0] + counts[1]
	if total != 8 {
		t.Errorf("expected 8 total, got %d", total)
	}
	// Both should have at least 3 (from phase 1)
	if counts[0] < 3 || counts[1] < 3 {
		t.Errorf("expected ≥3 per node, got node0=%d node1=%d", counts[0], counts[1])
	}
}

// TestPlaceShards_OnlyCPUNodesNoMemory verifies empty specs edge case.
func TestPlaceShards_EmptySpecs(t *testing.T) {
	assignment, _ := PlaceShards(nil, 4, 1<<30)
	// All default to 0 (zero-value)
	for i, v := range assignment {
		if v != 0 {
			t.Errorf("shard %d: expected 0, got %d", i, v)
		}
	}
}

// TestPlaceShards_OnlyMemoryOnlyNodes verifies all-memory-only scenario (CXL).
func TestPlaceShards_OnlyMemoryOnlyNodes(t *testing.T) {
	specs := []NodeSpec{
		{ID: 2, MemGB: 512.0, HasCPUs: false},
		{ID: 3, MemGB: 512.0, HasCPUs: false},
	}

	perShard := uint64(64) * (1 << 30)
	assignment, memOnlyUsed := PlaceShards(specs, 4, perShard)

	counts := countNodes(assignment)
	if counts[2]+counts[3] != 4 {
		t.Errorf("expected 4 total, got %v", counts)
	}
	if len(memOnlyUsed) == 0 {
		t.Errorf("expected memory-only nodes used, got none")
	}
}

// TestPlaceShards_SingleNode verifies single node gets all shards.
func TestPlaceShards_SingleNode(t *testing.T) {
	specs := []NodeSpec{
		{ID: 0, MemGB: 1024.0, HasCPUs: true},
	}

	perShard := uint64(64) * (1 << 30)
	assignment, _ := PlaceShards(specs, 8, perShard)

	for i, v := range assignment {
		if v != 0 {
			t.Errorf("shard %d: expected node 0, got %d", i, v)
		}
	}
}

// TestNodeShardBudgets verifies per-node byte computation.
func TestNodeShardBudgets(t *testing.T) {
	assignment := []int{0, 0, 0, 1, 1, 1, 1, 1}
	perShard := uint64(64) * (1 << 30)

	budgets := NodeShardBudgets(assignment, perShard)
	if len(budgets) != 2 {
		t.Fatalf("expected 2 budgets, got %d", len(budgets))
	}

	if budgets[0].NodeID != 0 || budgets[0].Shards != 3 {
		t.Errorf("node 0: expected 3 shards, got %+v", budgets[0])
	}
	if budgets[0].BytesNeed != 3*64*(1<<30) {
		t.Errorf("node 0: bytes mismatch: got %d", budgets[0].BytesNeed)
	}
	if budgets[1].NodeID != 1 || budgets[1].Shards != 5 {
		t.Errorf("node 1: expected 5 shards, got %+v", budgets[1])
	}
}

// TestNodeShardBudgets_SkipsNegative verifies -1 entries are excluded.
func TestNodeShardBudgets_SkipsNegative(t *testing.T) {
	budgets := NodeShardBudgets([]int{-1, -1, -1}, 1<<30)
	if len(budgets) != 0 {
		t.Errorf("expected 0 budgets, got %d", len(budgets))
	}
}

// TestNodeTypeString verifies string representation.
func TestNodeTypeString(t *testing.T) {
	tests := []struct {
		nt   NodeType
		want string
	}{
		{NodeTypeFull, "full"},
		{NodeTypeMemoryOnly, "memory-only"},
		{NodeTypeCPUOnly, "cpu-only"},
		{NodeType(99), "unknown"},
	}
	for _, tt := range tests {
		if got := tt.nt.String(); got != tt.want {
			t.Errorf("NodeType(%d).String() = %q, want %q", tt.nt, got, tt.want)
		}
	}
}

// TestValidateNodeBudgets_NilTopology verifies nil safety.
func TestValidateNodeBudgets_NilTopology(t *testing.T) {
	var topo *Topology
	assignment := []int{0, 1, 0, 1}
	corrected, warning := topo.ValidateNodeBudgets(assignment, 1<<30)
	if warning != "" {
		t.Errorf("expected no warning for nil topology, got: %s", warning)
	}
	for i, v := range corrected {
		if v != assignment[i] {
			t.Errorf("index %d: got %d want %d", i, v, assignment[i])
		}
	}
}

// Note: redistributeByCapacity warning message tests require Linux build
// (sysfs dependency). The core algorithm is tested via PlaceShards above.

// TestPlaceShards_RoundRobinOrder verifies alternating assignment on equal nodes.
func TestPlaceShards_RoundRobinOrder(t *testing.T) {
	specs := []NodeSpec{
		{ID: 0, MemGB: 256.0, HasCPUs: true},
		{ID: 1, MemGB: 256.0, HasCPUs: true},
	}
	// 6 shards at 64 GB. Equal capacity → should alternate 0,1,0,1,0,1.
	perShard := uint64(64) * (1 << 30)
	assignment, _ := PlaceShards(specs, 6, perShard)

	expected := []int{0, 1, 0, 1, 0, 1}
	for i, want := range expected {
		if assignment[i] != want {
			t.Errorf("shard %d: got node %d, want %d (full: %v)", i, assignment[i], want, assignment)
			break
		}
	}
}

// TestPlaceShards_RoundRobinThreeNodes verifies round-robin across 3 nodes.
func TestPlaceShards_RoundRobinThreeNodes(t *testing.T) {
	specs := []NodeSpec{
		{ID: 0, MemGB: 512.0, HasCPUs: true},
		{ID: 1, MemGB: 512.0, HasCPUs: true},
		{ID: 2, MemGB: 512.0, HasCPUs: true},
	}
	perShard := uint64(64) * (1 << 30)
	assignment, _ := PlaceShards(specs, 6, perShard)

	expected := []int{0, 1, 2, 0, 1, 2}
	for i, want := range expected {
		if assignment[i] != want {
			t.Errorf("shard %d: got node %d, want %d (full: %v)", i, assignment[i], want, assignment)
			break
		}
	}
}

// TestPlaceShards_Phase3Oversubscription verifies phase 3 can exceed 90% threshold.
func TestPlaceShards_Phase3Oversubscription(t *testing.T) {
	// Two small nodes: 90% of 72 GB = 64.8 GB → max 1 shard at 64 GB each.
	specs := []NodeSpec{
		{ID: 0, MemGB: 72.0, HasCPUs: true},
		{ID: 1, MemGB: 72.0, HasCPUs: true},
	}
	perShard := uint64(64) * (1 << 30)
	assignment, _ := PlaceShards(specs, 4, perShard)

	counts := countNodes(assignment)
	total := counts[0] + counts[1]
	if total != 4 {
		t.Fatalf("expected 4 total shards, got %d", total)
	}
	// Phase 1 places 1 per node (2 total). Phase 3 puts remaining 2 on largest.
	// Since equal, picks first (node 0). Node 0 ends up with 3 shards = 192 GB
	// on a 72 GB node — well beyond 90%.
	oversubscribed := false
	for _, c := range counts {
		if c > 1 {
			oversubscribed = true
		}
	}
	if !oversubscribed {
		t.Errorf("expected phase 3 oversubscription (at least one node >1 shard), got %v", counts)
	}
}

// TestPlaceShards_UnequalCapacity verifies shards fill larger node more.
func TestPlaceShards_UnequalCapacity(t *testing.T) {
	specs := []NodeSpec{
		{ID: 0, MemGB: 128.0, HasCPUs: true},  // 90% = 115.2 GB → max 1 shard
		{ID: 1, MemGB: 512.0, HasCPUs: true},   // 90% = 460.8 GB → max 7 shards
	}
	perShard := uint64(64) * (1 << 30)
	assignment, _ := PlaceShards(specs, 8, perShard)

	counts := countNodes(assignment)
	if counts[0] != 1 {
		t.Errorf("small node 0: expected 1 shard, got %d", counts[0])
	}
	if counts[1] != 7 {
		t.Errorf("large node 1: expected 7 shards, got %d", counts[1])
	}
}

// --- Helpers ---

func countNodes(assignment []int) map[int]int {
	m := make(map[int]int)
	for _, n := range assignment {
		m[n]++
	}
	return m
}
