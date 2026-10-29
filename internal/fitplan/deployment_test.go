package fitplan

import (
	"encoding/json"
	"errors"
	"math"
	"testing"
)

// fak-test:runtime fast est=1s
// fak-test:justify why=Pin per-rank and shared-pool capacity contracts for unchanged model composition. when=changed:internal/fitplan/deployment*
func TestDeploymentFixedModelComposition(t *testing.T) {
	weights := WeightPartition{ReplicatedBytes: 8, ShardedBytes: 24, ShardCount: 8}
	one := Placement{Name: "one", Support: SupportSupported, SupportEvidence: "declared compatible collective", Weights: weights, Ranks: []RankPlacement{{Rank: 0, PoolID: "a", Shards: 8, KVBytes: 1, WorkspaceBytes: 1}}}
	two := one
	two.Name = "two"
	two.Ranks = []RankPlacement{{Rank: 0, PoolID: "a", Shards: 2, KVBytes: 1, WorkspaceBytes: 1}, {Rank: 1, PoolID: "b", Shards: 6, KVBytes: 1, WorkspaceBytes: 1}}
	in := DeploymentComparisonInput{Pools: []MemoryPool{{ID: "a", UsableBytes: 16}, {ID: "b", UsableBytes: 32}}, Placements: []Placement{one, two}}
	r, err := ComparePlacements(in)
	if err != nil {
		t.Fatal(err)
	}
	if r[0].CapacityStatus != StatusRefused || r[0].Eligible || r[1].CapacityStatus != StatusFits || !r[1].Eligible {
		t.Fatalf("reports=%+v", r)
	}
	if r[1].Ranks[0].RequiredBytes != 16 || r[1].Ranks[1].RequiredBytes != 28 {
		t.Fatalf("rank charges=%+v", r[1].Ranks)
	}
	equal := two
	equal.Ranks = []RankPlacement{{Rank: 0, PoolID: "a", Shards: 4, KVBytes: 1, WorkspaceBytes: 1}, {Rank: 1, PoolID: "b", Shards: 4, KVBytes: 1, WorkspaceBytes: 1}}
	equalReport, err := EvaluatePlacement(DeploymentInput{Pools: in.Pools, Placement: equal})
	if err != nil || equalReport.CapacityStatus != StatusRefused {
		t.Fatalf("equal split report=%+v error=%v", equalReport, err)
	}
	in.Placements[1].Weights.ShardedBytes++
	if _, err := ComparePlacements(in); !errors.Is(err, ErrInvalidPlacement) {
		t.Fatalf("error=%v", err)
	}
}

func deploymentFixture() DeploymentInput {
	return DeploymentInput{Pools: []MemoryPool{{ID: "shared", UsableBytes: 20}}, Placement: Placement{Name: "declared", Support: SupportSupported, SupportEvidence: "operator declaration", Weights: WeightPartition{ReplicatedBytes: 4, ShardedBytes: 9, ShardCount: 2}, Ranks: []RankPlacement{{Rank: 0, PoolID: "shared", Shards: 1}, {Rank: 1, PoolID: "shared", Shards: 1}}}}
}

// fak-test:runtime fast est=1s
// fak-test:justify why=Pin per-rank and shared-pool capacity contracts for unchanged model composition. when=changed:internal/fitplan/deployment*
func TestDeploymentPoolAndSupportEvidence(t *testing.T) {
	tests := []struct {
		name             string
		edit             func(*DeploymentInput)
		capacity, status Status
		eligible         bool
	}{
		{"shared pool", func(in *DeploymentInput) { in.Pools[0].UsableBytes = 17 }, StatusRefused, StatusRefused, false},
		{"rounded shard", func(in *DeploymentInput) {}, StatusFits, StatusFits, true},
		{"unknown capacity", func(in *DeploymentInput) { in.Pools[0].UsableBytes = 0 }, StatusUnknown, StatusUnknown, false},
		{"unsupported", func(in *DeploymentInput) { in.Placement.Support = SupportUnsupported }, StatusFits, StatusUnavailable, false},
		{"unknown support", func(in *DeploymentInput) { in.Placement.Support = "" }, StatusFits, StatusUnknown, false},
		{"missing evidence", func(in *DeploymentInput) { in.Placement.SupportEvidence = " " }, StatusFits, StatusUnknown, false},
		{"unused unknown pool", func(in *DeploymentInput) { in.Pools = append(in.Pools, MemoryPool{ID: "unused"}) }, StatusFits, StatusFits, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			in := deploymentFixture()
			tt.edit(&in)
			r, err := EvaluatePlacement(in)
			if err != nil {
				t.Fatal(err)
			}
			if r.CapacityStatus != tt.capacity || r.Status != tt.status || r.Eligible != tt.eligible {
				t.Fatalf("report=%+v", r)
			}
			if r.Pools[0].RequiredBytes != 18 {
				t.Fatalf("pool=%+v", r.Pools)
			}
		})
	}
}

// fak-test:runtime fast est=1s
// fak-test:justify why=Pin per-rank and shared-pool capacity contracts for unchanged model composition. when=changed:internal/fitplan/deployment*
func TestDeploymentRejectsInvalidAccounting(t *testing.T) {
	tests := map[string]func(*DeploymentInput){
		"duplicate pool":       func(in *DeploymentInput) { in.Pools = append(in.Pools, in.Pools[0]) },
		"negative capacity":    func(in *DeploymentInput) { in.Pools[0].UsableBytes = -1 },
		"missing pool":         func(in *DeploymentInput) { in.Placement.Ranks[0].PoolID = "missing" },
		"duplicate rank":       func(in *DeploymentInput) { in.Placement.Ranks[1].Rank = 0 },
		"negative live demand": func(in *DeploymentInput) { in.Placement.Ranks[0].KVBytes = -1 },
		"missing weights":      func(in *DeploymentInput) { in.Placement.Weights = WeightPartition{ShardCount: 2} },
		"missing shard":        func(in *DeploymentInput) { in.Placement.Ranks[1].Shards = 0 },
		"extra shard":          func(in *DeploymentInput) { in.Placement.Ranks[1].Shards = 2 },
		"rank overflow":        func(in *DeploymentInput) { in.Placement.Ranks[0].ExtraBytes = math.MaxInt64 },
		"pool overflow":        func(in *DeploymentInput) { in.Placement.Weights.ReplicatedBytes = math.MaxInt64 / 2 },
	}
	for name, edit := range tests {
		t.Run(name, func(t *testing.T) {
			in := deploymentFixture()
			edit(&in)
			if _, err := EvaluatePlacement(in); !errors.Is(err, ErrInvalidPlacement) {
				t.Fatalf("error=%v", err)
			}
		})
	}
}

// fak-test:runtime fast est=1s
// fak-test:justify why=Pin per-rank and shared-pool capacity contracts for unchanged model composition. when=changed:internal/fitplan/deployment*
func TestDeploymentLargeShardArithmeticAndJSON(t *testing.T) {
	in := DeploymentInput{Pools: []MemoryPool{{ID: "gpu", UsableBytes: math.MaxInt64}}, Placement: Placement{Name: "large", Support: SupportSupported, SupportEvidence: "declared", Weights: WeightPartition{ShardedBytes: math.MaxInt64, ShardCount: math.MaxInt32}, Ranks: []RankPlacement{{Rank: 0, PoolID: "gpu", Shards: math.MaxInt32}}}}
	r, err := EvaluatePlacement(in)
	if err != nil {
		t.Fatal(err)
	}
	if r.Ranks[0].RequiredBytes != math.MaxInt64 {
		t.Fatalf("report=%+v", r)
	}
	b, err := json.Marshal(in)
	if err != nil {
		t.Fatal(err)
	}
	var decoded DeploymentInput
	if err := json.Unmarshal(b, &decoded); err != nil {
		t.Fatal(err)
	}
	if _, err := EvaluatePlacement(decoded); err != nil {
		t.Fatal(err)
	}
	if got := ceilFraction(math.MaxInt32-1, math.MaxInt32-1, math.MaxInt32); got != math.MaxInt32-1 {
		t.Fatalf("ceil=%d", got)
	}
}
