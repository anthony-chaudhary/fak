package fitplan

import (
	"errors"
	"math"
	"reflect"
	"testing"
)

// fak-test:runtime fast est=1s
// fak-test:justify why=Verify automatic shard assignment preserves fixed costs and physical pool limits. when=changed:internal/fitplan/deployment*
func TestDeploymentCompose(t *testing.T) {
	in := DeploymentInput{Pools: []MemoryPool{{ID: "small", UsableBytes: 16}, {ID: "large", UsableBytes: 32}}, Placement: Placement{Name: "auto", Support: SupportUnknown, Weights: WeightPartition{ReplicatedBytes: 8, ShardedBytes: 24, ShardCount: 8}, Ranks: []RankPlacement{{Rank: 0, PoolID: "small", KVBytes: 1, WorkspaceBytes: 1}, {Rank: 1, PoolID: "large", KVBytes: 1, WorkspaceBytes: 1}}}}
	out, err := ComposePlacement(in)
	if err != nil {
		t.Fatal(err)
	}
	if out.Placement.Ranks[0].Shards != 2 || out.Placement.Ranks[1].Shards != 6 || in.Placement.Ranks[0].Shards != 0 {
		t.Fatalf("out=%+v input=%+v", out, in)
	}
	r, err := EvaluatePlacement(out)
	if err != nil || r.CapacityStatus != StatusFits || r.Eligible {
		t.Fatalf("report=%+v error=%v", r, err)
	}
	again, err := ComposePlacement(in)
	if err != nil || !reflect.DeepEqual(out, again) {
		t.Fatalf("not deterministic: %+v %v", again, err)
	}
	// Both fixed rank charges consume one pool before the first shard lands.
	in.Pools = []MemoryPool{{ID: "small", UsableBytes: 43}}
	in.Placement.Ranks[1].PoolID = "small"
	before := append([]RankPlacement(nil), in.Placement.Ranks...)
	if _, err := ComposePlacement(in); !errors.Is(err, ErrPlacementInsufficientCapacity) {
		t.Fatalf("shared pool error=%v", err)
	}
	if !reflect.DeepEqual(before, in.Placement.Ranks) {
		t.Fatal("failure mutated input")
	}
	in.Pools[0].UsableBytes = 44
	out, err = ComposePlacement(in)
	if err != nil || out.Placement.Ranks[0].Shards != 8 || out.Placement.Ranks[1].Shards != 0 {
		t.Fatalf("shared result=%+v error=%v", out, err)
	}
}

// fak-test:runtime fast est=1s
// fak-test:justify why=Keep automatic composition refusals typed and overflow safe. when=changed:internal/fitplan/deployment*
func TestDeploymentComposeRefusalsAndLargeBytes(t *testing.T) {
	in := deploymentFixture()
	for i := range in.Placement.Ranks {
		in.Placement.Ranks[i].Shards = 0
	}
	in.Pools[0].UsableBytes = 0
	if _, err := ComposePlacement(in); !errors.Is(err, ErrPlacementCapacityUnknown) {
		t.Fatalf("error=%v", err)
	}
	in.Pools[0].UsableBytes = 20
	in.Placement.Ranks[0].Shards = 1
	if _, err := ComposePlacement(in); !errors.Is(err, ErrInvalidPlacement) {
		t.Fatalf("error=%v", err)
	}
	in.Placement.Ranks[0].Shards = 0
	in.Placement.Ranks[0].KVBytes = -1
	if _, err := ComposePlacement(in); !errors.Is(err, ErrInvalidPlacement) {
		t.Fatalf("error=%v", err)
	}
	in = DeploymentInput{Pools: []MemoryPool{{ID: "gpu", UsableBytes: math.MaxInt64}}, Placement: Placement{Name: "large", Weights: WeightPartition{ShardedBytes: math.MaxInt64, ShardCount: math.MaxInt32}, Ranks: []RankPlacement{{PoolID: "gpu"}}}}
	out, err := ComposePlacement(in)
	if err != nil || out.Placement.Ranks[0].Shards != math.MaxInt32 {
		t.Fatalf("large=%+v error=%v", out, err)
	}
	in.Placement.Weights.ShardCount = math.MaxInt
	out, err = ComposePlacement(in)
	if err != nil || out.Placement.Ranks[0].Shards != math.MaxInt {
		t.Fatalf("max shard count=%+v error=%v", out, err)
	}
}
