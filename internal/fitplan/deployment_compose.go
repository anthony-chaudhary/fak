package fitplan

import (
	"errors"
	"fmt"
	"strings"
)

var (
	ErrPlacementCapacityUnknown      = errors.New("fitplan: placement capacity unknown")
	ErrPlacementInsufficientCapacity = errors.New("fitplan: insufficient placement capacity")
)

// ComposePlacement assigns uniform accounting shards in input rank order. It
// preserves fixed costs and support declarations, and promises no speed optimum
// or executable tensor split. Every rank must have zero shards on input.
func ComposePlacement(in DeploymentInput) (DeploymentInput, error) {
	out := in
	out.Pools = append([]MemoryPool(nil), in.Pools...)
	out.Placement.Ranks = append([]RankPlacement(nil), in.Placement.Ranks...)
	p := &out.Placement
	invalid := func(detail string) (DeploymentInput, error) {
		return DeploymentInput{}, fmt.Errorf("%w: %s", ErrInvalidPlacement, detail)
	}
	if strings.TrimSpace(p.Name) == "" || len(p.Ranks) == 0 || len(out.Pools) == 0 {
		return invalid("name, ranks and pools required")
	}
	if p.Support != "" && p.Support != SupportSupported && p.Support != SupportUnsupported && p.Support != SupportUnknown {
		return invalid("invalid support")
	}
	if p.Weights.ReplicatedBytes < 0 || p.Weights.ShardedBytes < 0 || p.Weights.ShardCount <= 0 || (p.Weights.ReplicatedBytes == 0 && p.Weights.ShardedBytes == 0) {
		return invalid("positive weight demand and shard count required")
	}
	remaining := map[string]int64{}
	for _, pool := range out.Pools {
		if strings.TrimSpace(pool.ID) == "" || pool.UsableBytes < 0 {
			return invalid("invalid pool")
		}
		if _, ok := remaining[pool.ID]; ok {
			return invalid("duplicate pool")
		}
		remaining[pool.ID] = pool.UsableBytes
	}
	fixed := map[string]int64{}
	ranks := map[int]bool{}
	unknown := false
	for _, rank := range p.Ranks {
		capacity, ok := remaining[rank.PoolID]
		if !ok || rank.Rank < 0 || ranks[rank.Rank] || rank.Shards != 0 {
			return invalid("invalid rank, pool or pinned shards")
		}
		ranks[rank.Rank] = true
		charge, ok := sumPlacementBytes(p.Weights.ReplicatedBytes, rank.KVBytes, rank.WorkspaceBytes, rank.ExtraBytes)
		if !ok {
			return invalid("negative demand or rank byte overflow")
		}
		total, ok := sumPlacementBytes(fixed[rank.PoolID], charge)
		if !ok {
			return invalid("pool byte overflow")
		}
		fixed[rank.PoolID] = total
		unknown = unknown || capacity == 0
	}
	if unknown {
		return DeploymentInput{}, ErrPlacementCapacityUnknown
	}
	for pool, charge := range fixed {
		if charge > remaining[pool] {
			return DeploymentInput{}, ErrPlacementInsufficientCapacity
		}
		remaining[pool] -= charge
	}
	unassigned := p.Weights.ShardCount
	for i := range p.Ranks {
		rank := &p.Ranks[i]
		lo, hi := 0, unassigned
		for lo < hi {
			mid := lo + (hi-lo)/2 + 1
			if placementShardBytes(p.Weights, mid) <= remaining[rank.PoolID] {
				lo = mid
			} else {
				hi = mid - 1
			}
		}
		rank.Shards = lo
		remaining[rank.PoolID] -= placementShardBytes(p.Weights, lo)
		unassigned -= lo
	}
	if unassigned != 0 {
		return DeploymentInput{}, ErrPlacementInsufficientCapacity
	}
	r, err := EvaluatePlacement(out)
	if err != nil {
		return DeploymentInput{}, err
	}
	if r.CapacityStatus != StatusFits {
		return DeploymentInput{}, ErrPlacementInsufficientCapacity
	}
	return out, nil
}

func placementShardBytes(weights WeightPartition, shards int) int64 {
	n, d := int64(shards), int64(weights.ShardCount)
	return (weights.ShardedBytes/d)*n + ceilFraction(weights.ShardedBytes%d, n, d)
}
