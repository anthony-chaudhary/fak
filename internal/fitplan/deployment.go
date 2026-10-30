package fitplan

import (
	"errors"
	"fmt"
	"math"
	"math/bits"
	"strings"
)

var ErrInvalidPlacement = errors.New("fitplan: invalid placement")

type RuntimeSupport string

const (
	SupportSupported   RuntimeSupport = "supported"
	SupportUnsupported RuntimeSupport = "unsupported"
	SupportUnknown     RuntimeSupport = "unknown"
)

// MemoryPool identifies one physical allocation pool. Aliases for unified memory
// must use the same ID; usable bytes already exclude operator headroom.
type MemoryPool struct {
	ID          string `json:"id"`
	UsableBytes int64  `json:"usable_bytes"`
}

// WeightPartition fixes the model encoding across compared placements. Shards
// are uniform accounting units, conservatively rounded upward per rank.
type WeightPartition struct {
	ReplicatedBytes int64 `json:"replicated_bytes"`
	ShardedBytes    int64 `json:"sharded_bytes"`
	ShardCount      int   `json:"shard_count"`
}

type RankPlacement struct {
	Rank           int    `json:"rank"`
	PoolID         string `json:"pool_id"`
	Shards         int    `json:"shards"`
	KVBytes        int64  `json:"kv_bytes"`
	WorkspaceBytes int64  `json:"workspace_bytes"`
	ExtraBytes     int64  `json:"extra_bytes"`
}

type Placement struct {
	Name            string          `json:"name"`
	Support         RuntimeSupport  `json:"support"`
	SupportEvidence string          `json:"support_evidence"`
	Weights         WeightPartition `json:"weights"`
	Ranks           []RankPlacement `json:"ranks"`
}

type DeploymentInput struct {
	Pools     []MemoryPool `json:"pools"`
	Placement Placement    `json:"placement"`
}

type DeploymentComparisonInput struct {
	Pools      []MemoryPool `json:"pools"`
	Placements []Placement  `json:"placements"`
}

type PlacementReason string

const (
	PlacementPoolUnknown        PlacementReason = "pool_capacity_unknown"
	PlacementPoolExceeded       PlacementReason = "pool_capacity_exceeded"
	PlacementRuntimeUnsupported PlacementReason = "runtime_unsupported"
	PlacementRuntimeUnknown     PlacementReason = "runtime_support_unknown"
)

type RankDemand struct {
	Rank            int    `json:"rank"`
	PoolID          string `json:"pool_id"`
	ReplicatedBytes int64  `json:"replicated_bytes"`
	ShardedBytes    int64  `json:"sharded_bytes"`
	KVBytes         int64  `json:"kv_bytes"`
	WorkspaceBytes  int64  `json:"workspace_bytes"`
	ExtraBytes      int64  `json:"extra_bytes"`
	RequiredBytes   int64  `json:"required_bytes"`
}

type PoolDemand struct {
	ID            string `json:"id"`
	UsableBytes   int64  `json:"usable_bytes"`
	RequiredBytes int64  `json:"required_bytes"`
	Status        Status `json:"status"`
}

type PlacementReport struct {
	Name            string            `json:"name"`
	Status          Status            `json:"status"`
	CapacityStatus  Status            `json:"capacity_status"`
	Eligible        bool              `json:"eligible"`
	Support         RuntimeSupport    `json:"support"`
	SupportEvidence string            `json:"support_evidence"`
	Reasons         []PlacementReason `json:"reasons"`
	Ranks           []RankDemand      `json:"ranks"`
	Pools           []PoolDemand      `json:"pools"`
	// NextTiers lists the concrete lower tiers that would admit this
	// placement, best-first, populated only for a capacity refusal over fully
	// known pools; omitted otherwise.
	NextTiers []NextTierStep `json:"next_tiers,omitempty"`
}

// EvaluatePlacement models declared residency, not backend execution. Eligible
// requires capacity and caller-declared support evidence, not qualification.
// KV, workspace and other live allocations must be supplied for each rank.
func EvaluatePlacement(in DeploymentInput) (PlacementReport, error) {
	p := in.Placement
	r := PlacementReport{Name: p.Name, Status: StatusFits, CapacityStatus: StatusFits, Support: p.Support, SupportEvidence: p.SupportEvidence, Reasons: []PlacementReason{}, Ranks: []RankDemand{}, Pools: []PoolDemand{}}
	invalid := func(detail string) (PlacementReport, error) {
		return PlacementReport{}, fmt.Errorf("%w: %s", ErrInvalidPlacement, detail)
	}
	if strings.TrimSpace(p.Name) == "" || len(p.Ranks) == 0 || len(in.Pools) == 0 {
		return invalid("name, ranks and pools required")
	}
	if p.Support != "" && p.Support != SupportSupported && p.Support != SupportUnsupported && p.Support != SupportUnknown {
		return invalid("invalid support")
	}
	if p.Weights.ReplicatedBytes < 0 || p.Weights.ShardedBytes < 0 || p.Weights.ShardCount <= 0 || (p.Weights.ReplicatedBytes == 0 && p.Weights.ShardedBytes == 0) {
		return invalid("positive weight demand and shard count required")
	}
	poolIndex := map[string]int{}
	for _, pool := range in.Pools {
		if strings.TrimSpace(pool.ID) == "" || pool.UsableBytes < 0 {
			return invalid("invalid pool")
		}
		if _, ok := poolIndex[pool.ID]; ok {
			return invalid("duplicate pool")
		}
		poolIndex[pool.ID] = len(r.Pools)
		r.Pools = append(r.Pools, PoolDemand{ID: pool.ID, UsableBytes: pool.UsableBytes, Status: StatusFits})
	}
	seenRanks := map[int]bool{}
	shards := 0
	used := map[int]bool{}
	for _, rank := range p.Ranks {
		idx, ok := poolIndex[rank.PoolID]
		if !ok || rank.Rank < 0 || seenRanks[rank.Rank] || rank.Shards < 0 || rank.Shards > p.Weights.ShardCount-shards {
			return invalid("invalid rank, pool or shard assignment")
		}
		seenRanks[rank.Rank] = true
		shards += rank.Shards
		// Divide first and use quotient/remainder multiplication to avoid
		// overflowing a mathematically representable shard allocation.
		part := placementShardBytes(p.Weights, rank.Shards)
		demand, ok := sumPlacementBytes(p.Weights.ReplicatedBytes, part, rank.KVBytes, rank.WorkspaceBytes, rank.ExtraBytes)
		if !ok {
			return invalid("negative demand or rank byte overflow")
		}
		total, ok := sumPlacementBytes(r.Pools[idx].RequiredBytes, demand)
		if !ok {
			return invalid("pool byte overflow")
		}
		r.Pools[idx].RequiredBytes = total
		used[idx] = true
		r.Ranks = append(r.Ranks, RankDemand{rank.Rank, rank.PoolID, p.Weights.ReplicatedBytes, part, rank.KVBytes, rank.WorkspaceBytes, rank.ExtraBytes, demand})
	}
	if shards != p.Weights.ShardCount {
		return invalid("assigned shards do not cover weight partition")
	}
	for i := range r.Pools {
		pool := &r.Pools[i]
		if !used[i] {
			continue
		}
		if pool.UsableBytes == 0 {
			pool.Status = StatusUnknown
			if r.CapacityStatus != StatusRefused {
				r.CapacityStatus = StatusUnknown
			}
			r.Reasons = appendReason(r.Reasons, PlacementPoolUnknown)
		} else if pool.RequiredBytes > pool.UsableBytes {
			pool.Status = StatusRefused
			r.CapacityStatus = StatusRefused
			r.Reasons = appendReason(r.Reasons, PlacementPoolExceeded)
		}
	}
	r.NextTiers = nextTiersFor(r.CapacityStatus, r.Pools, r.Ranks)
	r.Status = r.CapacityStatus
	switch {
	case p.Support == SupportUnsupported:
		r.Status = StatusUnavailable
		r.Reasons = appendReason(r.Reasons, PlacementRuntimeUnsupported)
	case p.Support != SupportSupported || strings.TrimSpace(p.SupportEvidence) == "":
		if r.Status != StatusRefused {
			r.Status = StatusUnknown
		}
		r.Reasons = appendReason(r.Reasons, PlacementRuntimeUnknown)
	default:
		r.Eligible = r.CapacityStatus == StatusFits
	}
	return r, nil
}

// ComparePlacements retains input order and makes no throughput ranking. Every
// candidate must retain the same model partition; workload residency is declared
// per rank because replication varies with placement.
func ComparePlacements(in DeploymentComparisonInput) ([]PlacementReport, error) {
	if len(in.Placements) == 0 {
		return nil, fmt.Errorf("%w: placements required", ErrInvalidPlacement)
	}
	reports := make([]PlacementReport, 0, len(in.Placements))
	names := map[string]bool{}
	for _, p := range in.Placements {
		if p.Weights != in.Placements[0].Weights || names[p.Name] {
			return nil, fmt.Errorf("%w: comparison requires same weights and unique names", ErrInvalidPlacement)
		}
		names[p.Name] = true
		r, err := EvaluatePlacement(DeploymentInput{Pools: in.Pools, Placement: p})
		if err != nil {
			return nil, err
		}
		reports = append(reports, r)
	}
	return reports, nil
}

func sumPlacementBytes(values ...int64) (int64, bool) {
	var total int64
	for _, v := range values {
		if v < 0 || v > math.MaxInt64-total {
			return 0, false
		}
		total += v
	}
	return total, true
}

func appendReason(reasons []PlacementReason, reason PlacementReason) []PlacementReason {
	for _, r := range reasons {
		if r == reason {
			return reasons
		}
	}
	return append(reasons, reason)
}

func ceilFraction(a, b, d int64) int64 {
	hi, lo := bits.Mul64(uint64(a), uint64(b))
	q, r := bits.Div64(hi, lo, uint64(d))
	if r > 0 {
		q++
	}
	return int64(q)
}
