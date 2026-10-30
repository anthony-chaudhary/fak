package fitplan

import (
	"fmt"
	"math"
)

// NextTier is the closed vocabulary of lower admission tiers a refused
// placement may step to. Enumeration is best-first: the tier that requires the
// least new declaration comes first.
type NextTier string

const (
	// NextTierHostOffload admits the placement by moving sharded, KV or
	// workspace bytes to a declared host pool.
	NextTierHostOffload NextTier = "host_offload"
	// NextTierMultiDevice admits the placement by declaring additional pool
	// capacity or re-sharding onto larger pools.
	NextTierMultiDevice NextTier = "multi_device"
	// NextTierStreamedDense is reserved for the streamed-dense route
	// (fak#13175). It is an open-route pointer, never an implemented
	// placement, and is not populated until the deployment grammar can
	// declare stream support.
	NextTierStreamedDense NextTier = "streamed_dense"
)

// NextTierStep is one concrete, byte-proven next tier for a refused placement.
// ShortfallBytes is the amount the tier must absorb to admit the placement;
// Note is fixed-wording modeled_only arithmetic, not execution evidence.
type NextTierStep struct {
	Tier           NextTier `json:"tier"`
	ShortfallBytes int64    `json:"shortfall_bytes"`
	Note           string   `json:"note,omitempty"`
}

// nextTiersFor enumerates the lower tiers that would admit a refused
// placement, best-first, from declared-pool arithmetic only. It returns nil
// unless the capacity verdict is a refusal over fully known pools: a used pool
// with unknown capacity proves no tier, and the pool_capacity_unknown reason
// says why. Over known pools multi_device is arithmetically always admissible,
// so a known-pool refusal always names at least one tier. A host pool is a
// declared pool no rank draws from; its whole usable capacity is spare for
// offloaded bytes.
func nextTiersFor(capacity Status, pools []PoolDemand, ranks []RankDemand) []NextTierStep {
	if capacity != StatusRefused {
		return nil
	}
	used := make(map[string]bool, len(ranks))
	for _, r := range ranks {
		used[r.PoolID] = true
	}
	var shortfall, movable, hostSpare int64
	for _, p := range pools {
		if p.UsableBytes == 0 && used[p.ID] {
			return nil
		}
		if p.RequiredBytes > p.UsableBytes {
			shortfall = addSaturating(shortfall, p.RequiredBytes-p.UsableBytes)
		}
		if !used[p.ID] {
			hostSpare = addSaturating(hostSpare, p.UsableBytes)
		}
	}
	for _, r := range ranks {
		movable = addSaturating(movable, r.ShardedBytes)
		movable = addSaturating(movable, r.KVBytes)
		movable = addSaturating(movable, r.WorkspaceBytes)
		movable = addSaturating(movable, r.ExtraBytes)
	}
	var steps []NextTierStep
	if shortfall > 0 && shortfall <= movable && shortfall <= hostSpare {
		steps = append(steps, NextTierStep{
			Tier:           NextTierHostOffload,
			ShortfallBytes: shortfall,
			Note:           fmt.Sprintf("move >= %d bytes of sharded, KV or workspace bytes to a declared host pool", shortfall),
		})
	}
	steps = append(steps, NextTierStep{
		Tier:           NextTierMultiDevice,
		ShortfallBytes: shortfall,
		Note:           fmt.Sprintf("declare >= %d bytes of additional pool capacity or re-shard onto larger pools", shortfall),
	})
	return steps
}

// addSaturating caps at math.MaxInt64 so tier arithmetic stays defined even
// when declared demands approach the int64 ceiling.
func addSaturating(sum, addend int64) int64 {
	if addend > 0 && sum > math.MaxInt64-addend {
		return math.MaxInt64
	}
	return sum + addend
}
