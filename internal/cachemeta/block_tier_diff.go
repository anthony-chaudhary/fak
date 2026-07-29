package cachemeta

import "sort"

// block_tier_diff.go computes a deterministic, per-block-hash diff event stream
// from two residency snapshots. A prefix-aware KV router keeps a map from a
// block hash to where the block lives (its tier) and how hot it is (its
// priority); between two ticks it wants the DELTA, not a full re-scan — which
// blocks appeared, which were dropped, and which moved tier or changed priority
// — each event keyed by the block hash so the consumer can update exactly the
// affected prefix span. This composes with the Mooncake / LMCache residency
// adapters already in this package: those normalize one live movement into the
// shared entry stream, while this folds two whole snapshots into the ordered
// change stream a router diffs against.
//
// The lowering is a pure, wall-clock-free function of the two input maps: same
// inputs always yield the same ordered slice (ordered by block hash), so a
// replay reproduces the stream exactly. Empty or nil snapshots fail closed to an
// empty stream rather than fabricating events.

// BlockTierState is the per-block-hash residency snapshot the diff folds over:
// which tier a block currently lives on and how hot it is. Priority is the
// router-facing hotness rank a higher value means the router prefers the replica
// that still holds this block on this tier.
type BlockTierState struct {
	Tier     ResidencyTier
	Priority int64
}

// BlockDeltaKind names which kind of change one diff event reports.
type BlockDeltaKind string

const (
	// BlockAdded — a block hash present in the current snapshot but not the
	// previous one. The old tier and priority are zero the new values come from
	// the current snapshot.
	BlockAdded BlockDeltaKind = "added"
	// BlockRemoved — a block hash present in the previous snapshot but gone from
	// the current one. The new tier and priority are zero the old values come
	// from the previous snapshot.
	BlockRemoved BlockDeltaKind = "removed"
	// BlockChanged — a block hash present in both snapshots whose tier or
	// priority (or both) differs. Both old and new values are carried the two
	// moved flags say which axis actually changed.
	BlockChanged BlockDeltaKind = "changed"
)

// BlockDelta is one event in the per-block-hash diff stream. It names the block
// by hash, says whether the block was added, removed, or changed, and carries
// the old and new tier and priority so the consumer never has to hold the prior
// snapshot itself. For an add the old fields are zero for a remove the new
// fields are zero for a change both are populated and the two moved flags mark
// which axis differs.
type BlockDelta struct {
	Hash          string
	Kind          BlockDeltaKind
	OldTier       ResidencyTier
	NewTier       ResidencyTier
	OldPriority   int64
	NewPriority   int64
	TierMoved     bool
	PriorityMoved bool
}

// DiffBlockStates folds a previous and a current per-block-hash snapshot into the
// deterministic diff event stream between them. A block only in the current
// snapshot yields an added event a block only in the previous one yields a
// removed event a block in both whose tier or priority differs yields a changed
// event that flags which axis moved. A block that is byte-identical across the
// two snapshots emits nothing. The result is ordered by block hash so the same
// two inputs always produce the identical ordered slice (reproducible replay),
// and both nil and empty inputs fail closed to an empty (nil) stream.
func DiffBlockStates(prev, cur map[string]BlockTierState) []BlockDelta {
	if len(prev) == 0 && len(cur) == 0 {
		return nil
	}
	// Gather the union of hashes so every event is visited exactly once, then
	// sort for a stable, replay-reproducible order.
	seen := make(map[string]struct{}, len(prev)+len(cur))
	hashes := make([]string, 0, len(prev)+len(cur))
	hashes = appendUnseenHashes(hashes, seen, prev)
	hashes = appendUnseenHashes(hashes, seen, cur)
	sort.Strings(hashes)

	out := make([]BlockDelta, 0, len(hashes))
	for _, h := range hashes {
		before, inPrev := prev[h]
		after, inCur := cur[h]
		switch {
		case inCur && !inPrev:
			out = append(out, BlockDelta{
				Hash:          h,
				Kind:          BlockAdded,
				NewTier:       after.Tier,
				NewPriority:   after.Priority,
				TierMoved:     true,
				PriorityMoved: true,
			})
		case inPrev && !inCur:
			out = append(out, BlockDelta{
				Hash:          h,
				Kind:          BlockRemoved,
				OldTier:       before.Tier,
				OldPriority:   before.Priority,
				TierMoved:     true,
				PriorityMoved: true,
			})
		default:
			tierMoved := before.Tier != after.Tier
			priorityMoved := before.Priority != after.Priority
			if !tierMoved && !priorityMoved {
				continue
			}
			out = append(out, BlockDelta{
				Hash:          h,
				Kind:          BlockChanged,
				OldTier:       before.Tier,
				NewTier:       after.Tier,
				OldPriority:   before.Priority,
				NewPriority:   after.Priority,
				TierMoved:     tierMoved,
				PriorityMoved: priorityMoved,
			})
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

// appendUnseenHashes appends every block hash of m that seen does not already carry,
// marking it seen. DiffBlockStates runs it once over prev and once over cur to gather
// their union so each hash yields exactly one event; the caller sorts afterwards, so
// map iteration order does not reach the result.
func appendUnseenHashes(hashes []string, seen map[string]struct{}, m map[string]BlockTierState) []string {
	for h := range m {
		if _, ok := seen[h]; !ok {
			seen[h] = struct{}{}
			hashes = append(hashes, h)
		}
	}
	return hashes
}
