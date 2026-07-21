package model

import (
	"errors"
	"fmt"
	"sort"
)

// active_rank_membership.go — pure-Go active-rank membership mask for fault-tolerant MoE
// expert dispatch/combine (#5291, epic #5289 mooncake-study). Clean-room borrow of
// Mooncake's fault-tolerant expert parallelism, which threads one mutable active_ranks
// membership vector through BOTH dispatch and combine: a dead/absent expert-owner rank is
// masked out of the all-to-all dispatch (it receives no traffic) and its contribution is
// skipped in the combine reduction, so the batch emits a partial-but-valid output over the
// live subset instead of hanging or failing closed on the whole collective
// (upstream Mooncake EP kernel: combine-skip near line 708, dispatch-timeout flip near 417).
//
// This file lands ONLY the deterministic, CPU-witnessable policy — the membership mask plus
// the mask-in-dispatch and skip-in-combine contract. No timeout/detection or network wiring
// lives here: the caller supplies which ranks are alive. The default all-alive mask is exactly
// today's behavior (every rank contributes), so this composes with v4_expert_collective.go's
// AllReduceSum path (the mask names WHICH ranks reduce) and with expert_balance.go's
// LoadBalancedExpertBands (which names which rank OWNS an expert) without touching either.

// ErrActiveRankMembership is returned when a membership mask or a masked combine is refused
// (malformed world size, out-of-range rank, empty live subset, or a width mismatch AMONG the
// live ranks). It is distinct from ErrV4ExpertPlacement: a masked-out rank is NOT an error —
// it is the whole point — so route-around never surfaces the placement refusal the un-masked
// collective raises for a rank/width hole.
var ErrActiveRankMembership = errors.New("model: active-rank membership refused")

// ActiveRankMask records which ranks of an EP group of WorldSize ranks are alive. A rank
// masked out (dead/absent) is skipped in dispatch (it receives no traffic) and contributes
// zero to the combine reduction. The zero value is not usable; build one with AllActive or
// NewActiveRankMask.
type ActiveRankMask struct {
	active []bool
}

// AllActive returns the default mask over a world of worldSize ranks in which every rank is
// alive — identical to the pre-mask behavior where all ranks contribute. It fails closed on a
// non-positive world size (an EP group has at least one rank).
func AllActive(worldSize int) (ActiveRankMask, error) {
	if worldSize <= 0 {
		return ActiveRankMask{}, fmt.Errorf("%w: world size %d, want > 0", ErrActiveRankMembership, worldSize)
	}
	active := make([]bool, worldSize)
	for r := range active {
		active[r] = true
	}
	return ActiveRankMask{active: active}, nil
}

// NewActiveRankMask builds a mask over worldSize ranks in which ONLY the ranks listed in
// alive are active; every rank absent from that membership set is masked out. This is the
// route-around ledger: alive={0,1,3} over a world of 4 leaves rank 2 dead. It fails closed on
// a non-positive world size, an out-of-range member, or a membership set with no live rank
// (an empty live subset has nothing to combine). Duplicate members are idempotent.
func NewActiveRankMask(worldSize int, alive []int) (ActiveRankMask, error) {
	if worldSize <= 0 {
		return ActiveRankMask{}, fmt.Errorf("%w: world size %d, want > 0", ErrActiveRankMembership, worldSize)
	}
	active := make([]bool, worldSize)
	live := 0
	for _, r := range alive {
		if r < 0 || r >= worldSize {
			return ActiveRankMask{}, fmt.Errorf("%w: member rank %d outside [0,%d)", ErrActiveRankMembership, r, worldSize)
		}
		if !active[r] {
			active[r] = true
			live++
		}
	}
	if live == 0 {
		return ActiveRankMask{}, fmt.Errorf("%w: membership set has no live rank over world %d", ErrActiveRankMembership, worldSize)
	}
	return ActiveRankMask{active: active}, nil
}

// WorldSize is the number of ranks the mask spans (alive plus dead).
func (m ActiveRankMask) WorldSize() int { return len(m.active) }

// Active reports whether rank r is alive. An out-of-range rank is treated as dead (false):
// there is no traffic for a rank that does not exist, so a lookup never panics.
func (m ActiveRankMask) Active(r int) bool {
	return r >= 0 && r < len(m.active) && m.active[r]
}

// ActiveCount is the number of live ranks — the size of the subset the combine reduces over.
func (m ActiveRankMask) ActiveCount() int {
	n := 0
	for _, a := range m.active {
		if a {
			n++
		}
	}
	return n
}

// ActiveRanks returns the live ranks in ascending order.
func (m ActiveRankMask) ActiveRanks() []int {
	out := make([]int, 0, len(m.active))
	for r, a := range m.active {
		if a {
			out = append(out, r)
		}
	}
	return out
}

// DeadRanks returns the masked-out ranks in ascending order — the holes route-around skips.
func (m ActiveRankMask) DeadRanks() []int {
	out := make([]int, 0, len(m.active))
	for r, a := range m.active {
		if !a {
			out = append(out, r)
		}
	}
	return out
}

// MaskDispatch splits a per-rank dispatch map (as V4ExpertPlacement.Dispatch produces) into the
// traffic that survives the mask and the experts orphaned by a dead owner rank. A dispatch bound
// for a live rank is kept verbatim; a dispatch bound for a masked-out rank is DROPPED fail-closed
// and its expert recorded in orphaned (ascending, deduped). This is the dispatch-side route-around:
// no traffic is ever emitted to a dead rank, and the orphaned experts' contribution is simply
// absent from the downstream combine (partial-but-valid), never a hung all-to-all.
//
// It fails closed if a dispatch key names a rank outside the mask's world (a dispatch built for a
// different world size cannot be silently mis-routed).
func MaskDispatch(dispatch map[int][]V4ExpertDispatch, mask ActiveRankMask) (map[int][]V4ExpertDispatch, []int, error) {
	live := make(map[int][]V4ExpertDispatch, len(dispatch))
	orphanSet := make(map[int]struct{})
	for rank, work := range dispatch {
		if rank < 0 || rank >= mask.WorldSize() {
			return nil, nil, fmt.Errorf("%w: dispatch rank %d outside [0,%d)", ErrActiveRankMembership, rank, mask.WorldSize())
		}
		if mask.Active(rank) {
			live[rank] = work
			continue
		}
		for _, item := range work {
			orphanSet[item.Expert] = struct{}{}
		}
	}
	orphaned := make([]int, 0, len(orphanSet))
	for e := range orphanSet {
		orphaned = append(orphaned, e)
	}
	sort.Ints(orphaned)
	return live, orphaned, nil
}

// CombineActivePartials sums the per-rank partial vectors of only the ACTIVE ranks, producing a
// partial-but-valid output over the live subset. partials is indexed by rank: partials[r] is rank
// r's contribution. A masked-out rank's partial is ignored entirely whether present, empty, or nil
// (it contributes zero and is never read for width) — that is the skip-in-reduction contract from
// combine-skip. A live rank with an empty/nil partial owns no picks this round and contributes a
// zero vector.
//
// The common width is taken from the first live rank that carries a non-empty partial; every OTHER
// live non-empty partial must match it, else the combine fails closed (ErrActiveRankMembership) — a
// width hole among the live ranks is a real inconsistency, unlike a dead rank which is expected.
// It also fails closed if len(partials) != mask.WorldSize() (partials built for a different world),
// or if no live rank carries any partial (empty live subset — nothing to reduce).
//
// It returns the summed vector and the count of live ranks that actually contributed (non-empty).
func CombineActivePartials(partials [][]float32, mask ActiveRankMask) ([]float32, int, error) {
	if len(partials) != mask.WorldSize() {
		return nil, 0, fmt.Errorf("%w: %d partials, want world size %d", ErrActiveRankMembership, len(partials), mask.WorldSize())
	}
	width := -1
	for r, p := range partials {
		if !mask.Active(r) || len(p) == 0 {
			continue
		}
		if width < 0 {
			width = len(p)
		} else if len(p) != width {
			return nil, 0, fmt.Errorf("%w: live rank %d partial width %d, want %d", ErrActiveRankMembership, r, len(p), width)
		}
	}
	if width <= 0 {
		return nil, 0, fmt.Errorf("%w: no live rank carries a partial (empty live subset)", ErrActiveRankMembership)
	}
	out := make([]float32, width)
	contributed := 0
	for r, p := range partials {
		if !mask.Active(r) || len(p) == 0 {
			continue
		}
		for i, v := range p {
			out[i] += v
		}
		contributed++
	}
	return out, contributed, nil
}
