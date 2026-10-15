package model

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math/bits"
)

// Full-forward history already contains every row. Incremental Step keeps its
// current token out of history until the head commits, so it must use the
// separate routedLayer binding rather than infer this callback's position.
type v41RoutedPrefillFunc func(layer int, picks [][]routePick, inputs [][]float32, scratch *v41ProjScratch) ([][]float32, error)

type v41EPNextIdentity func(layer, position int, prefill bool, picks []routePick, input []float32) [32]byte

type v41EPPrefillResultStorage struct {
	backingBytes, rowHeaderBytes, totalBytes uint64
}

// Check cumulative retained outputs before any selected work. A scalar reply
// retains at most H+10 words (the root's result is smaller); a panel retains
// N*(H+10)+11. There can be at most floor(T/2) panels. The outer T slice
// descriptors are additional storage, not another copy of the result backing.
// This is checked accounting, not a caller memory budget or context limit.
func v41EPPrefillResultStorageBound(tokens, width int) (v41EPPrefillResultStorage, error) {
	var storage v41EPPrefillResultStorage
	maxBytes := uint64(^uint(0) >> 1)
	if tokens <= 0 || width <= 0 || uint64(width) > maxBytes/4-v41EPHeaderWords {
		return storage, fmt.Errorf("%w: invalid cumulative Prefill result geometry", ErrV41ExpertParallel)
	}
	T, rowBytes := uint64(tokens), 4*(uint64(width)+v41EPHeaderWords)
	if T > maxBytes/rowBytes {
		return storage, fmt.Errorf("%w: cumulative Prefill result backing overflows", ErrV41ExpertParallel)
	}
	backing := T * rowBytes
	if T/2 > (maxBytes-backing)/(4*v41EPPanelHeaderWords) {
		return storage, fmt.Errorf("%w: cumulative Prefill panel headers overflow", ErrV41ExpertParallel)
	}
	backing += (T / 2) * (4 * v41EPPanelHeaderWords)
	const rowHeaderBytes = 3 * (bits.UintSize / 8)
	if T > (maxBytes-backing)/rowHeaderBytes {
		return storage, fmt.Errorf("%w: cumulative Prefill row descriptors overflow", ErrV41ExpertParallel)
	}
	storage.backingBytes, storage.rowHeaderBytes = backing, T*rowHeaderBytes
	storage.totalBytes = storage.backingBytes + storage.rowHeaderBytes
	return storage, nil
}

// v41EPPrefillGatherAdmitted preserves v41EPGatherPlan admission without building
// a discarded counts/shard table for every prefill token. A validated ownership
// plan covers each legal expert exactly once, so the final gather dimension is
// world*header + picks*width regardless of skew or duplicate picks. Positive
// headers and widths make its partial sums no larger than that final bound.
func v41EPPrefillGatherAdmitted(picks []routePick, owners TPPlan, width int) error {
	if err := owners.Validate(); err != nil || width <= 0 || width > v41EPMaxWords-v41EPHeaderWords || len(owners.Shards) > v41EPMaxWords/v41EPHeaderWords {
		return fmt.Errorf("%w: invalid ownership or row width", ErrV41ExpertParallel)
	}
	for _, pick := range picks {
		if pick.expert < 0 || pick.expert >= owners.Dim || !finite32(pick.weight) {
			return fmt.Errorf("%w: invalid routed pick", ErrV41ExpertParallel)
		}
	}
	remaining := v41EPMaxWords - len(owners.Shards)*v41EPHeaderWords
	if len(picks) > remaining/width {
		return fmt.Errorf("%w: gather dimensions overflow", ErrV41ExpertParallel)
	}
	return nil
}

// Shared deterministic window selection, used by execution and storage
// witnesses. Callers preflight per-token shapes/routes before entering here.
// Window admission matches v41EPPanelPlan without its discarded request table.
func v41EPPrefillWindowEnd(owners TPPlan, picks [][]routePick, width, first int) int {
	limit := min(v41EPOwnerPanelPicks, v41EPOwnerPanelMaxBytes/4/width)
	last, count := first, 0
	for last < len(picks) && last-first < v41EPOwnerPanelTokens && len(picks[last]) <= limit-count {
		count += len(picks[last])
		last++
	}
	for last-first > 1 {
		if _, cause := v41EPPanelAdmitted(owners, picks[first:last], width); cause == nil {
			break
		}
		last--
	}
	return last
}

// The common round header still carries an opaque digest. Distinct execution
// domains refuse mixed scalar/full-forward owners even when a repeated token
// happens to have identical routes and inputs at the same ordinal.
func v41EPPositionIdentity(digest [32]byte, position int, prefill bool) [32]byte {
	var identity [41]byte
	copy(identity[:32], digest[:])
	binary.LittleEndian.PutUint64(identity[32:40], uint64(position))
	if prefill {
		identity[40] = 1
	}
	return sha256.Sum256(identity[:])
}

// Each bounded window finishes all local expert work before its exact panel
// exchange. Every original per-token contribution and identity is retained,
// never rank-presummed. Output
// is published to the layer only after every window succeeds; a later failure
// invalidates the group, but cannot undo rounds a peer has already received.
func (m *Model) v41EPPrefillSum(c *v41ExpertParallelCollective, owners TPPlan, layer int, picks [][]routePick, inputs [][]float32, scratch *v41ProjScratch, st *v41ForwardState, next v41EPNextIdentity) (out [][]float32, err error) {
	defer func() {
		if p := recover(); p != nil {
			if c.failure == nil {
				c.failure = fmt.Errorf("%w: prefill panic: %v", ErrV41ExpertParallel, p)
			}
			c.g.abortV41EP()
			panic(p)
		}
		if err != nil {
			out = nil
			if c.failure == nil {
				c.failure = err
			}
			err = c.failure
			c.g.abortV41EP()
		}
	}()
	if c.failure != nil {
		return nil, c.failure
	}
	H := m.Cfg.HiddenSize
	if H <= 0 || st == nil || scratch == nil || next == nil || len(picks) == 0 || len(inputs) != len(picks) || len(st.history) != len(picks) {
		return nil, fmt.Errorf("%w: invalid full-forward history or panel", ErrV41ExpertParallel)
	}
	if _, cause := v41EPPrefillResultStorageBound(len(picks), H); cause != nil {
		return nil, cause
	}
	// All ranks choose the same boundary from total picks and common geometry.
	// Using this rank's owned count here could desynchronize skewed/empty owners.
	limit := min(v41EPOwnerPanelPicks, v41EPOwnerPanelMaxBytes/4/H)
	for token, row := range picks {
		if len(row) > limit || len(inputs[token]) != H {
			return nil, fmt.Errorf("%w: token %d exceeds owner panel bounds", ErrV41ExpertParallel, token)
		}
		if cause := v41EPPrefillGatherAdmitted(row, owners, H); cause != nil {
			return nil, cause
		}
		for _, value := range inputs[token] {
			if !finite32(value) {
				return nil, fmt.Errorf("%w: token %d has nonfinite input", ErrV41ExpertParallel, token)
			}
		}
	}
	// A singleton cold forward preserves the original slot-major contraction
	// order, including duplicate picks. It consumes the full-forward identity
	// domain but does not select or claim any multi-token batching.
	if len(picks) == 1 {
		digest := next(layer, 0, true, picks[0], inputs[0])
		row, cause := m.v41EPOwnerSum(c, owners, digest, layer, picks[0], inputs[0], scratch, st)
		if cause != nil {
			return nil, cause
		}
		return [][]float32{row}, nil
	}
	out = make([][]float32, len(picks))
	for first := 0; first < len(picks); {
		last := v41EPPrefillWindowEnd(owners, picks, H, first)
		// Per-token preflight guarantees progress before any selected work.
		if last == first {
			return nil, fmt.Errorf("%w: empty owner panel window", ErrV41ExpertParallel)
		}
		// Shorten solely from common geometry before any selected expert work.
		// A one-row tail retains the scalar protocol; a selected error below
		// is terminal and can never retry this admission or replay scalar work.
		rows, cause := m.v41EPOwnerPanelRows(layer, owners, c.g.Rank(), picks[first:last], inputs[first:last], scratch, st)
		if cause != nil {
			return nil, cause
		}
		digests := make([][32]byte, last-first)
		for token := first; token < last; token++ {
			digests[token-first] = next(layer, token, true, picks[token], inputs[token])
		}
		if last-first > 1 {
			reduced, cause := c.reducePanel(rows, v41EPPanelReduction{owners: owners, layer: layer, width: H, picks: picks[first:last], digests: digests, sum: st.routedSumFunc()})
			if cause != nil {
				return nil, cause
			}
			copy(out[first:last], reduced)
		} else {
			token, digest := first, digests[0]
			part := make([]float32, v41EPHeaderWords+len(rows[token-first]))
			v41EPHeader(part, digest, false)
			copy(part[v41EPHeaderWords:], rows[token-first])
			out[token], cause = c.reduce(part, v41EPReduction{owners: owners, digest: digest, layer: layer, picks: picks[token], width: H, sum: st.routedSumFunc()})
			if cause != nil {
				return nil, cause
			}
		}
		first = last
	}
	return out, nil
}
