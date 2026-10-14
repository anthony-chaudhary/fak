package model

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
)

// Full-forward history already contains every row. Incremental Step keeps its
// current token out of history until the head commits, so it must use the
// separate routedLayer binding rather than infer this callback's position.
type v41RoutedPrefillFunc func(layer int, picks [][]routePick, inputs [][]float32, scratch *v41ProjScratch) ([][]float32, error)

type v41EPNextIdentity func(layer, position int, prefill bool, picks []routePick, input []float32) [32]byte

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

// v41EPPrefillSum changes only owner-local contraction scheduling. Each bounded
// window finishes all local expert work before its first exact token reduction.
// Original contributions are sent independently, never rank-presummed. Output
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
	// All ranks choose the same boundary from total picks and common geometry.
	// Using this rank's owned count here could desynchronize skewed/empty owners.
	limit := min(v41EPOwnerPanelPicks, v41EPOwnerPanelMaxBytes/4/H)
	for token, row := range picks {
		if len(row) > limit || len(inputs[token]) != H {
			return nil, fmt.Errorf("%w: token %d exceeds owner panel bounds", ErrV41ExpertParallel, token)
		}
		if _, cause := v41EPGatherPlan(row, owners, H); cause != nil {
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
		last, count := first, 0
		for last < len(picks) && last-first < v41EPOwnerPanelTokens && len(picks[last]) <= limit-count {
			count += len(picks[last])
			last++
		}
		// Per-token preflight guarantees progress before any selected work.
		if last == first {
			return nil, fmt.Errorf("%w: empty owner panel window", ErrV41ExpertParallel)
		}
		rows, cause := m.v41EPOwnerPanelRows(layer, owners, c.g.Rank(), picks[first:last], inputs[first:last], scratch, st)
		if cause != nil {
			return nil, cause
		}
		for token := first; token < last; token++ {
			digest := next(layer, token, true, picks[token], inputs[token])
			part := make([]float32, v41EPHeaderWords+len(rows[token-first]))
			v41EPHeader(part, digest, false)
			copy(part[v41EPHeaderWords:], rows[token-first])
			out[token], cause = c.reduce(part, v41EPReduction{owners: owners, digest: digest, layer: layer, picks: picks[token], width: H, sum: st.routedSum})
			if cause != nil {
				return nil, cause
			}
		}
		first = last
	}
	return out, nil
}
