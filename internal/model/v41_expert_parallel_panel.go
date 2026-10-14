package model

import "fmt"

const (
	v41EPOwnerPanelTokens   = 32
	v41EPOwnerPanelPicks    = v41EPOwnerPanelTokens * V41RouterTopK
	v41EPOwnerPanelMaxBytes = 16 << 20
)

// v41EPOwnerPanelRows contracts one bounded prefill window, expert-major on
// this owner. Each returned token contains only this rank's H-wide published
// contributions, in original slot order, including duplicate expert picks.
// The rows own their storage; no scratch, backend or cohort lease is borrowed.
// Empty owners still return one explicit empty row per token.
//
// This helper neither sums nor communicates. Its caller must bind each token's
// original position/identity to its normal exact collective, and abort or send
// the failure header on error. No partial window is returned. Grouping changes
// execution order, never a pick's coefficient or BF16 publication placement.
// The caller supplies this model/rank's already checked EP owner binding and
// runs under its Session execution ownership; this helper does not admit EP.
func (m *Model) v41EPOwnerPanelRows(layer int, owners TPPlan, rank int, picks [][]routePick, inputs [][]float32, scratch *v41ProjScratch, st *v41ForwardState) (out [][]float32, err error) {
	defer func() {
		if p := recover(); p != nil {
			if cause, ok := p.(error); ok {
				err = cause
			} else {
				err = fmt.Errorf("expert panic: %v", p)
			}
		}
		if err != nil {
			out = nil
			err = fmt.Errorf("%w: owner %d layer %d panel: %w", ErrV41ExpertParallel, rank, layer, err)
		}
	}()
	if m == nil || scratch == nil || st == nil || m.Cfg.HiddenSize <= 0 || m.Cfg.MoEIntermediateSize <= 0 || layer < 0 || layer >= m.Cfg.NumLayers {
		return nil, fmt.Errorf("invalid model, layer or execution state")
	}
	if err := owners.Validate(); err != nil || owners.Dim != m.Cfg.NumExperts || rank < 0 || rank >= len(owners.Shards) {
		return nil, fmt.Errorf("invalid expert ownership")
	}
	if len(picks) > v41EPOwnerPanelTokens || len(inputs) != len(picks) || st.callbackOwner != nil && st.callbackOwner.M != m {
		return nil, fmt.Errorf("invalid token window or callback owner")
	}
	H, band := m.Cfg.HiddenSize, owners.Shards[rank]
	total, owned := 0, 0
	for _, token := range picks {
		if len(token) > v41EPOwnerPanelPicks-total {
			return nil, fmt.Errorf("routed row bound exceeded")
		}
		total += len(token)
		for _, pick := range token {
			if pick.expert < 0 || pick.expert >= owners.Dim || !finite32(pick.weight) {
				return nil, fmt.Errorf("invalid routed pick")
			}
			if pick.expert >= band.Lo && pick.expert < band.Hi {
				owned++
			}
		}
	}
	cells, fits := checkedMulInt(owned, H)
	bytes, sized := checkedMulInt(cells, 4)
	if !fits || !sized || bytes > v41EPOwnerPanelMaxBytes {
		return nil, fmt.Errorf("owner output byte bound exceeded")
	}
	// Validate the whole window before the first weight admission or contraction.
	// Non-BF16 finite inputs are legal: cohort admission declines them and the
	// unchanged scalar expert publishes its input before doing any arithmetic.
	for _, input := range inputs {
		if len(input) != H {
			return nil, fmt.Errorf("invalid hidden width")
		}
		for _, value := range input {
			if !finite32(value) {
				return nil, fmt.Errorf("nonfinite expert input")
			}
		}
	}
	flat := make([]float32, cells)
	out = make([][]float32, len(picks))
	offsets := make([][]int, len(picks))
	offset := 0
	for token, perToken := range picks {
		start := offset
		offsets[token] = make([]int, len(perToken))
		for slot, pick := range perToken {
			offsets[token][slot] = -1
			if pick.expert >= band.Lo && pick.expert < band.Hi {
				offsets[token][slot] = offset
				offset += H
			}
		}
		out[token] = flat[start:offset:offset]
	}
	publish := func(row v41ExpertRow, values []float32) error {
		if len(values) != H {
			return errV41ExpertResult
		}
		for _, value := range values {
			if !finite32(value) {
				return errV41ExpertResult
			}
		}
		at := offsets[row.Token][row.Slot]
		copy(flat[at:at+H], values)
		return nil
	}
	for _, group := range v41PlanExpertGroups(picks) {
		if group.Expert < band.Lo || group.Expert >= band.Hi {
			continue
		}
		stem := "ffn.experts." + itoa(group.Expert)
		selected := false
		if st.callbackOwner != nil && len(group.Rows) > 1 {
			for first := 0; first < len(group.Rows); {
				last := min(first+v41ExpertCohortRows, len(group.Rows))
				if len(group.Rows)-last == 1 {
					last-- // Keep a two-row tail rather than a selected singleton.
				}
				cohortInputs := make([][]float32, last-first)
				weights := make([]float32, last-first)
				for i, row := range group.Rows[first:last] {
					cohortInputs[i] = inputs[row.Token]
					weights[i] = picks[row.Token][row.Slot].weight
				}
				rows, outcome, cause := st.callbackOwner.v41RoutedExpertCohort(layer, stem, cohortInputs, weights)
				if outcome == v41RoutedExpertDeclined && cause == nil && !selected {
					break
				}
				if cause != nil {
					return nil, cause
				}
				if outcome != v41RoutedExpertHandled || len(rows) != len(cohortInputs) {
					return nil, errV41ExpertResult
				}
				for i, row := range group.Rows[first:last] {
					if err := publish(row, rows[i]); err != nil {
						return nil, err
					}
					m.v41NoteExpertContraction()
				}
				selected, first = true, last
			}
		}
		if selected {
			continue
		}
		// Only a preselection decline reaches scalar execution. Reuse one lazy
		// host triple per expert, without growing the layer cache or device ring.
		// The bound routed callback still forbids host fallback for device owners.
		var gate, up, down []float32
		ready := false
		for _, row := range group.Rows {
			values, cause := m.v41FullRoutedExpertPublished(layer, stem, inputs[row.Token], picks[row.Token][row.Slot].weight, m.Cfg, st.routedExpert, st.expertGateUp, st.expertDown,
				func() ([]float32, []float32, []float32, error) {
					if !ready {
						var cause error
						gate, up, down, cause = m.v41ExpertTripleInto(layer, stem, scratch, false)
						if cause != nil {
							return nil, nil, nil, cause
						}
						ready = true
					}
					return gate, up, down, nil
				}, func() ([]float32, error) { return m.hostExpertDown(layer, stem, scratch) }, false, st.gatherOps.bf16PublishFunc())
			if cause != nil {
				return nil, cause
			}
			if err := publish(row, values); err != nil {
				return nil, err
			}
		}
	}
	return out, nil
}
