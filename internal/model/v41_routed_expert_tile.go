package model

import "github.com/anthony-chaudhary/fak/internal/compute"

// The same-expert cold cohort may share Q2_K weight decoding across input rows.
// Each activation/down/publication remains in its original row order. Row-log
// gathers are fresh device copies, not aliases or host readbacks. Native tile
// availability includes its explicit opt-in; no additional default is enabled.
type v41ExpertTile struct {
	backend  compute.V41Q2KTokenPanelBackend
	copies   compute.V41RowLogBackend
	gateW    compute.Tensor
	upW      compute.Tensor
	gate     compute.Tensor
	up       compute.Tensor
	start    int
	end      int
	gridRows int
	note     func()
}

func v41ExpertTileSize(remaining int) int {
	for _, rows := range [...]int{8, 4, 2} {
		if remaining >= rows {
			return rows
		}
	}
	return 0 // A singleton retains the scalar kernel.
}

func newV41ExpertTile(be compute.Backend, gate, up compute.Tensor, count int) *v41ExpertTile {
	tile, haveTile := be.(compute.V41Q2KTokenPanelBackend)
	copies, haveCopies := be.(compute.V41RowLogBackend)
	if !haveTile || !haveCopies || count < 2 || gate.Dtype != compute.Q2_K || up.Dtype != compute.Q2_K || !copies.V41RowLogAvailable() {
		return nil
	}
	// Qualify every needed tile width before any upload or dispatch. Repeated
	// eight-row chunks need only one probe for each weight.
	var checked uint
	for remaining := count; remaining >= 2; {
		rows := v41ExpertTileSize(remaining)
		if checked&uint(rows) == 0 {
			if !tile.SupportsV41Q2KTokenPanel(gate, rows) || !tile.SupportsV41Q2KTokenPanel(up, rows) {
				return nil
			}
			checked |= uint(rows)
		}
		remaining -= rows
	}
	return &v41ExpertTile{backend: tile, copies: copies, gateW: gate, upW: up}
}

// qualifyColdGrid widens a cold cohort after the existing Q3 down schedule
// has qualified. Both exact Q2 weight bindings must admit the complete
// row extent before any upload; legacy modules keep their 8/4/2 schedule.
// Counts 2/4/8 already use one tile and need no additional capability probe.
// This does not change down batching or the ordered deferred-status window.
func (p *v41ExpertTile) qualifyColdGrid(count int) {
	if p != nil && count > 2 && count != 4 && count != 8 && count <= v41ExpertCohortRows && p.backend.SupportsV41Q2KTokenPanel(p.gateW, count) && p.backend.SupportsV41Q2KTokenPanel(p.upW, count) {
		p.gridRows = count
	}
}

// project transfers every fresh allocation to the cohort's existing row owner,
// including outputs returned alongside a selected error. Panel owners survive
// until the cohort's final status read and cleanup. A selected error never
// declines; only a nil helper or singleton uses the historical scalar path.
func (p *v41ExpertTile) project(s *Session, inputs [][]float32, row, hidden, inter int, allowPanelRow bool, chunkRows int, own func(compute.Tensor) bool, valid func(compute.Tensor, int) bool) (gate, up compute.Tensor, handled bool, panelRow compute.V41RoutedSwiGLUPanelRowBackend, panelChunk compute.V41RoutedSwiGLUPanelChunkBackend, err error) {
	if p == nil || (row >= p.end && len(inputs)-row == 1) {
		return compute.Tensor{}, compute.Tensor{}, false, nil, nil, nil
	}
	if row >= p.end {
		rows := v41ExpertTileSize(len(inputs) - row)
		if p.gridRows > 0 && len(inputs)-row == p.gridRows {
			rows = p.gridRows
		}
		packed := make([]float32, rows*hidden)
		for r := 0; r < rows; r++ {
			copy(packed[r*hidden:(r+1)*hidden], inputs[row+r])
		}
		x := s.uploadHostF32([]int{len(packed)}, packed, compute.MemoryActivation, "V4.1 same-expert token tile")
		if !own(x) || !valid(x, len(packed)) {
			return gate, up, true, nil, nil, errV41ExpertResult
		}
		project := func(weight compute.Tensor) (compute.Tensor, error) {
			out, e := p.backend.V41Q2KTokenPanel(weight, x, rows)
			fresh := own(out)
			if e != nil {
				return compute.Tensor{}, e
			}
			if !fresh || !valid(out, rows*inter) {
				return compute.Tensor{}, errV41ExpertResult
			}
			out.Shape = []int{rows, inter}
			return out, nil
		}
		if p.note != nil {
			p.note()
		}
		if p.gate, err = project(p.gateW); err != nil {
			return gate, up, true, nil, nil, err
		}
		if p.up, err = project(p.upW); err != nil {
			return gate, up, true, nil, nil, err
		}
		p.start, p.end = row, row+rows
	}
	// Both panels already belong to the cohort owner. Refusal happens before
	// either row gather or activation is issued. A selected row call remains
	// fatal on error; the caller must never replay it through the scalar path.
	if allowPanelRow && chunkRows >= 2 && row+chunkRows <= p.end {
		if direct, ok := p.backend.(compute.V41RoutedSwiGLUPanelChunkBackend); ok && direct.SupportsV41RoutedSwiGLUPanelChunk(p.gate, p.up, row-p.start, chunkRows) {
			return p.gate, p.up, true, nil, direct, nil
		}
	}
	if allowPanelRow {
		if direct, ok := p.backend.(compute.V41RoutedSwiGLUPanelRowBackend); ok && direct.SupportsV41RoutedSwiGLUPanelRow(p.gate, p.up, row-p.start) {
			return p.gate, p.up, true, direct, nil, nil
		}
	}
	copyRow := func(panel compute.Tensor) (compute.Tensor, error) {
		out, e := p.copies.V41RowLogGather([]compute.V41RowSpan{{Src: panel, Row: row - p.start, Rows: 1}}, inter)
		fresh := own(out)
		if e != nil {
			return compute.Tensor{}, e
		}
		if !fresh || len(out.Shape) != 2 || out.Shape[0] != 1 || out.Shape[1] != inter {
			return compute.Tensor{}, errV41ExpertResult
		}
		out.Shape = []int{inter}
		if !valid(out, inter) {
			return compute.Tensor{}, errV41ExpertResult
		}
		return out, nil
	}
	if gate, err = copyRow(p.gate); err != nil {
		return gate, up, true, nil, nil, err
	}
	up, err = copyRow(p.up)
	return gate, up, true, nil, nil, err
}
