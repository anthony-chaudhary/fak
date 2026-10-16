package model

import "github.com/anthony-chaudhary/fak/internal/compute"

// Down tiling is explicitly opt-in through the native capability. Each tile
// records its weighted, BF16-published activation rows before issuing any down.
// Thus a later activation recording failure can precede an unissued earlier
// scalar down. Only actually issued statuses participate in fault selection;
// no hypothetical verdict, prefix repair, or selected replay is permitted.
type v41ExpertDownTile struct {
	backend compute.V41Q3KTokenPanelBackend
	copies  compute.V41RowLogBackend
	weight  compute.Tensor
	note    func()
}

func newV41ExpertDownTile(be compute.Backend, weight compute.Tensor, count int) *v41ExpertDownTile {
	panel, havePanel := be.(compute.V41Q3KTokenPanelBackend)
	copies, haveCopies := be.(compute.V41RowLogBackend)
	if !havePanel || !haveCopies || count < 2 || weight.Dtype != compute.Q3_K || !copies.V41RowLogAvailable() {
		return nil
	}
	var checked uint
	for remaining := count; remaining >= 2; {
		rows := v41ExpertTileSize(remaining)
		if checked&uint(rows) == 0 {
			if !panel.SupportsV41Q3KTokenPanel(weight, rows) {
				return nil
			}
			checked |= uint(rows)
		}
		remaining -= rows
	}
	return &v41ExpertDownTile{backend: panel, copies: copies, weight: weight}
}

// The activations stay borrowed from the cohort. Every fresh packed/panel/row
// tensor transfers to its existing owner, including partial error results.
func (p *v41ExpertDownTile) project(activations []compute.Tensor, inter, hidden int, own func(compute.Tensor) bool, valid func(compute.Tensor, int) bool) (compute.Tensor, error) {
	spans := make([]compute.V41RowSpan, len(activations))
	for row, activation := range activations {
		activation.Shape = []int{1, inter}
		spans[row] = compute.V41RowSpan{Src: activation, Rows: 1}
	}
	x, err := p.copies.V41RowLogGather(spans, inter)
	fresh := own(x)
	if err != nil {
		return compute.Tensor{}, err
	}
	if !fresh || len(x.Shape) != 2 || x.Shape[0] != len(activations) || x.Shape[1] != inter {
		return compute.Tensor{}, errV41ExpertResult
	}
	x.Shape = []int{len(activations) * inter}
	if !valid(x, x.Shape[0]) {
		return compute.Tensor{}, errV41ExpertResult
	}
	return p.projectPacked(x, len(activations), inter, hidden, own, valid)
}

// The packed activation has already transferred to the cohort owner. It is
// borrowed here and must not be registered or released a second time.
func (p *v41ExpertDownTile) projectPacked(x compute.Tensor, rows, inter, hidden int, own func(compute.Tensor) bool, valid func(compute.Tensor, int) bool) (compute.Tensor, error) {
	if !valid(x, rows*inter) {
		return compute.Tensor{}, errV41ExpertResult
	}
	if p.note != nil {
		p.note()
	}
	y, err := p.backend.V41Q3KTokenPanel(p.weight, x, rows)
	fresh := own(y)
	if err != nil {
		return compute.Tensor{}, err
	}
	if !fresh || !valid(y, rows*hidden) {
		return compute.Tensor{}, errV41ExpertResult
	}
	y.Shape = []int{rows, hidden}
	return y, nil
}

func (p *v41ExpertDownTile) row(panel compute.Tensor, row, hidden int, own func(compute.Tensor) bool, valid func(compute.Tensor, int) bool) (compute.Tensor, error) {
	x, err := p.copies.V41RowLogGather([]compute.V41RowSpan{{Src: panel, Row: row, Rows: 1}}, hidden)
	fresh := own(x)
	if err != nil {
		return compute.Tensor{}, err
	}
	if !fresh || len(x.Shape) != 2 || x.Shape[0] != 1 || x.Shape[1] != hidden {
		return compute.Tensor{}, errV41ExpertResult
	}
	x.Shape = []int{hidden}
	if !valid(x, hidden) {
		return compute.Tensor{}, errV41ExpertResult
	}
	return x, nil
}
