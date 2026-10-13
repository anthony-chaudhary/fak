package model

import "slices"

// A selected owned cohort already fixes the model, layer, backend and execution
// scope. Only consecutive rows with the exact same borrowed tensor bindings can
// share a token tile; equal expert numbers or equal shapes alone are insufficient.
// Splitting never changes the flattened row order or a lane's routed-sum order.
func (o *v41ExpertCohortOptions) tileEnd(start, count int) int {
	if o == nil {
		return count // Cold prefill supplies one common triple.
	}
	first := o.triples[start]
	for end := start + 1; end < count; end++ {
		for i, weight := range o.triples[end] {
			prior := first[i]
			if weight.Buf() == nil || weight.Buf() != prior.Buf() || weight.Backend() != prior.Backend() ||
				weight.Dtype != prior.Dtype || weight.Layout != prior.Layout || weight.Quant != prior.Quant || !slices.Equal(weight.Shape, prior.Shape) {
				return end
			}
		}
	}
	return count
}
