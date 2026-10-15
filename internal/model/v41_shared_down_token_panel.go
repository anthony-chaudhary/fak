package model

import "github.com/anthony-chaudhary/fak/internal/compute"

// Only the private shared-expert down site calls this helper. Capability
// probes stay at that contraction's original position, after the unchanged
// gate, up and down-input publications. A missing capability declines before
// any selected operation; once selected, no failure can replay the legacy down.
// Grid-capable backends consume the existing full panel directly. Legacy
// modules retain the eight-row gather path below.
func (s *Session) v41SharedDownTokenPanel(expected compute.Backend, weight, input compute.Tensor, rows, out, in int, own func(compute.Tensor) bool) (compute.Tensor, bool, error) {
	const tileRows = 8
	if s != nil && (!backendHasComparableIdentity(expected) || !backendHasComparableIdentity(s.Backend) || s.Backend != expected) {
		return compute.Tensor{}, true, errV41ProjectionResult
	}
	if s == nil || s.M == nil || s.Backend == nil || !backendHasComparableIdentity(s.Backend) ||
		s.halClosed || s.halFailure != nil || !s.modelWeightsHeld || rows != v41ColdSharedPanelRows ||
		weight.Backend() != s.Backend || weight.Buf() == nil || weight.Dtype != compute.Q3_K || weight.Layout != compute.RowMajor ||
		len(weight.Shape) != 2 || weight.Shape[0] != out || weight.Shape[1] != in {
		return compute.Tensor{}, false, nil
	}
	if _, err := compute.PlanV41Q3KTokenPanel(out, in, tileRows); err != nil {
		return compute.Tensor{}, false, nil
	}
	inputCount, inputOK := checkedMulInt(rows, in)
	outputCount, outputOK := checkedMulInt(rows, out)
	if !inputOK || !outputOK || compute.ValidateV41RowLogAlloc(rows, in) != nil || compute.ValidateV41RowLogAlloc(rows, out) != nil ||
		input.Backend() != s.Backend || input.Buf() == nil || input.Dtype != compute.F32 || input.Layout != compute.RowMajor || input.Quant != nil ||
		len(input.Shape) != 1 || input.Shape[0] != inputCount {
		return compute.Tensor{}, false, nil
	}
	backend, model, scope, epoch := s.Backend, s.M, s.backendExecutionScope, s.cacheGeometryEpoch
	if scope == nil || !scope.live || scope.owner == nil {
		return compute.Tensor{}, false, nil
	}
	expectedOwner := scope.owner
	validScope := func() bool {
		if !backendHasComparableIdentity(s.Backend) || s.Backend != backend || s.M != model || s.backendExecutionScope != scope || s.cacheGeometryEpoch != epoch ||
			!scope.live || scope.owner != expectedOwner || s.halClosed || s.halFailure != nil || !s.modelWeightsHeld || model.lora != nil || model.prism != nil {
			return false
		}
		for _, member := range scope.sessions {
			if member == s {
				return true
			}
		}
		return false
	}
	if !validScope() {
		return compute.Tensor{}, false, nil
	}
	panel, havePanel := backend.(compute.V41Q3KTokenPanelBackend)
	recording, haveRecording := backend.(compute.V41StepBatchRecordingBackend)
	provider, haveOwner := backend.(compute.ForwardExecutionOwnerBackend)
	if !havePanel || !haveRecording || !haveOwner {
		return compute.Tensor{}, false, nil
	}
	owner := provider.ForwardExecutionOwner()
	if !validScope() {
		return compute.Tensor{}, true, errV41ProjectionResult
	}
	if owner == nil || owner != scope.owner {
		return compute.Tensor{}, true, errV41ProjectionResult
	}
	validTensor := func(t compute.Tensor, count int) bool {
		return t.Backend() == backend && t.Buf() != nil && t.Dtype == compute.F32 && t.Layout == compute.RowMajor && t.Quant == nil && t.Numel() == count
	}
	// The loaded-module capability, not the planner's range, selects the grid.
	// Probe at the same down-contraction boundary as the legacy tile route.
	available := panel.SupportsV41Q3KTokenPanel(weight, rows)
	if !validScope() {
		return compute.Tensor{}, true, errV41ProjectionResult
	}
	if available {
		active := recording.V41StepBatchRecording()
		if !validScope() {
			return compute.Tensor{}, true, errV41ProjectionResult
		}
		if !active {
			return compute.Tensor{}, false, nil
		}
		y, err := panel.V41Q3KTokenPanel(weight, input, rows)
		fresh := own(y)
		if err != nil {
			return compute.Tensor{}, true, err
		}
		if !fresh || !validScope() || !validTensor(y, outputCount) || len(y.Shape) != 1 || y.Shape[0] != outputCount {
			return compute.Tensor{}, true, errV41ProjectionResult
		}
		y.Shape = []int{rows, out}
		return y, true, nil
	}
	// An old optional module declines the full panel before selection. Keep
	// its previous eight-row tile/copy contract, including probe order.
	copies, haveCopies := backend.(compute.V41RowLogBackend)
	if !haveCopies {
		return compute.Tensor{}, false, nil
	}
	available = copies.V41RowLogAvailable()
	if !validScope() {
		return compute.Tensor{}, true, errV41ProjectionResult
	}
	if !available {
		return compute.Tensor{}, false, nil
	}
	available = panel.SupportsV41Q3KTokenPanel(weight, tileRows)
	if !validScope() {
		return compute.Tensor{}, true, errV41ProjectionResult
	}
	if !available {
		return compute.Tensor{}, false, nil
	}
	// Query last: an open window may have suspended recording. This snapshot
	// is useful only under the same already-held live execution owner.
	active := recording.V41StepBatchRecording()
	if !validScope() {
		return compute.Tensor{}, true, errV41ProjectionResult
	}
	if !active {
		return compute.Tensor{}, false, nil
	}
	input.Shape = []int{rows, in}
	spans := make([]compute.V41RowSpan, 0, rows/tileRows)
	for row := 0; row < rows; row += tileRows {
		x, err := copies.V41RowLogGather([]compute.V41RowSpan{{Src: input, Row: row, Rows: tileRows}}, in)
		fresh := own(x)
		if err != nil {
			return compute.Tensor{}, true, err
		}
		if !fresh || !validScope() || !validTensor(x, tileRows*in) || len(x.Shape) != 2 || x.Shape[0] != tileRows || x.Shape[1] != in {
			return compute.Tensor{}, true, errV41ProjectionResult
		}
		x.Shape = []int{tileRows * in}
		// Use the admitted typed operation: generic BatchedMatMul could
		// re-probe availability and silently choose a different contraction.
		y, err := panel.V41Q3KTokenPanel(weight, x, tileRows)
		fresh = own(y)
		if err != nil {
			return compute.Tensor{}, true, err
		}
		if !fresh || !validScope() || !validTensor(y, tileRows*out) || len(y.Shape) != 1 || y.Shape[0] != tileRows*out {
			return compute.Tensor{}, true, errV41ProjectionResult
		}
		y.Shape = []int{tileRows, out}
		spans = append(spans, compute.V41RowSpan{Src: y, Rows: tileRows})
	}
	joined, err := copies.V41RowLogGather(spans, out)
	fresh := own(joined)
	if err != nil {
		return compute.Tensor{}, true, err
	}
	if !fresh || !validScope() || !validTensor(joined, outputCount) || len(joined.Shape) != 2 || joined.Shape[0] != rows || joined.Shape[1] != out {
		return compute.Tensor{}, true, errV41ProjectionResult
	}
	// Every temporary, including this result, belongs to the outer chunk.
	// Its existing drain precedes cleanup and preserves row/stage fault order.
	return joined, true, nil
}
