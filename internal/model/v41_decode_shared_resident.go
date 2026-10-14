package model

import (
	"errors"
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// Only the lane's already-owned, matching callback binding may retain shared
// intermediates. A decline happens before any resident work is selected.
func (m *Model) v41SharedExpertStepOwner(cfg Config, scratch *v41ProjScratch) *Session {
	publish := scratch.gatherOps.bf16PublishFunc()
	if scratch.gatherOps != nil && scratch.denseProjection() != nil && scratch.sharedActivation != nil && publish != nil {
		s := scratch.gatherOps.owner
		if s != nil && s.M == m && s.Backend != nil && !s.halClosed && s.halFailure == nil &&
			s.v41Forward != nil && s.v41Forward.callbackOwner == s && s.v41Forward.gatherOps == scratch.gatherOps &&
			cfg.HiddenSize == m.Cfg.HiddenSize && cfg.MoEIntermediateSize == m.Cfg.MoEIntermediateSize && cfg.SwigluLimit == m.Cfg.SwigluLimit {
			scope := s.backendExecutionScope
			full, err := v41ForwardGeometry(cfg)
			if err == nil && full && scope != nil && scope.live && scope.owner != nil {
				owner := backendExecutionOwner(s.Backend)
				if owner == nil || owner == scope.owner {
					return s
				}
			}
		}
	}
	return nil
}

// v41SharedExpertForStep retains the shared-only control: four existing
// publication kernels and their ordered records at one final row read.
func (m *Model) v41SharedExpertForStep(layer int, input []float32, cfg Config, scratch *v41ProjScratch) ([]float32, error) {
	if s := m.v41SharedExpertStepOwner(cfg, scratch); s != nil {
		output, selected, cause := s.v41DeferredSharedExpertChain(layer)(input)
		if selected || cause != nil {
			return output, cause
		}
	}
	return m.v41SharedExpertSwiGLUWithPublish(layer, input, cfg, scratch.denseProjection(), scratch.sharedActivation, scratch.gatherOps.bf16PublishFunc())
}

// v41SharedExpertAddForStep consumes the live shared output before its cleanup,
// retaining the original routed-then-shared F32 addition. Only the sum crosses
// the host boundary. Neither a tensor nor an owner scope escapes this call.
func (m *Model) v41SharedExpertAddForStep(layer int, input, routed []float32, cfg Config, routeCfg v41RouterConfig, scratch *v41ProjScratch) ([]float32, error) {
	return m.v41SharedExpertAddThenForStep(layer, input, routed, cfg, routeCfg, scratch, nil, nil)
}

// afterChecked borrows the finite-validated, unrounded sum only when the caller
// belongs to the same owned session. A declined chain retains the host control.
func (m *Model) v41SharedExpertAddThenForStep(layer int, input, routed []float32, cfg Config, routeCfg v41RouterConfig, scratch *v41ProjScratch, owner *Session, afterChecked func(compute.Tensor) error) ([]float32, error) {
	if s := m.v41SharedExpertStepOwner(cfg, scratch); s != nil && scratch.sharedAdd != nil {
		consume := func(shared compute.Tensor, own func(compute.Tensor) bool, stage *string) (compute.Tensor, error) {
			if routeCfg.SharedCount < 1 {
				return compute.Tensor{}, &v4RouteError{Field: "v41_shared", Reason: fmt.Sprintf("want >=1, got %d", routeCfg.SharedCount)}
			}
			if len(routed) == 0 {
				return compute.Tensor{}, &v4RouteError{Field: "shared_add", Reason: "empty operand"}
			}
			if len(routed) != cfg.HiddenSize {
				return compute.Tensor{}, &v4RouteError{Field: "shared_add", Reason: fmt.Sprintf("width %d does not match %d", len(routed), cfg.HiddenSize)}
			}
			for i, v := range routed {
				if !finite32(v) {
					return compute.Tensor{}, &v4RouteError{Field: "routed", Reason: fmt.Sprintf("non-finite value at %d", i)}
				}
			}
			*stage = "routed upload"
			acc := s.uploadHostF32([]int{len(routed)}, append([]float32(nil), routed...), compute.MemoryActivation, "V4.1 shared-add accumulator")
			if !own(acc) || acc.Dtype != compute.F32 || acc.Layout != compute.RowMajor || acc.Quant != nil || len(acc.Shape) != 1 || acc.Shape[0] != len(routed) {
				return compute.Tensor{}, errV41DeviceSumResult
			}
			*stage = "shared-add"
			s.Backend.AddInPlace(acc, shared)
			return acc, nil
		}
		if owner != s {
			afterChecked = nil
		}
		output, selected, cause := s.v41SharedExpertChain(layer, true, consume, afterChecked)(input)
		if selected || cause != nil {
			var operation *BackendForwardOperationError
			if errors.As(cause, &operation) && operation.Path == "v41-shared-add" {
				cause = v41StageErr(v41StageMoE, layer, cause)
			}
			return output, cause
		}
	}
	shared, err := m.v41SharedExpertForStep(layer, input, cfg, scratch)
	if err != nil {
		return nil, err
	}
	output, err := v41SharedExpertAddWith(layer, routed, shared, routeCfg, scratch.sharedAdd)
	if err != nil {
		return nil, v41StageErr(v41StageMoE, layer, err)
	}
	return output, nil
}
