package model

// v41SharedExpertForStep reuses the qualified resident scalar shared expert
// inside one lane's already-owned continuation. It retains all four checked
// publication kernels and observes their ordered records at one final read.
// No state or tensor survives the call. A decline uses the original callbacks,
// and a selected error is returned without replay or another lane's retirement.
func (m *Model) v41SharedExpertForStep(layer int, input []float32, cfg Config, scratch *v41ProjScratch) ([]float32, error) {
	publish := scratch.gatherOps.bf16PublishFunc()
	if scratch.gatherOps != nil && scratch.denseProjection != nil && scratch.sharedActivation != nil && publish != nil {
		s := scratch.gatherOps.owner
		if s != nil && s.M == m && s.Backend != nil && !s.halClosed && s.halFailure == nil &&
			s.v41Forward != nil && s.v41Forward.callbackOwner == s && s.v41Forward.gatherOps == scratch.gatherOps &&
			cfg.HiddenSize == m.Cfg.HiddenSize && cfg.MoEIntermediateSize == m.Cfg.MoEIntermediateSize && cfg.SwigluLimit == m.Cfg.SwigluLimit {
			scope := s.backendExecutionScope
			full, err := v41ForwardGeometry(cfg)
			if err == nil && full && scope != nil && scope.live && scope.owner != nil {
				owner := backendExecutionOwner(s.Backend)
				if owner == nil || owner == scope.owner {
					output, selected, cause := s.v41DeferredSharedExpertChain(layer)(input)
					if selected || cause != nil {
						return output, cause
					}
				}
			}
		}
	}
	return m.v41SharedExpertSwiGLUWithPublish(layer, input, cfg, scratch.denseProjection, scratch.sharedActivation, publish)
}
