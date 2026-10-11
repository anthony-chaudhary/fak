package model

// Under ExecutionPolicyDeviceOnly a projection, activation or expert seam that
// is unbound, or that declines at run time, would hand the op back to the host
// arm. These wrappers turn both into the typed device-only refusal instead
// (errors.Is errV41DeviceOnlyHostOp), retire the session, and never replay on
// the host (Refs #13668).

func (s *Session) v41DeviceOnlyDeclined(layer int, op string) error {
	cause := &v41DeviceOnlyHostOpError{ops: []v41HostOp{{name: op, site: "declined device seam"}}}
	backend := ""
	if s.Backend != nil {
		backend = s.Backend.Name()
	}
	closed := &BackendForwardOperationError{Backend: backend, Forward: ForwardPathKind("deepseek41"), Path: "device-only", Layer: layer, Stage: op + " declined", Cause: cause}
	s.hostFallbackObserved = true
	s.halFailure = closed
	s.halClosed = true
	s.Close()
	return closed
}

func (s *Session) v41BindDeviceOnlyDeclineRefusals(st *v41ForwardState) {
	st.denseProjection = v41DeviceOnlyDense(s, st.denseProjection)
	st.groupedOutput = v41DeviceOnlyGrouped(s, st.groupedOutput)
	st.engramProjection = v41DeviceOnlyEngram(s, st.engramProjection)
	st.mhcProjection = v41DeviceOnlyMHCProjection(s, st.mhcProjection)
	st.mhcFFNProjection = v41DeviceOnlyMHCProjection(s, st.mhcFFNProjection)
	st.sharedActivation = v41DeviceOnlySharedActivation(s, st.sharedActivation)
	st.expertGateUp = v41DeviceOnlyGateUp(s, st.expertGateUp)
	st.expertDown = v41DeviceOnlyDown(s, st.expertDown)
}

func v41DeviceOnlyDense(s *Session, f v41DenseProjectionFunc) v41DenseProjectionFunc {
	return func(layer int, leaf string, panel []float32, out, in, rows int) ([]float32, v41DenseProjectionOutcome, error) {
		if f != nil {
			y, outcome, err := f(layer, leaf, panel, out, in, rows)
			if outcome != v41ProjectionDeclined {
				return y, outcome, err
			}
		}
		return nil, v41ProjectionError, s.v41DeviceOnlyDeclined(layer, "dense-projection")
	}
}

func v41DeviceOnlyGrouped(s *Session, f v41GroupedOutputFunc) v41GroupedOutputFunc {
	return func(layer int, o []float32, heads, headDim, groups, rank, dim int) ([]float32, v41DenseProjectionOutcome, error) {
		if f != nil {
			y, outcome, err := f(layer, o, heads, headDim, groups, rank, dim)
			if outcome != v41ProjectionDeclined {
				return y, outcome, err
			}
		}
		return nil, v41ProjectionError, s.v41DeviceOnlyDeclined(layer, "grouped-output-projection")
	}
}

func v41DeviceOnlyEngram(s *Session, f v41EngramProjectionFunc) v41EngramProjectionFunc {
	return func(layer int, row []float32, out, in int) ([]float32, v41DenseProjectionOutcome, error) {
		if f != nil {
			y, outcome, err := f(layer, row, out, in)
			if outcome != v41ProjectionDeclined {
				return y, outcome, err
			}
		}
		return nil, v41ProjectionError, s.v41DeviceOnlyDeclined(layer, "engram-projection")
	}
}

func v41DeviceOnlyMHCProjection(s *Session, f v41MHCProjectionFunc) v41MHCProjectionFunc {
	return func(layer int, input []float32, H int, eps float32, full, transposed bool) ([]float32, v41DenseProjectionOutcome, error) {
		if f != nil {
			y, outcome, err := f(layer, input, H, eps, full, transposed)
			if outcome != v41ProjectionDeclined {
				return y, outcome, err
			}
		}
		return nil, v41ProjectionError, s.v41DeviceOnlyDeclined(layer, "mhc-projection")
	}
}

func v41DeviceOnlySharedActivation(s *Session, f v41SharedActivationFunc) v41SharedActivationFunc {
	return func(layer int, gate, up []float32, limit float32) ([]float32, v41DenseProjectionOutcome, error) {
		if f != nil {
			y, outcome, err := f(layer, gate, up, limit)
			if outcome != v41ProjectionDeclined {
				return y, outcome, err
			}
		}
		return nil, v41ProjectionError, s.v41DeviceOnlyDeclined(layer, "shared-expert-swiglu")
	}
}

func v41DeviceOnlyDown(s *Session, f v41ExpertDownFunc) v41ExpertDownFunc {
	return func(layer int, stem string, fused []float32) ([]float32, v41ExpertDownOutcome, error) {
		if f != nil {
			y, outcome, err := f(layer, stem, fused)
			if outcome != v41DownDeclined {
				return y, outcome, err
			}
		}
		return nil, v41DownError, s.v41DeviceOnlyDeclined(layer, "routed-expert-down")
	}
}
