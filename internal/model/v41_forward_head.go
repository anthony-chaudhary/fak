package model

import "fmt"

// v41Head retains the host final norm and resident LM head for callers without
// a session-owned projection callback.
func (m *Model) v41Head(x []float32) ([]float32, error) {
	return m.v41HeadWithProjection(x, nil)
}

func (m *Model) v41HeadWithProjection(x []float32, project v41DenseProjectionFunc) ([]float32, error) {
	return m.v41HeadWithFinalNorm(x, project, nil)
}

func (m *Model) v41HeadWithFinalNorm(x []float32, project v41DenseProjectionFunc, normalize v41FinalNormFunc) ([]float32, error) {
	return m.v41HeadWithBF16Boundary(x, project, normalize, nil)
}

func (m *Model) v41HeadWithBF16Boundary(x []float32, project v41DenseProjectionFunc, normalize v41FinalNormFunc, round v41BF16BoundaryFunc) ([]float32, error) {
	return m.v41HeadWithLogitScale(x, project, normalize, round, nil, nil)
}

func (m *Model) v41HeadWithLogitScale(x []float32, project v41DenseProjectionFunc, normalize v41FinalNormFunc, round v41BF16BoundaryFunc, scale v41LogitScaleFunc, refuse v41LogitHostRefusal) ([]float32, error) {
	return m.v41HeadWithLogitScaleBound(x, project, &v41FinalNormBinding{normalize: normalize, round: round}, nil, scale, refuse)
}

func (m *Model) v41HeadWithLogitScaleBound(x []float32, project v41DenseProjectionFunc, binding *v41FinalNormBinding, state *v41ForwardState, scale v41LogitScaleFunc, refuse v41LogitHostRefusal) ([]float32, error) {
	return m.v41HeadWithLogitScaleBoundObserved(x, project, binding, state, scale, refuse, nil)
}

func (m *Model) v41HeadWithLogitScaleBoundObserved(x []float32, project v41DenseProjectionFunc, binding *v41FinalNormBinding, state *v41ForwardState, scale v41LogitScaleFunc, refuse v41LogitHostRefusal, observe func(string)) ([]float32, error) {
	xf, err := m.v41PrepareHeadInputBoundObserved(x, binding, state, observe)
	if err != nil {
		return nil, err
	}
	logits, err := m.v41ProjectHeadInput(xf, project, 1)
	if err != nil {
		return nil, err
	}
	return v41ScaleHeadLogitsObserved(logits, m.Cfg, scale, refuse, observe)
}

// v41PrepareHeadInput retains the serial final-normalization and BF16 boundaries.
// A cohort prepares each row separately; normalization never crosses sessions.
func (m *Model) v41PrepareHeadInput(x []float32, normalize v41FinalNormFunc, round v41BF16BoundaryFunc) ([]float32, error) {
	return m.v41PrepareHeadInputBound(x, &v41FinalNormBinding{normalize: normalize, round: round}, nil)
}

func (m *Model) v41PrepareHeadInputBound(x []float32, binding *v41FinalNormBinding, state *v41ForwardState) ([]float32, error) {
	return m.v41PrepareHeadInputBoundObserved(x, binding, state, nil)
}

func (m *Model) v41PrepareHeadInputBoundObserved(x []float32, binding *v41FinalNormBinding, state *v41ForwardState, observe func(string)) ([]float32, error) {
	normalize, round := binding.normFunc(), binding.roundFunc()
	if !m.has("model.norm.weight") {
		return nil, v41StageErr(v41StageFinalNorm, -1,
			fmt.Errorf("%w: missing model.norm.weight", ErrV41ForwardStage))
	}
	if len(x) != m.Cfg.HiddenSize {
		return nil, v41StageErr(v41StageFinalNorm, -1,
			fmt.Errorf("%w: final norm width %d, want %d", ErrV41ForwardStage, len(x), m.Cfg.HiddenSize))
	}
	if !m.hasWeight("lm_head.weight") && !m.hasWeight("model.embed_tokens.weight") {
		return nil, v41StageErr(v41StageHead, -1,
			fmt.Errorf("%w: no lm_head.weight and no tied embedding", ErrV41ForwardStage))
	}
	full, err := v41ForwardGeometry(m.Cfg)
	if err != nil {
		return nil, err
	}
	var xf []float32
	if full {
		if values, selected, cause := binding.try(m, state, x); selected {
			if cause != nil {
				return nil, &V41ProjectionOperationError{Layer: -1, Leaf: "model.norm.weight", Stage: string(v41StageFinalNorm), Cause: v41StageErr(v41StageFinalNorm, -1, cause)}
			}
			return values, nil
		}
		xf, err = m.v41FullFinalNormWithBF16BoundaryObserved(x, normalize, round, observe)
		if err != nil {
			return nil, err
		}
	} else if normalize == nil {
		if observe != nil && len(x) > 0 && !m.Cfg.LayerNorm {
			observe("rms-norm")
		}
		xf = m.finalNorm(x)
	} else {
		var err error
		xf, err = normalize(x)
		if err == nil && len(xf) != m.Cfg.HiddenSize {
			err = errV41ProjectionResult
		}
		if err == nil {
			for _, v := range xf {
				if !finite32(v) {
					err = errV41ProjectionResult
					break
				}
			}
		}
		if err != nil {
			return nil, &V41ProjectionOperationError{Layer: -1, Leaf: "model.norm.weight", Stage: string(v41StageFinalNorm), Cause: v41StageErr(v41StageFinalNorm, -1, err)}
		}
	}
	return xf, nil
}

func (m *Model) v41ProjectHeadInput(xf []float32, project v41DenseProjectionFunc, rows int) ([]float32, error) {
	if project == nil {
		for _, v := range xf {
			if !finite32(v) {
				return nil, v41StageErr(v41StageHead, -1, errV41ProjectionResult)
			}
		}
	}
	return m.v41ProjectionRows(-1, m.headName(), xf, m.Cfg.VocabSize, m.Cfg.HiddenSize, rows, project)
}

// Skipping a projection must not skip its host finite-input refusal. Selected
// normalization and BF16 operations have already checked their own boundaries.
func v41ValidateDiscardedHeadInput(row []float32) error {
	for _, v := range row {
		if !finite32(v) {
			return v41StageErr(v41StageHead, -1, errV41ProjectionResult)
		}
	}
	return nil
}

// v41FullFinalNorm owns the reference's BF16 -> learned F32 RMSNorm -> BF16
// boundary before ParallelHead widens to F32. The callback itself remains F32;
// generic norm flags belong to the reduced compatibility path, not this graph.
func (m *Model) v41FullFinalNorm(input []float32, normalize v41FinalNormFunc) ([]float32, error) {
	return m.v41FullFinalNormWithBF16Boundary(input, normalize, nil)
}

func (m *Model) v41FullFinalNormWithBF16Boundary(input []float32, normalize v41FinalNormFunc, round v41BF16BoundaryFunc) ([]float32, error) {
	return m.v41FullFinalNormWithBF16BoundaryObserved(input, normalize, round, nil)
}

func (m *Model) v41FullFinalNormWithBF16BoundaryObserved(input []float32, normalize v41FinalNormFunc, round v41BF16BoundaryFunc, observe func(string)) ([]float32, error) {
	fail := func(cause error) error {
		return &V41ProjectionOperationError{Layer: -1, Leaf: "model.norm.weight", Stage: string(v41StageFinalNorm), Cause: v41StageErr(v41StageFinalNorm, -1, cause)}
	}
	gain := m.tensor("model.norm.weight")
	eps := float32(m.Cfg.RMSNormEps)
	if len(input) == 0 || len(input) != m.Cfg.HiddenSize || len(gain) != len(input) || !finite32(eps) || eps <= 0 {
		return nil, fail(errV41ProjectionResult)
	}
	for _, value := range gain {
		if !finite32(value) {
			return nil, fail(errV41ProjectionResult)
		}
	}
	staged, err := v41HeadBF16BoundaryObserved("model.norm.weight input", input, round, observe)
	if err != nil {
		if round == nil {
			return nil, fail(errV41ProjectionResult)
		}
		return nil, fail(err)
	}
	var values []float32
	if normalize == nil {
		if observe != nil {
			observe("rms-norm")
		}
		values = rmsnorm(staged, gain, eps)
	} else {
		var err error
		values, err = normalize(staged)
		if err != nil {
			return nil, fail(err)
		}
	}
	if len(values) != len(input) {
		return nil, fail(errV41ProjectionResult)
	}
	out, err := v41HeadBF16BoundaryObserved("model.norm.weight output", values, round, observe)
	if err != nil {
		if round == nil {
			return nil, fail(errV41ProjectionResult)
		}
		return nil, fail(err)
	}
	return out, nil
}

func v41HeadBF16BoundaryObserved(site string, values []float32, round v41BF16BoundaryFunc, observe func(string)) ([]float32, error) {
	out, err := v41BF16BoundaryCopy(-1, site, values, round)
	if observe != nil && round == nil && len(values) > 0 && finite32(values[0]) {
		observe("bf16-boundary")
	}
	return out, err
}
