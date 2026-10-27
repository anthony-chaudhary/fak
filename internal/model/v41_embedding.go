package model

import "fmt"

func (m *Model) v41EmbeddingPanel(ids []int) ([]float32, error) {
	return m.v41EmbeddingPanelOn(ids, nil)
}

// v41EmbeddingPanelOn gathers through the session's device seam when ops binds
// one (Ref #13668). That seam binds only a unit embedding scale, so its rows
// are already the scaled panel; a selected failure is never retried on the host.
func (m *Model) v41EmbeddingPanelOn(ids []int, ops *v41DeviceGatherOps) ([]float32, error) {
	if gather := ops.embeddingGatherFunc(); gather != nil {
		panel, err := gather(ids)
		if err == nil {
			if want, ok := checkedMulInt(len(ids), m.Cfg.HiddenSize); !ok || len(panel) != want {
				err = errV41EmbeddingGatherResult
			}
		}
		if err != nil {
			return nil, v41StageErr(v41StageEmbedding, -1, err)
		}
		return panel, nil
	}
	cfg := m.Cfg
	H := cfg.HiddenSize
	count, ok := checkedMulInt(len(ids), H)
	if !ok || H <= 0 {
		return nil, v41StageErr(v41StageEmbedding, -1, errV41ProjectionResult)
	}
	var panel []float32
	if m.has("model.embed_tokens.weight") {
		embed := m.tensor("model.embed_tokens.weight")
		expected, valid := checkedMulInt(cfg.VocabSize, H)
		if !valid || len(embed) < expected {
			return nil, v41StageErr(v41StageEmbedding, -1, fmt.Errorf("%w: invalid embedding table length", ErrV41ForwardStage))
		}
		panel = make([]float32, count)
		for t, id := range ids {
			copy(panel[t*H:(t+1)*H], embed[id*H:(id+1)*H])
		}
	} else if m.Q2KEmbedding != nil {
		var err error
		panel, err = m.Q2KEmbedding.GatherRows(ids, 1)
		if err != nil {
			return nil, v41StageErr(v41StageEmbedding, -1, err)
		}
		if len(panel) != count {
			return nil, v41StageErr(v41StageEmbedding, -1, errV41ProjectionResult)
		}
	} else {
		return nil, v41StageErr(v41StageEmbedding, -1, fmt.Errorf("%w: missing model.embed_tokens.weight", ErrV41ForwardStage))
	}
	for t := range ids {
		scaleEmbedInPlace(panel[t*H:(t+1)*H], cfg)
	}
	if len(ids) > 0 && ops != nil {
		ops.owner.observeHostArm("embedding-panel")
	}
	return panel, nil
}
