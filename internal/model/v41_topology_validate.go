package model

import (
	"errors"
	"fmt"
)

var errV41TopologyIndexCrossesKV = errors.New("index source segment crosses a kv source boundary")

// v41ValidateSharedTopology refuses a shared-attention schedule in which a
// compressed layer reuses an index selection computed over a different KV
// stream than the one it attends: kvOwner(indexSource(l)) must equal kvOwner(l)
// (DeepInfer cross_layer.py _validate_map_refines). fak's index keys are owned
// by the KV sources, so the index-topk ⊆ index-K ⊆ KV chain is one check.
// Resolution reuses the forward's planner; ratio admission stays with it.
// Adapted from DeepInfer at 774ce244; see third_party/deepinfer-v41-admission/NOTICE.
func v41ValidateSharedTopology(c Config, d41 *DeepSeekV41Config) error {
	if d41 == nil || len(d41.KVSourceLayerIDs) == 0 {
		return nil
	}
	cfg := c
	cfg.DeepSeekV41 = d41
	roles := v41AttentionRoles(cfg)
	kvOwner := func(p V41AttentionPlan) int {
		if p.KVSourceLayer < 0 {
			return p.Layer
		}
		return p.KVSourceLayer
	}
	for l := 0; l < cfg.NumLayers; l++ {
		if v41CompressRatioAt(cfg, l) == 0 {
			continue
		}
		plan, err := v41AttentionPlanFor(cfg, l, roles)
		if err != nil || plan.IndexSourceLayer < 0 || plan.IndexSourceLayer == l {
			continue
		}
		src, err := v41AttentionPlanFor(cfg, plan.IndexSourceLayer, roles)
		if err != nil {
			continue
		}
		if own, srcOwn := kvOwner(plan), kvOwner(src); srcOwn != own {
			return fmt.Errorf("%w: %w: layer %d reads index source %d, whose kv stream is layer %d, but attends layer %d's",
				ErrV41ConfigAdmission, errV41TopologyIndexCrossesKV, l, plan.IndexSourceLayer, srcOwn, own)
		}
	}
	return nil
}
