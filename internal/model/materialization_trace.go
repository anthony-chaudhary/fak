package model

import "fmt"

// MaterializationStep is one load/prefill/decode/verify/cache representation edge.
type MaterializationStep struct {
	Stage      string `json:"stage"`
	From       string `json:"from"`
	To         string `json:"to"`
	Bytes      int64  `json:"bytes"`
	HostStaged bool   `json:"host_staged"`
	Required   bool   `json:"required"`
}

type MaterializationReceipt struct {
	Schema                  string  `json:"schema"`
	Engine                  string  `json:"engine"`
	AcceptedTokens          int     `json:"accepted_tokens"`
	InputSteps              int     `json:"input_steps"`
	KeptSteps               int     `json:"kept_steps"`
	RemovedSteps            int     `json:"removed_steps"`
	InputBytes              int64   `json:"input_bytes"`
	RemovedBytes            int64   `json:"removed_bytes"`
	HostStagingBytesRemoved int64   `json:"host_staging_bytes_removed"`
	BytesRemovedPerAccepted float64 `json:"bytes_removed_per_accepted_token"`
	QualityConstraint       string  `json:"quality_constraint"`
	Rollback                string  `json:"rollback"`
}

// EliminateRedundantMaterialization removes identity conversions and adjacent
// A→B→A round trips only when neither edge is required. It never invents a new
// representation edge, so the kept graph remains explicit and reversible.
func EliminateRedundantMaterialization(steps []MaterializationStep, acceptedTokens int) ([]MaterializationStep, MaterializationReceipt, error) {
	if acceptedTokens < 0 {
		return nil, MaterializationReceipt{}, fmt.Errorf("model: negative accepted tokens")
	}
	r := MaterializationReceipt{Schema: "fak-materialization-receipt/1", Engine: "fak-native", AcceptedTokens: acceptedTokens, InputSteps: len(steps), QualityConstraint: "identical tensor values, shape, dtype, layout, and accepted tokens", Rollback: "retain explicit conversion graph"}
	kept := make([]MaterializationStep, 0, len(steps))
	remove := func(s MaterializationStep) {
		r.RemovedSteps++
		r.RemovedBytes += s.Bytes
		if s.HostStaged {
			r.HostStagingBytesRemoved += s.Bytes
		}
	}
	for _, s := range steps {
		if s.Stage == "" || s.From == "" || s.To == "" || s.Bytes < 0 {
			return nil, MaterializationReceipt{}, fmt.Errorf("model: invalid materialization step")
		}
		r.InputBytes += s.Bytes
		if !s.Required && s.From == s.To {
			remove(s)
			continue
		}
		if !s.Required && len(kept) > 0 {
			prev := kept[len(kept)-1]
			if !prev.Required && prev.From == s.To && prev.To == s.From {
				kept = kept[:len(kept)-1]
				remove(prev)
				remove(s)
				continue
			}
		}
		kept = append(kept, s)
	}
	r.KeptSteps = len(kept)
	if acceptedTokens > 0 {
		r.BytesRemovedPerAccepted = float64(r.RemovedBytes) / float64(acceptedTokens)
	}
	return kept, r, nil
}
