package agent

import (
	"strings"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// requantizedWeightsReceipt is the live resident-vs-file receipt on the serving backend;
// ok is false without a device backend or a recorded file size.
func (p *InKernelPlanner) requantizedWeightsReceipt() (model.WeightResidencyReceipt, bool) {
	if p == nil || p.m == nil || p.backend == nil {
		return model.WeightResidencyReceipt{}, false
	}
	r := p.m.WeightResidency(p.backend)
	return r, r.FileWeightBytes > 0
}

// requantizedWeightsClause is the trailing perf-line clause while the served model holds
// weights resident past model.ResidentBytesRatioAlarm; empty otherwise.
func (p *InKernelPlanner) requantizedWeightsClause() string {
	r, ok := p.requantizedWeightsReceipt()
	if !ok {
		return ""
	}
	return r.PerfClause()
}

func (p *InKernelPlanner) fillRequantizedWeights(out *WeightResidency) {
	r, ok := p.requantizedWeightsReceipt()
	if !ok {
		return
	}
	out.FileWeightBytes = r.FileWeightBytes
	out.ResidentWeightBytes = r.ResidentWeightBytes
	out.ResidentToFileRatio = r.Ratio
	out.RequantizedTypes = strings.Join(r.RequantizedTypes, ",")
	out.RequantizedDegraded = r.Degraded
}
