package agent

import (
	"sync"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// inKernelWeightResidencyCache memoizes the resident store byte split. The stores are fixed
// once the planner holds a loaded model, so the O(tensor) walk runs once; only the LM-head
// route is re-read per call (a Metal residency promotion can move it after startup).
type inKernelWeightResidencyCache struct {
	once sync.Once
	snap WeightResidency
	ok   bool
}

// WeightResidency implements WeightResidencyReporter (fak#13567). The byte split is computed
// once, primed at planner construction (before any request decodes), so probes never walk the
// store maps while decode may lazily fill q8w or a Metal upload may drop a CPU copy. Only the
// LM-head route is re-read per call; LMHeadRoute takes q8Mu/metalQ4KMu read paths and no
// planner lock, so a /healthz probe can never deadlock against an in-flight decode.
func (p *InKernelPlanner) WeightResidency() (WeightResidency, bool) {
	if p == nil || p.m == nil {
		return WeightResidency{}, false
	}
	c := &p.weightResidency
	c.once.Do(func() {
		r := p.m.ResidentReport()
		if r == nil || r.TotalResidentBytes <= 0 {
			return
		}
		c.snap = WeightResidencyFromReport(r)
		c.ok = true
	})
	if !c.ok {
		return WeightResidency{}, false
	}
	out := c.snap
	out.LMHead = p.m.LMHeadRoute()
	return out, true
}

// WeightResidencyFromReport projects a model.ResidentReport onto the gateway-neutral
// WeightResidency shape (shared by the gateway /healthz and the turnkey `fak up` /healthz).
func WeightResidencyFromReport(r *model.ResidentReport) WeightResidency {
	return WeightResidency{
		TotalResidentBytes:  r.TotalResidentBytes,
		F32Bytes:            r.F32Bytes,
		Q8Bytes:             r.Q8Bytes,
		Q4KBytes:            r.Q4KBytes,
		KQuantBytes:         r.KQuantBytes,
		Q2Bytes:             r.Q2Bytes,
		Q2KEmbedBytes:       r.Q2KEmbedBytes,
		PQ2EmbedBytes:       r.PQ2EmbedBytes,
		Q4KEmbedBytes:       r.Q4KEmbedBytes,
		Q6KEmbedBytes:       r.Q6KEmbedBytes,
		TiedEmbedF32Bytes:   r.TiedEmbedF32Bytes,
		TiedHeadQ8Bytes:     r.TiedHeadQ8Bytes,
		DecodeBytesPerToken: r.DecodeBytesPerToken,
		DecodeGiBPerToken:   r.DecodeGiBPerToken,
		LMHead:              r.LMHead,
	}
}
