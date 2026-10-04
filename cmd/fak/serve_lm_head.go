package main

import (
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/agent"
	fakmodel "github.com/anthony-chaudhary/fak/internal/model"
)

// serveLMHeadReadyLine renders the LIVE LM-head route and resident weight total at the point a
// local serve becomes ready (fak#13567). The "resident-layout" load message is formatted before the
// planner's Metal residency promotion runs, so its lm_head can still read cpu-*; this line is
// sampled after promotion so "lm_head=metal-q6k" (or cpu-q8, ...) reflects what decode will use.
// It returns "" when no in-kernel model is loaded (proxy/mock serves), so nothing is emitted.
func serveLMHeadReadyLine(m *fakmodel.Model) string {
	if m == nil {
		return ""
	}
	r := m.ResidentReport()
	if r == nil {
		return ""
	}
	return fmt.Sprintf("lm_head=%s resident_total=%.2fMiB q6k_embed=%.2fMiB tied_embed_f32=%.2fMiB tied_head_q8=%.2fMiB decode=%.2fGiB/tok",
		r.LMHead, lmHeadMiB(r.TotalResidentBytes), lmHeadMiB(r.Q6KEmbedBytes), lmHeadMiB(r.TiedEmbedF32Bytes), lmHeadMiB(r.TiedHeadQ8Bytes), r.DecodeGiBPerToken)
}

func lmHeadMiB(b int64) float64 { return float64(b) / (1 << 20) }

// residentWeightsLiveView is the turnkey /healthz resident_weights block: the load-time
// resident store byte split (the startup ResidentReport, taken before any decode) plus the LIVE
// lm_head route, in the same shape the gateway /healthz reports. Probes never re-walk the store
// maps, which decode may write concurrently. nil (key absent) when no native model is loaded or
// no startup snapshot exists — never a fabricated zero.
func residentWeightsLiveView(m *fakmodel.Model, startup *fakmodel.ResidentReport) *agent.WeightResidency {
	if m == nil || startup == nil || startup.TotalResidentBytes <= 0 {
		return nil
	}
	wr := agent.WeightResidencyFromReport(startup)
	wr.LMHead = m.LMHeadRoute()
	return &wr
}
