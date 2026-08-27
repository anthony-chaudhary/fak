package rehydrate

import (
	"context"
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/resume"
)

// CacheProjection binds the re-entry cache rung to the same TTL-aware cost
// projection used by the resume planner. A cold projection refuses the first
// post-wake action with COLD_CACHE and carries the re-warm plan that replaces
// any warm-cache latency or price assumption.
type CacheProjection struct {
	Report resume.Report
}

// NewCacheProjection evaluates the provider-cache posture once at the
// rehydration boundary. Callers retain the report as the cost-model witness.
func NewCacheProjection(in resume.Input) CacheProjection {
	return CacheProjection{Report: resume.Plan(in)}
}

// Rung returns the cache revalidation rung consumed by Gate.
func (p CacheProjection) Rung() Rung {
	return NewRung(ColdCache, func(context.Context) Verdict {
		if p.Report.Posture == resume.PostureCold {
			return Refuse(ColdCache, fmt.Sprintf("COLD_TTL: prefix %s; strategy=%s", p.Report.PostureReason, p.Report.Recommended))
		}
		return Clear()
	})
}
