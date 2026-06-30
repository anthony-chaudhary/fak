package vcachescore

import (
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/vcachewarm"
)

const DefaultReadinessSchema = "fak.cache.default_readiness.v1"

// DefaultReadinessReport is the hard gate for enabling cache behavior by
// default. It consumes the richer score report but fails closed on provenance
// collapse, cold-path regressions, unsupported active paths, or a low
// default-usefulness verdict.
type DefaultReadinessReport struct {
	Schema                 string            `json:"schema"`
	OK                     bool              `json:"ok"`
	Verdict                string            `json:"verdict"`
	Reasons                []string          `json:"reasons"`
	DefaultUsefulnessScore int               `json:"default_usefulness_score"`
	DefaultUsefulnessGrade string            `json:"default_usefulness_grade"`
	DefaultUsefulness      string            `json:"default_usefulness"`
	ColdPathCorrect        bool              `json:"cold_path_correct"`
	PlaneProvenance        map[string]string `json:"plane_provenance"`
}

// DefaultReadiness folds a vCache score into the binary default-on readiness
// gate. "Useful by default" is stronger than "2x economics": provider rebates
// can help cost/latency, but they do not prove fak-owned cache activation or
// trust.
func DefaultReadiness(rep Report) DefaultReadinessReport {
	out := DefaultReadinessReport{
		Schema:                 DefaultReadinessSchema,
		Verdict:                "default_ready",
		DefaultUsefulnessScore: rep.DefaultUsefulness.Score,
		DefaultUsefulnessGrade: rep.DefaultUsefulness.Grade,
		DefaultUsefulness:      rep.DefaultUsefulness.Verdict,
		ColdPathCorrect:        rep.ColdPathCorrect && rep.DefaultUsefulness.ColdPathCorrect,
		PlaneProvenance: map[string]string{
			"provider_observed":        rep.Planes.ProviderObserved.Provenance,
			"kernel_witnessed":         rep.Planes.KernelWitnessed.Provenance,
			"context_witnessed":        rep.Planes.ContextWitnessed.Provenance,
			"external_engine_observed": rep.Planes.ExternalEngineObserved.Provenance,
			"forecast":                 rep.Planes.Forecast.Provenance,
		},
	}
	if rep.DefaultUsefulness.Schema != "fak.cache.default_usefulness.v1" {
		out.Reasons = append(out.Reasons, "default_usefulness schema missing or unknown")
	}
	if !out.ColdPathCorrect {
		out.Reasons = append(out.Reasons, "cold-path correctness is not proven independent of cache hits")
	}
	if rep.DefaultUsefulness.Verdict != "default_ready" {
		out.Reasons = append(out.Reasons, fmt.Sprintf("default_usefulness verdict %q is not default_ready", rep.DefaultUsefulness.Verdict))
	}
	checkPlane(&out.Reasons, "provider_observed", rep.Planes.ProviderObserved, "OBSERVED")
	checkPlane(&out.Reasons, "kernel_witnessed", rep.Planes.KernelWitnessed, "WITNESSED")
	checkPlane(&out.Reasons, "context_witnessed", rep.Planes.ContextWitnessed, "WITNESSED")
	checkPlane(&out.Reasons, "external_engine_observed", rep.Planes.ExternalEngineObserved, "OBSERVED")
	if !rep.Planes.Forecast.Available || rep.Planes.Forecast.Provenance != "FORECAST" {
		out.Reasons = append(out.Reasons, "forecast plane must remain a separate FORECAST plane")
	}
	if hasUnsupportedActiveReason(rep.Planes.Forecast) {
		out.Reasons = append(out.Reasons, "forecast plane carries unsupported active-cache reason")
	}
	out.OK = len(out.Reasons) == 0
	if !out.OK {
		out.Verdict = "blocked"
	}
	return out
}

func checkPlane(reasons *[]string, name string, plane PlaneValueReport, availableProvenance string) {
	if hasUnsupportedActiveReason(plane) {
		*reasons = append(*reasons, fmt.Sprintf("%s carries unsupported active-cache reason", name))
	}
	if plane.Available {
		if plane.Provenance != availableProvenance {
			*reasons = append(*reasons, fmt.Sprintf("%s provenance %q must be %s", name, plane.Provenance, availableProvenance))
		}
		return
	}
	if plane.Provenance != "MISSING" {
		*reasons = append(*reasons, fmt.Sprintf("%s unavailable provenance %q must be MISSING", name, plane.Provenance))
	}
}

func hasUnsupportedActiveReason(plane PlaneValueReport) bool {
	reason := strings.ToLower(strings.TrimSpace(plane.Reason))
	return reason == string(vcachewarm.ReasonUnsupportedActiveCacheCapability) ||
		strings.Contains(reason, string(vcachewarm.ReasonUnsupportedActiveCacheCapability))
}
