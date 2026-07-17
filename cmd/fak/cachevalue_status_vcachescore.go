package main

// Optional vcache score digest -- summarizes the provider prompt-cache scorecard.
// Split out of cachevalue_status.go along this concern seam so the cachevalue
// status dispatch surface stays steerable as new digests land (steerability
// dispatch_god_file). Behavior-preserving code motion -- same package, no logic change.

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/vcachescore"
)

func loadCachevalueVCacheScoreStatus(path string) (cachevalueVCacheScoreDigest, []cachevalueStatusRow) {
	digest := cachevalueVCacheScoreDigest{Path: path, Status: "unavailable"}
	b, err := os.ReadFile(path)
	if err != nil {
		digest.Error = err.Error()
		return digest, []cachevalueStatusRow{vcacheScoreUnavailableRow(path, err.Error())}
	}
	var rep vcachescore.Report
	if err := json.Unmarshal(b, &rep); err != nil {
		digest.Error = err.Error()
		return digest, []cachevalueStatusRow{vcacheScoreUnavailableRow(path, "decode: "+err.Error())}
	}
	digest.Status = nonEmpty(rep.Status, vcacheScoreReportStatus(rep))
	digest.Grade = rep.Grade
	digest.Score = rep.Score
	digest.ActiveSource = rep.ActiveSource
	digest.ActiveMultiplier = rep.ActiveMultiplier
	digest.TwoXBetter = rep.TwoXBetter
	digest.DefaultUsefulness = rep.DefaultUsefulness.Verdict
	digest.AgenticActivation = rep.AgenticActivation.Active
	digest.ProviderObserved = vcacheScorePlaneLabel(rep.Planes.ProviderObserved)
	digest.KernelWitnessed = vcacheScorePlaneLabel(rep.Planes.KernelWitnessed)
	digest.ContextWitnessed = vcacheScorePlaneLabel(rep.Planes.ContextWitnessed)
	digest.ExternalEngineObserved = vcacheScorePlaneLabel(rep.Planes.ExternalEngineObserved)
	digest.Forecast = vcacheScorePlaneLabel(rep.Planes.Forecast)
	return digest, rowsFromVCacheScoreReport(rep, path)
}

func vcacheScoreUnavailableRow(path, reason string) cachevalueStatusRow {
	return cachevalueStatusRow{
		Plane:         "vcache_score",
		Component:     "vcache_score_report",
		Owner:         "fak",
		Dependency:    "local_json_report",
		Fidelity:      "diagnostic",
		Evidence:      "MISSING",
		Status:        "unavailable",
		FailureDomain: "vcache_score_artifact",
		SessionImpact: "vCache score evidence could not be folded, so provider/fak/external/forecast attribution is incomplete",
		Reason:        fmt.Sprintf("%s: %s", path, reason),
		NextAction:    "fak vcache score --json",
	}
}

func rowsFromVCacheScoreReport(rep vcachescore.Report, path string) []cachevalueStatusRow {
	rows := []cachevalueStatusRow{{
		Plane:         "vcache_score",
		Component:     "vcache_score_report",
		Owner:         "fak",
		Dependency:    "local_json_report",
		Fidelity:      "diagnostic",
		Evidence:      vcacheScoreReportEvidence(rep),
		Status:        vcacheScoreReportStatus(rep),
		FailureDomain: "vcache_score",
		SessionImpact: "vCache score artifact separates provider rebates, fak-owned KV/context value, external engine cache, and forecasted value",
		Reason:        vcacheScoreReportReason(rep, path),
		NextAction:    vcacheScoreReportNextAction(rep),
	}}
	rows = append(rows,
		rowFromVCacheScorePlane("provider_prompt_cache", "vcache_score_provider_observed", "provider", "provider_usage_snapshot", "lossless", "provider_telemetry", rep.Planes.ProviderObserved, "provider-reported cache economics; missing here points at telemetry/snapshot gaps, not fak-native KV", "fak vcache observe --transcript FILE --json"),
		rowFromVCacheScorePlane("kernel_tool_cache", "vcache_score_kernel_witnessed", "fak", "cachevalue_ledger", "lossless", "fak", rep.Planes.KernelWitnessed, "pure-fak KV reuse witness; missing here means no fak-owned reuse value was supplied to the scorecard", "fak vcache score --kernel-ledger default --json"),
		rowFromVCacheScorePlane("managed_context", "vcache_score_context_witnessed", "fak", "vcache_context_snapshot", "lossy", "fak_context_planner", rep.Planes.ContextWitnessed, "O(1) context/drop witness; separates lossy context shrink from provider cache hit/miss behavior", "fak vcache context-witness --json"),
		rowFromVCacheScorePlane("external_engine_cache", "vcache_score_external_engine_observed", "external", "external_engine_snapshot", "passive", "external_engine", rep.Planes.ExternalEngineObserved, "external serving-engine prefix-cache evidence; failures here belong to the driver or sidecar, not fak-native KV", "fak vcache score --external-engine-events N --external-engine-hit-rate F --json"),
		rowFromVCacheScorePlane("forecast", "vcache_score_forecast", "fak", "score_model", "forecast", "forecast", rep.Planes.Forecast, "deterministic score forecast; useful for planning but not a realized provider/fak/external witness", "fak vcache score --telemetry FILE --json"),
	)
	return rows
}

func rowFromVCacheScorePlane(plane, component, owner, dependency, fidelity, failureDomain string, p vcachescore.PlaneValueReport, impact, missingAction string) cachevalueStatusRow {
	status := vcacheScorePlaneStatus(p)
	next := ""
	if status == "missing" {
		next = missingAction
	}
	return cachevalueStatusRow{
		Plane:         plane,
		Component:     component,
		Owner:         owner,
		Dependency:    dependency,
		Fidelity:      fidelity,
		Evidence:      vcacheScorePlaneEvidence(p),
		Status:        status,
		FailureDomain: failureDomain,
		SessionImpact: impact,
		Reason:        vcacheScorePlaneReason(p),
		NextAction:    next,
	}
}

func vcacheScoreReportEvidence(rep vcachescore.Report) string {
	if hasVCacheScoreRealizedPlane(rep) {
		return "WITNESSED"
	}
	if rep.Planes.Forecast.Available {
		return "FORECAST"
	}
	return "MISSING"
}

func vcacheScoreReportStatus(rep vcachescore.Report) string {
	switch {
	case strings.TrimSpace(rep.Schema) == "" && strings.TrimSpace(rep.Status) == "" &&
		rep.Planes == (vcachescore.PlaneReport{}):
		return "missing"
	case !hasVCacheScoreRealizedPlane(rep) && rep.Planes.Forecast.Available:
		return "forecast_only"
	case strings.EqualFold(rep.DefaultUsefulness.Verdict, "not_ready"):
		return "not_ready"
	case strings.EqualFold(rep.DefaultUsefulness.Verdict, "partial"):
		return "partial"
	case !rep.TwoXBetter:
		return "not_2x"
	default:
		return "measured"
	}
}

func vcacheScoreReportReason(rep vcachescore.Report, path string) string {
	return fmt.Sprintf("status=%s grade=%s score=%d active=%s multiplier=%.2fx two_x=%v default=%s activation=%v source=%s",
		nonEmpty(rep.Status, "-"),
		nonEmpty(rep.Grade, "-"),
		rep.Score,
		nonEmpty(rep.ActiveSource, "-"),
		rep.ActiveMultiplier,
		rep.TwoXBetter,
		nonEmpty(rep.DefaultUsefulness.Verdict, "-"),
		rep.AgenticActivation.Active,
		path)
}

func vcacheScoreReportNextAction(rep vcachescore.Report) string {
	switch vcacheScoreReportStatus(rep) {
	case "missing":
		return "fak vcache score --json"
	case "forecast_only":
		return "supply provider telemetry, cachevalue ledger, context snapshot, or external engine evidence to fak vcache score"
	case "not_2x":
		return strings.Join(nonEmptyActions(rep.Actions), "; ")
	case "not_ready", "partial":
		if len(rep.Actions) > 0 {
			return strings.Join(nonEmptyActions(rep.Actions), "; ")
		}
		return "add realized fak/provider/external plane evidence before treating vCache as default-ready"
	default:
		return ""
	}
}

func hasVCacheScoreRealizedPlane(rep vcachescore.Report) bool {
	return rep.Planes.ProviderObserved.Available ||
		rep.Planes.KernelWitnessed.Available ||
		rep.Planes.ContextWitnessed.Available ||
		rep.Planes.ExternalEngineObserved.Available
}

func vcacheScorePlaneStatus(p vcachescore.PlaneValueReport) string {
	if !p.Available {
		return "missing"
	}
	switch strings.ToUpper(strings.TrimSpace(p.Provenance)) {
	case "OBSERVED":
		return "observed"
	case "WITNESSED":
		return "measured"
	case "FORECAST":
		return "forecast"
	default:
		return "measured"
	}
}

func vcacheScorePlaneEvidence(p vcachescore.PlaneValueReport) string {
	if strings.TrimSpace(p.Provenance) != "" {
		return p.Provenance
	}
	if !p.Available {
		return "MISSING"
	}
	return "unknown"
}

func vcacheScorePlaneLabel(p vcachescore.PlaneValueReport) string {
	if !p.Available {
		return "MISSING"
	}
	return nonEmpty(p.Provenance, "AVAILABLE")
}

func vcacheScorePlaneReason(p vcachescore.PlaneValueReport) string {
	if strings.TrimSpace(p.Reason) == "" {
		return fmt.Sprintf("available=%v provenance=%s multiplier=%.2fx saved=%.1f baseline=%.1f hit=%.2f%% cost=%.1f",
			p.Available,
			vcacheScorePlaneEvidence(p),
			p.Multiplier,
			p.SavedTokenEquiv,
			p.BaselineTokenEquiv,
			100*p.HitRate,
			p.CostTokenEquiv)
	}
	return fmt.Sprintf("%s (multiplier=%.2fx saved=%.1f baseline=%.1f hit=%.2f%% cost=%.1f)",
		p.Reason,
		p.Multiplier,
		p.SavedTokenEquiv,
		p.BaselineTokenEquiv,
		100*p.HitRate,
		p.CostTokenEquiv)
}

func nonEmptyActions(actions []string) []string {
	out := make([]string, 0, len(actions))
	for _, action := range actions {
		if strings.TrimSpace(action) != "" {
			out = append(out, strings.TrimSpace(action))
		}
	}
	if len(out) == 0 {
		return []string{"rerun fak vcache score with realized plane evidence"}
	}
	if len(out) > 3 {
		return out[:3]
	}
	return out
}
