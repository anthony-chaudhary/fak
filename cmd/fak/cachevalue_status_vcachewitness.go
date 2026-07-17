package main

// Optional vcache context-witness digest -- summarizes the context-witness
// evidence behind cache-key stability.
// Split out of cachevalue_status.go along this concern seam so the cachevalue
// status dispatch surface stays steerable as new digests land (steerability
// dispatch_god_file). Behavior-preserving code motion -- same package, no logic change.

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/ablate"
	"github.com/anthony-chaudhary/fak/internal/vcachescore"
)

func loadCachevalueVCacheContextWitnessStatus(path string) (cachevalueVCacheContextWitnessDigest, []cachevalueStatusRow) {
	digest := cachevalueVCacheContextWitnessDigest{Path: path, Status: "unavailable"}
	b, err := os.ReadFile(path)
	if err != nil {
		digest.Error = err.Error()
		return digest, []cachevalueStatusRow{vcacheContextWitnessUnavailableRow(path, err.Error())}
	}
	var rep vcacheContextWitnessReport
	if err := json.Unmarshal(b, &rep); err != nil {
		digest.Error = err.Error()
		return digest, []cachevalueStatusRow{vcacheContextWitnessUnavailableRow(path, "decode: "+err.Error())}
	}
	digest.Status = vcacheContextWitnessStatus(rep)
	digest.FailureDomain = vcacheContextWitnessFailureDomain(rep)
	digest.Fixture = rep.Fixture
	digest.Wire = rep.Wire
	digest.Snapshot = rep.Snapshot
	digest.ReplayExit = rep.ReplayExit
	digest.ScoreExit = rep.ScoreExit
	digest.ScoreStatus = rep.ScoreStatus
	digest.ContextWitnessed = vcacheContextWitnessPlaneLabel(rep.ContextWitnessed)
	digest.ContextEvents = rep.ContextEvents
	digest.ContextShedTokens = rep.ContextShedTokens
	return digest, rowsFromVCacheContextWitnessReport(rep, path)
}

func vcacheContextWitnessUnavailableRow(path, reason string) cachevalueStatusRow {
	return cachevalueStatusRow{
		Plane:         "managed_context",
		Component:     "vcache_context_witness_report",
		Owner:         "fak",
		Dependency:    "local_json_report",
		Fidelity:      "lossy",
		Evidence:      "MISSING",
		Status:        "unavailable",
		FailureDomain: "vcache_context_witness_artifact",
		SessionImpact: "fak context-witness evidence could not be folded, so lossy context behavior remains unevidenced",
		Reason:        fmt.Sprintf("%s: %s", path, reason),
		NextAction:    "fak vcache context-witness --json",
	}
}

func rowsFromVCacheContextWitnessReport(rep vcacheContextWitnessReport, path string) []cachevalueStatusRow {
	rows := []cachevalueStatusRow{{
		Plane:         "managed_context",
		Component:     "vcache_context_witness_report",
		Owner:         "fak",
		Dependency:    "guard_replay_context_snapshot",
		Fidelity:      "lossy",
		Evidence:      vcacheContextWitnessEvidence(rep),
		Status:        vcacheContextWitnessStatus(rep),
		FailureDomain: vcacheContextWitnessFailureDomain(rep),
		SessionImpact: "fak-owned context replay proves lossy context shedding separately from provider prompt-cache hit/miss behavior",
		Reason:        vcacheContextWitnessReason(rep, path),
		NextAction:    vcacheContextWitnessNextAction(rep),
	}}
	rows = append(rows, cachevalueStatusRow{
		Plane:         "managed_context",
		Component:     "vcache_context_witness_plane",
		Owner:         "fak",
		Dependency:    "vcache_context_snapshot",
		Fidelity:      "lossy",
		Evidence:      vcacheContextWitnessPlaneEvidence(rep.ContextWitnessed),
		Status:        vcacheContextWitnessPlaneStatus(rep.ContextWitnessed),
		FailureDomain: "fak_context_planner",
		SessionImpact: "context-witness plane carries the shed-token economics used by fak vcache score",
		Reason:        vcacheScorePlaneReason(rep.ContextWitnessed),
		NextAction:    vcacheContextWitnessPlaneNextAction(rep.ContextWitnessed),
	})
	return rows
}

func vcacheContextWitnessStatus(rep vcacheContextWitnessReport) string {
	switch {
	case strings.TrimSpace(rep.Schema) == "" && strings.TrimSpace(rep.Snapshot) == "":
		return "missing"
	case rep.ReplayExit != 0:
		return "replay_failed"
	case rep.ScoreExit != 0:
		return "score_failed"
	case !rep.ContextWitnessed.Available || rep.ContextEvents == 0:
		return "missing"
	default:
		return "measured"
	}
}

func vcacheContextWitnessEvidence(rep vcacheContextWitnessReport) string {
	if vcacheContextWitnessStatus(rep) == "measured" {
		return "WITNESSED"
	}
	if rep.ContextWitnessed.Available {
		return vcacheContextWitnessPlaneEvidence(rep.ContextWitnessed)
	}
	return "MISSING"
}

func vcacheContextWitnessFailureDomain(rep vcacheContextWitnessReport) string {
	switch vcacheContextWitnessStatus(rep) {
	case "replay_failed":
		return "fak_context_replay"
	case "score_failed":
		return "fak_context_score"
	case "missing":
		return "fak_context_evidence_gap"
	default:
		return "fak_context_planner"
	}
}

func vcacheContextWitnessReason(rep vcacheContextWitnessReport, path string) string {
	return fmt.Sprintf("fixture=%s wire=%s snapshot=%s replay_exit=%d score_exit=%d score_status=%s context=%s events=%d shed=%.1f source=%s",
		nonEmpty(rep.Fixture, "-"),
		nonEmpty(rep.Wire, "-"),
		nonEmpty(rep.Snapshot, "-"),
		rep.ReplayExit,
		rep.ScoreExit,
		nonEmpty(rep.ScoreStatus, "-"),
		vcacheContextWitnessPlaneLabel(rep.ContextWitnessed),
		rep.ContextEvents,
		rep.ContextShedTokens,
		path)
}

func vcacheContextWitnessNextAction(rep vcacheContextWitnessReport) string {
	switch vcacheContextWitnessStatus(rep) {
	case "replay_failed":
		return "rerun fak vcache context-witness and inspect guard replay failure"
	case "score_failed":
		return "rerun fak vcache score --context-snapshot " + nonEmpty(rep.Snapshot, "FILE") + " --json"
	case "missing":
		return "rerun fak vcache context-witness --json and ensure the replay emits fak_context_* counters"
	default:
		return ""
	}
}

func vcacheContextWitnessPlaneStatus(p vcachescore.PlaneValueReport) string {
	if !p.Available {
		return "missing"
	}
	return "measured"
}

func vcacheContextWitnessPlaneEvidence(p vcachescore.PlaneValueReport) string {
	if strings.TrimSpace(p.Provenance) != "" {
		return p.Provenance
	}
	if p.Available {
		return "WITNESSED"
	}
	return "MISSING"
}

func vcacheContextWitnessPlaneLabel(p vcachescore.PlaneValueReport) string {
	if !p.Available {
		return "MISSING"
	}
	return nonEmpty(p.Provenance, "WITNESSED")
}

func vcacheContextWitnessPlaneNextAction(p vcachescore.PlaneValueReport) string {
	if p.Available {
		return ""
	}
	return "fak vcache context-witness --json"
}

func cacheAblationStatusRow() cachevalueStatusRow {
	features := cacheAblationFeatures()
	return cachevalueStatusRow{
		Plane:         "diagnostics",
		Component:     "cache_ablation_runner",
		Owner:         "fak",
		Dependency:    "subprocess_reexec",
		Fidelity:      "diagnostic",
		Evidence:      "configured",
		Status:        "available",
		FailureDomain: "fak_diagnostics",
		SessionImpact: "use this when a bad session needs a controlled fak-native versus provider-cache comparison",
		Reason:        "can sweep cache/cache-adjacent runtime features: " + strings.Join(features, ","),
		NextAction:    "fak ablate --sweep " + strings.Join(features, ",") + " --json",
	}
}

func cacheAblationFeatures() []string {
	cacheFeatures := map[string]bool{
		"bp_plan":       true,
		"compressor":    true,
		"ctxplan_seam":  true,
		"prefix_guard":  true,
		"radix":         true,
		"ttl_1h":        true,
		"uncached_trim": true,
		"vdso":          true,
	}
	var out []string
	for _, feature := range ablate.KnownFeatures() {
		if cacheFeatures[feature] {
			out = append(out, feature)
		}
	}
	if len(out) == 0 {
		out = []string{"vdso"}
	}
	return out
}
