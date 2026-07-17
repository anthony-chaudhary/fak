package main

// Optional vcache context-join digest -- summarizes how context snapshots join
// to provider cache keys.
// Split out of cachevalue_status.go along this concern seam so the cachevalue
// status dispatch surface stays steerable as new digests land (steerability
// dispatch_god_file). Behavior-preserving code motion -- same package, no logic change.

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/vcacheobserve"
)

func loadCachevalueVCacheContextJoinStatus(path string) (cachevalueVCacheContextJoinDigest, []cachevalueStatusRow) {
	digest := cachevalueVCacheContextJoinDigest{Path: path, Status: "unavailable"}
	b, err := os.ReadFile(path)
	if err != nil {
		digest.Error = err.Error()
		return digest, []cachevalueStatusRow{vcacheContextJoinUnavailableRow(path, err.Error())}
	}
	var rep vcacheobserve.JoinReport
	if err := json.Unmarshal(b, &rep); err != nil {
		digest.Error = err.Error()
		return digest, []cachevalueStatusRow{vcacheContextJoinUnavailableRow(path, "decode: "+err.Error())}
	}
	digest.Status = vcacheContextJoinStatus(rep)
	digest.FailureDomain = vcacheContextJoinFailureDomain(rep)
	digest.Turns = rep.Turns
	digest.Events = rep.Events
	digest.TotalChanges = vcacheContextJoinTotalChanges(rep)
	digest.PlanningAttributed = rep.Summary.PlanningAttributed
	digest.ProviderAttributed = rep.Summary.ProviderAttributed
	return digest, rowsFromVCacheContextJoinReport(rep, path)
}

func vcacheContextJoinUnavailableRow(path, reason string) cachevalueStatusRow {
	return cachevalueStatusRow{
		Plane:         "managed_context",
		Component:     "vcache_context_join_report",
		Owner:         "fak",
		Dependency:    "local_json_report",
		Fidelity:      "diagnostic",
		Evidence:      "MISSING",
		Status:        "unavailable",
		FailureDomain: "vcache_context_join_artifact",
		SessionImpact: "context-join evidence could not be folded, so fak context-planning and provider cache behavior remain ambiguous",
		Reason:        fmt.Sprintf("%s: %s", path, reason),
		NextAction:    "fak vcache context-join --telemetry FILE --events FILE --json",
	}
}

func rowsFromVCacheContextJoinReport(rep vcacheobserve.JoinReport, path string) []cachevalueStatusRow {
	rows := []cachevalueStatusRow{{
		Plane:         "managed_context",
		Component:     "vcache_context_join_report",
		Owner:         "fak",
		Dependency:    "context_lifecycle_events",
		Fidelity:      "diagnostic",
		Evidence:      vcacheContextJoinEvidence(rep),
		Status:        vcacheContextJoinStatus(rep),
		FailureDomain: vcacheContextJoinFailureDomain(rep),
		SessionImpact: "context-join separates fak managed-context events from natural provider-cache behavior for bad-session attribution",
		Reason:        vcacheContextJoinReportReason(rep, path),
		NextAction:    vcacheContextJoinNextAction(rep),
	}}
	for i, change := range rep.Changes {
		rows = append(rows, rowFromVCacheContextJoinChange(change, i))
	}
	return rows
}

func rowFromVCacheContextJoinChange(change vcacheobserve.AttributedChange, idx int) cachevalueStatusRow {
	owner, dependency, fidelity, evidence, status, failureDomain := vcacheContextJoinChangeAttribution(change)
	return cachevalueStatusRow{
		Plane:         "managed_context",
		Component:     fmt.Sprintf("context_join:%s:%s:%d", nonEmpty(change.Family, "unknown"), nonEmpty(string(change.Change), "change"), idx+1),
		Owner:         owner,
		Dependency:    dependency,
		Fidelity:      fidelity,
		Evidence:      evidence,
		Status:        status,
		FailureDomain: failureDomain,
		SessionImpact: vcacheContextJoinChangeImpact(change),
		Reason:        vcacheContextJoinChangeReason(change),
		NextAction:    vcacheContextJoinChangeNextAction(change),
	}
}

func vcacheContextJoinChangeAttribution(change vcacheobserve.AttributedChange) (owner, dependency, fidelity, evidence, status, failureDomain string) {
	switch change.Cause {
	case vcacheobserve.CausePlanning:
		return "fak", "context_lifecycle_events", vcacheContextPlanningFidelity(change.MatchedEvent), "WITNESSED", string(vcacheobserve.CausePlanning), "fak_context_planner"
	case vcacheobserve.CauseProviderBehavior:
		return "provider", "provider_cache_telemetry", "passive", "OBSERVED", string(vcacheobserve.CauseProviderBehavior), "provider_cache_behavior"
	default:
		return "unknown", "provider_cache_telemetry", "diagnostic", "OBSERVED", "unattributed", "evidence_gap"
	}
}

func vcacheContextPlanningFidelity(ev *vcacheobserve.LifecycleEvent) string {
	if ev == nil {
		return "diagnostic"
	}
	switch ev.Kind {
	case vcacheobserve.EventCompaction, vcacheobserve.EventPrefixMutation:
		return "lossy"
	case vcacheobserve.EventContextReset:
		return "recoverable"
	case vcacheobserve.EventPageFault:
		return "passive"
	default:
		return "diagnostic"
	}
}

func vcacheContextJoinStatus(rep vcacheobserve.JoinReport) string {
	switch {
	case strings.TrimSpace(rep.Schema) == "" && rep.Turns == 0 && rep.Events == 0 && len(rep.Changes) == 0:
		return "missing"
	case vcacheContextJoinTotalChanges(rep) == 0:
		return "no-op"
	default:
		return "measured"
	}
}

func vcacheContextJoinEvidence(rep vcacheobserve.JoinReport) string {
	if strings.TrimSpace(rep.Schema) == "" && rep.Turns == 0 && rep.Events == 0 && len(rep.Changes) == 0 {
		return "MISSING"
	}
	if rep.Summary.PlanningAttributed > 0 || rep.Events > 0 {
		return "WITNESSED"
	}
	if rep.Summary.ProviderAttributed > 0 || len(rep.Changes) > 0 {
		return "OBSERVED"
	}
	return "WITNESSED"
}

func vcacheContextJoinFailureDomain(rep vcacheobserve.JoinReport) string {
	planning := rep.Summary.PlanningAttributed
	provider := rep.Summary.ProviderAttributed
	switch {
	case planning > 0 && provider > 0:
		return "mixed_context_provider"
	case planning > 0:
		return "fak_context_planner"
	case provider > 0:
		return "provider_cache_behavior"
	case vcacheContextJoinTotalChanges(rep) == 0:
		return "none"
	default:
		return "evidence_gap"
	}
}

func vcacheContextJoinTotalChanges(rep vcacheobserve.JoinReport) int {
	if rep.Summary.TotalChanges > 0 {
		return rep.Summary.TotalChanges
	}
	return len(rep.Changes)
}

func vcacheContextJoinReportReason(rep vcacheobserve.JoinReport, path string) string {
	return fmt.Sprintf("turns=%d events=%d changes=%d planning=%d provider=%d source=%s",
		rep.Turns,
		rep.Events,
		vcacheContextJoinTotalChanges(rep),
		rep.Summary.PlanningAttributed,
		rep.Summary.ProviderAttributed,
		path)
}

func vcacheContextJoinNextAction(rep vcacheobserve.JoinReport) string {
	planning := rep.Summary.PlanningAttributed
	provider := rep.Summary.ProviderAttributed
	switch {
	case vcacheContextJoinStatus(rep) == "missing":
		return "fak vcache context-join --telemetry FILE --events FILE --json"
	case planning > 0 && provider > 0:
		return "inspect matched fak lifecycle events and provider TTL/prefix behavior"
	case planning > 0:
		return "inspect matched fak lifecycle events with fak vcache context-witness --json"
	case provider > 0:
		return "inspect provider TTL/prefix churn or run fak vcache actions --json"
	default:
		return ""
	}
}

func vcacheContextJoinChangeImpact(change vcacheobserve.AttributedChange) string {
	switch change.Cause {
	case vcacheobserve.CausePlanning:
		return "a fak managed-context lifecycle event explains the provider cost change; inspect reset/compaction/page-fault handling before blaming natural provider cache behavior"
	case vcacheobserve.CauseProviderBehavior:
		return "the cost change had no nearby fak lifecycle event; suspect provider TTL, prefix churn, or provider cache-window behavior"
	default:
		return "the cost change is not attributable from the supplied lifecycle and provider telemetry"
	}
}

func vcacheContextJoinChangeReason(change vcacheobserve.AttributedChange) string {
	parts := []string{
		fmt.Sprintf("family=%s change=%s cause=%s", nonEmpty(change.Family, "unknown"), nonEmpty(string(change.Change), "change"), nonEmpty(string(change.Cause), "unknown")),
	}
	if strings.TrimSpace(change.Detail) != "" {
		parts = append(parts, change.Detail)
	}
	if change.MatchedEvent != nil {
		parts = append(parts, fmt.Sprintf("matched=%s outcome=%s detail=%s",
			change.MatchedEvent.Kind,
			nonEmpty(change.MatchedEvent.Outcome, "-"),
			nonEmpty(change.MatchedEvent.Detail, "-")))
	}
	return strings.Join(parts, " ")
}

func vcacheContextJoinChangeNextAction(change vcacheobserve.AttributedChange) string {
	switch change.Cause {
	case vcacheobserve.CausePlanning:
		if change.MatchedEvent != nil {
			return "inspect fak lifecycle event " + string(change.MatchedEvent.Kind)
		}
		return "rerun fak vcache context-join with lifecycle events"
	case vcacheobserve.CauseProviderBehavior:
		return "inspect provider TTL/prefix churn or run fak vcache actions --json"
	default:
		return "rerun fak vcache context-join with a complete lifecycle-event stream"
	}
}
