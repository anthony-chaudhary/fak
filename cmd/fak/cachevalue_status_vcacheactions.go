package main

// Optional vcache actions digest -- summarizes planned warm/miss provider
// cache actions and their witnessed transport.
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

func loadCachevalueVCacheActionsStatus(path string) (cachevalueVCacheActionsDigest, []cachevalueStatusRow) {
	digest := cachevalueVCacheActionsDigest{Path: path, Status: "unavailable"}
	b, err := os.ReadFile(path)
	if err != nil {
		digest.Error = err.Error()
		return digest, []cachevalueStatusRow{vcacheActionsUnavailableRow(path, err.Error())}
	}
	var plan vcacheobserve.ProviderActionPlan
	if err := json.Unmarshal(b, &plan); err != nil {
		digest.Error = err.Error()
		return digest, []cachevalueStatusRow{vcacheActionsUnavailableRow(path, "decode: "+err.Error())}
	}
	digest.Status = vcacheActionsStatus(plan)
	digest.Turns = plan.Turns
	digest.FamilyCount = plan.FamilyCount
	digest.Noop = plan.Counts.Noop
	digest.Ready = plan.Counts.Ready
	digest.Gated = plan.Counts.Gated
	digest.TransportMode = plan.Transport.Mode
	digest.TransportReady = plan.Transport.Ready
	digest.TransportReason = plan.Transport.Reason
	return digest, rowsFromVCacheActionsPlan(plan, path)
}

func vcacheActionsUnavailableRow(path, reason string) cachevalueStatusRow {
	return cachevalueStatusRow{
		Plane:         "provider_prompt_cache_control",
		Component:     "vcache_actions_report",
		Owner:         "fak",
		Dependency:    "local_json_report",
		Fidelity:      "diagnostic",
		Evidence:      "MISSING",
		Status:        "unavailable",
		FailureDomain: "vcache_actions_artifact",
		SessionImpact: "provider-cache action-plan evidence could not be folded, so transport gating attribution is incomplete",
		Reason:        fmt.Sprintf("%s: %s", path, reason),
		NextAction:    "fak vcache actions --json --out FILE",
	}
}

func rowsFromVCacheActionsPlan(plan vcacheobserve.ProviderActionPlan, path string) []cachevalueStatusRow {
	rows := []cachevalueStatusRow{
		{
			Plane:         "provider_prompt_cache_control",
			Component:     "vcache_actions_report",
			Owner:         "fak",
			Dependency:    "provider_action_transport",
			Fidelity:      "lossless",
			Evidence:      "DECISION",
			Status:        vcacheActionsStatus(plan),
			FailureDomain: vcacheActionsFailureDomain(plan),
			SessionImpact: "provider-cache actions are fak-authored decisions over provider telemetry; ready/gated rows are not proof that a provider warm already executed",
			Reason:        vcacheActionsReportReason(plan, path),
			NextAction:    vcacheActionsNextAction(plan),
		},
		{
			Plane:         "provider_prompt_cache_control",
			Component:     "vcache_actions_transport",
			Owner:         "provider",
			Dependency:    "provider_action_transport",
			Fidelity:      "lossless",
			Evidence:      vcacheActionsTransportEvidence(plan),
			Status:        vcacheActionsTransportStatus(plan),
			FailureDomain: "provider_transport",
			SessionImpact: "spendful provider-cache actions require provider transport plus byte-identical prefix evidence; gated transport is provider/API boundary, not fak-native KV",
			Reason:        plan.Transport.Reason,
			NextAction:    vcacheActionsTransportNextAction(plan),
		},
	}
	for _, action := range plan.Actions {
		rows = append(rows, rowFromVCacheAction(action))
	}
	return rows
}

func rowFromVCacheAction(action vcacheobserve.ProviderAction) cachevalueStatusRow {
	owner, dependency, failureDomain := vcacheActionAttribution(action)
	return cachevalueStatusRow{
		Plane:         "provider_prompt_cache_control",
		Component:     "vcache_action:" + nonEmpty(action.Family, "unknown") + ":" + nonEmpty(action.Action, "unknown"),
		Owner:         owner,
		Dependency:    dependency,
		Fidelity:      "lossless",
		Evidence:      "DECISION",
		Status:        string(action.State),
		FailureDomain: failureDomain,
		SessionImpact: "one provider-cache family action candidate; state names whether this is a no-op, locally/provider-ready, or still gated by missing transport evidence",
		Reason:        vcacheActionReason(action),
		NextAction:    vcacheActionNextAction(action),
	}
}

func vcacheActionsStatus(plan vcacheobserve.ProviderActionPlan) string {
	switch {
	case strings.TrimSpace(plan.Schema) == "" && len(plan.Actions) == 0:
		return "missing"
	case plan.Counts.Gated > 0:
		return "gated"
	case plan.Counts.Ready > 0:
		return "ready"
	default:
		return "no-op"
	}
}

func vcacheActionsFailureDomain(plan vcacheobserve.ProviderActionPlan) string {
	if plan.Counts.Gated > 0 {
		return "provider_transport"
	}
	return "provider_prompt_cache_control"
}

func vcacheActionsReportReason(plan vcacheobserve.ProviderActionPlan, path string) string {
	return fmt.Sprintf("turns=%d families=%d noop=%d ready=%d gated=%d transport=%s ready=%v source=%s law=%s",
		plan.Turns,
		plan.FamilyCount,
		plan.Counts.Noop,
		plan.Counts.Ready,
		plan.Counts.Gated,
		nonEmpty(plan.Transport.Mode, "-"),
		plan.Transport.Ready,
		path,
		nonEmpty(plan.CorrectnessLaw, "-"))
}

func vcacheActionsNextAction(plan vcacheobserve.ProviderActionPlan) string {
	if plan.Counts.Gated == 0 {
		return ""
	}
	return "rerun fak vcache actions with required provider transport witnesses, e.g. --heartbeat-transport/--explicit-cache-transport --prefix-witness"
}

func vcacheActionsTransportStatus(plan vcacheobserve.ProviderActionPlan) string {
	if plan.Transport.Ready {
		return "ready"
	}
	if plan.Counts.Gated > 0 {
		return "gated"
	}
	return "no-op"
}

func vcacheActionsTransportEvidence(plan vcacheobserve.ProviderActionPlan) string {
	if plan.Transport.Ready || plan.Transport.Witness != nil {
		return "WITNESSED"
	}
	return "MISSING"
}

func vcacheActionsTransportNextAction(plan vcacheobserve.ProviderActionPlan) string {
	if vcacheActionsTransportStatus(plan) != "gated" {
		return ""
	}
	return vcacheActionsNextAction(plan)
}

func vcacheActionAttribution(action vcacheobserve.ProviderAction) (owner, dependency, failureDomain string) {
	switch action.Action {
	case "evict_manifest", "no_cache":
		return "fak", "local_provider_manifest", "fak_provider_manifest"
	case "heartbeat_pin", "explicit_cache":
		return "provider", "provider_action_transport", "provider_transport"
	case "ride_natural", "lazy_rebuild":
		return "provider", "natural_provider_cache_window", "provider_prompt_cache_control"
	default:
		return "unknown", "provider_action_transport", "provider_transport"
	}
}

func vcacheActionReason(action vcacheobserve.ProviderAction) string {
	missing := missingVCacheActionRequires(action.Requires, action.Witnessed)
	parts := []string{
		fmt.Sprintf("decision=%s action=%s state=%s turns=%d saved=%.1f", action.Decision, action.Action, action.State, action.Turns, action.SavedTokenEquiv),
	}
	if len(action.Requires) > 0 {
		parts = append(parts, "requires="+strings.Join(action.Requires, ","))
	}
	if len(action.Witnessed) > 0 {
		parts = append(parts, "witnessed="+strings.Join(action.Witnessed, ","))
	}
	if len(missing) > 0 {
		parts = append(parts, "missing="+strings.Join(missing, ","))
	}
	if strings.TrimSpace(action.Reason) != "" {
		parts = append(parts, action.Reason)
	}
	return strings.Join(parts, " ")
}

func vcacheActionNextAction(action vcacheobserve.ProviderAction) string {
	if action.State != vcacheobserve.ActionGated {
		return ""
	}
	missing := missingVCacheActionRequires(action.Requires, action.Witnessed)
	if len(missing) == 0 {
		return "inspect provider action transport witness"
	}
	return "supply provider action witness: " + strings.Join(missing, ",")
}

func missingVCacheActionRequires(requires, witnessed []string) []string {
	if len(requires) == 0 {
		return nil
	}
	seen := map[string]bool{}
	for _, w := range witnessed {
		seen[w] = true
	}
	var missing []string
	for _, req := range requires {
		if !seen[req] {
			missing = append(missing, req)
		}
	}
	return missing
}
