package main

// Optional vcache observe digest -- summarizes passively observed provider
// cache telemetry.
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

func loadCachevalueVCacheObserveStatus(path string) (cachevalueVCacheObserveDigest, []cachevalueStatusRow) {
	digest := cachevalueVCacheObserveDigest{Path: path, Status: "unavailable"}
	b, err := os.ReadFile(path)
	if err != nil {
		digest.Error = err.Error()
		return digest, []cachevalueStatusRow{vcacheObserveUnavailableRow(path, err.Error())}
	}
	var rep vcacheobserve.Report
	if err := json.Unmarshal(b, &rep); err != nil {
		digest.Error = err.Error()
		return digest, []cachevalueStatusRow{vcacheObserveUnavailableRow(path, "decode: "+err.Error())}
	}
	digest = cachevalueVCacheObserveDigestFromReport(path, rep)
	return digest, rowsFromVCacheObserveReport(rep, path)
}

func loadCachevalueVCacheObserveFromSession(path string) (cachevalueVCacheObserveDigest, []cachevalueStatusRow) {
	source := "session:" + path
	digest := cachevalueVCacheObserveDigest{Path: source, Status: "unavailable"}
	turns, err := readObserveTranscript(path)
	if err != nil {
		digest.Error = err.Error()
		return digest, []cachevalueStatusRow{vcacheObserveUnavailableRow(source, err.Error())}
	}
	rep := vcacheobserve.ObserveWithOptions(turns, vcacheobserve.DefaultOptions())
	digest = cachevalueVCacheObserveDigestFromReport(source, rep)
	return digest, rowsFromVCacheObserveReport(rep, source)
}

func cachevalueVCacheObserveDigestFromReport(path string, rep vcacheobserve.Report) cachevalueVCacheObserveDigest {
	digest := cachevalueVCacheObserveDigest{Path: path}
	digest.Status = vcacheObserveStatus(rep)
	digest.FailureDomain = vcacheObserveFailureDomain(rep)
	digest.Turns = rep.Turns
	digest.FamilyCount = rep.FamilyCount
	digest.TurnsReordered = rep.TurnsReordered
	digest.OutOfOrderTurns = rep.OutOfOrderTurns
	digest.HitRate = rep.HitRate
	digest.Multiplier = rep.Multiplier
	digest.SavedTokenEquiv = rep.Aggregate.SavedTokenEquiv
	digest.CacheReadTokens = rep.Aggregate.CacheReadTokens
	digest.CacheCreationTokens = rep.Aggregate.CacheCreationTokens
	digest.FalseWarm = rep.Prediction.FalseWarm
	digest.FalseWarmRate = rep.Prediction.FalseWarmRate()
	return digest
}

func vcacheObserveUnavailableRow(path, reason string) cachevalueStatusRow {
	return cachevalueStatusRow{
		Plane:         "provider_prompt_cache",
		Component:     "vcache_observe_report",
		Owner:         "provider",
		Dependency:    "local_json_report",
		Fidelity:      "lossless",
		Evidence:      "MISSING",
		Status:        "unavailable",
		FailureDomain: "vcache_observe_artifact",
		SessionImpact: "provider-cache observation evidence could not be folded, so provider hit/miss behavior remains ambiguous",
		Reason:        fmt.Sprintf("%s: %s", path, reason),
		NextAction:    "fak vcache observe --transcript FILE --json",
	}
}

func rowsFromVCacheObserveReport(rep vcacheobserve.Report, path string) []cachevalueStatusRow {
	rows := []cachevalueStatusRow{{
		Plane:         "provider_prompt_cache",
		Component:     "vcache_observe_report",
		Owner:         "provider",
		Dependency:    "provider_usage_fields",
		Fidelity:      "lossless",
		Evidence:      vcacheObserveEvidence(rep),
		Status:        vcacheObserveStatus(rep),
		FailureDomain: vcacheObserveFailureDomain(rep),
		SessionImpact: "direct provider-cache telemetry separates realized provider rebates and misses from fak-owned KV/context behavior",
		Reason:        vcacheObserveReportReason(rep, path),
		NextAction:    vcacheObserveNextAction(rep),
	}}
	for _, slice := range rep.OwnerSlices {
		rows = append(rows, rowFromVCacheObserveOwnerSlice(slice))
	}
	for _, family := range rep.Families {
		rows = append(rows, rowFromVCacheObserveFamily(family))
	}
	return rows
}

func rowFromVCacheObserveOwnerSlice(slice vcacheobserve.OwnerSlice) cachevalueStatusRow {
	owner := nonEmpty(slice.Owner, "unknown")
	mechanism := nonEmpty(slice.Mechanism, "unknown")
	status := vcacheObserveOwnerSliceStatus(slice)
	return cachevalueStatusRow{
		Plane:         "provider_prompt_cache",
		Component:     "vcache_observe_owner:" + componentKey(owner) + ":" + componentKey(mechanism),
		Owner:         owner,
		Dependency:    vcacheObserveOwnerDependency(owner),
		Fidelity:      vcacheObserveOwnerFidelity(slice),
		Evidence:      string(slice.Provenance),
		Status:        status,
		FailureDomain: vcacheObserveOwnerFailureDomain(slice),
		SessionImpact: vcacheObserveOwnerImpact(slice),
		Reason:        vcacheObserveOwnerReason(slice),
		NextAction:    vcacheObserveOwnerNextAction(slice),
	}
}

func rowFromVCacheObserveFamily(family vcacheobserve.Family) cachevalueStatusRow {
	status := vcacheObserveFamilyStatus(family)
	return cachevalueStatusRow{
		Plane:         "provider_prompt_cache",
		Component:     "vcache_observe_family:" + componentKey(nonEmpty(family.Key, "unknown")),
		Owner:         "provider",
		Dependency:    "provider_usage_fields",
		Fidelity:      "lossless",
		Evidence:      "OBSERVED",
		Status:        status,
		FailureDomain: vcacheObserveFamilyFailureDomain(family),
		SessionImpact: vcacheObserveFamilyImpact(family),
		Reason:        vcacheObserveFamilyReason(family),
		NextAction:    vcacheObserveFamilyNextAction(family),
	}
}

func vcacheObserveStatus(rep vcacheobserve.Report) string {
	switch {
	case strings.TrimSpace(rep.Schema) == "" && rep.Turns == 0 && rep.FamilyCount == 0:
		return "missing"
	case rep.Prediction.FalseWarm > 0:
		return "false_warm"
	case rep.TurnsReordered:
		return "turns_reordered"
	case rep.Aggregate.CacheReadTokens > 0:
		return "observed"
	case rep.Aggregate.CacheCreationTokens > 0:
		return "cold_write_only"
	default:
		return "no_usage"
	}
}

func vcacheObserveEvidence(rep vcacheobserve.Report) string {
	if strings.TrimSpace(rep.Schema) == "" && rep.Turns == 0 && rep.FamilyCount == 0 {
		return "MISSING"
	}
	return "OBSERVED"
}

func vcacheObserveFailureDomain(rep vcacheobserve.Report) string {
	switch vcacheObserveStatus(rep) {
	case "false_warm":
		return "provider_cache_prediction"
	case "turns_reordered":
		return "provider_telemetry_ordering"
	case "cold_write_only":
		return "provider_cache_window"
	case "missing", "no_usage":
		return "provider_telemetry"
	default:
		return "provider_prompt_cache"
	}
}

func vcacheObserveReportReason(rep vcacheobserve.Report, path string) string {
	return fmt.Sprintf("turns=%d families=%d hit=%.2f%% multiplier=%.2fx saved=%.1f cache_read=%.0f cache_create=%.0f false_warm=%d false_warm_rate=%.2f%% reordered=%v/%d source=%s",
		rep.Turns,
		rep.FamilyCount,
		100*rep.HitRate,
		rep.Multiplier,
		rep.Aggregate.SavedTokenEquiv,
		rep.Aggregate.CacheReadTokens,
		rep.Aggregate.CacheCreationTokens,
		rep.Prediction.FalseWarm,
		100*rep.Prediction.FalseWarmRate(),
		rep.TurnsReordered,
		rep.OutOfOrderTurns,
		path)
}

func vcacheObserveNextAction(rep vcacheobserve.Report) string {
	switch vcacheObserveStatus(rep) {
	case "missing", "no_usage":
		return "fak vcache observe --transcript FILE --json"
	case "false_warm":
		return "run fak vcache context-join --events FILE and inspect provider TTL/prefix calibration"
	case "turns_reordered":
		return "sort or timestamp provider telemetry before comparing TTL-sensitive cache behavior"
	case "cold_write_only":
		return "run fak vcache context-join --events FILE or fak vcache actions --json"
	default:
		return ""
	}
}

func vcacheObserveOwnerSliceStatus(slice vcacheobserve.OwnerSlice) string {
	switch slice.Provenance {
	case vcacheobserve.Observed:
		if slice.SavedTokenEquiv > 0 || slice.CacheReadTokens > 0 {
			return "observed"
		}
		if slice.CacheCreationTokens > 0 {
			return "cold_write_only"
		}
		return "no_effect"
	case vcacheobserve.NotObserved:
		return "not_observed"
	case vcacheobserve.Decision:
		return "ready"
	default:
		return "unknown"
	}
}

func vcacheObserveOwnerDependency(owner string) string {
	switch strings.ToLower(strings.TrimSpace(owner)) {
	case "provider":
		return "provider_usage_fields"
	case "fak":
		return "fak_cache_witness"
	default:
		return "observe_report"
	}
}

func vcacheObserveOwnerFidelity(slice vcacheobserve.OwnerSlice) string {
	if slice.Provenance == vcacheobserve.NotObserved {
		return "diagnostic"
	}
	if strings.EqualFold(slice.Owner, "provider") {
		return "lossless"
	}
	return "diagnostic"
}

func vcacheObserveOwnerFailureDomain(slice vcacheobserve.OwnerSlice) string {
	if slice.Provenance == vcacheobserve.NotObserved {
		return "evidence_gap"
	}
	if strings.EqualFold(slice.Owner, "provider") {
		return "provider_prompt_cache"
	}
	return cachevalueFailureDomain(slice.Owner, slice.Mechanism)
}

func vcacheObserveOwnerImpact(slice vcacheobserve.OwnerSlice) string {
	if slice.Provenance == vcacheobserve.NotObserved {
		return "this observe source cannot prove fak-authored KV/context effects; use fak witnesses before attributing a bad session to fak"
	}
	if strings.EqualFold(slice.Owner, "provider") {
		return "provider-reported prompt-cache economics; these are external rebates/misses, not fak-native KV hits"
	}
	return "owner slice from the vcache observe report"
}

func vcacheObserveOwnerReason(slice vcacheobserve.OwnerSlice) string {
	return fmt.Sprintf("mechanism=%s provenance=%s saved=%.1f cache_read=%.0f cache_create=%.0f evidence=%s",
		nonEmpty(slice.Mechanism, "unknown"),
		slice.Provenance,
		slice.SavedTokenEquiv,
		slice.CacheReadTokens,
		slice.CacheCreationTokens,
		nonEmpty(slice.Evidence, "-"))
}

func vcacheObserveOwnerNextAction(slice vcacheobserve.OwnerSlice) string {
	if slice.Provenance != vcacheobserve.NotObserved {
		return ""
	}
	return "add fak-owned witnesses with fak vcache score --kernel-ledger default --json or fak vcache context-witness --json"
}

func vcacheObserveFamilyStatus(family vcacheobserve.Family) string {
	switch {
	case family.Prediction.FalseWarm > 0:
		return "false_warm"
	case family.TurnsReordered:
		return "turns_reordered"
	case family.CacheReadTokens > 0:
		return "observed"
	case family.CacheCreationTokens > 0:
		return "cold_write_only"
	default:
		return "no_usage"
	}
}

func vcacheObserveFamilyFailureDomain(family vcacheobserve.Family) string {
	switch vcacheObserveFamilyStatus(family) {
	case "false_warm":
		return "provider_cache_prediction"
	case "turns_reordered":
		return "provider_telemetry_ordering"
	case "cold_write_only":
		return "provider_cache_window"
	case "no_usage":
		return "provider_telemetry"
	default:
		return "provider_prompt_cache"
	}
}

func vcacheObserveFamilyImpact(family vcacheobserve.Family) string {
	switch vcacheObserveFamilyStatus(family) {
	case "false_warm":
		return "fak's warmth belief expected a provider cache hit but provider telemetry missed; inspect TTL, prefix churn, and context events"
	case "turns_reordered":
		return "same-family telemetry arrived out of order; TTL-sensitive attribution was repaired by sorting but source ordering should be fixed"
	case "cold_write_only":
		return "provider cache writes happened without reads for this family; suspect cold start, TTL expiry, or prefix churn"
	default:
		return "per-family provider prompt-cache observation"
	}
}

func vcacheObserveFamilyReason(family vcacheobserve.Family) string {
	return fmt.Sprintf("turns=%d hit=%.2f%% cache_read=%d cache_create=%d saved=%.1f governor=%s false_warm=%d false_warm_rate=%.2f%% reordered=%v/%d",
		family.Turns,
		100*family.HitRate,
		family.CacheReadTokens,
		family.CacheCreationTokens,
		family.Economics.SavedTokenEquiv,
		family.GovernorDecision,
		family.Prediction.FalseWarm,
		100*family.Prediction.FalseWarmRate(),
		family.TurnsReordered,
		family.OutOfOrderTurns)
}

func vcacheObserveFamilyNextAction(family vcacheobserve.Family) string {
	switch vcacheObserveFamilyStatus(family) {
	case "false_warm":
		return "run fak vcache context-join --events FILE for family " + nonEmpty(family.Key, "unknown")
	case "turns_reordered":
		return "sort or timestamp telemetry for family " + nonEmpty(family.Key, "unknown")
	case "cold_write_only":
		return "inspect TTL/prefix churn for family " + nonEmpty(family.Key, "unknown")
	default:
		return ""
	}
}

func componentKey(s string) string {
	s = strings.TrimSpace(s)
	if s == "" {
		return "unknown"
	}
	replacer := strings.NewReplacer(" ", "_", "\t", "_", "\n", "_", ":", "_", "/", "_", "\\", "_")
	return replacer.Replace(s)
}
