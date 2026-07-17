package main

// Optional session digest -- folds a sessionaudit.Session into the report's
// per-session cache-value view.
// Split out of cachevalue_status.go along this concern seam so the cachevalue
// status dispatch surface stays steerable as new digests land (steerability
// dispatch_god_file). Behavior-preserving code motion -- same package, no logic change.

import (
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/sessionaudit"
)

func cachevalueSessionDigestFromSession(s sessionaudit.Session) cachevalueSessionDigest {
	total := s.TotalInputTokens
	if total == 0 {
		total = s.Tokens.Input + s.Tokens.CacheRead + s.Tokens.CacheCreate
	}
	status, domain, finding := cachevalueSessionHeadline(s, total)
	return cachevalueSessionDigest{
		Path:               s.Path,
		Session:            nonEmptySessionName(s),
		Status:             status,
		LikelyDomain:       domain,
		Finding:            finding,
		AssistantTurns:     s.AssistantTurns,
		ToolCalls:          s.NToolUse,
		ReadOnlyToolFrac:   s.ReadOnlyFrac,
		InputTokens:        s.Tokens.Input,
		OutputTokens:       s.Tokens.Output,
		CacheReadTokens:    s.Tokens.CacheRead,
		CacheCreateTokens:  s.Tokens.CacheCreate,
		TotalContextTokens: total,
		CacheHitFrac:       s.CacheHitFrac,
		IORatio:            s.IORatio,
		CostUSD:            s.CostUSD,
		Error:              s.Error,
	}
}

func rowsFromSessionDiagnosis(s sessionaudit.Session) []cachevalueStatusRow {
	if s.Error != "" {
		return []cachevalueStatusRow{{
			Plane:         "session",
			Component:     "session_transcript",
			Owner:         "unknown",
			Dependency:    "session_file",
			Fidelity:      "passive",
			Evidence:      "MISSING",
			Status:        "unavailable",
			FailureDomain: "transcript",
			SessionImpact: "session diagnosis cannot attribute cache behavior because the transcript could not be read",
			Reason:        s.Error,
			NextAction:    "pass a readable Claude Code transcript JSONL to --session",
		}}
	}
	total := s.TotalInputTokens
	if total == 0 {
		total = s.Tokens.Input + s.Tokens.CacheRead + s.Tokens.CacheCreate
	}
	rows := []cachevalueStatusRow{
		sessionProviderCacheRow(s, total),
		sessionWorkloadRow(s, total),
		sessionFakContextVisibilityRow(s),
	}
	return rows
}

func sessionProviderCacheRow(s sessionaudit.Session, total int64) cachevalueStatusRow {
	status := "no_provider_cache_evidence"
	reason := fmt.Sprintf("session has %d input, %d cache-read, and %d cache-create token(s)", s.Tokens.Input, s.Tokens.CacheRead, s.Tokens.CacheCreate)
	impact := "provider prompt-cache counters are absent, so a bad session may be prompt churn, an uncached provider path, or missing telemetry"
	next := "fak vcache observe --transcript " + quotePathForHint(s.Path) + " --json"
	if total == 0 || s.AssistantTurns == 0 {
		status = "no_usage"
		impact = "transcript has no assistant usage rows, so cache behavior is not attributable from this file"
		next = "verify the session transcript contains assistant message usage blocks"
	} else if s.Tokens.CacheRead > 0 {
		status = "observed"
		impact = "provider cache-read tokens are present; if the session still went badly, inspect workload pressure and fak context rows next"
		next = ""
	} else if s.Tokens.CacheCreate > 0 {
		status = "cold_write_only"
		impact = "provider cache writes happened without later reads; this points at cold start, prefix churn, or TTL/window behavior outside fak-native KV"
		next = "fak vcache context-join --events FILE"
	}
	return cachevalueStatusRow{
		Plane:         "session_provider_prompt_cache",
		Component:     "session_provider_cache",
		Owner:         "provider",
		Dependency:    "session_transcript_usage",
		Fidelity:      "lossless",
		Evidence:      "OBSERVED",
		Status:        status,
		FailureDomain: sessionProviderFailureDomain(status),
		SessionImpact: impact,
		Reason:        reason,
		NextAction:    next,
	}
}

func sessionWorkloadRow(s sessionaudit.Session, total int64) cachevalueStatusRow {
	status := "measured"
	if total == 0 && s.Tokens.Output == 0 {
		status = "no_usage"
	} else if sessionHighContextPressure(s, total) {
		status = "high_pressure"
	}
	reason := fmt.Sprintf("total_context=%d output=%d io_ratio=%s tool_result_chars=%d read_only_tool_frac=%s",
		total, s.Tokens.Output, fmtFloatPtr(s.IORatio), s.ToolResultChars, fmtFloatPtr(s.ReadOnlyFrac))
	next := ""
	if status == "high_pressure" {
		next = "fak headroom bench --via native"
	}
	return cachevalueStatusRow{
		Plane:         "session_workload",
		Component:     "session_context_pressure",
		Owner:         "workload",
		Dependency:    "session_transcript_usage",
		Fidelity:      "passive",
		Evidence:      "OBSERVED",
		Status:        status,
		FailureDomain: "workload",
		SessionImpact: "large tool/prompt context can make a session bad even when cache machinery is working; this is not by itself a fak cache fault",
		Reason:        reason,
		NextAction:    next,
	}
}

func sessionFakContextVisibilityRow(s sessionaudit.Session) cachevalueStatusRow {
	return cachevalueStatusRow{
		Plane:         "session_fak_context",
		Component:     "session_fak_context_events",
		Owner:         "fak",
		Dependency:    "vcache_context_snapshot",
		Fidelity:      "lossy",
		Evidence:      "MISSING",
		Status:        "not_observed_from_transcript",
		FailureDomain: "evidence_gap",
		SessionImpact: "Claude transcripts show provider usage counters but do not prove fak context drops/compaction; use a fak context snapshot before blaming fak context planning",
		Reason:        "transcript " + nonEmptySessionName(s) + " has no fak_context_* counters in the sessionaudit shape",
		NextAction:    "fak vcache context-witness --json",
	}
}

func cachevalueSessionHeadline(s sessionaudit.Session, total int64) (status, domain, finding string) {
	if s.Error != "" {
		return "unavailable", "transcript", s.Error
	}
	if total == 0 || s.AssistantTurns == 0 {
		return "no_usage", "transcript", "no assistant usage rows were found, so cache behavior is not attributable"
	}
	if s.Tokens.CacheRead == 0 && s.Tokens.CacheCreate > 0 {
		return "partial", "provider_prompt_cache", "provider cache writes occurred but no cache reads were observed; suspect cold start, prefix churn, or TTL/window behavior"
	}
	if s.Tokens.CacheRead == 0 {
		return "partial", "provider_telemetry", "no provider cache-read evidence was observed in the transcript"
	}
	if sessionHighContextPressure(s, total) {
		return "partial", "workload", "provider cache reads are present, but context/workload pressure is high"
	}
	return "observed", "provider_prompt_cache", "provider cache-read evidence is present in the session transcript"
}

func sessionHighContextPressure(s sessionaudit.Session, total int64) bool {
	if total >= 200_000 {
		return true
	}
	return s.IORatio != nil && *s.IORatio >= 100
}

func sessionProviderFailureDomain(status string) string {
	switch status {
	case "observed":
		return "provider"
	case "cold_write_only":
		return "provider_cache_window"
	case "no_usage":
		return "transcript"
	default:
		return "provider_telemetry_or_prompt_churn"
	}
}
