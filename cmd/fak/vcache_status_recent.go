package main

import (
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/vcacheobserve"
	"github.com/anthony-chaudhary/fak/internal/vcachesnapshot"
)

func applyRecentVCacheObservation(rep *vcacheStatusReport, path, contextPath string) {
	path = strings.TrimSpace(path)
	if path == "" || strings.EqualFold(path, "off") {
		applyRecentVCacheContextOnlyObservation(rep, contextPath)
		return
	}
	turns, ok, err := vcachesnapshot.Read(path)
	if err != nil {
		rep.RecentObservationError = fmt.Sprintf("%s: %v", path, err)
		return
	}
	if !ok {
		applyRecentVCacheContextOnlyObservation(rep, contextPath)
		return
	}
	providerTurns := vcacheobserve.ProviderTelemetryTurns(turns)
	obs := vcacheobserve.Observe(providerTurns, vcacheobserve.DefaultMultipliers())
	recent := vcacheRecentObservation{
		Source:              "snapshot",
		Path:                path,
		Turns:               len(turns),
		ProviderStatus:      "MISSING",
		CacheReadTokens:     obs.Aggregate.CacheReadTokens,
		CacheCreationTokens: obs.Aggregate.CacheCreationTokens,
		HitRate:             obs.HitRate,
		Multiplier:          obs.Multiplier,
		SavedTokenEquiv:     obs.Aggregate.SavedTokenEquiv,
		FalseWarmRate:       obs.Prediction.FalseWarmRate(),
		FalseColdRate:       obs.Prediction.FalseColdRate(),
	}
	if len(providerTurns) > 0 {
		recent.ProviderStatus = string(obs.Aggregate.Status)
		recent.GovernorDecision = dominantVCacheGovernorDecision(obs.Families)
	}
	accumulateRecentVCacheContext(&recent, turns)
	recent.ContextStatus, recent.ContextReason = recentVCacheContextStatus(recent)
	if recent.ContextStatus == "MISSING" {
		applyRecentVCacheContextSnapshot(&recent, contextPath)
	}
	rep.RecentObservation = &recent
	if len(providerTurns) > 0 {
		rep.LiveProvider = fmt.Sprintf("provider-cache window wired; recent snapshot observed %d provider turn(s) at %.2fx multiplier with %.2f%% false-warm; provider action planner live, heartbeat/explicit-cache execution gated",
			len(providerTurns), recent.Multiplier, 100*recent.FalseWarmRate)
	} else {
		rep.LiveProvider = fmt.Sprintf("provider-cache window wired; recent snapshot has no provider-cache telemetry; context status %s with %d event(s); provider action planner waiting on provider turns",
			recent.ContextStatus, recent.ContextEvents)
	}
}

func applyRecentVCacheContextOnlyObservation(rep *vcacheStatusReport, contextPath string) {
	contextPath = strings.TrimSpace(contextPath)
	if contextPath == "" || strings.EqualFold(contextPath, "off") {
		return
	}
	contextPath, readContextSnapshot := resolveVCacheContextSnapshotPath(contextPath)
	if !readContextSnapshot {
		return
	}
	turns, ok, err := vcachesnapshot.Read(contextPath)
	if err != nil {
		rep.RecentObservationError = fmt.Sprintf("%s: %v", contextPath, err)
		return
	}
	if !ok {
		return
	}
	recent := vcacheRecentObservation{
		Source:         "context_snapshot",
		Path:           contextPath,
		Turns:          len(turns),
		ProviderStatus: "MISSING",
		ContextPath:    contextPath,
	}
	accumulateRecentVCacheContext(&recent, turns)
	recent.ContextStatus, recent.ContextReason = recentVCacheContextStatus(recent)
	if recent.ContextStatus != "WITNESSED" {
		return
	}
	recent.ContextReason = "separate context snapshot includes fak_context_* counters from a guard/serve context event"
	rep.RecentObservation = &recent
	rep.LiveProvider = fmt.Sprintf("provider-cache window wired; no provider-cache telemetry found; context status WITNESSED from %s with %d event(s); provider action planner waiting on provider turns",
		contextPath, recent.ContextEvents)
}

func applyRecentVCacheContextSnapshot(recent *vcacheRecentObservation, contextPath string) bool {
	if recent == nil {
		return false
	}
	contextPath = strings.TrimSpace(contextPath)
	if contextPath == "" || strings.EqualFold(contextPath, "off") {
		return false
	}
	resolved, readContextSnapshot := resolveVCacheContextSnapshotPath(contextPath)
	if !readContextSnapshot || resolved == recent.Path {
		return false
	}
	turns, ok, err := vcachesnapshot.Read(resolved)
	if err != nil || !ok {
		return false
	}
	var ctx vcacheRecentObservation
	accumulateRecentVCacheContext(&ctx, turns)
	status, _ := recentVCacheContextStatus(ctx)
	if status != "WITNESSED" {
		return false
	}
	recent.ContextPath = resolved
	recent.ContextEvents = ctx.ContextEvents
	recent.ContextShedTokens = ctx.ContextShedTokens
	recent.ContextDroppedTurns = ctx.ContextDroppedTurns
	recent.ContextBaselineTokens = ctx.ContextBaselineTokens
	recent.ContextCostTokens = ctx.ContextCostTokens
	recent.ContextStatus = "WITNESSED"
	recent.ContextReason = "separate context snapshot includes fak_context_* counters from a guard/serve context event"
	return true
}

func accumulateRecentVCacheContext(recent *vcacheRecentObservation, turns []vcacheobserve.Turn) {
	if recent == nil {
		return
	}
	for _, turn := range turns {
		recent.ContextEvents += turn.ContextEvents
		recent.ContextShedTokens += turn.ContextShedTokens
		recent.ContextDroppedTurns += turn.ContextDroppedTurns
		recent.ContextBaselineTokens += turn.ContextBaselineTokens
		recent.ContextCostTokens += turn.ContextCostTokens
	}
}

func recentVCacheContextStatus(recent vcacheRecentObservation) (string, string) {
	if recent.ContextEvents > 0 || recent.ContextShedTokens > 0 || recent.ContextDroppedTurns > 0 ||
		recent.ContextBaselineTokens > 0 || recent.ContextCostTokens > 0 {
		return "WITNESSED", "snapshot includes fak_context_* counters from a guard/serve context event"
	}
	return "MISSING", "snapshot has provider-cache turns but no fak_context_* counters; it predates context instrumentation or no managed-context event fired"
}
