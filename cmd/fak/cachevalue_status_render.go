package main

import (
	"fmt"
	"io"
	"path/filepath"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/headroom"
	"github.com/anthony-chaudhary/fak/internal/sessionaudit"
)

func renderCachevalueStatus(w io.Writer, rep cachevalueStatusReport) {
	fmt.Fprintf(w, "cachevalue status: %s - %s\n", rep.Verdict, rep.Summary)
	fmt.Fprintf(w, "sources: kernel=%s savings=%s usage=%s\n", rep.Sources.KernelLedger, rep.Sources.SavingsLedger, rep.Sources.UsageLedger)
	if strings.TrimSpace(rep.Sources.ArtifactDir) != "" {
		fmt.Fprintf(w, "artifacts: dir=%s\n", rep.Sources.ArtifactDir)
	}
	fmt.Fprintf(w, "headroom: selected=%s reachable=%v url=%s\n", rep.Headroom.Selected, rep.Headroom.HeadroomReachable, rep.Sources.HeadroomURL)
	fmt.Fprintf(w, "vcache: provider_actions=%s transport=%s recent_provider=%s recent_context=%s\n",
		rep.VCache.ProviderActions, rep.VCache.ProviderActionTransport, cachevalueEmptyDash(rep.VCache.RecentProviderStatus), cachevalueEmptyDash(rep.VCache.RecentContextStatus))
	if rep.Value.RejectedTierAccesses > 0 {
		fmt.Fprintf(w, "value: rejected_tier_accesses=%d\n", rep.Value.RejectedTierAccesses)
	}
	if rep.Session != nil {
		s := rep.Session
		fmt.Fprintf(w, "session: %s status=%s likely=%s turns=%d cache_read=%d cache_create=%d total_context=%d io=%s finding=%s\n",
			s.Session,
			s.Status,
			s.LikelyDomain,
			s.AssistantTurns,
			s.CacheReadTokens,
			s.CacheCreateTokens,
			s.TotalContextTokens,
			fmtFloatPtr(s.IORatio),
			s.Finding,
		)
	}
	if rep.Ablation != nil {
		a := rep.Ablation
		fmt.Fprintf(w, "ablation: status=%s runs=%d dropped=%d diag=%d child_exit=%d workload_mismatch=%d cache_effects=%d active=%d unavailable=%d path=%s\n",
			a.Status, a.Runs, a.DroppedArms, a.DroppedWithDiagnostics, a.DroppedChildExits, a.DroppedWorkloadMismatches,
			a.CacheEffects, a.ActiveEffects, a.UnavailableEffects, a.Path)
	}
	if rep.HeadroomBench != nil {
		h := rep.HeadroomBench
		fmt.Fprintf(w, "headroom bench: status=%s compressor=%s samples=%d saved=%.2f%% path=%s\n",
			h.Status, h.Compressor, h.Samples, h.SavedRatio*100, h.Path)
	}
	if rep.VCacheScore != nil {
		v := rep.VCacheScore
		fmt.Fprintf(w, "vcache score: status=%s grade=%s score=%d active=%s multiplier=%.2fx two_x=%v default=%s activation=%v path=%s\n",
			v.Status, cachevalueEmptyDash(v.Grade), v.Score, cachevalueEmptyDash(v.ActiveSource),
			v.ActiveMultiplier, v.TwoXBetter, cachevalueEmptyDash(v.DefaultUsefulness), v.AgenticActivation, v.Path)
	}
	if rep.VCacheActions != nil {
		a := rep.VCacheActions
		fmt.Fprintf(w, "vcache actions: status=%s families=%d noop=%d ready=%d gated=%d transport=%s ready=%v path=%s\n",
			a.Status, a.FamilyCount, a.Noop, a.Ready, a.Gated, cachevalueEmptyDash(a.TransportMode), a.TransportReady, a.Path)
	}
	if rep.VCacheObserve != nil {
		o := rep.VCacheObserve
		fmt.Fprintf(w, "vcache observe: status=%s domain=%s turns=%d families=%d hit=%.2f%% multiplier=%.2fx saved=%.1f false_warm=%d rate=%.2f%% path=%s\n",
			o.Status, cachevalueEmptyDash(o.FailureDomain), o.Turns, o.FamilyCount, 100*o.HitRate, o.Multiplier,
			o.SavedTokenEquiv, o.FalseWarm, 100*o.FalseWarmRate, o.Path)
	}
	if rep.VCacheContextJoin != nil {
		c := rep.VCacheContextJoin
		fmt.Fprintf(w, "vcache context-join: status=%s domain=%s changes=%d planning=%d provider=%d turns=%d events=%d path=%s\n",
			c.Status, cachevalueEmptyDash(c.FailureDomain), c.TotalChanges, c.PlanningAttributed, c.ProviderAttributed, c.Turns, c.Events, c.Path)
	}
	if rep.VCacheContextWitness != nil {
		c := rep.VCacheContextWitness
		fmt.Fprintf(w, "vcache context-witness: status=%s domain=%s context=%s events=%d shed=%.1f replay=%d score=%d path=%s\n",
			c.Status, cachevalueEmptyDash(c.FailureDomain), cachevalueEmptyDash(c.ContextWitnessed),
			c.ContextEvents, c.ContextShedTokens, c.ReplayExit, c.ScoreExit, c.Path)
	}
	renderCachevalueAttribution(w, rep.Attribution)
	fmt.Fprintln(w, "\ncache-plane rollup:")
	fmt.Fprintf(w, "  %-30s %-42s %-9s %-30s %-11s %-10s %-24s %s\n",
		"plane", "component", "owner", "status", "fidelity", "evidence", "dependency", "impact / next")
	for _, row := range rep.Rows {
		component := row.Component
		if row.Selected {
			component = "*" + component
		}
		fmt.Fprintf(w, "  %-30s %-42s %-9s %-30s %-11s %-10s %-24s %s\n",
			truncStatus(row.Plane, 30),
			truncStatus(component, 42),
			truncStatus(row.Owner, 9),
			truncStatus(row.Status, 30),
			truncStatus(row.Fidelity, 11),
			truncStatus(row.Evidence, 10),
			truncStatus(row.Dependency, 24),
			rowImpactNext(row),
		)
	}
	if len(rep.NextActions) > 0 {
		fmt.Fprintln(w, "\nnext actions:")
		for _, action := range rep.NextActions {
			fmt.Fprintf(w, "- %s\n", action)
		}
	}
}

func renderCachevalueAttribution(w io.Writer, a cachevalueStatusAttribution) {
	fmt.Fprintf(w, "attribution: problem owners=%s\n", cachevalueFormatFindings(a.ProblemOwners))
	fmt.Fprintf(w, "             problem domains=%s\n", cachevalueFormatFindings(a.ProblemDomains))
	fmt.Fprintf(w, "             problem fidelity=%s\n", cachevalueFormatFindings(a.ProblemFidelity))
	fmt.Fprintf(w, "             fidelity=%s evidence=%s\n", cachevalueFormatIntMap(a.Fidelities), cachevalueFormatIntMap(a.Evidence))
}

func cachevalueFormatFindings(findings []cachevalueStatusFinding) string {
	if len(findings) == 0 {
		return "none"
	}
	parts := make([]string, 0, len(findings))
	for _, f := range findings {
		part := fmt.Sprintf("%s=%d", f.Key, f.Problems)
		if strings.TrimSpace(f.Example) != "" {
			part += " [" + f.Example + "]"
		}
		parts = append(parts, part)
	}
	return strings.Join(parts, "; ")
}

func cachevalueFormatIntMap(m map[string]int) string {
	if len(m) == 0 {
		return "none"
	}
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool {
		if m[keys[i]] != m[keys[j]] {
			return m[keys[i]] > m[keys[j]]
		}
		return keys[i] < keys[j]
	})
	parts := make([]string, 0, len(keys))
	for _, k := range keys {
		parts = append(parts, fmt.Sprintf("%s=%d", k, m[k]))
	}
	return strings.Join(parts, ",")
}

func cachevalueComponentDependency(component string) string {
	switch component {
	case "kernel_prefix_reuse":
		return "cachevalue_ledger"
	case "provider_prompt_cache":
		return "provider_usage_ledger"
	case "compaction_shed", "guard_serve_usage_ledger":
		return "gateway_usage_ledger"
	default:
		return "ledger"
	}
}

func cachevalueComponentImpact(component string) string {
	switch component {
	case "kernel_prefix_reuse":
		return "fak-native KV reuse; inspect fak guard/serve cache admission when insufficient"
	case "provider_prompt_cache":
		return "provider-reported cache counters; missing/dollar-blind evidence is provider telemetry or pricing"
	case "compaction_shed":
		return "fak-authored context compaction; lossy token shedding is separate from lossless cache hits"
	case "guard_serve_usage_ledger":
		return "fak gateway counters; missing rows make session-level attribution incomplete"
	default:
		return "cache plane status row"
	}
}

func cachevalueFailureDomain(owner, component string) string {
	switch strings.ToLower(strings.TrimSpace(owner)) {
	case "provider":
		return "provider"
	case "external":
		return "external:" + strings.ToLower(strings.TrimSpace(component))
	case "fak":
		return "fak"
	default:
		return "unknown"
	}
}

func headroomSessionImpact(p headroom.PluginStatus) string {
	if p.Name == headroom.HeadroomName && p.Status == "unavailable" {
		if p.Selected {
			return "selected external sidecar is down; compression passes original bytes through, so bad compression value is external headroom, not fak core"
		}
		return "external sidecar is down but not selected; current sessions do not depend on it"
	}
	if p.Name == headroom.NoopName && p.Selected {
		return "compression is intentionally off; large tool outputs are not a headroom failure"
	}
	if p.Name == headroom.NativeName && p.Selected {
		return "in-process fak compressor is active; compression behavior is fak-owned"
	}
	return "registered compressor capability"
}

func headroomNextAction(p headroom.PluginStatus) string {
	switch {
	case p.Name == headroom.HeadroomName && p.Status == "unavailable":
		return "start headroom proxy or select FAK_COMPRESSOR=native/noop"
	case p.Name == headroom.NoopName && p.Selected:
		return "set FAK_COMPRESSOR=native or FAK_COMPRESSOR=headroom to enable context compression"
	case p.Name == headroom.NativeName && (p.Status == "active" || p.Status == "available"):
		return "fak headroom bench --via native"
	case p.Name == headroom.HeadroomName && (p.Status == "active" || p.Status == "available"):
		return "fak headroom bench --via headroom"
	default:
		return ""
	}
}

func statusReady(v string) string {
	v = strings.TrimSpace(strings.ToLower(v))
	if v == "" {
		return "missing"
	}
	if v == "ready" || v == "proven" {
		return "ready"
	}
	return v
}

func providerActionStatus(v vcacheProviderActionStatus) string {
	if strings.EqualFold(strings.TrimSpace(v.Transport), "decision_only") {
		return "gated"
	}
	return statusReady(v.Verifier)
}

func vcacheRecentProviderStatus(rep vcacheStatusReport) string {
	if rep.RecentObservation == nil {
		return ""
	}
	return rep.RecentObservation.ProviderStatus
}

func vcacheRecentContextStatus(rep vcacheStatusReport) string {
	if rep.RecentObservation == nil {
		return ""
	}
	return rep.RecentObservation.ContextStatus
}

func recentProviderObservationStatus(status string) string {
	if strings.EqualFold(strings.TrimSpace(status), "MISSING") || strings.TrimSpace(status) == "" {
		return "missing"
	}
	return "observed"
}

func recentContextObservationStatus(status string) string {
	if strings.EqualFold(strings.TrimSpace(status), "WITNESSED") {
		return "measured"
	}
	return "missing"
}

func recentProviderObservationReason(recent vcacheRecentObservation) string {
	if strings.EqualFold(strings.TrimSpace(recent.ProviderStatus), "MISSING") {
		return fmt.Sprintf("%d turn(s), no provider-cache telemetry in snapshot", recent.Turns)
	}
	return fmt.Sprintf("%d turn(s), provider %s, multiplier %.2fx, saved %.1f token-equiv, false-warm %.2f%%",
		recent.Turns, recent.ProviderStatus, recent.Multiplier, recent.SavedTokenEquiv, 100*recent.FalseWarmRate)
}

func cachevalueStatusCounts(rows []cachevalueStatusRow) map[string]int {
	counts := map[string]int{}
	for _, row := range rows {
		counts[row.Status]++
	}
	return counts
}

func cachevalueStatusAttributionFromRows(rows []cachevalueStatusRow) cachevalueStatusAttribution {
	owners := map[string]cachevalueStatusBucket{}
	domains := map[string]cachevalueStatusBucket{}
	fidelityBuckets := map[string]cachevalueStatusBucket{}
	fidelities := map[string]int{}
	evidence := map[string]int{}
	ownerExamples := map[string]cachevalueStatusRow{}
	domainExamples := map[string]cachevalueStatusRow{}
	fidelityExamples := map[string]cachevalueStatusRow{}

	for _, row := range rows {
		owner := cachevalueAttributionKey(row.Owner, "unknown")
		domain := cachevalueAttributionKey(row.FailureDomain, "unknown")
		fidelity := cachevalueAttributionKey(row.Fidelity, "unknown")
		ev := cachevalueAttributionKey(row.Evidence, "unknown")
		working := cachevalueRowWorking(row)
		problem := cachevalueRowProblem(row)

		cachevalueBumpBucket(owners, owner, working, problem)
		cachevalueBumpBucket(domains, domain, working, problem)
		cachevalueBumpBucket(fidelityBuckets, fidelity, working, problem)
		fidelities[fidelity]++
		evidence[ev]++
		if problem {
			cachevalueRememberExample(ownerExamples, owner, row)
			cachevalueRememberExample(domainExamples, domain, row)
			cachevalueRememberExample(fidelityExamples, fidelity, row)
		}
	}

	return cachevalueStatusAttribution{
		Owners:          owners,
		Fidelities:      fidelities,
		Evidence:        evidence,
		FailureDomains:  domains,
		ProblemOwners:   cachevalueProblemFindings(owners, ownerExamples),
		ProblemDomains:  cachevalueProblemFindings(domains, domainExamples),
		ProblemFidelity: cachevalueProblemFindings(fidelityBuckets, fidelityExamples),
	}
}

func cachevalueBumpBucket(m map[string]cachevalueStatusBucket, key string, working, problem bool) {
	b := m[key]
	b.Total++
	if working {
		b.Working++
	}
	if problem {
		b.Problem++
	}
	m[key] = b
}

func cachevalueRememberExample(m map[string]cachevalueStatusRow, key string, row cachevalueStatusRow) {
	if _, ok := m[key]; ok {
		return
	}
	m[key] = row
}

func cachevalueProblemFindings(buckets map[string]cachevalueStatusBucket, examples map[string]cachevalueStatusRow) []cachevalueStatusFinding {
	keys := make([]string, 0, len(buckets))
	for key, b := range buckets {
		if b.Problem > 0 {
			keys = append(keys, key)
		}
	}
	sort.Slice(keys, func(i, j int) bool {
		a, b := buckets[keys[i]], buckets[keys[j]]
		if a.Problem != b.Problem {
			return a.Problem > b.Problem
		}
		if a.Working != b.Working {
			return a.Working > b.Working
		}
		return keys[i] < keys[j]
	})
	if len(keys) > 8 {
		keys = keys[:8]
	}
	out := make([]cachevalueStatusFinding, 0, len(keys))
	for _, key := range keys {
		b := buckets[key]
		row := examples[key]
		out = append(out, cachevalueStatusFinding{
			Key:        key,
			Problems:   b.Problem,
			Working:    b.Working,
			Example:    cachevalueFindingExample(row),
			NextAction: row.NextAction,
		})
	}
	return out
}

func cachevalueFindingExample(row cachevalueStatusRow) string {
	if row.Component == "" {
		return ""
	}
	detail := strings.TrimSpace(row.Reason)
	if detail == "" {
		detail = strings.TrimSpace(row.SessionImpact)
	}
	if detail == "" {
		return row.Component + " status=" + row.Status
	}
	return row.Component + " status=" + row.Status + ": " + truncStatus(detail, 180)
}

func cachevalueAttributionKey(v, fallback string) string {
	v = strings.TrimSpace(v)
	if v == "" {
		return fallback
	}
	return v
}

func cachevalueStatusVerdict(rows []cachevalueStatusRow) (string, string) {
	working, problems := 0, 0
	for _, row := range rows {
		if cachevalueRowWorking(row) {
			working++
		}
		if cachevalueRowProblem(row) {
			problems++
		}
	}
	if problems == 0 {
		return "OK", fmt.Sprintf("%d cache component(s) report working or intentionally inactive", working)
	}
	if working == 0 {
		return "INSUFFICIENT", fmt.Sprintf("%d cache component(s) lack evidence or are unavailable", problems)
	}
	return "PARTIAL", fmt.Sprintf("%d working/available component(s), %d component(s) missing, gated, unavailable, or dollar-blind", working, problems)
}

func cachevalueVerdictRank(v string) (int, bool) {
	switch strings.ToUpper(strings.TrimSpace(v)) {
	case "OK":
		return 0, true
	case "PARTIAL":
		return 1, true
	case "INSUFFICIENT":
		return 2, true
	}
	return 0, false
}

// cachevalueStatusGateExit maps the folded verdict onto the opt-in gate exit:
// 1 when the verdict is at or worse than the caller-chosen floor, else 0. An
// unrecognized verdict or floor never gates (the floor is validated at flag
// parse; the verdict set is closed by cachevalueStatusVerdict).
func cachevalueStatusGateExit(verdict, floor string) int {
	floorRank, ok := cachevalueVerdictRank(floor)
	if !ok {
		return 0
	}
	rank, ok := cachevalueVerdictRank(verdict)
	if !ok || rank < floorRank {
		return 0
	}
	return 1
}

func cachevalueRowWorking(row cachevalueStatusRow) bool {
	switch row.Status {
	case "measured", "active", "available", "ready", "observed", "no-op", "saved", "no_effect", "forecast":
		return true
	default:
		return false
	}
}

func cachevalueRowProblem(row cachevalueStatusRow) bool {
	switch row.Status {
	case "missing", "insufficient", "dollar_blind", "gated", "partial", "dropped", "no_saving",
		"cold_write_only", "high_pressure", "no_provider_cache_evidence",
		"no_usage", "not_observed_from_transcript", "forecast_only", "not_2x", "not_ready", "error",
		"context_planning", "provider_cache_behavior", "unattributed", "false_warm", "turns_reordered", "not_observed",
		"replay_failed", "score_failed":
		return true
	case "unavailable":
		return row.Selected || row.Component == "recent_vcache_snapshot" ||
			row.Component == "ablation_report" || row.Component == "vcache_score_report" ||
			row.Component == "vcache_context_join_report" || row.Component == "vcache_observe_report" ||
			row.Component == "vcache_context_witness_report" ||
			strings.HasPrefix(row.Component, "headroom_bench")
	default:
		return false
	}
}

func cachevalueStatusNextActions(rows []cachevalueStatusRow) []string {
	seen := map[string]bool{}
	var actions []string
	for _, row := range rows {
		if !cachevalueRowProblem(row) || strings.TrimSpace(row.NextAction) == "" {
			continue
		}
		action := row.Component + ": " + row.NextAction
		if seen[action] {
			continue
		}
		seen[action] = true
		actions = append(actions, action)
	}
	sort.Strings(actions)
	if len(actions) > 8 {
		return actions[:8]
	}
	return actions
}

func cachevalueNonEmpty(v, fallback string) string {
	if strings.TrimSpace(v) == "" {
		return fallback
	}
	return strings.TrimSpace(v)
}

func rowImpactNext(row cachevalueStatusRow) string {
	impact := row.SessionImpact
	if cachevalueShowRowReason(row) && strings.TrimSpace(row.Reason) != "" {
		impact += " reason=" + row.Reason
	}
	if strings.TrimSpace(row.NextAction) == "" {
		return impact
	}
	return impact + " next=" + row.NextAction
}

func cachevalueShowRowReason(row cachevalueStatusRow) bool {
	if row.Component == "vcache_context_witness_report" || row.Component == "vcache_context_witness_plane" {
		return true
	}
	switch row.Status {
	case "dropped", "unavailable", "error", "no_saving", "not_2x", "forecast_only", "gated",
		"context_planning", "provider_cache_behavior", "unattributed", "false_warm", "turns_reordered", "not_observed",
		"replay_failed", "score_failed":
		return true
	default:
		return false
	}
}

func truncStatus(s string, n int) string {
	if len(s) <= n {
		return s
	}
	if n <= 1 {
		return s[:n]
	}
	return s[:n-1] + "~"
}

func cachevalueEmptyDash(s string) string {
	if strings.TrimSpace(s) == "" {
		return "-"
	}
	return s
}

func nonEmptySessionName(s sessionaudit.Session) string {
	if strings.TrimSpace(s.Session) != "" {
		return s.Session
	}
	if strings.TrimSpace(s.Path) != "" {
		base := filepath.Base(s.Path)
		return strings.TrimSuffix(base, filepath.Ext(base))
	}
	return "unknown_session"
}

func quotePathForHint(path string) string {
	path = strings.TrimSpace(path)
	if path == "" {
		return "FILE"
	}
	if strings.ContainsAny(path, " \t\"'") {
		return fmt.Sprintf("%q", path)
	}
	return path
}

func fmtFloatPtr(v *float64) string {
	if v == nil {
		return "-"
	}
	return fmt.Sprintf("%.3f", *v)
}
