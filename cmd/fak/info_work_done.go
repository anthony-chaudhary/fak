package main

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"math"
	"sort"
	"strings"
	"time"
)

const (
	guardInfoWorkDoneSourceSchema         = "fak.info.work-source/1"
	guardInfoWorkDoneSchema               = "fak.info.work-done/1"
	guardInfoWorkDoneBaselineID           = "direct-provider/v1"
	guardInfoWorkDoneBaselineEffectiveUTC = "2026-08-14"
	guardInfoWorkDoneComparisonScope      = "same observed session; fak-local reuse disabled on the baseline arm"
)

// guardInfoWorkDone is the shared accounting object behind the default TUI and `fak info
// --json`. Units stay separate because token estimates, avoided calls, and latency
// counterfactuals are different claims and cannot be honestly collapsed into one score.
type guardInfoWorkDone struct {
	Schema   string                    `json:"schema"`
	Window   string                    `json:"window"`
	Baseline guardInfoWorkDoneBaseline `json:"baseline"`
	Metrics  guardInfoWorkDoneMetrics  `json:"metrics"`
	Sources  []guardInfoWorkDoneSource `json:"sources,omitempty"`
}

type guardInfoWorkDoneBaseline struct {
	ID                  string `json:"id"`
	Label               string `json:"label"`
	Revision            int    `json:"revision"`
	EffectiveUTC        string `json:"effective_utc"`
	ConfigurationSHA256 string `json:"configuration_sha256"`
	ComparisonScope     string `json:"comparison_scope"`
	CandidateArm        string `json:"candidate_arm"`
	BaselineArm         string `json:"baseline_arm"`
}

type guardInfoWorkDoneMetrics struct {
	InputTokensAvoided guardInfoWorkDoneMetric `json:"input_tokens_avoided"`
	ModelCallsAvoided  guardInfoWorkDoneMetric `json:"model_calls_avoided"`
	WaitSecondsAvoided guardInfoWorkDoneMetric `json:"wait_seconds_avoided"`
}

type guardInfoWorkDoneMetric struct {
	Available         bool    `json:"available"`
	Value             float64 `json:"value,omitempty"`
	IntegerValue      *uint64 `json:"integer_value,omitempty"`
	UnavailableReason string  `json:"unavailable_reason,omitempty"`
	Unit              string  `json:"unit"`
	Evidence          string  `json:"evidence"`
	BaselineID        string  `json:"baseline_id"`
	Basis             string  `json:"basis,omitempty"`
}

type guardInfoWorkDoneSource struct {
	Schema              string  `json:"schema"`
	ID                  string  `json:"id"`
	Owner               string  `json:"owner"`
	Disposition         string  `json:"disposition"`
	Label               string  `json:"label"`
	Events              uint64  `json:"events"`
	EventCountAvailable bool    `json:"event_count_available"`
	InputTokenEquiv     float64 `json:"input_token_equiv"`
	ModelCallsAvoided   uint64  `json:"model_calls_avoided"`
	Evidence            string  `json:"evidence"`
	ExclusivityGroup    string  `json:"exclusivity_group"`
}

func guardInfoWorkDoneFromVars(v guardInfoVars) guardInfoWorkDone {
	w := guardInfoWorkDone{
		Schema:   guardInfoWorkDoneSchema,
		Window:   "observed_session",
		Baseline: guardInfoDirectProviderBaseline(),
		Metrics: guardInfoWorkDoneMetrics{
			InputTokensAvoided: guardInfoWorkDoneMetric{Unit: "input_tokens", Evidence: "unavailable", BaselineID: guardInfoWorkDoneBaselineID, UnavailableReason: "provider_cache_usage_not_reported"},
			ModelCallsAvoided:  guardInfoWorkDoneMetric{Unit: "model_calls", Evidence: "unavailable", BaselineID: guardInfoWorkDoneBaselineID, UnavailableReason: "local_avoidance_counters_not_reported"},
			WaitSecondsAvoided: guardInfoWorkDoneMetric{Unit: "seconds", Evidence: "unavailable", BaselineID: guardInfoWorkDoneBaselineID, UnavailableReason: "observed_turn_latency_not_reported"},
		},
	}
	if v.VCache != nil {
		w.Metrics.InputTokensAvoided = guardInfoWorkDoneMetric{
			Available: true, Value: guardInfoSaved(v), IntegerValue: integerMetricValue(guardInfoSaved(v)), Unit: "input_tokens", Evidence: "observed",
			BaselineID: guardInfoWorkDoneBaselineID, Basis: "provider-reported cache usage; token-equivalent delta against the declared baseline arm",
		}
	}
	if v.CacheAttribution != nil || v.Adjudication != nil {
		w.Metrics.ModelCallsAvoided = guardInfoWorkDoneMetric{
			Available: true, Value: float64(guardInfoTurnsSaved(v)), IntegerValue: ptrUint64(guardInfoTurnsSaved(v)), Unit: "model_calls", Evidence: "witnessed",
			BaselineID: guardInfoWorkDoneBaselineID, Basis: "fak-local engine calls skipped",
		}
	}
	if w.Metrics.ModelCallsAvoided.Available && v.Adjudication != nil {
		if seconds, ok := timeSavedSeconds(w.Metrics.ModelCallsAvoided.Value, *v.Adjudication); ok {
			w.Metrics.WaitSecondsAvoided = guardInfoWorkDoneMetric{
				Available: true, Value: seconds, Unit: "seconds", Evidence: "modeled",
				BaselineID: guardInfoWorkDoneBaselineID, Basis: "avoided calls multiplied by observed current-session mean end-to-end turn latency",
			}
		}
	}
	if a := v.CacheAttribution; a != nil {
		w.Sources = guardInfoWorkDoneSources(*a)
	} else {
		w.Sources = []guardInfoWorkDoneSource{guardInfoUnknownWorkSource()}
	}
	return w
}

func integerMetricValue(v float64) *uint64 {
	if v < 0 || math.IsNaN(v) || math.IsInf(v, 0) || math.Trunc(v) != v || v > math.MaxUint64 {
		return nil
	}
	return ptrUint64(uint64(v))
}

func ptrUint64(v uint64) *uint64 { return &v }

const guardInfoWorkDoneQuerySchema = "fak.info.work-done-query/1"

type guardInfoWorkDoneQuery struct {
	Schema      string              `json:"schema"`
	GeneratedAt string              `json:"generated_at"`
	Window      guardInfoWorkWindow `json:"window"`
	WorkDone    guardInfoWorkDone   `json:"work_done"`
}

type guardInfoWorkWindow struct {
	Kind          string `json:"kind"`
	StartUTC      string `json:"start_utc,omitempty"`
	EndUTC        string `json:"end_utc"`
	DurationNanos int64  `json:"duration_nanos"`
	Reset         bool   `json:"reset"`
	ResetReason   string `json:"reset_reason,omitempty"`
}

func guardInfoSessionWorkDoneQuery(v guardInfoVars, at time.Time) guardInfoWorkDoneQuery {
	return guardInfoWorkDoneQuery{
		Schema: guardInfoWorkDoneQuerySchema, GeneratedAt: at.UTC().Format(time.RFC3339Nano),
		Window:   guardInfoWorkWindow{Kind: "session_total", EndUTC: at.UTC().Format(time.RFC3339Nano), DurationNanos: int64(time.Duration(v.Gateway.UptimeSeconds * float64(time.Second)))},
		WorkDone: guardInfoWorkDoneFromVars(v),
	}
}

func guardInfoBoundedWorkDoneQuery(before, after guardInfoVars, start, end time.Time) guardInfoWorkDoneQuery {
	a, b := guardInfoWorkDoneFromVars(before), guardInfoWorkDoneFromVars(after)
	q := guardInfoWorkDoneQuery{
		Schema: guardInfoWorkDoneQuerySchema, GeneratedAt: end.UTC().Format(time.RFC3339Nano),
		Window:   guardInfoWorkWindow{Kind: "bounded", StartUTC: start.UTC().Format(time.RFC3339Nano), EndUTC: end.UTC().Format(time.RFC3339Nano), DurationNanos: end.Sub(start).Nanoseconds()},
		WorkDone: b,
	}
	q.WorkDone.Window = "bounded"
	if !guardInfoWorkDoneBaselineCompatible(a.Baseline, b.Baseline) {
		q.Window.Reset, q.Window.ResetReason = true, "baseline_changed"
		q.WorkDone = guardInfoUnavailableWindowWorkDone(b, "baseline_changed_during_window")
		return q
	}
	if workDoneCountersRegressed(a, b) {
		q.Window.Reset, q.Window.ResetReason = true, "session_counters_reset"
		q.WorkDone = guardInfoUnavailableWindowWorkDone(b, "session_counters_reset_during_window")
		return q
	}
	q.WorkDone.Metrics.InputTokensAvoided = subtractWorkMetric(a.Metrics.InputTokensAvoided, b.Metrics.InputTokensAvoided)
	q.WorkDone.Metrics.ModelCallsAvoided = subtractWorkMetric(a.Metrics.ModelCallsAvoided, b.Metrics.ModelCallsAvoided)
	q.WorkDone.Metrics.WaitSecondsAvoided = subtractWorkMetric(a.Metrics.WaitSecondsAvoided, b.Metrics.WaitSecondsAvoided)
	q.WorkDone.Sources = subtractWorkSources(a.Sources, b.Sources)
	return q
}

func workDoneCountersRegressed(a, b guardInfoWorkDone) bool {
	for _, pair := range [][2]guardInfoWorkDoneMetric{{a.Metrics.InputTokensAvoided, b.Metrics.InputTokensAvoided}, {a.Metrics.ModelCallsAvoided, b.Metrics.ModelCallsAvoided}, {a.Metrics.WaitSecondsAvoided, b.Metrics.WaitSecondsAvoided}} {
		if pair[0].Available && pair[1].Available && pair[1].Value < pair[0].Value {
			return true
		}
	}
	return false
}

func subtractWorkMetric(a, b guardInfoWorkDoneMetric) guardInfoWorkDoneMetric {
	if !a.Available || !b.Available {
		b.Available, b.Value, b.IntegerValue, b.Evidence = false, 0, nil, "unavailable"
		b.UnavailableReason = "window_endpoint_unavailable"
		return b
	}
	b.Value -= a.Value
	if b.Unit == "model_calls" {
		b.IntegerValue = ptrUint64(uint64(b.Value))
	}
	return b
}

func subtractWorkSources(a, b []guardInfoWorkDoneSource) []guardInfoWorkDoneSource {
	before := map[string]guardInfoWorkDoneSource{}
	for _, source := range a {
		before[source.ID] = source
	}
	out := make([]guardInfoWorkDoneSource, 0, len(b))
	for _, source := range b {
		prior := before[source.ID]
		source.InputTokenEquiv -= prior.InputTokenEquiv
		if source.ModelCallsAvoided < prior.ModelCallsAvoided || (source.EventCountAvailable && prior.EventCountAvailable && source.Events < prior.Events) {
			continue
		}
		source.ModelCallsAvoided -= prior.ModelCallsAvoided
		if source.EventCountAvailable && prior.EventCountAvailable {
			source.Events -= prior.Events
		} else {
			source.EventCountAvailable, source.Events = false, 0
		}
		if source.InputTokenEquiv != 0 || source.ModelCallsAvoided != 0 || source.Events != 0 {
			out = append(out, source)
		}
	}
	return out
}

func guardInfoUnavailableWindowWorkDone(w guardInfoWorkDone, reason string) guardInfoWorkDone {
	metrics := []*guardInfoWorkDoneMetric{&w.Metrics.InputTokensAvoided, &w.Metrics.ModelCallsAvoided, &w.Metrics.WaitSecondsAvoided}
	for _, metric := range metrics {
		metric.Available, metric.Value, metric.IntegerValue, metric.Evidence, metric.UnavailableReason = false, 0, nil, "unavailable", reason
	}
	w.Sources = nil
	return w
}

func guardInfoWorkDoneSources(a guardInfoCacheAttribution) []guardInfoWorkDoneSource {
	const tokenGroup = "input_token_equiv_owner/v1"
	const callGroup = "avoided_model_call_path/v1"
	var out []guardInfoWorkDoneSource
	add := func(source guardInfoWorkDoneSource) {
		source.Schema = guardInfoWorkDoneSourceSchema
		out = append(out, source)
	}
	if a.ProviderPromptCacheReadTokenEquiv > 0 || a.ProviderTokenEquiv > 0 {
		add(guardInfoWorkDoneSource{ID: "provider_cache", Owner: "provider", Disposition: "loaded", Label: "provider prefix cache",
			InputTokenEquiv: a.ProviderTokenEquiv, Evidence: "observed", ExclusivityGroup: tokenGroup})
	}
	if a.FakCompactionShedTokens > 0 {
		add(guardInfoWorkDoneSource{ID: "context_reduction", Owner: "fak", Disposition: "reduced", Label: "context compaction",
			InputTokenEquiv: float64(a.FakCompactionShedTokens), Evidence: "witnessed", ExclusivityGroup: tokenGroup})
	}
	if a.FakKVPrefixReusedTokens > 0 {
		add(guardInfoWorkDoneSource{ID: "fak_prefix_reuse", Owner: "fak", Disposition: "loaded", Label: "fak prefix reuse",
			InputTokenEquiv: float64(a.FakKVPrefixReusedTokens), Evidence: "witnessed", ExclusivityGroup: tokenGroup})
	}
	classifiedFakTokens := float64(a.FakCompactionShedTokens + a.FakKVPrefixReusedTokens)
	if a.FakTokenEquiv > classifiedFakTokens {
		add(guardInfoWorkDoneSource{ID: "unknown", Owner: "fak", Disposition: "unknown", Label: "unclassified context reduction",
			InputTokenEquiv: a.FakTokenEquiv - classifiedFakTokens, Evidence: "unavailable", ExclusivityGroup: tokenGroup})
	}
	if a.FakResponseMemoCalls > 0 {
		add(guardInfoWorkDoneSource{ID: "fak_response_reuse", Owner: "fak", Disposition: "served", Label: "response memo",
			Events: a.FakResponseMemoCalls, EventCountAvailable: true, ModelCallsAvoided: a.FakResponseMemoCalls, Evidence: "witnessed", ExclusivityGroup: callGroup})
	}
	if a.FakInlineServedCalls > 0 {
		add(guardInfoWorkDoneSource{ID: "inline_tool_local", Owner: "fak", Disposition: "served", Label: "inline/tool local",
			Events: a.FakInlineServedCalls, EventCountAvailable: true, ModelCallsAvoided: a.FakInlineServedCalls, Evidence: "witnessed", ExclusivityGroup: callGroup})
	}
	knownCalls := a.FakResponseMemoCalls + a.FakInlineServedCalls
	if a.FakVDSOAvoidedCalls > knownCalls {
		unknown := a.FakVDSOAvoidedCalls - knownCalls
		add(guardInfoWorkDoneSource{ID: "unknown", Owner: "unknown", Disposition: "unknown", Label: "unclassified local reuse",
			Events: unknown, EventCountAvailable: true, ModelCallsAvoided: unknown, Evidence: "unavailable", ExclusivityGroup: callGroup})
	}
	if len(out) == 0 {
		return []guardInfoWorkDoneSource{{Schema: guardInfoWorkDoneSourceSchema, ID: "cold_direct", Owner: "provider", Disposition: "loaded", Label: "cold/direct path", Evidence: "observed", ExclusivityGroup: "none"}}
	}
	rank := map[string]int{"provider_cache": 0, "context_reduction": 1, "fak_prefix_reuse": 2, "fak_response_reuse": 3, "inline_tool_local": 4, "cold_direct": 5, "unknown": 6}
	sort.SliceStable(out, func(i, j int) bool { return rank[out[i].ID] < rank[out[j].ID] })
	return out
}

func guardInfoUnknownWorkSource() guardInfoWorkDoneSource {
	return guardInfoWorkDoneSource{Schema: guardInfoWorkDoneSourceSchema, ID: "unknown", Owner: "unknown", Disposition: "unknown", Label: "source unavailable", Evidence: "unavailable", ExclusivityGroup: "none"}
}

func guardInfoWorkDoneReconciles(w guardInfoWorkDone) bool {
	var tokens float64
	var calls uint64
	for _, source := range w.Sources {
		tokens += source.InputTokenEquiv
		calls += source.ModelCallsAvoided
	}
	if w.Metrics.InputTokensAvoided.Available && tokens != w.Metrics.InputTokensAvoided.Value {
		return false
	}
	if w.Metrics.ModelCallsAvoided.Available && float64(calls) != w.Metrics.ModelCallsAvoided.Value {
		return false
	}
	return true
}

func guardInfoDirectProviderBaseline() guardInfoWorkDoneBaseline {
	b := guardInfoWorkDoneBaseline{
		ID: guardInfoWorkDoneBaselineID, Label: "direct provider path", Revision: 1,
		EffectiveUTC: guardInfoWorkDoneBaselineEffectiveUTC, ComparisonScope: guardInfoWorkDoneComparisonScope,
		CandidateArm: "current session: provider cache and fak-local reuse enabled as configured",
		BaselineArm:  "same provider/session workload: fak-local response reuse and inline serving disabled",
	}
	canonical := strings.Join([]string{b.ID, b.EffectiveUTC, b.ComparisonScope, b.CandidateArm, b.BaselineArm}, "\x00")
	sum := sha256.Sum256([]byte(canonical))
	b.ConfigurationSHA256 = "sha256:" + hex.EncodeToString(sum[:])
	return b
}

func guardInfoWorkDoneBaselineCompatible(a, b guardInfoWorkDoneBaseline) bool {
	return a.ID != "" && a.ID == b.ID && a.ConfigurationSHA256 != "" && a.ConfigurationSHA256 == b.ConfigurationSHA256
}

func guardInfoWorkDoneBaselineDetailRows(w guardInfoWorkDone) []string {
	b := w.Baseline
	return []string{
		fmt.Sprintf(" base   %s · r%d · effective %s", b.ID, b.Revision, b.EffectiveUTC),
		" scope  " + b.ComparisonScope,
		" fak    " + b.CandidateArm,
		" alt    " + b.BaselineArm,
		" config " + b.ConfigurationSHA256,
	}
}

func ptrGuardInfoWorkDone(v guardInfoWorkDone) *guardInfoWorkDone { return &v }

func guardInfoWorkDoneRows(ctx guardInfoPanelCtx, level guardInfoPanelLevel) []string {
	w := guardInfoWorkDoneFromVars(ctx.v)
	metric := func(m guardInfoWorkDoneMetric, available string) string {
		if !m.Available {
			return "unavailable"
		}
		return available
	}
	tokens := metric(w.Metrics.InputTokensAvoided, guardInfoSignedShortCount(w.Metrics.InputTokensAvoided.Value)+" input tok")
	calls := metric(w.Metrics.ModelCallsAvoided, guardInfoShortCount(int(w.Metrics.ModelCallsAvoided.Value))+" model calls")
	seconds := metric(w.Metrics.WaitSecondsAvoided, fmt.Sprintf("%.0fs wait", w.Metrics.WaitSecondsAvoided.Value))
	if level == guardPanelMini {
		return []string{fmt.Sprintf(" work  %s avoided · %s avoided", tokens, calls)}
	}
	rows := []string{
		fmt.Sprintf(" work  vs %s r%d · observed session", w.Baseline.Label, w.Baseline.Revision),
		fmt.Sprintf("       %s avoided · %s avoided · %s avoided", tokens, calls, seconds),
	}
	if source := guardInfoWorkDoneSourceText(w); source != "" {
		rows = append(rows, " from  "+source+" · Cache tab for ablation")
	} else {
		rows = append(rows, " from  source unavailable · Cache tab for ablation")
	}
	if ctx.v.WorkHistory != nil {
		rows = append(rows, guardInfoWorkHistoryRows(*ctx.v.WorkHistory)...)
	}
	return rows
}

func guardInfoSignedShortCount(v float64) string {
	if v < 0 {
		return "-" + guardInfoShortCount(int(-v))
	}
	return "+" + guardInfoShortCount(int(v))
}

func guardInfoWorkDoneSourceText(w guardInfoWorkDone) string {
	var providerTokens, fakTokens float64
	var calls uint64
	for _, source := range w.Sources {
		if source.Owner == "provider" {
			providerTokens += source.InputTokenEquiv
		}
		if source.Owner == "fak" {
			fakTokens += source.InputTokenEquiv
		}
		calls += source.ModelCallsAvoided
	}
	var parts []string
	if providerTokens > 0 {
		parts = append(parts, fmt.Sprintf("provider ~%s tok", guardInfoShortCount(int(providerTokens))))
	}
	if fakTokens > 0 {
		parts = append(parts, fmt.Sprintf("fak ~%s tok", guardInfoShortCount(int(fakTokens))))
	}
	if calls > 0 {
		parts = append(parts, fmt.Sprintf("fak %s calls", guardInfoShortCount(int(calls))))
	}
	if len(parts) == 0 && len(w.Sources) > 0 {
		return w.Sources[0].Label
	}
	return strings.Join(parts, " + ")
}

func guardInfoWorkDoneSourceRows(w guardInfoWorkDone) []string {
	rows := []string{" sources (exclusive within each group; do not add across units):"}
	for _, source := range w.Sources {
		effect := "no measured saving"
		if source.InputTokenEquiv != 0 {
			effect = fmt.Sprintf("~%s input tok", guardInfoShortCount(int(source.InputTokenEquiv)))
		}
		if source.ModelCallsAvoided > 0 {
			effect = fmt.Sprintf("%s model calls", guardInfoShortCount(int(source.ModelCallsAvoided)))
		}
		count := "events unavailable"
		if source.EventCountAvailable {
			count = "events " + guardInfoShortCount(int(source.Events))
		}
		rows = append(rows, fmt.Sprintf("  %-8s from %-21s · %s · %s · %s/%s · group %s",
			source.Disposition, source.Label, count, effect, source.Owner, source.Evidence, source.ExclusivityGroup))
	}
	return rows
}
