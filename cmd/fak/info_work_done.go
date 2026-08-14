package main

import (
	"fmt"
	"strings"
)

const (
	guardInfoWorkDoneSchema     = "fak.info.work-done/1"
	guardInfoWorkDoneBaselineID = "direct-provider/v1"
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
	ID    string `json:"id"`
	Label string `json:"label"`
}

type guardInfoWorkDoneMetrics struct {
	InputTokensAvoided guardInfoWorkDoneMetric `json:"input_tokens_avoided"`
	ModelCallsAvoided  guardInfoWorkDoneMetric `json:"model_calls_avoided"`
	WaitSecondsAvoided guardInfoWorkDoneMetric `json:"wait_seconds_avoided"`
}

type guardInfoWorkDoneMetric struct {
	Available bool    `json:"available"`
	Value     float64 `json:"value,omitempty"`
	Unit      string  `json:"unit"`
	Evidence  string  `json:"evidence"`
	Basis     string  `json:"basis,omitempty"`
}

type guardInfoWorkDoneSource struct {
	ID               string  `json:"id"`
	Label            string  `json:"label"`
	InputTokenEquiv  float64 `json:"input_token_equiv"`
	Evidence         string  `json:"evidence"`
	ExclusivityGroup string  `json:"exclusivity_group"`
}

func guardInfoWorkDoneFromVars(v guardInfoVars) guardInfoWorkDone {
	w := guardInfoWorkDone{
		Schema:   guardInfoWorkDoneSchema,
		Window:   "observed_session",
		Baseline: guardInfoWorkDoneBaseline{ID: guardInfoWorkDoneBaselineID, Label: "direct provider path"},
		Metrics: guardInfoWorkDoneMetrics{
			InputTokensAvoided: guardInfoWorkDoneMetric{Unit: "input_tokens", Evidence: "unavailable"},
			ModelCallsAvoided:  guardInfoWorkDoneMetric{Unit: "model_calls", Evidence: "unavailable"},
			WaitSecondsAvoided: guardInfoWorkDoneMetric{Unit: "seconds", Evidence: "unavailable"},
		},
	}
	if v.VCache != nil {
		w.Metrics.InputTokensAvoided = guardInfoWorkDoneMetric{
			Available: true, Value: guardInfoSaved(v), Unit: "input_tokens", Evidence: "estimated",
			Basis: "provider-reported cache usage and configured token prices",
		}
	}
	if v.CacheAttribution != nil || v.Adjudication != nil {
		w.Metrics.ModelCallsAvoided = guardInfoWorkDoneMetric{
			Available: true, Value: float64(guardInfoTurnsSaved(v)), Unit: "model_calls", Evidence: "witnessed",
			Basis: "fak-local engine calls skipped",
		}
	}
	if w.Metrics.ModelCallsAvoided.Available && v.Adjudication != nil {
		if seconds, ok := timeSavedSeconds(w.Metrics.ModelCallsAvoided.Value, *v.Adjudication); ok {
			w.Metrics.WaitSecondsAvoided = guardInfoWorkDoneMetric{
				Available: true, Value: seconds, Unit: "seconds", Evidence: "modeled_from_observed_session_mean",
				Basis: "avoided calls multiplied by observed mean end-to-end turn latency",
			}
		}
	}
	if a := v.CacheAttribution; a != nil {
		if a.ProviderTokenEquiv > 0 {
			w.Sources = append(w.Sources, guardInfoWorkDoneSource{
				ID: "provider_cache", Label: "provider cache", InputTokenEquiv: a.ProviderTokenEquiv,
				Evidence: "observed", ExclusivityGroup: "cache_token_equiv_owner",
			})
		}
		if a.FakTokenEquiv > 0 {
			w.Sources = append(w.Sources, guardInfoWorkDoneSource{
				ID: "fak_reuse", Label: "fak reuse", InputTokenEquiv: a.FakTokenEquiv,
				Evidence: "witnessed", ExclusivityGroup: "cache_token_equiv_owner",
			})
		}
	}
	return w
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
		" work  vs " + w.Baseline.Label + " · observed session",
		fmt.Sprintf("       %s avoided · %s avoided · %s avoided", tokens, calls, seconds),
	}
	if source := guardInfoWorkDoneSourceText(w); source != "" {
		rows = append(rows, " from  "+source+" · Cache tab for ablation")
	} else {
		rows = append(rows, " from  source unavailable · Cache tab for ablation")
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
	var parts []string
	for _, source := range w.Sources {
		parts = append(parts, fmt.Sprintf("%s ~%s tok", source.Label, guardInfoShortCount(int(source.InputTokenEquiv))))
	}
	return strings.Join(parts, " + ")
}
