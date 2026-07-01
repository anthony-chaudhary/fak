package memorystability

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
)

const (
	Schema        = "fleet-memory-stability-governor/1"
	Stable        = "STABLE_IN_BUDGET"
	Frozen        = "DRIFT_FROZEN"
	DefaultTau    = 0.20
	DefaultBudget = 0.50
	DefaultFloor  = 0.60
)

type Stability struct {
	Reliability float64 `json:"reliability"`
	Recency     float64 `json:"recency"`
	Consistency float64 `json:"consistency"`
	Overall     float64 `json:"overall"`
}

type Cycle struct {
	Cycle      int       `json:"cycle"`
	Manifest   string    `json:"manifest"`
	Drift      float64   `json:"drift"`
	Cumulative float64   `json:"cumulative"`
	BelowTau   bool      `json:"below_tau"`
	Stability  Stability `json:"stability"`
	Admitted   bool      `json:"admitted"`
}

type Forget struct {
	Cycle    int     `json:"cycle"`
	Manifest string  `json:"manifest"`
	Overall  float64 `json:"overall"`
	Reason   string  `json:"reason"`
}

type Metrics struct {
	DriftSlope         float64 `json:"drift_slope"`
	CumulativeDrift    float64 `json:"cumulative_drift"`
	InBudgetCumulative float64 `json:"in_budget_cumulative"`
	Budget             float64 `json:"budget"`
	Tau                float64 `json:"tau"`
	Floor              float64 `json:"floor"`
	Frozen             bool    `json:"frozen"`
	FreezeCycle        *int    `json:"freeze_cycle"`
	RollbackTo         *string `json:"rollback_to"`
	SubThresholdBreach bool    `json:"sub_threshold_breach"`
}

type Report struct {
	Schema     string         `json:"schema"`
	Workspace  string         `json:"workspace"`
	Verdict    string         `json:"verdict"`
	OK         bool           `json:"ok"`
	Finding    string         `json:"finding"`
	NextAction string         `json:"next_action"`
	Error      string         `json:"error,omitempty"`
	Baseline   string         `json:"baseline,omitempty"`
	Metrics    Metrics        `json:"metrics,omitempty"`
	Cycles     []Cycle        `json:"cycles"`
	Forget     []Forget       `json:"forget,omitempty"`
	Totals     map[string]int `json:"totals,omitempty"`
}

func Clamp01(x float64) float64 {
	if x < 0 {
		return 0
	}
	if x > 1 {
		return 1
	}
	return x
}

func CycleDrift(rec map[string]any) float64 {
	probes := max(1, asInt(rec["probes"], 1))
	divergence := Clamp01(asFloat(rec["divergence"], 0))
	leak := asInt(rec["stale_reuse"], 0) + asInt(rec["poison_leakage"], 0)
	return Clamp01(divergence + float64(leak)/float64(probes))
}

func StabilityScore(rec map[string]any) Stability {
	probes := max(1, asInt(rec["probes"], 1))
	reliability := Clamp01(1.0 - float64(asInt(rec["poison_leakage"], 0))/float64(probes))
	recency := Clamp01(1.0 - float64(asInt(rec["stale_reuse"], 0))/float64(probes))
	consistency := Clamp01(1.0 - asFloat(rec["divergence"], 0))
	overall := math.Min(reliability, math.Min(recency, consistency))
	return Stability{
		Reliability: round6(reliability),
		Recency:     round6(recency),
		Consistency: round6(consistency),
		Overall:     round6(overall),
	}
}

func DriftSlope(drifts []float64) float64 {
	n := len(drifts)
	if n < 2 {
		return 0
	}
	meanX := float64(n-1) / 2.0
	var sum float64
	for _, d := range drifts {
		sum += d
	}
	meanY := sum / float64(n)
	var num, den float64
	for i, d := range drifts {
		x := float64(i)
		num += (x - meanX) * (d - meanY)
		den += (x - meanX) * (x - meanX)
	}
	if den == 0 {
		return 0
	}
	return num / den
}

func BuildPayload(workspace string, trajectory []map[string]any, baseline string, tau, budget, floor, slopeTol float64, errorText string) Report {
	if errorText != "" {
		return Report{
			Schema:     Schema,
			Workspace:  workspace,
			Verdict:    Frozen,
			OK:         false,
			Finding:    errorText,
			NextAction: "fix the trajectory source",
			Error:      errorText,
			Cycles:     []Cycle{},
		}
	}
	if len(trajectory) == 0 {
		return Report{
			Schema:     Schema,
			Workspace:  workspace,
			Verdict:    Stable,
			OK:         true,
			Finding:    "empty trajectory - nothing to govern",
			NextAction: "freeze a baseline snapshot and replay a cycle",
			Metrics: Metrics{
				DriftSlope:         0,
				CumulativeDrift:    0,
				InBudgetCumulative: 0,
				Budget:             budget,
				Tau:                tau,
				Floor:              floor,
				Frozen:             false,
			},
			Cycles: []Cycle{},
			Forget: []Forget{},
		}
	}
	if baseline == "" {
		baseline = manifest(trajectory[0], 0)
	}
	drifts := make([]float64, len(trajectory))
	for i, rec := range trajectory {
		drifts[i] = CycleDrift(rec)
	}
	var cumulative float64
	var breachCumulative *float64
	var freezeCycle *int
	var rollbackTo *string
	lastInBudget := baseline
	var cycles []Cycle
	var forget []Forget
	frozen := false
	for i, rec := range trajectory {
		d := drifts[i]
		score := StabilityScore(rec)
		mid := manifest(rec, i)
		post := cumulative + d
		scoreOK := score.Overall >= floor
		budgetOK := post <= budget+1e-9
		admit := i == 0 || (scoreOK && budgetOK && !frozen)
		cycles = append(cycles, Cycle{
			Cycle:      i,
			Manifest:   mid,
			Drift:      round6(d),
			Cumulative: round6(post),
			BelowTau:   d < tau,
			Stability:  score,
			Admitted:   admit,
		})
		if i == 0 {
			cumulative = post
			continue
		}
		if !scoreOK {
			forget = append(forget, Forget{Cycle: i, Manifest: mid, Overall: score.Overall, Reason: "stability score below floor"})
		}
		if !frozen && !budgetOK {
			frozen = true
			fc := i
			freezeCycle = &fc
			rt := lastInBudget
			rollbackTo = &rt
			b := post
			breachCumulative = &b
		}
		if admit {
			cumulative = post
			lastInBudget = mid
		}
	}
	slope := DriftSlope(drifts)
	drifting := slope > slopeTol
	reportedCumulative := cumulative
	if breachCumulative != nil {
		reportedCumulative = *breachCumulative
	}
	ok := !frozen && !drifting
	verdict := Stable
	var finding, nextAction string
	if frozen {
		verdict = Frozen
		finding = fmt.Sprintf("cumulative drift %.4g breached budget %.4g at cycle %d; store FROZEN fail-closed", round4(reportedCumulative), budget, *freezeCycle)
		nextAction = fmt.Sprintf("rollback to in-budget snapshot %s; forget low-stability cycles", *rollbackTo)
	} else if drifting {
		finding = fmt.Sprintf("drift slope %.4g > tol %.4g (still in budget) - store is trending toward breach", round4(slope), slopeTol)
		nextAction = "tighten fold loss / forget contradictory items before the next cycle"
	} else {
		finding = fmt.Sprintf("drift slope %.4g flat, cumulative in budget - store stable", round4(slope))
		nextAction = "continue evolving; re-validate each new cycle against the baseline"
	}
	subThresholdBreach := false
	if freezeCycle != nil {
		subThresholdBreach = true
		for _, c := range cycles[1 : *freezeCycle+1] {
			if !c.BelowTau {
				subThresholdBreach = false
				break
			}
		}
	}
	return Report{
		Schema:     Schema,
		Workspace:  workspace,
		Verdict:    verdict,
		OK:         ok,
		Finding:    finding,
		NextAction: nextAction,
		Baseline:   baseline,
		Metrics: Metrics{
			DriftSlope:         round6(slope),
			CumulativeDrift:    round6(reportedCumulative),
			InBudgetCumulative: round6(cumulative),
			Budget:             budget,
			Tau:                tau,
			Floor:              floor,
			Frozen:             frozen,
			FreezeCycle:        freezeCycle,
			RollbackTo:         rollbackTo,
			SubThresholdBreach: subThresholdBreach,
		},
		Cycles: cycles,
		Forget: forget,
		Totals: map[string]int{"cycles": len(trajectory), "forget": len(forget)},
	}
}

func Collect(workspace, trajectoryPath string, tau, budget, floor float64) Report {
	if workspace == "" {
		workspace = "."
	}
	abs, _ := filepath.Abs(workspace)
	if trajectoryPath == "" {
		return BuildPayload(abs, nil, "", tau, budget, floor, 0.05, "")
	}
	data, err := os.ReadFile(trajectoryPath)
	if err != nil {
		return BuildPayload(abs, nil, "", tau, budget, floor, 0.05, fmt.Sprintf("cannot read trajectory %s: %v", trajectoryPath, err))
	}
	var raw any
	if err := json.Unmarshal(data, &raw); err != nil {
		return BuildPayload(abs, nil, "", tau, budget, floor, 0.05, fmt.Sprintf("cannot read trajectory %s: %v", trajectoryPath, err))
	}
	var baseline string
	var trajectory []map[string]any
	switch v := raw.(type) {
	case []any:
		trajectory = toRecords(v)
	case map[string]any:
		if b, ok := v["baseline"].(string); ok {
			baseline = b
		}
		arr, ok := v["trajectory"].([]any)
		if !ok {
			return BuildPayload(abs, nil, "", tau, budget, floor, 0.05, "trajectory must be a JSON list (or {trajectory:[...]})")
		}
		trajectory = toRecords(arr)
	default:
		return BuildPayload(abs, nil, "", tau, budget, floor, 0.05, "trajectory must be a JSON list (or {trajectory:[...]})")
	}
	return BuildPayload(abs, trajectory, baseline, tau, budget, floor, 0.05, "")
}

func Render(report Report) string {
	m := report.Metrics
	lines := []string{
		fmt.Sprintf("memory-stability governor: %s (%s)", report.Verdict, report.Finding),
		fmt.Sprintf("slope=%v  cumulative=%v/budget=%v  frozen=%v", m.DriftSlope, m.CumulativeDrift, m.Budget, m.Frozen),
		"next: " + report.NextAction,
	}
	if m.Frozen {
		rb := ""
		if m.RollbackTo != nil {
			rb = *m.RollbackTo
		}
		fc := -1
		if m.FreezeCycle != nil {
			fc = *m.FreezeCycle
		}
		lines = append(lines, fmt.Sprintf("  ROLLBACK -> %s (freeze at cycle %d)", rb, fc))
		if m.SubThresholdBreach {
			lines = append(lines, "  NOTE: every breaching cycle was BELOW tau - a per-reload gate would have stayed silent (trajectory-only catch).")
		}
	}
	if len(report.Forget) > 0 {
		lines = append(lines, "  FORGET (stability below floor):")
		for i, f := range report.Forget {
			if i >= 20 {
				break
			}
			lines = append(lines, fmt.Sprintf("    cycle %3d %s  overall=%v", f.Cycle, f.Manifest, f.Overall))
		}
	}
	return strings.Join(lines, "\n")
}

func ExitCode(report Report) int {
	if report.Metrics.Frozen || report.Error != "" {
		return 1
	}
	return 0
}

func MarshalJSON(report Report) ([]byte, error) {
	return json.MarshalIndent(report, "", "  ")
}

func manifest(rec map[string]any, i int) string {
	if s, ok := rec["manifest"].(string); ok && s != "" {
		return s
	}
	return fmt.Sprintf("cycle-%d", i)
}

func toRecords(arr []any) []map[string]any {
	out := make([]map[string]any, 0, len(arr))
	for _, item := range arr {
		if m, ok := item.(map[string]any); ok {
			out = append(out, m)
		}
	}
	return out
}

func asFloat(v any, fallback float64) float64 {
	switch x := v.(type) {
	case float64:
		return x
	case float32:
		return float64(x)
	case int:
		return float64(x)
	case int64:
		return float64(x)
	case json.Number:
		f, err := x.Float64()
		if err == nil {
			return f
		}
	case string:
		var f float64
		if _, err := fmt.Sscan(strings.TrimSpace(x), &f); err == nil {
			return f
		}
	}
	return fallback
}

func asInt(v any, fallback int) int {
	switch x := v.(type) {
	case int:
		return x
	case int64:
		return int(x)
	case float64:
		return int(x)
	case json.Number:
		i, err := x.Int64()
		if err == nil {
			return int(i)
		}
	case string:
		var i int
		if _, err := fmt.Sscan(strings.TrimSpace(x), &i); err == nil {
			return i
		}
	}
	return fallback
}

func round6(x float64) float64 { return math.Round(x*1_000_000) / 1_000_000 }
func round4(x float64) float64 { return math.Round(x*10_000) / 10_000 }
