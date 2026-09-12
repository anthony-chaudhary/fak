package agentbench

import (
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agentbench/profileplan"
)

type normalMetric struct {
	Known          bool   `json:"known"`
	Passed         bool   `json:"passed"`
	ObservedMillis int64  `json:"observed_ms,omitempty"`
	LimitMillis    int64  `json:"limit_ms"`
	Reason         string `json:"reason,omitempty"`
}

type normalSLOResult struct {
	ColdFirstOutput    normalMetric `json:"cold_first_output"`
	ColdComplete       normalMetric `json:"cold_complete"`
	NewAreaFirstOutput normalMetric `json:"new_area_first_output"`
	WarmTTFTP50        normalMetric `json:"warm_ttft_p50"`
	WarmTTFTP90        normalMetric `json:"warm_ttft_p90"`
	ShortDecisionP90   normalMetric `json:"short_decision_p90"`
	MediumCompleteP90  normalMetric `json:"medium_complete_p90"`
	LongCompleteP90    normalMetric `json:"long_complete_p90"`
	ProgressGapMaximum normalMetric `json:"progress_gap_maximum"`
	WarmComplete       normalMetric `json:"warm_complete"`
}

type normalCellResult struct {
	Name        string `json:"name"`
	Phase       string `json:"phase"`
	Concurrency int    `json:"concurrency"`
	Expected    int    `json:"expected"`
	Terminal    int    `json:"terminal"`
	Completed   int    `json:"completed"`
	Complete    bool   `json:"complete"`
	Passing     bool   `json:"passing"`
	Reason      string `json:"reason,omitempty"`
}

type normalQualification struct {
	Qualified bool               `json:"qualified"`
	Reason    string             `json:"reason"`
	Unknowns  []string           `json:"unknowns,omitempty"`
	Cells     []normalCellResult `json:"cells"`
	SLOs      normalSLOResult    `json:"slos"`
}

func summarizeNormalQualification(plan profileplan.Plan, cells []normalCell, events []lifecycleEvent, selected int) normalQualification {
	q := normalQualification{Reason: "normal qualification incomplete"}
	terminal := map[string][]lifecycleEvent{}
	for _, e := range events {
		if e.Event == "end" || e.Event == "request_terminal" || e.EventType == "terminal" || e.EventType == "end" {
			terminal[e.Rung] = append(terminal[e.Rung], e)
		}
		if hardLifecycleFailure(e) {
			q.Reason = fmt.Sprintf("hard failure in %s: %s", e.Rung, e.Status)
			q.Cells = classifyNormalCells(cells, terminal)
			return q
		}
	}
	q.Cells = classifyNormalCells(cells, terminal)
	for _, cell := range q.Cells {
		if !cell.Complete || (cell.Phase != "probe" && !cell.Passing) {
			q.Reason = cell.Reason
			return q
		}
	}
	steadyFound := false
	for _, cell := range cells {
		if cell.Phase == "steady" {
			steadyFound = true
			if cell.Concurrency != selected {
				q.Reason = fmt.Sprintf("steady ran at C%d, selected capacity is C%d", cell.Concurrency, selected)
				return q
			}
		}
	}
	if !steadyFound {
		q.Reason = "selected steady cell is absent"
		return q
	}
	q.SLOs = deriveNormalSLOs(events)
	for name, metric := range normalMetrics(q.SLOs) {
		if !metric.Known {
			q.Unknowns = append(q.Unknowns, name+": "+metric.Reason)
		} else if !metric.Passed {
			q.Reason = name + " exceeded its SLO"
			return q
		}
	}
	if len(q.Unknowns) > 0 {
		sort.Strings(q.Unknowns)
		q.Reason = "required SLO metrics are unknown"
		return q
	}
	q.Qualified = true
	q.Reason = "all complete cells and required observed SLOs passed"
	return q
}

func classifyNormalCells(cells []normalCell, terminal map[string][]lifecycleEvent) []normalCellResult {
	result := make([]normalCellResult, 0, len(cells))
	for _, cell := range cells {
		row := normalCellResult{Name: cell.Name, Phase: cell.Phase, Concurrency: cell.Concurrency, Expected: cell.Sessions * cell.TurnsPerSession}
		for _, e := range terminal[cell.Name] {
			row.Terminal++
			if e.Status == "completed" && e.ModelIdentityStatus == "observed" && e.UsageKnown != nil && *e.UsageKnown {
				row.Completed++
			}
		}
		row.Complete = row.Terminal == row.Expected
		row.Passing = row.Complete && row.Completed == row.Expected
		if !row.Complete {
			row.Reason = fmt.Sprintf("cell %s terminal coverage %d/%d", cell.Name, row.Terminal, row.Expected)
		} else if !row.Passing {
			row.Reason = fmt.Sprintf("cell %s has unqualified terminals", cell.Name)
		}
		result = append(result, row)
	}
	return result
}

func hardLifecycleFailure(e lifecycleEvent) bool {
	s := strings.ToLower(e.Status + " " + e.Error)
	return strings.Contains(s, "oom") || strings.Contains(s, "crash") || strings.Contains(s, "corrupt") || strings.Contains(s, "safety") || e.Status == "canceled"
}

func deriveNormalSLOs(events []lifecycleEvent) normalSLOResult {
	first := map[int][]time.Time{}
	tools := map[int][]time.Time{}
	terminals := map[string][]lifecycleEvent{}
	all := map[string][]time.Time{}
	for _, e := range events {
		at, ok := eventTime(e)
		if ok {
			all[e.RequestID] = append(all[e.RequestID], at)
		}
		if e.Event == "first_output" || e.EventType == "first_output" {
			first[e.Sequence] = append(first[e.Sequence], at)
		}
		if e.Event == "tool" || e.EventType == "tool_call" {
			tools[e.Sequence] = append(tools[e.Sequence], at)
		}
		if e.Event == "end" || e.EventType == "terminal" {
			terminals[e.Rung] = append(terminals[e.Rung], e)
		}
	}
	probe := terminals["probe-c1"]
	coldFirst := latencyFor(probe, first, func(e lifecycleEvent) bool { return e.Turn == 1 })
	coldDone := durations(probe, func(e lifecycleEvent) bool { return e.Turn == 1 })
	steady := terminals["steady"]
	warmTTFT := latencyFor(steady, first, func(e lifecycleEvent) bool { return e.Turn > 1 })
	short := decisionLatencies(steady, tools, func(e lifecycleEvent) bool { return e.Turn >= 2 && e.Turn <= 8 })
	medium := durations(steady, func(e lifecycleEvent) bool { return e.Turn >= 9 && e.Turn <= 11 })
	long := durations(steady, func(e lifecycleEvent) bool { return e.Turn == 12 })
	newArea := latencyFor(terminals["new-area-system-warm"], first, func(e lifecycleEvent) bool { return e.Turn == 2 })
	gaps, gapRequests := warmProgressGaps(events, steady)
	gapMetric := metricMax(gaps, 10000)
	if gapRequests < 88 {
		gapMetric = normalMetric{LimitMillis: 10000, Reason: fmt.Sprintf("observed progress for %d/88 warm requests", gapRequests)}
	}
	return normalSLOResult{ColdFirstOutput: metricMaxExpected(coldFirst, 8, 60000), ColdComplete: metricMaxExpected(coldDone, 8, 120000), NewAreaFirstOutput: metricMaxExpected(newArea, 1, 30000), WarmTTFTP50: metricPercentileExpected(warmTTFT, 88, 50, 2000), WarmTTFTP90: metricPercentileExpected(warmTTFT, 88, 90, 5000), ShortDecisionP90: metricPercentileExpected(short, 56, 90, 10000), MediumCompleteP90: metricPercentileExpected(medium, 24, 90, 20000), LongCompleteP90: metricPercentileExpected(long, 8, 90, 60000), ProgressGapMaximum: gapMetric, WarmComplete: metricMaxExpected(durations(steady, func(e lifecycleEvent) bool { return e.Turn > 1 }), 88, 90000)}
}

func eventTime(e lifecycleEvent) (time.Time, bool) {
	raw := e.StartedAt
	if e.Event == "end" || e.EventType == "terminal" {
		raw = e.CompletedAt
	}
	t, err := time.Parse(time.RFC3339Nano, raw)
	return t, err == nil
}
func latencyFor(es []lifecycleEvent, first map[int][]time.Time, keep func(lifecycleEvent) bool) []int64 {
	var v []int64
	for _, e := range es {
		if !keep(e) {
			continue
		}
		start, err := time.Parse(time.RFC3339Nano, e.StartedAt)
		if err != nil || len(first[e.Sequence]) == 0 {
			continue
		}
		v = append(v, first[e.Sequence][0].Sub(start).Milliseconds())
	}
	return v
}
func durations(es []lifecycleEvent, keep func(lifecycleEvent) bool) []int64 {
	var v []int64
	for _, e := range es {
		if keep(e) && e.DurationMilliseconds >= 0 {
			v = append(v, e.DurationMilliseconds)
		}
	}
	return v
}
func progressGaps(bySession map[string][]time.Time) []int64 {
	var v []int64
	for _, ts := range bySession {
		sort.Slice(ts, func(i, j int) bool { return ts[i].Before(ts[j]) })
		for i := 1; i < len(ts); i++ {
			v = append(v, ts[i].Sub(ts[i-1]).Milliseconds())
		}
	}
	return v
}
func decisionLatencies(es []lifecycleEvent, tools map[int][]time.Time, keep func(lifecycleEvent) bool) []int64 {
	var v []int64
	for _, e := range es {
		if !keep(e) {
			continue
		}
		start, err := time.Parse(time.RFC3339Nano, e.StartedAt)
		if err != nil {
			continue
		}
		end, err := time.Parse(time.RFC3339Nano, e.CompletedAt)
		if err != nil {
			continue
		}
		if e.FinishReason == "tool_calls" {
			if len(tools[e.Sequence]) == 0 {
				continue
			}
			end = tools[e.Sequence][0]
		}
		v = append(v, end.Sub(start).Milliseconds())
	}
	return v
}
func warmProgressGaps(events, terminals []lifecycleEvent) ([]int64, int) {
	warm := map[int]bool{}
	for _, e := range terminals {
		if e.Turn > 1 {
			warm[e.Sequence] = true
		}
	}
	points := map[int][]time.Time{}
	for _, e := range events {
		if !warm[e.Sequence] {
			continue
		}
		if e.Event != "progress" && e.Event != "first_output" && e.Event != "end" && e.EventType != "terminal" {
			continue
		}
		if at, ok := eventTime(e); ok {
			points[e.Sequence] = append(points[e.Sequence], at)
		}
	}
	var gaps []int64
	complete := 0
	for _, ts := range points {
		if len(ts) < 2 {
			continue
		}
		complete++
		sort.Slice(ts, func(i, j int) bool { return ts[i].Before(ts[j]) })
		for i := 1; i < len(ts); i++ {
			gaps = append(gaps, ts[i].Sub(ts[i-1]).Milliseconds())
		}
	}
	return gaps, complete
}
func metricMaxExpected(v []int64, expected int, limit int64) normalMetric {
	if expected <= 0 || len(v) < expected {
		return normalMetric{LimitMillis: limit, Reason: fmt.Sprintf("observed %d/%d required samples", len(v), expected)}
	}
	return metricMax(v, limit)
}
func metricPercentileExpected(v []int64, expected, p int, limit int64) normalMetric {
	if expected <= 0 || len(v) < expected {
		return normalMetric{LimitMillis: limit, Reason: fmt.Sprintf("observed %d/%d required samples", len(v), expected)}
	}
	return metricPercentile(v, p, limit)
}
func metricMax(v []int64, limit int64) normalMetric {
	if len(v) == 0 {
		return normalMetric{LimitMillis: limit, Reason: "no observed samples"}
	}
	sort.Slice(v, func(i, j int) bool { return v[i] < v[j] })
	x := v[len(v)-1]
	return normalMetric{Known: true, Passed: x <= limit, ObservedMillis: x, LimitMillis: limit}
}
func metricPercentile(v []int64, p int, limit int64) normalMetric {
	if len(v) == 0 {
		return normalMetric{LimitMillis: limit, Reason: "no observed samples"}
	}
	sort.Slice(v, func(i, j int) bool { return v[i] < v[j] })
	i := (len(v)*p+99)/100 - 1
	if i < 0 {
		i = 0
	}
	x := v[i]
	return normalMetric{Known: true, Passed: x <= limit, ObservedMillis: x, LimitMillis: limit}
}
func normalMetrics(s normalSLOResult) map[string]normalMetric {
	return map[string]normalMetric{"cold first output": s.ColdFirstOutput, "cold complete": s.ColdComplete, "new area first output": s.NewAreaFirstOutput, "warm TTFT p50": s.WarmTTFTP50, "warm TTFT p90": s.WarmTTFTP90, "short decision p90": s.ShortDecisionP90, "medium complete p90": s.MediumCompleteP90, "long complete p90": s.LongCompleteP90, "progress gap maximum": s.ProgressGapMaximum, "warm complete": s.WarmComplete}
}
