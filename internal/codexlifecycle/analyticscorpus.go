// analyticscorpus.go — the corpus half of #4767: fold a whole rollout store
// through AnalyzeRollout and report duration/TTFT percentiles, typed outcome
// totals, ranked outliers, and actionable repeated-cause findings.
//
// SCRUBBING CONTRACT. Everything this report exports is stable-and-scrubbed:
// rollout/turn UUIDs, tool names, closed class/reason tokens, hashed loop
// signatures, and apply_patch file paths. Raw commands and result bodies are never
// exported — they were already dropped at ingestion (see analytics.go).
//
// FINDINGS, NOT REMEDIATION. Repeated causes are emitted as Finding rows with
// stable reason tokens sized for `dos unstick` / issue filing. The report itself
// fixes nothing; its job is to hand the next actor a checkable cause.
package codexlifecycle

import (
	"fmt"
	"os"
	"sort"
	"strings"
	"time"
)

// Percentiles is a nearest-rank summary over one observed distribution (seconds).
type Percentiles struct {
	N   int     `json:"n"`
	P50 float64 `json:"p50"`
	P90 float64 `json:"p90"`
	P95 float64 `json:"p95"`
	P99 float64 `json:"p99"`
	Max float64 `json:"max"`
}

func percentiles(vals []float64) Percentiles {
	p := Percentiles{N: len(vals)}
	if len(vals) == 0 {
		return p
	}
	sort.Float64s(vals)
	rank := func(q float64) float64 {
		i := int(q*float64(len(vals))+0.5) - 1
		if i < 0 {
			i = 0
		}
		if i >= len(vals) {
			i = len(vals) - 1
		}
		return vals[i]
	}
	p.P50, p.P90, p.P95, p.P99 = rank(0.50), rank(0.90), rank(0.95), rank(0.99)
	p.Max = vals[len(vals)-1]
	return p
}

// ReasonRow is one ranked class/reason bucket.
type ReasonRow struct {
	Class  ToolClass `json:"class"`
	Reason string    `json:"reason"`
	Count  int       `json:"count"`
}

// ToolAgg is one per-tool rollup.
type ToolAgg struct {
	Calls    int   `json:"calls"`
	Failures int   `json:"failures"`
	Timeouts int   `json:"timeouts"`
	MS       int64 `json:"ms"`
}

// TaskOutlier is one ranked long task with its typed critical-path attribution.
type TaskOutlier struct {
	Session   string        `json:"session"` // rollout UUID (already opaque)
	TurnID    string        `json:"turn_id"`
	Outcome   Outcome       `json:"outcome"`
	DurationS float64       `json:"duration_s"`
	TTFTS     float64       `json:"ttft_s"`
	IdleS     float64       `json:"idle_s"`
	Top       []Contributor `json:"top,omitempty"`
}

// Finding is one actionable repeated cause, shaped for `dos unstick` / an issue.
type Finding struct {
	Reason string `json:"reason"` // stable token, e.g. repeated_failure:exit_1
	Count  int    `json:"count"`
	Action string `json:"action"`
}

// ResumeCohort measures metadata-grounded fresh headless goal continuations.
type ResumeCohort struct {
	Started           int            `json:"started"`
	UsefulWorkReached int            `json:"useful_work_reached"`
	Completed         int            `json:"completed"`
	Crashed           int            `json:"crashed"`
	Superseded        int            `json:"superseded"`
	FailureReasons    map[string]int `json:"failure_reasons,omitempty"`
}

// AnalyticsCorpus is the whole-store #4767 report.
type AnalyticsCorpus struct {
	Root       string `json:"root"`
	Sessions   int    `json:"sessions"`
	Unreadable int    `json:"unreadable,omitempty"`

	Tasks     int `json:"tasks"`
	Completed int `json:"completed"`
	ToolCalls int `json:"tool_calls"`

	Duration Percentiles `json:"duration"` // seconds, over completed tasks' recorded durations
	TTFT     Percentiles `json:"ttft"`     // seconds, over tasks with an observable first token

	Outcomes map[Outcome]int   `json:"outcomes"`
	Classes  map[ToolClass]int `json:"classes"`
	Reasons  []ReasonRow       `json:"reasons,omitempty"`

	ByTool   map[string]*ToolAgg `json:"by_tool,omitempty"`
	TopTasks []TaskOutlier       `json:"top_tasks,omitempty"`

	TimeoutKills int64 `json:"timeout_kills"`
	SleepPolls   int64 `json:"sleep_polls"`
	StallGaps    int64 `json:"stall_gaps"`

	Findings            []Finding    `json:"findings,omitempty"`
	FreshHeadlessResume ResumeCohort `json:"fresh_headless_resume"`
}

// HardFailureCount counts calls in classes that belong in a failure ranking.
// Expected negatives and control exits are visible in Classes but excluded here —
// that exclusion is the whole point of the typed vocabulary.
func (c AnalyticsCorpus) HardFailureCount() int {
	return c.Classes[ToolFailure] + c.Classes[ToolTimeout]
}

// ScanAnalyticsCorpus folds every rollout under root. Unreadable files are counted,
// never fatal. topN bounds the ranked outlier table.
func ScanAnalyticsCorpus(root string, opt ScanOptions, topN int) (AnalyticsCorpus, error) {
	now := opt.Now
	if now.IsZero() {
		now = time.Now()
	}
	if topN <= 0 {
		topN = 10
	}
	c := AnalyticsCorpus{
		Root:                root,
		Outcomes:            map[Outcome]int{},
		Classes:             map[ToolClass]int{},
		ByTool:              map[string]*ToolAgg{},
		FreshHeadlessResume: ResumeCohort{FailureReasons: map[string]int{}},
	}
	reasons := map[ReasonRow]int{}
	var durations, ttfts []float64
	var outliers []TaskOutlier

	paths, err := rolloutPaths(root, opt.Limit)
	if err != nil {
		return c, err
	}
	for _, p := range paths {
		info, statErr := os.Stat(p)
		if statErr != nil {
			c.Unreadable++
			continue
		}
		fh, openErr := os.Open(p)
		if openErr != nil {
			c.Unreadable++
			continue
		}
		meta, records, parseErr := ReadAnalyticsRollout(fh)
		_ = fh.Close()
		if parseErr != nil {
			c.Unreadable++
			continue
		}
		if opt.CWD != "" && !sameDir(meta.CWD, opt.CWD) {
			continue
		}
		fresh := opt.FreshWithin > 0 && now.Sub(info.ModTime()) <= opt.FreshWithin
		ra := AnalyzeRollout(meta, records, fresh)
		if len(ra.Tasks) == 0 && ra.Calls == 0 {
			continue
		}
		c.Sessions++
		if isFreshHeadlessResume(meta, records) {
			foldResumeCohort(&c.FreshHeadlessResume, ra)
		}
		c.ToolCalls += ra.Calls
		for _, o := range ra.Outcomes {
			c.Classes[o.Class]++
			reasons[ReasonRow{Class: o.Class, Reason: o.Reason}]++
			agg := c.ByTool[o.Tool]
			if agg == nil {
				agg = &ToolAgg{}
				c.ByTool[o.Tool] = agg
			}
			agg.Calls++
			agg.MS += o.SpanMS
			if o.Class == ToolTimeout {
				agg.Timeouts++
			} else if o.Class.CountsAsFailure() {
				agg.Failures++
			}
		}
		c.TimeoutKills += ra.Behavior.TimeoutKills
		c.SleepPolls += ra.Behavior.SleepPolls
		c.StallGaps += ra.Behavior.StallGaps
		for _, t := range ra.Tasks {
			c.Tasks++
			c.Outcomes[t.Outcome]++
			durS := float64(t.RecordedMS) / 1000.0
			if durS <= 0 {
				durS = float64(t.WallMS) / 1000.0
			}
			if t.Outcome == Complete {
				c.Completed++
				if durS > 0 {
					durations = append(durations, durS)
				}
			}
			if t.TTFTMS >= 0 {
				ttfts = append(ttfts, float64(t.TTFTMS)/1000.0)
			}
			if durS > 0 {
				top := t.Critical
				if len(top) > 3 {
					top = top[:3]
				}
				outliers = append(outliers, TaskOutlier{
					Session: meta.RolloutID, TurnID: t.TurnID, Outcome: t.Outcome,
					DurationS: durS, TTFTS: float64(t.TTFTMS) / 1000.0,
					IdleS: float64(t.IdleMS) / 1000.0, Top: top,
				})
			}
		}
	}

	c.Duration = percentiles(durations)
	c.TTFT = percentiles(ttfts)

	for row, n := range reasons {
		row.Count = n
		c.Reasons = append(c.Reasons, row)
	}
	sort.Slice(c.Reasons, func(a, b int) bool {
		if c.Reasons[a].Count != c.Reasons[b].Count {
			return c.Reasons[a].Count > c.Reasons[b].Count
		}
		return c.Reasons[a].Reason < c.Reasons[b].Reason
	})

	sort.Slice(outliers, func(a, b int) bool { return outliers[a].DurationS > outliers[b].DurationS })
	if len(outliers) > topN {
		outliers = outliers[:topN]
	}
	c.TopTasks = outliers

	c.Findings = corpusFindings(c)
	return c, nil
}

// isFreshHeadlessResume uses only structured rollout metadata and the harness-owned
// goal continuation envelope. codex_exec proves a non-interactive headless process;
// the envelope proves this fresh process continues an existing durable goal.
func isFreshHeadlessResume(meta Meta, recs []ARecord) bool {
	if !strings.EqualFold(meta.Originator, "codex_exec") || !strings.EqualFold(meta.Source, "exec") {
		return false
	}
	for _, r := range recs {
		if r.GoalContinuation {
			return true
		}
	}
	return false
}

func foldResumeCohort(c *ResumeCohort, ra RolloutAnalytics) {
	for _, task := range ra.Tasks {
		c.Started++
		useful := false
		for class, n := range task.Classes {
			if n > 0 && (class == ToolOK || class == ToolExpectedNegative) {
				useful = true
				break
			}
		}
		if useful {
			c.UsefulWorkReached++
		}
		switch task.Outcome {
		case Complete:
			c.Completed++
		case Superseded:
			c.Superseded++
		case Live:
			// Started already captures the still-running cohort member; it is not terminal.
		case Aborted, ProcessDeath:
			if task.TrailingEmptyAbort {
				continue
			}
			c.Crashed++
			stage := "before_useful_work"
			if useful {
				stage = "after_useful_work"
			}
			reason := string(task.Outcome)
			c.FailureReasons[stage+":"+reason]++
		}
	}
}

// findingMin is the repetition floor for a corpus-level finding: below it a cause
// is noise, at or above it the cause is a candidate for `dos unstick`.
const findingMin = 25

func corpusFindings(c AnalyticsCorpus) []Finding {
	var out []Finding
	for _, r := range c.Reasons {
		if !r.Class.CountsAsFailure() || r.Count < findingMin {
			continue
		}
		out = append(out, Finding{
			Reason: "repeated_failure:" + r.Reason,
			Count:  r.Count,
			Action: fmt.Sprintf("recurring %s tool outcome (%d calls) — dos unstick / issue candidate", r.Reason, r.Count),
		})
	}
	if c.StallGaps >= findingMin {
		out = append(out, Finding{
			Reason: "repeated_idle_gap",
			Count:  int(c.StallGaps),
			Action: fmt.Sprintf("%d idle gaps over the stall threshold — check harness stalls / human-wait boundaries", c.StallGaps),
		})
	}
	if c.SleepPolls >= findingMin {
		out = append(out, Finding{
			Reason: "foreground_polling",
			Count:  int(c.SleepPolls),
			Action: fmt.Sprintf("%d foreground sleep polls — replace with event-driven waits", c.SleepPolls),
		})
	}
	return out
}
