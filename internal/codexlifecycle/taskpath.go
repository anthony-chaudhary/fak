// taskpath.go — per-task critical-path decomposition and the #2365 behavioral
// detectors ported to native Codex event shape (#4767).
//
// THE DECOMPOSITION. Every reconciled task's wall time is split into four typed
// buckets that sum to the observable timeline:
//
//	tool  — spans from a function_call to its own function_call_output
//	wait  — the same spans when the command is a registered blocking wait
//	model — inter-event gaps that END at a model-emitted record, up to the stall
//	        threshold ("model-active time where observable" — the model was the
//	        only thing that could have been running)
//	idle  — the remainder of any gap beyond the stall threshold: nobody was
//	        demonstrably working (harness stall, human/peer wait, scheduler pause)
//
// The split is deliberately conservative: it never claims model activity beyond
// the stall threshold, and it attributes a gap ending at an output to the tool
// that produced it. Top critical-path contributors are ranked from these buckets,
// so "an 8.9-hour task" decomposes into typed causes instead of one opaque number.
package codexlifecycle

import (
	"sort"
	"time"
)

// Detector thresholds — the #2365 values, kept identical so the two lenses agree.
const (
	repeatFailureMinCodex = 3       // same tool+head signature failing this often is a stuck retry loop
	fileChurnMinCodex     = 5       // a file patched this often may be a rewrite/flip-flop loop
	successRepeatMinCodex = 8       // identical successful calls this often is a poll loop / storm
	stallGapMS            = 300_000 // a gap this long with zero records is idle/stall, not model work
)

// Contributor is one ranked critical-path bucket ("model", "idle", "wait",
// "tool:<name>") with its attributed milliseconds.
type Contributor struct {
	Category string `json:"category"`
	MS       int64  `json:"ms"`
}

// CallOutcome is one typed, scrubbed call row: class + reason + span, no command.
type CallOutcome struct {
	TurnID     string     `json:"turn_id,omitempty"`
	Tool       string     `json:"tool"`
	Class      ToolClass  `json:"class"`
	Reason     string     `json:"reason"`
	Confidence Confidence `json:"confidence"`
	SpanMS     int64      `json:"span_ms,omitempty"`
	Sig        string     `json:"sig,omitempty"` // hashed tool+head signature
}

// TaskAnalytics is one reconciled task with its critical-path decomposition.
type TaskAnalytics struct {
	TurnID     string     `json:"turn_id"`
	Outcome    Outcome    `json:"outcome"`
	Provenance Provenance `json:"provenance"`

	WallMS     int64 `json:"wall_ms,omitempty"`     // start → typed end (0 for live)
	RecordedMS int64 `json:"recorded_ms,omitempty"` // producer-recorded duration_ms when observed
	TTFTMS     int64 `json:"ttft_ms"`               // start → first model-emitted record; -1 unobserved

	ToolMS  int64 `json:"tool_ms"`
	WaitMS  int64 `json:"wait_ms"`
	ModelMS int64 `json:"model_ms"`
	IdleMS  int64 `json:"idle_ms"`

	ToolCalls   int `json:"tool_calls"`
	Compactions int `json:"compactions,omitempty"`
	IdleGaps    int `json:"idle_gaps,omitempty"`

	Classes  map[ToolClass]int `json:"classes,omitempty"`
	Critical []Contributor     `json:"critical,omitempty"`

	TrailingEmptyAbort bool `json:"trailing_empty_abort,omitempty"`
}

// CodexBehavior is the #2365 behavioral lens ported to Codex event shape. All rows
// are scrubbed: signatures are hashes, churn rows carry file paths only.
type CodexBehavior struct {
	TimeoutKills int64   `json:"timeout_kills"`
	SleepPolls   int64   `json:"sleep_polls"`
	StallGaps    int64   `json:"stall_gaps"`
	MaxGapS      float64 `json:"max_gap_s"`

	RepeatFailures   []SigRow   `json:"repeat_failures,omitempty"`
	MaxRepeatFailure int64      `json:"max_repeat_failure"`
	SuccessLoops     []SigRow   `json:"success_loops,omitempty"`
	MaxSuccessLoop   int64      `json:"max_success_loop"`
	EditChurn        []ChurnRow `json:"edit_churn,omitempty"`
	MaxEditChurn     int64      `json:"max_edit_churn"`
}

// SigRow is one repeat-failure / success-loop offender, keyed by hashed signature.
type SigRow struct {
	Tool  string `json:"tool"`
	Sig   string `json:"sig"`
	Count int64  `json:"count"`
}

// ChurnRow is one per-file patch-churn offender.
type ChurnRow struct {
	File  string `json:"file"`
	Count int64  `json:"count"`
}

// RolloutAnalytics is the full per-rollout report.
type RolloutAnalytics struct {
	Meta                       Meta            `json:"meta"`
	Tasks                      []TaskAnalytics `json:"tasks"`
	Calls                      int             `json:"calls"`
	Outcomes                   []CallOutcome   `json:"outcomes,omitempty"`
	Behavior                   CodexBehavior   `json:"behavior"`
	SubstantiveCompleted       bool            `json:"substantive_completed,omitempty"`
	CompletedWithTrailingAbort bool            `json:"completed_with_trailing_abort,omitempty"`
}

// AnalyzeRollout joins calls/results/tasks by their ids, types every outcome, and
// decomposes each reconciled task's wall time. fresh has the same meaning as in
// Fold: it decides live vs process-death for the final open start, which in turn
// decides live_tail vs interrupted for calls missing their result at the tail.
func AnalyzeRollout(meta Meta, records []ARecord, fresh bool) RolloutAnalytics {
	// 1. Lifecycle: reuse the #4785 exactly-once fold for typed task boundaries.
	var events []Event
	for _, r := range records {
		switch r.Kind {
		case KindStarted, KindComplete, KindAborted:
			events = append(events, Event{
				Kind: r.Kind, TurnID: r.TurnID,
				Timestamp: r.TS.UTC().Format(time.RFC3339Nano),
				Reason:    r.Reason, DurationMS: r.DurationMS,
			})
		}
	}
	rep := Fold(events, fresh)
	out := RolloutAnalytics{Meta: meta}

	// 2. Positional task attribution: records between one task_started and the next
	// belong to that task (rollouts are single-threaded within a file).
	taskIdx := -1
	recTask := make([]int, len(records))
	taskByTurn := map[string]int{}
	for i, t := range rep.Tasks {
		taskByTurn[t.TurnID] = i
	}
	for i, r := range records {
		if r.Kind == KindStarted {
			if j, ok := taskByTurn[r.TurnID]; ok {
				taskIdx = j
			}
		}
		recTask[i] = taskIdx
	}

	// 3. Join calls to outputs by call id; type every call outcome.
	outIdx := map[string]int{} // call_id -> record index of its output
	for i, r := range records {
		if r.Kind == "function_call_output" && r.CallID != "" {
			if _, dup := outIdx[r.CallID]; !dup {
				outIdx[r.CallID] = i
			}
		}
	}
	lastOutputInTask := map[int]int{} // task -> last output record index
	for i, r := range records {
		if r.Kind == "function_call_output" {
			lastOutputInTask[recTask[i]] = i
		}
	}

	tasks := make([]TaskAnalytics, len(rep.Tasks))
	for i, t := range rep.Tasks {
		tasks[i] = TaskAnalytics{
			TurnID: t.TurnID, Outcome: t.Outcome, Provenance: t.Provenance,
			RecordedMS: t.DurationMS, TTFTMS: -1, Classes: map[ToolClass]int{},
		}
		if t.StartedAt != "" && t.EndedAt != "" {
			s, e1 := time.Parse(time.RFC3339, t.StartedAt)
			e, e2 := time.Parse(time.RFC3339, t.EndedAt)
			if e1 == nil && e2 == nil && e.After(s) {
				tasks[i].WallMS = e.Sub(s).Milliseconds()
			}
		}
	}

	callWait := map[string]bool{} // call_id -> wait-classified command
	toolMSByTask := map[int]map[string]int64{}
	behavior := &out.Behavior
	failCounts := map[string]*SigRow{}
	okCounts := map[string]*SigRow{}
	churn := map[string]int64{}

	for i, r := range records {
		ti := recTask[i]
		if r.Kind != kindToolCall {
			continue
		}
		out.Calls++
		if ti >= 0 {
			tasks[ti].ToolCalls++
		}
		callWait[r.CallID] = waitCommandRE.MatchString(r.Head)
		if sleepPollRE.MatchString(r.Head) {
			behavior.SleepPolls++
		}
		for _, f := range r.Targets {
			churn[f]++
		}

		var class ToolClass
		var reason string
		var conf Confidence
		var spanMS int64
		if oi, ok := outIdx[r.CallID]; ok && r.CallID != "" {
			env := records[oi].Env
			class, reason, conf = ClassifyOutcome(r.Head, env)
			if records[oi].TS.After(r.TS) {
				spanMS = records[oi].TS.Sub(r.TS).Milliseconds()
			}
		} else {
			// MISSING RESULT: typed off the reconciled task boundary, never counted
			// as success and never as a generic error (#4767 issue guidance).
			class, reason, conf = missingResultClass(ti, i, tasks, lastOutputInTask)
		}
		if class == ToolTimeout {
			behavior.TimeoutKills++
		}
		sig := sigHash(r.Tool, r.Head)
		switch {
		case class.CountsAsFailure():
			bumpSig(failCounts, r.Tool, sig)
		case class == ToolOK:
			bumpSig(okCounts, r.Tool, sig)
		}
		if ti >= 0 {
			tasks[ti].Classes[class]++
			if spanMS > 0 {
				if toolMSByTask[ti] == nil {
					toolMSByTask[ti] = map[string]int64{}
				}
				if callWait[r.CallID] {
					tasks[ti].WaitMS += spanMS
				} else {
					tasks[ti].ToolMS += spanMS
					toolMSByTask[ti][r.Tool] += spanMS
				}
			}
		}
		out.Outcomes = append(out.Outcomes, CallOutcome{
			TurnID: turnOf(ti, tasks), Tool: r.Tool, Class: class, Reason: reason,
			Confidence: conf, SpanMS: spanMS, Sig: sig,
		})
	}

	// 4. Timeline decomposition: sequential gap attribution per task.
	decomposeTimeline(records, recTask, tasks, behavior)

	// 5. Critical-path contributors per task.
	for i := range tasks {
		tasks[i].Critical = contributors(&tasks[i], toolMSByTask[i])
	}

	behavior.RepeatFailures, behavior.MaxRepeatFailure = sigRows(failCounts, repeatFailureMinCodex)
	behavior.SuccessLoops, behavior.MaxSuccessLoop = sigRows(okCounts, successRepeatMinCodex)
	for f, n := range churn {
		if n > behavior.MaxEditChurn {
			behavior.MaxEditChurn = n
		}
		if n >= fileChurnMinCodex {
			behavior.EditChurn = append(behavior.EditChurn, ChurnRow{File: f, Count: n})
		}
	}
	sort.Slice(behavior.EditChurn, func(a, b int) bool { return behavior.EditChurn[a].Count > behavior.EditChurn[b].Count })

	for i := range tasks {
		if tasks[i].Outcome == Complete {
			out.SubstantiveCompleted = true
		}
		if i > 0 && tasks[i-1].Outcome == Complete && tasks[i].Outcome == Aborted {
			dur := tasks[i].RecordedMS
			if dur <= 0 {
				dur = tasks[i].WallMS
			}
			if tasks[i].ToolCalls == 0 && dur <= 2000 {
				tasks[i].TrailingEmptyAbort = true
			}
		}
	}
	if len(tasks) > 1 && tasks[len(tasks)-1].TrailingEmptyAbort {
		out.CompletedWithTrailingAbort = true
	}

	out.Tasks = tasks
	return out
}

func turnOf(ti int, tasks []TaskAnalytics) string {
	if ti >= 0 && ti < len(tasks) {
		return tasks[ti].TurnID
	}
	return ""
}

// missingResultClass types a call with no output using the reconciled boundary:
// a later output in the SAME task proves the result is simply missing; otherwise
// the owning task's typed terminal decides interrupted vs live tail.
func missingResultClass(ti, callIdx int, tasks []TaskAnalytics, lastOutputInTask map[int]int) (ToolClass, string, Confidence) {
	if ti < 0 || ti >= len(tasks) {
		return ToolMissingResult, "no_owning_task", ConfidenceInferred
	}
	if last, ok := lastOutputInTask[ti]; ok && last > callIdx {
		return ToolMissingResult, "later_output_in_task", ConfidenceObserved
	}
	switch tasks[ti].Outcome {
	case Complete:
		return ToolMissingResult, "task_completed_without_result", ConfidenceObserved
	case Live:
		return ToolLiveTail, "rollout_fresh_call_may_still_run", ConfidenceInferred
	default: // Aborted, Superseded, ProcessDeath
		return ToolInterrupted, "task_boundary_" + string(tasks[ti].Outcome), ConfidenceInferred
	}
}

// decomposeTimeline walks records in file order and attributes every inter-event
// gap: a gap ending at a call OUTPUT belongs to that tool (already booked from the
// call span, so it is skipped here); a gap ending at a model-emitted record while
// no call is outstanding is split model-vs-idle at the stall threshold. TTFT is
// the first model-emitted record (tool call or token count) after the task start.
func decomposeTimeline(records []ARecord, recTask []int, tasks []TaskAnalytics, behavior *CodexBehavior) {
	type cursor struct {
		startAt time.Time // the task's own start timestamp
		at      time.Time // last seen record timestamp
		inCall  bool      // a function_call is outstanding (its span owns this time)
	}
	cursors := make([]cursor, len(tasks))
	for i, r := range records {
		ti := recTask[i]
		if ti < 0 || r.TS.IsZero() {
			continue
		}
		t := &tasks[ti]
		cur := &cursors[ti]
		if r.Kind == KindStarted {
			cur.startAt, cur.at = r.TS, r.TS
			continue
		}
		if r.Kind == kindCompacted {
			t.Compactions++
		}
		if cur.at.IsZero() { // records before any start in this task: anchor only
			cur.at = r.TS
			continue
		}
		if r.Kind == "function_call_output" {
			// Time up to an output is the call span — already booked as tool/wait.
			cur.at, cur.inCall = r.TS, false
			continue
		}
		modelEmitted := r.Kind == kindToolCall || r.Kind == kindTokens ||
			r.Kind == KindComplete || r.Kind == KindAborted
		if !modelEmitted {
			continue // compaction and friends: not a timeline anchor
		}
		if t.TTFTMS < 0 && !cur.startAt.IsZero() && (r.Kind == kindToolCall || r.Kind == kindTokens) {
			t.TTFTMS = r.TS.Sub(cur.startAt).Milliseconds()
		}
		if gap := r.TS.Sub(cur.at).Milliseconds(); gap > 0 && !cur.inCall {
			if gap > stallGapMS {
				t.ModelMS += stallGapMS
				t.IdleMS += gap - stallGapMS
				t.IdleGaps++
				behavior.StallGaps++
				if s := float64(gap) / 1000.0; s > behavior.MaxGapS {
					behavior.MaxGapS = s
				}
			} else {
				t.ModelMS += gap
			}
		}
		if r.Kind == kindToolCall {
			cur.inCall = true
		}
		cur.at = r.TS
	}
}

func contributors(t *TaskAnalytics, byTool map[string]int64) []Contributor {
	var rows []Contributor
	if t.ModelMS > 0 {
		rows = append(rows, Contributor{"model", t.ModelMS})
	}
	if t.IdleMS > 0 {
		rows = append(rows, Contributor{"idle", t.IdleMS})
	}
	if t.WaitMS > 0 {
		rows = append(rows, Contributor{"wait", t.WaitMS})
	}
	for tool, ms := range byTool {
		rows = append(rows, Contributor{"tool:" + tool, ms})
	}
	sort.Slice(rows, func(a, b int) bool {
		if rows[a].MS != rows[b].MS {
			return rows[a].MS > rows[b].MS
		}
		return rows[a].Category < rows[b].Category
	})
	return rows
}

func bumpSig(m map[string]*SigRow, tool, sig string) {
	row := m[sig]
	if row == nil {
		row = &SigRow{Tool: tool, Sig: sig}
		m[sig] = row
	}
	row.Count++
}

func sigRows(m map[string]*SigRow, min int64) ([]SigRow, int64) {
	var rows []SigRow
	var max int64
	for _, r := range m {
		if r.Count > max {
			max = r.Count
		}
		if r.Count >= min {
			rows = append(rows, *r)
		}
	}
	sort.Slice(rows, func(a, b int) bool {
		if rows[a].Count != rows[b].Count {
			return rows[a].Count > rows[b].Count
		}
		return rows[a].Sig < rows[b].Sig
	})
	return rows, max
}
