package fleetmon

import (
	"fmt"
	"sort"
	"time"
)

// MonitorSchema tags the monitor's JSON payload.
const MonitorSchema = "fak-fleet-monitor/1"

// Classification is the closed set of worker states the monitor emits. Every
// state is derived from registry/process/transcript EVIDENCE — a worker's own
// "busy" is never sufficient for any of them.
type Classification string

const (
	ClassHealthy         Classification = "healthy"              // PID alive and the transcript is advancing
	ClassCompletedFinal  Classification = "completed-final"      // idle with a final report on the transcript
	ClassDead            Classification = "dead"                 // registry-active but the worker PID is gone
	ClassStaleTranscript Classification = "stale-transcript"     // PID alive but the transcript has not advanced past the threshold
	ClassAuthRateBlocked Classification = "auth-or-rate-blocked" // a current auth/rate/credit/access blocker
	ClassStaleChild      Classification = "stale-child-command"  // PID alive but wedged on a stale child process tree
	ClassAttention       Classification = "attention"            // evidence is insufficient/ambiguous — inspect
)

// Thresholds bound the staleness decisions. Defaults match the issue: a 20-minute
// transcript-idle floor, a 5-minute simple-child floor, a 10-minute test-child
// floor.
type Thresholds struct {
	StaleTranscript  time.Duration
	StaleChildSimple time.Duration
	StaleChildTest   time.Duration
}

// DefaultThresholds returns the issue's default staleness floors.
func DefaultThresholds() Thresholds {
	return Thresholds{
		StaleTranscript:  20 * time.Minute,
		StaleChildSimple: 5 * time.Minute,
		StaleChildTest:   10 * time.Minute,
	}
}

// WorkerEvidence is the injected, per-worker evidence bundle the classifier folds.
// The cmd layer gathers it (registry row + PID liveness + CPU delta + transcript
// read + janitor scan); the classifier itself does no I/O.
type WorkerEvidence struct {
	Issue          int
	Session        string
	Account        string
	RegistryDisp   string           // sessions.json Disp: LIVE / DONE / USER_CLOSED / INFRA_AUTH / ...
	RegistryAction string           // sessions.json Action: BLOCKED_AUTH / ...
	HasPID         bool             // whether a worker PID is known at all
	PID            int              //
	PIDAlive       bool             // OS liveness of PID (meaningful only when HasPID)
	CPUDeltaSec    *float64         // CPU seconds consumed since the previous sample (nil = unknown)
	PrevLines      *int             // transcript line count at the previous sample (nil = first sample)
	Transcript     TranscriptSignal //
	StaleChildren  []ChildCommand   // stale child commands the janitor scan attributed to this worker
}

// WorkerSample is the classified, machine-readable monitor row for one worker.
type WorkerSample struct {
	Issue            int            `json:"issue,omitempty"`
	Session          string         `json:"session"`
	Account          string         `json:"account,omitempty"`
	Class            Classification `json:"class"`
	RegistryStatus   string         `json:"registry_status,omitempty"`
	PID              int            `json:"pid,omitempty"`
	PIDAlive         bool           `json:"pid_alive"`
	CPUDeltaSec      *float64       `json:"cpu_delta_sec,omitempty"`
	TranscriptAgeSec *float64       `json:"transcript_age_sec,omitempty"`
	LineDelta        *int           `json:"line_delta,omitempty"`
	FinalReport      bool           `json:"final_report"`
	Blocker          string         `json:"blocker,omitempty"`
	ChildSummary     string         `json:"child_summary,omitempty"`
	Reasons          []string       `json:"reasons"`
}

// Classify folds one worker's evidence into a classified sample. The order of
// the decision ladder is deliberate and evidence-first:
//
//  1. dead — the ONLY class that fires on a missing process, and it needs a known
//     PID that is not alive. A worker whose PID is alive is NEVER called dead, so
//     an all-PIDs-alive run raises zero false dead-worker alerts.
//  2. auth-or-rate-blocked — a current blocker (registry auth row or a transcript
//     tail signature) that no final report has cleared.
//  3. completed-final — an idle worker whose transcript ends on a final report.
//  4. stale-child-command — alive, no final report, wedged on a stale child tree.
//  5. stale-transcript — alive, no final report, transcript idle past the floor.
//  6. healthy — alive and advancing (fresh transcript, growing lines, or CPU burn).
//  7. attention — anything left: the evidence does not decide, so ask a human.
func Classify(ev WorkerEvidence, now time.Time, th Thresholds) WorkerSample {
	if th == (Thresholds{}) {
		th = DefaultThresholds()
	}
	s := WorkerSample{
		Issue:          ev.Issue,
		Session:        ev.Session,
		Account:        ev.Account,
		PID:            ev.PID,
		PIDAlive:       ev.HasPID && ev.PIDAlive,
		CPUDeltaSec:    ev.CPUDeltaSec,
		FinalReport:    ev.Transcript.FinalReport,
		RegistryStatus: registryStatus(ev),
		ChildSummary:   childSummary(ev.StaleChildren),
	}

	var ageSec *float64
	if ev.Transcript.HasTimestamp {
		v := now.Sub(ev.Transcript.LastTimestamp).Seconds()
		if v < 0 {
			v = 0
		}
		ageSec = &v
		s.TranscriptAgeSec = &v
	}
	if ev.PrevLines != nil {
		d := ev.Transcript.Lines - *ev.PrevLines
		s.LineDelta = &d
	}

	stale := ageSec != nil && *ageSec > th.StaleTranscript.Seconds()
	// activelyGrowing is forward progress we can SEE this sample: the line count
	// grew, or CPU was burned. Freshness alone is NOT active growth — a worker that
	// just stopped (final report) has a fresh transcript but is not still working.
	activelyGrowing := (s.LineDelta != nil && *s.LineDelta > 0) || (ev.CPUDeltaSec != nil && *ev.CPUDeltaSec > 0)
	advancing := isAdvancing(ev, ageSec, s.LineDelta, th)
	blocked := ev.Transcript.Blocker != "" || ev.RegistryAction == "BLOCKED_AUTH" || ev.RegistryDisp == "INFRA_AUTH"

	switch {
	case ev.HasPID && !ev.PIDAlive:
		s.Class = ClassDead
		s.Reasons = append(s.Reasons, fmt.Sprintf("registry %s but worker PID %d is not alive", orNone(ev.RegistryDisp), ev.PID))

	case blocked && !ev.Transcript.FinalReport:
		s.Class = ClassAuthRateBlocked
		s.Blocker = blockerReason(ev)
		s.Reasons = append(s.Reasons, "current blocker: "+s.Blocker)

	case ev.Transcript.FinalReport && !activelyGrowing:
		s.Class = ClassCompletedFinal
		s.Reasons = append(s.Reasons, "transcript ended on a final report and is idle")

	case ev.HasPID && ev.PIDAlive && len(ev.StaleChildren) > 0 && !ev.Transcript.FinalReport:
		s.Class = ClassStaleChild
		s.Reasons = append(s.Reasons, "PID alive but wedged on stale child: "+s.ChildSummary)

	case ev.HasPID && ev.PIDAlive && stale && !ev.Transcript.FinalReport:
		s.Class = ClassStaleTranscript
		s.Reasons = append(s.Reasons, fmt.Sprintf("transcript idle %s > %s, PID alive", roundDur(*ageSec), th.StaleTranscript))

	case ev.HasPID && ev.PIDAlive && advancing:
		s.Class = ClassHealthy
		s.Reasons = append(s.Reasons, advancingReason(ev, ageSec, s.LineDelta))

	default:
		s.Class = ClassAttention
		s.Reasons = append(s.Reasons, attentionReason(ev, ageSec))
	}
	return s
}

// isAdvancing reports whether a worker shows forward progress this sample: the
// transcript is fresh (younger than the stale floor), OR its line count grew,
// OR it burned CPU since the previous sample. Any one is enough — the monitor
// errs toward "still working" so it does not alarm on a briefly quiet worker.
func isAdvancing(ev WorkerEvidence, ageSec *float64, lineDelta *int, th Thresholds) bool {
	if lineDelta != nil && *lineDelta > 0 {
		return true
	}
	if ev.CPUDeltaSec != nil && *ev.CPUDeltaSec > 0 {
		return true
	}
	if ageSec != nil && *ageSec <= th.StaleTranscript.Seconds() {
		return true
	}
	return false
}

func registryStatus(ev WorkerEvidence) string {
	if ev.RegistryAction != "" && ev.RegistryAction != "OK" {
		return ev.RegistryDisp + "/" + ev.RegistryAction
	}
	return ev.RegistryDisp
}

func blockerReason(ev WorkerEvidence) string {
	if ev.Transcript.Blocker != "" {
		return ev.Transcript.Blocker
	}
	if ev.RegistryAction == "BLOCKED_AUTH" || ev.RegistryDisp == "INFRA_AUTH" {
		return "registry reports auth block"
	}
	return "blocked"
}

func childSummary(children []ChildCommand) string {
	if len(children) == 0 {
		return ""
	}
	c := children[0]
	extra := ""
	if len(children) > 1 {
		extra = fmt.Sprintf(" (+%d more)", len(children)-1)
	}
	return fmt.Sprintf("%s pid %d age %s [%s]%s", c.Name, c.RootPID, roundDur(float64(c.AgeSec)), c.Class, extra)
}

func advancingReason(ev WorkerEvidence, ageSec *float64, lineDelta *int) string {
	switch {
	case lineDelta != nil && *lineDelta > 0:
		return fmt.Sprintf("transcript advanced +%d lines", *lineDelta)
	case ev.CPUDeltaSec != nil && *ev.CPUDeltaSec > 0:
		return fmt.Sprintf("burned %.1fs CPU since last sample", *ev.CPUDeltaSec)
	case ageSec != nil:
		return fmt.Sprintf("transcript fresh (%s old)", roundDur(*ageSec))
	default:
		return "advancing"
	}
}

func attentionReason(ev WorkerEvidence, ageSec *float64) string {
	switch {
	case !ev.HasPID && !ev.Transcript.Exists:
		return "no worker PID and no transcript — cannot witness liveness"
	case !ev.HasPID:
		return "no worker PID known — cannot confirm liveness"
	case !ev.Transcript.Exists:
		return "worker PID alive but no transcript yet"
	case ageSec == nil:
		return "worker PID alive but transcript has no timestamp"
	default:
		return "PID alive, transcript idle but under the stale floor — inspect"
	}
}

func orNone(s string) string {
	if s == "" {
		return "(no status)"
	}
	return s
}

func roundDur(sec float64) string {
	d := time.Duration(sec) * time.Second
	return d.Round(time.Second).String()
}

// SortSamples orders samples for a stable operator table: attention-worthy
// classes first (dead, blocked, stale, attention), then by issue number.
func SortSamples(samples []WorkerSample) {
	sort.SliceStable(samples, func(i, j int) bool {
		pi, pj := classPriority(samples[i].Class), classPriority(samples[j].Class)
		if pi != pj {
			return pi < pj
		}
		if samples[i].Issue != samples[j].Issue {
			return samples[i].Issue < samples[j].Issue
		}
		return samples[i].Session < samples[j].Session
	})
}

func classPriority(c Classification) int {
	switch c {
	case ClassDead:
		return 0
	case ClassAuthRateBlocked:
		return 1
	case ClassStaleChild:
		return 2
	case ClassStaleTranscript:
		return 3
	case ClassAttention:
		return 4
	case ClassHealthy:
		return 5
	case ClassCompletedFinal:
		return 6
	default:
		return 7
	}
}

// MonitorPayload is the full machine-readable monitor output the cmd layer emits
// with --json.
type MonitorPayload struct {
	Schema      string                 `json:"schema"`
	GeneratedAt string                 `json:"generated_at"`
	RunID       string                 `json:"run_id,omitempty"`
	Total       int                    `json:"total"`
	ByClass     map[Classification]int `json:"by_class"`
	Workers     []WorkerSample         `json:"workers"`
}

// NewMonitorPayload assembles the payload from classified samples, stamping the
// class histogram so a control pane reads the run's health at a glance.
func NewMonitorPayload(runID string, samples []WorkerSample, now time.Time) MonitorPayload {
	SortSamples(samples)
	byClass := map[Classification]int{}
	for _, s := range samples {
		byClass[s.Class]++
	}
	return MonitorPayload{
		Schema:      MonitorSchema,
		GeneratedAt: now.UTC().Format(time.RFC3339),
		RunID:       runID,
		Total:       len(samples),
		ByClass:     byClass,
		Workers:     samples,
	}
}
