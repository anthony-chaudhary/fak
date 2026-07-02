package resume

import (
	"fmt"
	"sort"
	"strings"
)

// WatchdogStatusSchema is the machine-readable contract for the live-drain readout.
const WatchdogStatusSchema = "fak.resume-watchdog.status.v1"

// WatchdogDrainVerdict is the steward's closed verdict for the AUTO_RESUME queue.
type WatchdogDrainVerdict string

const (
	WatchdogDrainGreen WatchdogDrainVerdict = "green"
	WatchdogDrainRed   WatchdogDrainVerdict = "red"
)

// WatchdogMTTRStatus is the per-session state in the watchdog drain report.
type WatchdogMTTRStatus string

const (
	WatchdogMTTRQueued           WatchdogMTTRStatus = "queued"
	WatchdogMTTRLaunchedUnproven WatchdogMTTRStatus = "launched_unproven"
	WatchdogMTTRRecovered        WatchdogMTTRStatus = "recovered"
)

// WatchdogStatusInput is the pure input to FoldWatchdogStatus. Events are typed facts
// parsed from the durable resume ledger; Plan is the current AUTO_RESUME plan.
type WatchdogStatusInput struct {
	Mode           string                `json:"mode"`
	NowUnix        int64                 `json:"now_unix"`
	SilentSeconds  int64                 `json:"silent_seconds"`
	MonotonicTicks int                   `json:"monotonic_ticks"`
	Plan           []WatchdogPlanRow     `json:"plan,omitempty"`
	Events         []WatchdogStatusEvent `json:"events,omitempty"`
}

// WatchdogStatusEvent is one ledger fact the drain steward can trust without reading
// transcript content: queued/detected, launched/resumed, progress, or a queue-depth
// snapshot. Unknown fields stay shell-side.
type WatchdogStatusEvent struct {
	UnixSeconds         int64  `json:"unix_seconds,omitempty"`
	Session             string `json:"session,omitempty"`
	Phase               string `json:"phase,omitempty"`
	Mode                string `json:"mode,omitempty"`
	AutoResumeDepth     int    `json:"auto_resume_depth,omitempty"`
	NewTurns            int    `json:"new_turns,omitempty"`
	CommitSHA           string `json:"commit_sha,omitempty"`
	LedgerProgress      bool   `json:"ledger_progress,omitempty"`
	DetectedUnix        int64  `json:"detected_at,omitempty"`
	ResumedUnix         int64  `json:"resumed_at,omitempty"`
	ProgressWitnessUnix int64  `json:"progress_witnessed_at,omitempty"`
}

// WatchdogMTTRRow is the row-level evidence for one session's recovery journey. A row is
// recovered only when a launch/resume is followed by independent progress evidence.
type WatchdogMTTRRow struct {
	Session             string             `json:"session"`
	Status              WatchdogMTTRStatus `json:"status"`
	Mode                string             `json:"mode"`
	DetectedAt          int64              `json:"detected_at,omitempty"`
	ResumedAt           int64              `json:"resumed_at,omitempty"`
	ProgressWitnessedAt int64              `json:"progress_witnessed_at,omitempty"`
	SilentSeconds       int64              `json:"silent_seconds,omitempty"`
	Evidence            string             `json:"evidence,omitempty"`
}

// WatchdogDrainStatus is the one-command answer to "is recovery draining?".
type WatchdogDrainStatus struct {
	Schema                   string               `json:"schema"`
	Mode                     string               `json:"mode"`
	Verdict                  WatchdogDrainVerdict `json:"verdict"`
	AutoResumeDepth          int                  `json:"auto_resume_depth"`
	AutoResumeMonotonicTicks int                  `json:"auto_resume_monotonic_ticks,omitempty"`
	SilentSeconds            int64                `json:"silent_seconds,omitempty"`
	SilentHours              float64              `json:"silent_hours,omitempty"`
	MTTRSessions             []WatchdogMTTRRow    `json:"mttr_sessions"`
	Reasons                  []string             `json:"reasons,omitempty"`
}

type watchdogSessionFold struct {
	session    string
	detectedAt int64
	launches   []int64
	progresses []watchdogProgress
	closed     bool
}

type watchdogProgress struct {
	at       int64
	evidence string
}

type watchdogDepthSample struct {
	at    int64
	mode  string
	depth int
}

// FoldWatchdogStatus folds the current AUTO_RESUME plan plus durable ledger evidence into
// a drain verdict. A launched row alone never recovers a session: recovery needs progress
// evidence (new turns, a commit, or an explicit ledger-progress witness).
func FoldWatchdogStatus(in WatchdogStatusInput) WatchdogDrainStatus {
	now := in.NowUnix
	mode := normalizeWatchdogMode(in.Mode)
	bySession := map[string]*watchdogSessionFold{}
	depthSamples := make([]watchdogDepthSample, 0)
	currentDepth := len(in.Plan)
	planSessions := map[string]bool{}
	for _, row := range in.Plan {
		if row.Session != "" {
			planSessions[row.Session] = true
		}
	}
	hasCurrentPlan := in.Plan != nil

	events := append([]WatchdogStatusEvent(nil), in.Events...)
	sort.SliceStable(events, func(i, j int) bool { return events[i].UnixSeconds < events[j].UnixSeconds })
	for _, e := range events {
		phase := normalizeWatchdogPhase(e.Phase)
		at := firstNonZero(e.UnixSeconds, e.DetectedUnix, e.ResumedUnix, e.ProgressWitnessUnix)
		if phase == "status" || phase == "tick" || phase == "snapshot" {
			depthSamples = append(depthSamples, watchdogDepthSample{at: at, mode: normalizeWatchdogMode(firstNonEmpty(e.Mode, mode)), depth: e.AutoResumeDepth})
			currentDepth = e.AutoResumeDepth
		}
		if e.Session == "" {
			continue
		}
		if e.DetectedUnix > 0 {
			f := watchdogFoldFor(bySession, e.Session)
			f.beginCycle(e.DetectedUnix)
		}
		switch phase {
		case "queued", "detected", "auto_resume":
			watchdogFoldFor(bySession, e.Session).beginCycle(e.UnixSeconds)
		case "launched", "resumed":
			watchdogFoldFor(bySession, e.Session).recordLaunch(firstNonZero(e.ResumedUnix, e.UnixSeconds))
		case "settled", "operator_settled", "consolidated":
			watchdogFoldFor(bySession, e.Session).close()
		}
		if e.ProgressWitnessUnix > 0 {
			watchdogFoldFor(bySession, e.Session).recordProgress(e.ProgressWitnessUnix, "progress_witnessed_at")
		}
		if e.NewTurns > 0 && e.UnixSeconds > 0 {
			watchdogFoldFor(bySession, e.Session).recordProgress(e.UnixSeconds, fmt.Sprintf("new_turns:%d", e.NewTurns))
		}
		if phase == "progress" && e.NewTurns <= 0 && e.UnixSeconds > 0 {
			watchdogFoldFor(bySession, e.Session).recordProgress(e.UnixSeconds, "progress_row")
		}
		if strings.TrimSpace(e.CommitSHA) != "" && e.UnixSeconds > 0 {
			watchdogFoldFor(bySession, e.Session).recordProgress(e.UnixSeconds, "commit:"+strings.TrimSpace(e.CommitSHA))
		}
		if e.LedgerProgress && e.UnixSeconds > 0 {
			watchdogFoldFor(bySession, e.Session).recordProgress(e.UnixSeconds, "ledger_progress")
		}
	}
	for _, row := range in.Plan {
		if row.Session == "" {
			continue
		}
		f := watchdogFoldFor(bySession, row.Session)
		if f.closed || f.detectedAt == 0 || f.recovered() {
			f.beginCycle(now)
		}
	}
	if hasCurrentPlan && now > 0 {
		depthSamples = append(depthSamples, watchdogDepthSample{at: now, mode: mode, depth: len(in.Plan)})
		currentDepth = len(in.Plan)
	}

	rows := make([]WatchdogMTTRRow, 0, len(bySession))
	var maxSilent int64
	for _, f := range bySession {
		if f.closed {
			continue
		}
		if hasCurrentPlan && !planSessions[f.session] {
			continue
		}
		row := foldWatchdogMTTRRow(*f, mode, now)
		if row.SilentSeconds > maxSilent {
			maxSilent = row.SilentSeconds
		}
		rows = append(rows, row)
	}
	sort.SliceStable(rows, func(i, j int) bool {
		ri, rj := watchdogMTTRRank(rows[i].Status), watchdogMTTRRank(rows[j].Status)
		if ri != rj {
			return ri < rj
		}
		return rows[i].Session < rows[j].Session
	})

	reasons := make([]string, 0)
	monotonic := monotonicGrowthTicks(depthSamples, in.MonotonicTicks)
	if monotonic > 0 {
		reasons = append(reasons, fmt.Sprintf("AUTO_RESUME depth grew monotonically for %d ticks", monotonic))
	}
	if in.SilentSeconds > 0 && maxSilent >= in.SilentSeconds {
		reasons = append(reasons, fmt.Sprintf("oldest unrecovered AUTO_RESUME row silent for %.1fh", float64(maxSilent)/3600))
	}
	if mode == "DRY-RUN" && currentDepth > 0 {
		reasons = append(reasons, "watchdog is DRY-RUN with queued AUTO_RESUME rows")
	}

	verdict := WatchdogDrainGreen
	if len(reasons) > 0 {
		verdict = WatchdogDrainRed
	}
	return WatchdogDrainStatus{
		Schema:                   WatchdogStatusSchema,
		Mode:                     mode,
		Verdict:                  verdict,
		AutoResumeDepth:          currentDepth,
		AutoResumeMonotonicTicks: monotonic,
		SilentSeconds:            maxSilent,
		SilentHours:              float64(maxSilent) / 3600,
		MTTRSessions:             rows,
		Reasons:                  reasons,
	}
}

func watchdogFoldFor(m map[string]*watchdogSessionFold, session string) *watchdogSessionFold {
	if f := m[session]; f != nil {
		return f
	}
	f := &watchdogSessionFold{session: session}
	m[session] = f
	return f
}

func (f *watchdogSessionFold) beginCycle(at int64) {
	if at <= 0 {
		at = f.detectedAt
	}
	f.detectedAt = at
	f.launches = nil
	f.progresses = nil
	f.closed = false
}

func (f *watchdogSessionFold) recordLaunch(at int64) {
	if at <= 0 {
		return
	}
	if f.recovered() {
		f.beginCycle(at)
	}
	if f.detectedAt == 0 {
		f.detectedAt = at
	}
	f.launches = append(f.launches, at)
	f.closed = false
}

func (f *watchdogSessionFold) recordProgress(at int64, evidence string) {
	if at <= 0 {
		return
	}
	if f.detectedAt == 0 {
		f.detectedAt = at
	}
	f.progresses = append(f.progresses, watchdogProgress{at: at, evidence: evidence})
	f.closed = false
}

func (f *watchdogSessionFold) close() {
	f.detectedAt = 0
	f.launches = nil
	f.progresses = nil
	f.closed = true
}

func (f watchdogSessionFold) recovered() bool {
	progressAt, _ := firstProgressAfterLaunch(f.detectedAt, f.launches, f.progresses)
	return resumeAtForProgress(f.detectedAt, f.launches, progressAt) > 0 && progressAt > 0
}

func foldWatchdogMTTRRow(f watchdogSessionFold, mode string, now int64) WatchdogMTTRRow {
	sort.Slice(f.launches, func(i, j int) bool { return f.launches[i] < f.launches[j] })
	sort.Slice(f.progresses, func(i, j int) bool { return f.progresses[i].at < f.progresses[j].at })
	progressAt, evidence := firstProgressAfterLaunch(f.detectedAt, f.launches, f.progresses)
	resumedAt := resumeAtForProgress(f.detectedAt, f.launches, progressAt)
	status := WatchdogMTTRQueued
	if resumedAt > 0 {
		status = WatchdogMTTRLaunchedUnproven
	}
	if resumedAt > 0 && progressAt > 0 {
		status = WatchdogMTTRRecovered
	}
	silent := int64(0)
	if status != WatchdogMTTRRecovered && f.detectedAt > 0 && now > f.detectedAt {
		silent = now - f.detectedAt
	}
	return WatchdogMTTRRow{
		Session:             f.session,
		Status:              status,
		Mode:                mode,
		DetectedAt:          f.detectedAt,
		ResumedAt:           resumedAt,
		ProgressWitnessedAt: progressAt,
		SilentSeconds:       silent,
		Evidence:            evidence,
	}
}

func firstProgressAfterLaunch(detectedAt int64, launches []int64, progresses []watchdogProgress) (int64, string) {
	if len(launches) == 0 {
		return 0, ""
	}
	for _, p := range progresses {
		for _, l := range launches {
			if launchInCycle(detectedAt, l) && p.at > l {
				return p.at, p.evidence
			}
		}
	}
	return 0, ""
}

func resumeAtForProgress(detectedAt int64, launches []int64, progressAt int64) int64 {
	if len(launches) == 0 {
		return 0
	}
	if progressAt <= 0 {
		for i := len(launches) - 1; i >= 0; i-- {
			if launchInCycle(detectedAt, launches[i]) {
				return launches[i]
			}
		}
		return 0
	}
	resumedAt := int64(0)
	for _, l := range launches {
		if launchInCycle(detectedAt, l) && l <= progressAt {
			resumedAt = l
		}
	}
	return resumedAt
}

func launchInCycle(detectedAt, launchAt int64) bool {
	return launchAt > 0 && (detectedAt <= 0 || launchAt >= detectedAt)
}

func monotonicGrowthTicks(samples []watchdogDepthSample, ticks int) int {
	if ticks <= 1 || len(samples) < ticks {
		return 0
	}
	sort.SliceStable(samples, func(i, j int) bool { return samples[i].at < samples[j].at })
	tail := samples[len(samples)-ticks:]
	for i := 1; i < len(tail); i++ {
		if tail[i].depth <= tail[i-1].depth {
			return 0
		}
	}
	return ticks
}

func normalizeWatchdogMode(mode string) string {
	mode = strings.ToUpper(strings.TrimSpace(mode))
	if mode == "" {
		return "UNKNOWN"
	}
	return mode
}

func normalizeWatchdogPhase(phase string) string {
	phase = strings.ToLower(strings.TrimSpace(phase))
	if phase == "" {
		return "launched"
	}
	return phase
}

func watchdogMTTRRank(s WatchdogMTTRStatus) int {
	switch s {
	case WatchdogMTTRQueued:
		return 0
	case WatchdogMTTRLaunchedUnproven:
		return 1
	default:
		return 2
	}
}

func setEarliest(dst *int64, v int64) {
	if v <= 0 {
		return
	}
	if *dst == 0 || v < *dst {
		*dst = v
	}
}

func firstNonZero(vals ...int64) int64 {
	for _, v := range vals {
		if v > 0 {
			return v
		}
	}
	return 0
}

func firstNonEmpty(vals ...string) string {
	for _, v := range vals {
		if strings.TrimSpace(v) != "" {
			return v
		}
	}
	return ""
}
