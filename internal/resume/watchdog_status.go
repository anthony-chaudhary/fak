package resume

import (
	"fmt"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/strmatch"
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
	WatchdogMTTRAuthBlocked      WatchdogMTTRStatus = "auth_blocked"
	WatchdogMTTRRecovered        WatchdogMTTRStatus = "recovered"
)

// WatchdogStatusInput is the pure input to FoldWatchdogStatus. Events are typed facts
// parsed from the durable resume ledger; Plan is the current AUTO_RESUME plan.
type WatchdogStatusInput struct {
	Mode            string `json:"mode"`
	NowUnix         int64  `json:"now_unix"`
	SilentSeconds   int64  `json:"silent_seconds"`
	UnprovenSeconds int64  `json:"unproven_seconds,omitempty"`
	// LaunchStaleSeconds arms the "auto-resume not launching" alarm (#3460): when the
	// plan carries queued sessions but the durable ledger shows no launch within this
	// many seconds, the resume layer is queuing work it never launches — the dead-but-
	// silent failure a fabricated "resumed just now" would otherwise mask. 0 disables it
	// (the pure default; the CLI arms it).
	LaunchStaleSeconds int64 `json:"launch_stale_seconds,omitempty"`
	MonotonicTicks     int   `json:"monotonic_ticks"`
	// BacklogThreshold / BacklogTicks arm the post-reset backlog SLO gate (#3582):
	// BOTTLENECK-MAP §7's manual decision — "if auto_resume is still >= N with 0 throttled
	// accounts, the cap is the real limiter" — as a standing detector. The gate fires only
	// when the last BacklogTicks depth samples ALL exceed BacklogThreshold *and* the roster
	// carries zero throttled accounts, i.e. the backlog outlived the throttle reset, so
	// recovery capacity (not transient account pressure) is what is binding. BacklogThreshold
	// <= 0 or BacklogTicks <= 0 disables it (the pure default; the CLI arms it).
	BacklogThreshold int `json:"backlog_threshold,omitempty"`
	BacklogTicks     int `json:"backlog_ticks,omitempty"`
	// ThrottledAccounts is the roster's current throttled-seat count; ThrottledAccountsKnown
	// says whether that count could actually be read. The gate FAILS CLOSED on an unknown
	// roster (never pages), because "0 throttled" is exactly what an unreadable roster and a
	// genuinely clear roster would both look like — and paging on the former is crying wolf.
	ThrottledAccounts      int  `json:"throttled_accounts,omitempty"`
	ThrottledAccountsKnown bool `json:"throttled_accounts_known,omitempty"`

	Plan   []WatchdogPlanRow     `json:"plan,omitempty"`
	Events []WatchdogStatusEvent `json:"events,omitempty"`
}

// WatchdogPageBacklogPersists is the closed reason code for the post-reset backlog page
// (#3582): the AUTO_RESUME backlog stayed above threshold across consecutive ticks while
// zero accounts were throttled.
const WatchdogPageBacklogPersists = "resume_backlog_persists_after_reset"

// WatchdogPage is the structured page the drain fold emits when a scoped SLO gate trips.
// Signature is the DEDUP key: it is deliberately built from the gate identity alone (reason
// + threshold), never from the live depth or a timestamp, so a gate that keeps firing tick
// after tick refreshes ONE occurrence-counted issue/toast instead of spamming one per tick.
type WatchdogPage struct {
	Reason            string `json:"reason"`
	Signature         string `json:"signature"`
	Depth             int    `json:"depth"`
	Threshold         int    `json:"threshold"`
	Ticks             int    `json:"ticks"`
	ThrottledAccounts int    `json:"throttled_accounts"`
	Detail            string `json:"detail"`
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
	UnprovenSeconds     int64              `json:"unproven_seconds,omitempty"`
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
	UnprovenSeconds          int64                `json:"unproven_seconds,omitempty"`
	UnprovenHours            float64              `json:"unproven_hours,omitempty"`
	MTTRSessions             []WatchdogMTTRRow    `json:"mttr_sessions"`
	Reasons                  []string             `json:"reasons,omitempty"`
	// Page is the structured SLO page to route to notify/guardcomplaint, or nil when no
	// gate tripped. At most one page per fold — the shell dedups it by Page.Signature.
	Page *WatchdogPage `json:"page,omitempty"`
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
	planDisps := map[string][]string{}
	for _, row := range in.Plan {
		if row.Session != "" {
			planSessions[row.Session] = true
			planDisps[row.Session] = append(planDisps[row.Session], row.Disp)
		}
	}
	hasCurrentPlan := in.Plan != nil

	// lastLedgerLaunchUnix is the newest real launch/resume event across all sessions —
	// the ground truth for "is auto-resume actually launching?" (#3460). lastTickMode is
	// the mode of the newest real TICK the watchdog recorded, i.e. the INSTALLED task's
	// mode — distinct from in.Mode, which only echoes this side-effect-free --status read.
	var lastLedgerLaunchUnix int64
	lastTickMode := ""
	events := append([]WatchdogStatusEvent(nil), in.Events...)
	sort.SliceStable(events, func(i, j int) bool { return events[i].UnixSeconds < events[j].UnixSeconds })
	for _, e := range events {
		phase := normalizeWatchdogPhase(e.Phase)
		at := firstNonZero(e.UnixSeconds, e.DetectedUnix, e.ResumedUnix, e.ProgressWitnessUnix)
		if phase == "status" || phase == "tick" || phase == "snapshot" {
			depthSamples = append(depthSamples, watchdogDepthSample{at: at, mode: normalizeWatchdogMode(strmatch.FirstNonBlank(e.Mode, mode)), depth: e.AutoResumeDepth})
			currentDepth = e.AutoResumeDepth
			lastTickMode = normalizeWatchdogMode(strmatch.FirstNonBlank(e.Mode, mode))
		}
		if e.Session == "" {
			continue
		}
		if e.DetectedUnix > 0 {
			f := watchdogFoldFor(bySession, e.Session)
			f.beginCycle(e.DetectedUnix)
		}
		if phaseIsLaunchToken(phase) {
			// A launch token ("launched"/"resumed") — single-sourced with Attempt.IsLaunch
			// via phaseIsLaunchToken so the two readers never drift on a named token (#4333).
			launchAt := firstNonZero(e.ResumedUnix, e.UnixSeconds)
			watchdogFoldFor(bySession, e.Session).recordLaunch(launchAt)
			if launchAt > lastLedgerLaunchUnix {
				lastLedgerLaunchUnix = launchAt
			}
		} else {
			switch phase {
			case "queued", "detected", "auto_resume":
				watchdogFoldFor(bySession, e.Session).beginCycle(e.UnixSeconds)
			case "settled", "operator_settled", "consolidated":
				watchdogFoldFor(bySession, e.Session).close()
			}
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
		// A session in the live plan must render as a row, but its detected/resumed times
		// come only from real ledger events — never fabricated from `now` (#3460): a plan-
		// only session with no ledger record reads detected=—/resumed=—, and a settled or
		// recovered session that re-appears reopens as a bare queued row with an unknown
		// detection time until a real queued/detected/launch row supplies it.
		if f.closed || f.recovered() {
			f.reopenQueued()
		}
	}
	if hasCurrentPlan && now > 0 {
		depthSamples = append(depthSamples, watchdogDepthSample{at: now, mode: mode, depth: len(in.Plan)})
		currentDepth = len(in.Plan)
	}

	rows := make([]WatchdogMTTRRow, 0, len(bySession))
	var maxSilent int64
	var maxUnproven int64
	staleUnproven := 0
	authBlocked := 0
	queuedPending := 0
	for _, f := range bySession {
		if f.closed {
			continue
		}
		if hasCurrentPlan && !planSessions[f.session] {
			continue
		}
		row := foldWatchdogMTTRRow(*f, mode, now)
		if hasCurrentPlan && planAuthBlocked(planDisps[f.session]) && row.Status != WatchdogMTTRRecovered {
			row.Status = WatchdogMTTRAuthBlocked
			row.UnprovenSeconds = 0
			row.Evidence = "plan_disp:" + strings.Join(uniqueNonEmpty(planDisps[f.session]), ",")
			authBlocked++
		}
		// A queued row is one auto-resume has NOT launched this cycle (resumedAt==0, and
		// not auth-blocked): the population the staleness alarm watches (#3460).
		if row.Status == WatchdogMTTRQueued {
			queuedPending++
		}
		if row.SilentSeconds > maxSilent {
			maxSilent = row.SilentSeconds
		}
		if row.UnprovenSeconds > maxUnproven {
			maxUnproven = row.UnprovenSeconds
		}
		if in.UnprovenSeconds > 0 && row.UnprovenSeconds >= in.UnprovenSeconds {
			staleUnproven++
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
	if in.UnprovenSeconds > 0 && staleUnproven > 0 {
		reasons = append(reasons, fmt.Sprintf("%d launched resume(s) unproven for >= %.1fm", staleUnproven, float64(in.UnprovenSeconds)/60))
	}
	if authBlocked > 0 {
		reasons = append(reasons, fmt.Sprintf("%d AUTO_RESUME row(s) require auth/login before resume", authBlocked))
	}
	// The installed watchdog's mode is the mode of the last real TICK the ledger recorded
	// — never in.Mode, which merely echoes THIS side-effect-free --status read (always dry
	// by nature). Only flag DRY-RUN when the actual watchdog last ticked dry (the installed
	// task lacks -Live, #3321), not because a status read is dry (#3460). With no recorded
	// tick we cannot prove the installed mode, so we stay silent rather than cry wolf.
	if lastTickMode == "DRY-RUN" && currentDepth > 0 {
		reasons = append(reasons, "installed watchdog last ticked DRY-RUN with queued AUTO_RESUME rows (reinstall with -Live)")
	}
	// RED headline (#3460): a live plan with queued sessions but no ledger launch within the
	// window means auto-resume is queuing work it never launches — the dead-but-silent
	// failure a fabricated "resumed just now" would mask. Prepend so it leads the reasons.
	if in.LaunchStaleSeconds > 0 && len(in.Plan) > 0 && queuedPending > 0 {
		if lastLedgerLaunchUnix <= 0 {
			reasons = append([]string{fmt.Sprintf("AUTO-RESUME NOT LAUNCHING: no ledger launch on record, %d queued", queuedPending)}, reasons...)
		} else if age := now - lastLedgerLaunchUnix; age >= in.LaunchStaleSeconds {
			reasons = append([]string{fmt.Sprintf("AUTO-RESUME NOT LAUNCHING: last ledger launch %s ago, %d queued", watchdogAge(age), queuedPending)}, reasons...)
		}
	}

	// Post-reset backlog SLO gate (#3582). Prepend: when recovery capacity is the binding
	// limiter, that is the headline an operator (or the MaxPerTick auto-scaler) must read
	// first — every other reason here is downstream of it.
	page := foldWatchdogBacklogPage(depthSamples, in)
	if page != nil {
		reasons = append([]string{page.Detail}, reasons...)
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
		UnprovenSeconds:          maxUnproven,
		UnprovenHours:            float64(maxUnproven) / 3600,
		MTTRSessions:             rows,
		Reasons:                  reasons,
		Page:                     page,
	}
}

// foldWatchdogBacklogPage decides the post-reset backlog page (#3582): the last BacklogTicks
// depth samples must ALL sit strictly above BacklogThreshold while the roster reports zero
// throttled accounts. Depth alone is not the signal — a deep backlog WITH throttled seats is
// the transient account pressure §4 already expects to clear itself at the next reset, and
// paging on it would train an operator to ignore the page. It is the backlog that OUTLIVES
// the throttle that proves recovery capacity is the real limiter.
//
// Returns nil (no page) when the gate is disarmed, when the ledger has not yet accumulated
// BacklogTicks samples, when any sample is at or below threshold, when any seat is throttled,
// or when the roster could not be read at all.
func foldWatchdogBacklogPage(samples []watchdogDepthSample, in WatchdogStatusInput) *WatchdogPage {
	if in.BacklogThreshold <= 0 || in.BacklogTicks <= 0 {
		return nil
	}
	// Fail closed on an unreadable roster: an absent count is not proof of zero throttled.
	if !in.ThrottledAccountsKnown || in.ThrottledAccounts != 0 {
		return nil
	}
	if len(samples) < in.BacklogTicks {
		return nil
	}
	ordered := append([]watchdogDepthSample(nil), samples...)
	sort.SliceStable(ordered, func(i, j int) bool { return ordered[i].at < ordered[j].at })
	tail := ordered[len(ordered)-in.BacklogTicks:]
	for _, s := range tail {
		if s.depth <= in.BacklogThreshold {
			return nil
		}
	}
	depth := tail[len(tail)-1].depth
	return &WatchdogPage{
		Reason: WatchdogPageBacklogPersists,
		// Gate identity only — stable across every tick this keeps firing, so the shell
		// refreshes one occurrence-counted issue instead of filing one per tick.
		Signature:         fmt.Sprintf("%s:threshold=%d", WatchdogPageBacklogPersists, in.BacklogThreshold),
		Depth:             depth,
		Threshold:         in.BacklogThreshold,
		Ticks:             in.BacklogTicks,
		ThrottledAccounts: 0,
		Detail: fmt.Sprintf("RESUME BACKLOG PERSISTS AFTER RESET: auto_resume depth stayed > %d for %d consecutive ticks (now %d) with 0 throttled accounts — recovery capacity, not account throttling, is the limiter",
			in.BacklogThreshold, in.BacklogTicks, depth),
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

// reopenQueued clears a closed/recovered fold back to a bare queued state WITHOUT
// inventing a detection time (#3460): the row still renders (its session is in the
// live plan) but its detected/resumed columns stay — until a real ledger event
// supplies them, instead of reading "resumed just now" off a week-dead ledger.
func (f *watchdogSessionFold) reopenQueued() {
	f.detectedAt = 0
	f.launches = nil
	f.progresses = nil
	f.closed = false
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
	unproven := int64(0)
	if status == WatchdogMTTRLaunchedUnproven && resumedAt > 0 && now > resumedAt {
		unproven = now - resumedAt
	}
	return WatchdogMTTRRow{
		Session:             f.session,
		Status:              status,
		Mode:                mode,
		DetectedAt:          f.detectedAt,
		ResumedAt:           resumedAt,
		ProgressWitnessedAt: progressAt,
		SilentSeconds:       silent,
		UnprovenSeconds:     unproven,
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

// watchdogPhaseUnknown marks a ledger row that never recorded its phase. A blank
// phase used to fold to "launched" (#3801), silently counting bookkeeping rows as
// real resumes and inflating the drain/launch metric; keeping it distinct makes a
// missing phase visible as missing, never a launch. Real launch rows always write
// an explicit "launched"/"resumed" phase.
const watchdogPhaseUnknown = "phase_unknown"

// normalizeWatchdogPhase folds a ledger row's phase to the token the fold reasons over: a
// blank phase folds to phase_unknown (never "launched"), so a missing phase stays visible as
// missing instead of minting a phantom launched_unproven MTTR row (#3801).
//
// The launch verdict itself is single-sourced in phaseIsLaunchToken (outcome.go), which both
// this fold and Attempt.IsLaunch consult, so the two readers agree on every non-empty token
// (#4333). They diverge on the empty phase ON PURPOSE: this fold treats it as phase_unknown
// (a launch it cannot prove, excluded from the MTTR view), while IsLaunch counts it as a
// fired spawn (the pre-phase legacy rows still burn attempt budget). See the shared table
// test TestPhaseClassifierSharedVocabulary (outcome_test.go).
func normalizeWatchdogPhase(phase string) string {
	phase = strings.ToLower(strings.TrimSpace(phase))
	if phase == "" {
		return watchdogPhaseUnknown
	}
	return phase
}

func watchdogMTTRRank(s WatchdogMTTRStatus) int {
	switch s {
	case WatchdogMTTRQueued:
		return 0
	case WatchdogMTTRAuthBlocked:
		return 1
	case WatchdogMTTRLaunchedUnproven:
		return 2
	default:
		return 3
	}
}

func planAuthBlocked(disps []string) bool {
	seen := false
	for _, disp := range disps {
		disp = strings.ToUpper(strings.TrimSpace(disp))
		if disp == "" {
			continue
		}
		seen = true
		if !strings.Contains(disp, "AUTH") {
			return false
		}
	}
	return seen
}

func uniqueNonEmpty(vals []string) []string {
	out := make([]string, 0, len(vals))
	seen := map[string]bool{}
	for _, v := range vals {
		v = strings.TrimSpace(v)
		if v == "" || seen[v] {
			continue
		}
		seen[v] = true
		out = append(out, v)
	}
	sort.Strings(out)
	return out
}

// watchdogAge renders a staleness age compactly for the "not launching" headline:
// days past a day, hours past an hour, else whole minutes — enough scale for a
// week-dead ledger to read as "7.0d ago", not "168.0h ago".
func watchdogAge(sec int64) string {
	switch {
	case sec >= 86400:
		return fmt.Sprintf("%.1fd", float64(sec)/86400)
	case sec >= 3600:
		return fmt.Sprintf("%.1fh", float64(sec)/3600)
	default:
		return fmt.Sprintf("%.0fm", float64(sec)/60)
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
