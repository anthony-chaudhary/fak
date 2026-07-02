// Package operatortouches is the R1 babysitting counter (#2270, epic #2269):
// a pure fold over loop-event ledgers (internal/loopmgr, fak.loop-event.v1)
// that measures how much HUMAN supervision a fleet actually consumed, per
// witnessed unit of shipped work. Spine: docs/notes/CONCEPT-NO-BABYSITTING-2026-07-01.md
// ("nobody watches a healthy fleet" — touches per witnessed unit is a ratchet
// that only goes down; you cannot retire what you cannot count).
//
// The classifier is data over ledger rows — source/principal/reason tokens
// against declared tables — never a judge model reading transcripts. A source
// neither table claims is counted as UNKNOWN and surfaced for classification,
// not silently binned.
//
// v0 honesty fences: mttr_sessions and escalation_handling_p50 report not_yet
// until their source rows exist (the resume-drain witness #2273 / #1146, and
// the R2 escalation packet's ack row #2271). silent_hours is computed from
// the loop ledger alone (heartbeat/end gaps), not yet from a liveness oracle
// (#750).
package operatortouches

import (
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/loopmgr"
)

// Schema tags the JSON report.
const Schema = "fak.operator-touches.v1"

// DefaultSilentBound is the silence threshold past which a run with no fresh
// ledger row counts as sitting undetected (law B4: silence is bounded).
const DefaultSilentBound = 30 * time.Minute

// SourceClass is the human/machine/unknown verdict for an event source.
type SourceClass string

const (
	SourceHuman   SourceClass = "human"
	SourceMachine SourceClass = "machine"
	SourceUnknown SourceClass = "unknown"
)

// MachineSources are trigger sources that are schedulers or registered
// producers — never a human touch. Data, not heuristic: extend the table when
// a new producer ships, never infer.
var MachineSources = map[string]bool{
	"cron": true, "launchd": true, "systemd": true, "task-scheduler": true,
	"schedule": true, "timer": true, "github": true, "ci": true, "api": true,
	"watchdog": true, "supervisor": true, "bgloop": true,
	"issue_resolve_dispatch": true, "issue_resolve_progress": true,
}

// HumanSources are control-plane sources that carry a human principal's
// intent onto the ledger.
var HumanSources = map[string]bool{
	"human": true, "manual": true, "operator": true, "cli": true,
	"terminal": true, "slack": true, "chat": true, "whatsapp": true,
	"phone": true,
}

// ClassifySource is the data-table verdict for one source string.
func ClassifySource(source string) SourceClass {
	s := strings.ToLower(strings.TrimSpace(source))
	if s == "" {
		return SourceUnknown
	}
	if MachineSources[s] {
		return SourceMachine
	}
	if HumanSources[s] {
		return SourceHuman
	}
	return SourceUnknown
}

// The seven watches from the spine note, plus the honest bucket.
const (
	WatchLiveness     = "liveness"
	WatchTruth        = "truth"
	WatchSafety       = "safety"
	WatchCollision    = "collision"
	WatchBudget       = "budget"
	WatchUnblock      = "unblock"
	WatchQuality      = "quality"
	WatchUnclassified = "unclassified"
)

// reasonHints maps reason-token substrings (upper-cased match) to a watch.
// Order matters: first hit wins. Declared data, tuned as real reasons appear.
var reasonHints = []struct {
	needle string
	watch  string
}{
	{"RESUME", WatchLiveness}, {"STUCK", WatchLiveness}, {"DEAD", WatchLiveness},
	{"CRASH", WatchLiveness}, {"HANG", WatchLiveness}, {"THROTTLE", WatchLiveness},
	{"429", WatchLiveness}, {"529", WatchLiveness},
	{"WITNESS", WatchTruth}, {"CLAIM", WatchTruth},
	{"POLICY", WatchSafety}, {"DENY", WatchSafety}, {"LEAK", WatchSafety},
	{"QUARANT", WatchSafety},
	{"COLLISION", WatchCollision}, {"LEASE", WatchCollision}, {"MERGE", WatchCollision},
	{"BUDGET", WatchBudget}, {"OVERHEAD", WatchBudget},
	{"ESCALAT", WatchUnblock}, {"APPROV", WatchUnblock}, {"AUTH", WatchUnblock},
	{"LOGIN", WatchUnblock},
	{"QUALITY", WatchQuality}, {"REVIEW", WatchQuality},
}

// kindWatch is the fallback classification when the reason carries no hint:
// what a human doing this KIND of control event is most plausibly watching.
var kindWatch = map[loopmgr.EventKind]string{
	loopmgr.EventFire:      WatchLiveness,
	loopmgr.EventArmed:     WatchLiveness,
	loopmgr.EventStart:     WatchLiveness,
	loopmgr.EventHeartbeat: WatchLiveness,
	loopmgr.EventEnd:       WatchLiveness,
	loopmgr.EventWitness:   WatchTruth,
	loopmgr.EventNotify:    WatchUnblock,
	loopmgr.EventAdmit:     WatchCollision,
}

// WatchFor classifies one event into a watch: reason tokens first, then the
// event kind, then the honest bucket.
func WatchFor(ev loopmgr.Event) string {
	reason := strings.ToUpper(ev.Reason)
	for _, h := range reasonHints {
		if strings.Contains(reason, h.needle) {
			return h.watch
		}
	}
	if w, ok := kindWatch[ev.Kind]; ok {
		return w
	}
	return WatchUnclassified
}

// KPIStatus is measured-or-not-yet; a not_yet always names its missing witness.
type KPIStatus string

const (
	KPIMeasured KPIStatus = "measured"
	KPINotYet   KPIStatus = "not_yet"
)

// KPI is one headline number with provenance posture.
type KPI struct {
	Status  KPIStatus `json:"status"`
	Value   float64   `json:"value,omitempty"`
	Unit    string    `json:"unit,omitempty"`
	Missing string    `json:"missing,omitempty"`
}

// Touch is one human-classified control event, with the row identity that
// proves it (ledger seq + loop id).
type Touch struct {
	Seq        uint64 `json:"seq"`
	TSUnixNano int64  `json:"ts_unix_nano"`
	LoopID     string `json:"loop_id"`
	RunID      string `json:"run_id,omitempty"`
	Kind       string `json:"kind"`
	Source     string `json:"source"`
	Principal  string `json:"principal,omitempty"`
	Reason     string `json:"reason,omitempty"`
	Watch      string `json:"watch"`
}

// SilentRun is one run that sat past the silence bound with no fresh row.
type SilentRun struct {
	LoopID       string  `json:"loop_id"`
	RunID        string  `json:"run_id"`
	LastKind     string  `json:"last_kind"`
	SilentHours  float64 `json:"silent_hours"`
	Open         bool    `json:"open"` // true: still no terminal row at AsOf
	LastUnixNano int64   `json:"last_unix_nano"`
}

// Report is the fak.operator-touches.v1 fold.
type Report struct {
	Schema       string `json:"schema"`
	AsOfUnixNano int64  `json:"as_of_unix_nano"`

	Events         int            `json:"events"`
	Touches        []Touch        `json:"touches"`
	TouchesByWatch map[string]int `json:"touches_by_watch"`
	UnknownSources map[string]int `json:"unknown_sources,omitempty"`

	WitnessedDone  int `json:"witnessed_done"`
	ClaimedDone    int `json:"claimed_done"`
	WitnessRefused int `json:"witness_refused"`

	TouchesPerWitnessedUnit KPI `json:"touches_per_witnessed_unit"`
	SilentHours             KPI `json:"silent_hours"`
	MTTRSessions            KPI `json:"mttr_sessions"`
	EscalationHandlingP50   KPI `json:"escalation_handling_p50"`

	SilentRuns []SilentRun `json:"silent_runs,omitempty"`
}

// Params tunes the fold; zero values take the documented defaults.
type Params struct {
	AsOf        time.Time
	SilentBound time.Duration
}

// Fold computes the R1 report from loop-event rows (any number of ledgers,
// pre-concatenated). Pure: no I/O, no clock — AsOf comes from the caller.
func Fold(events []loopmgr.Event, p Params) Report {
	if p.SilentBound <= 0 {
		p.SilentBound = DefaultSilentBound
	}
	asOf := p.AsOf
	if asOf.IsZero() && len(events) > 0 {
		// Deterministic fallback: the newest row. Callers wanting "now" pass it.
		var maxTS int64
		for _, ev := range events {
			if ev.TSUnixNano > maxTS {
				maxTS = ev.TSUnixNano
			}
		}
		asOf = time.Unix(0, maxTS).UTC()
	}

	r := Report{
		Schema:         Schema,
		AsOfUnixNano:   asOf.UnixNano(),
		Events:         len(events),
		TouchesByWatch: map[string]int{},
		UnknownSources: map[string]int{},
	}

	type runKey struct{ loop, run string }
	runs := map[runKey][]loopmgr.Event{}

	for _, ev := range events {
		switch ClassifySource(ev.Source) {
		case SourceHuman:
			watch := WatchFor(ev)
			r.Touches = append(r.Touches, Touch{
				Seq: ev.Seq, TSUnixNano: ev.TSUnixNano,
				LoopID: ev.LoopID, RunID: ev.RunID,
				Kind: string(ev.Kind), Source: ev.Source,
				Principal: ev.Principal, Reason: ev.Reason, Watch: watch,
			})
			r.TouchesByWatch[watch]++
		case SourceUnknown:
			if s := strings.TrimSpace(ev.Source); s != "" {
				r.UnknownSources[s]++
			}
		}

		if ev.Kind == loopmgr.EventWitness || ev.Kind == loopmgr.EventEnd {
			switch ev.Status {
			case loopmgr.StatusWitnessedDone:
				r.WitnessedDone++
			case loopmgr.StatusClaimedDone:
				r.ClaimedDone++
			case loopmgr.StatusWitnessRefused:
				r.WitnessRefused++
			}
		}

		if ev.RunID != "" {
			k := runKey{ev.LoopID, ev.RunID}
			runs[k] = append(runs[k], ev)
		}
	}
	sort.Slice(r.Touches, func(i, j int) bool { return r.Touches[i].TSUnixNano < r.Touches[j].TSUnixNano })

	// Silent hours: per run, the gap past the bound with no fresh row —
	// open (no terminal end row at AsOf) or closed (a terminal failure end
	// arrived only after the gap; the gap is how long it sat unnoticed).
	var totalSilent float64
	keys := make([]runKey, 0, len(runs))
	for k := range runs {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool {
		if keys[i].loop != keys[j].loop {
			return keys[i].loop < keys[j].loop
		}
		return keys[i].run < keys[j].run
	})
	for _, k := range keys {
		evs := runs[k]
		sort.Slice(evs, func(i, j int) bool { return evs[i].TSUnixNano < evs[j].TSUnixNano })
		started := false
		var end *loopmgr.Event
		for i := range evs {
			switch evs[i].Kind {
			case loopmgr.EventStart:
				started = true
			case loopmgr.EventEnd:
				end = &evs[i]
			}
		}
		if !started {
			continue
		}
		if end == nil {
			last := evs[len(evs)-1]
			gap := asOf.Sub(time.Unix(0, last.TSUnixNano))
			if gap > p.SilentBound {
				h := gap.Hours()
				totalSilent += h
				r.SilentRuns = append(r.SilentRuns, SilentRun{
					LoopID: k.loop, RunID: k.run, LastKind: string(last.Kind),
					SilentHours: h, Open: true, LastUnixNano: last.TSUnixNano,
				})
			}
			continue
		}
		if end.Status == loopmgr.StatusFailed || end.Status == loopmgr.StatusCanceled {
			// The last sign of life before the terminal row.
			var prev *loopmgr.Event
			for i := range evs {
				if evs[i].TSUnixNano < end.TSUnixNano && evs[i].Kind != loopmgr.EventEnd {
					prev = &evs[i]
				}
			}
			if prev != nil {
				gap := time.Unix(0, end.TSUnixNano).Sub(time.Unix(0, prev.TSUnixNano))
				if gap > p.SilentBound {
					h := gap.Hours()
					totalSilent += h
					r.SilentRuns = append(r.SilentRuns, SilentRun{
						LoopID: k.loop, RunID: k.run, LastKind: string(prev.Kind),
						SilentHours: h, Open: false, LastUnixNano: prev.TSUnixNano,
					})
				}
			}
		}
	}
	r.SilentHours = KPI{Status: KPIMeasured, Value: totalSilent, Unit: "hours"}

	if r.WitnessedDone > 0 {
		r.TouchesPerWitnessedUnit = KPI{
			Status: KPIMeasured,
			Value:  float64(len(r.Touches)) / float64(r.WitnessedDone),
			Unit:   "touches/witnessed_done",
		}
	} else {
		r.TouchesPerWitnessedUnit = KPI{
			Status:  KPINotYet,
			Missing: "no witnessed_done rows in the window (denominator 0) — counts reported, ratio undefined",
		}
	}

	// Escalation handling: needs the R2 packet's ack row (#2271). If a notify
	// row already carries an ack_unix_nano metric, compute the p50; otherwise
	// not_yet with the missing witness named.
	var handling []float64
	for _, ev := range events {
		if ev.Kind != loopmgr.EventNotify || ev.Metrics == nil {
			continue
		}
		if ack, ok := ev.Metrics["ack_unix_nano"]; ok && ack > ev.TSUnixNano {
			handling = append(handling, time.Duration(ack-ev.TSUnixNano).Seconds())
		}
	}
	if len(handling) > 0 {
		sort.Float64s(handling)
		r.EscalationHandlingP50 = KPI{Status: KPIMeasured, Value: handling[len(handling)/2], Unit: "seconds"}
	} else {
		r.EscalationHandlingP50 = KPI{
			Status:  KPINotYet,
			Missing: "no acknowledged notify rows — needs the fak.escalation.v1 ack row (#2271)",
		}
	}

	r.MTTRSessions = KPI{
		Status:  KPINotYet,
		Missing: "resume watchdog ledger fold not wired — needs the recovery drain witness (#2273 / #1146)",
	}

	return r
}
