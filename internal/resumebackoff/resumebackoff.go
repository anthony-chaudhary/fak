// Package resumebackoff contains the pure resume-storm containment fold.
package resumebackoff

import (
	"sort"
	"time"
)

const ReasonSignatureParked = "SIGNATURE_PARKED"
const ReasonBackoff = "SIGNATURE_BACKOFF"
const ReasonCrashLoopQuarantined = "CRASH_LOOP_QUARANTINED"

type Event struct {
	Session, Signature string
	At                 time.Time
}
type Input struct {
	Session, Signature    string
	Now                   time.Time
	History               []Event
	Base, Ceiling, Window time.Duration
	ParkThreshold         int
	CrashLoopBudget       int
}
type Decision struct {
	Eligible     bool          `json:"eligible"`
	Reason       string        `json:"reason"`
	Repeat       int           `json:"repeat"`
	Delay        time.Duration `json:"delay"`
	NextEligible time.Time     `json:"next_eligible,omitempty"`
	Parked       bool          `json:"parked"`
	Quarantined  bool          `json:"quarantined,omitempty"`
	Sessions     []string      `json:"sessions,omitempty"`
}

func Decide(in Input) Decision {
	if in.Base <= 0 {
		in.Base = time.Minute
	}
	if in.Ceiling <= 0 {
		in.Ceiling = time.Hour
	}
	if in.Window <= 0 {
		in.Window = time.Hour
	}
	if in.ParkThreshold <= 0 {
		in.ParkThreshold = 3
	}
	if in.CrashLoopBudget <= 0 {
		in.CrashLoopBudget = 3
	}
	sessions := map[string]bool{}
	cutoff := in.Now.Add(-in.Window)
	var own []Event
	for _, e := range in.History {
		if e.Signature != in.Signature {
			continue
		}
		if !e.At.Before(cutoff) {
			sessions[e.Session] = true
		}
		if e.Session == in.Session && !e.At.Before(cutoff) {
			own = append(own, e)
		}
	}
	if in.Session != "" {
		sessions[in.Session] = true
	}
	names := make([]string, 0, len(sessions))
	for s := range sessions {
		names = append(names, s)
	}
	sort.Strings(names)
	if len(names) >= in.ParkThreshold {
		return Decision{Reason: ReasonSignatureParked, Parked: true, Sessions: names}
	}
	// Consecutive repeats reset when this session's latest signature changed.
	latestSig := ""
	latestAt := time.Time{}
	for _, e := range in.History {
		if e.Session == in.Session && e.At.After(latestAt) {
			latestAt = e.At
			latestSig = e.Signature
		}
	}
	if latestSig != in.Signature {
		return Decision{Eligible: true, Sessions: names}
	}
	sort.Slice(own, func(i, j int) bool { return own[i].At.Before(own[j].At) })
	repeat := len(own)
	if repeat >= in.CrashLoopBudget {
		return Decision{
			Eligible:    false,
			Reason:      ReasonCrashLoopQuarantined,
			Repeat:      repeat,
			Parked:      true,
			Quarantined: true,
			Sessions:    names,
		}
	}
	delay := in.Base
	for i := 1; i < repeat; i++ {
		if delay >= in.Ceiling/2 {
			delay = in.Ceiling
			break
		}
		delay *= 2
	}
	if delay > in.Ceiling {
		delay = in.Ceiling
	}
	next := latestAt.Add(delay)
	if in.Now.Before(next) {
		return Decision{Reason: ReasonBackoff, Repeat: repeat, Delay: delay, NextEligible: next, Sessions: names}
	}
	return Decision{Eligible: true, Repeat: repeat, Delay: delay, NextEligible: next, Sessions: names}
}
