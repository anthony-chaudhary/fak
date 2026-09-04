package fleetbottleneck

import "sort"

// Class designates the category of fleet bottleneck identified during evaluation.
type Class string

const (
	// ClassSeats indicates saturation of available execution slots across machines.
	ClassSeats Class = "seats"
	// ClassThrottle indicates execution capacity diminished by active rate limits.
	ClassThrottle Class = "throttle"
	// ClassResume indicates worker backlog waiting on session restoration.
	ClassResume Class = "resume"
	// ClassHost indicates compute host resource strain or high CPU load.
	ClassHost Class = "host"
	// ClassAuth indicates worker starvation due to invalid or missing credentials.
	ClassAuth Class = "auth"
)

// Snapshot captures operational fleet metrics across machines, seats, and workers.
type Snapshot struct {
	Machines       int
	Sessions       int
	SeatCapacity   int
	ThrottledSeats int
	HealthySeats   int
	ResumeBacklog  int
	HostLoad       float64
	AuthBlocked    int
}

// Bottleneck describes an identified constraint with its severity score and diagnostic text.
type Bottleneck struct {
	Class    Class
	Score    float64
	Evidence string
}

// Rank evaluates fleet operational metrics and orders active constraints by severity.
//
// Invariant: fleet bottleneck ranking is fail-closed and deterministic.
// Guard: empty snapshots or non-positive machine counts return nil without error.
// Precondition: caller provides a valid Snapshot with non-negative metrics.
func Rank(s Snapshot) []Bottleneck {
	if s.Machines <= 0 {
		return nil
	}
	var out []Bottleneck
	add := func(class Class, score float64, evidence string) {
		if score > 0 {
			out = append(out, Bottleneck{class, score, evidence})
		}
	}
	if s.AuthBlocked > 0 {
		add(ClassAuth, .9+min(float64(s.AuthBlocked)/10, .1), itoa(s.AuthBlocked)+" auth-blocked")
	}
	if s.ThrottledSeats > 0 {
		denom := s.ThrottledSeats + s.HealthySeats
		ratio := 1.0
		if denom > 0 {
			ratio = float64(s.ThrottledSeats) / float64(denom)
		}
		add(ClassThrottle, .75+.25*ratio, itoa(s.ThrottledSeats)+" throttled seats")
	}
	if s.ResumeBacklog > 0 {
		add(ClassResume, .7+.25*min(float64(s.ResumeBacklog)/20, 1), itoa(s.ResumeBacklog)+" queued resumes")
	}
	if s.HostLoad >= .8 {
		add(ClassHost, .65+.3*min((s.HostLoad-.8)/.2, 1), percent(s.HostLoad)+" host load")
	}
	if s.SeatCapacity > 0 && s.Sessions >= s.SeatCapacity {
		add(ClassSeats, .8+min(float64(s.Sessions-s.SeatCapacity)/float64(s.SeatCapacity), .15), itoa(s.Sessions)+"/"+itoa(s.SeatCapacity)+" sessions")
	}
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].Score != out[j].Score {
			return out[i].Score > out[j].Score
		}
		return out[i].Class < out[j].Class
	})
	return out
}
func min(a, b float64) float64 {
	if a < b {
		return a
	}
	return b
}
func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	b := make([]byte, 0, 12)
	for n > 0 {
		b = append([]byte{byte('0' + n%10)}, b...)
		n /= 10
	}
	return string(b)
}
func percent(v float64) string { return itoa(int(v*100)) + "%" }
