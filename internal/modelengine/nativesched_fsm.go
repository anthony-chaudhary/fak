// Copyright (c) 2026 LightSeek Foundation
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

package modelengine

import "fmt"

// Native scheduler lane FSM.
//
// A lane's lifecycle phase is DERIVED, not stored: schedLane.state only records
// prefilling vs decode, while waiting and preempted are scheduler-list membership
// (s.waiting / s.preempted) and terminal is schedLane.terminal. This file adds no
// new state; it declares which phase-to-phase edges are legal and gives the
// state-assignment sites one checked entry point (setLaneStateLocked) so an illegal
// edge is a typed error instead of a silent overwrite.
//
// Adapted from tokenspeed's typed request FSM with declared legal transitions
// (lightseekorg/tokenspeed at b174a3186d9a6eb3192389afbb25611a976eefc7, MIT).
// The upstream file is not vendored; only the "explicit transition table +
// checked setter" convention is borrowed, with runtime rather than
// compile-time checking. Scheduling decisions are unchanged: every edge the
// scheduler takes today is in the table.

type schedLanePhase uint8

const (
	schedPhaseWaiting schedLanePhase = iota
	schedPhasePrefilling
	schedPhaseDecode
	schedPhasePreempted
	schedPhaseTerminal
)

func (p schedLanePhase) String() string {
	switch p {
	case schedPhaseWaiting:
		return "waiting"
	case schedPhasePrefilling:
		return "prefilling"
	case schedPhaseDecode:
		return "decode"
	case schedPhasePreempted:
		return "preempted"
	case schedPhaseTerminal:
		return "terminal"
	}
	return fmt.Sprintf("phase(%d)", uint8(p))
}

// schedLaneTransitions is the declared set of legal phase edges. Self edges are
// legal (idempotent re-assignment). Terminal is absorbing. decode->prefilling is
// only reachable through preemption (decode->preempted->prefilling).
var schedLaneTransitions = map[schedLanePhase]map[schedLanePhase]bool{
	schedPhaseWaiting: {
		schedPhaseWaiting:    true,
		schedPhasePrefilling: true,
		schedPhaseDecode:     true,
		schedPhaseTerminal:   true,
	},
	schedPhasePrefilling: {
		schedPhasePrefilling: true,
		schedPhaseDecode:     true,
		schedPhasePreempted:  true,
		schedPhaseTerminal:   true,
	},
	schedPhaseDecode: {
		schedPhaseDecode:    true,
		schedPhasePreempted: true,
		schedPhaseTerminal:  true,
	},
	schedPhasePreempted: {
		schedPhasePreempted:  true,
		schedPhasePrefilling: true,
		schedPhaseDecode:     true,
		schedPhaseTerminal:   true,
	},
	schedPhaseTerminal: {
		schedPhaseTerminal: true,
	},
}

// errSchedLaneIllegalTransition is the typed refusal for an edge outside
// schedLaneTransitions.
type errSchedLaneIllegalTransition struct {
	From, To schedLanePhase
}

func (e *errSchedLaneIllegalTransition) Error() string {
	return fmt.Sprintf("modelengine: illegal native scheduler lane transition %s -> %s", e.From, e.To)
}

func checkSchedLaneTransition(from, to schedLanePhase) error {
	if schedLaneTransitions[from][to] {
		return nil
	}
	return &errSchedLaneIllegalTransition{From: from, To: to}
}

func schedLanePhaseForState(st schedLaneState) schedLanePhase {
	if st == schedLanePrefilling {
		return schedPhasePrefilling
	}
	return schedPhaseDecode
}

// lanePhaseLocked derives ln's current phase. Caller holds s.mu.
func (s *NativeScheduler) lanePhaseLocked(ln *schedLane) schedLanePhase {
	if ln.terminal {
		return schedPhaseTerminal
	}
	for _, p := range s.preempted {
		if p == ln {
			return schedPhasePreempted
		}
	}
	for _, w := range s.waiting {
		if w == ln {
			return schedPhaseWaiting
		}
	}
	return schedLanePhaseForState(ln.state)
}

// setLaneStateLocked assigns ln.state = to after checking the derived phase edge
// against schedLaneTransitions. On an illegal edge the lane is left unchanged and
// a *errSchedLaneIllegalTransition is returned. Caller holds s.mu.
func (s *NativeScheduler) setLaneStateLocked(ln *schedLane, to schedLaneState) error {
	if err := checkSchedLaneTransition(s.lanePhaseLocked(ln), schedLanePhaseForState(to)); err != nil {
		return err
	}
	ln.state = to
	return nil
}
