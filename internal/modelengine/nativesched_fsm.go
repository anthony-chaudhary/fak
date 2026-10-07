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
// (lightseekorg/tokenspeed tokenspeed-scheduler/csrc/fsm/states.h:32 at
// b174a3186d9a6eb3192389afbb25611a976eefc7, MIT). The upstream file is not
// vendored; only the "explicit transition table + checked setter" convention is
// borrowed. Scheduling decisions are unchanged: every edge the scheduler takes
// today is in the table.

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
