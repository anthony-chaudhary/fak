package dojo

import (
	"sort"
	"sync"
)

// run.go holds the environment/lever contract and the pure orchestration that
// turns (scenarios x levers) into scored episodes. A Scenario is the workload
// (an offline corpus of real transcripts today, a live feed later); a Lever is
// one optimization under test that declares a theory and measures reality over a
// scenario. The concrete levers (which read transcripts) live in the
// cmd/fak/dojo.go shell — this file stays pure so the loop is unit-testable with
// fake levers.

// Scenario is a workload the gym runs a lever against. Offline mode replays a
// corpus of recorded transcripts (deterministic, hardware-free, CI-able); the
// live mode (a running session feed) is the same shape with a different source.
type Scenario struct {
	Name   string `json:"name"`
	Mode   string `json:"mode"`   // "offline" (replay corpus) | "live" (future)
	Corpus string `json:"corpus"` // directory of transcripts, for offline mode
	Note   string `json:"note"`
}

// ScoredInput is one (prediction, outcome) pair a lever emits for one metric: the
// theory it declared and the billed reality it measured. Run scores it.
type ScoredInput struct {
	Prediction Prediction `json:"prediction"`
	Outcome    Outcome    `json:"outcome"`
}

// Lever is one optimization under test. Episodes runs it against a scenario and
// returns one ScoredInput per metric it scores (a single lever can score several
// metrics — e.g. posture accuracy AND cold-cost calibration). A lever that
// produced no comparable metric on this scenario returns an empty slice, not an
// error.
type Lever interface {
	Name() string
	Episodes(s Scenario) ([]ScoredInput, error)
}

var (
	leverRegistryMu sync.RWMutex
	leverRegistry   = map[string]Lever{}
)

// RegisterLever registers a Lever in the package-level registry so it is recognized
// across the dojo gym. It panics on a nil lever, an empty name, or a duplicate name.
func RegisterLever(l Lever) Lever {
	if l == nil {
		panic("dojo: cannot register nil lever")
	}
	name := l.Name()
	if name == "" {
		panic("dojo: cannot register lever with empty name")
	}
	leverRegistryMu.Lock()
	defer leverRegistryMu.Unlock()
	if _, exists := leverRegistry[name]; exists {
		panic("dojo: lever already registered: " + name)
	}
	leverRegistry[name] = l
	return l
}

// LookupLever retrieves a registered Lever by name.
func LookupLever(name string) (Lever, bool) {
	leverRegistryMu.RLock()
	defer leverRegistryMu.RUnlock()
	l, ok := leverRegistry[name]
	return l, ok
}

// RegisteredLevers returns all registered levers sorted by name.
func RegisteredLevers() []Lever {
	leverRegistryMu.RLock()
	defer leverRegistryMu.RUnlock()
	res := make([]Lever, 0, len(leverRegistry))
	for _, l := range leverRegistry {
		res = append(res, l)
	}
	sort.Slice(res, func(i, j int) bool {
		return res[i].Name() < res[j].Name()
	})
	return res
}

// RunError records a lever that failed to run against a scenario, so a single
// broken lever degrades to a recorded error rather than aborting the whole run.
type RunError struct {
	Lever    string `json:"lever"`
	Scenario string `json:"scenario"`
	Err      string `json:"err"`
}

// Run drives every lever over every scenario and scores each emitted input into
// an episode. It is deterministic in scenario-then-lever order and total: a
// lever error is collected (never fatal), and a lever with nothing comparable
// simply contributes no episodes.
func Run(scenarios []Scenario, levers []Lever, band CalibBand) ([]Episode, []RunError) {
	var episodes []Episode
	var errs []RunError
	for _, s := range scenarios {
		for _, lv := range levers {
			ins, err := lv.Episodes(s)
			if err != nil {
				errs = append(errs, RunError{Lever: lv.Name(), Scenario: s.Name, Err: err.Error()})
				continue
			}
			for _, in := range ins {
				episodes = append(episodes, Score(s.Name, in.Prediction, in.Outcome, band))
			}
		}
	}
	return episodes, errs
}
