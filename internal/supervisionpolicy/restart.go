package supervisionpolicy

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"time"
)

// RestartClass declares which exits make a child eligible for restart.
type RestartClass string

const (
	RestartPermanent RestartClass = "permanent"
	RestartTransient RestartClass = "transient"
	RestartTemporary RestartClass = "temporary"
)

// FailureKind classifies an observed child exit. Deterministic failures are
// quarantined immediately rather than consuming resources in a retry loop.
type FailureKind string

const (
	FailureClean         FailureKind = "clean"
	FailureTransient     FailureKind = "transient"
	FailureDeterministic FailureKind = "deterministic"
)

// EscalationTarget bounds the blast radius when a child cannot be restarted.
type EscalationTarget string

const (
	EscalateChild  EscalationTarget = "child"
	EscalateDomain EscalationTarget = "domain"
)

// ChildSpec is the portable supervision contract attached to a launched child.
// Independent children must use one-for-one strategy; shared fate is therefore
// explicit in serialized launch plans rather than inferred by a watchdog.
type ChildSpec struct {
	Owner         string           `json:"owner"`
	FaultDomain   DomainID         `json:"fault_domain"`
	Restart       RestartClass     `json:"restart"`
	Strategy      Strategy         `json:"strategy"`
	MaxAttempts   uint32           `json:"max_attempts"`
	RollingWindow time.Duration    `json:"rolling_window"`
	BaseBackoff   time.Duration    `json:"base_backoff"`
	MaxBackoff    time.Duration    `json:"max_backoff"`
	Jitter        time.Duration    `json:"jitter"`
	StableReset   time.Duration    `json:"stable_reset"`
	Escalation    EscalationTarget `json:"escalation"`
}

// IndependentChildSpec returns the gen/next default used by orchestration.
func IndependentChildSpec(owner string, domain DomainID) ChildSpec {
	return ChildSpec{Owner: owner, FaultDomain: domain, Restart: RestartTransient,
		Strategy: StrategyOneForOne, MaxAttempts: 3, RollingWindow: 5 * time.Minute,
		BaseBackoff: time.Second, MaxBackoff: 30 * time.Second, Jitter: 250 * time.Millisecond,
		StableReset: 10 * time.Minute, Escalation: EscalateChild}
}

// ChildState is durable crash-loop accounting for one child identity.
type ChildState struct {
	Failures    []time.Time `json:"failures,omitempty"`
	Quarantined bool        `json:"quarantined,omitempty"`
	Reason      string      `json:"reason,omitempty"`
	LastReceipt EvidenceRef `json:"last_receipt,omitempty"`
}

// RestartDecision is the typed effect returned after recording an exit.
type RestartDecision struct {
	Action     Action           `json:"action"`
	Outcome    Outcome          `json:"outcome"`
	Escalation EscalationTarget `json:"escalation,omitempty"`
	RetryAt    time.Time        `json:"retry_at,omitempty"`
	State      ChildState       `json:"state"`
}

// StateStore persists per-child counters across coordinator restarts.
type StateStore interface {
	Load(MemberID) (ChildState, error)
	Save(MemberID, ChildState) error
}

// RecordExit atomically evaluates one child exit against its persisted budget.
func RecordExit(store StateStore, child MemberID, spec ChildSpec, kind FailureKind, started, now time.Time, receipt EvidenceRef) (RestartDecision, error) {
	if store == nil || child == "" || !validChildSpec(spec) || now.Before(started) {
		return RestartDecision{Action: ActionHold, Outcome: OutcomeInvalidBudget}, errors.New("invalid child supervision request")
	}
	state, err := store.Load(child)
	if err != nil {
		return RestartDecision{}, err
	}
	if now.Sub(started) >= spec.StableReset {
		state.Failures = nil
		state.Quarantined = false
		state.Reason = ""
	}
	state.LastReceipt = receipt
	state.Failures = failuresWithin(state.Failures, now.Add(-spec.RollingWindow), now)

	eligible := spec.Restart == RestartPermanent || (spec.Restart == RestartTransient && kind != FailureClean)
	if kind == FailureDeterministic {
		state.Quarantined, state.Reason = true, "deterministic_failure"
	} else if !eligible {
		state.Reason = "restart_not_requested"
	} else if uint32(len(state.Failures)) >= spec.MaxAttempts {
		state.Quarantined, state.Reason = true, "restart_exhausted"
	} else {
		state.Failures = append(state.Failures, now)
		if err := store.Save(child, state); err != nil {
			return RestartDecision{}, err
		}
		return RestartDecision{Action: ActionRestart, Outcome: OutcomeRecover, RetryAt: now.Add(exponentialBackoff(spec, child, uint32(len(state.Failures)))), State: state}, nil
	}
	if err := store.Save(child, state); err != nil {
		return RestartDecision{}, err
	}
	action, outcome := ActionHold, OutcomeStrategyHold
	if state.Quarantined {
		outcome = OutcomeBudgetExhausted
		if spec.Escalation == EscalateDomain {
			action = ActionEscalate
		}
	}
	return RestartDecision{Action: action, Outcome: outcome, Escalation: spec.Escalation, State: state}, nil
}

func validChildSpec(s ChildSpec) bool {
	validRestart := s.Restart == RestartPermanent || s.Restart == RestartTransient || s.Restart == RestartTemporary
	validEscalation := s.Escalation == EscalateChild || s.Escalation == EscalateDomain
	return s.Owner != "" && s.FaultDomain != "" && validRestart && validStrategy(s.Strategy) &&
		s.MaxAttempts > 0 && s.RollingWindow > 0 && s.StableReset > 0 && s.BaseBackoff >= 0 &&
		s.MaxBackoff >= s.BaseBackoff && s.Jitter >= 0 && validEscalation
}

func failuresWithin(in []time.Time, from, now time.Time) []time.Time {
	out := make([]time.Time, 0, len(in))
	for _, failure := range in {
		if !failure.Before(from) && !failure.After(now) {
			out = append(out, failure)
		}
	}
	return out
}

func exponentialBackoff(s ChildSpec, child MemberID, attempt uint32) time.Duration {
	delay := s.BaseBackoff
	for i := uint32(1); i < attempt && delay < s.MaxBackoff; i++ {
		if delay > s.MaxBackoff/2 {
			delay = s.MaxBackoff
		} else {
			delay *= 2
		}
	}
	if delay > s.MaxBackoff {
		delay = s.MaxBackoff
	}
	if s.Jitter == 0 {
		return delay
	}
	sum := sha256.Sum256([]byte(fmt.Sprintf("%s/%d", child, attempt)))
	return delay + time.Duration(binary.LittleEndian.Uint64(sum[:8])%uint64(s.Jitter+1))
}

// FileStore is a JSON state store. Save uses replace-by-rename so a coordinator
// crash cannot expose a partially written crash-loop budget.
type FileStore struct{ Path string }
type stateFile struct {
	Children map[MemberID]ChildState `json:"children"`
}

func (s FileStore) Load(child MemberID) (ChildState, error) {
	all, err := s.load()
	if err != nil {
		return ChildState{}, err
	}
	return all.Children[child], nil
}
func (s FileStore) Save(child MemberID, state ChildState) error {
	all, err := s.load()
	if err != nil {
		return err
	}
	all.Children[child] = state
	data, err := json.MarshalIndent(all, "", "  ")
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(s.Path), 0o755); err != nil {
		return err
	}
	tmp := s.Path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o600); err != nil {
		return err
	}
	return os.Rename(tmp, s.Path)
}
func (s FileStore) load() (stateFile, error) {
	all := stateFile{Children: map[MemberID]ChildState{}}
	data, err := os.ReadFile(s.Path)
	if errors.Is(err, os.ErrNotExist) {
		return all, nil
	}
	if err != nil {
		return all, err
	}
	if err := json.Unmarshal(data, &all); err != nil {
		return all, err
	}
	if all.Children == nil {
		all.Children = map[MemberID]ChildState{}
	}
	return all, nil
}
