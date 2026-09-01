package orchestration

import (
	"errors"
	"sort"
	"strings"
	"time"
)

// WaitAuthority describes how strongly a harness wait observation can govern
// canonical worker state. Structured events are native; process inspection is
// retained only as a degraded fallback for harnesses without those events.
type WaitAuthority string

const (
	WaitAuthorityNative   WaitAuthority = "native"
	WaitAuthorityDegraded WaitAuthority = "degraded"
)

// WaitPhase is the canonical lifecycle phase of a bounded multi-worker wait.
type WaitPhase string

const (
	WaitPhaseStarted   WaitPhase = "started"
	WaitPhaseCompleted WaitPhase = "completed"
)

// WaitTargetStatus deliberately uses harness-neutral values. Adapters must map
// provider-specific status names before calling ReconcileWaitLifecycle.
type WaitTargetStatus string

const (
	WaitTargetRunning   WaitTargetStatus = "running"
	WaitTargetCompleted WaitTargetStatus = "completed"
	WaitTargetFailed    WaitTargetStatus = "failed"
)

// WaitTarget is one target returned by a structured wait completion.
type WaitTarget struct {
	ID     string           `json:"id"`
	Status WaitTargetStatus `json:"status"`
}

// WaitLifecycleEvent is the provider-neutral adapter boundary for a bounded
// worker wait. RequestedTimeout records caller intent; EffectiveTimeout records
// the positive timeout actually enforced by the harness.
type WaitLifecycleEvent struct {
	Phase            WaitPhase     `json:"phase"`
	TargetIDs        []string      `json:"target_ids"`
	RequestedTimeout time.Duration `json:"requested_timeout"`
	EffectiveTimeout time.Duration `json:"effective_timeout"`
	TimedOut         bool          `json:"timed_out,omitempty"`
	Targets          []WaitTarget  `json:"targets,omitempty"`
}

// WaitLifecycleState is safe to expose in status JSON: it contains identifiers,
// bounded timing, canonical statuses, and authority, but no prompts or paths.
type WaitLifecycleState struct {
	Phase            WaitPhase     `json:"phase"`
	TargetIDs        []string      `json:"target_ids"`
	RequestedTimeout time.Duration `json:"requested_timeout"`
	EffectiveTimeout time.Duration `json:"effective_timeout"`
	TimedOut         bool          `json:"timed_out"`
	Targets          []WaitTarget  `json:"targets,omitempty"`
	Authority        WaitAuthority `json:"authority"`
	Source           string        `json:"source"`
}

// ReconcileWaitLifecycle validates and canonicalizes a structured harness wait.
// A timeout is a completed bounded wait, not worker failure: running targets stay
// running until a later terminal completion reconciles them exactly once.
func ReconcileWaitLifecycle(event WaitLifecycleEvent) (WaitLifecycleState, error) {
	if event.Phase != WaitPhaseStarted && event.Phase != WaitPhaseCompleted {
		return WaitLifecycleState{}, errors.New("wait lifecycle phase is invalid")
	}
	if event.EffectiveTimeout <= 0 {
		return WaitLifecycleState{}, errors.New("wait lifecycle effective timeout must be positive")
	}
	ids, err := canonicalWaitIDs(event.TargetIDs)
	if err != nil {
		return WaitLifecycleState{}, err
	}
	state := WaitLifecycleState{
		Phase: event.Phase, TargetIDs: ids,
		RequestedTimeout: event.RequestedTimeout, EffectiveTimeout: event.EffectiveTimeout,
		TimedOut: event.TimedOut, Authority: WaitAuthorityNative, Source: "structured_wait",
	}
	if event.Phase == WaitPhaseStarted {
		if event.TimedOut || len(event.Targets) != 0 {
			return WaitLifecycleState{}, errors.New("wait start cannot contain completion state")
		}
		return state, nil
	}

	seen := make(map[string]struct{}, len(event.Targets))
	allowed := make(map[string]struct{}, len(ids))
	for _, id := range ids {
		allowed[id] = struct{}{}
	}
	for _, target := range event.Targets {
		target.ID = strings.TrimSpace(target.ID)
		if _, ok := allowed[target.ID]; !ok {
			return WaitLifecycleState{}, errors.New("wait result contains an unknown target")
		}
		if _, ok := seen[target.ID]; ok {
			return WaitLifecycleState{}, errors.New("wait result contains a duplicate target")
		}
		seen[target.ID] = struct{}{}
		switch target.Status {
		case WaitTargetRunning, WaitTargetCompleted, WaitTargetFailed:
		default:
			return WaitLifecycleState{}, errors.New("wait result contains an unknown target status")
		}
		state.Targets = append(state.Targets, target)
	}
	sort.Slice(state.Targets, func(i, j int) bool { return state.Targets[i].ID < state.Targets[j].ID })
	return state, nil
}

// DegradedWaitLifecycle records the legacy process-liveness fallback. Callers
// should use it only when the harness exposes no structured wait events.
func DegradedWaitLifecycle(targetIDs []string, effectiveTimeout time.Duration) (WaitLifecycleState, error) {
	ids, err := canonicalWaitIDs(targetIDs)
	if err != nil {
		return WaitLifecycleState{}, err
	}
	if effectiveTimeout <= 0 {
		return WaitLifecycleState{}, errors.New("wait lifecycle effective timeout must be positive")
	}
	return WaitLifecycleState{Phase: WaitPhaseStarted, TargetIDs: ids, EffectiveTimeout: effectiveTimeout, Authority: WaitAuthorityDegraded, Source: "process_liveness"}, nil
}

func canonicalWaitIDs(ids []string) ([]string, error) {
	seen := make(map[string]struct{}, len(ids))
	out := make([]string, 0, len(ids))
	for _, id := range ids {
		id = strings.TrimSpace(id)
		if id == "" {
			return nil, errors.New("wait target id is required")
		}
		if _, ok := seen[id]; ok {
			return nil, errors.New("wait target ids must be unique")
		}
		seen[id] = struct{}{}
		out = append(out, id)
	}
	if len(out) == 0 {
		return nil, errors.New("wait targets are required")
	}
	sort.Strings(out)
	return out, nil
}
