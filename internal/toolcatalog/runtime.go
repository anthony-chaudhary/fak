package toolcatalog

import (
	"errors"
	"fmt"
	"sync"
)

// AdmitCode is a closed refusal class for runtime registration.
type AdmitCode string

const (
	AdmitMalformed       AdmitCode = "MALFORMED_PROGRAM"
	AdmitUnknownVersion  AdmitCode = "UNKNOWN_VERSION"
	AdmitDigestMismatch  AdmitCode = "DIGEST_MISMATCH"
	AdmitPolicyWidening  AdmitCode = "POLICY_FLOOR_WIDENING"
	AdmitNameCollision   AdmitCode = "NAME_COLLISION"
	AdmitPolicyDeny      AdmitCode = "POLICY_DENY"
	AdmitSnapshotFailure AdmitCode = "SNAPSHOT_FAILURE"
)

// AdmitError preserves a machine-readable runtime-admit refusal.
type AdmitError struct {
	Code AdmitCode
	Err  error
}

func (e *AdmitError) Error() string { return fmt.Sprintf("%s: %v", e.Code, e.Err) }
func (e *AdmitError) Unwrap() error { return e.Err }

// IsAdmitCode reports whether err carries code.
func IsAdmitCode(err error, code AdmitCode) bool {
	var target *AdmitError
	return errors.As(err, &target) && target.Code == code
}

// LiveFailure is a closed failure class eligible for automatic revert.
type LiveFailure string

const (
	LiveFailureExecutor    LiveFailure = "EXECUTOR_ERROR"
	LiveFailureResultAdmit LiveFailure = "RESULT_ADMIT_RED"
	LiveFailureQuarantine  LiveFailure = "QUARANTINE_TRIP"
)

// SwapKind describes a witnessed catalog generation transition.
type SwapKind string

const (
	SwapAdmit      SwapKind = "admit"
	SwapAutoRevert SwapKind = "auto_revert"
	SwapRevert     SwapKind = "revert"
)

// RuntimeSwap is the journal-ready record of one live generation transition.
type RuntimeSwap struct {
	Kind          SwapKind `json:"kind"`
	PriorDigest   string   `json:"prior_digest"`
	CurrentDigest string   `json:"current_digest"`
	Registration  string   `json:"registration_digest,omitempty"`
	Reason        string   `json:"reason,omitempty"`
}

// RuntimeHooks bind the catalog-only generation manager to policy, page-table,
// and journal owners without introducing upward package dependencies.
type RuntimeHooks struct {
	Policy   func(Registration) error
	Register func(Snapshot) error
	Swap     func(RuntimeSwap)
}

type runtimeAdmission struct {
	prior string
	calls int
}

// Runtime owns immutable, digest-addressed catalog generations for a running host.
type Runtime struct {
	mu            sync.RWMutex
	registrations []Registration
	current       Snapshot
	snapshots     map[string]Snapshot
	admitted      map[string]runtimeAdmission
	hooks         RuntimeHooks
}

// NewRuntime starts a generation manager at an already validated snapshot.
func NewRuntime(registrations []Registration, initial Snapshot, hooks RuntimeHooks) (*Runtime, error) {
	if err := ValidateSnapshot(initial); err != nil {
		return nil, err
	}
	for _, registration := range registrations {
		if err := ValidateRegistration(registration); err != nil {
			return nil, err
		}
	}
	r := &Runtime{
		registrations: append([]Registration(nil), registrations...),
		current:       cloneSnapshot(initial),
		snapshots:     map[string]Snapshot{initial.Digest: cloneSnapshot(initial)},
		admitted:      make(map[string]runtimeAdmission),
		hooks:         hooks,
	}
	return r, nil
}

// Pin returns the immutable generation used for one turn.
func (r *Runtime) Pin() Snapshot {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return cloneSnapshot(r.current)
}

// Snapshot resolves a historical generation for replay.
func (r *Runtime) Snapshot(digest string) (Snapshot, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	snapshot, ok := r.snapshots[digest]
	if !ok {
		return Snapshot{}, fmt.Errorf("TOOL_SNAPSHOT_UNKNOWN: %s", digest)
	}
	return cloneSnapshot(snapshot), nil
}

// Admit verifies and publishes a registration for turns pinned after this call.
func (r *Runtime) Admit(registration Registration) (Snapshot, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if err := classifyRegistration(registration); err != nil {
		return Snapshot{}, err
	}
	for _, existing := range r.registrations {
		if existing.Program.Name == registration.Program.Name {
			return Snapshot{}, &AdmitError{Code: AdmitNameCollision, Err: fmt.Errorf("canonical name %q already registered", registration.Program.Name)}
		}
	}
	if r.hooks.Policy != nil {
		if err := r.hooks.Policy(registration); err != nil {
			var refusal *AdmitError
			if errors.As(err, &refusal) && (refusal.Code == AdmitPolicyWidening || refusal.Code == AdmitPolicyDeny) {
				return Snapshot{}, refusal
			}
			return Snapshot{}, &AdmitError{Code: AdmitPolicyDeny, Err: err}
		}
	}
	registrations := append(append([]Registration(nil), r.registrations...), registration)
	selected := make([]string, 0, len(r.current.Tools)+1)
	for _, tool := range r.current.Tools {
		selected = append(selected, tool.Canonical)
	}
	selected = append(selected, registration.Program.Name)
	next, err := Expose(registrations, selected, r.current.Dialect)
	if err != nil {
		return Snapshot{}, &AdmitError{Code: AdmitNameCollision, Err: err}
	}
	if r.hooks.Register != nil {
		if err := r.hooks.Register(next); err != nil {
			return Snapshot{}, &AdmitError{Code: AdmitSnapshotFailure, Err: err}
		}
	}
	prior := r.current.Digest
	r.registrations = registrations
	r.current = cloneSnapshot(next)
	r.snapshots[next.Digest] = cloneSnapshot(next)
	r.admitted[registration.Digest] = runtimeAdmission{prior: prior}
	r.emit(RuntimeSwap{Kind: SwapAdmit, PriorDigest: prior, CurrentDigest: next.Digest, Registration: registration.Digest})
	return cloneSnapshot(next), nil
}

// Report records one of an admitted registration's first live calls. A typed
// failure reverts immediately; nil marks a successful call and closes the
// bounded three-call observation window when reached.
func (r *Runtime) Report(registrationDigest string, failure LiveFailure, cause error) (bool, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	admission, ok := r.admitted[registrationDigest]
	if !ok {
		return false, nil
	}
	admission.calls++
	if cause == nil {
		if admission.calls >= 3 {
			delete(r.admitted, registrationDigest)
		} else {
			r.admitted[registrationDigest] = admission
		}
		return false, nil
	}
	if failure != LiveFailureExecutor && failure != LiveFailureResultAdmit && failure != LiveFailureQuarantine {
		return false, fmt.Errorf("LIVE_FAILURE_UNKNOWN: %q", failure)
	}
	return true, r.revertLocked(admission.prior, SwapAutoRevert, registrationDigest, string(failure))
}

// Revert restores a retained digest-addressed generation.
func (r *Runtime) Revert(digest string) (Snapshot, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if err := r.revertLocked(digest, SwapRevert, "", "operator"); err != nil {
		return Snapshot{}, err
	}
	return cloneSnapshot(r.current), nil
}

// List returns retained generations, including pre-admit snapshots for replay.
func (r *Runtime) List() []Snapshot {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]Snapshot, 0, len(r.snapshots))
	for _, snapshot := range r.snapshots {
		out = append(out, cloneSnapshot(snapshot))
	}
	return out
}

func (r *Runtime) revertLocked(digest string, kind SwapKind, registration, reason string) error {
	target, ok := r.snapshots[digest]
	if !ok {
		return fmt.Errorf("TOOL_SNAPSHOT_UNKNOWN: %s", digest)
	}
	prior := r.current.Digest
	r.current = cloneSnapshot(target)
	active := make(map[string]bool, len(target.Tools))
	for _, tool := range target.Tools {
		active[tool.Canonical] = true
	}
	registrations := r.registrations[:0]
	for _, candidate := range r.registrations {
		if active[candidate.Program.Name] {
			registrations = append(registrations, candidate)
		}
	}
	r.registrations = registrations
	delete(r.admitted, registration)
	r.emit(RuntimeSwap{Kind: kind, PriorDigest: prior, CurrentDigest: digest, Registration: registration, Reason: reason})
	return nil
}

func (r *Runtime) emit(swap RuntimeSwap) {
	if r.hooks.Swap != nil {
		r.hooks.Swap(swap)
	}
}

func classifyRegistration(registration Registration) error {
	if registration.Program.Version != ProgramVersion {
		return &AdmitError{Code: AdmitUnknownVersion, Err: fmt.Errorf("version %q", registration.Program.Version)}
	}
	if err := validateProgram(registration.Program); err != nil {
		return &AdmitError{Code: AdmitMalformed, Err: err}
	}
	expected, err := newRegistration(registration.Program, registration.Source)
	if err != nil {
		return &AdmitError{Code: AdmitMalformed, Err: err}
	}
	if expected.Digest != registration.Digest {
		return &AdmitError{Code: AdmitDigestMismatch, Err: fmt.Errorf("got %s want %s", registration.Digest, expected.Digest)}
	}
	return nil
}

func cloneSnapshot(snapshot Snapshot) Snapshot {
	snapshot.Tools = append([]ModelTool(nil), snapshot.Tools...)
	snapshot.Omitted = append([]Omission(nil), snapshot.Omitted...)
	return snapshot
}
