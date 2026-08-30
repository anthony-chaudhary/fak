package harnesskit

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strings"
	"sync"
)

// RuntimeReadiness is the state a mounted extension reports at the launch gate.
type RuntimeReadiness struct {
	State           LifecycleState `json:"state"`
	MissingServices []string       `json:"missing_services,omitempty"`
	Err             error          `json:"-"`
}

// ReadinessReporter lets a runtime distinguish successful construction from
// actually reaching its declared running state.
type ReadinessReporter interface {
	Readiness() RuntimeReadiness
}

// LifecycleRecord is one enabled extension's mounted-state readback.
type LifecycleRecord struct {
	ExtensionID     string         `json:"extension_id"`
	State           LifecycleState `json:"state"`
	MissingServices []string       `json:"missing_services,omitempty"`
	Failure         string         `json:"failure,omitempty"`
	Evidence        string         `json:"evidence,omitempty"`
}

type lifecycleEntry struct {
	LifecycleRecord
	cause error
}

// LifecycleRegistry records every enabled component from the resolved lock,
// including components whose factory or runtime never mounted.
type LifecycleRegistry struct {
	mu   sync.RWMutex
	rows map[string]lifecycleEntry
}

func newLifecycleRegistry(components []LockedComponent) *LifecycleRegistry {
	registry := &LifecycleRegistry{rows: make(map[string]lifecycleEntry, len(components))}
	for _, component := range components {
		registry.rows[component.ID] = lifecycleEntry{LifecycleRecord: LifecycleRecord{ExtensionID: component.ID, State: StateDeclared}}
	}
	return registry
}

func (r *LifecycleRegistry) set(id string, state LifecycleState, missing []string, failure, evidence string, cause error) {
	if r == nil {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.rows[id] = lifecycleEntry{LifecycleRecord: LifecycleRecord{
		ExtensionID:     id,
		State:           state,
		MissingServices: normalizedServices(missing),
		Failure:         failure,
		Evidence:        evidence,
	}, cause: cause}
}

// Snapshot returns a stable copy ordered by extension ID.
func (r *LifecycleRegistry) Snapshot() []LifecycleRecord {
	entries := r.entries()
	out := make([]LifecycleRecord, len(entries))
	for i, entry := range entries {
		out[i] = cloneLifecycleRecord(entry.LifecycleRecord)
	}
	return out
}

func (r *LifecycleRegistry) entries() []lifecycleEntry {
	if r == nil {
		return nil
	}
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]lifecycleEntry, 0, len(r.rows))
	for _, entry := range r.rows {
		entry.LifecycleRecord = cloneLifecycleRecord(entry.LifecycleRecord)
		out = append(out, entry)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].ExtensionID < out[j].ExtensionID })
	return out
}

// Activation owns every runtime started from one resolved product lock.
type Activation struct {
	registry  *LifecycleRegistry
	mounted   []mountedExtension
	drainOnce sync.Once
	drainErr  error
	closeOnce sync.Once
	closeErr  error
}

type mountedExtension struct {
	id      string
	runtime Runtime
}

// Registry returns the mounted-state registry owned by this activation.
func (a *Activation) Registry() *LifecycleRegistry {
	if a == nil {
		return nil
	}
	return a.registry
}

// Drain stops accepted work in reverse mount order.
func (a *Activation) Drain(ctx context.Context) error {
	if a == nil {
		return nil
	}
	a.drainOnce.Do(func() {
		var errs []error
		for i := len(a.mounted) - 1; i >= 0; i-- {
			mounted := a.mounted[i]
			a.registry.set(mounted.id, StateDraining, nil, "", "owner scope draining", nil)
			if err := mounted.runtime.Drain(ctx); err != nil {
				a.registry.set(mounted.id, StateFailed, nil, err.Error(), "drain failed", err)
				errs = append(errs, fmt.Errorf("drain extension %s: %w", mounted.id, err))
			}
		}
		a.drainErr = errors.Join(errs...)
	})
	return a.drainErr
}

// Close releases every runtime in reverse mount order. It is idempotent.
func (a *Activation) Close() error {
	if a == nil {
		return nil
	}
	a.closeOnce.Do(func() {
		var errs []error
		for i := len(a.mounted) - 1; i >= 0; i-- {
			mounted := a.mounted[i]
			if err := mounted.runtime.Close(); err != nil {
				a.registry.set(mounted.id, StateFailed, nil, err.Error(), "close failed", err)
				errs = append(errs, fmt.Errorf("close extension %s: %w", mounted.id, err))
				continue
			}
			a.registry.set(mounted.id, StateClosed, nil, "", "owner scope closed", nil)
		}
		a.closeErr = errors.Join(errs...)
	})
	return a.closeErr
}

// ActivationError reports every enabled extension that failed the readiness
// gate while preserving the original runtime failures in its error chain.
type ActivationError struct {
	rows       []LifecycleRecord
	failures   []LifecycleRecord
	causes     []error
	cleanupErr error
}

// CompatibilityError preserves the complete machine report for an activation
// refused before any factory is started.
type CompatibilityError struct {
	Report CompatibilityReport
}

func (e *CompatibilityError) Error() string {
	if e == nil {
		return "<nil>"
	}
	return e.Report.Error()
}

func (e *ActivationError) Error() string {
	if e == nil {
		return "<nil>"
	}
	noun := "extensions"
	if len(e.failures) == 1 {
		noun = "extension"
	}
	lines := []string{fmt.Sprintf("harnesskit activation: %d %s did not reach running", len(e.failures), noun)}
	for _, row := range e.failures {
		lines = append(lines, formatReadinessFailure(row))
	}
	if e.cleanupErr != nil {
		lines = append(lines, "owner-scope cleanup: "+e.cleanupErr.Error())
	}
	return strings.Join(lines, "\n")
}

// Unwrap retains runtime and cleanup causes for errors.Is/errors.As callers.
func (e *ActivationError) Unwrap() []error {
	if e == nil {
		return nil
	}
	out := append([]error(nil), e.causes...)
	if e.cleanupErr != nil {
		out = append(out, e.cleanupErr)
	}
	return out
}

// Snapshot returns the refused mounted-state readback before cleanup changed
// the live registry to its terminal states.
func (e *ActivationError) Snapshot() []LifecycleRecord {
	if e == nil {
		return nil
	}
	out := make([]LifecycleRecord, len(e.rows))
	for i, row := range e.rows {
		out[i] = cloneLifecycleRecord(row)
	}
	return out
}

// Activate mounts every enabled component in a verified resolved lock, audits
// the settled registry, and returns only when every row is running.
func Activate(ctx context.Context, lock ProductLock, services Services, factories map[string]Factory) (*Activation, error) {
	verified, err := verifyActivationLock(lock)
	if err != nil {
		return nil, &Error{Code: CodeInvalid, Op: "activate", Err: err}
	}
	active := &Activation{registry: newLifecycleRegistry(verified.Components)}
	for _, component := range verified.Components {
		factory, ok := factories[component.ID]
		if !ok || factory == nil {
			active.registry.set(component.ID, StateDeclared, nil, "", "factory not mounted", nil)
			continue
		}
		manifest := factory.Manifest()
		if manifest.ID != component.ID {
			cause := fmt.Errorf("factory manifest id %q does not match locked extension %q", manifest.ID, component.ID)
			active.registry.set(component.ID, StateFailed, nil, cause.Error(), "factory manifest mismatch", cause)
			continue
		}
		active.registry.set(component.ID, StateStarting, nil, "", "factory starting", nil)
		runtime, startErr := factory.Start(ctx, services)
		if startErr != nil {
			active.registry.set(component.ID, StateFailed, nil, startErr.Error(), "factory start failed", startErr)
			continue
		}
		if runtime == nil {
			cause := errors.New("factory returned a nil runtime")
			active.registry.set(component.ID, StateFailed, nil, cause.Error(), "factory start failed", cause)
			continue
		}
		active.mounted = append(active.mounted, mountedExtension{id: component.ID, runtime: runtime})
	}

	for _, mounted := range active.mounted {
		readiness := RuntimeReadiness{State: StateRunning}
		if reporter, ok := mounted.runtime.(ReadinessReporter); ok {
			readiness = reporter.Readiness()
		}
		state := readiness.State
		if state == "" {
			state = StateStarting
		}
		if state == StateRunning && (readiness.Err != nil || len(readiness.MissingServices) > 0) {
			if readiness.Err == nil {
				readiness.Err = errors.New("running runtime reported missing services")
			}
			state = StateFailed
		}
		failure := ""
		if readiness.Err != nil {
			failure = readiness.Err.Error()
		}
		if state == StateFailed && readiness.Err == nil {
			readiness.Err = errors.New("runtime reported failed without an original cause")
			failure = readiness.Err.Error()
		}
		active.registry.set(mounted.id, state, readiness.MissingServices, failure, "runtime readiness readback", readiness.Err)
	}

	activationErr := auditActivation(active.registry)
	if activationErr == nil {
		return active, nil
	}
	activationErr.cleanupErr = active.Close()
	return nil, activationErr
}

// ActivateCompatible negotiates the builder and host contracts before lock
// verification or any Factory.Start call. Legacy Activate remains available
// for callers whose compatibility is established by an older host boundary.
func ActivateCompatible(ctx context.Context, lock ProductLock, services Services, factories map[string]Factory, builder BuilderContract, host RuntimeContract) (*Activation, CompatibilityReport, error) {
	report := NegotiateCompatibility(builder, host)
	if !report.Compatible {
		return nil, report, &Error{Code: CodeUnsupported, Op: "activate compatibility", Err: &CompatibilityError{Report: report}}
	}
	active, err := Activate(ctx, lock, services, factories)
	return active, report, err
}

func verifyActivationLock(lock ProductLock) (ProductLock, error) {
	raw, err := json.Marshal(lock)
	if err != nil {
		return ProductLock{}, fmt.Errorf("encode product lock: %w", err)
	}
	return ParseProductLock(raw)
}

func auditActivation(registry *LifecycleRegistry) *ActivationError {
	entries := registry.entries()
	rows := make([]LifecycleRecord, 0, len(entries))
	var refused []LifecycleRecord
	var causes []error
	for _, entry := range entries {
		rows = append(rows, entry.LifecycleRecord)
		if entry.State == StateRunning {
			continue
		}
		refused = append(refused, entry.LifecycleRecord)
		if entry.cause != nil {
			causes = append(causes, entry.cause)
		}
	}
	if len(refused) == 0 {
		return nil
	}
	return &ActivationError{rows: rows, failures: refused, causes: causes}
}

func formatReadinessFailure(row LifecycleRecord) string {
	switch row.State {
	case StateDeclared:
		if row.Evidence == "factory not mounted" {
			return fmt.Sprintf("%s: absent (%s)", row.ExtensionID, row.Evidence)
		}
		return fmt.Sprintf("%s: pending (%s)", row.ExtensionID, evidenceOrUnknown(row.Evidence))
	case StateStarting:
		missing := "unknown"
		if len(row.MissingServices) > 0 {
			missing = strings.Join(row.MissingServices, ", ")
		}
		subject := "services"
		if len(row.MissingServices) == 1 {
			subject = "service"
		}
		return fmt.Sprintf("%s: pending (waiting for %s: %s)", row.ExtensionID, subject, missing)
	case StateFailed:
		return fmt.Sprintf("%s: failed: %s", row.ExtensionID, evidenceOrUnknown(row.Failure))
	default:
		return fmt.Sprintf("%s: state %s (%s)", row.ExtensionID, row.State, evidenceOrUnknown(row.Evidence))
	}
}

func evidenceOrUnknown(value string) string {
	if value == "" {
		return "unknown"
	}
	return value
}

func normalizedServices(services []string) []string {
	seen := map[string]bool{}
	out := make([]string, 0, len(services))
	for _, service := range services {
		service = strings.TrimSpace(service)
		if service == "" || seen[service] {
			continue
		}
		seen[service] = true
		out = append(out, service)
	}
	sort.Strings(out)
	return out
}

func cloneLifecycleRecord(row LifecycleRecord) LifecycleRecord {
	row.MissingServices = append([]string(nil), row.MissingServices...)
	return row
}
