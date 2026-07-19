// Package servicespec defines the portable fak.service.v1 desired-state and
// restart-semantics contract (#4749, parent #4748). Service intent was encoded
// independently in SCM verbs, Scheduled Task scripts, cron concepts, and
// deployment docs; this leaf is the ONE versioned schema all of them render
// from: identity, command, cwd, environment references, readiness, restart /
// backoff, checkpoint/resume, dependencies, and intentional-stop state.
//
// Two axes are kept strictly disjoint:
//
//   - DESIRED state ("desired-running" / "desired-stopped") is intent — what
//     the operator or a reconciler wants. It has exactly two values.
//   - OBSERVED state (starting / ready / degraded / failed / fenced / stopped /
//     unknown) is what a supervisor last saw. No string is shared between the
//     axes, so a reader can never conflate intent with observation.
//
// Recurring jobs (#749) versus always-on services: a KindService with
// desired=running is a steady-state obligation — ANY exit (even clean) leaves
// the desire unmet and the supervisor restarts it. A KindJob is a bounded run
// whose SCHEDULE owns recurrence: a clean exit is completion (never restarted
// here); only a non-clean exit is retried under the same backoff contract.
//
// The leaf is pure and portable: no SCM, systemd, launchd, or cron calls —
// platform supervisors (#4750, #4756) render from and report into this
// contract. Secrets are referenced (EnvRef.SecretRef), never serialized.
package servicespec

import (
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strings"
)

// SchemaV1 names the versioned desired-state schema this package parses.
const SchemaV1 = "fak.service.v1"

// ObservedSchemaV1 names the observed-state schema a supervisor reports.
const ObservedSchemaV1 = "fak.service.observed.v1"

// Identity gives the service, the node it is bound to, and the concrete
// workload instance stable names. Node and Service are required; Workload
// defaults to Service (a single-instance service is its own workload).
type Identity struct {
	Node     string `json:"node"`
	Service  string `json:"service"`
	Workload string `json:"workload,omitempty"`
}

// Kind separates always-on services from recurring jobs (#749).
type Kind string

const (
	// KindService is an always-on service: desired=running is a steady-state
	// obligation, so any exit — clean included — is restarted.
	KindService Kind = "service"
	// KindJob is a bounded recurring run: the schedule owns recurrence, a
	// clean exit is completion, only failures are retried.
	KindJob Kind = "job"
)

// DesiredState is the intent axis. Exactly two values: a service is wanted
// running or wanted stopped. Everything else is observation, not desire. The
// wire strings carry the "desired-" prefix (the issue's own vocabulary) so no
// desired value ever shares a string with an observed Phase — a log line or a
// JSON field can never be misread across the axes.
type DesiredState string

const (
	DesiredRunning DesiredState = "desired-running"
	DesiredStopped DesiredState = "desired-stopped"
)

// Phase is the observed axis — what the supervisor last saw, never what is
// wanted.
type Phase string

const (
	PhaseStarting Phase = "starting" // launched, readiness not yet met
	PhaseReady    Phase = "ready"    // readiness met
	PhaseDegraded Phase = "degraded" // readiness previously met, now impaired
	PhaseFailed   Phase = "failed"   // exited or unhealthy, restart pending
	PhaseFenced   Phase = "fenced"   // held off: circuit open or operator fence
	PhaseStopped  Phase = "stopped"  // intentionally not running
	PhaseUnknown  Phase = "unknown"  // observation lost
)

// AllPhases enumerates the observed vocabulary (stable order).
var AllPhases = []Phase{PhaseStarting, PhaseReady, PhaseDegraded, PhaseFailed, PhaseFenced, PhaseStopped, PhaseUnknown}

// phaseNext is the legal observed-state transition table. Observation loss
// (-> unknown) is always legal and is handled in CanTransition, not listed.
var phaseNext = map[Phase][]Phase{
	PhaseStarting: {PhaseReady, PhaseFailed, PhaseStopped},
	PhaseReady:    {PhaseDegraded, PhaseFailed, PhaseStopped},
	PhaseDegraded: {PhaseReady, PhaseFailed, PhaseStopped},
	PhaseFailed:   {PhaseStarting, PhaseFenced, PhaseStopped},
	PhaseFenced:   {PhaseStarting, PhaseStopped},
	PhaseStopped:  {PhaseStarting},
	PhaseUnknown:  {PhaseStarting, PhaseReady, PhaseDegraded, PhaseFailed, PhaseFenced, PhaseStopped},
}

// CanTransition reports whether an observed-state transition is legal.
// Every phase may decay to unknown (observation loss is always possible);
// ready is only reachable through starting (or a re-observation from unknown),
// never directly from stopped or fenced.
func CanTransition(from, to Phase) bool {
	if to == PhaseUnknown {
		return from != PhaseUnknown
	}
	for _, n := range phaseNext[from] {
		if n == to {
			return true
		}
	}
	return false
}

// ExitClass classifies why a run ended. Every restart decision starts from one
// of these; no exit is left unclassified.
type ExitClass string

const (
	ExitClean          ExitClass = "clean"            // process exited with success
	ExitCrash          ExitClass = "crash"            // nonzero exit, signal, or panic
	ExitWatchdog       ExitClass = "watchdog-timeout" // liveness/readiness deadline missed
	ExitBootRecovery   ExitClass = "boot-recovery"    // host rebooted; the service did not fail
	ExitDependencyLoss ExitClass = "dependency-loss"  // a depends_on service left ready
	ExitOperatorStop   ExitClass = "operator-stop"    // a human or control plane stopped it on purpose
)

// AllExitClasses enumerates the exit vocabulary (stable order).
var AllExitClasses = []ExitClass{ExitClean, ExitCrash, ExitWatchdog, ExitBootRecovery, ExitDependencyLoss, ExitOperatorStop}

// EnvRef is one environment binding. A ref carries EITHER a literal value OR
// an opaque secret reference — never both, so a resolved secret can never be
// serialized next to its reference. The contract never carries secret bytes.
type EnvRef struct {
	Name      string `json:"name"`
	Value     string `json:"value,omitempty"`
	SecretRef string `json:"secret_ref,omitempty"`
}

// Readiness declares how a supervisor decides PhaseStarting -> PhaseReady.
// Kind and Target are opaque to this leaf (platform supervisors interpret
// them); the contract only fixes their shape and the deadline.
type Readiness struct {
	Kind      string `json:"kind"`
	Target    string `json:"target,omitempty"`
	TimeoutMS int64  `json:"timeout_ms,omitempty"`
}

// Restart-policy defaults, filled by Normalize when a spec omits them.
const (
	DefaultInitialBackoffMS  int64 = 1000
	DefaultMaxBackoffMS      int64 = 60000
	DefaultBackoffFactor     int64 = 2
	DefaultWindowMS          int64 = 600000
	DefaultWindowMaxRestarts int   = 5
	DefaultStableRunResetMS  int64 = 300000
)

// RestartPolicy bounds how a supervisor re-runs an exited workload: bounded
// exponential backoff (initial * factor^attempt, capped at max), a rolling
// restart-window cap that opens the circuit, and a stable-run threshold that
// resets the attempt counter. All durations are integer milliseconds so the
// wire form is deterministic on every platform.
type RestartPolicy struct {
	InitialBackoffMS  int64 `json:"initial_backoff_ms"`
	MaxBackoffMS      int64 `json:"max_backoff_ms"`
	BackoffFactor     int64 `json:"backoff_factor"`
	WindowMS          int64 `json:"window_ms"`
	WindowMaxRestarts int   `json:"window_max_restarts"`
	StableRunResetMS  int64 `json:"stable_run_reset_ms"`
}

// Spec is the fak.service.v1 desired-state document.
type Spec struct {
	Schema        string        `json:"schema"`
	Identity      Identity      `json:"identity"`
	Kind          Kind          `json:"kind"`
	Desired       DesiredState  `json:"desired"`
	Command       []string      `json:"command"`
	Dir           string        `json:"dir,omitempty"`
	Env           []EnvRef      `json:"env,omitempty"`
	Readiness     *Readiness    `json:"readiness,omitempty"`
	Restart       RestartPolicy `json:"restart"`
	CheckpointDir string        `json:"checkpoint_dir,omitempty"`
	DependsOn     []string      `json:"depends_on,omitempty"`
}

// Observed is the fak.service.observed.v1 report a supervisor emits. It never
// carries intent — Desired lives only in Spec.
type Observed struct {
	Schema   string      `json:"schema"`
	Identity Identity    `json:"identity"`
	Phase    Phase       `json:"phase"`
	Attempt  int         `json:"attempt,omitempty"`
	LastExit *ExitRecord `json:"last_exit,omitempty"`
}

// ExitRecord classifies the most recent run's end.
type ExitRecord struct {
	Class    ExitClass `json:"class"`
	Code     int       `json:"code,omitempty"`
	AtUnixMS int64     `json:"at_unix_ms,omitempty"`
	RunMS    int64     `json:"run_ms,omitempty"`
}

// ParseSpec strictly decodes, validates, and normalizes one fak.service.v1
// document. Unknown fields are refused (a typo'd field must never silently
// drop intent). The returned spec is normalized: defaults filled, env sorted
// by name, depends_on sorted and deduplicated.
func ParseSpec(data []byte) (*Spec, error) {
	dec := json.NewDecoder(strings.NewReader(string(data)))
	dec.DisallowUnknownFields()
	var s Spec
	if err := dec.Decode(&s); err != nil {
		if strings.Contains(err.Error(), "unknown field") {
			return nil, fmt.Errorf("servicespec: %w", err)
		}
		return nil, fmt.Errorf("servicespec: parse: %w", err)
	}
	s.Normalize()
	if err := s.Validate(); err != nil {
		return nil, err
	}
	return &s, nil
}

// Normalize fills portable defaults and puts list fields in canonical order so
// CanonicalJSON is deterministic. Safe to call repeatedly (idempotent).
func (s *Spec) Normalize() {
	if s.Kind == "" {
		s.Kind = KindService
	}
	if s.Identity.Workload == "" {
		s.Identity.Workload = s.Identity.Service
	}
	p := &s.Restart
	if p.InitialBackoffMS == 0 {
		p.InitialBackoffMS = DefaultInitialBackoffMS
	}
	if p.MaxBackoffMS == 0 {
		p.MaxBackoffMS = DefaultMaxBackoffMS
	}
	if p.BackoffFactor == 0 {
		p.BackoffFactor = DefaultBackoffFactor
	}
	if p.WindowMS == 0 {
		p.WindowMS = DefaultWindowMS
	}
	if p.WindowMaxRestarts == 0 {
		p.WindowMaxRestarts = DefaultWindowMaxRestarts
	}
	if p.StableRunResetMS == 0 {
		p.StableRunResetMS = DefaultStableRunResetMS
	}
	sort.SliceStable(s.Env, func(i, j int) bool { return s.Env[i].Name < s.Env[j].Name })
	sort.Strings(s.DependsOn)
	s.DependsOn = dedupeSorted(s.DependsOn)
}

func dedupeSorted(in []string) []string {
	out := in[:0]
	for i, v := range in {
		if i == 0 || v != in[i-1] {
			out = append(out, v)
		}
	}
	return out
}

// Validate checks the normalized spec against the v1 contract.
func (s *Spec) Validate() error {
	if s.Schema != SchemaV1 {
		return fmt.Errorf("servicespec: schema %q is not %q", s.Schema, SchemaV1)
	}
	if strings.TrimSpace(s.Identity.Node) == "" {
		return errors.New("servicespec: identity.node is required")
	}
	if strings.TrimSpace(s.Identity.Service) == "" {
		return errors.New("servicespec: identity.service is required")
	}
	if s.Kind != KindService && s.Kind != KindJob {
		return fmt.Errorf("servicespec: kind %q is not %q or %q", s.Kind, KindService, KindJob)
	}
	if s.Desired != DesiredRunning && s.Desired != DesiredStopped {
		return fmt.Errorf("servicespec: desired %q is not %q or %q", s.Desired, DesiredRunning, DesiredStopped)
	}
	if len(s.Command) == 0 || strings.TrimSpace(s.Command[0]) == "" {
		return errors.New("servicespec: command is required")
	}
	for i, e := range s.Env {
		if strings.TrimSpace(e.Name) == "" {
			return fmt.Errorf("servicespec: env[%d] has no name", i)
		}
		if e.Value != "" && e.SecretRef != "" {
			return fmt.Errorf("servicespec: env %q sets both value and secret_ref — a secret is referenced, never serialized", e.Name)
		}
	}
	p := s.Restart
	if p.InitialBackoffMS < 0 {
		return fmt.Errorf("servicespec: initial_backoff_ms %d is negative", p.InitialBackoffMS)
	}
	if p.MaxBackoffMS < p.InitialBackoffMS {
		return fmt.Errorf("servicespec: max_backoff_ms %d is below initial_backoff_ms %d", p.MaxBackoffMS, p.InitialBackoffMS)
	}
	if p.BackoffFactor < 1 {
		return fmt.Errorf("servicespec: backoff_factor %d must be >= 1", p.BackoffFactor)
	}
	if p.WindowMS < 0 || p.StableRunResetMS < 0 || p.WindowMaxRestarts < 0 {
		return errors.New("servicespec: restart window/reset bounds must be non-negative")
	}
	for _, d := range s.DependsOn {
		if d == s.Identity.Service {
			return fmt.Errorf("servicespec: depends_on lists the service itself (%q)", d)
		}
	}
	return nil
}

// CanonicalJSON returns the deterministic wire form of the normalized spec:
// fixed field order, sorted lists, integer durations. Parsing the canonical
// form and re-canonicalizing is the identity.
func (s *Spec) CanonicalJSON() ([]byte, error) {
	c := *s
	c.Normalize()
	if err := c.Validate(); err != nil {
		return nil, err
	}
	return json.MarshalIndent(&c, "", "  ")
}

// Decision reasons — the closed vocabulary a supervisor logs with each
// restart decision.
const (
	ReasonDesiredStopped = "desired-stopped"
	ReasonOperatorStop   = "operator-stop"
	ReasonJobComplete    = "job-complete"
	ReasonCircuitOpen    = "circuit-open"
	ReasonBootRecovery   = "boot-recovery"
	ReasonRestart        = "restart"
)

// RestartInput is what a supervisor knows when a run ends.
type RestartInput struct {
	Kind    Kind
	Desired DesiredState
	Class   ExitClass
	// Attempt is the consecutive restart count since the last stable run.
	Attempt int
	// RunMS is how long the exited run lasted.
	RunMS int64
	// WindowCount is how many restarts already happened inside the rolling
	// restart window.
	WindowCount int
}

// RestartDecision is the portable verdict: restart or not, after what delay,
// and whether the circuit is now open (PhaseFenced until an operator or a
// window expiry releases it).
type RestartDecision struct {
	Restart     bool   `json:"restart"`
	DelayMS     int64  `json:"delay_ms"`
	CircuitOpen bool   `json:"circuit_open"`
	NextAttempt int    `json:"next_attempt"`
	Reason      string `json:"reason"`
}

// Decide applies the v1 restart semantics, in priority order:
//
//  1. desired=stopped never restarts (intent wins).
//  2. operator-stop never restarts even under desired=running: an intentional
//     stop is recorded, not fought.
//  3. a KindJob clean exit is completion — the schedule owns the next run.
//  4. the rolling restart-window cap opens the circuit (no restart, fenced).
//  5. boot-recovery restarts immediately with a fresh attempt counter: the
//     HOST failed, not the service, so prior backoff does not carry over.
//  6. a run that outlived stable_run_reset_ms resets the attempt counter.
//  7. otherwise restart after bounded exponential backoff:
//     min(initial * factor^attempt, max).
func (p RestartPolicy) Decide(in RestartInput) RestartDecision {
	if in.Desired == DesiredStopped {
		return RestartDecision{Reason: ReasonDesiredStopped, NextAttempt: in.Attempt}
	}
	if in.Class == ExitOperatorStop {
		return RestartDecision{Reason: ReasonOperatorStop, NextAttempt: in.Attempt}
	}
	if in.Kind == KindJob && in.Class == ExitClean {
		return RestartDecision{Reason: ReasonJobComplete}
	}
	if p.WindowMaxRestarts > 0 && in.WindowCount >= p.WindowMaxRestarts {
		return RestartDecision{CircuitOpen: true, Reason: ReasonCircuitOpen, NextAttempt: in.Attempt}
	}
	if in.Class == ExitBootRecovery {
		return RestartDecision{Restart: true, DelayMS: 0, NextAttempt: 1, Reason: ReasonBootRecovery}
	}
	attempt := in.Attempt
	if p.StableRunResetMS > 0 && in.RunMS >= p.StableRunResetMS {
		attempt = 0
	}
	return RestartDecision{
		Restart:     true,
		DelayMS:     p.backoff(attempt),
		NextAttempt: attempt + 1,
		Reason:      ReasonRestart,
	}
}

// backoff computes min(initial * factor^attempt, max) without overflowing.
func (p RestartPolicy) backoff(attempt int) int64 {
	d := p.InitialBackoffMS
	for i := 0; i < attempt; i++ {
		if d >= p.MaxBackoffMS || p.BackoffFactor <= 1 {
			return min64(d, p.MaxBackoffMS)
		}
		if d > p.MaxBackoffMS/p.BackoffFactor {
			return p.MaxBackoffMS
		}
		d *= p.BackoffFactor
	}
	return min64(d, p.MaxBackoffMS)
}

func min64(a, b int64) int64 {
	if a < b {
		return a
	}
	return b
}
