package guard

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"
)

// Supervisor child process monitoring and exit classification (#11481).
//
// When an upstream provider returns HTTP 429 rate limiting on LLM completion
// endpoints, the guarded child process exits with code 1. Rather than treating
// this rate-limit exit as an abnormal crash that burns the RESTART_HOP budget
// and terminates with CRASH_RESTART_EXHAUSTED, the supervisor classifies it as
// backpressure (PROVIDER_RATE_LIMITED / backpressure pause).
//
// Invariant: Rate-limit exits DO NOT increment crash restart counts or burn
// RESTART_HOP budgets. The crash restart runway is reserved for genuine abnormal
// crashes (panics, OOMs, SIGSEGVs, unhandled process failures).

const (
	// Supervisor states.
	StateRunning               = "RUNNING"
	StateProviderRateLimited   = "PROVIDER_RATE_LIMITED"
	StateCrashRestart          = "CRASH_RESTART"
	StateCrashRestartExhausted = "CRASH_RESTART_EXHAUSTED"
	StateCleanExit             = "CLEAN_EXIT"

	// Supervisor actions.
	ActionBackpressurePause = "BACKPRESSURE_PAUSE"
	ActionCrashRestart      = "CRASH_RESTART"
	ActionExhausted         = "CRASH_RESTART_EXHAUSTED"
	ActionTerminate         = "TERMINATE"

	// Reason tokens from closed vocabulary.
	ReasonProviderRateLimited   = "PROVIDER_RATE_LIMITED"
	ReasonCrashRestartExhausted = "CRASH_RESTART_EXHAUSTED"
	ReasonNormalExit            = "NORMAL_EXIT"
	ReasonChildCrash            = "CHILD_CRASH"

	// Observed event kinds and details.
	EventUpstreamFailure        = "UPSTREAM_FAILURE"
	EventAccountRotation        = "ACCOUNT_ROTATION"
	RotationProviderRateLimited = "provider_rate_limited"

	// Defaults.
	DefaultCrashRestartLimit        = 3
	DefaultInitialBackpressurePause = 1 * time.Second
	DefaultMaxBackpressurePause     = 30 * time.Second
	DefaultCrashRestartInitialDelay = 250 * time.Millisecond
	DefaultCrashRestartMaxDelay     = 2 * time.Second
)

// UpstreamFailureReceipt is bounded evidence for an upstream failure event.
type UpstreamFailureReceipt struct {
	HTTPStatus        int               `json:"http_status,omitempty"`
	EmittingLayer     string            `json:"emitting_layer,omitempty"`
	Cause             string            `json:"cause,omitempty"`
	TargetID          string            `json:"target_id,omitempty"`
	ProviderRequestID string            `json:"provider_request_id,omitempty"`
	RetryReason       string            `json:"retry_reason,omitempty"`
	RetryAfter        string            `json:"retry_after,omitempty"`
	Diagnostic        map[string]string `json:"diagnostic_headers,omitempty"`
}

// AccountRotationEvent describes an account rotation observed during child execution.
type AccountRotationEvent struct {
	Seat   string `json:"seat,omitempty"`
	Reason string `json:"reason,omitempty"`
	Detail string `json:"detail,omitempty"`
}

// SupervisorEvent represents an observed lifecycle or gateway event.
type SupervisorEvent struct {
	Kind      string    `json:"kind"`
	Timestamp time.Time `json:"timestamp,omitempty"`
	TraceID   string    `json:"trace_id,omitempty"`
	Reason    string    `json:"reason,omitempty"`
	Payload   string    `json:"payload,omitempty"`
	Status    int       `json:"status,omitempty"`
}

// ChildExitEvidence collects signals preceding or accompanying a child process exit.
type ChildExitEvidence struct {
	ExitCode        int
	RunErr          error
	UpstreamFailure *UpstreamFailureReceipt
	AccountRotation *AccountRotationEvent
	RotationReason  string
	Events          []SupervisorEvent
	RawEvidence     string
}

// SupervisorDecision encapsulates the classification and required action for a child exit.
type SupervisorDecision struct {
	State                 string        `json:"state"`
	Action                string        `json:"action"`
	Reason                string        `json:"reason"`
	BurnsBudget           bool          `json:"burns_budget"`
	RestartHop            int           `json:"restart_hop"`
	BackpressurePause     time.Duration `json:"backpressure_pause_ms"`
	ConsecutiveRateLimits int           `json:"consecutive_rate_limits"`
	Detail                string        `json:"detail,omitempty"`
}

// SupervisorConfig controls crash limits and backpressure parameters.
type SupervisorConfig struct {
	CrashRestartLimit        int
	InitialBackpressurePause time.Duration
	MaxBackpressurePause     time.Duration
	CrashRestartInitialDelay time.Duration
	CrashRestartMaxDelay     time.Duration
}

// DefaultSupervisorConfig returns production-default supervision settings.
func DefaultSupervisorConfig() SupervisorConfig {
	return SupervisorConfig{
		CrashRestartLimit:        DefaultCrashRestartLimit,
		InitialBackpressurePause: DefaultInitialBackpressurePause,
		MaxBackpressurePause:     DefaultMaxBackpressurePause,
		CrashRestartInitialDelay: DefaultCrashRestartInitialDelay,
		CrashRestartMaxDelay:     DefaultCrashRestartMaxDelay,
	}
}

// Supervisor monitors guarded child process execution, classifying exits into
// clean completion, backpressure pauses, or crash restarts.
type Supervisor struct {
	mu                    sync.Mutex
	cfg                   SupervisorConfig
	state                 string
	crashRestarts         int
	consecutiveRateLimits int
	events                []SupervisorEvent
	lastDecision          SupervisorDecision
}

// ChildSupervisor is an alias for Supervisor for embedders and callers.
type ChildSupervisor = Supervisor

// ProcessSupervisor is an alias for Supervisor for embedders and callers.
type ProcessSupervisor = Supervisor

// NewSupervisor constructs a child process supervisor with optional config overrides.
func NewSupervisor(cfgs ...SupervisorConfig) *Supervisor {
	cfg := DefaultSupervisorConfig()
	if len(cfgs) > 0 {
		c := cfgs[0]
		if c.CrashRestartLimit > 0 {
			cfg.CrashRestartLimit = c.CrashRestartLimit
		}
		if c.InitialBackpressurePause > 0 {
			cfg.InitialBackpressurePause = c.InitialBackpressurePause
		}
		if c.MaxBackpressurePause > 0 {
			cfg.MaxBackpressurePause = c.MaxBackpressurePause
		}
		if c.CrashRestartInitialDelay > 0 {
			cfg.CrashRestartInitialDelay = c.CrashRestartInitialDelay
		}
		if c.CrashRestartMaxDelay > 0 {
			cfg.CrashRestartMaxDelay = c.CrashRestartMaxDelay
		}
	}
	return &Supervisor{
		cfg:   cfg,
		state: StateRunning,
	}
}

// NewProcessSupervisor is an alias constructor for NewSupervisor.
func NewProcessSupervisor(cfgs ...SupervisorConfig) *Supervisor {
	return NewSupervisor(cfgs...)
}

// RecordUpstreamFailure records an upstream failure receipt on the supervisor.
func (s *Supervisor) RecordUpstreamFailure(status int, cause string, diagnostic ...map[string]string) {
	s.mu.Lock()
	defer s.mu.Unlock()

	var diag map[string]string
	if len(diagnostic) > 0 {
		diag = diagnostic[0]
	}
	ev := SupervisorEvent{
		Kind:      EventUpstreamFailure,
		Timestamp: time.Now(),
		Status:    status,
		Reason:    cause,
		Payload:   fmt.Sprintf(`{"http_status":%d,"cause":%q}`, status, cause),
	}
	if diag != nil {
		if raw, err := json.Marshal(diag); err == nil {
			ev.Payload = fmt.Sprintf(`{"http_status":%d,"cause":%q,"diagnostic_headers":%s}`, status, cause, string(raw))
		}
	}
	s.events = append(s.events, ev)
}

// RecordUpstreamFailureReceipt records a structured upstream failure receipt.
func (s *Supervisor) RecordUpstreamFailureReceipt(r UpstreamFailureReceipt) {
	s.mu.Lock()
	defer s.mu.Unlock()

	raw, _ := json.Marshal(r)
	s.events = append(s.events, SupervisorEvent{
		Kind:      EventUpstreamFailure,
		Timestamp: time.Now(),
		Status:    r.HTTPStatus,
		Reason:    r.Cause,
		Payload:   string(raw),
	})
}

// RecordAccountRotation records an account rotation event.
func (s *Supervisor) RecordAccountRotation(reason string, seat ...string) {
	s.mu.Lock()
	defer s.mu.Unlock()

	st := ""
	if len(seat) > 0 {
		st = seat[0]
	}
	payload := reason
	if st != "" {
		payload = st + ":" + reason
	}
	s.events = append(s.events, SupervisorEvent{
		Kind:      EventAccountRotation,
		Timestamp: time.Now(),
		Reason:    reason,
		Payload:   payload,
	})
}

// RecordEvent appends an arbitrary lifecycle event.
func (s *Supervisor) RecordEvent(event SupervisorEvent) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.events = append(s.events, event)
}

// HandleChildExit classifies the exit of a child process and updates supervisor state.
func (s *Supervisor) HandleChildExit(exitCode int, runErr error, extraEvidence ...ChildExitEvidence) SupervisorDecision {
	s.mu.Lock()
	defer s.mu.Unlock()

	evidence := ChildExitEvidence{
		ExitCode: exitCode,
		RunErr:   runErr,
		Events:   append([]SupervisorEvent(nil), s.events...),
	}
	if len(extraEvidence) > 0 {
		extra := extraEvidence[0]
		if extra.UpstreamFailure != nil {
			evidence.UpstreamFailure = extra.UpstreamFailure
		}
		if extra.AccountRotation != nil {
			evidence.AccountRotation = extra.AccountRotation
		}
		if extra.RotationReason != "" {
			evidence.RotationReason = extra.RotationReason
		}
		if extra.RawEvidence != "" {
			evidence.RawEvidence = extra.RawEvidence
		}
		if len(extra.Events) > 0 {
			evidence.Events = append(evidence.Events, extra.Events...)
		}
	}

	decision := s.classifyLocked(exitCode, runErr, evidence)
	s.lastDecision = decision
	s.state = decision.State
	return decision
}

func (s *Supervisor) classifyLocked(exitCode int, runErr error, evidence ChildExitEvidence) SupervisorDecision {
	// Clean exit.
	if exitCode == 0 && runErr == nil {
		s.consecutiveRateLimits = 0
		return SupervisorDecision{
			State:       StateCleanExit,
			Action:      ActionTerminate,
			Reason:      ReasonNormalExit,
			BurnsBudget: false,
			RestartHop:  s.crashRestarts,
			Detail:      "process finished cleanly",
		}
	}

	// Detect if this non-zero exit is preceded or accompanied by rate limiting.
	isRateLimited, trigger := DetectRateLimit(evidence)
	if isRateLimited {
		s.consecutiveRateLimits++
		pause := s.calculateBackpressurePause(s.consecutiveRateLimits, evidence)
		return SupervisorDecision{
			State:                 StateProviderRateLimited,
			Action:                ActionBackpressurePause,
			Reason:                ReasonProviderRateLimited,
			BurnsBudget:           false, // DO NOT burn crash restart budget on rate limits!
			RestartHop:            s.crashRestarts,
			BackpressurePause:     pause,
			ConsecutiveRateLimits: s.consecutiveRateLimits,
			Detail:                fmt.Sprintf("rate-limit backpressure detected (%s); holding restart hop budget at %d/%d", trigger, s.crashRestarts, s.cfg.CrashRestartLimit),
		}
	}

	// Genuine abnormal crash (not rate limited).
	s.consecutiveRateLimits = 0
	if s.crashRestarts >= s.cfg.CrashRestartLimit {
		return SupervisorDecision{
			State:       StateCrashRestartExhausted,
			Action:      ActionExhausted,
			Reason:      ReasonCrashRestartExhausted,
			BurnsBudget: false,
			RestartHop:  s.crashRestarts,
			Detail:      fmt.Sprintf("crash restart budget exhausted (%d/%d attempts spent)", s.crashRestarts, s.cfg.CrashRestartLimit),
		}
	}

	s.crashRestarts++
	delay := s.calculateCrashDelay(s.crashRestarts)
	return SupervisorDecision{
		State:             StateCrashRestart,
		Action:            ActionCrashRestart,
		Reason:            ReasonChildCrash,
		BurnsBudget:       true, // Genuine crash burns 1 RESTART_HOP
		RestartHop:        s.crashRestarts,
		BackpressurePause: delay,
		Detail:            fmt.Sprintf("child process crashed (exit %d); restart attempt %d/%d", exitCode, s.crashRestarts, s.cfg.CrashRestartLimit),
	}
}

func (s *Supervisor) calculateBackpressurePause(consecutive int, evidence ChildExitEvidence) time.Duration {
	// If explicit Retry-After was provided in evidence, honor it within bounds.
	if evidence.UpstreamFailure != nil {
		if dur, ok := parseRetryAfter(evidence.UpstreamFailure.RetryAfter); ok {
			return s.clampPause(dur)
		}
		if evidence.UpstreamFailure.Diagnostic != nil {
			if dur, ok := parseRetryAfter(evidence.UpstreamFailure.Diagnostic["Retry-After"]); ok {
				return s.clampPause(dur)
			}
			if dur, ok := parseRetryAfter(evidence.UpstreamFailure.Diagnostic["retry-after"]); ok {
				return s.clampPause(dur)
			}
		}
	}

	// Default exponential backpressure pause: 1s, 2s, 4s, 8s, up to MaxBackpressurePause.
	shift := consecutive - 1
	if shift < 0 {
		shift = 0
	}
	if shift > 5 {
		shift = 5
	}
	delay := s.cfg.InitialBackpressurePause * time.Duration(1<<uint(shift))
	return s.clampPause(delay)
}

func (s *Supervisor) clampPause(d time.Duration) time.Duration {
	if d < s.cfg.InitialBackpressurePause {
		d = s.cfg.InitialBackpressurePause
	}
	if d > s.cfg.MaxBackpressurePause {
		d = s.cfg.MaxBackpressurePause
	}
	return d
}

func (s *Supervisor) calculateCrashDelay(attempt int) time.Duration {
	if attempt <= 1 {
		return s.cfg.CrashRestartInitialDelay
	}
	delay := s.cfg.CrashRestartInitialDelay * time.Duration(1<<uint(attempt-1))
	if delay > s.cfg.CrashRestartMaxDelay {
		delay = s.cfg.CrashRestartMaxDelay
	}
	return delay
}

// CrashRestarts reports the number of genuine crash restart hops consumed.
func (s *Supervisor) CrashRestarts() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.crashRestarts
}

// ConsecutiveRateLimits reports the number of consecutive rate limit backpressure events.
func (s *Supervisor) ConsecutiveRateLimits() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.consecutiveRateLimits
}

// State returns the current supervisor state.
func (s *Supervisor) State() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.state
}

// LastDecision returns the most recent supervisor decision.
func (s *Supervisor) LastDecision() SupervisorDecision {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.lastDecision
}

// ClearEvents resets recorded events for a fresh child run.
func (s *Supervisor) ClearEvents() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.events = nil
}

// Reset clears both restart counts and recorded events.
func (s *Supervisor) Reset() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.state = StateRunning
	s.crashRestarts = 0
	s.consecutiveRateLimits = 0
	s.events = nil
	s.lastDecision = SupervisorDecision{}
}

// DetectRateLimit evaluates exit evidence to detect if an exit was caused by
// upstream HTTP 429 rate limiting or provider_rate_limited account rotation.
func DetectRateLimit(evidence ChildExitEvidence) (bool, string) {
	// 1. Check structured UpstreamFailure
	if evidence.UpstreamFailure != nil {
		if evidence.UpstreamFailure.HTTPStatus == http.StatusTooManyRequests {
			return true, fmt.Sprintf("UPSTREAM_FAILURE http_status: 429 (%s)", evidence.UpstreamFailure.Cause)
		}
	}

	// 2. Check structured AccountRotation
	if evidence.AccountRotation != nil {
		if strings.Contains(evidence.AccountRotation.Reason, RotationProviderRateLimited) ||
			strings.Contains(evidence.AccountRotation.Detail, RotationProviderRateLimited) {
			return true, fmt.Sprintf("ACCOUNT_ROTATION: %s", RotationProviderRateLimited)
		}
	}

	// 3. Check RotationReason string
	if strings.Contains(evidence.RotationReason, RotationProviderRateLimited) {
		return true, fmt.Sprintf("ACCOUNT_ROTATION: %s", RotationProviderRateLimited)
	}

	// 4. Inspect recorded supervisor events in reverse order (most recent first)
	for i := len(evidence.Events) - 1; i >= 0; i-- {
		ev := evidence.Events[i]
		if ev.Kind == EventUpstreamFailure {
			if ev.Status == http.StatusTooManyRequests {
				return true, "UPSTREAM_FAILURE with http_status: 429"
			}
			if strings.Contains(ev.Payload, `"http_status":429`) ||
				strings.Contains(ev.Payload, `"http_status": 429`) ||
				strings.Contains(ev.Payload, `"status":429`) ||
				strings.Contains(ev.Payload, `"status": 429`) ||
				strings.Contains(ev.Payload, "status=429") ||
				strings.Contains(ev.Payload, "http_status: 429") ||
				strings.Contains(ev.Payload, "429 Too Many Requests") {
				return true, "UPSTREAM_FAILURE with http_status: 429"
			}
			// Attempt JSON unmarshal if payload looks like JSON
			if strings.HasPrefix(strings.TrimSpace(ev.Payload), "{") {
				var parsed UpstreamFailureReceipt
				if err := json.Unmarshal([]byte(ev.Payload), &parsed); err == nil && parsed.HTTPStatus == http.StatusTooManyRequests {
					return true, "UPSTREAM_FAILURE with http_status: 429"
				}
			}
		}

		if ev.Kind == EventAccountRotation {
			if strings.Contains(ev.Reason, RotationProviderRateLimited) ||
				strings.Contains(ev.Payload, RotationProviderRateLimited) {
				return true, "ACCOUNT_ROTATION: provider_rate_limited"
			}
		}

		// Combined reason tokens in event
		if strings.Contains(ev.Reason, "ACCOUNT_ROTATION: provider_rate_limited") ||
			strings.Contains(ev.Reason, "ACCOUNT_ROTATION:provider_rate_limited") ||
			strings.Contains(ev.Reason, "UPSTREAM_FAILURE: 429") {
			return true, ev.Reason
		}
	}

	// 5. Inspect raw evidence text if provided
	if raw := evidence.RawEvidence; raw != "" {
		if (strings.Contains(raw, EventUpstreamFailure) || strings.Contains(raw, "upstream")) &&
			(strings.Contains(raw, "429") || strings.Contains(raw, "rate_limited")) {
			return true, "raw evidence matches UPSTREAM_FAILURE 429"
		}
		if strings.Contains(raw, EventAccountRotation) && strings.Contains(raw, RotationProviderRateLimited) {
			return true, "raw evidence matches ACCOUNT_ROTATION provider_rate_limited"
		}
	}

	return false, ""
}

// IsRateLimitExit reports whether the given exit evidence represents an upstream rate limit.
func IsRateLimitExit(evidence ChildExitEvidence) bool {
	rateLimited, _ := DetectRateLimit(evidence)
	return rateLimited
}

// ClassifyChildExit is a pure function that classifies a child exit against given restart parameters.
func ClassifyChildExit(exitCode int, runErr error, evidence ChildExitEvidence, restartsSoFar, restartLimit int) SupervisorDecision {
	sup := NewSupervisor(SupervisorConfig{CrashRestartLimit: restartLimit})
	sup.crashRestarts = restartsSoFar
	return sup.classifyLocked(exitCode, runErr, evidence)
}

func parseRetryAfter(raw string) (time.Duration, bool) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return 0, false
	}
	if strings.HasSuffix(raw, "s") {
		raw = strings.TrimSuffix(raw, "s")
	}
	if secs, err := strconv.Atoi(raw); err == nil && secs > 0 {
		return time.Duration(secs) * time.Second, true
	}
	return 0, false
}
