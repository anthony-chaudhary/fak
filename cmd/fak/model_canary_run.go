package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"net"
	"net/url"
	"os"
	"os/signal"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"time"
)

const (
	modelCanaryRunConfigSchema  = "fak.model-canary-run-config/v1"
	modelCanaryRunReceiptSchema = "fak.model-canary-run-receipt/v1"
)

type modelCanaryRunConfig struct {
	Schema    string                     `json:"schema"`
	Lease     modelCanaryLeaseConfig     `json:"lease"`
	Incumbent modelCanaryIncumbentConfig `json:"incumbent"`
	Candidate modelCanaryCandidateConfig `json:"candidate"`
	Request   modelCanaryRequestConfig   `json:"request"`
	Watcher   modelCanaryWatcherConfig   `json:"watcher"`
	Cleanup   modelCanaryCleanupConfig   `json:"cleanup"`
}

type modelCanaryLeaseConfig struct {
	Path    string `json:"path,omitempty"`
	Timeout string `json:"timeout"`
}

type modelCanaryIncumbentConfig struct {
	ListenerPort       int      `json:"listener_port"`
	LaunchdTarget      string   `json:"launchd_target"`
	ExpectedArgvSHA256 string   `json:"expected_argv_sha256"`
	RestorePlist       string   `json:"restore_plist"`
	RestorePlistSHA256 string   `json:"restore_plist_sha256"`
	RestoreCommand     []string `json:"restore_command"`
	StableEndpoints    []string `json:"stable_endpoints"`
}

type modelCanaryCandidateConfig struct {
	Engine             string            `json:"engine"`
	Command            []string          `json:"command"`
	Environment        map[string]string `json:"environment,omitempty"`
	ListenerPort       int               `json:"listener_port"`
	ReadinessEndpoints []string          `json:"readiness_endpoints"`
	ReadinessTimeout   string            `json:"readiness_timeout"`
}

type modelCanaryRequestConfig struct {
	Command     []string          `json:"command"`
	Environment map[string]string `json:"environment,omitempty"`
	Deadline    string            `json:"deadline"`
}

type modelCanaryWatcherConfig struct {
	Interval                   string `json:"interval"`
	ConsecutiveCrossings       int    `json:"consecutive_crossings"`
	MaximumRSSBytes            int64  `json:"maximum_rss_bytes"`
	MaximumFootprintBytes      int64  `json:"maximum_footprint_bytes"`
	MaximumSwapGrowthBytes     int64  `json:"maximum_swap_growth_bytes"`
	MinimumSystemFreePercent   int    `json:"minimum_system_free_percent"`
	MinimumMemorystatusPercent int    `json:"minimum_memorystatus_percent"`
}

type modelCanaryCleanupConfig struct {
	CandidateTERMTimeout string `json:"candidate_term_timeout"`
	RestoreTimeout       string `json:"restore_timeout"`
	StabilityDuration    string `json:"stability_duration"`
	ProbeInterval        string `json:"probe_interval"`
}

type modelCanaryDurations struct {
	LeaseTimeout      time.Duration
	ReadinessTimeout  time.Duration
	RequestDeadline   time.Duration
	SampleInterval    time.Duration
	CandidateTERM     time.Duration
	RestoreTimeout    time.Duration
	StabilityDuration time.Duration
	ProbeInterval     time.Duration
}

type modelCanaryPhase string

const (
	modelCanaryPhaseConfigValidated     modelCanaryPhase = "config_validated"
	modelCanaryPhasePreflightComplete   modelCanaryPhase = "preflight_complete"
	modelCanaryPhaseLeaseAcquired       modelCanaryPhase = "lease_acquired"
	modelCanaryPhaseIncumbentVerified   modelCanaryPhase = "incumbent_verified"
	modelCanaryPhaseIncumbentStopped    modelCanaryPhase = "incumbent_stopped"
	modelCanaryPhaseCandidateStarted    modelCanaryPhase = "candidate_started"
	modelCanaryPhaseCandidateReady      modelCanaryPhase = "candidate_ready"
	modelCanaryPhaseRequestStarted      modelCanaryPhase = "request_started"
	modelCanaryPhaseMonitoring          modelCanaryPhase = "monitoring"
	modelCanaryPhaseTerminalDecision    modelCanaryPhase = "terminal_decision"
	modelCanaryPhaseRequestStopped      modelCanaryPhase = "request_stopped"
	modelCanaryPhaseCandidateTerminated modelCanaryPhase = "candidate_terminated"
	modelCanaryPhaseIncumbentRestored   modelCanaryPhase = "incumbent_restored"
	modelCanaryPhaseEndpointsStable     modelCanaryPhase = "endpoints_stable"
	modelCanaryPhaseLeaseReleased       modelCanaryPhase = "lease_released"
	modelCanaryPhaseComplete            modelCanaryPhase = "complete"
	modelCanaryPhaseRefused             modelCanaryPhase = "refused"
)

const (
	modelCanaryReasonUnsupportedPlatform        = "UNSUPPORTED_PLATFORM"
	modelCanaryReasonConfigInvalid              = "CONFIG_INVALID"
	modelCanaryReasonPreflightFailed            = "PREFLIGHT_FAILED"
	modelCanaryReasonLeaseFailed                = "LEASE_FAILED"
	modelCanaryReasonIncumbentIdentityMismatch  = "INCUMBENT_IDENTITY_MISMATCH"
	modelCanaryReasonBootoutFailed              = "BOOTOUT_FAILED"
	modelCanaryReasonCandidateStartFailed       = "CANDIDATE_START_FAILED"
	modelCanaryReasonCandidateReadinessFailed   = "CANDIDATE_READINESS_FAILED"
	modelCanaryReasonRequestStartFailed         = "REQUEST_START_FAILED"
	modelCanaryReasonRequestEvidenceUnavailable = "REQUEST_EVIDENCE_UNAVAILABLE"
	modelCanaryReasonRequestDeadline            = "REQUEST_DEADLINE"
	modelCanaryReasonObservationUnavailable     = "OBSERVATION_UNAVAILABLE"
	modelCanaryReasonGuardTripped               = "GUARD_TRIPPED"
	modelCanaryReasonCandidateIdentityMismatch  = "CANDIDATE_IDENTITY_MISMATCH"
	modelCanaryReasonCandidateTERMFailed        = "CANDIDATE_TERM_FAILED"
	modelCanaryReasonRestoreFailed              = "RESTORE_FAILED"
	modelCanaryReasonStabilityFailed            = "STABILITY_FAILED"
	modelCanaryReasonLeaseReleaseFailed         = "LEASE_RELEASE_FAILED"
	modelCanaryReasonCanceled                   = "CANCELED"
)

type modelCanaryRefusal struct {
	Reason string
	Phase  modelCanaryPhase
	Detail string
}

func (e *modelCanaryRefusal) Error() string {
	if e == nil {
		return "model canary refused"
	}
	return fmt.Sprintf("%s at %s: %s", e.Reason, e.Phase, e.Detail)
}

type modelCanaryProcessIdentity struct {
	PID        int    `json:"pid"`
	StartedAt  string `json:"started_at"`
	ArgvSHA256 string `json:"argv_sha256"`
}

func (i modelCanaryProcessIdentity) valid() bool {
	if i.PID <= 0 || strings.TrimSpace(i.StartedAt) == "" || !validSHA256(i.ArgvSHA256) {
		return false
	}
	_, err := time.Parse(time.RFC3339Nano, i.StartedAt)
	return err == nil
}

func (i modelCanaryProcessIdentity) equal(other modelCanaryProcessIdentity) bool {
	return i.PID == other.PID && i.StartedAt == other.StartedAt && i.ArgvSHA256 == other.ArgvSHA256
}

type modelCanaryPreflight struct {
	Incumbent          modelCanaryProcessIdentity `json:"incumbent"`
	BaselineSwapBytes  int64                      `json:"baseline_swap_bytes"`
	Tools              map[string]string          `json:"tools"`
	ExecutableSHA256   map[string]string          `json:"executable_sha256,omitempty"`
	RestorePlistSHA256 string                     `json:"restore_plist_sha256"`
}

type modelCanarySample struct {
	Sequence            int                        `json:"sequence"`
	ObservedAt          string                     `json:"observed_at"`
	Candidate           modelCanaryProcessIdentity `json:"candidate"`
	RSSBytes            int64                      `json:"rss_bytes"`
	FootprintBytes      int64                      `json:"footprint_bytes"`
	SwapUsedBytes       int64                      `json:"swap_used_bytes"`
	SwapGrowthBytes     int64                      `json:"swap_growth_bytes"`
	SystemFreePercent   int                        `json:"system_free_percent"`
	MemorystatusPercent int                        `json:"memorystatus_percent"`
	Raw                 map[string]string          `json:"raw"`
	RawSHA256           map[string]string          `json:"raw_sha256"`
}

type modelCanaryGuardState struct {
	RSSCrossings          int    `json:"rss_crossings"`
	FootprintCrossings    int    `json:"footprint_crossings"`
	SwapGrowthCrossings   int    `json:"swap_growth_crossings"`
	SystemFreeCrossings   int    `json:"system_free_crossings"`
	MemorystatusCrossings int    `json:"memorystatus_crossings"`
	TrippedMetric         string `json:"tripped_metric,omitempty"`
	TripSequence          int    `json:"trip_sequence,omitempty"`
}

type modelCanaryEvent struct {
	Sequence int              `json:"sequence"`
	At       string           `json:"at"`
	Phase    modelCanaryPhase `json:"phase"`
	Action   string           `json:"action"`
	Status   string           `json:"status"`
	Detail   string           `json:"detail,omitempty"`
}

type modelCanaryCommandBinding struct {
	CandidateArgvSHA256       string `json:"candidate_argv_sha256"`
	CandidateEnvSHA256        string `json:"candidate_environment_sha256"`
	CandidateExecutableSHA256 string `json:"candidate_executable_sha256,omitempty"`
	RequestArgvSHA256         string `json:"request_argv_sha256"`
	RequestEnvSHA256          string `json:"request_environment_sha256"`
	RequestExecutableSHA256   string `json:"request_executable_sha256,omitempty"`
	RestoreArgvSHA256         string `json:"restore_argv_sha256"`
	RestoreExecutableSHA256   string `json:"restore_executable_sha256,omitempty"`
}

type modelCanaryRequestEvidence struct {
	CompletedAt  string `json:"completed_at"`
	Stdout       string `json:"stdout"`
	StdoutBytes  int    `json:"stdout_bytes"`
	StdoutSHA256 string `json:"stdout_sha256"`
	Stderr       string `json:"stderr"`
	StderrBytes  int    `json:"stderr_bytes"`
	StderrSHA256 string `json:"stderr_sha256"`
}

type modelCanaryRunReceipt struct {
	Schema               string                      `json:"schema"`
	ConfigSHA256         string                      `json:"config_sha256"`
	EvidenceSHA256       string                      `json:"evidence_sha256"`
	Engine               string                      `json:"engine,omitempty"`
	Platform             string                      `json:"platform"`
	Architecture         string                      `json:"architecture"`
	StartedAt            string                      `json:"started_at"`
	CompletedAt          string                      `json:"completed_at"`
	Outcome              string                      `json:"outcome"`
	TerminalPhase        modelCanaryPhase            `json:"terminal_phase"`
	Reason               string                      `json:"reason,omitempty"`
	Detail               string                      `json:"detail,omitempty"`
	Commands             modelCanaryCommandBinding   `json:"commands"`
	Preflight            *modelCanaryPreflight       `json:"preflight,omitempty"`
	Candidate            *modelCanaryProcessIdentity `json:"candidate,omitempty"`
	Request              *modelCanaryProcessIdentity `json:"request,omitempty"`
	RestoredIncumbent    *modelCanaryProcessIdentity `json:"restored_incumbent,omitempty"`
	RequestExitCode      *int                        `json:"request_exit_code,omitempty"`
	RequestEvidence      *modelCanaryRequestEvidence `json:"request_evidence,omitempty"`
	Guard                modelCanaryGuardState       `json:"guard"`
	Samples              []modelCanarySample         `json:"samples"`
	Events               []modelCanaryEvent          `json:"events"`
	LeaseReleased        bool                        `json:"lease_released"`
	RestorationAttempted bool                        `json:"restoration_attempted"`
}

type modelCanaryLease interface {
	Release() error
}

// modelCanaryProcess deliberately carries no os.Process. Live process ownership stays in the
// Darwin adapter; deterministic tests can use a small token while the common transaction sees
// only the identity that must be rechecked before a signal.
type modelCanaryProcess struct {
	Identity modelCanaryProcessIdentity
	Handle   any
}

// modelCanaryRunDeps is the host-independent transaction boundary. Every mutating operation is
// injected so the complete handoff, watcher, cleanup, restoration, and lease ordering can be
// exercised without a model, GPU, launchd, or a fixed port.
type modelCanaryRunDeps struct {
	Platform           string
	Architecture       string
	Now                func() time.Time
	Preflight          func(context.Context, modelCanaryRunConfig) (modelCanaryPreflight, error)
	AcquireLease       func(context.Context, modelCanaryLeaseConfig, time.Duration) (modelCanaryLease, error)
	VerifyIncumbent    func(context.Context, modelCanaryRunConfig, modelCanaryProcessIdentity) (modelCanaryProcessIdentity, error)
	BootoutIncumbent   func(context.Context, modelCanaryRunConfig, modelCanaryProcessIdentity, time.Duration) error
	StartCandidate     func(context.Context, modelCanaryCandidateConfig) (modelCanaryProcess, error)
	WaitCandidateReady func(context.Context, modelCanaryRunConfig, modelCanaryProcess, time.Duration) error
	StartRequest       func(context.Context, modelCanaryRequestConfig) (modelCanaryProcess, error)
	PollRequest        func(modelCanaryProcess) (done bool, exitCode int, err error)
	RequestEvidence    func(modelCanaryProcess) (modelCanaryRequestEvidence, error)
	StopRequest        func(context.Context, modelCanaryProcess, time.Duration) error
	Sample             func(context.Context, modelCanaryProcess, int64) (modelCanarySample, error)
	Sleep              func(context.Context, time.Duration) error
	TermCandidate      func(context.Context, modelCanaryProcess, time.Duration) error
	RestoreIncumbent   func(context.Context, modelCanaryRunConfig, time.Duration) error
	EndpointsStable    func(context.Context, modelCanaryRunConfig, time.Duration, time.Duration) (modelCanaryProcessIdentity, error)
}

func runModelCanaryRun(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("model canary-run", flag.ContinueOnError)
	fs.SetOutput(stderr)
	configPath := fs.String("config", "", "strict model-canary-run config JSON")
	receiptPath := fs.String("receipt", "", "atomic terminal receipt JSON")
	fs.Usage = func() {
		fmt.Fprintln(stderr, "usage: fak model canary-run --config CONFIG.json --receipt RECEIPT.json")
		fs.PrintDefaults()
	}
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 0 || strings.TrimSpace(*configPath) == "" || strings.TrimSpace(*receiptPath) == "" {
		fs.Usage()
		return 2
	}

	started := time.Now().UTC()
	raw, err := os.ReadFile(*configPath)
	if err != nil {
		fmt.Fprintf(stderr, "fak model canary-run: read config: %v\n", err)
		return 1
	}
	configSHA := digestBytes(raw)
	var cfg modelCanaryRunConfig
	if err := decodeModelCanaryStrict(raw, &cfg); err != nil {
		return failModelCanaryConfig(stderr, *receiptPath, started, configSHA, err)
	}
	durations, err := validateModelCanaryConfig(cfg)
	if err != nil {
		return failModelCanaryConfig(stderr, *receiptPath, started, configSHA, err)
	}

	deps, depErr := modelCanaryLiveDependencies()
	var receipt modelCanaryRunReceipt
	if depErr != nil {
		receipt = newModelCanaryReceipt(started, configSHA, runtime.GOOS, runtime.GOARCH)
		receipt.Engine = cfg.Candidate.Engine
		refuseModelCanary(&receipt, modelCanaryPhasePreflightComplete, modelCanaryReasonUnsupportedPlatform, depErr.Error())
		finishModelCanaryReceipt(&receipt, time.Now().UTC())
	} else {
		runCtx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
		receipt = executeModelCanaryRun(runCtx, cfg, durations, configSHA, deps)
		stop()
	}
	if err := writeModelCanaryReceiptAtomic(*receiptPath, receipt); err != nil {
		fmt.Fprintf(stderr, "fak model canary-run: write terminal receipt: %v\n", err)
		return 1
	}
	encoded, _ := json.Marshal(receipt)
	fmt.Fprintln(stdout, string(encoded))
	if receipt.Outcome != "complete" {
		fmt.Fprintf(stderr, "fak model canary-run: %s at %s: %s\n", receipt.Reason, receipt.TerminalPhase, receipt.Detail)
		return 1
	}
	return 0
}

func executeModelCanaryRun(ctx context.Context, cfg modelCanaryRunConfig, durations modelCanaryDurations, configSHA string, deps modelCanaryRunDeps) modelCanaryRunReceipt {
	now := deps.Now
	if now == nil {
		now = time.Now
	}
	receipt := newModelCanaryReceipt(now().UTC(), configSHA, deps.Platform, deps.Architecture)
	receipt.Engine = cfg.Candidate.Engine
	receipt.Commands = modelCanaryCommandBinding{
		CandidateArgvSHA256: hashModelCanaryArgv(cfg.Candidate.Command),
		CandidateEnvSHA256:  hashModelCanaryEnvironment(cfg.Candidate.Environment),
		RequestArgvSHA256:   hashModelCanaryArgv(cfg.Request.Command),
		RequestEnvSHA256:    hashModelCanaryEnvironment(cfg.Request.Environment),
		RestoreArgvSHA256:   hashModelCanaryArgv(cfg.Incumbent.RestoreCommand),
	}
	addModelCanaryEvent(&receipt, now(), modelCanaryPhaseConfigValidated, "validate_config", "ok", "")

	preflight, err := deps.Preflight(ctx, cfg)
	if err != nil {
		refuseModelCanary(&receipt, modelCanaryPhasePreflightComplete, modelCanaryReasonPreflightFailed, err.Error())
		addModelCanaryEvent(&receipt, now(), modelCanaryPhasePreflightComplete, "preflight", "refused", err.Error())
		finishModelCanaryReceipt(&receipt, now().UTC())
		return receipt
	}
	if err := validateModelCanaryPreflight(cfg, preflight); err != nil {
		detail := err.Error()
		refuseModelCanary(&receipt, modelCanaryPhasePreflightComplete, modelCanaryReasonPreflightFailed, detail)
		addModelCanaryEvent(&receipt, now(), modelCanaryPhasePreflightComplete, "preflight", "refused", detail)
		finishModelCanaryReceipt(&receipt, now().UTC())
		return receipt
	}
	receipt.Preflight = &preflight
	receipt.Commands.CandidateExecutableSHA256 = preflight.ExecutableSHA256["candidate"]
	receipt.Commands.RequestExecutableSHA256 = preflight.ExecutableSHA256["request"]
	receipt.Commands.RestoreExecutableSHA256 = preflight.ExecutableSHA256["restore"]
	addModelCanaryEvent(&receipt, now(), modelCanaryPhasePreflightComplete, "preflight", "ok", "all live parsers and declared identities verified before lease")

	lease, err := deps.AcquireLease(ctx, cfg.Lease, durations.LeaseTimeout)
	if err != nil {
		refuseModelCanary(&receipt, modelCanaryPhaseLeaseAcquired, modelCanaryReasonLeaseFailed, err.Error())
		addModelCanaryEvent(&receipt, now(), modelCanaryPhaseLeaseAcquired, "acquire_gpu_lease", "refused", err.Error())
		finishModelCanaryReceipt(&receipt, now().UTC())
		return receipt
	}
	addModelCanaryEvent(&receipt, now(), modelCanaryPhaseLeaseAcquired, "acquire_gpu_lease", "ok", "")

	bootoutAttempted := false
	var candidate, request modelCanaryProcess
	requestRunning := false
	terminalSet := false

	setFailure := func(phase modelCanaryPhase, reason string, cause error) {
		if cause == nil {
			return
		}
		refuseModelCanary(&receipt, phase, reason, cause.Error())
		terminalSet = true
	}

	verified, err := deps.VerifyIncumbent(ctx, cfg, preflight.Incumbent)
	if err != nil || !verified.equal(preflight.Incumbent) {
		if err == nil {
			err = errors.New("incumbent PID/start/argv identity changed after lease acquisition")
		}
		setFailure(modelCanaryPhaseIncumbentVerified, modelCanaryReasonIncumbentIdentityMismatch, err)
		addModelCanaryEvent(&receipt, now(), modelCanaryPhaseIncumbentVerified, "reverify_incumbent", "refused", err.Error())
	} else {
		addModelCanaryEvent(&receipt, now(), modelCanaryPhaseIncumbentVerified, "reverify_incumbent", "ok", "")
		bootoutAttempted = true // bootout can mutate before returning an error, so restoration is owed from here.
		if err = deps.BootoutIncumbent(ctx, cfg, verified, durations.CandidateTERM); err != nil {
			setFailure(modelCanaryPhaseIncumbentStopped, modelCanaryReasonBootoutFailed, err)
			addModelCanaryEvent(&receipt, now(), modelCanaryPhaseIncumbentStopped, "bootout_incumbent", "error", err.Error())
		} else {
			addModelCanaryEvent(&receipt, now(), modelCanaryPhaseIncumbentStopped, "bootout_incumbent", "ok", "")
		}
	}

	if !terminalSet {
		candidate, err = deps.StartCandidate(ctx, cfg.Candidate)
		if err != nil || !candidate.Identity.valid() {
			if err == nil {
				err = errors.New("candidate start returned an incomplete PID/start/argv identity")
			}
			setFailure(modelCanaryPhaseCandidateStarted, modelCanaryReasonCandidateStartFailed, err)
			addModelCanaryEvent(&receipt, now(), modelCanaryPhaseCandidateStarted, "start_candidate", "error", err.Error())
		} else {
			receipt.Candidate = &candidate.Identity
			addModelCanaryEvent(&receipt, now(), modelCanaryPhaseCandidateStarted, "start_candidate", "ok", "")
		}
	}
	if !terminalSet {
		if err = deps.WaitCandidateReady(ctx, cfg, candidate, durations.ReadinessTimeout); err != nil {
			setFailure(modelCanaryPhaseCandidateReady, modelCanaryReasonCandidateReadinessFailed, err)
			addModelCanaryEvent(&receipt, now(), modelCanaryPhaseCandidateReady, "wait_candidate_ready", "error", err.Error())
		} else {
			addModelCanaryEvent(&receipt, now(), modelCanaryPhaseCandidateReady, "wait_candidate_ready", "ok", "")
		}
	}
	if !terminalSet {
		request, err = deps.StartRequest(ctx, cfg.Request)
		if err != nil || !request.Identity.valid() {
			if err == nil {
				err = errors.New("request start returned an incomplete PID/start/argv identity")
			}
			setFailure(modelCanaryPhaseRequestStarted, modelCanaryReasonRequestStartFailed, err)
			addModelCanaryEvent(&receipt, now(), modelCanaryPhaseRequestStarted, "start_request", "error", err.Error())
		} else {
			requestRunning = true
			receipt.Request = &request.Identity
			addModelCanaryEvent(&receipt, now(), modelCanaryPhaseRequestStarted, "start_request", "ok", "")
		}
	}

	monitorModelCanary(modelCanaryMonitorState{
		ctx: ctx, cfg: cfg, durations: durations, deps: deps, receipt: &receipt,
		candidate: candidate, request: request, preflight: preflight, now: now,
		setFailure: setFailure, requestRunning: &requestRunning, terminalSet: &terminalSet,
	})

	// Cleanup is intentionally detached from request/session cancellation. Its bound covers all
	// candidate TERM, exact restore, and stability work; the lease remains held until it ends.
	cleanupBudget := 2*durations.CandidateTERM + durations.RestoreTimeout + durations.StabilityDuration + 2*durations.ProbeInterval
	if cleanupBudget <= 0 {
		cleanupBudget = time.Minute
	}
	cleanupCtx, cancelCleanup := context.WithTimeout(context.WithoutCancel(ctx), cleanupBudget)
	defer cancelCleanup()

	if requestRunning {
		if err := deps.StopRequest(cleanupCtx, request, durations.CandidateTERM); err != nil {
			setFailure(modelCanaryPhaseRequestStopped, modelCanaryReasonCandidateTERMFailed, fmt.Errorf("stop request: %w", err))
			addModelCanaryEvent(&receipt, now(), modelCanaryPhaseRequestStopped, "stop_request", "error", err.Error())
		} else {
			addModelCanaryEvent(&receipt, now(), modelCanaryPhaseRequestStopped, "stop_request", "ok", "")
		}
	}
	if candidate.Identity.PID > 0 {
		if err := deps.TermCandidate(cleanupCtx, candidate, durations.CandidateTERM); err != nil {
			reason := modelCanaryReasonCandidateTERMFailed
			var refusal *modelCanaryRefusal
			if errors.As(err, &refusal) && refusal.Reason == modelCanaryReasonCandidateIdentityMismatch {
				reason = modelCanaryReasonCandidateIdentityMismatch
			}
			setFailure(modelCanaryPhaseCandidateTerminated, reason, err)
			addModelCanaryEvent(&receipt, now(), modelCanaryPhaseCandidateTerminated, "term_candidate", "refused", err.Error())
		} else {
			addModelCanaryEvent(&receipt, now(), modelCanaryPhaseCandidateTerminated, "term_candidate", "ok", "TERM only; exact identity rechecked")
		}
	}

	if bootoutAttempted {
		receipt.RestorationAttempted = true
		if err := deps.RestoreIncumbent(cleanupCtx, cfg, durations.RestoreTimeout); err != nil {
			setFailure(modelCanaryPhaseIncumbentRestored, modelCanaryReasonRestoreFailed, err)
			addModelCanaryEvent(&receipt, now(), modelCanaryPhaseIncumbentRestored, "restore_incumbent", "error", err.Error())
		} else {
			addModelCanaryEvent(&receipt, now(), modelCanaryPhaseIncumbentRestored, "restore_incumbent", "ok", "exact declared restore command")
			restored, stableErr := deps.EndpointsStable(cleanupCtx, cfg, durations.StabilityDuration, durations.ProbeInterval)
			if stableErr != nil {
				setFailure(modelCanaryPhaseEndpointsStable, modelCanaryReasonStabilityFailed, stableErr)
				addModelCanaryEvent(&receipt, now(), modelCanaryPhaseEndpointsStable, "prove_restoration_stable", "error", stableErr.Error())
			} else if !restored.valid() || restored.ArgvSHA256 != preflight.Incumbent.ArgvSHA256 {
				err := errors.New("restored incumbent launchd/argv identity does not match preflight incumbent")
				setFailure(modelCanaryPhaseEndpointsStable, modelCanaryReasonStabilityFailed, err)
				addModelCanaryEvent(&receipt, now(), modelCanaryPhaseEndpointsStable, "prove_restoration_stable", "refused", err.Error())
			} else {
				receipt.RestoredIncumbent = &restored
				addModelCanaryEvent(&receipt, now(), modelCanaryPhaseEndpointsStable, "prove_restoration_stable", "ok", "")
			}
		}
	}

	// Release is deliberately the last external action, even when restoration failed.
	if err := lease.Release(); err != nil {
		setFailure(modelCanaryPhaseLeaseReleased, modelCanaryReasonLeaseReleaseFailed, err)
		addModelCanaryEvent(&receipt, now(), modelCanaryPhaseLeaseReleased, "release_gpu_lease", "error", err.Error())
	} else {
		receipt.LeaseReleased = true
		addModelCanaryEvent(&receipt, now(), modelCanaryPhaseLeaseReleased, "release_gpu_lease", "ok", "")
	}

	if !terminalSet || receipt.Outcome == "complete" {
		if receipt.RestorationAttempted && receipt.RestoredIncumbent == nil {
			refuseModelCanary(&receipt, modelCanaryPhaseEndpointsStable, modelCanaryReasonStabilityFailed, "restoration did not reach an identity-bound stable endpoint state")
		} else if !receipt.LeaseReleased {
			refuseModelCanary(&receipt, modelCanaryPhaseLeaseReleased, modelCanaryReasonLeaseReleaseFailed, "GPU lease release was not witnessed")
		} else {
			receipt.Outcome = "complete"
			receipt.TerminalPhase = modelCanaryPhaseComplete
			receipt.Reason, receipt.Detail = "", ""
		}
	}
	if receipt.Outcome != "complete" && receipt.TerminalPhase == "" {
		refuseModelCanary(&receipt, modelCanaryPhaseRefused, modelCanaryReasonPreflightFailed, "transaction refused without a terminal phase")
	}
	finishModelCanaryReceipt(&receipt, now().UTC())
	return receipt
}

func validateModelCanaryRequestEvidence(evidence modelCanaryRequestEvidence) error {
	if _, err := time.Parse(time.RFC3339Nano, evidence.CompletedAt); err != nil {
		return errors.New("request evidence completed_at is missing or malformed")
	}
	if evidence.StdoutBytes != len([]byte(evidence.Stdout)) || digestBytes([]byte(evidence.Stdout)) != evidence.StdoutSHA256 {
		return errors.New("request stdout byte count or SHA256 binding is invalid")
	}
	if evidence.StderrBytes != len([]byte(evidence.Stderr)) || digestBytes([]byte(evidence.Stderr)) != evidence.StderrSHA256 {
		return errors.New("request stderr byte count or SHA256 binding is invalid")
	}
	return nil
}

func validateModelCanaryPreflight(cfg modelCanaryRunConfig, preflight modelCanaryPreflight) error {
	if !preflight.Incumbent.valid() || preflight.BaselineSwapBytes < 0 || !validSHA256(preflight.RestorePlistSHA256) {
		return errors.New("preflight returned incomplete incumbent, swap, or restore-plist identity")
	}
	if !sameModelCanaryDigest(preflight.Incumbent.ArgvSHA256, cfg.Incumbent.ExpectedArgvSHA256) {
		return errors.New("preflight incumbent argv identity differs from the strict config")
	}
	if !sameModelCanaryDigest(preflight.RestorePlistSHA256, cfg.Incumbent.RestorePlistSHA256) {
		return errors.New("preflight restore-plist identity differs from the strict config")
	}
	for _, name := range []string{"lsof", "ps", "footprint", "sysctl", "memory_pressure", "launchctl", "candidate", "request", "restore"} {
		path := strings.TrimSpace(preflight.Tools[name])
		if path == "" || !filepath.IsAbs(path) {
			return fmt.Errorf("preflight tool %s has no absolute executable path", name)
		}
		if !validSHA256(preflight.ExecutableSHA256[name]) {
			return fmt.Errorf("preflight tool %s has no valid executable SHA256", name)
		}
	}
	return nil
}

func newModelCanaryReceipt(start time.Time, configSHA, platform, architecture string) modelCanaryRunReceipt {
	return modelCanaryRunReceipt{
		Schema: modelCanaryRunReceiptSchema, ConfigSHA256: configSHA,
		Platform: platform, Architecture: architecture,
		StartedAt: start.UTC().Format(time.RFC3339Nano), Outcome: "refused",
		Samples: []modelCanarySample{}, Events: []modelCanaryEvent{},
	}
}

func refuseModelCanary(receipt *modelCanaryRunReceipt, phase modelCanaryPhase, reason, detail string) {
	receipt.Outcome, receipt.TerminalPhase = "refused", phase
	receipt.Reason, receipt.Detail = reason, detail
}

func addModelCanaryEvent(receipt *modelCanaryRunReceipt, at time.Time, phase modelCanaryPhase, action, status, detail string) {
	receipt.Events = append(receipt.Events, modelCanaryEvent{
		Sequence: len(receipt.Events) + 1, At: at.UTC().Format(time.RFC3339Nano),
		Phase: phase, Action: action, Status: status, Detail: detail,
	})
}

func finishModelCanaryReceipt(receipt *modelCanaryRunReceipt, completed time.Time) {
	receipt.CompletedAt = completed.UTC().Format(time.RFC3339Nano)
	receipt.EvidenceSHA256 = ""
	data, err := json.Marshal(receipt)
	if err == nil {
		receipt.EvidenceSHA256 = digestBytes(data)
	}
}

func recomputeModelCanaryEvidenceSHA256(receipt modelCanaryRunReceipt) (string, error) {
	receipt.EvidenceSHA256 = ""
	data, err := json.Marshal(receipt)
	if err != nil {
		return "", err
	}
	return digestBytes(data), nil
}

func decodeModelCanaryStrict(raw []byte, target any) error {
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(target); err != nil {
		return fmt.Errorf("strict config decode: %w", err)
	}
	var extra any
	if err := decoder.Decode(&extra); !errors.Is(err, io.EOF) {
		if err == nil {
			return errors.New("strict config decode: multiple JSON values are not allowed")
		}
		return fmt.Errorf("strict config decode trailing data: %w", err)
	}
	return nil
}

func validateModelCanaryConfig(cfg modelCanaryRunConfig) (modelCanaryDurations, error) {
	var durations modelCanaryDurations
	if cfg.Schema != modelCanaryRunConfigSchema {
		return durations, fmt.Errorf("schema must be %q", modelCanaryRunConfigSchema)
	}
	parsePositive := func(name, raw string) (time.Duration, error) {
		value, err := time.ParseDuration(raw)
		if err != nil || value <= 0 {
			return 0, fmt.Errorf("%s must be a positive Go duration", name)
		}
		return value, nil
	}
	var err error
	fields := []struct {
		name string
		raw  string
		dst  *time.Duration
	}{
		{"lease.timeout", cfg.Lease.Timeout, &durations.LeaseTimeout},
		{"candidate.readiness_timeout", cfg.Candidate.ReadinessTimeout, &durations.ReadinessTimeout},
		{"request.deadline", cfg.Request.Deadline, &durations.RequestDeadline},
		{"watcher.interval", cfg.Watcher.Interval, &durations.SampleInterval},
		{"cleanup.candidate_term_timeout", cfg.Cleanup.CandidateTERMTimeout, &durations.CandidateTERM},
		{"cleanup.restore_timeout", cfg.Cleanup.RestoreTimeout, &durations.RestoreTimeout},
		{"cleanup.stability_duration", cfg.Cleanup.StabilityDuration, &durations.StabilityDuration},
		{"cleanup.probe_interval", cfg.Cleanup.ProbeInterval, &durations.ProbeInterval},
	}
	for _, field := range fields {
		if *field.dst, err = parsePositive(field.name, field.raw); err != nil {
			return durations, err
		}
	}
	if err := validateModelCanaryPort("incumbent.listener_port", cfg.Incumbent.ListenerPort); err != nil {
		return durations, err
	}
	if err := validateModelCanaryPort("candidate.listener_port", cfg.Candidate.ListenerPort); err != nil {
		return durations, err
	}
	if cfg.Candidate.ListenerPort != cfg.Incumbent.ListenerPort {
		return durations, errors.New("candidate.listener_port must equal incumbent.listener_port for an exact handoff")
	}
	if strings.TrimSpace(cfg.Incumbent.LaunchdTarget) == "" || strings.ContainsAny(cfg.Incumbent.LaunchdTarget, "\r\n\x00") {
		return durations, errors.New("incumbent.launchd_target is required and must be one line")
	}
	if !validSHA256(cfg.Incumbent.ExpectedArgvSHA256) {
		return durations, errors.New("incumbent.expected_argv_sha256 must be a sha256 digest")
	}
	if !filepath.IsAbs(cfg.Incumbent.RestorePlist) {
		return durations, errors.New("incumbent.restore_plist must be an absolute path")
	}
	if !validSHA256(cfg.Incumbent.RestorePlistSHA256) {
		return durations, errors.New("incumbent.restore_plist_sha256 must be a sha256 digest")
	}
	for name, command := range map[string][]string{"candidate.command": cfg.Candidate.Command, "request.command": cfg.Request.Command, "incumbent.restore_command": cfg.Incumbent.RestoreCommand} {
		if err := validateModelCanaryCommand(name, command); err != nil {
			return durations, err
		}
	}
	if cfg.Candidate.Engine != "fak-native" {
		return durations, errors.New("candidate.engine must be \"fak-native\"; canary-run never falls back to another inference engine")
	}
	if err := validateModelCanaryEnvironment("candidate.environment", cfg.Candidate.Environment); err != nil {
		return durations, err
	}
	if err := validateModelCanaryEnvironment("request.environment", cfg.Request.Environment); err != nil {
		return durations, err
	}
	if err := validateModelCanaryEndpoints("candidate.readiness_endpoints", cfg.Candidate.ReadinessEndpoints, cfg.Candidate.ListenerPort); err != nil {
		return durations, err
	}
	if err := validateModelCanaryEndpoints("incumbent.stable_endpoints", cfg.Incumbent.StableEndpoints, cfg.Incumbent.ListenerPort); err != nil {
		return durations, err
	}
	if cfg.Watcher.ConsecutiveCrossings < 1 {
		return durations, errors.New("watcher.consecutive_crossings must be at least 1")
	}
	if cfg.Watcher.MaximumRSSBytes <= 0 || cfg.Watcher.MaximumFootprintBytes <= 0 || cfg.Watcher.MaximumSwapGrowthBytes <= 0 {
		return durations, errors.New("watcher maximum byte thresholds must all be positive")
	}
	if cfg.Watcher.MinimumSystemFreePercent < 1 || cfg.Watcher.MinimumSystemFreePercent > 100 || cfg.Watcher.MinimumMemorystatusPercent < 1 || cfg.Watcher.MinimumMemorystatusPercent > 100 {
		return durations, errors.New("watcher minimum percent thresholds must be in [1,100]")
	}
	return durations, nil
}

func validateModelCanaryPort(name string, port int) error {
	if port < 1 || port > 65535 {
		return fmt.Errorf("%s must be in [1,65535]", name)
	}
	return nil
}

func validateModelCanaryCommand(name string, argv []string) error {
	if len(argv) == 0 || strings.TrimSpace(argv[0]) == "" {
		return fmt.Errorf("%s must declare an executable and arguments", name)
	}
	for _, arg := range argv {
		if strings.ContainsRune(arg, '\x00') {
			return fmt.Errorf("%s contains a NUL argument", name)
		}
	}
	return nil
}

func validateModelCanaryEnvironment(name string, env map[string]string) error {
	for key, value := range env {
		if key == "" || strings.ContainsAny(key, "=\x00") || strings.ContainsRune(value, '\x00') {
			return fmt.Errorf("%s contains an invalid key or value", name)
		}
	}
	return nil
}

func validateModelCanaryEndpoints(name string, endpoints []string, port int) error {
	if len(endpoints) == 0 {
		return fmt.Errorf("%s must declare at least one endpoint", name)
	}
	seen := make(map[string]bool, len(endpoints))
	for _, endpoint := range endpoints {
		parsed, err := url.Parse(endpoint)
		if err != nil || (parsed.Scheme != "http" && parsed.Scheme != "https") || parsed.Hostname() == "" || parsed.User != nil {
			return fmt.Errorf("%s contains invalid HTTP endpoint %q", name, endpoint)
		}
		ip := net.ParseIP(parsed.Hostname())
		if parsed.Hostname() != "localhost" && (ip == nil || !ip.IsLoopback()) {
			return fmt.Errorf("%s endpoint %q is not loopback", name, endpoint)
		}
		endpointPort := parsed.Port()
		if endpointPort == "" {
			if (parsed.Scheme == "http" && port != 80) || (parsed.Scheme == "https" && port != 443) {
				return fmt.Errorf("%s endpoint %q does not name listener port %d", name, endpoint, port)
			}
		} else if endpointPort != strconv.Itoa(port) {
			return fmt.Errorf("%s endpoint %q does not name listener port %d", name, endpoint, port)
		}
		if seen[endpoint] {
			return fmt.Errorf("%s contains duplicate endpoint %q", name, endpoint)
		}
		seen[endpoint] = true
	}
	return nil
}

func validateModelCanarySampleAt(sample modelCanarySample, expected modelCanaryProcessIdentity, baselineSwap int64, previous, now time.Time, interval time.Duration) (time.Time, error) {
	if !sample.Candidate.equal(expected) {
		return time.Time{}, errors.New("sample candidate PID/start/argv identity does not match the launched candidate")
	}
	observed, err := time.Parse(time.RFC3339Nano, sample.ObservedAt)
	if err != nil {
		return time.Time{}, errors.New("sample observed_at is missing or malformed")
	}
	observed = observed.UTC()
	if !previous.IsZero() && !observed.After(previous) {
		return time.Time{}, errors.New("sample observed_at is stale or non-monotonic")
	}
	maxAge := 3 * interval
	if maxAge < 5*time.Second {
		maxAge = 5 * time.Second
	}
	if observed.After(now.Add(time.Second)) || now.Sub(observed) > maxAge {
		return time.Time{}, errors.New("sample observed_at is stale or in the future")
	}
	if sample.RSSBytes <= 0 || sample.FootprintBytes <= 0 || sample.SwapUsedBytes < 0 || sample.SystemFreePercent < 0 || sample.SystemFreePercent > 100 || sample.MemorystatusPercent < 0 || sample.MemorystatusPercent > 100 {
		return time.Time{}, errors.New("sample contains a non-positive process byte value, negative system byte value, or out-of-range percentage")
	}
	for _, source := range []string{"ps", "footprint", "swap", "memory_pressure", "memorystatus"} {
		raw, rawOK := sample.Raw[source]
		digest, digestOK := sample.RawSHA256[source]
		if !rawOK || !digestOK || digestBytes([]byte(raw)) != digest {
			return time.Time{}, fmt.Errorf("sample source %s is absent or its raw binding is invalid", source)
		}
	}
	rss, err := parseModelCanaryRSS([]byte(sample.Raw["ps"]), expected.PID)
	if err != nil || rss != sample.RSSBytes {
		return time.Time{}, errors.New("sample rss_bytes does not independently recompute from raw BSD ps")
	}
	footprint, err := parseModelCanaryFootprint([]byte(sample.Raw["footprint"]))
	if err != nil || footprint != sample.FootprintBytes {
		return time.Time{}, errors.New("sample footprint_bytes does not independently recompute from raw footprint output")
	}
	swap, err := parseModelCanarySwap([]byte(sample.Raw["swap"]))
	if err != nil || swap != sample.SwapUsedBytes || swap-baselineSwap != sample.SwapGrowthBytes {
		return time.Time{}, errors.New("sample swap values do not independently recompute from raw vm.swapusage and the preflight baseline")
	}
	systemFree, err := parseModelCanaryMemoryPressure([]byte(sample.Raw["memory_pressure"]))
	if err != nil || systemFree != sample.SystemFreePercent {
		return time.Time{}, errors.New("sample system_free_percent does not independently recompute from raw memory_pressure output")
	}
	memorystatus, err := parseModelCanaryMemorystatus([]byte(sample.Raw["memorystatus"]))
	if err != nil || memorystatus != sample.MemorystatusPercent {
		return time.Time{}, errors.New("sample memorystatus_percent does not independently recompute from raw kern.memorystatus_level")
	}
	return observed, nil
}

func foldModelCanaryGuard(state modelCanaryGuardState, cfg modelCanaryWatcherConfig, sample modelCanarySample) modelCanaryGuardState {
	if state.TrippedMetric != "" {
		return state
	}
	state.RSSCrossings = nextModelCanaryCrossing(state.RSSCrossings, sample.RSSBytes > cfg.MaximumRSSBytes)
	state.FootprintCrossings = nextModelCanaryCrossing(state.FootprintCrossings, sample.FootprintBytes > cfg.MaximumFootprintBytes)
	state.SwapGrowthCrossings = nextModelCanaryCrossing(state.SwapGrowthCrossings, sample.SwapGrowthBytes > cfg.MaximumSwapGrowthBytes)
	state.SystemFreeCrossings = nextModelCanaryCrossing(state.SystemFreeCrossings, sample.SystemFreePercent < cfg.MinimumSystemFreePercent)
	state.MemorystatusCrossings = nextModelCanaryCrossing(state.MemorystatusCrossings, sample.MemorystatusPercent < cfg.MinimumMemorystatusPercent)
	for _, metric := range []struct {
		name  string
		count int
	}{
		{"rss_bytes", state.RSSCrossings},
		{"footprint_bytes", state.FootprintCrossings},
		{"swap_growth_bytes", state.SwapGrowthCrossings},
		{"system_free_percent", state.SystemFreeCrossings},
		{"memorystatus_percent", state.MemorystatusCrossings},
	} {
		if metric.count >= cfg.ConsecutiveCrossings {
			state.TrippedMetric, state.TripSequence = metric.name, sample.Sequence
			break
		}
	}
	return state
}

func nextModelCanaryCrossing(current int, crossed bool) int {
	if !crossed {
		return 0
	}
	return current + 1
}

func writeModelCanaryReceiptAtomic(path string, receipt modelCanaryRunReceipt) error {
	if strings.TrimSpace(path) == "" {
		return errors.New("receipt path is required")
	}
	if receipt.Schema != modelCanaryRunReceiptSchema || receipt.CompletedAt == "" || receipt.TerminalPhase == "" || receipt.Outcome == "" {
		return errors.New("terminal receipt is incomplete")
	}
	want, err := recomputeModelCanaryEvidenceSHA256(receipt)
	if err != nil || receipt.EvidenceSHA256 != want {
		return errors.New("terminal receipt evidence hash is absent or stale")
	}
	data, err := json.MarshalIndent(receipt, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	dir := filepath.Dir(path)
	tmp, err := os.CreateTemp(dir, ".model-canary-run-*.tmp")
	if err != nil {
		return err
	}
	tmpPath := tmp.Name()
	closed := false
	defer func() {
		if !closed {
			_ = tmp.Close()
		}
		_ = os.Remove(tmpPath)
	}()
	if err := tmp.Chmod(0o600); err != nil {
		return err
	}
	if _, err := tmp.Write(data); err != nil {
		return err
	}
	if err := tmp.Sync(); err != nil {
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	closed = true
	return os.Rename(tmpPath, path)
}

func hashModelCanaryArgv(argv []string) string {
	encoded, _ := json.Marshal(argv)
	return digestBytes(encoded)
}

func hashModelCanaryEnvironment(env map[string]string) string {
	return hashModelCanaryArgv(modelCanaryEnvironmentRows(env))
}

// modelCanaryEnvironmentRows returns the complete environment passed to a canary-owned
// process. The live adapter must not inherit ambient variables that are absent from the
// strict config because the receipt only binds these rows.
func modelCanaryEnvironmentRows(env map[string]string) []string {
	keys := make([]string, 0, len(env))
	for key := range env {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	rows := make([]string, 0, len(keys))
	for _, key := range keys {
		rows = append(rows, key+"="+env[key])
	}
	return rows
}

func parseDarwinModelCanaryLaunchctl(raw []byte, target string) (int, string, error) {
	lines := nonEmptyModelCanaryLines(raw)
	if len(lines) == 0 || !strings.Contains(strings.TrimSpace(lines[0]), target+" = {") {
		return 0, "", errors.New("launchctl print output does not name the exact declared target")
	}
	var pid int
	var path string
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "pid = ") {
			value, err := strconv.Atoi(strings.TrimSpace(strings.TrimPrefix(trimmed, "pid = ")))
			if err != nil || value <= 0 || pid != 0 {
				return 0, "", errors.New("launchctl print contains malformed or duplicate pid")
			}
			pid = value
		}
		if strings.HasPrefix(trimmed, "path = ") {
			value := strings.TrimSpace(strings.TrimPrefix(trimmed, "path = "))
			if !isDarwinAbsolutePath(value) || path != "" {
				return 0, "", errors.New("launchctl print contains malformed or duplicate plist path")
			}
			path = filepath.Clean(value)
		}
	}
	if pid <= 0 || path == "" {
		return 0, "", errors.New("launchctl print omitted PID or plist path")
	}
	return pid, path, nil
}

func isDarwinAbsolutePath(path string) bool {
	return strings.HasPrefix(path, "/") && !strings.ContainsAny(path, "\r\n\x00")
}

func digestBytes(raw []byte) string {
	digest := sha256.Sum256(raw)
	return "sha256:" + hex.EncodeToString(digest[:])
}

func validSHA256(raw string) bool {
	raw = strings.TrimPrefix(strings.ToLower(strings.TrimSpace(raw)), "sha256:")
	if len(raw) != sha256.Size*2 {
		return false
	}
	_, err := hex.DecodeString(raw)
	return err == nil
}

func sameModelCanaryDigest(left, right string) bool {
	left = strings.TrimPrefix(strings.ToLower(strings.TrimSpace(left)), "sha256:")
	right = strings.TrimPrefix(strings.ToLower(strings.TrimSpace(right)), "sha256:")
	return left != "" && left == right
}

func errorDetail(err error) string {
	if err == nil {
		return ""
	}
	return err.Error()
}

type modelCanaryLsofProcess struct {
	PID     int
	Command string
}

// classifyModelCanaryLsofExit recognizes only BSD lsof's documented no-match shape:
// exit 1 with no stdout or stderr. Tool errors, diagnostics, and empty successful output
// remain observation failures rather than being inferred as an absent listener.
func classifyModelCanaryLsofExit(stdout, stderr []byte, exitCode int) (bool, error) {
	stdoutEmpty := len(bytes.TrimSpace(stdout)) == 0
	stderrEmpty := len(bytes.TrimSpace(stderr)) == 0
	switch {
	case exitCode == 0 && !stdoutEmpty:
		return true, nil
	case exitCode == 1 && stdoutEmpty && stderrEmpty:
		return false, nil
	case exitCode == 0:
		return false, errors.New("lsof succeeded with empty output")
	default:
		return false, fmt.Errorf("lsof observation failed with exit %d and stderr %q", exitCode, strings.TrimSpace(string(stderr)))
	}
}

func parseModelCanaryLsof(raw []byte, port int) (modelCanaryLsofProcess, error) {
	if port < 1 || port > 65535 {
		return modelCanaryLsofProcess{}, errors.New("invalid listener port")
	}
	text := strings.ReplaceAll(string(raw), "\r\n", "\n")
	var current modelCanaryLsofProcess
	matches := make(map[int]modelCanaryLsofProcess)
	for _, line := range strings.Split(text, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		switch line[0] {
		case 'p':
			pid, err := strconv.Atoi(strings.TrimSpace(line[1:]))
			if err != nil || pid <= 0 {
				return modelCanaryLsofProcess{}, fmt.Errorf("malformed lsof PID field %q", line)
			}
			current = modelCanaryLsofProcess{PID: pid}
		case 'c':
			if current.PID <= 0 || strings.TrimSpace(line[1:]) == "" {
				return modelCanaryLsofProcess{}, fmt.Errorf("malformed lsof command field %q", line)
			}
			current.Command = strings.TrimSpace(line[1:])
		case 'n':
			if current.PID <= 0 {
				return modelCanaryLsofProcess{}, errors.New("lsof listener field preceded its PID")
			}
			name := strings.TrimSpace(line[1:])
			if modelCanaryListenerNameHasPort(name, port) {
				matches[current.PID] = current
			}
		case 'f':
			if current.PID <= 0 || strings.TrimSpace(line[1:]) == "" {
				return modelCanaryLsofProcess{}, fmt.Errorf("malformed lsof descriptor field %q", line)
			}
		case 't':
			kind := strings.TrimSpace(line[1:])
			if current.PID <= 0 || (kind != "IPv4" && kind != "IPv6" && kind != "TCP") {
				return modelCanaryLsofProcess{}, fmt.Errorf("malformed lsof listener type field %q", line)
			}
		default:
			return modelCanaryLsofProcess{}, fmt.Errorf("unexpected lsof field %q; require -Fpcftn output", line)
		}
	}
	if len(matches) != 1 {
		return modelCanaryLsofProcess{}, fmt.Errorf("listener port %d resolved to %d distinct owners", port, len(matches))
	}
	for _, match := range matches {
		if match.Command == "" {
			return modelCanaryLsofProcess{}, errors.New("lsof listener owner has no command field")
		}
		return match, nil
	}
	panic("unreachable")
}

func modelCanaryListenerNameHasPort(name string, port int) bool {
	marker := ":" + strconv.Itoa(port)
	index := strings.LastIndex(name, marker)
	if index < 0 {
		return false
	}
	tail := name[index+len(marker):]
	return tail == "" || strings.HasPrefix(tail, " ") || strings.HasPrefix(tail, "->")
}

var modelCanaryPSIdentityPattern = regexp.MustCompile(`^\s*([0-9]+)\s+((?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+[ 0-9][0-9]?\s+[0-9]{2}:[0-9]{2}:[0-9]{2}\s+[0-9]{4})\s+(.+?)\s*$`)

func parseModelCanaryPS(raw []byte, location *time.Location) (modelCanaryProcessIdentity, string, error) {
	lines := nonEmptyModelCanaryLines(raw)
	if len(lines) != 1 {
		return modelCanaryProcessIdentity{}, "", fmt.Errorf("BSD ps identity output has %d non-empty lines, want 1", len(lines))
	}
	match := modelCanaryPSIdentityPattern.FindStringSubmatch(lines[0])
	if match == nil {
		return modelCanaryProcessIdentity{}, "", errors.New("malformed BSD ps identity output")
	}
	pid, _ := strconv.Atoi(match[1])
	if location == nil {
		location = time.Local
	}
	started, err := time.ParseInLocation("Mon Jan _2 15:04:05 2006", match[2], location)
	if err != nil || pid <= 0 || strings.TrimSpace(match[3]) == "" {
		return modelCanaryProcessIdentity{}, "", errors.New("malformed BSD ps PID/start/command identity")
	}
	command := strings.TrimSpace(match[3])
	return modelCanaryProcessIdentity{PID: pid, StartedAt: started.UTC().Format(time.RFC3339Nano), ArgvSHA256: digestBytes([]byte(command))}, command, nil
}

func parseModelCanaryRSS(raw []byte, expectedPID int) (int64, error) {
	lines := nonEmptyModelCanaryLines(raw)
	if len(lines) != 1 {
		return 0, fmt.Errorf("BSD ps RSS output has %d non-empty lines, want 1", len(lines))
	}
	fields := strings.Fields(lines[0])
	if len(fields) != 2 {
		return 0, errors.New("malformed BSD ps RSS output; require pid and rss_kib")
	}
	pid, pidErr := strconv.Atoi(fields[0])
	rssKiB, rssErr := strconv.ParseInt(fields[1], 10, 64)
	if pidErr != nil || rssErr != nil || pid != expectedPID || rssKiB < 0 || rssKiB > math.MaxInt64/1024 {
		return 0, errors.New("BSD ps RSS PID/value is malformed or does not match the exact candidate")
	}
	return rssKiB * 1024, nil
}

var modelCanaryFootprintPattern = regexp.MustCompile(`(?i)^\s*(?:physical footprint|phys_footprint)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)\s*([kmgtp]?i?b?)\s*$`)

func parseModelCanaryFootprint(raw []byte) (int64, error) {
	var values []int64
	for _, line := range nonEmptyModelCanaryLines(raw) {
		match := modelCanaryFootprintPattern.FindStringSubmatch(line)
		if match == nil {
			continue
		}
		value, err := parseModelCanaryBytes(match[1], match[2])
		if err != nil {
			return 0, fmt.Errorf("malformed footprint value: %w", err)
		}
		values = append(values, value)
	}
	if len(values) != 1 {
		return 0, fmt.Errorf("footprint output contains %d exact physical-footprint values, want 1", len(values))
	}
	return values[0], nil
}

var modelCanarySwapPattern = regexp.MustCompile(`(?i)\bused\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*([kmgtp]i?b?)\b`)

func parseModelCanarySwap(raw []byte) (int64, error) {
	matches := modelCanarySwapPattern.FindAllStringSubmatch(string(raw), -1)
	if len(matches) != 1 {
		return 0, fmt.Errorf("vm.swapusage output contains %d used values, want 1", len(matches))
	}
	// Darwin prints decimal numbers with a unit suffix (for example 2817.50M), but the
	// unit is binary: the committed live witness binds 2817.50M to 2,954,362,880 bytes.
	// Parse in Go so a GNU-awk-only match extension can never consume a launch again.
	value, err := parseModelCanaryBytes(matches[0][1], matches[0][2])
	if err != nil {
		return 0, fmt.Errorf("malformed vm.swapusage used value: %w", err)
	}
	return value, nil
}

var modelCanaryMemoryPressurePattern = regexp.MustCompile(`(?im)^\s*system-wide memory free percentage:\s*([0-9]+)%\s*$`)

func parseModelCanaryMemoryPressure(raw []byte) (int, error) {
	matches := modelCanaryMemoryPressurePattern.FindAllSubmatch(raw, -1)
	if len(matches) != 1 {
		return 0, fmt.Errorf("memory_pressure output contains %d system-free percentages, want 1", len(matches))
	}
	value, _ := strconv.Atoi(string(matches[0][1]))
	if value < 0 || value > 100 {
		return 0, errors.New("memory_pressure system-free percentage is outside [0,100]")
	}
	return value, nil
}

func parseModelCanaryMemorystatus(raw []byte) (int, error) {
	lines := nonEmptyModelCanaryLines(raw)
	if len(lines) != 1 {
		return 0, fmt.Errorf("kern.memorystatus_level output has %d non-empty lines, want 1", len(lines))
	}
	value, err := strconv.Atoi(strings.TrimSpace(lines[0]))
	if err != nil || value < 0 || value > 100 {
		return 0, errors.New("kern.memorystatus_level must be one integer in [0,100]")
	}
	return value, nil
}

func parseModelCanaryBytes(number, unit string) (int64, error) {
	value, err := strconv.ParseFloat(number, 64)
	if err != nil || math.IsNaN(value) || math.IsInf(value, 0) || value < 0 {
		return 0, errors.New("invalid non-negative decimal byte value")
	}
	unit = strings.ToUpper(strings.TrimSpace(unit))
	unit = strings.TrimSuffix(unit, "B")
	unit = strings.TrimSuffix(unit, "I")
	power := 0
	switch unit {
	case "":
	case "K":
		power = 1
	case "M":
		power = 2
	case "G":
		power = 3
	case "T":
		power = 4
	case "P":
		power = 5
	default:
		return 0, fmt.Errorf("unsupported byte unit %q", unit)
	}
	scaled := value * math.Pow(1024, float64(power))
	if scaled > math.MaxInt64 {
		return 0, errors.New("byte value overflows int64")
	}
	return int64(math.Round(scaled)), nil
}

func nonEmptyModelCanaryLines(raw []byte) []string {
	text := strings.ReplaceAll(string(raw), "\r\n", "\n")
	var lines []string
	for _, line := range strings.Split(text, "\n") {
		if strings.TrimSpace(line) != "" {
			lines = append(lines, line)
		}
	}
	return lines
}

func failModelCanaryConfig(stderr io.Writer, receiptPath string, started time.Time, configSHA string, err error) int {
	receipt := newModelCanaryReceipt(started, configSHA, runtime.GOOS, runtime.GOARCH)
	refuseModelCanary(&receipt, modelCanaryPhaseConfigValidated, modelCanaryReasonConfigInvalid, err.Error())
	finishModelCanaryReceipt(&receipt, time.Now().UTC())
	if writeErr := writeModelCanaryReceiptAtomic(receiptPath, receipt); writeErr != nil {
		fmt.Fprintf(stderr, "fak model canary-run: %v; write terminal receipt: %v\n", err, writeErr)
		return 1
	}
	fmt.Fprintf(stderr, "fak model canary-run: %v\n", err)
	return 1
}
