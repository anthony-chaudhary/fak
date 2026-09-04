package servingsupervision

import (
	"errors"
	"strings"
	"time"
)

// ServingReceiptSchema is the canonical schema for serving supervision receipts.
// Invariant: Emitted receipts must strictly match this schema string.
// Guard: Consumers fail closed if an unexpected receipt schema is encountered.
const ServingReceiptSchema = "fak-serving-receipt/1"

// Engine constants ensure all serving execution is recorded as FAK-native.
// Invariant: Serving execution must report EngineNative or EngineInKernel.
// Guard: Non-native backends or silent fallback engines are strictly forbidden.
const (
	EngineNative   = "native"
	EngineInKernel = "inkernel"
)

// ServingRole designates the architectural role of a component in the serving topology.
// Invariant: Component roles determine failure domain boundaries and recovery rules.
type ServingRole string

// Architectural roles recognized by the serving supervisor.
// Invariant: Each role has distinct restart scoping and isolation guarantees.
const (
	RoleController ServingRole = "controller"
	RoleProxy      ServingRole = "proxy"
	RoleRouter     ServingRole = "router"
	RoleReplica    ServingRole = "replica"
	RoleKVFabric   ServingRole = "kv_fabric"
)

// ServingPhase tracks the lifecycle phase of a serving component or domain.
// Invariant: Phase transitions follow deterministic state machine rules.
// Guard: Traffic is only admitted when phase is PhaseReady.
type ServingPhase string

// Lifecycle phases for serving components and failure domains.
// Invariant: Components start in PhaseStarting, transition to PhaseReady on passing readiness,
// and enter PhaseDraining, PhaseRecovering, PhaseQuarantined, PhaseStopped, or PhaseFailed upon conditions.
// Guard: Inactive, draining, recovering, quarantined, or failed phases reject traffic immediately.
const (
	PhaseStarting    ServingPhase = "starting"
	PhaseReady       ServingPhase = "ready"
	PhaseDraining    ServingPhase = "draining"
	PhaseRecovering  ServingPhase = "recovering"
	PhaseQuarantined ServingPhase = "quarantined"
	PhaseStopped     ServingPhase = "stopped"
	PhaseFailed      ServingPhase = "failed"
)

// ServingErrorKind classifies failures to determine the minimum necessary restart scope.
// Invariant: Error kind dictates the failure blast radius and recovery action.
// Guard: Unclassified or unrecognized errors fail closed to ErrorKindWorkerProcessFailure.
type ServingErrorKind string

// Classified error kinds for serving supervision.
// Invariant: Application errors are isolated from process and infrastructure errors.
const (
	ErrorKindRequestApplication   ServingErrorKind = "request_application"
	ErrorKindWorkerProcessFailure ServingErrorKind = "worker_process_failure"
	ErrorKindFailedReadiness      ServingErrorKind = "failed_readiness"
	ErrorKindFailedLiveness       ServingErrorKind = "failed_liveness"
	ErrorKindModelStateCorruption ServingErrorKind = "model_state_corruption"
	ErrorKindKVFabricFailure      ServingErrorKind = "kv_fabric_failure"
	ErrorKindControllerFailure    ServingErrorKind = "controller_failure"
)

// RestartScope defines the blast radius of a recovery action.
// Invariant: Scopes form a hierarchy from ScopeNone up to ScopeRootFatal.
// Guard: Recovery actions are restricted to the lowest-reasonable scope.
type RestartScope string

// Restart blast radius scopes.
// Invariant: ScopeNone affects no processes; ScopeLeafOnly restarts only the failing worker;
// ScopeDeploymentDomain restarts the coupled domain; ScopeQuarantine locks the component out;
// ScopeRootFatal stops the cluster root.
const (
	ScopeNone             RestartScope = "none"
	ScopeLeafOnly         RestartScope = "leaf_only"
	ScopeDeploymentDomain RestartScope = "deployment_domain"
	ScopeQuarantine       RestartScope = "quarantine"
	ScopeRootFatal        RestartScope = "root_fatal"
)

// ServingReceipt is the immutable audit record of a serving lifecycle or recovery action.
// Invariant: Receipts capture generation changes, drain metrics, and engine provenance.
// Guard: FallbackUsed must be false for valid FAK-native execution.
type ServingReceipt struct {
	Schema          string           `json:"schema"`
	Timestamp       time.Time        `json:"timestamp"`
	DomainID        string           `json:"domain_id"`
	MemberID        string           `json:"member_id"`
	Role            ServingRole      `json:"role"`
	ObservedGen     uint64           `json:"observed_gen"`
	NextGen         uint64           `json:"next_gen"`
	ErrorKind       ServingErrorKind `json:"error_kind,omitempty"`
	RestartScope    RestartScope     `json:"restart_scope"`
	InflightDrained int              `json:"inflight_drained"`
	InflightLost    int              `json:"inflight_lost"`
	DrainDuration   time.Duration    `json:"drain_duration"`
	Quarantined     bool             `json:"quarantined"`
	Engine          string           `json:"engine"`
	FallbackUsed    bool             `json:"fallback_used"`
}

// ServingDomainSpec declares configuration and boundary constraints for a failure domain.
// Invariant: DomainID must uniquely identify the failure domain across the topology.
// Guard: DrainTimeout and RestartBudget must be non-negative.
type ServingDomainSpec struct {
	DomainID       string        `json:"domain_id"`
	ControllerID   string        `json:"controller_id"`
	DrainTimeout   time.Duration `json:"drain_timeout"`
	RestartBudget  int           `json:"restart_budget"`
	CoupledDomains []string      `json:"coupled_domains,omitempty"`
	Role           ServingRole   `json:"role,omitempty"`
}

// Sentinel errors for standard serving conditions.
// Invariant: Standardized error sentinels enable deterministic error classification.
// Guard: Specific sentinels map directly to deterministic ServingErrorKind and RestartScope values.
var (
	// ErrRequestApplication signals a client or request payload validation failure.
	// Invariant: Application errors must never trigger replica restart (ScopeNone).
	ErrRequestApplication = errors.New("request application error")

	// ErrWorkerProcessFailure signals an unexpected worker process crash or exit.
	// Guard: Triggers leaf-only restart bounded by the replica's restart budget.
	ErrWorkerProcessFailure = errors.New("worker process failure")

	// ErrFailedReadiness signals that a replica's readiness check probe failed.
	// Guard: Prevents unready replicas from entering PhaseReady and receiving traffic.
	ErrFailedReadiness = errors.New("failed readiness probe")

	// ErrFailedLiveness signals that a replica's liveness heartbeat or watchdog timed out.
	// Guard: Triggers leaf restart to recover potentially hung or deadlocked workers.
	ErrFailedLiveness = errors.New("failed liveness probe")

	// ErrModelStateCorruption signals detected weight corruption or NaN/Inf tensor values.
	// Guard: Expands restart blast radius to ScopeDeploymentDomain to purge corrupted state.
	ErrModelStateCorruption = errors.New("model state corruption")

	// ErrKVFabricFailure signals a failure or disconnect in the shared KV cache fabric.
	// Guard: Triggers deployment-domain recovery to re-establish coherent cache state.
	ErrKVFabricFailure = errors.New("kv fabric failure")

	// ErrControllerFailure signals a failure in the serving controller process.
	// Guard: Allows root supervisor to restore controller without tearing down healthy workers.
	ErrControllerFailure = errors.New("controller failure")

	// ErrTrafficWithdrawn signals that traffic is rejected because the target is not in PhaseReady.
	// Guard: Fail-closed boundary protecting draining, recovering, or quarantined domains.
	ErrTrafficWithdrawn = errors.New("traffic withdrawn: domain is not ready")

	// ErrBudgetExhausted signals that a component has exceeded its allowed restart budget.
	// Guard: Transitions component into PhaseQuarantined to halt cascading restart loops.
	ErrBudgetExhausted = errors.New("restart budget exhausted")

	// ErrNoHealthyReplicas signals that no replicas in the pool are currently ready to receive traffic.
	// Guard: Returned by ingress routing when all replicas are unready, draining, or quarantined.
	ErrNoHealthyReplicas = errors.New("no healthy replicas available")
)

// ClassifiedError wraps an underlying error with an explicit kind and restart scope.
// Invariant: Preserves underlying error causality while attaching supervision classification.
// Guard: Unwrap returns the underlying error for standard errors.Is / errors.As unwrapping.
type ClassifiedError struct {
	Kind  ServingErrorKind
	Scope RestartScope
	Err   error
}

// Error returns the string representation of the classified error.
// Invariant: Returns underlying error string if present, otherwise returns string representation of Kind.
func (e *ClassifiedError) Error() string {
	if e.Err != nil {
		return e.Err.Error()
	}
	return string(e.Kind)
}

// Unwrap returns the underlying error.
// Invariant: Enables standard Go error unwrapping chains.
func (e *ClassifiedError) Unwrap() error {
	return e.Err
}

// WrapClassifiedError annotates an error with a typed ServingErrorKind and RestartScope.
// Invariant: Returns nil if err is nil.
// Guard: Attaches supervision metadata without masking the underlying error chain.
func WrapClassifiedError(kind ServingErrorKind, scope RestartScope, err error) error {
	if err == nil {
		return nil
	}
	return &ClassifiedError{
		Kind:  kind,
		Scope: scope,
		Err:   err,
	}
}

// ClassifyError categorizes an error into its ServingErrorKind and lowest-reasonable RestartScope.
// Invariant: Maps errors deterministically to error kind and restart scope.
// Guard: Unknown errors fail closed to ErrorKindWorkerProcessFailure with ScopeLeafOnly; nil maps to ScopeNone.
func ClassifyError(err error) (ServingErrorKind, RestartScope) {
	if err == nil {
		return "", ScopeNone
	}

	var classified *ClassifiedError
	if errors.As(err, &classified) {
		return classified.Kind, classified.Scope
	}

	if errors.Is(err, ErrRequestApplication) {
		return ErrorKindRequestApplication, ScopeNone
	}
	if errors.Is(err, ErrModelStateCorruption) {
		return ErrorKindModelStateCorruption, ScopeDeploymentDomain
	}
	if errors.Is(err, ErrKVFabricFailure) {
		return ErrorKindKVFabricFailure, ScopeDeploymentDomain
	}
	if errors.Is(err, ErrFailedReadiness) {
		return ErrorKindFailedReadiness, ScopeLeafOnly
	}
	if errors.Is(err, ErrFailedLiveness) {
		return ErrorKindFailedLiveness, ScopeLeafOnly
	}
	if errors.Is(err, ErrControllerFailure) {
		return ErrorKindControllerFailure, ScopeLeafOnly
	}
	if errors.Is(err, ErrWorkerProcessFailure) {
		return ErrorKindWorkerProcessFailure, ScopeLeafOnly
	}

	msg := strings.ToLower(err.Error())
	switch {
	case strings.Contains(msg, "model state corruption") ||
		strings.Contains(msg, "model_state_corruption") ||
		strings.Contains(msg, "corrupt state") ||
		strings.Contains(msg, "corrupted weights") ||
		strings.Contains(msg, "nan tensor") ||
		strings.Contains(msg, "tensor corruption"):
		return ErrorKindModelStateCorruption, ScopeDeploymentDomain

	case strings.Contains(msg, "kv fabric") ||
		strings.Contains(msg, "kv_fabric") ||
		strings.Contains(msg, "fabric failure"):
		return ErrorKindKVFabricFailure, ScopeDeploymentDomain

	case strings.Contains(msg, "readiness probe") ||
		strings.Contains(msg, "failed readiness") ||
		strings.Contains(msg, "unready"):
		return ErrorKindFailedReadiness, ScopeLeafOnly

	case strings.Contains(msg, "liveness probe") ||
		strings.Contains(msg, "failed liveness") ||
		strings.Contains(msg, "deadlock") ||
		strings.Contains(msg, "heartbeat missed"):
		return ErrorKindFailedLiveness, ScopeLeafOnly

	case strings.Contains(msg, "request application") ||
		strings.Contains(msg, "request error") ||
		strings.Contains(msg, "bad request") ||
		strings.Contains(msg, "invalid argument") ||
		strings.Contains(msg, "validation failed") ||
		strings.Contains(msg, "invalid payload") ||
		strings.Contains(msg, "context length") ||
		strings.Contains(msg, "prompt error") ||
		strings.Contains(msg, "user error"):
		return ErrorKindRequestApplication, ScopeNone

	case strings.Contains(msg, "controller failure") ||
		strings.Contains(msg, "controller crash"):
		return ErrorKindControllerFailure, ScopeLeafOnly

	default:
		return ErrorKindWorkerProcessFailure, ScopeLeafOnly
	}
}
