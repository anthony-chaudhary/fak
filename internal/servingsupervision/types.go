package servingsupervision

import (
	"errors"
	"strings"
	"time"
)

// ServingReceiptSchema is the canonical schema for serving supervision receipts.
const ServingReceiptSchema = "fak-serving-receipt/1"

// Engine constants ensure all serving execution is recorded as FAK-native.
const (
	EngineNative   = "native"
	EngineInKernel = "inkernel"
)

// ServingRole designates the architectural role of a component in the serving topology.
type ServingRole string

const (
	RoleController ServingRole = "controller"
	RoleProxy      ServingRole = "proxy"
	RoleRouter     ServingRole = "router"
	RoleReplica    ServingRole = "replica"
	RoleKVFabric   ServingRole = "kv_fabric"
)

// ServingPhase tracks the lifecycle phase of a serving component or domain.
type ServingPhase string

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
type ServingErrorKind string

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
type RestartScope string

const (
	ScopeNone             RestartScope = "none"
	ScopeLeafOnly         RestartScope = "leaf_only"
	ScopeDeploymentDomain RestartScope = "deployment_domain"
	ScopeQuarantine       RestartScope = "quarantine"
	ScopeRootFatal        RestartScope = "root_fatal"
)

// ServingReceipt is the immutable audit record of a serving lifecycle or recovery action.
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
type ServingDomainSpec struct {
	DomainID       string        `json:"domain_id"`
	ControllerID   string        `json:"controller_id"`
	DrainTimeout   time.Duration `json:"drain_timeout"`
	RestartBudget  int           `json:"restart_budget"`
	CoupledDomains []string      `json:"coupled_domains,omitempty"`
	Role           ServingRole   `json:"role,omitempty"`
}

// Sentinel errors for standard serving conditions.
var (
	ErrRequestApplication   = errors.New("request application error")
	ErrWorkerProcessFailure = errors.New("worker process failure")
	ErrFailedReadiness      = errors.New("failed readiness probe")
	ErrFailedLiveness       = errors.New("failed liveness probe")
	ErrModelStateCorruption = errors.New("model state corruption")
	ErrKVFabricFailure      = errors.New("kv fabric failure")
	ErrControllerFailure    = errors.New("controller failure")
	ErrTrafficWithdrawn     = errors.New("traffic withdrawn: domain is not ready")
	ErrBudgetExhausted      = errors.New("restart budget exhausted")
	ErrNoHealthyReplicas    = errors.New("no healthy replicas available")
)

// ClassifiedError wraps an underlying error with an explicit kind and restart scope.
type ClassifiedError struct {
	Kind  ServingErrorKind
	Scope RestartScope
	Err   error
}

func (e *ClassifiedError) Error() string {
	if e.Err != nil {
		return e.Err.Error()
	}
	return string(e.Kind)
}

func (e *ClassifiedError) Unwrap() error {
	return e.Err
}

// WrapClassifiedError annotates an error with a typed ServingErrorKind and RestartScope.
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
