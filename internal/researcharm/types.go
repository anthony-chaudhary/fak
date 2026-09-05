package researcharm

import (
	"errors"
	"time"
)

var (
	// ErrArmConcurrencyExceeded is returned when an arm has reached its maximum in-flight request limit.
	ErrArmConcurrencyExceeded = errors.New("researcharm: arm concurrency limit reached")

	// ErrExclusiveLeaseHeld is returned when an exclusive lease is held by a different arm.
	ErrExclusiveLeaseHeld = errors.New("researcharm: exclusive lease held by another arm")

	// ErrLeaseRequired is returned when server requires an active lease to admit requests.
	ErrLeaseRequired = errors.New("researcharm: active lease required for arm")

	// ErrLeaseNotFound is returned when attempting to renew or release a non-existent lease.
	ErrLeaseNotFound = errors.New("researcharm: lease not found or expired")

	// ErrInvalidLeaseToken is returned when the supplied lease release token does not match.
	ErrInvalidLeaseToken = errors.New("researcharm: invalid lease token")

	// ErrArmNotFound is returned when an arm is not found in the registry.
	ErrArmNotFound = errors.New("researcharm: arm not found")
)

// Origin represents the caller attribution metadata derived from an HTTP request.
type Origin struct {
	ArmID         string `json:"arm_id"`
	ArmGroup      string `json:"arm_group"`
	CallerPID     int    `json:"caller_pid,omitempty"`
	CallerProcess string `json:"caller_process,omitempty"`
	RemoteAddr    string `json:"remote_addr,omitempty"`
	UserAgent     string `json:"user_agent,omitempty"`
	Explicit      bool   `json:"explicit"` // true if explicitly specified via header or query
}

// LeaseMode specifies whether a lease allows concurrent sharing or demands exclusive model execution.
type LeaseMode string

const (
	// LeaseModeShared allows multiple arms to run concurrently within their respective concurrency budgets.
	LeaseModeShared LeaseMode = "shared"

	// LeaseModeExclusive locks the server so ONLY this arm may issue requests (ideal for isolated latency benchmarks).
	LeaseModeExclusive LeaseMode = "exclusive"
)

// LeaseInfo describes an active lease held by an arm on the server.
type LeaseInfo struct {
	ID          string    `json:"id"`
	ArmID       string    `json:"arm_id"`
	HolderPID   int       `json:"holder_pid,omitempty"`
	Mode        LeaseMode `json:"mode"`
	Concurrency int       `json:"concurrency,omitempty"`
	Token       string    `json:"token,omitempty"` // Secret token required to release the lease
	CreatedAt   time.Time `json:"created_at"`
	ExpiresAt   time.Time `json:"expires_at"`
}

// ArmInfo provides an aggregated view of a research project arm's traffic and state.
type ArmInfo struct {
	ID             string     `json:"id"`
	Group          string     `json:"group"`
	DisplayName    string     `json:"display_name"`
	MaxConcurrency int        `json:"max_concurrency"` // 0 = unlimited
	ActiveRequests int        `json:"active_requests"`
	TotalRequests  int64      `json:"total_requests"`
	TotalTokens    int64      `json:"total_tokens"`
	ErrorCount     int64      `json:"error_count"`
	RecentPIDs     []int      `json:"recent_pids,omitempty"`
	LastSeen       time.Time  `json:"last_seen"`
	ActiveLease    *LeaseInfo `json:"active_lease,omitempty"`
}

// InflightRequest describes a single request currently being served.
type InflightRequest struct {
	RequestID     string    `json:"request_id"`
	ArmID         string    `json:"arm_id"`
	ArmGroup      string    `json:"arm_group"`
	CallerPID     int       `json:"caller_pid,omitempty"`
	CallerProcess string    `json:"caller_process,omitempty"`
	Endpoint      string    `json:"endpoint"`
	TraceID       string    `json:"trace_id,omitempty"`
	StartedAt     time.Time `json:"started_at"`
	RemoteAddr    string    `json:"remote_addr,omitempty"`
	UserAgent     string    `json:"user_agent,omitempty"`
}

// Snapshot is the comprehensive state of all arms, active traffic, and leases.
type Snapshot struct {
	Timestamp     time.Time         `json:"timestamp"`
	TotalInflight int               `json:"total_inflight"`
	TotalArms     int               `json:"total_arms"`
	Arms          []ArmInfo         `json:"arms"`
	Inflight      []InflightRequest `json:"inflight"`
	ActiveLeases  []LeaseInfo       `json:"active_leases"`
}

// LeaseRequest specifies parameters for acquiring or renewing a lease.
type LeaseRequest struct {
	ArmID       string        `json:"arm_id"`
	HolderPID   int           `json:"holder_pid,omitempty"`
	Mode        LeaseMode     `json:"mode"`
	Concurrency int           `json:"concurrency,omitempty"`
	TTL         time.Duration `json:"ttl,omitempty"` // defaults to 5m if <= 0
}

// LimitRequest specifies parameters for updating an arm's concurrency limit.
type LimitRequest struct {
	ArmID          string `json:"arm_id"`
	MaxConcurrency int    `json:"max_concurrency"`
}
