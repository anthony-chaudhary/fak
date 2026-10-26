// Package remotemaint defines the fail-closed authority contract for privileged
// remote maintenance. Callers select only a pre-reviewed profile and operation
// digest; every other field comes from the trusted host authority.
package remotemaint

import (
	"context"
	"errors"
	"time"
)

const (
	OperationSystemdProfileInstallReload = "systemd-profile-install-reload"
	ReceiptSchema                        = "fak.remote-maintenance.receipt.v1"
	ConfigSchema                         = "fak.remote-maintenance.config.v1"
	RequestEnvelopeSchema                = "fak.remote-maintenance.request.v1"
	ResponseEnvelopeSchema               = "fak.remote-maintenance.response.v1"
)

var (
	ErrUnavailable    = errors.New("remote maintenance unavailable")
	ErrInvalidRequest = errors.New("invalid remote maintenance request")
	ErrScope          = errors.New("remote maintenance scope mismatch")
	ErrExpired        = errors.New("remote maintenance authority expired")
	ErrChanged        = errors.New("remote maintenance authority changed")
	ErrAuthority      = errors.New("remote maintenance authority denied")
	ErrUnknownEffect  = errors.New("remote maintenance effect unknown")
	ErrRollbackFailed = errors.New("remote maintenance rollback failed")
)

const (
	OutcomeApplied        = "applied"
	OutcomeUnchanged      = "unchanged"
	OutcomeRolledBack     = "rolled_back"
	OutcomeUnknownEffect  = "unknown_effect"
	OutcomeRollbackFailed = "rollback_failed"
)

type BackupDisposition string

const (
	BackupUnknown     BackupDisposition = "unknown"
	BackupRetained    BackupDisposition = "retained"
	BackupRestored    BackupDisposition = "restored"
	BackupNotRequired BackupDisposition = "not_required"
)

type Request struct {
	ProfileID       string `json:"profile_id"`
	OperationDigest string `json:"operation_digest"`
}

type LeasePin struct {
	NodeID    string    `json:"node_id"`
	ID        string    `json:"id"`
	Epoch     uint64    `json:"epoch"`
	ExpiresAt time.Time `json:"expires_at"`
}

type PathPrefix struct {
	Directory      string `json:"directory"`
	FilenamePrefix string `json:"filename_prefix"`
}

type Grant struct {
	ProfileID        string       `json:"profile_id"`
	OperationDigest  string       `json:"operation_digest"`
	ContextID        string       `json:"context_id"`
	Target           string       `json:"target"`
	HostKey          string       `json:"host_key"`
	Operation        string       `json:"operation"`
	PayloadSHA256    string       `json:"payload_sha256"`
	Holder           string       `json:"holder"`
	BootID           string       `json:"boot_id"`
	PriorStateSHA256 string       `json:"prior_state_sha256"`
	Paths            []string     `json:"paths"`
	PathPrefixes     []PathPrefix `json:"path_prefixes"`
	Units            []string     `json:"units"`
	LeasePins        []LeasePin   `json:"lease_pins"`
	ExpiresAt        time.Time    `json:"expires_at"`
}

type Receipt struct {
	Schema            string            `json:"schema"`
	GrantDigest       string            `json:"grant_digest"`
	ProfileID         string            `json:"profile_id"`
	OperationDigest   string            `json:"operation_digest"`
	ContextID         string            `json:"context_id"`
	Target            string            `json:"target"`
	HostKey           string            `json:"host_key"`
	Operation         string            `json:"operation"`
	BootID            string            `json:"boot_id"`
	StartedAt         time.Time         `json:"started_at"`
	EndedAt           time.Time         `json:"ended_at"`
	Outcome           string            `json:"outcome"`
	Verified          bool              `json:"verified"`
	BackupDisposition BackupDisposition `json:"backup_disposition,omitempty"`
}

type Authority interface {
	Resolve(context.Context, Request) (Grant, error)
	Validate(context.Context, Grant) error
	Execute(context.Context, Grant) (Receipt, error)
}
