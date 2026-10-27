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
	OperationSystemdProfileReadback      = "systemd-profile-readback"
	OperationMaintainLeaseAcquire        = "maintain-lease-acquire"
	OperationMaintainLeaseRelease        = "maintain-lease-release"
	OperationInstalledBootStagePlan      = "installed-boot-stage-plan"
	OperationInstalledBootApply          = "installed-boot-apply"
	OperationInstalledBootReadback       = "installed-boot-readback"
	OperationInstalledBootRollback       = "installed-boot-rollback"
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

// TrustedReservation is a finite host-selected local maintenance lease scope.
// It is never populated from tool arguments. Grant.Paths includes the actual
// store, lock and journal paths and the narrow state directory conservatively
// covering random atomic-write temporaries. It carries no SSH authority.
type TrustedReservation struct {
	WorkspaceRoot      string   `json:"workspace_root"`
	NodeIDs            []string `json:"node_ids"`
	Kind               string   `json:"kind"`
	TTLSeconds         int64    `json:"ttl_seconds"`
	CatalogGeneration  string   `json:"catalog_generation"`
	ExpectedStoreEpoch uint64   `json:"expected_store_epoch"`
}

// SystemdProfileReadback carries fresh passive observations of the fixed
// profile. The complete graph remains ordinary result content for all floors.
type SystemdProfileReadback struct {
	Hostname  string             `json:"hostname"`
	Files     []SystemdFileState `json:"files"`
	UnitGraph string             `json:"unit_graph"`
}

type SystemdFileState struct {
	Path   string `json:"path"`
	SHA256 string `json:"sha256"`
	Mode   string `json:"mode"`
	UID    uint32 `json:"uid"`
	GID    uint32 `json:"gid"`
}

type Grant struct {
	ProfileID        string              `json:"profile_id"`
	OperationDigest  string              `json:"operation_digest"`
	ContextID        string              `json:"context_id"`
	Target           string              `json:"target"`
	HostKey          string              `json:"host_key"`
	Operation        string              `json:"operation"`
	PayloadSHA256    string              `json:"payload_sha256"`
	Holder           string              `json:"holder"`
	BootID           string              `json:"boot_id"`
	PriorStateSHA256 string              `json:"prior_state_sha256"`
	Paths            []string            `json:"paths"`
	PathPrefixes     []PathPrefix        `json:"path_prefixes"`
	Units            []string            `json:"units"`
	LeasePins        []LeasePin          `json:"lease_pins"`
	ExpiresAt        time.Time           `json:"expires_at"`
	Reservation      *TrustedReservation `json:"reservation,omitempty"`
	InstalledBoot    *InstalledBootScope `json:"installed_boot,omitempty"`
}

type Receipt struct {
	Schema            string                    `json:"schema"`
	GrantDigest       string                    `json:"grant_digest"`
	ProfileID         string                    `json:"profile_id"`
	OperationDigest   string                    `json:"operation_digest"`
	ContextID         string                    `json:"context_id"`
	Target            string                    `json:"target"`
	HostKey           string                    `json:"host_key"`
	Operation         string                    `json:"operation"`
	BootID            string                    `json:"boot_id"`
	StartedAt         time.Time                 `json:"started_at"`
	EndedAt           time.Time                 `json:"ended_at"`
	Outcome           string                    `json:"outcome"`
	Verified          bool                      `json:"verified"`
	BackupDisposition BackupDisposition         `json:"backup_disposition,omitempty"`
	ReservationLeases []LeasePin                `json:"reservation_leases,omitempty"`
	Readback          *SystemdProfileReadback   `json:"readback,omitempty"`
	InstalledBoot     *InstalledBootObservation `json:"installed_boot,omitempty"`
}

type Authority interface {
	Resolve(context.Context, Request) (Grant, error)
	Validate(context.Context, Grant) error
	Execute(context.Context, Grant) (Receipt, error)
}
