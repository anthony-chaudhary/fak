// Package service is the public deployable-lifecycle facade for service
// render / control / readback / restore: one Manager per (scope, name) pair
// that snapshots the prior registration, applies a new one, and — on any
// failure after the snapshot — restores the prior registration AND the prior
// running state.
//
// It is a facade, not a second implementation. The platform primitives it
// drives already exist in this module: systemctl control and typed read-back
// in internal/servicewatchdog, launchd projection and print parsing in
// internal/systemservice, the portable desired-state contract in
// internal/servicespec, and the Windows SCM semantics (snapshot → stop →
// sha256-named staged executable → config update → recovery ladder → start →
// read back, restore on failure) established by cmd/fak's `fak service`
// install path. This package owns its own narrow command and SCM boundaries so
// every driver is fake-testable on every GOOS: the systemd, launchd, and
// scheduled-task drivers run against an injected Command on any host, and the
// SCM driver runs against an injected SCMBoundary on any host.
//
// Two scopes: ScopeSystem (SCM / systemd system unit / LaunchDaemon) and
// ScopeUser (Scheduled Task / systemctl --user / LaunchAgent). Document
// publishing (unit file, plist, task XML) stages a temp file beside the target
// and activates it through the L3 swap facade (pkg/deploykit/swap), so a
// concurrent reader never sees a partial document.
package service

import (
	"errors"
	"time"
)

// Start-type values, owned here as plain ints so the package stays testable
// without importing golang.org/x/sys/windows/svc. Verified against x/sys
// v0.49.0 (svc.StartAutomatic/StartManual/StartDisabled).
const (
	StartAutomatic = 2
	StartManual    = 3
	StartDisabled  = 4
)

// ErrorNormal is the SCM error-control level fak services install with
// (x/sys v0.49.0 svc.ErrorNormal).
const ErrorNormal = 1

// Service state values (x/sys v0.49.0 svc.Stopped/StartPending/StopPending/Running).
const (
	StateStopped      = 1
	StateStartPending = 2
	StateStopPending  = 3
	StateRunning      = 4
)

// ActionRestart is the SCM recovery action kind "restart"
// (x/sys v0.49.0 windows.SC_ACTION_RESTART, surfaced as mgr.ServiceRestart).
const ActionRestart = 1

// CmdStop is the SCM control command "stop"
// (x/sys v0.49.0 windows.SERVICE_CONTROL_STOP, surfaced as svc.Stop).
const CmdStop = 1

// SidTypeRestricted is SERVICE_SID_TYPE_RESTRICTED: the service SID carries no
// user-group privileges (x/sys v0.49.0 windows.SERVICE_SID_TYPE_RESTRICTED).
const SidTypeRestricted = 3

// ErrServiceDoesNotExist wraps the SCM not-found refusal (ERROR_SERVICE_DOES_NOT_EXIST)
// so callers can distinguish "no such service" from transport errors on every GOOS.
var ErrServiceDoesNotExist = errors.New("deploykit/service: service does not exist")

// SCMConfig is the plain-value mirror of the Windows SCM service configuration
// (golang.org/x/sys/windows/svc/mgr.Config). The adapter in scm_windows.go
// converts to/from the x/sys type; the driver logic only ever sees these.
type SCMConfig struct {
	StartType        int
	ErrorControl     int
	BinaryPathName   string
	DisplayName      string
	Description      string
	ServiceStartName string
	SidType          int
}

// RecoveryAction is one rung of the SCM failure-action ladder.
type RecoveryAction struct {
	Type  int
	Delay time.Duration
}

// Status is the plain-value mirror of the SCM service status query result.
type Status struct {
	State int
	PID   uint32
}

// SCMService is the narrow service-handle boundary the Windows system driver
// drives. Implementations must be safe for sequential use from one goroutine.
type SCMService interface {
	Close() error
	Config() (SCMConfig, error)
	UpdateConfig(cfg SCMConfig) error
	RecoveryActions() ([]RecoveryAction, error)
	ResetPeriod() (uint32, error)
	SetRecoveryActions(actions []RecoveryAction, resetPeriodSec uint32) error
	RecoveryActionsOnNonCrashFailures() (bool, error)
	SetRecoveryActionsOnNonCrashFailures(on bool) error
	Query() (Status, error)
	Control(cmd int) (Status, error)
	Start(args ...string) error
	Delete() error
}

// SCMBoundary is the manager-level boundary: open or create a service by name
// and disconnect when done. OpenService must return an error wrapping
// ErrServiceDoesNotExist when the name is unknown.
type SCMBoundary interface {
	Disconnect() error
	OpenService(name string) (SCMService, error)
	CreateService(name, path string, cfg SCMConfig, args ...string) (SCMService, error)
}
