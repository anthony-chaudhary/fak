// Package scmbridge completes the Windows supervision plane (#4756, parent
// #4748, foundation #4749): ONE desired-state / read-back contract shared by
// the SCM LocalService control plane and the Scheduled-Task S4U /
// InteractiveToken recovery bridge, so machine services, interactive agents,
// boot recovery, and crash recovery cannot diverge or double-launch.
//
// The placement rule the leaf encodes (and refuses violations of):
//
//   - Machine-owned always-on services belong to SCM, run as the least
//     privilege NT AUTHORITY\LocalService principal, start at boot, and carry
//     SCM recovery actions derived from the ONE servicespec.RestartPolicy.
//   - SCM cannot enter the interactive desktop. Scheduled Tasks are
//     reserved for exactly two session-shaped roles: the S4U watchdog (runs
//     unattended without a stored password, boot-triggered) and the
//     InteractiveToken broker (enters the logged-on desktop, logon-triggered).
//   - A recurring job never becomes an SCM service; the task schedule owns
//     recurrence (#749 semantics from servicespec).
//
// Everything here is pure and deterministic: no SCM or Task Scheduler calls,
// logical clocks only. Platform read-back (cmd/fak `service bridge --live`)
// produces the Observed document; Reconcile diffs it against the projection
// of a fak.service.v1 spec across the axes the issue names — installed binary
// provenance, service/task principal, action, trigger, recovery, and
// desired-stop state.
package scmbridge

import (
	"errors"
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/servicespec"
)

// ProjectionSchemaV1 names the versioned Windows projection document.
const ProjectionSchemaV1 = "fak.service.windows.v1"

// Role is the closed vocabulary of Windows launch roles. Exactly one of the
// three launchers may own a workload at a time (see LaunchFence).
type Role string

const (
	// RoleMachine is a machine-owned always-on service: SCM, LocalService.
	RoleMachine Role = "machine-service"
	// RoleWatchdog is the unattended S4U scheduled task that supervises and
	// resurrects work without a desktop and without a stored password.
	RoleWatchdog Role = "watchdog-task"
	// RoleBroker is the InteractiveToken scheduled task that bridges into the
	// logged-on desktop where SCM cannot go.
	RoleBroker Role = "session-broker"
)

// AllRoles enumerates the vocabulary (stable order).
var AllRoles = []Role{RoleMachine, RoleWatchdog, RoleBroker}

// Manager is which native supervisor owns the unit.
type Manager string

const (
	ManagerSCM  Manager = "windows-scm"
	ManagerTask Manager = "windows-scheduled-task"
)

// Logon types for scheduled-task roles (SCM services carry none — their
// principal is the service account itself).
const (
	LogonS4U         = "S4U"
	LogonInteractive = "InteractiveToken"
)

// MachinePrincipal is the least-privilege account every machine-owned SCM
// service runs under (matches the landed `fak service install` projection).
const MachinePrincipal = "NT AUTHORITY\\LocalService"

// RecoveryStep is one SCM failure action: restart after DelayMS.
type RecoveryStep struct {
	DelayMS int64 `json:"delay_ms"`
}

// Projection is the desired Windows form of one workload in one role — the
// document both the installer and the reconciler read, derived from the ONE
// portable spec so the two planes cannot drift apart.
type Projection struct {
	Schema   string               `json:"schema"`
	Identity servicespec.Identity `json:"identity"`
	Role     Role                 `json:"role"`
	Manager  Manager              `json:"manager"`
	UnitName string               `json:"unit_name"`
	// Principal is the service account (machine) or task run-as account.
	Principal string `json:"principal"`
	// LogonType is S4U (watchdog) or InteractiveToken (broker); empty for SCM.
	LogonType string   `json:"logon_type,omitempty"`
	Command   []string `json:"command"`
	// BinarySHA256 pins the installed binary provenance (hex, optional).
	BinarySHA256 string `json:"binary_sha256,omitempty"`
	// StartOnBoot / StartOnLogon are the trigger axis: SCM auto-start or task
	// boot trigger versus the broker's logon trigger.
	StartOnBoot  bool `json:"start_on_boot"`
	StartOnLogon bool `json:"start_on_logon"`
	// Recovery is the SCM failure-action ladder derived from
	// servicespec.RestartPolicy (machine role only).
	Recovery []RecoveryStep `json:"recovery,omitempty"`
	// RecoveryResetSec mirrors SCM's failure-count reset period, derived from
	// the spec's stable-run reset.
	RecoveryResetSec int64 `json:"recovery_reset_sec,omitempty"`
	// DesiredStopped projects desired=stopped: unit present but start-disabled.
	DesiredStopped bool `json:"desired_stopped"`
}

// ProjectInput carries the environment-specific facts the portable spec does
// not encode.
type ProjectInput struct {
	Role Role
	// BinarySHA256 is the expected installed-binary digest (hex, optional).
	BinarySHA256 string
	// TaskPrincipal is the account for watchdog/broker tasks (required there;
	// ignored for the machine role, which is always MachinePrincipal).
	TaskPrincipal string
}

// Placement refusals — the closed reasons Project fails.
var (
	ErrUnknownRole = errors.New("scmbridge: unknown role")
	// ErrJobOnSCM: a recurring job never becomes an SCM service — the task
	// schedule owns recurrence.
	ErrJobOnSCM = errors.New("scmbridge: a recurring job cannot be a machine SCM service; use a scheduled-task role")
	// ErrTaskNeedsPrincipal: S4U and InteractiveToken tasks run as a named
	// account; there is no anonymous desktop.
	ErrTaskNeedsPrincipal = errors.New("scmbridge: watchdog/broker roles require a task principal")
)

// Project derives the desired Windows form of the spec for one role. The
// placement rule is enforced here, once: machine work on SCM under
// LocalService, desktop-shaped work on Scheduled Tasks only.
func Project(spec *servicespec.Spec, in ProjectInput) (*Projection, error) {
	if spec == nil {
		return nil, errors.New("scmbridge: nil spec")
	}
	s := *spec
	s.Normalize()
	if err := s.Validate(); err != nil {
		return nil, err
	}
	p := &Projection{
		Schema:         ProjectionSchemaV1,
		Identity:       s.Identity,
		Role:           in.Role,
		UnitName:       s.Identity.Workload,
		Command:        append([]string(nil), s.Command...),
		BinarySHA256:   strings.ToLower(in.BinarySHA256),
		DesiredStopped: s.Desired == servicespec.DesiredStopped,
	}
	switch in.Role {
	case RoleMachine:
		if s.Kind == servicespec.KindJob {
			return nil, ErrJobOnSCM
		}
		p.Manager = ManagerSCM
		p.Principal = MachinePrincipal
		p.StartOnBoot = !p.DesiredStopped
		p.Recovery = recoveryLadder(s.Restart)
		p.RecoveryResetSec = s.Restart.StableRunResetMS / 1000
	case RoleWatchdog:
		if in.TaskPrincipal == "" {
			return nil, ErrTaskNeedsPrincipal
		}
		p.Manager = ManagerTask
		p.Principal = in.TaskPrincipal
		p.LogonType = LogonS4U
		p.StartOnBoot = !p.DesiredStopped
	case RoleBroker:
		if in.TaskPrincipal == "" {
			return nil, ErrTaskNeedsPrincipal
		}
		p.Manager = ManagerTask
		p.Principal = in.TaskPrincipal
		p.LogonType = LogonInteractive
		p.StartOnLogon = !p.DesiredStopped
	default:
		return nil, fmt.Errorf("%w: %q", ErrUnknownRole, in.Role)
	}
	return p, nil
}

// recoveryLadder maps the portable RestartPolicy onto SCM's three failure
// actions: the first three backoff delays of the ONE restart contract.
func recoveryLadder(rp servicespec.RestartPolicy) []RecoveryStep {
	out := make([]RecoveryStep, 0, 3)
	for attempt := 0; attempt < 3; attempt++ {
		d := rp.Decide(servicespec.RestartInput{
			Kind:    servicespec.KindService,
			Desired: servicespec.DesiredRunning,
			Class:   servicespec.ExitCrash,
			Attempt: attempt,
		})
		out = append(out, RecoveryStep{DelayMS: d.DelayMS})
	}
	return out
}

// Observed is the authoritative native read-back document: what SCM or the
// Task Scheduler actually has installed, plus live status. cmd/fak captures
// it (`service bridge --live` for SCM); operators can also feed an exported
// document. Zero-valued optional fields mean "not read", not "empty".
type Observed struct {
	Present          bool           `json:"present"`
	Manager          Manager        `json:"manager,omitempty"`
	UnitName         string         `json:"unit_name,omitempty"`
	Principal        string         `json:"principal,omitempty"`
	LogonType        string         `json:"logon_type,omitempty"`
	Command          []string       `json:"command,omitempty"`
	BinarySHA256     string         `json:"binary_sha256,omitempty"`
	StartOnBoot      bool           `json:"start_on_boot,omitempty"`
	StartOnLogon     bool           `json:"start_on_logon,omitempty"`
	Recovery         []RecoveryStep `json:"recovery,omitempty"`
	RecoveryResetSec int64          `json:"recovery_reset_sec,omitempty"`
	// StartDisabled is the native desired-stop read-back: SCM start type
	// disabled, or the task disabled.
	StartDisabled bool `json:"start_disabled,omitempty"`
	// Status is the native state string (SCM: running/stopped/start-pending/…;
	// tasks: Running/Ready/Queued/Disabled).
	Status string `json:"status,omitempty"`
	PID    int    `json:"pid,omitempty"`
}

// Divergence axes — the closed vocabulary of ways the native install can
// drift from the desired projection.
const (
	AxisAbsent      = "absent"
	AxisManager     = "manager"
	AxisUnit        = "unit"
	AxisPrincipal   = "principal"
	AxisLogonType   = "logon-type"
	AxisAction      = "action"
	AxisProvenance  = "binary-provenance"
	AxisTrigger     = "trigger"
	AxisRecovery    = "recovery"
	AxisDesiredStop = "desired-stop"
)

// Divergence is one axis where observed != desired.
type Divergence struct {
	Axis string `json:"axis"`
	Want string `json:"want"`
	Got  string `json:"got"`
}

// Report is the reconcile verdict.
type Report struct {
	InSync      bool         `json:"in_sync"`
	Divergences []Divergence `json:"divergences,omitempty"`
}

// Reconcile diffs the authoritative read-back against the desired projection
// across every axis the contract names. Optional observed fields that were
// not read (zero-valued digest, empty recovery) are skipped, never guessed.
func Reconcile(want *Projection, got Observed) Report {
	var out []Divergence
	add := func(axis, w, g string) { out = append(out, Divergence{Axis: axis, Want: w, Got: g}) }
	if !got.Present {
		return Report{Divergences: []Divergence{{Axis: AxisAbsent, Want: string(want.Manager) + " " + want.UnitName, Got: "not installed"}}}
	}
	if got.Manager != "" && got.Manager != want.Manager {
		add(AxisManager, string(want.Manager), string(got.Manager))
	}
	if got.UnitName != "" && got.UnitName != want.UnitName {
		add(AxisUnit, want.UnitName, got.UnitName)
	}
	if got.Principal != "" && !strings.EqualFold(normalizePrincipal(got.Principal), normalizePrincipal(want.Principal)) {
		add(AxisPrincipal, want.Principal, got.Principal)
	}
	if want.LogonType != "" && got.LogonType != "" && !strings.EqualFold(got.LogonType, want.LogonType) {
		add(AxisLogonType, want.LogonType, got.LogonType)
	}
	if len(got.Command) > 0 && strings.Join(got.Command, " ") != strings.Join(want.Command, " ") {
		add(AxisAction, strings.Join(want.Command, " "), strings.Join(got.Command, " "))
	}
	if want.BinarySHA256 != "" && got.BinarySHA256 != "" && !strings.EqualFold(got.BinarySHA256, want.BinarySHA256) {
		add(AxisProvenance, want.BinarySHA256, got.BinarySHA256)
	}
	if got.StartOnBoot != want.StartOnBoot || got.StartOnLogon != want.StartOnLogon {
		add(AxisTrigger,
			fmt.Sprintf("boot=%v logon=%v", want.StartOnBoot, want.StartOnLogon),
			fmt.Sprintf("boot=%v logon=%v", got.StartOnBoot, got.StartOnLogon))
	}
	if len(want.Recovery) > 0 && len(got.Recovery) > 0 && !sameRecovery(want, got) {
		add(AxisRecovery,
			fmt.Sprintf("%v reset=%ds", delays(want.Recovery), want.RecoveryResetSec),
			fmt.Sprintf("%v reset=%ds", delays(got.Recovery), got.RecoveryResetSec))
	}
	if got.StartDisabled != want.DesiredStopped {
		add(AxisDesiredStop,
			fmt.Sprintf("start_disabled=%v", want.DesiredStopped),
			fmt.Sprintf("start_disabled=%v", got.StartDisabled))
	}
	return Report{InSync: len(out) == 0, Divergences: out}
}

// normalizePrincipal folds the equivalent spellings of well-known service
// accounts ("LocalService" == "NT AUTHORITY\LocalService").
func normalizePrincipal(p string) string {
	p = strings.TrimSpace(p)
	if i := strings.LastIndexByte(p, '\\'); i >= 0 && strings.EqualFold(p[:i], "NT AUTHORITY") {
		return p[i+1:]
	}
	return p
}

func sameRecovery(want *Projection, got Observed) bool {
	if len(want.Recovery) != len(got.Recovery) {
		return false
	}
	for i := range want.Recovery {
		if want.Recovery[i] != got.Recovery[i] {
			return false
		}
	}
	// A zero got reset means "not read" — skip, never guess.
	return got.RecoveryResetSec == 0 || got.RecoveryResetSec == want.RecoveryResetSec
}

func delays(steps []RecoveryStep) []int64 {
	out := make([]int64, len(steps))
	for i, s := range steps {
		out[i] = s.DelayMS
	}
	return out
}

// PhaseFromSCMState maps the authoritative SCM service state onto the
// portable observed axis. Running with a live PID is ready; a running state
// without a process is still starting (SCM has admitted it but the image is
// not up).
func PhaseFromSCMState(state string, pid int) servicespec.Phase {
	switch strings.ToLower(strings.TrimSpace(state)) {
	case "running":
		if pid > 0 {
			return servicespec.PhaseReady
		}
		return servicespec.PhaseStarting
	case "start-pending", "continue-pending":
		return servicespec.PhaseStarting
	case "pause-pending", "paused":
		return servicespec.PhaseDegraded
	case "stop-pending", "stopped":
		return servicespec.PhaseStopped
	default:
		return servicespec.PhaseUnknown
	}
}

// PhaseFromTaskState maps Task Scheduler states. "Ready" is an ARMED task
// waiting on its trigger — for the logon-triggered broker that is the desired
// between-logons shape, so it reads as an intentional stop, not a failure.
func PhaseFromTaskState(state string) servicespec.Phase {
	switch strings.ToLower(strings.TrimSpace(state)) {
	case "running":
		return servicespec.PhaseReady
	case "queued":
		return servicespec.PhaseStarting
	case "ready", "disabled":
		return servicespec.PhaseStopped
	default:
		return servicespec.PhaseUnknown
	}
}

// ExecutableFromCommandLine extracts the installed image path from a native
// service command line so its provenance can be digested: a quoted path is
// taken verbatim; an unquoted path with spaces is resolved by extending
// through the spaces until exists() accepts (the canonical resolution for the
// classic unquoted-service-path shape). exists never being satisfied returns
// the first space-delimited token.
func ExecutableFromCommandLine(cmdline string, exists func(string) bool) string {
	cmdline = strings.TrimSpace(cmdline)
	if cmdline == "" {
		return ""
	}
	if cmdline[0] == '"' {
		if i := strings.IndexByte(cmdline[1:], '"'); i >= 0 {
			return cmdline[1 : 1+i]
		}
		return strings.TrimPrefix(cmdline, "\"")
	}
	first := cmdline
	if i := strings.IndexByte(cmdline, ' '); i >= 0 {
		first = cmdline[:i]
	}
	if exists == nil {
		return first
	}
	// Try every space-delimited prefix, shortest first, until a real file is
	// named.
	for i := 0; i <= len(cmdline); i++ {
		if i == len(cmdline) || cmdline[i] == ' ' {
			if cand := cmdline[:i]; cand != "" && exists(cand) {
				return cand
			}
		}
	}
	return first
}
