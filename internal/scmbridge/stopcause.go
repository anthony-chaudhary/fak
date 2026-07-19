package scmbridge

import (
	"github.com/anthony-chaudhary/fak/internal/servicespec"
)

// StopCause is the closed vocabulary of WHY a Windows-supervised run ended.
// It is deliberately finer than servicespec.ExitClass: the contract's rule
// (#4756) is that user logoff and operator stop are each their own cause and
// NEITHER is a crash — a logoff must never be fought with crash-restart, and
// an operator stop must never be resurrected.
type StopCause string

const (
	CauseCrash        StopCause = "crash"
	CauseOperatorStop StopCause = "operator-stop"
	CauseUserLogoff   StopCause = "user-logoff"
	CauseHostReboot   StopCause = "host-reboot"
	CauseCleanExit    StopCause = "clean-exit"
)

// AllStopCauses enumerates the vocabulary (stable order).
var AllStopCauses = []StopCause{CauseCrash, CauseOperatorStop, CauseUserLogoff, CauseHostReboot, CauseCleanExit}

// StopReport is the native stop evidence one sensor saw. Every field is what
// Windows itself reports — native event IDs, SCM exit codes, Task Scheduler
// instance IDs, and the interactive session identity — so the classification
// (and the ledger row built from it) stays bound to read-back, not to the
// supervisor's opinion.
type StopReport struct {
	// Provider is the native event provider ("Service Control Manager",
	// "Microsoft-Windows-TaskScheduler", "EventLog", "USER32", …).
	Provider string `json:"provider,omitempty"`
	// NativeEventID is the Windows event ID that reported the stop
	// (7031/7034 crash, 7036 state change, 6008 dirty boot, 1074 initiated
	// shutdown, 4647 user-initiated logoff, 102/103 task end, …).
	NativeEventID int `json:"native_event_id,omitempty"`
	// OperatorVerb marks a stop that arrived through an explicit control
	// verb: an SCM stop control, `schtasks /End`, or `fak service uninstall`.
	OperatorVerb bool `json:"operator_verb,omitempty"`
	// UserLogoff marks a WTS logoff notification / event 4647 correlated stop —
	// the interactive session ended, taking InteractiveToken work with it.
	UserLogoff bool `json:"user_logoff,omitempty"`
	// DirtyBoot marks event 6008 / Kernel-Power 41: the previous shutdown
	// was unexpected — the HOST failed, not the workload.
	DirtyBoot bool `json:"dirty_boot,omitempty"`
	// InitiatedShutdown marks event 1074: a requested shutdown/restart.
	InitiatedShutdown bool `json:"initiated_shutdown,omitempty"`
	// ExitCode is the service win32/service-specific exit code or the task
	// action result code.
	ExitCode int `json:"exit_code,omitempty"`
	// TaskInstance is the Task Scheduler instance (activity) ID.
	TaskInstance string `json:"task_instance,omitempty"`
	// Session is the interactive session identity the work was bound to
	// (e.g. "1/console"), carried into the ledger so a resume can prove it
	// re-entered the SAME logical session lineage.
	Session string `json:"session,omitempty"`
}

// Classify maps native stop evidence onto (cause, portable exit class), in
// priority order: host evidence beats session evidence beats operator verbs
// beats exit codes. User logoff maps to the operator-stop EXIT CLASS (it is
// intent-shaped: never fought by restart) while keeping its own CAUSE so the
// resume rule can re-arm on logon instead of treating it as final.
func Classify(r StopReport) (StopCause, servicespec.ExitClass) {
	switch {
	case r.DirtyBoot || r.NativeEventID == 6008 || (r.Provider == "Microsoft-Windows-Kernel-Power" && r.NativeEventID == 41):
		return CauseHostReboot, servicespec.ExitBootRecovery
	case r.InitiatedShutdown || r.NativeEventID == 1074:
		return CauseHostReboot, servicespec.ExitBootRecovery
	case r.UserLogoff || r.NativeEventID == 4647:
		return CauseUserLogoff, servicespec.ExitOperatorStop
	case r.OperatorVerb:
		return CauseOperatorStop, servicespec.ExitOperatorStop
	case r.Provider == "Service Control Manager" && (r.NativeEventID == 7031 || r.NativeEventID == 7034):
		return CauseCrash, servicespec.ExitCrash
	case r.ExitCode != 0:
		return CauseCrash, servicespec.ExitCrash
	default:
		return CauseCleanExit, servicespec.ExitClean
	}
}

// Resume is the deterministic resume ruling for one classified stop.
type Resume struct {
	Restart bool  `json:"restart"`
	DelayMS int64 `json:"delay_ms,omitempty"`
	// AwaitLogon re-arms the broker's logon trigger instead of restarting:
	// the desktop is gone, and only a logon brings it back.
	AwaitLogon bool `json:"await_logon,omitempty"`
	// Fenced mirrors circuit-open: the restart window is exhausted.
	Fenced bool   `json:"fenced,omitempty"`
	Reason string `json:"reason"`
}

// Resume-rule reasons beyond the servicespec restart vocabulary.
const (
	ReasonAwaitLogon = "await-logon"
	// ReasonMisplacedLogoff: a logoff-correlated death of a machine service
	// or S4U watchdog — those roles must not depend on a desktop, so this is
	// surfaced for reconciliation instead of blindly restarted.
	ReasonMisplacedLogoff = "logoff-outside-desktop-role"
)

// RuleResume applies the ONE restart contract to a classified Windows stop.
// It derives the exit class from the cause (single chokepoint — in.Class is
// overwritten) and overlays the two Windows-only rules: a broker taken down
// by logoff waits for the next logon, and a logoff-correlated death outside
// the desktop role is surfaced, not restarted.
func RuleResume(role Role, cause StopCause, p servicespec.RestartPolicy, in servicespec.RestartInput) Resume {
	if cause == CauseUserLogoff {
		if role == RoleBroker {
			return Resume{AwaitLogon: true, Reason: ReasonAwaitLogon}
		}
		return Resume{Reason: ReasonMisplacedLogoff}
	}
	switch cause {
	case CauseCrash:
		in.Class = servicespec.ExitCrash
	case CauseOperatorStop:
		in.Class = servicespec.ExitOperatorStop
	case CauseHostReboot:
		in.Class = servicespec.ExitBootRecovery
	case CauseCleanExit:
		in.Class = servicespec.ExitClean
	}
	d := p.Decide(in)
	return Resume{Restart: d.Restart, DelayMS: d.DelayMS, Fenced: d.CircuitOpen, Reason: d.Reason}
}
