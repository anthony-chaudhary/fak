// quarantine.go — the desired-population (target) admission gate every resume/recovery
// actuator must pass before it may put work back on the fleet.
//
// # The backdoor this closes
//
// An operator quarantines the fleet by declaring a desired worker population of ZERO
// (`dos loop --target 0` => AT_TARGET, alive=0). That declaration binds the supervisors,
// which read it — and nothing else. The RECOVERY half of the fleet never read it at all:
// the owner-seat resume task re-enabled the issue-dispatch tasks after a seat reset, the
// resume watchdog kept firing `claude --resume`, and the stranded-recovery task kept
// re-launching stranded work, each on its own schedule and each with its own idea of how
// many workers ought to exist (#6505). A quarantine that only the supervisors honor is not
// a quarantine: it is a race between the operator and whichever recovery timer ticks next.
//
// [AdmitQuarantine] is that missing primitive — one total function over (declared
// population, what the actor wants to DO) that every recovery path consults, so target 0
// refuses reset, resume, re-enable, restart and refill ALIKE rather than only the paths
// that happened to be wired to a supervisor.
//
// # Fail closed, on purpose
//
// Every other guard in this package is fail-OPEN per key: an unreadable process table or
// a missing roster must not strand a genuinely-crashed session (see [WatchdogGuards]).
// This gate inverts that default for the ACTING classes, because the two errors are not
// symmetric. Wrongly refusing a recovery costs one tick of latency — the next tick with a
// readable target admits it. Wrongly ADMITTING one puts live workers on a fleet the
// operator believes is drained, which is exactly the harm audited on 2026-08-11: a
// supervisor kept a target of 5 and spawned twelve workers while the authoritative view
// reported alive=0. So an UNKNOWN population refuses just as a zero one does
// (ReasonFleetTargetUnknown), and only a positive, explicitly DECLARED population admits.
//
// Read-only classes ([RecoveryStatusRead]) are never refused: the operator needs status
// MOST while quarantined, and reporting cannot start work. This is the same
// class-not-process line internal/fleetfreeze draws for its freeze.
//
// # Pure by construction
//
// No clock, no I/O, no env read. The SHELL folds the declared population into a
// [FleetPosture] (from the control-pane config's target, the loop's declared target, or
// whatever the fleet's declared SSOT is) and stamps its provenance; this leaf only
// decides. Same posture + same action in, same verdict out — which is what makes the
// target-0 regression a deterministic proof rather than a timing-dependent flake.
package resume

import "fmt"

// Structured quarantine verdict reasons. Closed vocabulary in the spirit of the
// AdmitSource refusal set (source_governor.go): a refusal carries a token a downstream
// can route on, never free-text drift.
const (
	// ReasonQuarantineAdmitted: the declared population is positive (or the action is
	// read-only), so the quarantine gate does not stand in the way.
	ReasonQuarantineAdmitted = "QUARANTINE_ADMITTED"
	// ReasonFleetQuarantined: the operator declared a desired population of zero. Every
	// acting class is refused, including the reset/recovery paths that historically had
	// their own targets.
	ReasonFleetQuarantined = "FLEET_QUARANTINED"
	// ReasonFleetTargetUnknown: no declared population was supplied, or the read of it
	// failed. Refused for the acting classes — an unreadable target must never read as
	// permission to start work.
	ReasonFleetTargetUnknown = "FLEET_TARGET_UNKNOWN"
)

// FleetTargetState is the closed vocabulary for what the shell learned about the declared
// desired worker population this tick. It is carried separately from the number so that
// "declared zero" (a deliberate quarantine) and "could not tell" (a failed read) are
// distinguishable facts rather than both collapsing onto the int 0.
type FleetTargetState string

const (
	// FleetTargetUndeclared is the zero value: the caller supplied no population at all.
	// [AdmitQuarantine] refuses the acting classes on it — a caller that forgot to fold
	// the SSOT is precisely the bypass this gate exists to close.
	FleetTargetUndeclared FleetTargetState = ""
	// FleetTargetDeclared: the SSOT was read and DesiredWorkers is authoritative.
	FleetTargetDeclared FleetTargetState = "declared"
	// FleetTargetUnreadable: a read was attempted and failed (missing/malformed config,
	// unreachable loop). Treated exactly like undeclared, but kept distinct so status can
	// tell "nobody wired it" apart from "the wiring broke".
	FleetTargetUnreadable FleetTargetState = "unreadable"
)

// FleetPosture is the folded desired-population SSOT one tick decides against: the
// declared worker target plus where it came from and why. The shell builds it; this leaf
// never reads it from disk.
type FleetPosture struct {
	// State says whether DesiredWorkers may be trusted at all.
	State FleetTargetState `json:"state,omitempty"`
	// DesiredWorkers is the declared population. Meaningful only when State is
	// FleetTargetDeclared; zero (or negative) there is the operator's quarantine.
	DesiredWorkers int `json:"desired_workers,omitempty"`
	// Source names the SSOT the number came from (e.g. "control-pane config target",
	// "dos loop --target"), so a refusal points at the thing an operator would change.
	Source string `json:"source,omitempty"`
	// Reason is the operator's stated reason for the posture, carried into the refusal
	// summary so a paged human sees WHY the fleet is held.
	Reason string `json:"reason,omitempty"`
}

// DeclaredFleetTarget builds a posture from a successfully-read SSOT value. A
// non-positive workers value is the quarantine, not an error.
func DeclaredFleetTarget(workers int, source, reason string) FleetPosture {
	return FleetPosture{State: FleetTargetDeclared, DesiredWorkers: workers, Source: source, Reason: reason}
}

// UnreadableFleetTarget builds the posture a FAILED SSOT read produces. detail is the
// read error, carried as the posture's reason so the refusal names it.
func UnreadableFleetTarget(source, detail string) FleetPosture {
	return FleetPosture{State: FleetTargetUnreadable, Source: source, Reason: detail}
}

// Quarantined reports whether the posture is an explicit, declared target-0 hold. False
// for an undeclared/unreadable posture — that is UNKNOWN, not proof of a quarantine, and
// the two refuse under different reasons.
func (p FleetPosture) Quarantined() bool {
	return p.State == FleetTargetDeclared && p.DesiredWorkers <= 0
}

// RecoveryAction is the closed vocabulary of what a recovery/resume actuator wants to do,
// classified by EFFECT rather than by which task or script is asking. Classifying by
// effect is what stops a new bypass from being invented: an unregistered supervisor and a
// scheduled task that both refill workers land on the same member and get the same answer.
type RecoveryAction string

const (
	// RecoveryResumeSession: fire `claude --resume` for a dead session (the resume
	// watchdog's launch).
	RecoveryResumeSession RecoveryAction = "resume_session"
	// RecoveryEnableDispatchTask: flip a dispatch Scheduled Task from disabled to enabled
	// (what FleetOwnerSeatResume does after an owner-seat reset).
	RecoveryEnableDispatchTask RecoveryAction = "enable_dispatch_task"
	// RecoveryStartDispatchTask: start an already-enabled dispatch task NOW.
	RecoveryStartDispatchTask RecoveryAction = "start_dispatch_task"
	// RecoveryOwnerSeatReset: run the owner-seat reset flow, whose tail re-enables
	// dispatch. Named separately from the enable it performs so status can attribute the
	// refusal to the flow an operator recognizes.
	RecoveryOwnerSeatReset RecoveryAction = "owner_seat_reset"
	// RecoveryStrandedRecovery: re-launch work whose worker was lost (stranded recovery).
	RecoveryStrandedRecovery RecoveryAction = "stranded_recovery"
	// RecoveryRefillWorkers: top the live worker population back up to some target — the
	// class the unmanaged supervisor and the manual refill both belong to.
	RecoveryRefillWorkers RecoveryAction = "refill_workers"
	// RecoveryStatusRead: read and report fleet/recovery state. The one read-only class,
	// ALWAYS admitted: an operator needs status most while quarantined.
	RecoveryStatusRead RecoveryAction = "status_read"
)

// ActingRecoveryActions is every class that can put work back on the fleet — the exact
// set target 0 must refuse. Exported so a caller (and the regression witness) can
// enumerate the gate's obligation instead of re-listing it by hand.
//
//enumlint:exempt RecoveryStatusRead is read-only by definition and therefore excluded from acting actions.
var ActingRecoveryActions = []RecoveryAction{
	RecoveryResumeSession,
	RecoveryEnableDispatchTask,
	RecoveryStartDispatchTask,
	RecoveryOwnerSeatReset,
	RecoveryStrandedRecovery,
	RecoveryRefillWorkers,
}

// Acts reports whether the action can put work back on the fleet. Every action except
// the read-only status class acts — including an UNKNOWN token, which is treated as
// acting so an unrecognized future actuator fails closed rather than slipping through.
func (a RecoveryAction) Acts() bool { return a != RecoveryStatusRead }

// QuarantineDecision is the verdict for one (posture, action) pair: admit plus the
// closed reason token and a human summary naming the SSOT an operator would change.
type QuarantineDecision struct {
	Admit          bool           `json:"admit"`
	Action         RecoveryAction `json:"action"`
	Reason         string         `json:"reason"`
	Summary        string         `json:"summary"`
	DesiredWorkers int            `json:"desired_workers"`
}

// AdmitQuarantine decides whether act may proceed under the declared population p. Total
// over any input, and the whole safety property in one sentence: NO acting class is ever
// admitted without a positive, explicitly DECLARED desired population.
func AdmitQuarantine(p FleetPosture, act RecoveryAction) QuarantineDecision {
	d := QuarantineDecision{Action: act, DesiredWorkers: p.DesiredWorkers}
	if !act.Acts() {
		d.Admit, d.Reason = true, ReasonQuarantineAdmitted
		d.Summary = fmt.Sprintf("%s is read-only; never held by a quarantine", act)
		return d
	}
	switch {
	case p.State != FleetTargetDeclared:
		d.Reason = ReasonFleetTargetUnknown
		d.Summary = fmt.Sprintf("desired worker population is unknown (%s); %s refused — "+
			"an unreadable target must never read as permission to start work",
			postureOrigin(p), act)
	case p.DesiredWorkers <= 0:
		d.Reason = ReasonFleetQuarantined
		d.Summary = fmt.Sprintf("fleet is quarantined at target %d (%s); %s refused — "+
			"recovery may not put work back on a fleet the operator drained",
			p.DesiredWorkers, postureOrigin(p), act)
	default:
		d.Admit, d.Reason = true, ReasonQuarantineAdmitted
		d.Summary = fmt.Sprintf("declared target %d (%s) admits %s", p.DesiredWorkers, postureOrigin(p), act)
	}
	return d
}

// postureOrigin renders the provenance half of a summary: the SSOT the number came from
// plus the operator's stated reason, each omitted when the shell did not supply it.
func postureOrigin(p FleetPosture) string {
	src := p.Source
	if src == "" {
		src = "no declared source"
	}
	if p.Reason == "" {
		return src
	}
	return src + ": " + p.Reason
}

// --- the enumerated actors (what status must be able to name) -------------------------

// RecoveryActorKind separates automation that a task scheduler owns from automation that
// nothing owns. The 2026-08-11 audit found the second kind twice — an unmanaged
// supervisor process and a hand-run refill — after the scheduled-task inventory had been
// declared clean, so an enumeration that only covers Scheduled Tasks is not an
// enumeration.
type RecoveryActorKind string

const (
	// ActorScheduledTask: a registered Windows Scheduled Task.
	ActorScheduledTask RecoveryActorKind = "scheduled_task"
	// ActorUnmanagedProcess: a long-running supervisor process no scheduler registered
	// (the isolated-nemotron supervisor found maintaining target=5).
	ActorUnmanagedProcess RecoveryActorKind = "unmanaged_process"
	// ActorOperatorSession: a human/agent session issuing refills by hand.
	ActorOperatorSession RecoveryActorKind = "operator_session"
)

// RecoveryActor is one piece of automation capable of changing dispatch-task enablement
// or worker population. The list is the audited inventory from #6505 (plus the
// target-8/target-4 supervisors from #6502), kept in code so status can enumerate it and
// so a new bypass has an obvious place to be declared.
type RecoveryActor struct {
	// Name is the Scheduled Task name, process name, or operator label.
	Name string            `json:"name"`
	Kind RecoveryActorKind `json:"kind"`
	// Action is the effect class the actor performs — the member AdmitQuarantine judges.
	Action RecoveryAction `json:"action"`
	// Actuator is the script/command that does the work, so status points somewhere.
	Actuator string `json:"actuator,omitempty"`
	// Gated reports whether this actor's DECISION CORE consults AdmitQuarantine. It does
	// NOT claim the actor's shell already folds a declared FleetPosture: a tick that
	// supplies FleetTargetUndeclared still leaves the row-level guard inert (see
	// DecideWatchdogRow). Ungated actors are the standing exposure.
	Gated bool   `json:"gated"`
	Note  string `json:"note,omitempty"`
}

// recoveryActors is the audited inventory. Ordered most-recently-audited last so the
// list reads as the history it is.
var recoveryActors = []RecoveryActor{
	{
		Name: "FleetResumeWatchdog", Kind: ActorScheduledTask, Action: RecoveryResumeSession,
		Actuator: "tools/fleet_resume_watchdog.ps1", Gated: true,
		Note: "DecideWatchdogRow applies the quarantine guard above every per-row fact; " +
			"the tick must still fold a declared FleetPosture for it to bind",
	},
	{
		Name: "FleetOwnerSeatResume", Kind: ActorScheduledTask, Action: RecoveryOwnerSeatReset,
		Actuator: "%LOCALAPPDATA%/Fleet/resume_on_owner_after_reset.ps1",
		Note:     "re-enables and starts FleetIssueDispatch* after an owner-seat reset",
	},
	{
		Name: "FleetStrandedRecovery", Kind: ActorScheduledTask, Action: RecoveryStrandedRecovery,
		Actuator: "%LOCALAPPDATA%/Fleet/stranded_recovery.ps1",
		Note:     "re-launches work whose worker was lost",
	},
	{
		Name: "FleetDOSDispatchWatchdog", Kind: ActorScheduledTask, Action: RecoveryStartDispatchTask,
		Actuator: "tools/dos_supervisor_watchdog.py", Note: "ran -Target 8 every five minutes (#6502)",
	},
	{
		Name: "FleetSupervisorWatchdog", Kind: ActorScheduledTask, Action: RecoveryStartDispatchTask,
		Actuator: "tools/fleet_supervisor_watchdog.ps1", Note: "ran -Target 4 every five minutes (#6502)",
	},
	{
		Name: "FakOvernightIsolatedNemotron", Kind: ActorScheduledTask, Action: RecoveryRefillWorkers,
		Actuator: "_scratch/isolated-nemotron-tick.ps1",
		Note:     "one-minute refill registered DURING the audit; recreated the supervisor after it was stopped",
	},
	{
		Name: "isolated-nemotron-supervisor", Kind: ActorUnmanagedProcess, Action: RecoveryRefillWorkers,
		Actuator: "_scratch/isolated-nemotron-supervisor.ps1",
		Note:     "held target=5 and spawned 12 workers while the authoritative view reported alive=0",
	},
	{
		Name: "operator manual refill", Kind: ActorOperatorSession, Action: RecoveryRefillWorkers,
		Note: "a session launched a worker at 21:59:42 while logging task_disabled_by_policy",
	},
}

// RecoveryActors returns the enumerated inventory (a copy — the registry is not
// caller-mutable).
func RecoveryActors() []RecoveryActor {
	return append([]RecoveryActor(nil), recoveryActors...)
}

// UngatedRecoveryActors returns the acting actors whose decision core does NOT consult
// AdmitQuarantine — the standing bypass surface a status page must show rather than
// report a clean inventory.
func UngatedRecoveryActors() []RecoveryActor {
	var out []RecoveryActor
	for _, a := range recoveryActors {
		if a.Action.Acts() && !a.Gated {
			out = append(out, a)
		}
	}
	return out
}

// --- lifting the quarantine -----------------------------------------------------------

// Re-enable refusal reasons, checked in the fixed order AdmitReEnable applies.
const (
	// ReasonReEnableAdmitted: every precondition is met; the population may be raised.
	ReasonReEnableAdmitted = "REENABLE_ADMITTED"
	// ReasonTargetNotRaised: the request does not actually ask for a positive population,
	// so there is nothing to admit.
	ReasonTargetNotRaised = "TARGET_NOT_RAISED"
	// ReasonCanaryUnwitnessed: no witnessed canary proved a single worker can survive.
	ReasonCanaryUnwitnessed = "CANARY_UNWITNESSED"
	// ReasonLauncherGateOpen: the launcher defect that reports success for dead workers
	// (#6492) is unresolved — a re-enable would re-enter the state that produced the audit.
	ReasonLauncherGateOpen = "LAUNCHER_GATE_OPEN"
	// ReasonStatusGateOpen: the status defect (#6495) is unresolved, so the operator could
	// not SEE a bad re-enable even if it happened.
	ReasonStatusGateOpen = "STATUS_GATE_OPEN"
)

// ReEnableRequest is the evidence an operator offers for lifting a quarantine. Every
// field defaults to "unproven", so the zero value admits nothing.
type ReEnableRequest struct {
	// DesiredWorkers is the population being asked for; must be positive.
	DesiredWorkers int `json:"desired_workers"`
	// CanaryWitnessed: a bounded canary ran and its worker was PROVEN alive/productive —
	// not merely spawned (the exact distinction #6492 collapsed).
	CanaryWitnessed bool `json:"canary_witnessed"`
	// LauncherGateResolved / StatusGateResolved: the audited launcher (#6492) and status
	// (#6495) defects are closed.
	LauncherGateResolved bool `json:"launcher_gate_resolved"`
	StatusGateResolved   bool `json:"status_gate_resolved"`
}

// AdmitReEnable decides whether a quarantine may be lifted. Gates are checked in a fixed
// order so the reason is deterministic; the first failing gate wins. The order is
// target → canary → launcher → status: a request that does not raise the population is
// not a re-enable at all, then the empirical proof, then the two defects that made the
// old evidence untrustworthy.
func AdmitReEnable(req ReEnableRequest) QuarantineDecision {
	d := QuarantineDecision{Action: RecoveryEnableDispatchTask, DesiredWorkers: req.DesiredWorkers}
	switch {
	case req.DesiredWorkers <= 0:
		d.Reason, d.Summary = ReasonTargetNotRaised,
			fmt.Sprintf("requested target %d is not a re-enable; the quarantine stands", req.DesiredWorkers)
	case !req.CanaryWitnessed:
		d.Reason, d.Summary = ReasonCanaryUnwitnessed,
			"no witnessed canary: prove one worker survives and produces before raising the target"
	case !req.LauncherGateResolved:
		d.Reason, d.Summary = ReasonLauncherGateOpen,
			"launcher still reports success for dead workers (#6492); a re-enable would re-enter the audited state"
	case !req.StatusGateResolved:
		d.Reason, d.Summary = ReasonStatusGateOpen,
			"status cannot yet show a bad re-enable (#6495); lift the quarantine only when it is observable"
	default:
		d.Admit, d.Reason = true, ReasonReEnableAdmitted
		d.Summary = fmt.Sprintf("witnessed canary and resolved launcher/status gates admit target %d",
			req.DesiredWorkers)
	}
	return d
}
