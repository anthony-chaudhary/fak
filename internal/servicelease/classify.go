package servicelease

import (
	"encoding/json"

	"github.com/anthony-chaudhary/fak/internal/servicespec"
)

// Condition is the reconciled control-plane classification of one workload on
// one node — the issue's closed observed-state vocabulary (#4752). It is a
// VIEW folded from three evidence channels (pull heartbeat, native-manager
// read-back, control-plane records); it never carries intent.
type Condition string

const (
	CondHealthy              Condition = "healthy"
	CondProcessCrashed       Condition = "process-crashed"
	CondHostRebooted         Condition = "host-rebooted"
	CondNetworkPartitioned   Condition = "network-partitioned"
	CondIntentionallyStopped Condition = "intentionally-stopped"
	CondDependencyBlocked    Condition = "dependency-blocked"
	CondUnknown              Condition = "unknown"
)

// AllConditions enumerates the classification vocabulary (stable order).
var AllConditions = []Condition{
	CondHealthy, CondProcessCrashed, CondHostRebooted, CondNetworkPartitioned,
	CondIntentionallyStopped, CondDependencyBlocked, CondUnknown,
}

// Evidence is everything the control plane knows about one workload on one
// node at one instant, across the three channels it must reconcile:
//
//   - the pull-based node heartbeat (LastHeartbeatMS, HeartbeatBootID),
//   - the native service-manager read-back the node last reported (ReadBack,
//     a fak.service.observed.v1 document), and
//   - the control plane's own durable records (KnownBootID, DesiredStopped,
//     DependencyBlocked).
//
// All times are the same explicit logical clock the Table uses.
type Evidence struct {
	NowMS int64 `json:"now_ms"`

	// Pull-based heartbeat channel.
	LastHeartbeatMS    int64  `json:"last_heartbeat_ms"`
	HeartbeatBootID    string `json:"heartbeat_boot_id,omitempty"`
	HeartbeatTimeoutMS int64  `json:"heartbeat_timeout_ms"`

	// Native-manager read-back channel (nil if never reported).
	ReadBack *servicespec.Observed `json:"read_back,omitempty"`

	// Control-plane records.
	KnownBootID       string `json:"known_boot_id,omitempty"`
	DesiredStopped    bool   `json:"desired_stopped,omitempty"`
	DependencyBlocked bool   `json:"dependency_blocked,omitempty"`
}

// heartbeatFresh reports whether the node's pull heartbeat is inside the
// timeout window.
func (e Evidence) heartbeatFresh() bool {
	return e.LastHeartbeatMS > 0 && e.NowMS-e.LastHeartbeatMS <= e.HeartbeatTimeoutMS
}

// Classify folds the three evidence channels into one Condition, in fixed
// precedence order:
//
//  1. intent wins: desired-stopped (or an operator-stop read-back) is
//     intentionally-stopped, reachable or not.
//  2. a fresh heartbeat with a boot ID differing from the control plane's
//     record is host-rebooted — the node is back but as a NEW incarnation.
//  3. a silent node (no heartbeat inside the window) is network-partitioned:
//     the control plane cannot distinguish dead from unreachable, so it
//     classifies the partition and lets FENCING own safety, never a guess.
//  4. a reachable node blocked on a dependency is dependency-blocked.
//  5. a reachable node whose read-back shows failed (or a crash exit) is
//     process-crashed — locally recoverable without the controller.
//  6. a reachable node with a ready/starting/degraded read-back is healthy.
//  7. anything else (e.g. reachable but no read-back yet) is unknown.
func Classify(e Evidence) Condition {
	if e.DesiredStopped {
		return CondIntentionallyStopped
	}
	if e.ReadBack != nil && e.ReadBack.LastExit != nil && e.ReadBack.LastExit.Class == servicespec.ExitOperatorStop {
		return CondIntentionallyStopped
	}
	if e.heartbeatFresh() && e.HeartbeatBootID != "" && e.KnownBootID != "" && e.HeartbeatBootID != e.KnownBootID {
		return CondHostRebooted
	}
	if !e.heartbeatFresh() {
		return CondNetworkPartitioned
	}
	if e.DependencyBlocked {
		return CondDependencyBlocked
	}
	if e.ReadBack == nil {
		return CondUnknown
	}
	switch e.ReadBack.Phase {
	case servicespec.PhaseFailed:
		return CondProcessCrashed
	case servicespec.PhaseReady, servicespec.PhaseStarting, servicespec.PhaseDegraded:
		return CondHealthy
	case servicespec.PhaseStopped:
		if e.ReadBack.LastExit != nil && e.ReadBack.LastExit.Class == servicespec.ExitBootRecovery {
			return CondHostRebooted
		}
		return CondProcessCrashed // stopped without intent: the process is down
	case servicespec.PhaseFenced:
		return CondDependencyBlocked // held off by circuit/operator fence, not crashed
	default:
		return CondUnknown
	}
}

// ReconcileSchemaV1 names the dry-run reconciliation-plan document this leaf emits.
const ReconcileSchemaV1 = "fak.service.lease-plan.v1"

// ActionKind is the closed set of reconciliation actions a plan may propose.
type ActionKind string

const (
	// ActionNone — nothing to do (healthy, stopped-as-desired, or blocked
	// where waiting is the policy).
	ActionNone ActionKind = "none"
	// ActionRestartLocal — the OWNING incarnation restarts its local process;
	// offline-capable, no controller round-trip, lease unchanged.
	ActionRestartLocal ActionKind = "restart-local"
	// ActionReassign — fencing permits granting the workload to a current
	// incarnation (old owner expired, superseded, or generation-bumped).
	ActionReassign ActionKind = "reassign"
	// ActionWaitLease — the owner is unreachable but its lease is still
	// valid; reassignment is REFUSED until lease/fencing policy permits it.
	ActionWaitLease ActionKind = "wait-lease"
)

// Plan is the dry-run JSON document a reconciler emits BEFORE mutating any
// native service state. This leaf only ever produces plans (DryRun is always
// true here); an executor that carries one out flips DryRun in its own record.
type Plan struct {
	Schema    string     `json:"schema"`
	Workload  string     `json:"workload"`
	Condition Condition  `json:"condition"`
	Action    ActionKind `json:"action"`
	Reason    string     `json:"reason"`
	DryRun    bool       `json:"dry_run"`
}

// JSON returns the deterministic wire form of the plan.
func (p Plan) JSON() ([]byte, error) { return json.MarshalIndent(&p, "", "  ") }

// BuildPlan classifies the evidence and proposes the one safe next action for
// the workload, consulting the fencing table before ever proposing a remote
// reassignment. It never mutates the table.
func BuildPlan(t *Table, workload string, e Evidence) Plan {
	p := Plan{Schema: ReconcileSchemaV1, Workload: workload, Condition: Classify(e), DryRun: true}
	switch p.Condition {
	case CondIntentionallyStopped:
		p.Action, p.Reason = ActionNone, "desired-stopped"
	case CondHealthy:
		p.Action, p.Reason = ActionNone, "healthy"
	case CondDependencyBlocked:
		p.Action, p.Reason = ActionNone, "dependency-blocked"
	case CondProcessCrashed:
		// The holder recovers locally under its own incarnation — no
		// controller involvement, no ownership change.
		p.Action, p.Reason = ActionRestartLocal, "local-recovery"
	case CondHostRebooted, CondNetworkPartitioned, CondUnknown:
		if ok, why := RemoteReassignAllowed(t, workload, e.NowMS); ok {
			p.Action, p.Reason = ActionReassign, why
		} else {
			p.Action, p.Reason = ActionWaitLease, why
		}
	}
	return p
}
