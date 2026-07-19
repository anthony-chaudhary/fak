package servicelease

import (
	"errors"
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/servicespec"
)

// This file is the fenced reconcile loop of #4752: the controller-side engine
// that drives one remote node toward its DESIRED fak.service.v1 state, where
// the authority to restart a node is itself a fenced lease. Two controllers
// (an active/standby pair, a control-plane retry after a partition, or a
// restarted controller racing its own previous epoch) can therefore never
// double-restart a node: exactly one holds the newest restart-authority token,
// and every reconcile step re-proves that token before proposing any action.
// A stale claimant is refused with the same closed sentinel vocabulary the
// workload fence uses (ErrStaleIncarnation / ErrNotHolder / ErrFenced /
// ErrLeaseExpired), and an ownership change simply moves the loop: the new
// owner's steps succeed, the old owner's steps refuse.
//
// The loop composes the leaf's existing parts instead of re-deciding them:
// Classify folds the evidence channels, BuildPlan proposes the safe action,
// servicespec.RestartPolicy paces restarts (bounded backoff, circuit-open —
// never an infinite restart storm), and the Table adjudicates every fence.
// Like the rest of the leaf it is pure and deterministic: time is an explicit
// logical-millisecond argument, plans are dry-run JSON documents, and nothing
// here touches native service state.

// ErrNoAuthority — the controller holds no restart-authority lease for the
// node; it must AcquireAuthority before it may reconcile.
var ErrNoAuthority = errors.New("servicelease: no restart authority held for node; acquire first")

// RestartAuthority names the per-node restart-authority row in the Table. The
// prefix namespaces controller fences away from real workload leases so the
// two can never collide in one table.
func RestartAuthority(node string) string { return "restart-authority/" + node }

// ActionStop is the reconcile action that drives desired-stopped over a node
// still observed running. It extends the plan vocabulary of BuildPlan, which
// classifies intent but does not itself chase it.
const ActionStop ActionKind = "stop"

// StepInput is everything one reconcile step needs: the desired spec, the
// observed evidence, and the logical clock. Intent is single-sourced from the
// spec — the step overwrites Evidence.DesiredStopped and Evidence.NowMS so a
// caller can never present observation under one intent and reconcile under
// another.
type StepInput struct {
	Spec     *servicespec.Spec
	Evidence Evidence
	NowMS    int64
}

// StepResult is one fenced reconcile verdict: the dry-run plan, the restart
// pacing decision when the plan proposes a restart, and the authority token
// the action is authorized under (an executor presents it as the fencing
// token on the effectful call).
type StepResult struct {
	Plan     Plan                         `json:"plan"`
	Decision *servicespec.RestartDecision `json:"decision,omitempty"`
	Token    FencingToken                 `json:"token"`
}

// pace is the per-workload restart pacing memory: consecutive attempt count
// and the timestamps of restarts issued inside the rolling window.
type pace struct {
	attempt  int
	restarts []int64
}

// Reconciler is one controller's fenced reconcile loop over a shared Table.
// Identity is an Incarnation of the CONTROLLER: Node is the controller ID,
// BootID is its epoch — a restarted controller is a new epoch that supersedes
// the old one the moment it acquires, exactly as a rebooted node supersedes
// its old boot. The zero value is not usable; construct with NewReconciler.
type Reconciler struct {
	table     *Table
	self      Incarnation
	authority map[string]*Lease // node -> restart-authority lease held
	paces     map[string]*pace  // workload -> restart pacing state
}

// NewReconciler builds a reconcile loop for one controller epoch over the
// shared fencing table.
func NewReconciler(t *Table, controllerID, epoch string) *Reconciler {
	return &Reconciler{
		table:     t,
		self:      Incarnation{Node: controllerID, BootID: epoch},
		authority: map[string]*Lease{},
		paces:     map[string]*pace{},
	}
}

// Self returns the controller incarnation this loop claims fences as.
func (r *Reconciler) Self() Incarnation { return r.self }

// AcquireAuthority claims the restart-authority fence for the node. It records
// this controller epoch (superseding any older epoch of the same controller)
// and then acquires under the table's normal rules: while ANOTHER controller
// holds a still-valid authority lease the claim is refused (ErrLeaseHeld) —
// the standby waits out the fence instead of creating a second restarter.
func (r *Reconciler) AcquireAuthority(node string, nowMS int64) (*Lease, error) {
	r.table.RecordIncarnation(r.self)
	l, err := r.table.Acquire(RestartAuthority(node), r.self, nowMS)
	if err != nil {
		return nil, err
	}
	r.authority[node] = l
	return l, nil
}

// Step runs one fenced reconcile pass for the spec's node and workload.
//
// The fence comes first: the step renews the node's restart-authority lease
// under this controller's token, and ANY refusal (stale epoch, superseded
// owner, expired lease) aborts the step with that sentinel, drops the local
// authority copy, and proposes nothing — a stale owner does not merely get a
// smaller plan, it gets no plan. Only the current fence owner ever reaches
// the desired-vs-observed comparison.
//
// With authority proven, the step folds evidence through Classify/BuildPlan,
// overrides intent-chasing (desired-stopped over a node still running becomes
// ActionStop), and paces any proposed restart through the spec's
// RestartPolicy: bounded exponential backoff, and circuit-open converts the
// restart into a held ActionNone (reason circuit-open) rather than a storm.
func (r *Reconciler) Step(in StepInput) (StepResult, error) {
	node := in.Spec.Identity.Node
	held := r.authority[node]
	if held == nil {
		return StepResult{}, fmt.Errorf("%w: %s", ErrNoAuthority, node)
	}
	renewed, err := r.table.Renew(RestartAuthority(node), r.self, held.Token, in.NowMS)
	if err != nil {
		delete(r.authority, node) // fenced out: stop acting on this node
		return StepResult{}, err
	}
	r.authority[node] = renewed

	e := in.Evidence
	e.NowMS = in.NowMS
	e.DesiredStopped = in.Spec.Desired == servicespec.DesiredStopped

	workload := in.Spec.Identity.Workload
	if workload == "" {
		workload = in.Spec.Identity.Service
	}
	res := StepResult{Plan: BuildPlan(r.table, workload, e), Token: renewed.Token}

	if e.DesiredStopped && observedRunning(e.ReadBack) {
		res.Plan.Action, res.Plan.Reason = ActionStop, "desired-stopped-still-running"
		return res, nil
	}
	if res.Plan.Action == ActionRestartLocal {
		d := r.decide(workload, in.Spec, e, in.NowMS)
		res.Decision = &d
		if !d.Restart {
			res.Plan.Action, res.Plan.Reason = ActionNone, d.Reason
		}
	}
	return res, nil
}

// observedRunning reports whether the native-manager read-back shows a live
// process (starting, ready, or degraded).
func observedRunning(o *servicespec.Observed) bool {
	if o == nil {
		return false
	}
	switch o.Phase {
	case servicespec.PhaseStarting, servicespec.PhaseReady, servicespec.PhaseDegraded:
		return true
	}
	return false
}

// decide paces one proposed restart through the spec's RestartPolicy using
// this loop's per-workload memory: the consecutive attempt count and the
// restarts issued inside the rolling window. A granted restart is recorded;
// a refused one (circuit open, operator stop) leaves the memory unchanged.
func (r *Reconciler) decide(workload string, spec *servicespec.Spec, e Evidence, nowMS int64) servicespec.RestartDecision {
	p := r.paces[workload]
	if p == nil {
		p = &pace{}
		r.paces[workload] = p
	}
	if w := spec.Restart.WindowMS; w > 0 {
		kept := p.restarts[:0]
		for _, at := range p.restarts {
			if nowMS-at < w {
				kept = append(kept, at)
			}
		}
		p.restarts = kept
	}
	class, runMS := servicespec.ExitCrash, int64(0)
	if e.ReadBack != nil && e.ReadBack.LastExit != nil {
		class, runMS = e.ReadBack.LastExit.Class, e.ReadBack.LastExit.RunMS
	}
	d := spec.Restart.Decide(servicespec.RestartInput{
		Kind:        spec.Kind,
		Desired:     spec.Desired,
		Class:       class,
		Attempt:     p.attempt,
		RunMS:       runMS,
		WindowCount: len(p.restarts),
	})
	if d.Restart {
		p.attempt = d.NextAttempt
		p.restarts = append(p.restarts, nowMS)
	}
	return d
}
