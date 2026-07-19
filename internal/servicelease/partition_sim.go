package servicelease

import (
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/servicespec"
)

// This file is the DETERMINISTIC partition simulation the issue's acceptance
// witness names (#4752): a logical-clock model of N nodes, one leased
// workload, and one control plane, where crash, reboot, partition, and
// delayed-heartbeat faults are injected as explicit events and every step is
// a pure function of the previous state. The invariant under test: at no step
// is there more than one VALID owner (a node whose effectful claims the
// fencing table would accept). A partitioned stale owner may still be
// RUNNING — the simulation counts it, and shows its claims refused.
//
// The same fault scripts are meant to drive the two-node LIVE test on real
// lab nodes later; only the transport and clock change, never the table.

// SimNode is one simulated node: its local truth (boot, process, lease copy)
// plus the controller-side view of it (last heartbeat received).
type SimNode struct {
	Name        string
	BootSeq     int    // increments on reboot
	BootID      string // derived from BootSeq
	Running     bool   // local process running the workload
	Partitioned bool   // true: no messages flow in either direction

	// LeaseCopy is the node's local copy of the lease it believes it holds.
	// It survives a crash (durable on the node) but not a reboot in this
	// model's worst case; tests may also probe the stale copy directly.
	LeaseCopy *Lease

	// Controller-side reception records (what the pull channel last delivered).
	hbAtMS   int64
	hbBootID string
	hbPhase  servicespec.Phase
}

// Incarnation returns the node's CURRENT incarnation.
func (n *SimNode) Incarnation() Incarnation {
	return Incarnation{Node: n.Name, BootID: n.BootID}
}

// Refusal records one fencing refusal witnessed during the simulation.
type Refusal struct {
	AtMS int64
	Node string
	Op   string
	Err  error
}

// Sim is the deterministic simulation state. All progression happens in
// Step(); fault events are explicit method calls between steps. No wall
// clock, no goroutines, no randomness — replaying the same script yields the
// same trace.
type Sim struct {
	Table    *Table
	Workload string
	Nodes    []*SimNode

	NowMS              int64
	StepMS             int64
	HeartbeatTimeoutMS int64

	LastReconcile Plan      // controller's most recent reconcile plan
	Refusals      []Refusal // every fencing refusal the script provoked
	MaxOwners     int       // high-water mark of simultaneous valid RUNNING owners
}

// NewSim builds a simulation over the named nodes with the given lease TTL,
// step size, and heartbeat timeout (all logical milliseconds). Every node
// starts booted, reachable, and heartbeating; no lease is granted yet.
func NewSim(workload string, leaseTTLMS, stepMS, hbTimeoutMS int64, nodeNames ...string) *Sim {
	s := &Sim{
		Table:              NewTable(leaseTTLMS),
		Workload:           workload,
		StepMS:             stepMS,
		HeartbeatTimeoutMS: hbTimeoutMS,
	}
	for _, name := range nodeNames {
		n := &SimNode{Name: name, BootSeq: 1}
		n.BootID = fmt.Sprintf("%s-boot-1", name)
		s.Nodes = append(s.Nodes, n)
		s.Table.RecordIncarnation(n.Incarnation())
	}
	return s
}

func (s *Sim) node(name string) *SimNode {
	for _, n := range s.Nodes {
		if n.Name == name {
			return n
		}
	}
	panic("servicelease sim: unknown node " + name)
}

// Grant bootstraps ownership: the named node acquires the workload lease and
// starts its process.
func (s *Sim) Grant(name string) error {
	n := s.node(name)
	l, err := s.Table.Acquire(s.Workload, n.Incarnation(), s.NowMS)
	if err != nil {
		return err
	}
	n.LeaseCopy = l
	n.Running = true
	return nil
}

// Crash kills the node's local process. The lease copy survives on disk.
func (s *Sim) Crash(name string) { s.node(name).Running = false }

// Reboot restarts the whole host as a NEW incarnation: process gone, local
// lease copy gone, boot ID changed. The controller learns of the new boot
// only when a heartbeat next arrives.
func (s *Sim) Reboot(name string) {
	n := s.node(name)
	n.BootSeq++
	n.BootID = fmt.Sprintf("%s-boot-%d", n.Name, n.BootSeq)
	n.Running = false
	n.LeaseCopy = nil
}

// Partition cuts the node off (heartbeats, renewals, and grants all stop
// flowing). The node keeps running whatever it was running.
func (s *Sim) Partition(name string) { s.node(name).Partitioned = true }

// Heal reconnects a partitioned node.
func (s *Sim) Heal(name string) { s.node(name).Partitioned = false }

// ValidRunningOwners lists nodes that are RUNNING the workload AND whose
// claims the fencing table would accept right now. The safety property of the
// whole design is that this never exceeds one.
func (s *Sim) ValidRunningOwners() []string {
	var out []string
	for _, n := range s.Nodes {
		if n.Running && n.LeaseCopy != nil &&
			s.Table.WouldAccept(s.Workload, n.Incarnation(), n.LeaseCopy.Token, s.NowMS) {
			out = append(out, n.Name)
		}
	}
	return out
}

// RunningNodes lists every node whose process is running, valid or not (a
// partitioned stale owner shows up here but not in ValidRunningOwners).
func (s *Sim) RunningNodes() []string {
	var out []string
	for _, n := range s.Nodes {
		if n.Running {
			out = append(out, n.Name)
		}
	}
	return out
}

func (s *Sim) refuse(node, op string, err error) {
	s.Refusals = append(s.Refusals, Refusal{AtMS: s.NowMS, Node: node, Op: op, Err: err})
}

// Step advances the logical clock one tick and runs the five deterministic
// phases in fixed order: (1) heartbeat delivery from reachable nodes — the
// pull channel that also teaches the table about new incarnations, (2)
// offline local recovery — a crashed holder restarts under its own
// incarnation with no controller contact, (3) renewal attempts by reachable
// running holders — a refusal FENCES the node: it stops its process, (4)
// controller reconcile — classify evidence, build the dry-run plan, execute
// it, (5) invariant accounting. It returns the number of simultaneously valid
// running owners after the step (the safety bound says this is always <= 1).
func (s *Sim) Step() int {
	s.NowMS += s.StepMS

	// (1) Heartbeats from reachable nodes.
	for _, n := range s.Nodes {
		if n.Partitioned {
			continue
		}
		n.hbAtMS = s.NowMS
		n.hbBootID = n.BootID
		if n.Running {
			n.hbPhase = servicespec.PhaseReady
		} else {
			n.hbPhase = servicespec.PhaseFailed
		}
		s.Table.RecordIncarnation(n.Incarnation())
	}

	// (2) Offline local recovery: a crashed node restarts its own process if
	// its own lease copy names its own current incarnation — no controller.
	for _, n := range s.Nodes {
		if !n.Running && LocalRestartAllowed(n.LeaseCopy, n.Incarnation()) {
			n.Running = true
		}
	}

	// (3) Renewals from reachable running holders. A fencing refusal is the
	// signal that ownership moved on: the node stops rather than split-brain.
	for _, n := range s.Nodes {
		if n.Partitioned || !n.Running || n.LeaseCopy == nil {
			continue
		}
		l, err := s.Table.Renew(s.Workload, n.Incarnation(), n.LeaseCopy.Token, s.NowMS)
		if err != nil {
			s.refuse(n.Name, "renew", err)
			n.Running = false
			n.LeaseCopy = nil
			continue
		}
		n.LeaseCopy = l
	}

	// (4) Controller reconcile: evidence about the lease holder (or the lack
	// of any lease), dry-run plan, then execution.
	s.reconcile()

	// (5) Invariant accounting.
	owners := len(s.ValidRunningOwners())
	if owners > s.MaxOwners {
		s.MaxOwners = owners
	}
	return owners
}

// reconcile is the controller half of a step.
func (s *Sim) reconcile() {
	l, ok := s.Table.Leases[s.Workload]
	if !ok {
		s.assign()
		return
	}
	holder := s.node(l.Holder.Node)
	ev := Evidence{
		NowMS:              s.NowMS,
		LastHeartbeatMS:    holder.hbAtMS,
		HeartbeatBootID:    holder.hbBootID,
		HeartbeatTimeoutMS: s.HeartbeatTimeoutMS,
		KnownBootID:        l.Holder.BootID,
	}
	if ev.LastHeartbeatMS > 0 {
		ev.ReadBack = &servicespec.Observed{
			Schema:   servicespec.ObservedSchemaV1,
			Identity: servicespec.Identity{Node: holder.Name, Service: s.Workload},
			Phase:    holder.hbPhase,
		}
	}
	s.LastReconcile = BuildPlan(s.Table, s.Workload, ev)
	switch s.LastReconcile.Action {
	case ActionReassign:
		s.assign()
	case ActionRestartLocal, ActionWaitLease, ActionNone:
		// restart-local is executed by the node itself in phase (2);
		// wait/none mutate nothing — exactly the dry-run contract.
	}
}

// assign grants the workload to the first reachable node whose incarnation is
// current (deterministic: slice order). If none qualifies, nothing happens
// this step.
func (s *Sim) assign() {
	for _, n := range s.Nodes {
		if n.Partitioned {
			continue
		}
		if !s.Table.currentIncarnation(n.Incarnation()) {
			continue
		}
		l, err := s.Table.Acquire(s.Workload, n.Incarnation(), s.NowMS)
		if err != nil {
			s.refuse(n.Name, "acquire", err)
			continue
		}
		n.LeaseCopy = l
		n.Running = true
		return
	}
}
