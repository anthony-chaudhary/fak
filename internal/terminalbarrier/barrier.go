// Package terminalbarrier coordinates the fail-closed pause barrier before terminal host replacement.
package terminalbarrier

import (
	"context"
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/fleetbus"
	"github.com/anthony-chaudhary/fak/internal/lifecycleadapter"
	"github.com/anthony-chaudhary/fak/internal/processforest"
)

type Member struct {
	ID         string
	Generation uint64
	Active     bool
	Adapter    lifecycleadapter.Adapter
}
type Forest struct {
	ID         string
	Authority  string
	Generation uint64
	Members    []Member
}
type Actuator interface {
	StopHost(context.Context) error
	RestoreHost(context.Context) error
}
type Report struct {
	TransactionID string                  `json:"transaction_id"`
	ForestID      string                  `json:"forest_id"`
	Verdict       string                  `json:"verdict"`
	Reason        string                  `json:"reason"`
	StopCalls     int                     `json:"stop_calls"`
	RestoreCalls  int                     `json:"restore_calls"`
	Acks          []fleetbus.LifecycleAck `json:"acks,omitempty"`
	Readback      []string                `json:"readback,omitempty"`
}
type Coordinator struct {
	Discover func() Forest
	Bus      fleetbus.LifecycleDirBus
	Actuator Actuator
	Now      func() time.Time
}

func (c Coordinator) Replace(ctx context.Context, pressured bool, forest Forest, transactionID string, deadline time.Time) Report {
	now := time.Now().UTC()
	if c.Now != nil {
		now = c.Now().UTC()
	}
	r := Report{TransactionID: transactionID, ForestID: forest.ID, Verdict: "ABSTAIN"}
	if !pressured {
		r.Verdict = "BELOW_THRESHOLD"
		r.Reason = "pressure below replacement threshold"
		return r
	}
	if forest.ID == "" || forest.Authority == "" || forest.Generation == 0 || transactionID == "" || deadline.IsZero() {
		r.Reason = "managed forest transaction incomplete"
		return r
	}
	active := activeMembers(forest.Members)
	if len(active) == 0 {
		r.Reason = "managed forest has no active members"
		return r
	}
	ids := make([]string, len(active))
	for i, m := range active {
		ids[i] = m.ID
	}
	if err := c.publish(forest, transactionID, fleetbus.LifecyclePrepare, deadline, ids); err != nil {
		r.Reason = "persist prepare requests: " + err.Error()
		return r
	}
	for _, m := range active {
		base := lifecycleadapter.Request{TransactionID: transactionID, ForestID: forest.ID, MemberID: m.ID, Generation: m.Generation, Deadline: deadline, Operation: lifecycleadapter.Prepare}
		n, res := lifecycleadapter.Execute(ctx, base, m.Adapter)
		if !n.Supported || res.State != lifecycleadapter.ResultCompleted {
			c.writeAck(&r, forest, m, fleetbus.LifecycleAckRefused, "prepare refused: "+res.Reason, "", now)
			r.Reason = "member prepare barrier incomplete"
			return r
		}
	}
	if err := c.publish(forest, transactionID, fleetbus.LifecyclePause, deadline, ids); err != nil {
		r.Reason = "persist pause requests: " + err.Error()
		return r
	}
	for _, m := range active {
		base := lifecycleadapter.Request{TransactionID: transactionID, ForestID: forest.ID, MemberID: m.ID, Generation: m.Generation, Deadline: deadline, Operation: lifecycleadapter.Pause}
		n, res := lifecycleadapter.Execute(ctx, base, m.Adapter)
		if !n.Supported || res.State != lifecycleadapter.ResultCompleted {
			c.writeAck(&r, forest, m, fleetbus.LifecycleAckRefused, "pause refused: "+res.Reason, "", now)
			r.Reason = "member pause barrier incomplete"
			return r
		}
		checkpoint := ""
		if n.Document.ApplicationCheckpoint {
			base.Operation = lifecycleadapter.Checkpoint
			n, res = lifecycleadapter.Execute(ctx, base, m.Adapter)
			if !n.Supported || res.State != lifecycleadapter.ResultCompleted {
				c.writeAck(&r, forest, m, fleetbus.LifecycleAckRefused, "checkpoint failed: "+res.Reason, "", now)
				r.Reason = "member checkpoint barrier incomplete"
				return r
			}
			checkpoint = res.CheckpointRef
			if checkpoint == "" {
				checkpoint = res.ReadbackRef
			}
			if checkpoint == "" {
				c.writeAck(&r, forest, m, fleetbus.LifecycleAckRefused, "checkpoint evidence missing", "", now)
				r.Reason = "member checkpoint evidence incomplete"
				return r
			}
		}
		readback := res.ReadbackRef
		if readback == "" {
			readback = fmt.Sprintf("%s:paused", m.ID)
		}
		c.writeAck(&r, forest, m, fleetbus.AckCompleted, "", checkpoint, now)
		r.Readback = append(r.Readback, readback)
	}
	// The coordinator never trusts its own in-memory tally: quiescence is proven
	// by reading the durable acknowledgements back off the bus.
	acks, err := c.Bus.ReadAcks(transactionID)
	if err != nil {
		r.Reason = "read lifecycle acknowledgements: " + err.Error()
		return r
	}
	r.Acks = acks
	if len(acks) != len(active) {
		r.Reason = fmt.Sprintf("durable acknowledgements incomplete: %d of %d members", len(acks), len(active))
		return r
	}
	for i, a := range acks {
		if a.State != fleetbus.AckCompleted || a.MemberID != active[i].ID || a.TransactionID != transactionID {
			r.Reason = "durable acknowledgement not completed for " + a.MemberID
			return r
		}
	}
	before := activeMembers(forest.Members)
	if c.Discover != nil {
		before = activeMembers(c.Discover().Members)
	}
	if !sameMembers(active, before) {
		r.Reason = "forest membership changed while quiescing"
		return r
	}
	if c.Actuator == nil {
		r.Reason = "host actuator unavailable"
		return r
	}
	if err := c.Actuator.StopHost(ctx); err != nil {
		r.Reason = "host replacement failed: " + err.Error()
		return r
	}
	r.StopCalls++
	if err := c.Actuator.RestoreHost(ctx); err != nil {
		r.Reason = "host restore failed: " + err.Error()
		return r
	}
	r.RestoreCalls++
	for _, m := range active {
		base := lifecycleadapter.Request{TransactionID: transactionID, ForestID: forest.ID, MemberID: m.ID, Generation: m.Generation, Operation: lifecycleadapter.Resume, Deadline: deadline}
		_, res := lifecycleadapter.Execute(ctx, base, m.Adapter)
		if res.State != lifecycleadapter.ResultCompleted {
			r.Reason = "member resume failed"
			return r
		}
		base.Operation = lifecycleadapter.Readiness
		_, res = lifecycleadapter.Execute(ctx, base, m.Adapter)
		if res.State != lifecycleadapter.ResultCompleted {
			r.Reason = "member readiness failed"
			return r
		}
		r.Readback = append(r.Readback, res.ReadbackRef)
	}
	r.Verdict = "READY"
	r.Reason = "all active members quiesced, host replaced, restored, resumed, and ready"
	return r
}
func (c Coordinator) publish(f Forest, transactionID string, action fleetbus.LifecycleAction, deadline time.Time, ids []string) error {
	return c.Bus.Broadcast(fleetbus.LifecycleRequest{Schema: fleetbus.LifecycleSchema, TransactionID: transactionID, ForestID: f.ID, Generation: f.Generation, Action: action, Deadline: deadline, Capability: "forest.pause", IdempotencyKey: transactionID + ":" + string(action), Authority: f.Authority}, ids)
}
func (c Coordinator) writeAck(r *Report, f Forest, m Member, state fleetbus.LifecycleAckState, reason, checkpoint string, at time.Time) {
	a := fleetbus.LifecycleAck{Schema: fleetbus.LifecycleSchema, TransactionID: r.TransactionID, ForestID: f.ID, MemberID: m.ID, Generation: m.Generation, State: state, Reason: reason, CheckpointRef: checkpoint, ReadbackRef: m.ID + ":readback", At: at}
	_ = c.Bus.WriteAck(a)
	r.Acks = append(r.Acks, a)
}

// MonitorLine renders the one operator-readable line the stall monitor logs for
// every relief sample: it always names the lifecycle transaction ID and the
// read-back captured for it, so an abstention is as auditable as a replacement.
func (r Report) MonitorLine() string {
	transaction, readback := r.TransactionID, strings.Join(r.Readback, ",")
	if transaction == "" {
		transaction = "none"
	}
	if readback == "" {
		readback = "none"
	}
	return fmt.Sprintf("terminal-relief barrier %s transaction=%s forest=%s stops=%d restores=%d readback=%s reason=%s", r.Verdict, transaction, r.ForestID, r.StopCalls, r.RestoreCalls, readback, r.Reason)
}
func activeMembers(in []Member) []Member {
	var out []Member
	for _, m := range in {
		if m.Active {
			out = append(out, m)
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	return out
}
func sameMembers(a, b []Member) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i].ID != b[i].ID || a[i].Generation != b[i].Generation {
			return false
		}
	}
	return true
}

// ManagedProcess is one live descendant of a pressured terminal host, named by
// the image the host started.
type ManagedProcess struct {
	MemberID string
	Image    string
}

// AdapterForImage returns the negotiated lifecycle adapter for a descendant image.
// An unrecognized image gets Unknown, whose failed negotiation is exactly what makes
// the barrier abstain rather than replace a host that owns unmanaged interactive work.
func AdapterForImage(image string) lifecycleadapter.Adapter {
	switch strings.ToLower(strings.TrimSuffix(strings.TrimSpace(image), ".exe")) {
	case "fak":
		return lifecycleadapter.NativeFAK()
	case "codex":
		return lifecycleadapter.Codex()
	case "claude":
		return lifecycleadapter.Claude()
	}
	return lifecycleadapter.Unknown(image)
}

// ForestUnderHost declares the managed forest discovered under a pressured terminal
// host so the barrier can address it as one lifecycle transaction.
func ForestUnderHost(hostPID int, generation uint64, procs []ManagedProcess) Forest {
	f := Forest{ID: fmt.Sprintf("terminal-host-%d", hostPID), Authority: fmt.Sprintf("terminal-relief:%d", hostPID), Generation: generation}
	for _, p := range procs {
		f.Members = append(f.Members, Member{ID: p.MemberID, Generation: generation, Active: true, Adapter: AdapterForImage(p.Image)})
	}
	return f
}
func ForestFromSnapshot(s processforest.Snapshot, adapters map[string]lifecycleadapter.Adapter) Forest {
	f := Forest{ID: s.ForestID, Authority: s.RootAuthority}
	for _, m := range s.Members {
		if m.Generation > f.Generation {
			f.Generation = m.Generation
		}
		f.Members = append(f.Members, Member{ID: m.MemberID, Generation: m.Generation, Active: m.State == processforest.StateActive, Adapter: adapters[m.MemberID]})
	}
	return f
}
