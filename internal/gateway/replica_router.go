package gateway

import (
	"context"
	"errors"
	"fmt"
	"sync/atomic"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

var ErrReplicaRouterEmpty = errors.New("gateway: replica router has no replicas")

// PlannerReplica is one statically-declared upstream in a gateway fleet.
type PlannerReplica struct {
	Name    string
	Planner agent.Planner
}

// ReplicaInfo is the read-only registry view exposed by ReplicaRouter.
type ReplicaInfo struct {
	Name  string
	Model string
}

// ReplicaRouter is an agent.Planner that dispatches turns across a fixed replica set.
type ReplicaRouter struct {
	model    string
	replicas []PlannerReplica
	next     atomic.Uint64

	// membership is the optional live health/drain/failover loop the router reads.
	// When nil the router stays policy-free (blind round-robin over every replica).
	// When attached (WithMembership), pick() routes only to replicas the loop
	// currently marks admissible — so an unhealthy or draining worker drops out of
	// the rotation within the health interval — and returns ErrNoHealthyWorker (a
	// typed verdict, never a silent drop) when none is admissible.
	membership *FleetMembership
}

// NewReplicaRouter builds a static, in-process planner fleet. It is intentionally
// policy-free: later residency/health work can choose smarter placement without changing
// the gateway's Planner seam.
func NewReplicaRouter(model string, replicas []PlannerReplica) (*ReplicaRouter, error) {
	if model == "" {
		return nil, errors.New("gateway: replica router model id is empty")
	}
	if len(replicas) == 0 {
		return nil, ErrReplicaRouterEmpty
	}
	seen := make(map[string]struct{}, len(replicas))
	cp := make([]PlannerReplica, len(replicas))
	for i, repl := range replicas {
		if repl.Name == "" {
			return nil, fmt.Errorf("gateway: replica %d has an empty name", i)
		}
		if repl.Planner == nil {
			return nil, fmt.Errorf("gateway: replica %q has nil planner", repl.Name)
		}
		if _, ok := seen[repl.Name]; ok {
			return nil, fmt.Errorf("gateway: duplicate replica name %q", repl.Name)
		}
		seen[repl.Name] = struct{}{}
		cp[i] = repl
	}
	return &ReplicaRouter{model: model, replicas: cp}, nil
}

// WithMembership attaches a live FleetMembership so the router routes only to
// admissible (healthy, non-draining) replicas. A replica is bound to a worker by
// Name == WorkerSpec.ID; a replica absent from membership, still unknown, drained,
// or unhealthy is dropped from the rotation, and a pick with no admissible worker
// returns ErrNoHealthyWorker instead of falling through to a dead upstream. Passing
// nil restores the policy-free blind round-robin. Returns r for chaining.
func (r *ReplicaRouter) WithMembership(m *FleetMembership) *ReplicaRouter {
	if r == nil {
		return nil
	}
	r.membership = m
	return r
}

func (r *ReplicaRouter) Model() string {
	if r == nil {
		return ""
	}
	return r.model
}

// Replicas returns a stable snapshot of the static registry.
func (r *ReplicaRouter) Replicas() []ReplicaInfo {
	if r == nil || len(r.replicas) == 0 {
		return nil
	}
	out := make([]ReplicaInfo, len(r.replicas))
	for i, repl := range r.replicas {
		out[i] = ReplicaInfo{Name: repl.Name, Model: repl.Planner.Model()}
	}
	return out
}

func (r *ReplicaRouter) Complete(ctx context.Context, messages []agent.Message, tools []agent.ToolDef, opts ...agent.SampleOpt) (*agent.Completion, error) {
	repl, err := r.pick()
	if err != nil {
		return nil, err
	}
	return repl.Planner.Complete(ctx, messages, tools, opts...)
}

func (r *ReplicaRouter) StreamingSupported() bool {
	if r == nil || len(r.replicas) == 0 {
		return false
	}
	for _, repl := range r.replicas {
		sp, ok := repl.Planner.(agent.StreamingPlanner)
		if !ok || !sp.StreamingSupported() {
			return false
		}
	}
	return true
}

func (r *ReplicaRouter) CompleteStream(ctx context.Context, sink agent.StreamSink, messages []agent.Message, tools []agent.ToolDef, opts ...agent.SampleOpt) (*agent.Completion, error) {
	repl, err := r.pick()
	if err != nil {
		return nil, err
	}
	sp, ok := repl.Planner.(agent.StreamingPlanner)
	if !ok || !sp.StreamingSupported() {
		return nil, agent.ErrStreamingUnsupported
	}
	return sp.CompleteStream(ctx, sink, messages, tools, opts...)
}

func (r *ReplicaRouter) pick() (PlannerReplica, error) {
	if r == nil || len(r.replicas) == 0 {
		return PlannerReplica{}, ErrReplicaRouterEmpty
	}
	n := uint64(len(r.replicas))
	start := r.next.Add(1) - 1 // advance the shared cursor exactly once per pick
	if r.membership == nil {
		return r.replicas[int(start%n)], nil
	}
	// Membership-gated: round-robin only over the replicas the live health/drain
	// loop currently admits, scanning forward from the cursor so picks still spread
	// across the admissible subset. An unhealthy or draining worker is simply not in
	// the set, so it drops from the rotation within the health interval; if nothing
	// is admissible we return the typed verdict rather than route to a dead upstream.
	adm := r.membership.Admissible()
	admit := make(map[string]struct{}, len(adm))
	for _, spec := range adm {
		admit[spec.ID] = struct{}{}
	}
	for i := uint64(0); i < n; i++ {
		repl := r.replicas[int((start+i)%n)]
		if _, ok := admit[repl.Name]; ok {
			return repl, nil
		}
	}
	return PlannerReplica{}, ErrNoHealthyWorker
}
