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
	idx := int((r.next.Add(1) - 1) % uint64(len(r.replicas)))
	return r.replicas[idx], nil
}
