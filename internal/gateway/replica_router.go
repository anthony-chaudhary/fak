package gateway

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"sync/atomic"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

var ErrReplicaRouterEmpty = errors.New("gateway: replica router has no replicas")

// PickPolicy is the pluggable placement policy behind ReplicaRouter.pick(). The
// skeleton supplies the candidate set (the live admissible replicas), the request's
// shared prefix (a leading run of stable segment identities — see prefixSegments),
// and a load function (each candidate's live in-flight count, 0 when unknown); the
// policy returns the chosen replica. Returning ok=false makes the router fall back to
// its built-in round-robin, so a policy is purely additive and never strands a request.
// CacheAwarePolicy is the issue-#41 implementation; nil leaves the router policy-free.
type PickPolicy interface {
	Pick(candidates []PlannerReplica, prefix []string, load func(name string) int) (PlannerReplica, bool)
}

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

	// policy is the optional cache-aware placement policy (issue #41). When nil the
	// router keeps its round-robin pick unchanged; when set (WithPickPolicy), pick()
	// scores the admissible candidates by prefix residency × inverse load and falls
	// back to round-robin only if the policy declines. It composes with membership:
	// the candidate set is the admissible subset, and the load function is each
	// admissible worker's live in-flight count.
	policy PickPolicy
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

// WithPickPolicy attaches a cache-aware placement policy (issue #41). pick() then asks
// the policy to choose among the admissible candidates, falling back to round-robin if
// the policy declines. Passing nil restores the policy-free round-robin. Returns r for
// chaining (composes with WithMembership).
func (r *ReplicaRouter) WithPickPolicy(p PickPolicy) *ReplicaRouter {
	if r == nil {
		return nil
	}
	r.policy = p
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
	repl, err := r.pickForMessages(messages)
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
	repl, err := r.pickForMessages(messages)
	if err != nil {
		return nil, err
	}
	sp, ok := repl.Planner.(agent.StreamingPlanner)
	if !ok || !sp.StreamingSupported() {
		return nil, agent.ErrStreamingUnsupported
	}
	return sp.CompleteStream(ctx, sink, messages, tools, opts...)
}

// pickForMessages derives the request's shared prefix from its messages (only when a
// cache-aware policy is attached — otherwise it is wasted work) and picks a replica.
func (r *ReplicaRouter) pickForMessages(messages []agent.Message) (PlannerReplica, error) {
	if r != nil && r.policy != nil {
		return r.pick(prefixSegments(messages))
	}
	return r.pick(nil)
}

func (r *ReplicaRouter) pick(prefix []string) (PlannerReplica, error) {
	if r == nil || len(r.replicas) == 0 {
		return PlannerReplica{}, ErrReplicaRouterEmpty
	}
	if r.policy != nil {
		if repl, err, handled := r.pickByPolicy(prefix); handled {
			return repl, err
		}
	}
	return r.pickRoundRobin()
}

// pickByPolicy runs the attached cache-aware policy over the admissible candidate set.
// handled=false means the policy declined and the caller should fall back to
// round-robin; handled=true carries the policy's decision (or the typed no-worker
// verdict when membership leaves nothing admissible).
func (r *ReplicaRouter) pickByPolicy(prefix []string) (repl PlannerReplica, err error, handled bool) {
	candidates, load := r.candidatesAndLoad()
	if len(candidates) == 0 {
		if r.membership != nil {
			return PlannerReplica{}, ErrNoHealthyWorker, true
		}
		return PlannerReplica{}, ErrReplicaRouterEmpty, true
	}
	if chosen, ok := r.policy.Pick(candidates, prefix, load); ok {
		return chosen, nil, true
	}
	return PlannerReplica{}, nil, false
}

// candidatesAndLoad returns the replicas the policy may place on (every replica, or —
// when membership is attached — only the admissible subset) plus a load function that
// reports each worker's live in-flight count (nil when there is no membership to read
// load from, so the policy scores on residency alone).
func (r *ReplicaRouter) candidatesAndLoad() ([]PlannerReplica, func(string) int) {
	if r.membership == nil {
		return r.replicas, nil
	}
	adm := r.membership.Admissible()
	admit := make(map[string]struct{}, len(adm))
	for _, spec := range adm {
		admit[spec.ID] = struct{}{}
	}
	candidates := make([]PlannerReplica, 0, len(r.replicas))
	for _, repl := range r.replicas {
		if _, ok := admit[repl.Name]; ok {
			candidates = append(candidates, repl)
		}
	}
	inflight := make(map[string]int, len(candidates))
	for _, st := range r.membership.Snapshot() {
		inflight[st.Spec.ID] = st.Inflight
	}
	return candidates, func(name string) int { return inflight[name] }
}

// pickRoundRobin is the policy-free placement: round-robin over every replica, or —
// when membership is attached — over the admissible subset, returning the typed
// no-worker verdict rather than routing to a dead upstream. This is the router's
// behavior whenever no cache-aware policy is attached or the policy declines.
func (r *ReplicaRouter) pickRoundRobin() (PlannerReplica, error) {
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

// prefixSegments lowers a request's messages into the shared-prefix segment run the
// residency index keys on: one stable digest per leading message (role + content), in
// order. Two requests that share a leading conversation (the same system prompt /
// agent scaffold / early turns) share that many leading segments, so the index's
// longest-common-prefix is their reusable-KV overlap — the gateway-level analogue of a
// token-block prefix, derived without a tokenizer in the routing path.
func prefixSegments(messages []agent.Message) []string {
	if len(messages) == 0 {
		return nil
	}
	segs := make([]string, len(messages))
	for i, m := range messages {
		sum := sha256.Sum256([]byte(m.Role + "\x00" + m.Content))
		segs[i] = hex.EncodeToString(sum[:12])
	}
	return segs
}
