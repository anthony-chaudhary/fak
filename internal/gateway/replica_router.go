package gateway

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	neturl "net/url"
	"strings"
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

	// Hedge is default-off and applies only to buffered Complete calls.
	Hedge *HedgePolicy
}

type reservedPlannerReplica struct {
	replica        PlannerReplica
	reservation    *fleetReservation
	prefix         []string
	decode         *decodeFootprintReservation
	decodeDecision DecodeFootprintRouteDecision
}

func (r reservedPlannerReplica) Release() {
	if r.decode != nil {
		r.decode.releaseOnce("released")
	}
	if r.reservation != nil {
		r.reservation.Release()
	}
}

func (r reservedPlannerReplica) finish(ctx context.Context, comp *agent.Completion, err error, stream bool) {
	if r.decode != nil {
		if comp != nil {
			r.decode.reconcileObserved(comp.Usage.CompletionTokens)
		}
		reason := "completion"
		switch {
		case (ctx != nil && ctx.Err() != nil) || errors.Is(err, context.Canceled):
			reason = "cancellation"
		case stream && err == nil:
			reason = "stream_completion"
		case err != nil:
			reason = "error"
		}
		r.decode.releaseOnce(reason)
	}
	if r.reservation != nil {
		r.reservation.Release()
	}
}

func (r reservedPlannerReplica) currentDecodeDecision() (DecodeFootprintRouteDecision, bool) {
	if r.decode == nil {
		return DecodeFootprintRouteDecision{}, false
	}
	return r.decode.decisionSnapshot()
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

// parseReplicaEntry splits an optional operator-chosen identity from a
// --replica-base-url value. "name=URL" yields (name, URL) so an operator can pin a
// stable id that survives a URL change (a live re-base's "same replica, new URL"); a
// bare "URL" yields ("", URL) and the caller derives the identity from the endpoint
// (deriveReplicaName). The split is robust to '=' inside a URL query string: the left
// side is taken as a name ONLY when it carries no ':' or '/', so "http://h/v1?k=v" stays
// one whole URL and never a spurious name "http://h/v1?k".
func parseReplicaEntry(raw string) (name, url string) {
	raw = strings.TrimSpace(raw)
	if i := strings.IndexByte(raw, '='); i > 0 {
		if lhs := raw[:i]; !strings.ContainsAny(lhs, ":/") {
			return strings.TrimSpace(lhs), strings.TrimSpace(raw[i+1:])
		}
	}
	return "", raw
}

// deriveReplicaName is the default, order-independent replica identity: replica-<6 hex>
// over the endpoint's scheme://host (host carries the port), so the SAME upstream keeps
// ONE identity — and thus one set of /metrics transition labels and one residency-index
// slot — regardless of its position in the flag list or a membership change that drops a
// peer (#3968). Positional replica-N naming silently reassigned identity on any reorder
// or removal: dropping URL 1 of 3 renamed the survivors and restarted their counters.
// Two entries that resolve to the same endpoint collide on this derived name and are
// rejected by NewReplicaRouter's duplicate-name check; give same-endpoint replicas the
// explicit name=URL form to keep them distinct.
func deriveReplicaName(rawURL string) string {
	key := strings.TrimSpace(rawURL)
	if u, err := neturl.Parse(key); err == nil && u.Host != "" {
		key = u.Scheme + "://" + u.Host
	}
	sum := sha256.Sum256([]byte(key))
	return "replica-" + hex.EncodeToString(sum[:3])
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

func (r *ReplicaRouter) pickDistinctReplica(primary string) (PlannerReplica, bool) {
	admit, err := r.admitSet()
	if err != nil {
		return PlannerReplica{}, false
	}
	for _, candidate := range r.replicas {
		if candidate.Name == primary {
			continue
		}
		if admit != nil {
			if _, ok := admit[candidate.Name]; !ok {
				continue
			}
		}
		return candidate, true
	}
	return PlannerReplica{}, false
}
func (r *ReplicaRouter) Complete(ctx context.Context, messages []agent.Message, tools []agent.ToolDef, opts ...agent.SampleOpt) (*agent.Completion, error) {
	route, err := r.reserveForMessages(messages, opts)
	if err != nil {
		return nil, err
	}
	repl := route.replica
	if r.Hedge == nil {
		return r.completeReserved(ctx, route, messages, tools, opts...)
	}
	if reason := hedgeIneligibility(r.Hedge, len(r.replicas), tools, opts); reason != "" {
		r.observeHedgeAbstention(repl, reason)
		return r.completeReserved(ctx, route, messages, tools, opts...)
	}
	return r.completeHedged(ctx, route, messages, tools, opts...)
}

func (r *ReplicaRouter) completeReserved(ctx context.Context, route reservedPlannerReplica, messages []agent.Message, tools []agent.ToolDef, opts ...agent.SampleOpt) (comp *agent.Completion, err error) {
	defer func() { route.finish(ctx, comp, err, false) }()
	tried := make(map[string]struct{}, len(r.replicas))
	for {
		comp, err = route.replica.Planner.Complete(ctx, messages, tools, opts...)
		if err == nil {
			return comp, nil
		}
		if route.reservation == nil || !replicaFallbackAllowed(ctx, err) {
			return nil, err
		}
		tried[route.replica.Name] = struct{}{}
		next, retargetErr := r.retargetReserved(route.reservation, route.decode, route.prefix, tried)
		if retargetErr != nil {
			if errors.Is(retargetErr, ErrNoWorkerForModel) {
				return nil, retargetErr
			}
			return nil, fmt.Errorf("%w: every admissible worker failed: %w", ErrNoHealthyWorker, err)
		}
		route.replica = next
		if decision, ok := route.currentDecodeDecision(); ok {
			route.decodeDecision = decision
		}
	}
}

func replicaFallbackAllowed(ctx context.Context, err error) bool {
	if err == nil {
		return false
	}
	if ctx != nil && ctx.Err() != nil {
		return false
	}
	return !errors.Is(err, context.Canceled) && !errors.Is(err, context.DeadlineExceeded)
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

func (r *ReplicaRouter) CompleteStream(ctx context.Context, sink agent.StreamSink, messages []agent.Message, tools []agent.ToolDef, opts ...agent.SampleOpt) (comp *agent.Completion, err error) {
	route, err := r.reserveForMessages(messages, opts)
	if err != nil {
		return nil, err
	}
	defer func() { route.finish(ctx, comp, err, true) }()
	repl := route.replica
	sp, ok := repl.Planner.(agent.StreamingPlanner)
	if !ok || !sp.StreamingSupported() {
		return nil, agent.ErrStreamingUnsupported
	}
	comp, err = sp.CompleteStream(ctx, sink, messages, tools, opts...)
	return comp, err
}

func (r *ReplicaRouter) reserveForMessages(messages []agent.Message, opts []agent.SampleOpt) (reservedPlannerReplica, error) {
	var prefix []string
	if r != nil && r.policy != nil {
		prefix = prefixSegments(messages)
	}
	return r.reserveWithDecode(prefix, nil, decodeFootprintRouteRequest{ExpectedOutputTokens: sampleMaxTokens(opts)})
}

func (r *ReplicaRouter) reserve(prefix []string, skip map[string]struct{}) (reservedPlannerReplica, error) {
	return r.reserveWithDecode(prefix, skip, decodeFootprintRouteRequest{})
}

func (r *ReplicaRouter) reserveWithDecode(prefix []string, skip map[string]struct{}, req decodeFootprintRouteRequest) (reservedPlannerReplica, error) {
	return r.reserveOnEngineWithDecode(prefix, skip, "", req)
}

func (r *ReplicaRouter) reserveOnEngineWithDecode(prefix []string, skip map[string]struct{}, engine EngineKind, req decodeFootprintRouteRequest) (reservedPlannerReplica, error) {
	if r == nil || len(r.replicas) == 0 {
		return reservedPlannerReplica{}, ErrReplicaRouterEmpty
	}
	if r.membership == nil {
		candidates := r.replicas
		if len(skip) > 0 {
			candidates = make([]PlannerReplica, 0, len(r.replicas))
			for _, repl := range r.replicas {
				if _, excluded := skip[repl.Name]; !excluded {
					candidates = append(candidates, repl)
				}
			}
		}
		if len(candidates) == 0 {
			return reservedPlannerReplica{}, ErrReplicaRouterEmpty
		}
		if policy, ok := r.policy.(decodeFootprintPickPolicy); ok {
			if repl, booking, picked := policy.reserveDecodeFootprint(candidates, prefix, nil, nil, req); picked {
				booking.setIdentity(engine, r.model)
				decision, _ := booking.decisionSnapshot()
				return reservedPlannerReplica{replica: repl, prefix: prefix, decode: booking, decodeDecision: decision}, nil
			}
		}
		if len(skip) > 0 {
			return reservedPlannerReplica{replica: candidates[0], prefix: prefix}, nil
		}
		repl, err := r.pick(prefix)
		return reservedPlannerReplica{replica: repl, prefix: prefix}, err
	}
	byName := make(map[string]PlannerReplica, len(r.replicas))
	allowed := make(map[string]struct{}, len(r.replicas))
	for _, repl := range r.replicas {
		byName[repl.Name] = repl
		allowed[repl.Name] = struct{}{}
	}
	var booking *decodeFootprintReservation
	reservation, err := r.membership.reserveForModel(r.model, engine, allowed, skip, r.reservationPicker(prefix, byName, req, nil, &booking))
	if err != nil {
		if booking != nil {
			booking.releaseOnce("booking_failure")
		}
		return reservedPlannerReplica{}, err
	}
	workerID := reservation.WorkerID()
	repl, ok := byName[workerID]
	if !ok {
		reservation.Release()
		if booking != nil {
			booking.releaseOnce("booking_failure")
		}
		return reservedPlannerReplica{}, ErrNoHealthyWorker
	}
	if booking != nil {
		booking.setIdentity(reservation.Engine(), r.model)
	}
	decision, _ := booking.decisionSnapshot()
	return reservedPlannerReplica{replica: repl, reservation: reservation, prefix: prefix, decode: booking, decodeDecision: decision}, nil
}

func (r *ReplicaRouter) retargetReserved(reservation *fleetReservation, booking *decodeFootprintReservation, prefix []string, skip map[string]struct{}) (PlannerReplica, error) {
	byName := make(map[string]PlannerReplica, len(r.replicas))
	allowed := make(map[string]struct{}, len(r.replicas))
	for _, repl := range r.replicas {
		byName[repl.Name] = repl
		allowed[repl.Name] = struct{}{}
	}
	workerID, err := reservation.Retarget(r.model, allowed, skip, r.reservationPicker(prefix, byName, decodeFootprintRouteRequest{}, booking, nil))
	if err != nil {
		return PlannerReplica{}, err
	}
	repl, ok := byName[workerID]
	if !ok {
		reservation.Release()
		return PlannerReplica{}, ErrNoHealthyWorker
	}
	if booking != nil {
		booking.setIdentity(reservation.Engine(), r.model)
	}
	return repl, nil
}

func (r *ReplicaRouter) reservationPicker(prefix []string, byName map[string]PlannerReplica, req decodeFootprintRouteRequest, existing *decodeFootprintReservation, capture **decodeFootprintReservation) fleetReservationPick {
	return func(statuses []WorkerStatus) (fleetReservationPickResult, bool) {
		candidates := make([]PlannerReplica, 0, len(statuses))
		inflight := make(map[string]int, len(statuses))
		bookedOutput := make(map[string]int, len(statuses))
		for _, status := range statuses {
			repl, ok := byName[status.Spec.ID]
			if !ok {
				continue
			}
			candidates = append(candidates, repl)
			inflight[repl.Name] = status.Inflight
			bookedOutput[repl.Name] = status.BookedOutputBlocks
		}
		if len(candidates) == 0 {
			return fleetReservationPickResult{}, false
		}
		if policy, ok := r.policy.(decodeFootprintPickPolicy); ok {
			load := func(name string) int { return inflight[name] }
			booked := func(name string) int { return bookedOutput[name] }
			var chosen PlannerReplica
			var booking *decodeFootprintReservation
			var picked bool
			if existing != nil {
				chosen, picked = policy.retargetDecodeFootprint(existing, candidates, prefix, load, booked)
			} else {
				chosen, booking, picked = policy.reserveDecodeFootprint(candidates, prefix, load, booked, req)
			}
			if picked {
				if capture != nil {
					*capture = booking
				}
				if _, valid := inflight[chosen.Name]; valid {
					return fleetReservationPickResult{workerID: chosen.Name, bookedOutputBlocks: bookingBlocks(booking, existing)}, true
				}
				if booking != nil {
					booking.releaseOnce("booking_failure")
				}
			}
		}
		if r.policy != nil {
			if chosen, ok := r.policy.Pick(candidates, prefix, func(name string) int { return inflight[name] }); ok {
				if _, valid := inflight[chosen.Name]; valid {
					return fleetReservationPickResult{workerID: chosen.Name}, true
				}
			}
		}
		workerID, ok := r.roundRobinCandidate(inflight)
		return fleetReservationPickResult{workerID: workerID}, ok
	}
}

func bookingBlocks(created, existing *decodeFootprintReservation) int {
	booking := created
	if booking == nil {
		booking = existing
	}
	return booking.bookedBlocks()
}

func (r *ReplicaRouter) roundRobinCandidate(candidates map[string]int) (string, bool) {
	if len(candidates) == 0 || len(r.replicas) == 0 {
		return "", false
	}
	n := uint64(len(r.replicas))
	start := r.next.Add(1) - 1
	for i := uint64(0); i < n; i++ {
		name := r.replicas[int((start+i)%n)].Name
		if _, ok := candidates[name]; ok {
			return name, true
		}
	}
	return "", false
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
	candidates, load, cerr := r.candidatesAndLoad()
	if cerr != nil {
		return PlannerReplica{}, cerr, true
	}
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

// admitSet is the router's membership read: the ids membership currently offers for
// THIS router's model, with the model filter applied BEFORE the health filter. A nil
// membership yields a nil set (the caller then keeps the blind rotation). The error is
// the typed placement verdict membership decided — ErrNoWorkerForModel when the roster
// holds no worker for r.model (a heterogeneous fleet that simply does not carry this
// model), ErrNoHealthyWorker when a holder exists but none is admissible. Routing that
// distinction up unchanged is the point: a config mistake must not read as an outage.
func (r *ReplicaRouter) admitSet() (map[string]struct{}, error) {
	if r.membership == nil {
		return nil, nil
	}
	adm, err := r.membership.CandidatesForModel(r.model)
	if err != nil {
		return nil, err
	}
	admit := make(map[string]struct{}, len(adm))
	for _, spec := range adm {
		admit[spec.ID] = struct{}{}
	}
	return admit, nil
}

// candidatesAndLoad returns the replicas the policy may place on (every replica, or —
// when membership is attached — only the subset admissible FOR THIS ROUTER'S MODEL)
// plus a load function that reports each worker's live in-flight count (nil when there
// is no membership to read load from, so the policy scores on residency alone). A
// non-nil error is membership's typed verdict and is returned to the caller as-is.
func (r *ReplicaRouter) candidatesAndLoad() ([]PlannerReplica, func(string) int, error) {
	if r.membership == nil {
		return r.replicas, nil, nil
	}
	admit, err := r.admitSet()
	if err != nil {
		return nil, nil, err
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
	return candidates, func(name string) int { return inflight[name] }, nil
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
	// loop currently admits FOR THIS ROUTER'S MODEL, scanning forward from the cursor
	// so picks still spread across the admissible subset. A worker that does not hold
	// r.model is filtered out before health is even consulted, so a heterogeneous
	// fleet never hands this router's request to a worker serving a different model;
	// an unhealthy or draining holder drops from the rotation within the health
	// interval; and when nothing is left we return the typed verdict membership
	// decided rather than route to a wrong or dead upstream.
	admit, err := r.admitSet()
	if err != nil {
		return PlannerReplica{}, err
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
