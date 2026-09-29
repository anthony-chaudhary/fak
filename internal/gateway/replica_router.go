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

var ErrReplicaDispatchEmpty = errors.New("gateway: replica dispatch has no replicas")

// PickPolicy is the pluggable placement policy behind ReplicaDispatch.pick(). The
// skeleton supplies the candidate set (the live admissible replicas), the request's
// shared prefix (a leading run of stable segment identities — see prefixSegments),
// and a load function (each candidate's live in-flight count, 0 when unknown); the
// policy returns the chosen replica. Returning ok=false makes the dispatch fall back to
// its built-in pure keyed rendezvous, so a policy is purely additive and never strands
// a request. CacheAwarePolicy is the issue-#41 implementation; nil leaves the dispatch
// policy-free.
type PickPolicy interface {
	Pick(candidates []PlannerReplica, prefix []string, load func(name string) int) (PlannerReplica, bool)
}

// PlannerReplica is one statically-declared upstream in a gateway fleet.
type PlannerReplica struct {
	Name    string
	Planner agent.Planner

	// Endpoint is the upstream dial URL this replica fronts (the operator's
	// --base-url/--replica-base-url value, after a name=URL split). It is carried
	// so the health loop can register the worker with the endpoint it actually
	// probes/dispatches against — WorkerSpec.Endpoint — rather than overloading
	// that field with the replica identity. Empty on a hand-built test replica; the
	// membership then records no endpoint, which the dispatch never reads for placement
	// (it binds by Name == WorkerSpec.ID).
	Endpoint string
}

// ReplicaInfo is the read-only registry view exposed by ReplicaDispatch.
type ReplicaInfo struct {
	Name  string
	Model string
}

// ReplicaDispatch is an agent.Planner that dispatches turns across a fixed replica set.
//
// Placement is a PURE keyed rendezvous over the declared (and, when membership is
// armed, admissible) replica set: the request's shared-prefix identity selects the
// winner, so the same request always lands on the same replica and there is no
// mutable counter to make a decision a function of history (see hrwScore). The
// failover path re-runs the same rendezvous with the failed replica removed, which
// yields the next distinct winner with no cursor.
type ReplicaDispatch struct {
	model    string
	replicas []PlannerReplica

	// membership is the optional live health/drain/failover loop the dispatch reads.
	// When nil the dispatch stays policy-free (pure keyed placement over every
	// replica). When attached (WithMembership), pick() routes only to replicas the
	// loop currently marks admissible — so an unhealthy or draining worker drops out
	// within the health interval — and returns ErrNoHealthyWorker (a typed verdict,
	// never a silent drop) when none is admissible. It is an atomic pointer because
	// the host arms it at Serve — with the gateway already accepting requests — so
	// the publish genuinely races the request-goroutine reads; every read goes
	// through liveMembership().
	membership atomic.Pointer[FleetMembership]

	// fleet is the live membership this dispatch was BUILT against but has not yet armed.
	// newProxyPlanner registers the configured replica roster here so the host can start
	// the health loop on the serve lifecycle context and arm admission ONCE, at Serve —
	// rather than at construction, where a not-yet-probed roster would read as an outage
	// for any request racing the first beat, and where construction would have to do a
	// blocking network probe. WithMembership promotes it to the live membership the
	// placement paths read (see runFleetHealthLoop). nil for a dispatch built without a
	// fleet (a lone upstream, or a hand-built test dispatch).
	fleet *FleetMembership

	// policy is the optional cache-aware placement policy (issue #41). When nil the
	// dispatch keeps its pure keyed pick unchanged; when set (WithPickPolicy), pick()
	// scores the admissible candidates by prefix residency × inverse load and falls
	// back to the keyed rendezvous only if the policy declines. It composes with
	// membership: the candidate set is the admissible subset, and the load function is
	// each admissible worker's live in-flight count.
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

// NewReplicaDispatch builds a static, in-process planner fleet. It is intentionally
// policy-free: later residency/health work can choose smarter placement without changing
// the gateway's Planner seam.
func NewReplicaDispatch(model string, replicas []PlannerReplica) (*ReplicaDispatch, error) {
	if model == "" {
		return nil, errors.New("gateway: replica dispatch model id is empty")
	}
	if len(replicas) == 0 {
		return nil, ErrReplicaDispatchEmpty
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
	return &ReplicaDispatch{model: model, replicas: cp}, nil
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
// rejected by NewReplicaDispatch's duplicate-name check; give same-endpoint replicas the
// explicit name=URL form to keep them distinct.
func deriveReplicaName(rawURL string) string {
	key := strings.TrimSpace(rawURL)
	if u, err := neturl.Parse(key); err == nil && u.Host != "" {
		key = u.Scheme + "://" + u.Host
	}
	sum := sha256.Sum256([]byte(key))
	return "replica-" + hex.EncodeToString(sum[:3])
}

// FleetMembership returns the live membership the dispatch was BUILT against, or nil when
// it was built without one. It is the handle the host hands to the health loop; the
// membership is unarmed until the host promotes it via WithMembership.
func (r *ReplicaDispatch) FleetMembership() *FleetMembership {
	if r == nil {
		return nil
	}
	return r.fleet
}

// WithMembership attaches a live FleetMembership so the dispatch routes only to
// admissible (healthy, non-draining) replicas. A replica is bound to a worker by
// Name == WorkerSpec.ID; a replica absent from membership, still unknown, drained,
// or unhealthy is dropped from the eligible set, and a pick with no admissible
// worker returns ErrNoHealthyWorker instead of falling through to a dead upstream.
// Passing nil restores the policy-free keyed placement over every replica. Returns
// r for chaining. It is safe to call concurrently with routing: the publish is a
// single atomic store the request-goroutine reads observe through liveMembership.
func (r *ReplicaDispatch) WithMembership(m *FleetMembership) *ReplicaDispatch {
	if r == nil {
		return nil
	}
	r.membership.Store(m)
	return r
}

// liveMembership reads the atomically-published membership, or nil when the dispatch
// is nil or no membership has been attached. It is the single read seam every
// placement path uses, so a concurrent WithMembership publish can never be read as a
// torn or stale pointer.
func (r *ReplicaDispatch) liveMembership() *FleetMembership {
	if r == nil {
		return nil
	}
	return r.membership.Load()
}

// WithPickPolicy attaches a cache-aware placement policy (issue #41). pick() then asks
// the policy to choose among the admissible candidates, falling back to the pure
// keyed rendezvous if the policy declines. Passing nil restores the policy-free
// keyed placement. Returns r for chaining (composes with WithMembership).
func (r *ReplicaDispatch) WithPickPolicy(p PickPolicy) *ReplicaDispatch {
	if r == nil {
		return nil
	}
	r.policy = p
	return r
}

func (r *ReplicaDispatch) Model() string {
	if r == nil {
		return ""
	}
	return r.model
}

// Replicas returns a stable snapshot of the static registry.
func (r *ReplicaDispatch) Replicas() []ReplicaInfo {
	if r == nil || len(r.replicas) == 0 {
		return nil
	}
	out := make([]ReplicaInfo, len(r.replicas))
	for i, repl := range r.replicas {
		out[i] = ReplicaInfo{Name: repl.Name, Model: repl.Planner.Model()}
	}
	return out
}

// WalkPlanners traverses each direct child planner in the replica set.
func (r *ReplicaDispatch) WalkPlanners(fn func(agent.Planner)) {
	if r == nil {
		return
	}
	for _, repl := range r.replicas {
		if repl.Planner != nil {
			fn(repl.Planner)
		}
	}
}

func (r *ReplicaDispatch) pickDistinctReplica(primary string) (PlannerReplica, bool) {
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
func (r *ReplicaDispatch) Complete(ctx context.Context, messages []agent.Message, tools []agent.ToolDef, opts ...agent.SampleOpt) (*agent.Completion, error) {
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

func (r *ReplicaDispatch) completeReserved(ctx context.Context, route reservedPlannerReplica, messages []agent.Message, tools []agent.ToolDef, opts ...agent.SampleOpt) (comp *agent.Completion, err error) {
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
	if IsHaltException(err) {
		return false
	}
	return !errors.Is(err, context.Canceled) && !errors.Is(err, context.DeadlineExceeded)
}

func (r *ReplicaDispatch) StreamingSupported() bool {
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

func (r *ReplicaDispatch) CompleteStream(ctx context.Context, sink agent.StreamSink, messages []agent.Message, tools []agent.ToolDef, opts ...agent.SampleOpt) (comp *agent.Completion, err error) {
	route, err := r.reserveForMessages(messages, opts)
	if err != nil {
		return nil, err
	}
	defer func() { route.finish(ctx, comp, err, true) }()
	tried := make(map[string]struct{}, len(r.replicas))
	// emitted counts content fragments already handed to the caller's sink. Once a
	// single fragment has crossed it, the client holds part of an answer and the
	// turn can NOT be retargeted — a replay against another replica would duplicate
	// content on the wire. So the retarget loop below is armed only while the sink
	// is still clean, exactly as the buffered path can retry a turn that wrote
	// nothing. A backend 4xx (405/404/408/429) is returned by the upstream BEFORE
	// any body fragment, so the common failure shape stays retargetable.
	emitted := false
	guarded := func(delta string) error {
		if delta != "" {
			emitted = true
		}
		if sink == nil {
			return nil
		}
		return sink(delta)
	}
	for {
		repl := route.replica
		sp, ok := repl.Planner.(agent.StreamingPlanner)
		if !ok || !sp.StreamingSupported() {
			return nil, agent.ErrStreamingUnsupported
		}
		comp, err = sp.CompleteStream(ctx, guarded, messages, tools, opts...)
		if err == nil {
			return comp, nil
		}
		// A retarget is legal only when nothing was emitted (no partial answer to
		// duplicate) and the reservation can select a fresh replica. This mirrors
		// completeReserved's `route.reservation == nil || !replicaFallbackAllowed`
		// guard; the emitted check is the streaming-only addition.
		if emitted || route.reservation == nil || !replicaFallbackAllowed(ctx, err) {
			return comp, err
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

func (r *ReplicaDispatch) reserveForMessages(messages []agent.Message, opts []agent.SampleOpt) (reservedPlannerReplica, error) {
	// The shared-prefix segment run is the request's stable placement identity: the
	// cache-aware policy keys on it, and the policy-free keyed rendezvous fallback
	// keys on it too (replicaRendezvousKey), so the same request replays to the same
	// replica with no counter. Compute it whenever the dispatch may place blindly as
	// well as when a policy is attached; a request with no messages yields an empty
	// key, which is still deterministic.
	var prefix []string
	if r != nil {
		prefix = prefixSegments(messages)
	}
	return r.reserveWithDecode(prefix, nil, decodeFootprintRouteRequest{ExpectedOutputTokens: sampleMaxTokens(opts)})
}

func (r *ReplicaDispatch) reserve(prefix []string, skip map[string]struct{}) (reservedPlannerReplica, error) {
	return r.reserveWithDecode(prefix, skip, decodeFootprintRouteRequest{})
}

func (r *ReplicaDispatch) reserveWithDecode(prefix []string, skip map[string]struct{}, req decodeFootprintRouteRequest) (reservedPlannerReplica, error) {
	return r.reserveOnEngineWithDecode(prefix, skip, "", req)
}

func (r *ReplicaDispatch) reserveOnEngineWithDecode(prefix []string, skip map[string]struct{}, engine EngineKind, req decodeFootprintRouteRequest) (reservedPlannerReplica, error) {
	if r == nil || len(r.replicas) == 0 {
		return reservedPlannerReplica{}, ErrReplicaDispatchEmpty
	}
	membership := r.liveMembership()
	if membership == nil {
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
			return reservedPlannerReplica{}, ErrReplicaDispatchEmpty
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
	reservation, err := membership.reserveForModel(r.model, engine, allowed, skip, r.reservationPicker(prefix, byName, req, nil, &booking))
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

func (r *ReplicaDispatch) retargetReserved(reservation *fleetReservation, booking *decodeFootprintReservation, prefix []string, skip map[string]struct{}) (PlannerReplica, error) {
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

func (r *ReplicaDispatch) reservationPicker(prefix []string, byName map[string]PlannerReplica, req decodeFootprintRouteRequest, existing *decodeFootprintReservation, capture **decodeFootprintReservation) fleetReservationPick {
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
		workerID, ok := r.keyedCandidate(inflight, prefix)
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

// keyedCandidate is the policy-free placement primitive: a keyed rendezvous over
// the eligible replicas, ordered by hrwScore(prefixKey, name). It is pure and
// stateless — the same (prefix, eligible set) always yields the same winner — and
// it never reads or advances a mutable cursor, so a decision is replayable. The
// candidate map is the set the caller has already filtered (admissible replicas
// the request may still try); an empty map reports false so the caller returns its
// typed verdict rather than a hang. The rendezvous key is the request's shared
// prefix (see prefixSegments); an empty/degenerate prefix still spreads because
// hrwScore hashes the member identity too.
func (r *ReplicaDispatch) keyedCandidate(candidates map[string]int, prefix []string) (string, bool) {
	if len(candidates) == 0 {
		return "", false
	}
	key := replicaRendezvousKey(prefix)
	var best string
	var bestScore uint64
	for name := range candidates {
		score := hrwScore(key, name)
		if best == "" || score > bestScore || (score == bestScore && name < best) {
			best, bestScore = name, score
		}
	}
	return best, best != ""
}

// replicaRendezvousKey folds the request's shared-prefix segment run into the
// stable string the rendezvous hashes against each member's name. An absent prefix
// yields "", which still spreads across members because hrwScore hashes the member
// identity alongside it.
func replicaRendezvousKey(prefix []string) string {
	if len(prefix) == 0 {
		return ""
	}
	return strings.Join(prefix, "\x00")
}

// hrwScore is the gateway's rendezvous hash: 64-bit FNV-1a over key || ':' ||
// memberID with a SplitMix64 avalanche finalizer. The construction is identical to
// the private fabric's routing.BalanceScore / platform/gateway.ComputeHRWScore, so
// a decision computed on either axis for the same identity agrees — the shared
// constant is DUPLICATED with this citation rather than imported across the
// boundary. It is a pure function of its inputs: no clock, no counter, no state.
func hrwScore(key, memberID string) uint64 {
	const (
		offset64 = 14695981039346656037
		prime64  = 1099511628211
	)
	hash := uint64(offset64)
	for i := 0; i < len(key); i++ {
		hash ^= uint64(key[i])
		hash *= prime64
	}
	hash ^= uint64(':')
	hash *= prime64
	for i := 0; i < len(memberID); i++ {
		hash ^= uint64(memberID[i])
		hash *= prime64
	}
	// SplitMix64 avalanche: FNV-1a alone leaves visible bit bias on keys that
	// differ only in a short suffix (the shape of a session/request id), so the
	// finalizer is load-bearing, not decoration.
	hash ^= hash >> 30
	hash *= 0xbf58476d1ce4e5b9
	hash ^= hash >> 27
	hash *= 0x94d049bb133111eb
	hash ^= hash >> 31
	return hash
}

// pick is the cached placement entry point: it asks the attached policy first and
// falls back to the pure keyed rendezvous when no policy is attached or the policy
// declines. handled=false from pickByPolicy means "use the fallback".
func (r *ReplicaDispatch) pick(prefix []string) (PlannerReplica, error) {
	if r == nil || len(r.replicas) == 0 {
		return PlannerReplica{}, ErrReplicaDispatchEmpty
	}
	if r.policy != nil {
		if repl, err, handled := r.pickByPolicy(prefix); handled {
			return repl, err
		}
	}
	return r.pickKeyed(prefix)
}

// pickRoundRobin is the policy-free placement. Despite the historical name it is
// now a KEYED rendezvous, not a rotation: the request's shared-prefix identity
// chooses the replica, so the decision is pure and replayable and no counter is
// consulted. It still returns the typed no-worker verdict (never a dead upstream)
// when membership leaves nothing admissible.
func (r *ReplicaDispatch) pickRoundRobin() (PlannerReplica, error) {
	return r.pickKeyed(nil)
}

// pickKeyed is pickRoundRobin's core, taking the request's shared prefix so the
// winner is a function of request identity. Over every configured replica (or,
// when membership is attached, the admissible subset), it selects the highest
// rendezvous score. A worker that does not hold r.model is filtered out before
// health is even consulted, so a heterogeneous fleet never hands this dispatch's
// request to a worker serving a different model; an unhealthy or draining holder
// drops from the admissible set within the health interval; and when nothing is
// left it returns the typed verdict membership decided rather than route to a
// wrong or dead upstream.
func (r *ReplicaDispatch) pickKeyed(prefix []string) (PlannerReplica, error) {
	if len(r.replicas) == 0 {
		return PlannerReplica{}, ErrReplicaDispatchEmpty
	}
	if r.liveMembership() == nil {
		key := replicaRendezvousKey(prefix)
		var best PlannerReplica
		var bestScore uint64
		for _, repl := range r.replicas {
			score := hrwScore(key, repl.Name)
			if best.Name == "" || score > bestScore || (score == bestScore && repl.Name < best.Name) {
				best, bestScore = repl, score
			}
		}
		return best, nil
	}
	admit, err := r.admitSet()
	if err != nil {
		return PlannerReplica{}, err
	}
	key := replicaRendezvousKey(prefix)
	var best PlannerReplica
	var bestScore uint64
	found := false
	for _, repl := range r.replicas {
		if _, ok := admit[repl.Name]; !ok {
			continue
		}
		score := hrwScore(key, repl.Name)
		if !found || score > bestScore || (score == bestScore && repl.Name < best.Name) {
			best, bestScore, found = repl, score, true
		}
	}
	if !found {
		return PlannerReplica{}, ErrNoHealthyWorker
	}
	return best, nil
}

// pickByPolicy runs the attached cache-aware policy over the admissible candidate set.
// handled=false means the policy declined and the caller should fall back to
// round-robin; handled=true carries the policy's decision (or the typed no-worker
// verdict when membership leaves nothing admissible).
func (r *ReplicaDispatch) pickByPolicy(prefix []string) (repl PlannerReplica, err error, handled bool) {
	candidates, load, cerr := r.candidatesAndLoad()
	if cerr != nil {
		return PlannerReplica{}, cerr, true
	}
	if len(candidates) == 0 {
		if r.liveMembership() != nil {
			return PlannerReplica{}, ErrNoHealthyWorker, true
		}
		return PlannerReplica{}, ErrReplicaDispatchEmpty, true
	}
	if chosen, ok := r.policy.Pick(candidates, prefix, load); ok {
		return chosen, nil, true
	}
	return PlannerReplica{}, nil, false
}

// admitSet is the dispatch's membership read: the ids membership currently offers for
// THIS dispatch's model, with the model filter applied BEFORE the health filter. A nil
// membership yields a nil set (the caller then keeps the blind rotation). The error is
// the typed placement verdict membership decided — ErrNoWorkerForModel when the roster
// holds no worker for r.model (a heterogeneous fleet that simply does not carry this
// model), ErrNoHealthyWorker when a holder exists but none is admissible. Routing that
// distinction up unchanged is the point: a config mistake must not read as an outage.
func (r *ReplicaDispatch) admitSet() (map[string]struct{}, error) {
	membership := r.liveMembership()
	if membership == nil {
		return nil, nil
	}
	adm, err := membership.CandidatesForModel(r.model)
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
func (r *ReplicaDispatch) candidatesAndLoad() ([]PlannerReplica, func(string) int, error) {
	membership := r.liveMembership()
	if membership == nil {
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
	for _, st := range membership.Snapshot() {
		inflight[st.Spec.ID] = st.Inflight
	}
	return candidates, func(name string) int { return inflight[name] }, nil
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
