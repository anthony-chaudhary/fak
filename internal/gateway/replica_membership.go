package gateway

// replica_membership.go — production wiring of the live health/drain registry
// (FleetMembership) into the ReplicaRouter (issue fak-private#2417, gap 1). The router
// has always been a blind round-robin over CONFIGURED replicas; here the gateway builds
// a membership over exactly those replicas and runs its health loop on the serve
// lifecycle context, so a dead upstream drops out of rotation and traffic fails over to
// a healthy peer. The loop is armed only once the first beat has been applied (see
// runFleetHealthLoop): until then the router keeps the blind round-robin it has always
// had, so construction does no network I/O and a request that races gateway startup
// reads the configured fleet, not a not-yet-probed outage. The loop is a supervised
// in-kernel bgloop ("fleet-health"), so startLoops starts it and stopLoops joins it.

import (
	"context"
	"time"
)

const (
	// fleetHealthInterval is how often the membership health loop probes each replica
	// once the gateway is serving. It is deliberately longer than one upstream round-trip
	// and shorter than a client's retry patience, so a dead replica leaves the rotation
	// promptly without hammering healthy peers.
	fleetHealthInterval = 5 * time.Second

	// replicaProbeTimeout bounds one readiness probe so a hung upstream cannot stall the
	// health tick (or the loop's parallel siblings) indefinitely; three seconds is well
	// inside fleetHealthInterval.
	replicaProbeTimeout = 3 * time.Second
)

// fleetProber is the narrow readiness surface a planner must expose to serve as a fleet
// health probe. *agent.HTTPPlanner implements it via ProbeReachability (a bounded,
// zero-generation OPTIONS against the exact configured provider endpoint).
type fleetProber interface {
	ProbeReachability(ctx context.Context) (int, error)
}

// newReplicaHealthProbe builds the Probe the fleet registry calls once per replica per
// tick. It reuses the production upstream readiness check — (*agent.HTTPPlanner).
// ProbeReachability — keyed on the SAME identity the router binds workers by
// (WorkerSpec.ID == PlannerReplica.Name), so a probe result lands on the replica the
// router would place. A replica whose planner does not expose the readiness check is
// never reported healthy (fail-closed). The probe is nil-safe and respects ctx, wrapping
// each call in replicaProbeTimeout so a hung upstream cannot stall the tick.
func newReplicaHealthProbe(replicas []PlannerReplica) Probe {
	probers := make(map[string]fleetProber, len(replicas))
	for _, repl := range replicas {
		if pr, ok := repl.Planner.(fleetProber); ok {
			probers[repl.Name] = pr
		}
	}
	return func(ctx context.Context, spec WorkerSpec) bool {
		pr := probers[spec.ID]
		if pr == nil {
			return false
		}
		pctx, cancel := context.WithTimeout(ctx, replicaProbeTimeout)
		defer cancel()
		_, err := pr.ProbeReachability(pctx)
		return err == nil
	}
}

// buildReplicaMembership registers one membership worker per configured replica, bound
// by the SAME identity the router uses (PlannerReplica.Name), and returns the registry
// with its health probe installed. Each worker declares the router's model, so the
// router's model-before-health filter (servesModel) admits it for this router's requests
// and a heterogeneous fleet is describable; an empty model normalizes to unconstrained,
// which is the fail-open reading that keeps an unmodeled deployment unchanged. The
// membership is returned UNARMED (see ReplicaRouter.fleet) — the host arms it at Serve
// after the first probe.
func buildReplicaMembership(replicas []PlannerReplica, model string) (*FleetMembership, error) {
	fm := NewFleetMembership(MembershipConfig{Probe: newReplicaHealthProbe(replicas)})
	for _, repl := range replicas {
		if err := fm.Add(WorkerSpec{
			ID:       repl.Name,
			Endpoint: repl.Endpoint,
			Models:   []string{model},
		}); err != nil {
			return nil, err
		}
	}
	return fm, nil
}

// runFleetHealthLoop is the fleet-health loop body the in-kernel supervisor owns
// (registered as "fleet-health" in newBgloopSupervisor). It arms the replica router's
// live membership and drives its health loop on the loop's lifecycle context — the
// same context startLoops derives from the serve context, so stopLoops cancels and
// joins it, and /v1/fak/loops shows it. It is a no-op when no fleet was wired (a lone
// upstream, the in-kernel model, or the mock path) so those deployments are
// byte-for-byte unchanged, and it returns when ctx is done. The first beat is applied
// SYNCHRONOUSLY before the router is armed, so admission reflects a real probe from the
// first routed turn rather than the all-unknown state a fresh registry starts in; the
// router stays on its blind round-robin until that probe returns, so a request racing
// arm-time still reads the configured fleet, never an outage. Each probe is bounded
// (replicaProbeTimeout) and ctx-honoring, so a hung upstream cannot wedge shutdown.
//
// The supervisor runs this body as a CONTINUOUS loop (Interval 0): RunHealthLoop paces
// itself with its own ticker and blocks until ctx is done, which is exactly the
// continuous contract bgloop.Loop documents — the Tick owns its own pacing.
func (s *Server) runFleetHealthLoop(ctx context.Context) error {
	if s == nil {
		return nil
	}
	s.fleetMu.Lock()
	fm := s.fleet
	s.fleetMu.Unlock()
	if fm == nil {
		return nil
	}
	fm.ProbeOnce(ctx)
	if s.fleetRouter != nil {
		s.fleetRouter.WithMembership(fm)
	}
	fm.RunHealthLoop(ctx, fleetHealthInterval)
	return nil
}
