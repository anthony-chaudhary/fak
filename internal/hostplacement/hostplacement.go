// Package hostplacement is a clean-room Go primitive for deterministic
// multi-host worker placement: a per-host headroom registry plus a pure
// placement function that picks the least-saturated, non-stale host to spill a
// dispatch worker onto, or falls back to staying local when no host is eligible.
//
// It is the smallest valuable slice of #3599 (multi-host worker placement).
// Dispatch is single-host today: a wave targets the local box only, so the only
// lever to 100x — more boxes — is unreachable. This package breaks the
// single-box ceiling with two pieces:
//
//   - A per-host headroom HEARTBEAT record {Hostname, LiveHeadroom, Saturation,
//     TS} with a staleness TTL, and a Registry that folds the latest heartbeat
//     per host (newest TS wins).
//   - A pure Place function: given a set of heartbeats, a saturation threshold,
//     a "now" timestamp, and a TTL, it picks the least-saturated host that is
//     (a) strictly below its saturation threshold and (b) fresh (TS within TTL
//     of now), or returns a "stay local" decision when none qualify.
//
// It is a PURE primitive over injected inputs: the caller passes now and the TTL
// explicitly, there is no wall clock, no real hostname discovery, and no
// ssh/exec. That keeps the placement decision trivially deterministic (the DoD
// table test advances "time" simply by passing later timestamps) and immune to
// the heavy peer WIP breaking cmd/fak / internal/dispatchtick.
//
// Intended landing site (NOT wired in this pass — the registry plus placement
// function plus its test is the shippable unit; wiring the live tick touches
// contended shared files and risks the build):
//
//	internal/dispatchtick/dispatchtick.go — the worker-launch seam
//	  (BuildWorkerCommand / GuardedLaunchCommand over WorkerLaunch). A live
//	  wiring reads a configured FAK_DISPATCH_HOSTS set, folds each host's
//	  heartbeat into a Registry, and calls Place before launching: an eligible
//	  remote host routes the worker to a RemoteHost executor (a thin seam for the
//	  second box), while StayLocal preserves today's single-host behaviour
//	  unchanged when no host is configured or eligible.
//
// Signals the live heartbeat derives from (per the ticket):
//
//	LiveHeadroom ← accounts_headroom (cmd/fak/accounts_headroom.go): how much
//	  offerable, non-throttled seat capacity a host still has.
//	Saturation   ← procguard: the fraction of the host's local worker capacity
//	  already consumed (process/handle/thread pressure, the #3153 sprawl lever).
package hostplacement

import (
	"sort"
	"time"
)

// Heartbeat is one host's most recent headroom report. It is the durable fact a
// host publishes to the shared registry; placement reads only these fields, so a
// host is fully described by its latest heartbeat.
type Heartbeat struct {
	// Hostname identifies the host. It is the registry key and the value Place
	// returns; it is treated opaquely (no DNS, no real hostname discovery here).
	Hostname string

	// LiveHeadroom is the host's remaining offerable capacity, derived from the
	// accounts_headroom signal. Larger means more room. It is not an eligibility
	// gate on its own (Saturation is), but it breaks ties between two equally
	// saturated eligible hosts: prefer the one with more headroom.
	LiveHeadroom float64

	// Saturation is the fraction of the host's local worker capacity already
	// consumed, derived from procguard (0 = idle, 1 = full). It is the primary
	// placement signal: eligibility requires Saturation strictly below the
	// caller's threshold, and Place picks the least-saturated eligible host.
	Saturation float64

	// TS is the time this heartbeat was produced. Freshness is judged against it:
	// a host is stale (unavailable) once now.Sub(TS) exceeds the TTL.
	TS time.Time
}

// Fresh reports whether this heartbeat is within ttl of now. A heartbeat dated
// in the future is treated as fresh (clock skew is tolerated forward); only age
// beyond the TTL marks a host stale. A non-positive ttl means "never fresh",
// which disables spill entirely (every host looks stale).
func (h Heartbeat) Fresh(now time.Time, ttl time.Duration) bool {
	if ttl <= 0 {
		return false
	}
	age := now.Sub(h.TS)
	if age < 0 {
		return true
	}
	return age <= ttl
}

// Eligible reports whether this heartbeat is a valid spill target at now: fresh
// within ttl AND strictly below the saturation threshold. A host exactly at the
// threshold is NOT eligible (the threshold is the ceiling, not an allowed value).
func (h Heartbeat) Eligible(threshold float64, now time.Time, ttl time.Duration) bool {
	return h.Fresh(now, ttl) && h.Saturation < threshold
}

// Registry folds the latest heartbeat per host. Observing a host that already
// has an entry keeps whichever heartbeat is newer by TS, so out-of-order or
// duplicate reports collapse to the freshest fact per host. The zero value is
// not usable; construct with NewRegistry.
type Registry struct {
	latest map[string]Heartbeat
}

// NewRegistry returns an empty registry ready to Observe heartbeats.
func NewRegistry() *Registry {
	return &Registry{latest: make(map[string]Heartbeat)}
}

// Observe folds one heartbeat into the registry. If a heartbeat for the same
// hostname is already present, the one with the newer TS is kept (a strictly
// older report is ignored); on an equal TS the newly observed value wins, so a
// re-publish at the same instant refreshes the payload.
func (r *Registry) Observe(hb Heartbeat) {
	if r.latest == nil {
		r.latest = make(map[string]Heartbeat)
	}
	if prev, ok := r.latest[hb.Hostname]; ok && hb.TS.Before(prev.TS) {
		return
	}
	r.latest[hb.Hostname] = hb
}

// Hosts returns the latest heartbeat per host, sorted by hostname for a stable,
// deterministic iteration order independent of map layout.
func (r *Registry) Hosts() []Heartbeat {
	out := make([]Heartbeat, 0, len(r.latest))
	for _, hb := range r.latest {
		out = append(out, hb)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Hostname < out[j].Hostname })
	return out
}

// Decision is the outcome of a placement query.
type Decision struct {
	// Host is the chosen spill target's hostname, or "" when StayLocal is true.
	Host string
	// StayLocal is true when no remote host is eligible and the worker should run
	// on the local box (today's single-host behaviour). It is the documented
	// fallback for the single-host and all-ineligible cases.
	StayLocal bool
	// Reason is a short, machine-stable token explaining the decision, for logs
	// and tests. See the Reason* constants.
	Reason string
}

// Reason tokens for a Decision. They are stable strings, safe to assert on.
const (
	// ReasonPlaced means an eligible host was chosen as the spill target.
	ReasonPlaced = "placed"
	// ReasonNoHosts means the candidate set was empty (e.g. no FAK_DISPATCH_HOSTS
	// configured) — the single-host default.
	ReasonNoHosts = "no_hosts"
	// ReasonNoneEligible means candidates existed but all were stale and/or at or
	// above the saturation threshold.
	ReasonNoneEligible = "none_eligible"
)

// Place picks the least-saturated eligible host among hosts and returns it, or a
// StayLocal decision when none qualify. Eligibility is Heartbeat.Eligible:
// strictly below threshold AND fresh within ttl of now.
//
// Selection is fully deterministic. Among eligible hosts the winner is the one
// with the lowest Saturation; ties are broken by higher LiveHeadroom (prefer the
// host with more offerable capacity), and any remaining tie by lexicographic
// hostname. Passing the same inputs always yields the same host, so the DoD
// table test needs no clock, goroutine, or ordering assumptions.
func Place(hosts []Heartbeat, threshold float64, now time.Time, ttl time.Duration) Decision {
	if len(hosts) == 0 {
		return Decision{StayLocal: true, Reason: ReasonNoHosts}
	}

	eligible := make([]Heartbeat, 0, len(hosts))
	for _, hb := range hosts {
		if hb.Eligible(threshold, now, ttl) {
			eligible = append(eligible, hb)
		}
	}
	if len(eligible) == 0 {
		return Decision{StayLocal: true, Reason: ReasonNoneEligible}
	}

	sort.Slice(eligible, func(i, j int) bool {
		a, b := eligible[i], eligible[j]
		if a.Saturation != b.Saturation {
			return a.Saturation < b.Saturation
		}
		if a.LiveHeadroom != b.LiveHeadroom {
			return a.LiveHeadroom > b.LiveHeadroom
		}
		return a.Hostname < b.Hostname
	})
	return Decision{Host: eligible[0].Hostname, Reason: ReasonPlaced}
}

// Place is the registry-bound convenience form of the package Place function: it
// folds the registry's latest-per-host heartbeats through the same pure
// selection. It is what the intended dispatchtick seam calls once per tick.
func (r *Registry) Place(threshold float64, now time.Time, ttl time.Duration) Decision {
	return Place(r.Hosts(), threshold, now, ttl)
}
