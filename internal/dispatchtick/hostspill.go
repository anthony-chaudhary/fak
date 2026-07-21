package dispatchtick

import (
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/hostplacement"
)

// Multi-host worker placement seam (#3599, first slice): dispatch is
// single-host today — a wave targets the local box only, so the only lever to
// 100x (more boxes) is unreachable. This file is the dispatch-side spill
// decision over the internal/hostplacement primitive: given the configured
// FAK_DISPATCH_HOSTS candidate set and a per-host headroom registry (latest
// heartbeat per host), ResolveHostSpill answers "place the next worker on a
// remote host, or stay local?" once per tick, BEFORE the worker-launch seam
// (BuildWorkerCommand / GuardedLaunchCommand over WorkerLaunch) builds the
// command.
//
// Like the rest of this package it is a PURE contract: the caller passes the
// raw env VALUE, the registry, and now/TTL explicitly — no os.Getenv, no wall
// clock, no ssh/exec — so the spill decision is trivially deterministic and
// the single-host default (no FAK_DISPATCH_HOSTS) is provably
// behavior-unchanged. Executing the placed worker on the chosen remote host (a
// RemoteHost executor behind the launch seam) is the follow-on slice; this
// seam only DECIDES.

// DispatchHostsEnv names the env knob holding the configured multi-host
// candidate set: a comma- and/or whitespace-separated list of hostnames the
// dispatch tick may spill workers onto. Unset or empty means single-host
// (today's behavior, unchanged). The caller reads the env and passes the raw
// VALUE to ResolveHostSpill, keeping this file os.Getenv-free.
const DispatchHostsEnv = "FAK_DISPATCH_HOSTS"

const (
	// DefaultSpillSaturationThreshold is the saturation ceiling a host must be
	// strictly below to accept a spilled worker when the caller passes a
	// non-positive threshold. 0.90 leaves one-tenth of a host's capacity as the
	// margin that absorbs heartbeat lag: a host reported just under full may
	// have grown since its last report.
	DefaultSpillSaturationThreshold = 0.90

	// DefaultHeartbeatTTL is the staleness ceiling applied when the caller
	// passes a non-positive TTL: a host whose latest heartbeat is older than
	// this is unavailable (never placed on). 90s tolerates two missed beats of
	// a ~30s heartbeat cadence before a host drops out of the candidate set.
	DefaultHeartbeatTTL = 90 * time.Second
)

// ParseDispatchHosts parses a raw FAK_DISPATCH_HOSTS value into the configured
// candidate hostname set: hosts are separated by commas and/or whitespace,
// each is trimmed, empties are dropped, and duplicates collapse to the first
// occurrence (order-preserving). Hostnames are opaque exact-match tokens here,
// same as in hostplacement — no DNS, no case folding. An empty or
// whitespace-only value returns nil: the single-host default.
func ParseDispatchHosts(raw string) []string {
	fields := strings.FieldsFunc(raw, func(r rune) bool {
		return r == ',' || r == ' ' || r == '\t' || r == '\n' || r == '\r'
	})
	seen := make(map[string]bool, len(fields))
	hosts := make([]string, 0, len(fields))
	for _, f := range fields {
		f = strings.TrimSpace(f)
		if f == "" || seen[f] {
			continue
		}
		seen[f] = true
		hosts = append(hosts, f)
	}
	if len(hosts) == 0 {
		return nil
	}
	return hosts
}

// ResolveHostSpill is the per-tick placement decision. hostsRaw is the raw
// FAK_DISPATCH_HOSTS value; reg holds the latest headroom heartbeat per host
// (nil is a valid empty registry). A non-positive threshold or ttl selects the
// package default (DefaultSpillSaturationThreshold / DefaultHeartbeatTTL) —
// at this seam a zero knob means "default", unlike the raw primitive where a
// non-positive TTL means "never fresh".
//
// The decision:
//   - No hosts configured -> stay local, ReasonNoHosts (the single-host
//     default; heartbeats in the registry are irrelevant).
//   - Otherwise only CONFIGURED hosts are candidates: a registry heartbeat for
//     an unconfigured host is ignored, and a configured host with no heartbeat
//     has no headroom fact and cannot be placed on.
//   - Among candidates, hostplacement.Place picks the least-saturated host
//     that is fresh within ttl of now and strictly below threshold — the
//     primary at capacity SPILLS the worker to the second host.
//   - None eligible (all full and/or stale, or no candidate has a heartbeat)
//     -> stay local, ReasonNoneEligible: the spill is DECLINED and the worker
//     runs on the local box exactly as before.
func ResolveHostSpill(hostsRaw string, reg *hostplacement.Registry, threshold float64, now time.Time, ttl time.Duration) hostplacement.Decision {
	configured := ParseDispatchHosts(hostsRaw)
	if len(configured) == 0 {
		return hostplacement.Decision{StayLocal: true, Reason: hostplacement.ReasonNoHosts}
	}
	if threshold <= 0 {
		threshold = DefaultSpillSaturationThreshold
	}
	if ttl <= 0 {
		ttl = DefaultHeartbeatTTL
	}
	allowed := make(map[string]bool, len(configured))
	for _, h := range configured {
		allowed[h] = true
	}
	candidates := []hostplacement.Heartbeat{}
	if reg != nil {
		for _, hb := range reg.Hosts() {
			if allowed[hb.Hostname] {
				candidates = append(candidates, hb)
			}
		}
	}
	if len(candidates) == 0 {
		// Hosts are configured but none has a headroom fact: that is a decline
		// (none eligible), not the single-host default.
		return hostplacement.Decision{StayLocal: true, Reason: hostplacement.ReasonNoneEligible}
	}
	return hostplacement.Place(candidates, threshold, now, ttl)
}
