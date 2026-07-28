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
// behavior-unchanged.
//
// Two halves live here. ResolveHostSpill DECIDES; HostSpill.PlaceLaunch (the
// launch seam, further down) APPLIES that decision to the argv the tick is
// about to spawn, so the decision reaches the command instead of being an
// unconsulted opinion. Actually spawning the rewritten argv — handing it to the
// dispatch shell's spawner — remains the cmd/fak call site's job; nothing here
// executes anything.

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

// ---------------------------------------------------------------------------
// Launch seam — applying the decision to the worker argv.
//
// ResolveHostSpill above only DECIDES. The live tick builds the worker argv
// (BuildWorkerCommand, then GuardedLaunchCommand fronts it with `fak guard`)
// and hands it straight to the spawner, so a decision nothing CONSULTS leaves
// dispatch single-host no matter what it answers. HostSpill.PlaceLaunch is that
// consult: it takes the argv the tick is about to spawn and returns the argv it
// should actually spawn — unchanged for every local outcome, rewritten through
// the RemoteHost executor when the decision spills the worker to another box.
// It stays PURE (no os.Getenv, no wall clock, no exec): the shell passes the
// raw env value, the registry, the local hostname and now, so the whole
// placement is one deterministic function of its inputs and the single-host
// default is provably byte-identical.

// Placement reason tokens. hostplacement.ReasonNoHosts and ReasonNoneEligible
// pass through from the decision unchanged; these four say what the LAUNCH seam
// did once a host WAS placed on.
const (
	// PlacementLocal means the placed host IS the local host: the argv is
	// spawned verbatim, exactly as before this seam existed. A configured
	// multi-host fleet whose primary still has headroom keeps running local.
	PlacementLocal = "placed_local"

	// PlacementRemote means a DIFFERENT host was placed on and the argv was
	// rewritten to execute there. This is the spill.
	PlacementRemote = "placed_remote"

	// PlacementLocalHostUnknown means a remote host was eligible but the caller
	// supplied no local hostname, so the seam cannot tell "spill" from "stay".
	// It refuses to rewrite: an unknown local identity must never silently turn
	// a local launch into a remote one (and a host that is really the local box
	// would be re-entered through the remote executor).
	PlacementLocalHostUnknown = "local_host_unknown"

	// PlacementNoCommand means there was no argv to place.
	PlacementNoCommand = "no_command"
)

// DefaultRemoteLaunchProgram is the RemoteHost executor used when the caller
// names none: plain ssh, which is the transport the fleet already reaches its
// other boxes over. It is a thin seam on purpose — this file only BUILDS the
// argv, it never executes it.
const DefaultRemoteLaunchProgram = "ssh"

// HostSpill bundles the per-tick placement inputs so the live call site is one
// line. The zero value is the single-host default: no hosts configured, so
// PlaceLaunch returns the caller's argv untouched.
type HostSpill struct {
	// LocalHost is the hostname of the box running this tick — the identity the
	// placed host is compared against to tell a local placement from a spill.
	// Empty means "unknown", which disables spilling (PlacementLocalHostUnknown).
	LocalHost string

	// HostsRaw is the raw FAK_DISPATCH_HOSTS value (see DispatchHostsEnv).
	// Empty is the single-host default.
	HostsRaw string

	// Registry holds the latest headroom heartbeat per host; nil is a valid
	// empty registry.
	Registry *hostplacement.Registry

	// Threshold is the saturation ceiling; non-positive selects
	// DefaultSpillSaturationThreshold.
	Threshold float64

	// TTL is the heartbeat staleness ceiling; non-positive selects
	// DefaultHeartbeatTTL.
	TTL time.Duration

	// RemoteProgram overrides the RemoteHost executor; empty selects
	// DefaultRemoteLaunchProgram.
	RemoteProgram string
}

// LaunchPlacement is what the launch seam decided to actually spawn.
type LaunchPlacement struct {
	// Command is the argv to spawn. It is a COPY of the caller's argv whenever
	// Remote is false, so every local outcome is byte-identical to the argv the
	// tick built and no caller slice is aliased.
	Command []string

	// Host is the host placed on, or "" when the decision stayed local.
	Host string

	// Remote is true only when Command was rewritten to run on another box.
	Remote bool

	// Reason is the launch-seam token (see the Placement* constants, plus the
	// hostplacement.Reason* tokens that pass through).
	Reason string

	// Decision is the raw ResolveHostSpill outcome that produced this placement.
	Decision hostplacement.Decision
}

// PlaceLaunch is the per-tick launch-seam consult: resolve the spill decision
// for now, then apply it to command.
//
// Local outcomes — no hosts configured, none eligible, the placed host is the
// local box, an unknown local identity, or an empty argv — all return the
// caller's argv unchanged, which is what makes the single-host default
// behavior-unchanged rather than merely "intended to be". Only a placed host
// that differs from LocalHost rewrites the argv, and then only by fronting it
// with the RemoteHost executor; the worker command itself is preserved verbatim
// inside the remote word so the guard front, model flags and prompt survive.
func (s HostSpill) PlaceLaunch(command []string, now time.Time) LaunchPlacement {
	decision := ResolveHostSpill(s.HostsRaw, s.Registry, s.Threshold, now, s.TTL)
	stay := LaunchPlacement{
		Command:  append([]string(nil), command...),
		Decision: decision,
		Reason:   decision.Reason,
	}
	if decision.StayLocal {
		return stay
	}
	stay.Host = decision.Host
	if len(command) == 0 {
		stay.Reason = PlacementNoCommand
		return stay
	}
	host := strings.TrimSpace(decision.Host)
	local := strings.TrimSpace(s.LocalHost)
	if local == "" {
		stay.Reason = PlacementLocalHostUnknown
		return stay
	}
	if host == local {
		stay.Reason = PlacementLocal
		return stay
	}
	return LaunchPlacement{
		Command:  remoteLaunchCommand(s.remoteProgram(), host, command),
		Host:     host,
		Remote:   true,
		Reason:   PlacementRemote,
		Decision: decision,
	}
}

func (s HostSpill) remoteProgram() string {
	if p := strings.TrimSpace(s.RemoteProgram); p != "" {
		return p
	}
	return DefaultRemoteLaunchProgram
}

// remoteLaunchCommand renders the RemoteHost executor argv for one worker: the
// executor, the host, and the worker command collapsed into a SINGLE
// shell-quoted word. Passing the command as one already-quoted word is the only
// safe shape here — a remote shell re-splits whatever follows the host, so
// handing it the argv element-by-element would tear the worker prompt (which
// carries spaces, quotes and newlines) apart at the far end.
func remoteLaunchCommand(program, host string, command []string) []string {
	if len(command) == 0 {
		return nil
	}
	return []string{program, host, shellQuoteArgv(command)}
}

// shellQuoteArgv joins an argv into one POSIX-shell-safe command line.
func shellQuoteArgv(argv []string) string {
	parts := make([]string, 0, len(argv))
	for _, a := range argv {
		parts = append(parts, shellQuoteArg(a))
	}
	return strings.Join(parts, " ")
}

// shellQuoteArg wraps one argument in single quotes unless every rune is
// already shell-inert. A literal single quote is emitted as close-escape-reopen,
// the only escape a POSIX single-quoted string admits.
func shellQuoteArg(s string) string {
	if s != "" && !strings.ContainsFunc(s, shellUnsafeRune) {
		return s
	}
	return "'" + strings.ReplaceAll(s, "'", `'\''`) + "'"
}

func shellUnsafeRune(r rune) bool {
	switch {
	case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9':
		return false
	}
	switch r {
	case '_', '@', '%', '+', '=', ':', ',', '.', '/', '-':
		return false
	}
	return true
}
