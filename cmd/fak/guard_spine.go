//go:build wip_sessionfleet

// GATED WIP — see guard_fleet.go: depends on the not-yet-committed gateway.SessionFleet surface,
// fenced behind //go:build wip_sessionfleet so the default build stays green. Remove this line
// once gateway.SessionFleet lands.
package main

import (
	"context"
	"os"
	"time"

	"github.com/anthony-chaudhary/fak/internal/fleetpane"
	"github.com/anthony-chaudhary/fak/internal/fleetspine"
	"github.com/anthony-chaudhary/fak/internal/gateway"
)

// guard_spine.go — wires the networking-aware self-discovery spine into `fak guard`. The spine
// lets machines find each other LIVE over the LAN via UDP-multicast heartbeats, so the `fak info`
// fleet panel shows peers that share NO git remote or filesystem (a fresh cloud box, a Mac verify
// node) within one heartbeat interval — instead of only machines wired into the same repo that
// have recently pushed a snapshot. The spine is a SECOND discovery source; guardFleetProvider
// unions it with the disk fold (see guard_fleet.go). Everything on the wire is display metadata
// (id/host/verdict/version) — never a token or payload — so the /debug/vars contract holds.
//
// Discovery is ON by default (zero-config), and degradation is load-bearing because of that: a
// box on a network that blocks multicast, or with no usable NIC, must fall back SILENTLY to the
// disk-only view and never fail the guard. So every build/bind error here returns the plain
// disk-only provider rather than propagating.

// newGuardFleetProviderMaybeSpine returns the fleet provider `fak guard` installs. When the
// self-discovery spine is enabled (default) and its multicast transport binds, it starts the
// advertise/listen/expiry loops on ctx (the guard-lifetime context, so the socket closes on
// shutdown) and returns a provider that unions live peers with the disk fold. On ANY failure —
// no repo/config, spine disabled, transport won't bind — it returns the plain disk-only provider,
// so guard never fails to start over a network issue. logf is the one-line non-fatal logger
// (may be nil).
func newGuardFleetProviderMaybeSpine(ctx context.Context, logf fleetspine.Logf) func() (gateway.SessionFleet, bool) {
	reg, ok := startGuardSpine(ctx, logf)
	if !ok {
		return newGuardFleetProvider()
	}
	return newGuardFleetProviderWithSpine(func() []map[string]any {
		return reg.MachineMaps(time.Time{}) // zero time => registry's own clock
	})
}

// spineLogf emits a one-line, non-fatal spine message when logf is set (nil is a no-op).
func spineLogf(logf fleetspine.Logf, format string, args ...any) {
	if logf != nil {
		logf(format, args...)
	}
}

// startGuardSpine loads the fleet config and, when the spine is enabled and its multicast
// transport binds, builds the registry, starts the three loops on ctx, and returns the registry
// the provider reads peers from. ok is false (registry nil) whenever the spine cannot or should
// not run — the caller then wires the disk-only provider.
func startGuardSpine(ctx context.Context, logf fleetspine.Logf) (*fleetspine.Registry, bool) {
	root, err := fleetpane.FindRepoRoot(".")
	if err != nil {
		return nil, false
	}
	cfg, err := fleetpane.LoadConfig(root)
	if err != nil {
		return nil, false
	}
	if !cfg.SpineEnabled {
		return nil, false
	}
	transport, err := fleetspine.NewUDPMulticastTransport(cfg.SpineGroup, cfg.SpinePort)
	if err != nil {
		spineLogf(logf, "fleetspine: disabled (multicast transport: %v)", err)
		return nil, false
	}
	selfID := fleetpane.MachineID(cfg)
	reg := fleetspine.NewRegistry(fleetspine.RegistryConfig{
		SelfID:      selfID,
		MissWindow:  time.Duration(cfg.SpinePeerStaleM * float64(time.Minute)),
		MinInterval: time.Duration(cfg.SpineAdvertiseS) * time.Second / 4,
	})
	spine := &fleetspine.Spine{
		Transport: transport,
		Registry:  reg,
		Interval:  time.Duration(cfg.SpineAdvertiseS) * time.Second,
		Snapshot:  guardSpineHeartbeat(selfID, root),
		Logf:      logf,
	}
	go spine.Run(ctx)
	spineLogf(logf, "fleetspine: self-discovery on (group %s:%d as %q)", cfg.SpineGroup, cfg.SpinePort, selfID)
	return reg, true
}

// guardSpineHeartbeat builds the closure that produces THIS machine's compact heartbeat each
// advertise tick. It is deliberately cheap — MachineID + hostname + app version + a fresh stamp,
// NO subprocess — because it fires on a short interval; the heavy per-machine snapshot the disk
// path collects (git/supervisor/monitor) is never run here. State is "OK": a running guard that
// can advertise is, by that fact, up.
func guardSpineHeartbeat(selfID, root string) func() fleetspine.Heartbeat {
	host, _ := os.Hostname()
	version := fleetpane.AppVersion(root)
	return func() fleetspine.Heartbeat {
		return fleetspine.Heartbeat{
			Schema:       fleetspine.HeartbeatSchema,
			ID:           selfID,
			Host:         host,
			State:        "OK",
			AppVersion:   version,
			GeneratedUTC: time.Now().UTC().Format(time.RFC3339),
		}
	}
}
