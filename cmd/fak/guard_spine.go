package main

import (
	"context"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/fleetpane"
	"github.com/anthony-chaudhary/fak/internal/fleetspine"
	"github.com/anthony-chaudhary/fak/internal/gateway"
)

const (
	guardSpineDefaultGroup      = "239.255.70.65"
	guardSpineDefaultPort       = 4765
	guardSpineDefaultAdvertiseS = 15.0
	guardSpineDefaultPeerStaleM = 1.0
)

func guardSpineEnabled() bool {
	return !strings.EqualFold(strings.TrimSpace(os.Getenv("FLEET_SPINE_ENABLED")), "false")
}
func guardSpineGroup() string {
	if v := strings.TrimSpace(os.Getenv("FLEET_SPINE_GROUP")); v != "" {
		return v
	}
	return guardSpineDefaultGroup
}
func guardSpinePort() int { return guardSpineEnvInt("FLEET_SPINE_PORT", guardSpineDefaultPort) }
func guardSpineAdvertiseSeconds() float64 {
	return guardSpineEnvFloat("FLEET_SPINE_ADVERTISE_S", guardSpineDefaultAdvertiseS)
}
func guardSpinePeerStaleMinutes() float64 {
	return guardSpineEnvFloat("FLEET_SPINE_PEER_STALE_M", guardSpineDefaultPeerStaleM)
}
func guardSpineEnvInt(key string, fallback int) int {
	v, err := strconv.Atoi(strings.TrimSpace(os.Getenv(key)))
	if err == nil && v > 0 {
		return v
	}
	return fallback
}
func guardSpineEnvFloat(key string, fallback float64) float64 {
	v, err := strconv.ParseFloat(strings.TrimSpace(os.Getenv(key)), 64)
	if err == nil && v > 0 {
		return v
	}
	return fallback
}

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
	if !guardSpineEnabled() {
		return nil, false
	}
	transport, err := fleetspine.NewUDPMulticastTransport(guardSpineGroup(), guardSpinePort())
	if err != nil {
		spineLogf(logf, "fleetspine: disabled (multicast transport: %v)", err)
		return nil, false
	}
	selfID := fleetpane.MachineID(cfg)
	reg := fleetspine.NewRegistry(fleetspine.RegistryConfig{
		SelfID:      selfID,
		MissWindow:  time.Duration(guardSpinePeerStaleMinutes() * float64(time.Minute)),
		MinInterval: time.Duration(guardSpineAdvertiseSeconds()) * time.Second / 4,
	})
	spine := &fleetspine.Spine{
		Transport: transport,
		Registry:  reg,
		Interval:  time.Duration(guardSpineAdvertiseSeconds()) * time.Second,
		Snapshot:  guardSpineHeartbeat(selfID, root),
		Logf:      logf,
	}
	go spine.Run(ctx)
	spineLogf(logf, "fleetspine: self-discovery on (group %s:%d as %q)", guardSpineGroup(), guardSpinePort(), selfID)
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
