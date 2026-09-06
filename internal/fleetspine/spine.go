// Package fleetspine is the networking-aware self-discovery spine for the fleet-control
// pane. It lets machines find each other LIVE over the LAN — via UDP-multicast heartbeats —
// instead of only through the git-mediated shared-filesystem path (each machine writing a
// snapshot to tools/_registry/machines/<id>.json that peers see only after a git pull).
//
// The spine is a second discovery source the `fak guard` fleet provider unions with the disk
// fold: a machine on the same network shows up in `fak info` within one heartbeat interval
// even with NO shared git remote or machine dir. It carries ONLY display metadata (id, host,
// verdict, sessions, version, timestamp) — never a token or a request payload — so the
// payload-free /debug/vars contract holds by construction.
//
// This file is the pure core: the Heartbeat wire message and the passive peer Registry. The
// registry is passive by design — peers PUSH heartbeats; the spine never probes them — so
// liveness is a pure time-since-last-heartbeat rule (the network analog of the fleetpane
// machine-snapshot staleness gate), not the active probe/hysteresis loop the gateway's
// worker-membership FleetMembership uses. Transport and the advertise/listen goroutines live
// in transport.go and loop.go.
package fleetspine

import (
	"sort"
	"sync"
	"time"
)

// Heartbeat is the compact wire message one guard multicasts to announce itself. It is a
// struct of display SCALARS only — no map, no payload field — so it mirrors the gateway's
// display-only SessionFleetMachine field-for-field and can never smuggle a token onto the
// wire. State carries the raw verdict ("OK"/"ACTION"); the RECEIVER derives staleness from
// GeneratedUTC, exactly as the disk path derives it from a snapshot's generated_utc.
type Heartbeat struct {
	Schema       string `json:"schema"`
	ID           string `json:"id"`
	Host         string `json:"host"`
	State        string `json:"state"`
	AppVersion   string `json:"app_version"`
	Sessions     int    `json:"sessions"`
	GeneratedUTC string `json:"generated_utc"`
	Endpoint     string `json:"endpoint,omitempty"`
}

// HeartbeatSchema tags the wire format so a future field add is non-breaking: a receiver on an
// older schema still reads the fields it knows and ignores the rest.
const HeartbeatSchema = "fak.fleetspine.heartbeat/v1"

// defaultMaxPeers bounds the peer map so a flood of distinct spoofed ids on the multicast
// group cannot grow it without limit. Once full, heartbeats from NEW ids are ignored while
// known ids keep updating — a departed peer still ages out via Expire, freeing a slot.
const defaultMaxPeers = 256

// peer is one tracked machine: the last heartbeat received from it plus when it arrived (the
// clock the staleness/expiry rules read, never a timestamp the peer itself controls).
type peer struct {
	hb       Heartbeat
	lastSeen time.Time
}

// RegistryConfig parameterizes a Registry. Zero values fall back to sane defaults so a caller
// can pass only what it overrides.
type RegistryConfig struct {
	// SelfID is this machine's sanitized id. Heartbeats whose id equals it are dropped
	// (multicast loopback returns our own packets, and the panel is peer-only by design).
	SelfID string
	// MissWindow is how long since the last heartbeat a peer stays live before it is marked
	// STALE. The network analog of fleetpane's machine_stale_min. Default 15m.
	MissWindow time.Duration
	// HardExpiry is how long since the last heartbeat before a peer is dropped from the map
	// entirely (a departed machine eventually disappears). Default 2×MissWindow.
	HardExpiry time.Duration
	// MinInterval rate-limits a single id: heartbeats arriving faster than this for an
	// already-known id are ignored, bounding a per-id flood. Default 0 (no rate limit).
	MinInterval time.Duration
	// MaxPeers caps the peer map. Default defaultMaxPeers.
	MaxPeers int
	// Now is the clock seam for deterministic tests; nil falls back to time.Now.
	Now func() time.Time
}

// Registry is the passive, concurrency-safe store of discovered peers. Ingest folds an
// incoming heartbeat in; Expire drops hard-expired peers; Snapshot / MachineMaps read the
// live view. All three take (or default) a now clock so tests drive expiry deterministically.
type Registry struct {
	mu          sync.Mutex
	selfID      string
	missWindow  time.Duration
	hardExpiry  time.Duration
	minInterval time.Duration
	maxPeers    int
	now         func() time.Time
	peers       map[string]*peer
}

// NewRegistry builds a Registry from cfg, applying defaults for any zero field.
func NewRegistry(cfg RegistryConfig) *Registry {
	miss := cfg.MissWindow
	if miss <= 0 {
		miss = 15 * time.Minute
	}
	hard := cfg.HardExpiry
	if hard <= 0 {
		hard = 2 * miss
	}
	maxPeers := cfg.MaxPeers
	if maxPeers <= 0 {
		maxPeers = defaultMaxPeers
	}
	return &Registry{
		selfID:      cfg.SelfID,
		missWindow:  miss,
		hardExpiry:  hard,
		minInterval: cfg.MinInterval,
		maxPeers:    maxPeers,
		now:         cfg.Now,
		peers:       make(map[string]*peer),
	}
}

func (r *Registry) clock() time.Time {
	if r.now != nil {
		return r.now()
	}
	return time.Now()
}

// Ingest folds one heartbeat into the registry, stamping it with now (the arrival time). It is
// the testable body of the listener loop. A heartbeat is dropped when it: is malformed (empty
// id, or a generated_utc that does not parse), is our own (id == selfID, the multicast
// self-echo), arrives too soon after the last one from a known id (per-id rate limit), or
// would grow the map past MaxPeers with a NEW id. Known ids always update, even at the cap.
func (r *Registry) Ingest(hb Heartbeat, now time.Time) {
	if hb.ID == "" || hb.ID == r.selfID {
		return
	}
	if !heartbeatTimestampOK(hb.GeneratedUTC) {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if existing, ok := r.peers[hb.ID]; ok {
		if r.minInterval > 0 && now.Sub(existing.lastSeen) < r.minInterval {
			return // rate-limited: too soon after the last heartbeat from this id
		}
		existing.hb = hb
		existing.lastSeen = now
		return
	}
	if len(r.peers) >= r.maxPeers {
		return // map full: ignore a brand-new id until a slot frees via Expire
	}
	r.peers[hb.ID] = &peer{hb: hb, lastSeen: now}
}

// Expire drops peers whose last heartbeat is older than HardExpiry, so a machine that has left
// the network eventually disappears from the view even with no new traffic.
func (r *Registry) Expire(now time.Time) {
	r.mu.Lock()
	defer r.mu.Unlock()
	for id, p := range r.peers {
		if now.Sub(p.lastSeen) > r.hardExpiry {
			delete(r.peers, id)
		}
	}
}

// Peer is one machine's view for a Snapshot consumer: its last heartbeat, when it was last
// seen, and whether it is currently stale (past MissWindow but not yet hard-expired).
type Peer struct {
	Heartbeat
	LastSeen time.Time
	Stale    bool
}

// Snapshot returns the live peer set (not hard-expired) sorted by id, each marked stale when
// now-LastSeen exceeds MissWindow. It also runs Expire under the same lock so a read drops
// departed peers as a side effect. now defaults to the registry clock when zero.
func (r *Registry) Snapshot(now time.Time) []Peer {
	if now.IsZero() {
		now = r.clock()
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	// Prune hard-expired peers inline (same rule as Expire) so a read is self-cleaning.
	ids := make([]string, 0, len(r.peers))
	for id, p := range r.peers {
		if now.Sub(p.lastSeen) > r.hardExpiry {
			delete(r.peers, id)
			continue
		}
		ids = append(ids, id)
	}
	sort.Strings(ids)
	out := make([]Peer, 0, len(ids))
	for _, id := range ids {
		p := r.peers[id]
		out = append(out, Peer{
			Heartbeat: p.hb,
			LastSeen:  p.lastSeen,
			Stale:     now.Sub(p.lastSeen) > r.missWindow,
		})
	}
	return out
}

// MachineMaps folds the live peers into the exact map[string]any shape the guard fleet fold
// reads (guardFleetFromDoc in cmd/fak/guard_fleet.go): the keys id, host, state, age_min,
// sessions, app_version, generated_utc. state is the peer's reported verdict, overridden to
// "STALE" once the peer is past the miss-window — the same rule fleetpane.SummarizeMachineSnapshot
// applies to a disk snapshot. age_min is minutes since the last heartbeat, so the union step
// can dedupe a disk-and-live machine newest-wins. now defaults to the registry clock.
func (r *Registry) MachineMaps(now time.Time) []map[string]any {
	peers := r.Snapshot(now)
	if now.IsZero() {
		now = r.clock()
	}
	out := make([]map[string]any, 0, len(peers))
	for _, p := range peers {
		state := p.State
		if p.Stale || state == "" {
			state = "STALE"
		}
		m := map[string]any{
			"id":            p.ID,
			"host":          p.Host,
			"state":         state,
			"age_min":       now.Sub(p.LastSeen).Minutes(),
			"sessions":      p.Sessions,
			"app_version":   p.AppVersion,
			"generated_utc": p.GeneratedUTC,
			"source":        "spine",
		}
		if p.Endpoint != "" {
			m["endpoint"] = p.Endpoint
		}
		out = append(out, m)
	}
	return out
}

// heartbeatTimestampOK reports whether a heartbeat's generated_utc is a parseable RFC3339
// stamp. A blank or garbage stamp makes the receiver's age math meaningless, so such a
// heartbeat is rejected rather than folded in with a bogus age.
func heartbeatTimestampOK(raw string) bool {
	if raw == "" {
		return false
	}
	if _, err := time.Parse(time.RFC3339, raw); err == nil {
		return true
	}
	_, err := time.Parse(time.RFC3339Nano, raw)
	return err == nil
}
