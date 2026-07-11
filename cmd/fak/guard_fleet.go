//go:build wip_sessionfleet

// GATED WIP: this file references gateway.SessionFleet / gateway.SessionFleetMachine /
// gateway.SetSessionFleetProvider, which are not committed to internal/gateway yet, so with the
// default build tags it does not compile. Per the repo's build-witness convention (AGENTS.md,
// internal/buildwitness), work that cannot yet build against committed symbols is fenced behind a
// `//go:build wip_<feature>` tag so the default build — and every other session — stays green
// while the WIP lives on disk. To re-enable: land gateway.SessionFleet et al., then delete this
// build line (and the twin lines in guard_spine.go / info_fleet.go). Build meanwhile with
// `-tags wip_sessionfleet`.
package main

import (
	"context"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/fleetpane"
	"github.com/anthony-chaudhary/fak/internal/gateway"
)

// guard_fleet.go — builds the live fleet-CONTROL-PANE provider `fak guard` hands the
// gateway (SetSessionFleetProvider) so the status area (`fak info`) shows cross-MACHINE
// health beside the session-local blocks: how many operator machines have published
// snapshots, how many are stale/needing-action, and a few rolled-up totals. This is the
// fleet-of-machines aggregate (fleetpane.FleetDoc / `fak fleetpane fleet`), distinct from
// the gateway's native worker-membership `fleet` behind the router.
//
// Unlike the endpoints provider (a cheap per-scrape FS glob), a fleet fold walks every
// published machine snapshot off disk, so this provider is re-read behind a short TTL
// cache: a /debug/vars scrape fires every tick, and the cross-machine view changes on the
// order of minutes (each machine republishes on its own control-tick), so a 30s cache
// keeps the pane live without re-walking the machine dir on every poll. It is snapshot-
// ONLY (FleetView with includeLiveLocal=false): no git/supervisor subprocess ever runs on
// the poll path, honoring the payload-free /debug/vars contract. Everything folded here is
// display metadata (machine ids, states, counts) — never a token or a request payload.

// guardFleetTTL is how long a folded fleet view is reused before the next scrape re-walks
// the published machine snapshots. Bounds the per-scrape cost of the fleet block to one
// disk walk per window regardless of scrape rate.
const guardFleetTTL = 30 * time.Second

// guardFleetMachineRows caps how many per-machine rows the block carries. The pane only
// renders a handful; a large fleet still folds into the aggregate totals, so the row list
// stays a bounded, most-relevant sample rather than the whole roster.
const guardFleetMachineRows = 8

// guardFleetProvider is the TTL-cached state behind the pull provider. now is a seam for
// deterministic tests; nil falls back to time.Now.
type guardFleetProvider struct {
	mu     sync.Mutex
	now    func() time.Time
	cached gateway.SessionFleet
	ok     bool
	at     time.Time // stamp of the last fold (whether ok or not), for the TTL gate
	primed bool      // whether a fold has run at least once (at is meaningful)

	// diskDoc is the snapshot-only disk fold, injected so tests can drive it without an
	// on-disk machine dir. nil uses the default FleetView off the discovered repo root.
	diskDoc func() (fleetpane.FleetDoc, bool)

	// spinePeers is the LIVE network-discovery source: the machine-maps of peers the
	// fleetspine registry has heard multicast heartbeats from, in the same map[string]any
	// shape guardFleetFromDoc reads. nil (the default) leaves the fold disk-only. The union
	// happens in foldView, so the 30s TTL cache covers the merged result.
	spinePeers func() []map[string]any
}

// newGuardFleetProvider returns the pull provider `fak guard` installs via
// SetSessionFleetProvider. It folds a snapshot-only fleet view (no subprocess) behind a
// guardFleetTTL cache, returning ok=false when there is no fleet to show (no repo root,
// unreadable config, or zero published machines) so the /debug/vars block is omitted then.
func newGuardFleetProvider() func() (gateway.SessionFleet, bool) {
	p := &guardFleetProvider{}
	return p.pull
}

// newGuardFleetProviderWithSpine is newGuardFleetProvider plus the live self-discovery spine:
// on each fold it unions the disk snapshots with the peers `peers` reports (machines heard over
// UDP-multicast heartbeats), so a box on the same LAN shows up even with no shared git remote.
// peers nil behaves exactly like newGuardFleetProvider (disk-only).
func newGuardFleetProviderWithSpine(peers func() []map[string]any) func() (gateway.SessionFleet, bool) {
	p := &guardFleetProvider{spinePeers: peers}
	return p.pull
}

func (p *guardFleetProvider) timeNow() time.Time {
	if p.now != nil {
		return p.now()
	}
	return time.Now()
}

// pull returns the cached fold when still fresh, else recomputes and re-stamps. The stamp
// is updated on every recompute (ok or not) so a fleet that folds to nothing does not
// re-walk the machine dir on every subsequent scrape within the window.
func (p *guardFleetProvider) pull() (gateway.SessionFleet, bool) {
	p.mu.Lock()
	defer p.mu.Unlock()
	now := p.timeNow()
	if p.primed && now.Sub(p.at) < guardFleetTTL {
		return p.cached, p.ok
	}
	fleet, ok := p.foldView()
	p.cached, p.ok, p.at, p.primed = fleet, ok, now, true
	return fleet, ok
}

// foldView produces the current SessionFleet: the disk snapshot fold UNIONED with the live
// spine peers (when a spine source is wired). The union is keyed by machine id, newest wins,
// so a machine present both on disk and live dedupes to its fresher record; the aggregate
// (verdict, stale/action counts, session totals) is then recomputed over the merged set. When
// both sources are empty the merged doc has no machines, so guardFleetFromDoc returns ok=false
// and the /debug/vars block is omitted — the zero-cost, disk-only-degradation contract.
func (p *guardFleetProvider) foldView() (gateway.SessionFleet, bool) {
	doc, ok := p.diskFold()
	if !ok {
		// No repo/config: there is no disk doc to union into. Fall back to a bare doc so a
		// spine-only fleet (peers on the LAN, no shared git) still renders.
		doc = fleetpane.FleetDoc{States: map[string]int{}, Totals: map[string]int{}}
	}
	if p.spinePeers != nil {
		if peers := p.spinePeers(); len(peers) > 0 {
			mergeSpineIntoDoc(&doc, peers)
		}
	}
	return guardFleetFromDoc(doc)
}

// diskFold reads the snapshot-only disk fold (injected in tests, else the default FleetView off
// the discovered repo root). ok is false on any read failure. Runs NO subprocess: FleetView
// with includeLiveLocal=false reads only the published per-machine snapshots.
func (p *guardFleetProvider) diskFold() (fleetpane.FleetDoc, bool) {
	if p.diskDoc != nil {
		return p.diskDoc()
	}
	return guardFleetSnapshotDoc()
}

// guardFleetSnapshotDoc folds the snapshot-only fleet aggregate off the discovered repo root
// into a fleetpane.FleetDoc. ok is false on any read failure (no repo root, unreadable config).
func guardFleetSnapshotDoc() (fleetpane.FleetDoc, bool) {
	root, err := fleetpane.FindRepoRoot(".")
	if err != nil {
		return fleetpane.FleetDoc{}, false
	}
	cfg, err := fleetpane.LoadConfig(root)
	if err != nil {
		return fleetpane.FleetDoc{}, false
	}
	return fleetpane.FleetView(context.Background(), cfg, false, false, fleetpane.Options{}), true
}

// mergeSpineIntoDoc unions the live spine peers into doc.Machines keyed by id (newest
// generated_utc wins over a disk entry with the same id), then recomputes the aggregate the
// fold reads — States (the stale/action counts) and Totals["sessions"] — plus the Verdict, so
// the summary reflects both discovery sources. It mirrors FleetView's own aggregation rule:
// verdict is ACTION when any machine is in a needs-attention state, else OK.
func mergeSpineIntoDoc(doc *fleetpane.FleetDoc, peers []map[string]any) {
	byID := make(map[string]int, len(doc.Machines))
	for i, m := range doc.Machines {
		byID[guardFleetString(m["id"])] = i
	}
	for _, peer := range peers {
		id := guardFleetString(peer["id"])
		if id == "" {
			continue
		}
		if idx, ok := byID[id]; ok {
			// Same machine on disk AND live: keep whichever snapshot is newer.
			if guardFleetGeneratedNewer(peer, doc.Machines[idx]) {
				doc.Machines[idx] = peer
			}
			continue
		}
		byID[id] = len(doc.Machines)
		doc.Machines = append(doc.Machines, peer)
	}
	guardFleetRecomputeAggregate(doc)
}

// guardFleetGeneratedNewer reports whether machine a's generated_utc is strictly newer than b's.
// An unparseable/absent stamp sorts oldest, so a live peer with a real stamp beats a stale disk
// entry with none.
func guardFleetGeneratedNewer(a, b map[string]any) bool {
	return guardFleetGeneratedAt(a).After(guardFleetGeneratedAt(b))
}

func guardFleetGeneratedAt(m map[string]any) time.Time {
	raw := guardFleetString(m["generated_utc"])
	if raw == "" {
		return time.Time{}
	}
	if ts, err := time.Parse(time.RFC3339, raw); err == nil {
		return ts
	}
	if ts, err := time.Parse(time.RFC3339Nano, raw); err == nil {
		return ts
	}
	return time.Time{}
}

// guardFleetRecomputeAggregate recomputes doc.States, doc.Totals["sessions"], and doc.Verdict
// from the (merged) machine list, following the same needs-attention rule FleetView uses.
func guardFleetRecomputeAggregate(doc *fleetpane.FleetDoc) {
	states := map[string]int{}
	sessions := 0
	action := false
	for _, m := range doc.Machines {
		state := guardFleetString(m["state"])
		if state == "" {
			state = "UNKNOWN"
		}
		states[state]++
		sessions += guardFleetInt(m["sessions"])
		switch state {
		case "ACTION", "STALE", "UNKNOWN", "INVALID":
			action = true
		}
	}
	if doc.States == nil {
		doc.States = map[string]int{}
	}
	// Overwrite the stale/action-relevant counts from the merged set; leave any pre-existing
	// disk-only keys the fold does not read untouched.
	for k := range doc.States {
		delete(doc.States, k)
	}
	for k, v := range states {
		doc.States[k] = v
	}
	if doc.Totals == nil {
		doc.Totals = map[string]int{}
	}
	doc.Totals["sessions"] = sessions
	switch {
	case len(doc.Machines) == 0:
		doc.Verdict = "EMPTY"
	case doc.Totals["version_mismatches"] > 0 || action:
		doc.Verdict = "ACTION"
	default:
		doc.Verdict = "OK"
	}
}

// guardFleetFromDoc folds a fleetpane.FleetDoc into the gateway's display-only
// SessionFleet. ok is false when the fleet holds no machines (verdict EMPTY / cold
// operator), so a machine with no peers omits the block instead of showing "machines=0".
func guardFleetFromDoc(doc fleetpane.FleetDoc) (gateway.SessionFleet, bool) {
	if len(doc.Machines) == 0 {
		return gateway.SessionFleet{}, false
	}
	fleet := gateway.SessionFleet{
		Verdict:           doc.Verdict,
		Machines:          len(doc.Machines),
		Stale:             doc.States["STALE"],
		Action:            doc.States["ACTION"],
		Sessions:          doc.Totals["sessions"],
		AuthBlocked:       doc.Totals["auth_blocked"],
		ThrottledSeats:    doc.Totals["throttled_seats"],
		HealthySeats:      doc.Totals["healthy_seats"],
		SeatCapacity:      doc.Totals["seat_capacity"],
		ResumeBacklog:     doc.Totals["auto_resume"],
		VersionMismatches: doc.Totals["version_mismatches"],
	}
	rows := make([]gateway.SessionFleetMachine, 0, min(len(doc.Machines), guardFleetMachineRows))
	for _, m := range doc.Machines {
		if len(rows) >= guardFleetMachineRows {
			break
		}
		rows = append(rows, gateway.SessionFleetMachine{
			ID:       guardFleetString(m["id"]),
			State:    guardFleetString(m["state"]),
			AgeMin:   guardFleetFloat(m["age_min"]),
			Sessions: guardFleetInt(m["sessions"]),
			Version:  guardFleetString(m["app_version"]),
		})
	}
	fleet.Rows = rows
	return fleet, true
}

// guardFleetString reads a display string from a snapshot map value, "" when absent.
func guardFleetString(v any) string {
	if s, ok := v.(string); ok {
		return s
	}
	return ""
}

// guardFleetInt reads an int count from a snapshot map value (JSON numbers decode as
// float64 through the FleetDoc's map[string]any machines), 0 when absent/non-numeric.
func guardFleetInt(v any) int {
	switch n := v.(type) {
	case int:
		return n
	case float64:
		return int(n)
	}
	return 0
}

// guardFleetFloat reads a float (snapshot age in minutes; may be a *float64 in the fold),
// 0 when absent/non-numeric.
func guardFleetFloat(v any) float64 {
	switch n := v.(type) {
	case float64:
		return n
	case *float64:
		if n != nil {
			return *n
		}
	case int:
		return float64(n)
	}
	return 0
}
