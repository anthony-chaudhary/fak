package main

// session_inventory.go — the DURABLE + FLEET view behind `fak session ls --durable`
// / `fak session ls --fleet` (issue #1203, epic #1193 Pillar 4). The default
// `fak session ls` reads the live gateway snapshot (the in-memory session.Table over
// GET /v1/fak/sessions), so it shows only THIS process's sessions and drops them on a
// restart / LRU eviction. This surface answers the operator's actual question —
// "how many sessions are alive across the fleet, and what is each one's cache posture?"
// — from DETERMINISTIC, non-self-reported sources:
//
//   - --durable : reads the durable C1 session registry (internal/session.Registry over
//     a FileStore), which survives a process restart and eviction. No gateway required.
//   - --fleet   : --durable PLUS a best-effort `git fetch` of the refs/fak/locks/*
//     namespace, then folds in every node's C2 session refs (internal/leaseref
//     refs/fak/locks/session-*) so a peer node's sessions appear here too. Deduped by id,
//     preferring the richer local C1 row over its own pushed C2 projection.
//
// Each row is sourced from the registry + an ORACLE, never the agent's self-report:
//   {id, host, pcb_state, liveness_class, cache_posture, age, parent_id}
//   - pcb_state    : the durable PCB run-state (RUNNING/THROTTLED/PAUSED/DRAINING/STOPPED).
//   - liveness_class: #750's witnessed vocabulary (internal/taskmgr.LivenessClass —
//     live / idle / stalled) read off the durable HEARTBEAT (LastSeen for C1, UpdatedAt
//     for C2). A running-family session (RUNNING/THROTTLED/DRAINING) whose heartbeat has
//     not advanced within the stale window — or that carries no heartbeat at all — is
//     STALLED (claims work, not progressing / garbage); PAUSED/STOPPED is idle.
//   - cache_posture: internal/resume.Plan's WARM/COLD projection over idle-since-last-stamp.
//   - age          : now - created-at (local) / now - updated-at (fleet-only C2 rows).
//   - parent_id    : the re-continuation lineage parent (the reset transaction's old trace).
//
// HONEST FENCE (liveness_class). #750's oracle (alive ∧ progressing ∧ not-garbage) is a
// per-turn heartbeat witness. An OFFLINE `session ls` reads one durable snapshot, so it
// classifies with the #750 vocabulary off the persisted heartbeat's freshness — a
// single-snapshot projection, not a live two-sample delta. A wedged session that stops
// re-stamping its descriptor reads STALLED (the wire this surface exists to expose); the
// live per-turn delta stays the gateway/taskmgr's job. cache_posture carries the same
// projection fence resume.go documents (a posture over idle-vs-TTL, never a witnessed
// provider-cache hit).

import (
	"context"
	"fmt"
	"io"
	"sort"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/anthony-chaudhary/fak/internal/leaseref"
	"github.com/anthony-chaudhary/fak/internal/pathutil"
	"github.com/anthony-chaudhary/fak/internal/resume"
	"github.com/anthony-chaudhary/fak/internal/session"
	"github.com/anthony-chaudhary/fak/internal/taskmgr"
)

// defaultSessionStaleWindow is the liveness heartbeat window: a running-family session
// whose durable heartbeat (LastSeen / UpdatedAt) has not advanced within it reads STALLED
// — "claims it is working, but not progressing". A healthy session re-stamps its
// descriptor on every drive change / Decide, so this is deliberately far tighter than the
// registry's 30-minute reap TTL; it is a heuristic, tunable with --stale.
const defaultSessionStaleWindow = 5 * time.Minute

// sessionInventoryRow is one deterministic, non-self-reported row of the durable/fleet
// inventory. Field names match the acceptance tuple {id, host, pcb_state, liveness_class,
// cache_posture, age, parent_id}; Source records whether the row came from the local C1
// registry or a folded-in C2 fleet ref.
type sessionInventoryRow struct {
	ID            string `json:"id"`
	Host          string `json:"host"`
	PCBState      string `json:"pcb_state"`
	LivenessClass string `json:"liveness_class"`
	CachePosture  string `json:"cache_posture"`
	AgeSeconds    int64  `json:"age_seconds"`
	ParentID      string `json:"parent_id,omitempty"`
	Source        string `json:"source"`
}

// sessionInventory is the rendered result: the rows plus the headline rollup (total,
// per-pcb-state counts, warm count) so the one-line count summary is immediate and --json
// carries the same numbers for a machine.
type sessionInventory struct {
	Count    int                   `json:"count"`
	Fleet    bool                  `json:"fleet"`
	ByState  map[string]int        `json:"by_state"`
	Warm     int                   `json:"warm"`
	Cold     int                   `json:"cold"`
	Sessions []sessionInventoryRow `json:"sessions"`
}

// sessionInventoryOpts carries the parsed `ls` flags into the offline inventory path.
type sessionInventoryOpts struct {
	asJSON       bool
	fleet        bool
	registryPath string
	remote       string
	staleWindow  time.Duration
}

// pcbStateOrder is the canonical status order the count summary and table sort by, so the
// rollup reads "3 RUNNING, 1 PAUSED, 1 STALLED, ..." in a stable, scan-friendly order
// regardless of map iteration. STALLED is the liveness-derived status a running-family row
// with a lapsed heartbeat rolls up as. States outside this set sort after, alphabetically.
var pcbStateOrder = map[string]int{
	"RUNNING": 0, "THROTTLED": 1, "PAUSED": 2, "DRAINING": 3, "STALLED": 4, "STOPPED": 5,
}

// runSessionInventory wires the real durable registry (and, for --fleet, the git-backed
// leaseref store) and renders the inventory. It never dials a gateway, so it answers
// after a restart / with no serve process running — the durable point of the surface.
func runSessionInventory(stdout, stderr io.Writer, opt sessionInventoryOpts) int {
	now := time.Now()
	path := strings.TrimSpace(opt.registryPath)
	if path == "" {
		path = defaultSessionRegistryPath()
	}
	path = pathutil.ExpandTilde(path)

	reg := session.NewRegistry(session.NewFileStore(path))
	local, err := reg.List(now)
	if err != nil {
		fmt.Fprintf(stderr, "fak session ls: read durable registry %s: %v\n", path, err)
		return 1
	}

	var fleetDescs []leaseref.SessionDescriptor
	if opt.fleet {
		remote := strings.TrimSpace(opt.remote)
		if remote == "" {
			remote = "origin"
		}
		store := leaseref.NewInDir("")
		ctx := context.Background()
		// Best-effort fetch: a fleet view of peers' sessions needs the remote refs, but a
		// missing remote / no network must degrade to "the C2 refs this clone already has",
		// never fail the whole listing (the same fail-open posture as leaseref publish).
		if _, ferr := store.Sync(ctx, remote, false, true); ferr != nil {
			fmt.Fprintf(stderr, "fak session ls --fleet: git fetch %s: %v (showing already-fetched C2 refs only)\n", remote, ferr)
		}
		live, _, lerr := store.LiveSessions(ctx, now)
		if lerr != nil {
			fmt.Fprintf(stderr, "fak session ls --fleet: read session refs: %v\n", lerr)
		} else {
			fleetDescs = live
		}
	}

	inv := buildSessionInventory(local, fleetDescs, now, opt.staleWindow, opt.fleet)
	if opt.asJSON {
		return emitSessionJSON(stdout, stderr, inv)
	}
	renderSessionInventory(stdout, inv)
	return 0
}

// buildSessionInventory folds the durable C1 descriptors and the (optional) C2 fleet
// descriptors into one deterministic inventory. It is pure over its inputs (an injected
// now, no I/O) so it is unit-testable to an exact rollup. A C1 row supersedes a C2 row
// with the same id (C1 carries the richer projection — parent, created-at); a peer's
// session appears only via C2. Rows are sorted by (host, id) for a stable view.
func buildSessionInventory(local []session.Descriptor, fleet []leaseref.SessionDescriptor, now time.Time, staleWindow time.Duration, isFleet bool) sessionInventory {
	if staleWindow <= 0 {
		staleWindow = defaultSessionStaleWindow
	}
	inv := sessionInventory{Fleet: isFleet, ByState: map[string]int{}}
	seen := make(map[string]bool, len(local)+len(fleet))

	for _, d := range local {
		id := strings.TrimSpace(d.ID)
		if id == "" {
			id = strings.TrimSpace(d.Trace)
		}
		if id == "" {
			continue
		}
		seen[id] = true
		// The heartbeat is the descriptor's LastSeen (re-stamped every drive change);
		// fall back to UpdatedAt when a row carried only the latter.
		beat := d.LastSeen
		if beat.IsZero() {
			beat = d.UpdatedAt
		}
		pcb := descriptorPCBState(d)
		inv.Sessions = append(inv.Sessions, sessionInventoryRow{
			ID:            id,
			Host:          d.Host,
			PCBState:      pcb,
			LivenessClass: string(inventoryLiveness(pcb, beat, now, staleWindow)),
			CachePosture:  sessionCachePosture(idleSecondsSince(beat, now)),
			AgeSeconds:    ageSecondsSince(d.CreatedAt, now),
			ParentID:      descriptorParentID(d),
			Source:        "local",
		})
	}

	if isFleet {
		for _, d := range fleet {
			id := strings.TrimSpace(d.ID)
			if id == "" || seen[id] {
				continue // deduped against the richer local C1 row
			}
			seen[id] = true
			beat := time.Unix(d.UpdatedAt, 0)
			if d.UpdatedAt == 0 {
				beat = time.Time{}
			}
			pcb := strings.ToUpper(strings.TrimSpace(d.PCBState))
			inv.Sessions = append(inv.Sessions, sessionInventoryRow{
				ID:            id,
				Host:          d.Host,
				PCBState:      pcb,
				LivenessClass: string(inventoryLiveness(pcb, beat, now, staleWindow)),
				CachePosture:  sessionCachePosture(idleSecondsSince(beat, now)),
				AgeSeconds:    idleSecondsSince(beat, now), // a C2 ref carries only updated_at; age-since-update is the best it holds
				Source:        "fleet",
			})
		}
	}

	sort.Slice(inv.Sessions, func(i, j int) bool {
		if inv.Sessions[i].Host != inv.Sessions[j].Host {
			return inv.Sessions[i].Host < inv.Sessions[j].Host
		}
		return inv.Sessions[i].ID < inv.Sessions[j].ID
	})

	for _, r := range inv.Sessions {
		inv.Count++
		// The rollup counts the EFFECTIVE status: a running-family row whose heartbeat
		// lapsed rolls up as STALLED, so the headline "3 RUNNING, 1 STALLED" separates the
		// working from the wedged — while the row keeps its raw pcb_state + liveness_class.
		inv.ByState[inventoryEffectiveStatus(r.PCBState, taskmgr.LivenessClass(r.LivenessClass))]++
		switch r.CachePosture {
		case "warm":
			inv.Warm++
		case "cold":
			inv.Cold++
		}
	}
	return inv
}

// inventoryLiveness classifies a row with #750's witnessed vocabulary
// (internal/taskmgr.LivenessClass) off the durable heartbeat. A session that claims to be
// advancing (RUNNING/THROTTLED/DRAINING) but whose heartbeat has not moved within the
// stale window — or that has NO heartbeat at all — is STALLED (claims work, not
// progressing / garbage). PAUSED/STOPPED/unknown is idle. Pure over its inputs.
func inventoryLiveness(pcb string, beat, now time.Time, staleWindow time.Duration) taskmgr.LivenessClass {
	switch strings.ToUpper(strings.TrimSpace(pcb)) {
	case "RUNNING", "THROTTLED", "DRAINING":
		if beat.IsZero() {
			return taskmgr.LivenessStalled
		}
		if staleWindow > 0 && now.Sub(beat) > staleWindow {
			return taskmgr.LivenessStalled
		}
		return taskmgr.LivenessLive
	default:
		return taskmgr.LivenessIdle
	}
}

// inventoryEffectiveStatus is the status the count summary rolls up by: the PCB state,
// except a running-family row whose liveness lapsed reports STALLED so the headline number
// distinguishes "3 RUNNING" from a wedged "1 STALLED".
func inventoryEffectiveStatus(pcb string, class taskmgr.LivenessClass) string {
	p := strings.ToUpper(strings.TrimSpace(pcb))
	if p == "" {
		p = "UNKNOWN"
	}
	if class == taskmgr.LivenessStalled && (p == "RUNNING" || p == "THROTTLED" || p == "DRAINING") {
		return "STALLED"
	}
	return p
}

// descriptorPCBState reads the durable PCB state, preferring the persisted PCBState string
// and falling back to the typed Run so an older row that stored only Run still renders.
func descriptorPCBState(d session.Descriptor) string {
	if s := strings.TrimSpace(d.PCBState); s != "" {
		return strings.ToUpper(s)
	}
	return strings.ToUpper(d.Run.String())
}

// descriptorParentID returns the re-continuation lineage parent for a durable row: the old
// trace of the reset transaction that minted this session (internal/session.ResetTransaction
// binds old→new trace on a budget-reset re-continuation). A session not produced by a reset
// carries no parent, so this is empty (rendered "-") — the honest "no parent" answer, not a
// missing one.
func descriptorParentID(d session.Descriptor) string {
	return strings.TrimSpace(d.ResetTransaction.OldTrace)
}

// sessionCachePosture projects WARM/COLD (or "unknown") from idle-since-last-stamp via
// internal/resume.Plan — the deterministic idle-vs-TTL projection the AC names, not an
// observation of the provider's cache. A negative idle (no timestamp) is "unknown".
func sessionCachePosture(idleSeconds int64) string {
	rep := resume.Plan(resume.Input{IdleSeconds: idleSeconds, TTL: resume.TTL5m})
	switch rep.Posture {
	case resume.PostureWarm, resume.PostureWarmHit:
		return "warm"
	case resume.PostureCold:
		return "cold"
	default:
		return "unknown"
	}
}

// idleSecondsSince returns how long since t as whole seconds, or -1 (unknown) for a zero
// timestamp — the sentinel resume.Plan reads as "cannot tell warm from cold".
func idleSecondsSince(t, now time.Time) int64 {
	if t.IsZero() {
		return -1
	}
	d := now.Sub(t)
	if d < 0 {
		d = 0
	}
	return int64(d / time.Second)
}

// ageSecondsSince returns the session age in whole seconds, or 0 for a zero created-at.
func ageSecondsSince(t, now time.Time) int64 {
	if t.IsZero() {
		return 0
	}
	d := now.Sub(t)
	if d < 0 {
		d = 0
	}
	return int64(d / time.Second)
}

// renderSessionInventory prints the count summary headline followed by the aligned table.
// The summary comes FIRST so the headline number ("how many, and how many warm") is the
// first thing an operator reads.
func renderSessionInventory(w io.Writer, inv sessionInventory) {
	fmt.Fprintln(w, sessionInventorySummary(inv))
	if inv.Count == 0 {
		return
	}
	tw := tabwriter.NewWriter(w, 0, 2, 2, ' ', 0)
	fmt.Fprintln(tw, "ID\tHOST\tSTATE\tLIVENESS\tCACHE\tAGE\tPARENT\tSOURCE")
	for _, r := range inv.Sessions {
		parent := r.ParentID
		if parent == "" {
			parent = "-"
		}
		host := r.Host
		if host == "" {
			host = "-"
		}
		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n",
			r.ID, host, r.PCBState, r.LivenessClass, r.CachePosture,
			(time.Duration(r.AgeSeconds) * time.Second).String(), parent, r.Source)
	}
	_ = tw.Flush()
}

// sessionInventorySummary renders the one-line headline: total, per-state counts in
// canonical PCB order, then the warm (and, when present, cold) cache tally. Example:
// "5 session(s) [fleet]: 3 RUNNING, 1 PAUSED, 1 STOPPED — 4 warm, 1 cold".
func sessionInventorySummary(inv sessionInventory) string {
	scope := ""
	if inv.Fleet {
		scope = " [fleet]"
	}
	if inv.Count == 0 {
		return "no sessions" + scope
	}
	states := make([]string, 0, len(inv.ByState))
	for s := range inv.ByState {
		states = append(states, s)
	}
	sort.Slice(states, func(i, j int) bool {
		oi, iok := pcbStateOrder[states[i]]
		oj, jok := pcbStateOrder[states[j]]
		if iok && jok {
			return oi < oj
		}
		if iok != jok {
			return iok // known states before unknown ones
		}
		return states[i] < states[j]
	})
	parts := make([]string, 0, len(states))
	for _, s := range states {
		parts = append(parts, fmt.Sprintf("%d %s", inv.ByState[s], s))
	}
	cache := fmt.Sprintf("%d warm", inv.Warm)
	if inv.Cold > 0 {
		cache += fmt.Sprintf(", %d cold", inv.Cold)
	}
	return fmt.Sprintf("%d session(s)%s: %s — %s", inv.Count, scope, strings.Join(parts, ", "), cache)
}
