// Package dosdecision revalidates DOS decision rows against the kernel's live
// lane-lease set.
//
// An `ARBITER_REFUSE` row is not a point-in-time artifact whose staleness is its
// age: it is a refusal *relative to a live contended lane* ("lane L cannot share
// live lane B", "lane L is already held by a live loop"). The moment the blocking
// lease is released, re-requesting L would be admitted, so the row is resolved —
// it is no longer human work. The kernel supersedes such a row when a LATER
// RELEASE/SCAVENGE entry survives in the lane journal, but a released lease whose
// freeing entry never reached the journal (a truncated/rotated/corrupt WAL tail)
// leaves the refusal standing forever: `dos lease-lane live` reports zero live
// leases while `dos decisions --all --json` still shows hours-old HUMAN rows
// citing lanes nobody holds (#6494).
//
// This package closes that gap on the read side: it revalidates each
// lease-dependent refusal against the CURRENT live set rather than against the
// journal's memory of how the lease ended. Nothing is deleted — a superseded row
// is annotated and moved to the history bucket, which the caller can still render
// explicitly.
package dosdecision

import (
	"fmt"
	"regexp"
	"sort"
	"strings"
)

// KindArbiterRefuse is the only decision kind whose liveness this package judges.
// Every other kind (LIVENESS, WEDGE, SOAK_GATE, host queue items, …) is passed
// through untouched: their resolution has nothing to do with a lane lease.
const KindArbiterRefuse = "ARBITER_REFUSE"

// ResolutionLeaseReleased is written into a superseded row's `resolution` field.
const ResolutionLeaseReleased = "LEASE_RELEASED"

// Row is one decision as the kernel (or the host adapter) emitted it. It is kept
// as the raw object so every field a future DOS release adds survives the
// round trip; this package only reads `kind`/`lane`/`reason_text` and adds the
// resolution annotation.
type Row map[string]any

// LiveSet is the current lane-lease live set (`dos lease-lane live`).
//
// Known distinguishes "the kernel answered, and holds zero leases" from "we could
// not read the kernel at all". Only the first may clear a refusal: an unreadable
// kernel must never be mistaken for an empty one, or a decision queue would empty
// itself the moment `dos` is missing from PATH.
type LiveSet struct {
	Lanes []string
	Known bool
}

// Result is the partition Revalidate produced.
type Result struct {
	// Active are the rows that are still unresolved work, in input order.
	Active []Row
	// Superseded are the lease-dependent refusals whose blocking lease is gone,
	// annotated with the resolution, in input order.
	Superseded []Row
	// Cleared counts the rows this call moved from Active to Superseded — the
	// cleanup number an unstick/replan loop reports so the work is measurable.
	Cleared int
}

// laneRefRe lifts every lane NAME a refusal's prose quotes: the kernel writes
// `lane 'X' cannot share live lane 'Y'`, `lane 'X' is already held by a live
// loop`, and `an exclusive lane is live (lane='Y', kind='K', …)`. Matching the
// generic `lane <quoted>` shape over-collects rather than under-collects, which is
// the safe direction: an extra watched lane can only keep a row ACTIVE.
var laneRefRe = regexp.MustCompile(`(?i)lane[\s=]+['"]([^'"]{1,120})['"]`)

// clusterDecorationRe strips the curated-cluster relic tail the kernel's own
// `_dynamic_lane_handle` drops, so `apply cluster (AFR, ALO)` and `apply` compare
// equal.
var clusterDecorationRe = regexp.MustCompile(`\s*(?:cluster\s*)?\([^)]*\)\s*$`)

// LaneKey normalizes a lane name for comparison: cluster decoration removed, the
// last path segment kept, case-folded. Decision rows carry the bare dynamic lane
// handle while live-lease records carry the raw lane name, so both sides are
// normalized before the membership test.
func LaneKey(s string) string {
	v := strings.TrimSpace(s)
	if v == "" {
		return ""
	}
	v = clusterDecorationRe.ReplaceAllString(v, "")
	v = strings.TrimSpace(v)
	if i := strings.LastIndexAny(v, "/\\"); i >= 0 {
		v = v[i+1:]
	}
	return strings.ToLower(strings.TrimSpace(v))
}

// holds reports whether the live set still holds lane.
func (s LiveSet) holds(lane string) bool {
	key := LaneKey(lane)
	if key == "" {
		return false
	}
	for _, l := range s.Lanes {
		if LaneKey(l) == key {
			return true
		}
	}
	return false
}

// BlockingLanes is the set of lanes a refusal is waiting on: the lanes its prose
// names plus the refused lane itself. The refused lane belongs in the set because
// an "already held by a live loop" refusal names no other lane and is resolved
// when the prior holder of that same lane lets go. The result is sorted so the
// resolution evidence is deterministic.
func BlockingLanes(r Row) []string {
	seen := map[string]bool{}
	add := func(s string) {
		if k := LaneKey(s); k != "" && !seen[k] {
			seen[k] = true
		}
	}
	add(rowString(r, "lane"))
	for _, m := range laneRefRe.FindAllStringSubmatch(rowString(r, "reason_text"), -1) {
		add(m[1])
	}
	out := make([]string, 0, len(seen))
	for k := range seen {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// NeedsLiveSet reports whether any row's status depends on the live lane-lease
// set. A caller reading the kernel is doing a subprocess round trip, so a page of
// LIVENESS/host rows should not pay for it.
func NeedsLiveSet(rows []Row) bool {
	for _, r := range rows {
		if !strings.EqualFold(rowString(r, "kind"), KindArbiterRefuse) {
			continue
		}
		if resolved, ok := r["resolved"].(bool); ok && resolved {
			continue
		}
		return true
	}
	return false
}

// Revalidate partitions rows into what is still unresolved and what the live set
// proves is over.
//
// A row is moved to Superseded only when all of the following hold: the live set
// is Known, the row is an ARBITER_REFUSE that is not already resolved, at least
// one blocking lane could be identified, and NONE of those lanes is live. A
// refusal whose blockers cannot be identified stays Active — an undecidable row is
// never silently dropped.
func Revalidate(rows []Row, live LiveSet) Result {
	res := Result{Active: []Row{}, Superseded: []Row{}}
	for _, r := range rows {
		if r == nil {
			continue
		}
		if !live.Known || !strings.EqualFold(rowString(r, "kind"), KindArbiterRefuse) {
			res.Active = append(res.Active, r)
			continue
		}
		if resolved, ok := r["resolved"].(bool); ok && resolved {
			// Already history when it reached us: keep it out of the active
			// queue, but it is not cleanup this call performed.
			res.Superseded = append(res.Superseded, r)
			continue
		}
		blockers := BlockingLanes(r)
		if len(blockers) == 0 {
			res.Active = append(res.Active, r)
			continue
		}
		stillHeld := []string{}
		for _, lane := range blockers {
			if live.holds(lane) {
				stillHeld = append(stillHeld, lane)
			}
		}
		if len(stillHeld) > 0 {
			res.Active = append(res.Active, r)
			continue
		}
		res.Superseded = append(res.Superseded, supersede(r, blockers))
		res.Cleared++
	}
	return res
}

// supersede returns an annotated COPY of the row: the caller's map is never
// mutated, and every original field is preserved so `--all` renders real history
// rather than a stub.
func supersede(r Row, blockers []string) Row {
	out := make(Row, len(r)+3)
	for k, v := range r {
		out[k] = v
	}
	out["resolved"] = true
	out["resolution"] = ResolutionLeaseReleased
	out["resolution_evidence"] = fmt.Sprintf(
		"blocking lane lease(s) %s are absent from `dos lease-lane live`; the collision this row records is over",
		strings.Join(blockers, ", "))
	return out
}

// rowString reads a string field, tolerating a missing or wrongly-typed value.
func rowString(r Row, key string) string {
	s, _ := r[key].(string)
	return s
}
