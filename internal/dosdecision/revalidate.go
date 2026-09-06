// Package dosdecision revalidates lease-dependent DOS decision rows against the
// kernel's live lane leases, moving resolved contention rows to history.
package dosdecision

import (
	"fmt"
	"regexp"
	"sort"
	"strings"
)

// KindArbiterRefuse is the decision kind whose contention liveness is judged.
const KindArbiterRefuse = "ARBITER_REFUSE"

// ResolutionLeaseReleased marks a decision superseded by a released lane lease.
const ResolutionLeaseReleased = "LEASE_RELEASED"

// Row is a raw decision object emitted by the kernel or host adapter.
type Row map[string]any

// LiveSet represents the kernel's active lane leases. Known is false when the
// live lease set could not be determined.
type LiveSet struct {
	Lanes []string
	Known bool
}

// Result is the partition produced by Revalidate.
type Result struct {
	Active     []Row
	Superseded []Row
	Cleared    int
}

var (
	laneRefRe           = regexp.MustCompile(`(?i)lane[\s=]+['"]([^'"]{1,120})['"]`)
	clusterDecorationRe = regexp.MustCompile(`\s*(?:cluster\s*)?\([^)]*\)\s*$`)
)

// LaneKey normalizes a lane name by stripping cluster decoration, keeping the
// terminal path segment, and lowercasing.
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

// BlockingLanes extracts all unique normalized lane keys a refusal depends on,
// including the refused lane itself, sorted deterministically.
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

// NeedsLiveSet reports whether any active row requires live-lease revalidation.
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

// Revalidate partitions rows into active work and superseded records whose
// blocking leases are no longer held.
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
			res.Superseded = append(res.Superseded, r)
			continue
		}
		blockers := BlockingLanes(r)
		if len(blockers) == 0 {
			res.Active = append(res.Active, r)
			continue
		}
		stillHeld := false
		for _, lane := range blockers {
			if live.holds(lane) {
				stillHeld = true
				break
			}
		}
		if stillHeld {
			res.Active = append(res.Active, r)
			continue
		}
		res.Superseded = append(res.Superseded, supersede(r, blockers))
		res.Cleared++
	}
	return res
}

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

func rowString(r Row, key string) string {
	s, _ := r[key].(string)
	return s
}
