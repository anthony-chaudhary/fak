package dispatchtick

import (
	"fmt"
	"strings"
)

// HostEnrollment is the pure, I/O-free descriptor the cmd/fak dispatch shell hands a
// running in-process microagent host (internal/microagent, M2) to Spawn one routed
// issue as a single microagent — the #2030 gen/second-next alternative to
// exec-spawning a detached guarded CLI per lane.
//
// It carries the SAME lease identity and fence tree the detached spawn path would hold
// (PlanHostEnrollment copies them verbatim), so lane-lease disjointness / per-agent
// audit (M11) is preserved BY CONSTRUCTION: whichever backend runs the lane, the same
// lane lease over the same tree serializes it against every peer, so the in-process
// path can never collide where the detached path would not. The AgentID is the host's
// per-agent identity — unique per issue, because a host id is one agent lifetime.
type HostEnrollment struct {
	AgentID string   `json:"agent_id"`
	Lane    string   `json:"lane"`
	Issue   int      `json:"issue"`
	LeaseID string   `json:"lease_id"`
	Tree    []string `json:"tree"`
}

// PlanHostEnrollment builds the pure enrollment descriptor for one routed issue. The
// agent id is a stable, per-issue identity derived from the lane lease id (the same
// clean token the detached path leases on) suffixed with the issue number, so it is
// unique per issue rather than per lane — a host refuses a duplicate id, and one lane
// may enroll several issues over its lifetime. The tree is COPIED from the routed lease
// tree so the host holds EXACTLY the fence the detached path would (the M11
// disjointness invariant); mutating the caller's slice can never reach into the plan.
func PlanHostEnrollment(lane string, issue int, leaseID string, tree []string) HostEnrollment {
	id := strings.TrimSpace(leaseID)
	if id == "" {
		id = "resolve-" + laneToken(lane)
	}
	return HostEnrollment{
		AgentID: fmt.Sprintf("%s-%d", id, issue),
		Lane:    lane,
		Issue:   issue,
		LeaseID: leaseID,
		Tree:    append([]string(nil), tree...),
	}
}

// laneToken sanitizes a lane name into an id-safe token: it keeps letters, digits, '-'
// and '_' and folds every other rune to '-'. It is the pure fallback used only when no
// lease id is supplied (the detached path always supplies one), so an enrollment id is
// never derived from a raw path-shaped lane like "internal/foo".
func laneToken(lane string) string {
	lane = strings.TrimSpace(lane)
	if lane == "" {
		return "unknown"
	}
	var b strings.Builder
	for _, r := range lane {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9', r == '-', r == '_':
			b.WriteRune(r)
		default:
			b.WriteRune('-')
		}
	}
	return b.String()
}
