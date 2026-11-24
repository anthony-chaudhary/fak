package routing

// TaskCandidate is the routing fabric's OWN view of a unit of work to place:
// just the identity and the placement-relevant attributes (lane, paths). It is
// deliberately NOT platform/dispatch.Candidate.
//
// The separation is load-bearing. The control plane (platform/dispatch) owns
// the full schedulable candidate — priority, state, contracts, GitHub identity,
// lifecycle machinery — and CONSUMES the fabric. If the fabric imported that
// model, the dependency direction would invert and the two planes would fuse,
// dragging leases and landing into every routing test (see
// docs/notes/ROUTING-FABRIC-VS-CONTROL-PLANE-2026-09-18.md). So the fabric
// carries only the inert slice of task identity it actually routes over, and
// the control plane maps its richer Candidate into a TaskCandidate at the call
// site.
type TaskCandidate struct {
	// ID is the task/ticket identity. A candidate with an empty ID still
	// participates (a nil candidate is what marks a request unrouted), but an
	// empty identity places no lane.
	ID string `json:"id,omitempty"`
	// Lane is the file-tree lane the work belongs to, when known.
	Lane string `json:"lane,omitempty"`
	// Paths is the changed-path set, when known, for path-sensitive placement.
	Paths []string `json:"paths,omitempty"`
}

// GetID returns the candidate identity, or "" for a nil candidate.
func (c *TaskCandidate) GetID() string {
	if c == nil {
		return ""
	}
	return c.ID
}
