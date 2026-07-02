package dispatchauto

import (
	"fmt"
	"sort"
	"strings"
)

// Node is one host that can run workers. Today the fleet dispatches on the
// local box only, so callers pass a single node (or none — an empty roster
// means "one implicit node with no seat ceiling"). The axis exists so the SAME
// fold balances a many-node fleet: a node's contribution to the wave is its
// seat headroom, and placement fills the least-utilized healthy node first.
type Node struct {
	Name    string `json:"name"`
	SeatCap int    `json:"seat_cap"` // max concurrent workers this node may host (0 = no per-node ceiling)
	Live    int    `json:"live"`     // live workers on this node
	Healthy bool   `json:"healthy"`
}

// Input is the complete set of live ceilings an auto-sized wave folds. Every
// field is a plain measurement the caller already has (preflight, the account
// switcher, the issue router); the fold itself does no I/O and reads no clock.
//
// Zero-value semantics are split by what a zero MEANS:
//   - DistinctPools and ReadyWork are hard facts: 0 pools or 0 ready units
//     really does mean "no wave" (Target 0).
//   - EffectiveCap, RequiredWorkers, and SharedContextTokens use 0 = "unset":
//     an absent ceiling never binds.
type Input struct {
	// EffectiveCap is the preflight's already-folded population ceiling
	// (min of configured max-workers, kernel lease target, host resource
	// cap, seat total). 0 = no preflight view (never binds).
	EffectiveCap int `json:"effective_cap"`
	// LiveWorkers is the current live worker population.
	LiveWorkers int `json:"live_workers"`
	// DistinctPools is how many DISTINCT, fresh rate-limit account pools the
	// switcher can allocate right now. A wave wider than this collapses onto
	// shared usage buckets and serializes, so it always binds.
	DistinctPools int `json:"distinct_pools"`
	// ReadyWork is how many dispatchable work units are ready (routed open
	// issues, ready leaves). Workers beyond this have nothing to take.
	ReadyWork int `json:"ready_work"`
	// RequiredWorkers is an optional throughput target (e.g. Little's-law
	// required concurrency). 0 = unset.
	RequiredWorkers int `json:"required_workers"`
	// Nodes is the optional many-node roster. Empty = one implicit local
	// node with no per-node seat ceiling.
	Nodes []Node `json:"nodes,omitempty"`
	// SharedContextTokens is an optional fleet-wide context-token budget to
	// slice evenly across the wave, so each worker starts with an explicit
	// context envelope instead of an unbounded one. 0 = unset.
	SharedContextTokens int `json:"shared_context_tokens"`
}

// Assignment places one to-be-launched worker on a node, carrying its context
// slice. Seq is the launch order (0-based).
type Assignment struct {
	Seq           int    `json:"seq"`
	Node          string `json:"node"`
	ContextTokens int    `json:"context_tokens,omitempty"`
}

// Plan is the auto-sizing decision: how many workers the fleet should be
// running (Target), how many to launch now to converge (Refill), and where.
type Plan struct {
	// Target is the steady-state worker population: the minimum of every
	// ceiling that is set, and of the hard facts (pools, ready work).
	Target int `json:"target"`
	// Refill is what to launch THIS tick: max(0, Target-LiveWorkers). A
	// caller that runs on a cadence converges the population to Target and
	// tops it back up as workers exit — the steady-state refill the one-shot
	// wave verbs lack.
	Refill int `json:"refill"`
	// Binding names the ceiling that bound Target (the vocabulary of
	// Ceilings keys), so "why only N?" is always answerable.
	Binding string `json:"binding"`
	// Ceilings records every term consulted: the set ones and the hard
	// facts, by name.
	Ceilings map[string]int `json:"ceilings"`
	// PerWorkerContextTokens is the even context slice each worker gets
	// (SharedContextTokens/Target), 0 when either side is unset/zero.
	PerWorkerContextTokens int `json:"per_worker_context_tokens,omitempty"`
	// Assignments places the Refill workers on healthy nodes, least
	// utilized first. Empty when Refill is 0.
	Assignments []Assignment `json:"assignments,omitempty"`
	Reason      string       `json:"reason"`
}

// Ceiling names (the closed vocabulary of Plan.Binding / Plan.Ceilings keys).
const (
	CeilingEffectiveCap    = "effective_cap"
	CeilingDistinctPools   = "distinct_pools"
	CeilingReadyWork       = "ready_work"
	CeilingRequiredWorkers = "required_workers"
	CeilingNodeHeadroom    = "node_headroom"
)

// PlanAuto folds the live ceilings into the auto-sized wave decision. Pure and
// deterministic: same Input in, same Plan out — no clock, no I/O.
func PlanAuto(in Input) Plan {
	ceilings := map[string]int{
		CeilingDistinctPools: in.DistinctPools,
		CeilingReadyWork:     in.ReadyWork,
	}
	// The binding scan walks a FIXED order so ties break deterministically,
	// hard facts first (they are the honest headline when equal).
	order := []string{CeilingDistinctPools, CeilingReadyWork}
	if in.EffectiveCap > 0 {
		ceilings[CeilingEffectiveCap] = in.EffectiveCap
		order = append(order, CeilingEffectiveCap)
	}
	if in.RequiredWorkers > 0 {
		ceilings[CeilingRequiredWorkers] = in.RequiredWorkers
		order = append(order, CeilingRequiredWorkers)
	}
	nodes := healthyNodes(in.Nodes)
	if headroom, bounded := nodeHeadroom(nodes, len(in.Nodes) > 0); bounded {
		ceilings[CeilingNodeHeadroom] = headroom
		order = append(order, CeilingNodeHeadroom)
	}

	target := -1
	binding := ""
	for _, name := range order {
		if target < 0 || ceilings[name] < target {
			target = ceilings[name]
			binding = name
		}
	}
	if target < 0 {
		target = 0
	}

	refill := target - in.LiveWorkers
	if refill < 0 {
		refill = 0
	}

	perWorker := 0
	if in.SharedContextTokens > 0 && target > 0 {
		perWorker = in.SharedContextTokens / target
	}

	return Plan{
		Target:                 target,
		Refill:                 refill,
		Binding:                binding,
		Ceilings:               ceilings,
		PerWorkerContextTokens: perWorker,
		Assignments:            placeWorkers(nodes, refill, perWorker),
		Reason:                 planReason(target, refill, binding, in.LiveWorkers),
	}
}

func healthyNodes(nodes []Node) []Node {
	out := make([]Node, 0, len(nodes))
	for _, n := range nodes {
		if n.Healthy {
			out = append(out, n)
		}
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}

// nodeHeadroom sums the healthy nodes' remaining seats. It reports bounded
// only when a roster was supplied: an empty roster means the caller has no
// node view, and an absent ceiling never binds. A supplied roster with no
// healthy node is a genuine 0.
func nodeHeadroom(healthy []Node, rosterSupplied bool) (int, bool) {
	if !rosterSupplied {
		return 0, false
	}
	total := 0
	unbounded := false
	for _, n := range healthy {
		if n.SeatCap <= 0 {
			unbounded = true
			continue
		}
		room := n.SeatCap - n.Live
		if room > 0 {
			total += room
		}
	}
	if unbounded && len(healthy) > 0 {
		// At least one healthy node has no per-node ceiling, so the roster
		// does not bound the wave.
		return 0, false
	}
	return total, true
}

// placeWorkers fills the least-utilized healthy node first (utilization is
// Live/SeatCap compared cross-multiplied, so no floats; an uncapped node
// counts as utilization 0 until it receives work). Ties break by node name so
// placement is deterministic.
func placeWorkers(healthy []Node, refill, perWorker int) []Assignment {
	if refill <= 0 {
		return nil
	}
	name := "local"
	if len(healthy) == 0 {
		out := make([]Assignment, refill)
		for i := range out {
			out[i] = Assignment{Seq: i, Node: name, ContextTokens: perWorker}
		}
		return out
	}
	load := make([]Node, len(healthy))
	copy(load, healthy)
	out := make([]Assignment, 0, refill)
	for seq := 0; seq < refill; seq++ {
		best := -1
		for i := range load {
			if load[i].SeatCap > 0 && load[i].Live >= load[i].SeatCap {
				continue // node full
			}
			if best < 0 || lessUtilized(load[i], load[best]) {
				best = i
			}
		}
		if best < 0 {
			break // every node full; the node_headroom ceiling already bounded Target, this is belt and braces
		}
		out = append(out, Assignment{Seq: seq, Node: load[best].Name, ContextTokens: perWorker})
		load[best].Live++
	}
	return out
}

// lessUtilized reports whether a is strictly less utilized than b, treating an
// uncapped node's utilization as Live/+inf capacity (i.e. compare by absolute
// live count against another uncapped node, and as ratio 0-ish vs a capped
// one). Cross-multiplication avoids float drift: a.Live/a.Cap < b.Live/b.Cap
// ⇔ a.Live*b.Cap < b.Live*a.Cap.
func lessUtilized(a, b Node) bool {
	switch {
	case a.SeatCap <= 0 && b.SeatCap <= 0:
		if a.Live != b.Live {
			return a.Live < b.Live
		}
	case a.SeatCap <= 0:
		return true
	case b.SeatCap <= 0:
		return false
	default:
		la, lb := a.Live*b.SeatCap, b.Live*a.SeatCap
		if la != lb {
			return la < lb
		}
	}
	return a.Name < b.Name
}

func planReason(target, refill int, binding string, live int) string {
	if target == 0 {
		return fmt.Sprintf("no wave: %s is 0", binding)
	}
	if refill == 0 {
		return fmt.Sprintf("converged: %d live >= target %d (bound by %s)", live, target, binding)
	}
	return fmt.Sprintf("refill %d worker(s): %d live, target %d (bound by %s)", refill, live, target, binding)
}

// String renders the plan as one operator line.
func (p Plan) String() string {
	parts := make([]string, 0, len(p.Ceilings))
	names := make([]string, 0, len(p.Ceilings))
	for name := range p.Ceilings {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		parts = append(parts, fmt.Sprintf("%s=%d", name, p.Ceilings[name]))
	}
	return fmt.Sprintf("target=%d refill=%d binding=%s (%s)", p.Target, p.Refill, p.Binding, strings.Join(parts, " "))
}
