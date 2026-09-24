// rotation.go â€” the GENERAL same-class model rotation primitive behind the
// ops-desk preset (issue #13502 revision 2). A desk declares CLASS pools: routed
// ids that are similar-class models (same hosted flash/hosted tier) a provider
// carries together, so the desk can ROTATE through them instead of dead-ending on
// one seat â€” e.g. hive-ai hosts BOTH zai-org/glm-5.3-flash AND
// deepseek-ai/DeepSeek-V4.1-Flash on one OpenAI-compatible wire, which witnessed
// the GLM<->DeepSeek rotation. Astra (gpt-6-astra) and deepseek-v4-pro are NOT
// available; they belong to no pool.
//
// Rotation lives at FOUR levels in modelroute, each a different seam:
//  1. aspect-level alternation: the routing MANIFEST bands work by shape
//     (complexity/prompt-window/latency/labels), alternating among the class
//     members deterministically per aspect â€” the ops-desk rules.
//  2. seat-level rotation within a class: THIS FILE (NextSeat/RotationPeers),
//     the pure round-robin step dispatch wiring applies per call when it wants
//     to spread load or step off a degraded/exhausted seat.
//  3. account/rung-level failover: the roster's Accounts â€” several endpoints
//     (hive-ai, Nebius, Modal rungs) may carry the SAME upstream id under
//     different credentials; Roster.Resolve picks the endpoint, placement.go
//     and serving.go own the live failover between rungs.
//  4. free-tier rotation: freetier.go's FreeFallbackChain (per-id try-order
//     chains the wiring walks on 429/quota exhaustion).
//
// This package stays PURE (no provider probes, no quota state, no I/O); levels
// 2..4 hand ordered CANDIDATES to the dispatch wiring, which owns execution.
package modelroute

import "fmt"

// RotationGroup is one similar-class pool: ordered routed ids that stand in for
// each other. Seats are ROUTED ids (manifest member model id), not wire names â€”
// the roster binds each to its account/upstream separately.
type RotationGroup struct {
	Class string   `json:"class"`
	Seats []string `json:"seats"`
}

// deskRotation declares the ops-desk pools. The desk's one class: BOTH seats are
// same-class HOSTED flash models and the same provider carries both â€” the
// witnessed hive-ai setup â€” so either can rotate onto the other.
var deskRotation = []RotationGroup{
	{
		Class: "hosted-flash",
		Seats: []string{"glm-5.3-flash", "deepseek-v4.1-flash"},
	},
}

// safeRotationSeat applies the routing-token rule to a seat id: seats live in
// manifests and plans, so they must be plain tokens (the routed-id delimiter
// rule, mirroring binding-model validation).
func safeRotationSeat(s string) error {
	if s == "" {
		return fmt.Errorf("modelroute: rotation seat is an empty token")
	}
	if err := safeRouteToken("binding model", s); err != nil {
		return fmt.Errorf("modelroute: rotation seat %q must be a plain routed id: %w", s, err)
	}
	return nil
}

// ValidateRotationGroups is the fail-loud gate over declared pools: a seat id
// belongs to EXACTLY ONE class (membership in two pools makes the rotation
// ambiguous), classes are unique, and every seat is a plain routed token.
func ValidateRotationGroups(groups []RotationGroup) error {
	classes := make(map[string]bool, len(groups))
	seats := make(map[string]string, 2*len(groups))
	for i, g := range groups {
		if g.Class == "" {
			return fmt.Errorf("modelroute: rotation group %d has an empty class", i)
		}
		if classes[g.Class] {
			return fmt.Errorf("modelroute: duplicate rotation class %q", g.Class)
		}
		classes[g.Class] = true
		if len(g.Seats) < 2 {
			return fmt.Errorf("modelroute: rotation class %q has %d seats; a pool needs >= 2 to be a rotation", g.Class, len(g.Seats))
		}
		for _, s := range g.Seats {
			if err := safeRotationSeat(s); err != nil {
				return err
			}
			if prior, dup := seats[s]; dup {
				return fmt.Errorf("modelroute: rotation seat %q belongs to two classes (%q and %q) â€” membership must be unambiguous", s, prior, g.Class)
			}
			seats[s] = g.Class
		}
	}
	return nil
}

// RotationClass returns the similar-class name of the pool containing id.
// ok is false for an id declared in NO pool (undeclared ids are not rotation
// candidates â€” e.g. a deleted/unavailable seat must not rotate anywhere).
func RotationClass(id string) (string, bool) {
	for _, g := range deskRotation {
		for _, s := range g.Seats {
			if s == id {
				return g.Class, true
			}
		}
	}
	return "", false
}

// RotationPeers returns the same-class alternates for id, in declared order,
// with id itself excluded. The returned slice is a COPY (callers may reorder or
// filter it for wiring). ok is false when id is declared in no pool.
func RotationPeers(id string) ([]string, bool) {
	for _, g := range deskRotation {
		for _, s := range g.Seats {
			if s != id {
				continue
			}
			peers := make([]string, 0, len(g.Seats)-1)
			for _, p := range g.Seats {
				if p != id {
					peers = append(peers, p)
				}
			}
			return peers, true
		}
	}
	return nil, false
}

// NextSeat is the pure round-robin STEP across a class pool: the seat AFTER
// current among its similar-class peers, cyclic (rotation wraps to the pool
// head). Pure over the declared pools â€” no state, no arrival time, so the
// deterministic Route spine is untouched: the WIRING decides when to apply a
// step (load spread, exhaustion, degraded seat); the kernel only provides the
// order. ok is false when current is declared in no pool.
func NextSeat(current string) (next, class string, ok bool) {
	for _, g := range deskRotation {
		for i, s := range g.Seats {
			if s != current {
				continue
			}
			return g.Seats[(i+1)%len(g.Seats)], g.Class, true
		}
	}
	return "", "", false
}

// DeskRotationGroups returns a copy of the declared desk pools for rendering and
// audit surfaces (the registry never leaks the mutable backing slice).
func DeskRotationGroups() []RotationGroup {
	out := make([]RotationGroup, len(deskRotation))
	for i, g := range deskRotation {
		out[i] = RotationGroup{Class: g.Class, Seats: append([]string(nil), g.Seats...)}
	}
	return out
}
