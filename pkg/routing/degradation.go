package routing

import (
	"fmt"
)

// ChainSchema identifies the degradation-chain manifest/receipt shape.
const ChainSchema = "routing-degradation/1"

// Rung is one named step in a degradation ladder: a capability contract matched
// against engines at routing time, not a concrete engine.
type Rung struct {
	// Name is the rung's identity in the audit trail. Required.
	Name string `json:"name"`
	// Surface narrows the rung to one execution surface. Empty means any.
	Surface Surface `json:"surface,omitempty"`
	// Model narrows the rung to one model handle (exact match). Empty is any.
	Model string `json:"model,omitempty"`
	// Tags are capability tags an engine must carry to fill this rung.
	// Empty means any engine matches.
	Tags []string `json:"tags,omitempty"`
}

// DegradationChain is the ordered ladder of rungs from the primary model down to
// the small always-on maintenance model, so the last rung still answers when the
// big models are down. It is pure data; Order and Plan never mutate it.
type DegradationChain struct {
	// Version is the policy/manifest version tag.
	Version string `json:"version"`
	// Rungs are evaluated in declared order; the last is the maintenance rung.
	Rungs []Rung `json:"rungs"`
}

// DefaultDegradationChain returns the standard ladder: "primary" (engines tagged
// "primary"), "fallback-small" (local engines tagged "small") and "maintenance"
// (any local engine). Only the final rung may be unconstrained: an unconstrained
// non-final rung would match every engine and nothing would ever demote.
func DefaultDegradationChain() DegradationChain {
	return DegradationChain{
		Version: ChainSchema,
		Rungs: []Rung{
			{Name: "primary", Tags: []string{"primary"}},
			{Name: "fallback-small", Surface: SurfaceLocal, Tags: []string{"small"}},
			{Name: "maintenance", Surface: SurfaceLocal},
		},
	}
}

// Validate checks the ladder's structural invariants and fails on the first
// violation: at least two rungs, named rungs, known surfaces, and a constraint on
// every non-final rung (the final rung is the catch-all by design).
func (c DegradationChain) Validate() error {
	if len(c.Rungs) < 2 {
		return fmt.Errorf("routing: degradation chain has %d rung(s); a ladder needs at least 2", len(c.Rungs))
	}
	for i, r := range c.Rungs {
		if r.Name == "" {
			return fmt.Errorf("routing: degradation chain rung %d has no name", i)
		}
		if r.Surface != "" && !r.Surface.IsKnown() {
			return fmt.Errorf("routing: degradation chain rung %q surface %q is not known", r.Name, r.Surface)
		}
		if i < len(c.Rungs)-1 && r.Surface == "" && r.Model == "" && len(r.Tags) == 0 {
			return fmt.Errorf("routing: degradation rung %q has no constraint (a non-final rung must constrain surface, model, or tags)", r.Name)
		}
	}
	return nil
}

// MaintenanceRung returns the last rung, or false when the chain is empty.
func (c DegradationChain) MaintenanceRung() (Rung, bool) {
	if len(c.Rungs) == 0 {
		return Rung{}, false
	}
	return c.Rungs[len(c.Rungs)-1], true
}

func matchesRung(e Engine, r Rung) bool {
	if r.Surface != "" && e.Surface != r.Surface {
		return false
	}
	if r.Model != "" && e.Model != r.Model {
		return false
	}
	return e.HasTag(r.Tags)
}

// Order returns the engines by rung, then engines matching no rung, then the
// maintenance rung last; declared order is kept within each group. Unmatched
// engines precede maintenance because they are presumed more capable than the
// deliberately small maintenance model.
func (c DegradationChain) Order(engines []Engine) []Engine {
	groups := make([][]Engine, len(c.Rungs)+1)
	for _, e := range engines {
		matched := len(c.Rungs) // no-rung bucket
		for i, r := range c.Rungs {
			if matchesRung(e, r) {
				matched = i
				break
			}
		}
		groups[matched] = append(groups[matched], e)
	}

	out := make([]Engine, 0, len(engines))
	for i := 0; i < len(c.Rungs)-1; i++ {
		out = append(out, groups[i]...)
	}
	out = append(out, groups[len(c.Rungs)]...)
	if len(c.Rungs) > 0 {
		out = append(out, groups[len(c.Rungs)-1]...)
	}
	return out
}

// RungPlan is one rung's legible receipt entry.
type RungPlan struct {
	// Name is the rung's identity.
	Name string `json:"name"`
	// Surface is the rung's surface constraint ("" = any).
	Surface Surface `json:"surface,omitempty"`
	// Model is the rung's model constraint ("" = any).
	Model string `json:"model,omitempty"`
	// Maintenance is true only for the last rung.
	Maintenance bool `json:"maintenance,omitempty"`
	// Eligible lists engine IDs whose first-matching rung is this one.
	Eligible []string `json:"eligible,omitempty"`
}

// DegradationPlan is the receipt answering "what degrades to what"; Plan selects
// nothing and changes nothing.
type DegradationPlan struct {
	// Schema identifies the receipt shape.
	Schema string `json:"schema"`
	// Chain names the chain version the plan was computed against.
	Chain string `json:"chain"`
	// Rungs are the per-rung plans in declared order.
	Rungs []RungPlan `json:"rungs"`
	// Audit is the legible step-by-step trail, one line per rung.
	Audit []string `json:"audit,omitempty"`
}

// Plan renders which engines fill which rung, each attributed to its first
// matching rung, with one audit line per rung.
func (c DegradationChain) Plan(engines []Engine) DegradationPlan {
	plan := DegradationPlan{
		Schema: ChainSchema,
		Chain:  c.Version,
		Rungs:  make([]RungPlan, 0, len(c.Rungs)),
	}
	for i, r := range c.Rungs {
		rp := RungPlan{
			Name:        r.Name,
			Surface:     r.Surface,
			Model:       r.Model,
			Maintenance: i == len(c.Rungs)-1,
		}
		for _, e := range engines {
			if c.firstRung(e) == i {
				rp.Eligible = append(rp.Eligible, e.ID)
			}
		}
		plan.Rungs = append(plan.Rungs, rp)

		switch {
		case len(rp.Eligible) > 0:
			plan.Audit = append(plan.Audit, fmt.Sprintf(
				"rung %d %q: %d engine(s) eligible [%s]",
				i+1, r.Name, len(rp.Eligible), joinIDs(rp.Eligible)))
		case rp.Maintenance:
			plan.Audit = append(plan.Audit, fmt.Sprintf(
				"rung %d %q: no engine eligible; maintenance interface unavailable",
				i+1, r.Name))
		default:
			plan.Audit = append(plan.Audit, fmt.Sprintf(
				"rung %d %q: no engine eligible; fallback to next rung",
				i+1, r.Name))
		}
	}
	return plan
}

// firstRung returns len(c.Rungs) when the engine matches no rung.
func (c DegradationChain) firstRung(e Engine) int {
	for i, r := range c.Rungs {
		if matchesRung(e, r) {
			return i
		}
	}
	return len(c.Rungs)
}

func joinIDs(ids []string) string {
	out := ""
	for i, id := range ids {
		if i > 0 {
			out += " "
		}
		out += id
	}
	return out
}
