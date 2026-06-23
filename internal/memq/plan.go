package memq

import (
	"fmt"
	"strings"
)

// PlanStep is one explained stage of a query: what the op does, whether it is an
// effect, whether it mutates durable state (and so needs a capability to APPLY), and
// the capability key that gates it.
type PlanStep struct {
	Index       int    `json:"index"`
	Kind        string `json:"kind"`
	Detail      string `json:"detail"`
	Effect      bool   `json:"effect"`
	Mutation    bool   `json:"mutation"`
	RequiresCap string `json:"requires_cap,omitempty"`
}

// Plan is the static explanation of a query — the "step through this before you run
// it" surface. It is produced WITHOUT a backend, so it commits to no reads or writes;
// it is a description of the pipeline the agent authored.
type Plan struct {
	Intent string     `json:"intent,omitempty"`
	Steps  []PlanStep `json:"steps"`
	// Effects lists the distinct effect kinds the pipeline would perform, and
	// Mutations the subset that change durable state (and so are proposal-only
	// without a Caps grant). An empty Mutations means the query is read-only.
	Effects   []string `json:"effects,omitempty"`
	Mutations []string `json:"mutations,omitempty"`
	Valid     bool     `json:"valid"`
	Error     string   `json:"error,omitempty"`
}

// Explain renders a query as a Plan without touching a backend. An invalid query
// still produces a Plan (Valid=false, Error set) so a caller can show the agent WHY
// its authored pipeline was refused, rather than a bare error.
func Explain(q Query) Plan {
	p := Plan{Intent: q.Intent, Valid: true}
	if err := Validate(q); err != nil {
		p.Valid = false
		p.Error = err.Error()
	}
	effSeen := map[string]bool{}
	mutSeen := map[string]bool{}
	for i, op := range q.Ops {
		step := PlanStep{
			Index:    i,
			Kind:     op.Kind,
			Detail:   detailOf(op),
			Effect:   IsEffect(op.Kind),
			Mutation: IsMutation(op.Kind),
		}
		if step.Mutation {
			step.RequiresCap = op.Kind
			if !mutSeen[op.Kind] {
				mutSeen[op.Kind] = true
				p.Mutations = append(p.Mutations, op.Kind)
			}
		}
		if step.Effect && !effSeen[op.Kind] {
			effSeen[op.Kind] = true
			p.Effects = append(p.Effects, op.Kind)
		}
		p.Steps = append(p.Steps, step)
	}
	return p
}

// detailOf is the human-readable one-liner for a step.
func detailOf(op Op) string {
	switch op.Kind {
	case OpScan:
		return "load every cell from the backend"
	case OpFilter:
		if op.Pred == nil {
			return "keep cells matching <nil predicate>"
		}
		return "keep cells where " + predString(*op.Pred)
	case OpRank:
		dir := "asc"
		if op.Desc {
			dir = "desc"
		}
		return fmt.Sprintf("sort by %s (%s)", op.By, dir)
	case OpLimit:
		return fmt.Sprintf("keep the first %d", op.K)
	case OpBudget:
		if op.Bytes <= 0 {
			return "keep all (no byte budget set)"
		}
		return fmt.Sprintf("keep the prefix whose cumulative size <= %d bytes", op.Bytes)
	case OpRender:
		return "materialize the set into context (read-only page-in through the trust gate; sealed cells refused)"
	case OpTombstone:
		r := op.Reason
		if r == "" {
			r = "requested by memq"
		}
		return "tombstone each cell (negative-only; bytes/row preserved): " + r
	case OpConsolidate:
		return "fold the set into one derived extractive disposition (no model; faithful by construction)"
	case OpReclassify:
		return "reclassify durability to " + op.By + " (never promotes to durable)"
	case OpPrune:
		return "reclaim unreferenced storage (GC; no model-visible effect)"
	}
	return op.Kind
}

// predString renders a predicate as a compact infix string for EXPLAIN.
func predString(p Pred) string {
	switch p.Op {
	case "", PredTrue:
		return "true"
	case PredAnd, PredOr:
		parts := make([]string, 0, len(p.Args))
		for _, a := range p.Args {
			parts = append(parts, predString(a))
		}
		sep := " AND "
		if p.Op == PredOr {
			sep = " OR "
		}
		return "(" + strings.Join(parts, sep) + ")"
	case PredNot:
		if len(p.Args) == 1 {
			return "NOT " + predString(p.Args[0])
		}
		return "NOT(?)"
	case PredMatch:
		return fmt.Sprintf("match(%q)", p.Value)
	}
	sym := map[string]string{PredEq: "==", PredNe: "!=", PredLt: "<", PredLe: "<=", PredGt: ">", PredGe: ">="}[p.Op]
	if sym == "" {
		sym = p.Op
	}
	return fmt.Sprintf("%s %s %q", p.Field, sym, p.Value)
}

// Text renders a Plan as an operator-readable block.
func (p Plan) Text() string {
	var b strings.Builder
	if p.Intent != "" {
		fmt.Fprintf(&b, "intent: %q\n", p.Intent)
	}
	if !p.Valid {
		fmt.Fprintf(&b, "INVALID: %s\n", p.Error)
	}
	for _, s := range p.Steps {
		tag := "     "
		if s.Mutation {
			tag = "MUT  "
		} else if s.Effect {
			tag = "eff  "
		}
		fmt.Fprintf(&b, "  [%d] %s %-12s %s\n", s.Index, tag, s.Kind, s.Detail)
	}
	if len(p.Mutations) == 0 {
		fmt.Fprintln(&b, "read-only: this query proposes no durable mutation")
	} else {
		fmt.Fprintf(&b, "mutations (proposal-only without caps): %s\n", strings.Join(p.Mutations, ", "))
	}
	return b.String()
}
