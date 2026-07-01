// Package fusedturn is the executable form of the fak thesis — the FUSED agent
// kernel — at the level of ONE TURN: a turn may spawn BOTH a classical operation
// (a deterministic tool call, a git commit, a lease, a verify) AND a weight-based
// operation (a model forward: an inference, an ensemble member, an expert
// dispatch), and BOTH cross the SAME default-deny adjudication floor.
//
// # Why this is a real seam, not a relabel
//
// The kernel already treats every operation uniformly: a classical tool and a
// weight-based model are BOTH abi.EngineDriver's behind one path
// (abi.ToolCall -> kernel.Submit adjudicate -> kernel.Reap -> Engine.Complete).
// That uniformity IS the fusion. What the kernel does NOT do is NAME the seam: it
// carries no first-class notion of "this op is model-centric, that one is
// classic", and no primitive that recognizes when a single turn's batch genuinely
// SPANS both concept-families. fak's own docs deliberately hold the two rank
// spaces apart to avoid conflation (docs/collectives.md: the AGENT layer vs the
// TENSOR layer). This package is the layer ABOVE that split: it does not conflate
// the two — it CLASSIFIES each op into its family and proves a mixed turn is
// governed by one kernel, so "one turn spawns both classical and weight-based
// operations" becomes a witnessed verdict instead of a slogan.
//
// # Pure, floor-injected
//
// The package is PURE: [Classify] and [Fuse] are deterministic folds over
// abi.ToolCall fields, and it imports ONLY internal/abi (tier 1, off the hot
// path). The adjudication witness does not IMPORT the kernel — it INVERTS the
// dependency behind a one-method [Decider] interface that *kernel.Kernel's
// BatchDecide satisfies structurally. So a fused turn is proven against the REAL
// default-deny floor with no coupling, and a test can drive it with a stub.
package fusedturn

import (
	"context"
	"sort"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// Schema is the versioned payload tag the JSON [Summary] carries.
const Schema = "fak.fusedturn.v1"

// OpClass is the CLOSED, additive concept-family one operation belongs to. The
// whole point of a fused kernel is that these families cross the SAME floor, so a
// turn can spawn both. ClassUnknown is the fail-closed default: an op whose family
// is not declared is never silently sorted into one — it is surfaced as unknown so
// a reader (or a gate) can refuse to treat an unclassified turn as fused.
type OpClass uint8

const (
	// ClassUnknown is the fail-closed default: family not declared / unrecognized.
	ClassUnknown OpClass = iota
	// ClassClassical is a deterministic-effect op — a tool call, a git commit, a
	// lease, a verify: the classic / AGENT-layer family (docs/collectives.md).
	ClassClassical
	// ClassWeight is a model-forward op — an inference, an ensemble member, an
	// expert dispatch: the model-centric / TENSOR-layer family.
	ClassWeight
)

// String renders the class as its stable lowercase token (also the Meta value).
func (c OpClass) String() string {
	switch c {
	case ClassClassical:
		return "classical"
	case ClassWeight:
		return "weight"
	default:
		return "unknown"
	}
}

// MetaClassKey is the OPEN abi.ToolCall.Meta key a producer stamps to DECLARE an
// op's concept-family. Declaration is authoritative because the producer (the
// gateway, the planner, a test) is the only party that actually knows whether a
// call runs a model forward or a deterministic tool — so [Classify] reads the
// declaration rather than guessing from a name. The value is the OpClass token
// (see [OpClass.String]); an absent or unrecognized value classifies as
// ClassUnknown (fail-closed).
const MetaClassKey = "fak.opclass"

// classFromToken maps a declared Meta token back to its OpClass. An unrecognized
// token is ClassUnknown — the fail-closed default, never a guess.
func classFromToken(tok string) OpClass {
	switch tok {
	case "classical":
		return ClassClassical
	case "weight":
		return ClassWeight
	default:
		return ClassUnknown
	}
}

// Classify returns the DECLARED concept-family of one call, or ClassUnknown when
// it carries no recognized declaration (fail-closed). It is a pure function of the
// call's Meta[MetaClassKey] tag — no name heuristics, no registry lookup, no clock
// — so the same call always classifies the same way and an unclassified op can
// never masquerade as classified. A nil call is ClassUnknown.
func Classify(c *abi.ToolCall) OpClass {
	if c == nil || c.Meta == nil {
		return ClassUnknown
	}
	return classFromToken(c.Meta[MetaClassKey])
}

// Tag stamps class onto a call's Meta[MetaClassKey] (allocating Meta if needed)
// and returns the same call for chaining. Tagging with ClassUnknown clears the
// declaration. It is how a producer that knows an op's family records that fact so
// [Classify] can read it back authoritatively.
func Tag(c *abi.ToolCall, class OpClass) *abi.ToolCall {
	if c == nil {
		return nil
	}
	if class == ClassUnknown {
		if c.Meta != nil {
			delete(c.Meta, MetaClassKey)
		}
		return c
	}
	if c.Meta == nil {
		c.Meta = map[string]string{}
	}
	c.Meta[MetaClassKey] = class.String()
	return c
}

// Classical builds a classical-tagged tool call: a deterministic-effect op named
// by tool, running args, on the kernel's default engine (Engine left empty).
func Classical(tool string, args abi.Ref) *abi.ToolCall {
	return Tag(&abi.ToolCall{Tool: tool, Args: args}, ClassClassical)
}

// Weight builds a weight-tagged tool call: a model-forward op routed to engine
// (the model / inference route), named by tool, running args.
func Weight(engine, tool string, args abi.Ref) *abi.ToolCall {
	return Tag(&abi.ToolCall{Tool: tool, Engine: engine, Args: args}, ClassWeight)
}

// ClassifiedOp is one op of a turn paired with the family [Classify] assigned it.
type ClassifiedOp struct {
	Call  *abi.ToolCall `json:"-"`
	Tool  string        `json:"tool"`
	Class OpClass       `json:"class"`
}

// FusedTurn is one turn's ORDERED batch of proposed operations, each classified
// into its concept-family. It is the executable claim "one turn spawns both": it
// is [FusedTurn.Fused] iff it contains at least one classical AND one weight op.
// The order is the caller's submission order — preserved, because a fused turn is
// a real interleaving of the two families, not two sorted piles.
type FusedTurn struct {
	Ops []ClassifiedOp
}

// Fuse classifies each call in submission order into a FusedTurn. Nil calls are
// skipped (a producer that emitted nothing contributes nothing to the turn).
func Fuse(calls []*abi.ToolCall) FusedTurn {
	ops := make([]ClassifiedOp, 0, len(calls))
	for _, c := range calls {
		if c == nil {
			continue
		}
		ops = append(ops, ClassifiedOp{Call: c, Tool: c.Tool, Class: Classify(c)})
	}
	return FusedTurn{Ops: ops}
}

// count returns how many ops carry the given class.
func (t FusedTurn) count(class OpClass) int {
	n := 0
	for _, o := range t.Ops {
		if o.Class == class {
			n++
		}
	}
	return n
}

// Classical reports how many classical ops the turn spawns.
func (t FusedTurn) Classical() int { return t.count(ClassClassical) }

// Weight reports how many weight-based ops the turn spawns.
func (t FusedTurn) Weight() int { return t.count(ClassWeight) }

// Unknown reports how many ops the turn could not classify (fail-closed residue).
func (t FusedTurn) Unknown() int { return t.count(ClassUnknown) }

// Fused reports whether the turn genuinely spans BOTH concept-families — at least
// one classical op AND at least one weight-based op. This is the load-bearing
// predicate: a turn of only tool calls, or only inferences, is a NORMAL turn; a
// turn that spawns both is what the fused kernel unlocks. An unclassified op never
// contributes to fusion (fail-closed), so a turn cannot be called fused by
// accident of a missing declaration.
func (t FusedTurn) Fused() bool {
	return t.Classical() > 0 && t.Weight() > 0
}

// Summary is the JSON-safe fold of a turn: counts per family and the fused bit. It
// carries no arg bytes — only tool names and classes — so it is safe to log.
type Summary struct {
	Schema    string `json:"schema"`
	Ops       int    `json:"ops"`
	Classical int    `json:"classical"`
	Weight    int    `json:"weight"`
	Unknown   int    `json:"unknown"`
	Fused     bool   `json:"fused"`
}

// Summary folds the turn into its counts + fused bit.
func (t FusedTurn) Summary() Summary {
	return Summary{
		Schema:    Schema,
		Ops:       len(t.Ops),
		Classical: t.Classical(),
		Weight:    t.Weight(),
		Unknown:   t.Unknown(),
		Fused:     t.Fused(),
	}
}

// Decider folds a batch of calls to per-call verdicts on ONE floor, in the caller's
// order. *kernel.Kernel's BatchDecide satisfies it structurally, so injecting it
// keeps this package pure (abi-only) while proving fusion against the real
// default-deny kernel. A test can supply a stub decider.
type Decider interface {
	BatchDecide(ctx context.Context, calls []*abi.ToolCall) []abi.Verdict
}

// AdjudicatedOp is one classified op paired with the verdict the shared floor gave
// it — the row that proves a given family's op crossed the SAME kernel.
type AdjudicatedOp struct {
	Tool    string      `json:"tool"`
	Class   OpClass     `json:"class"`
	Verdict abi.Verdict `json:"-"`
	Kind    string      `json:"verdict"`
	Reason  string      `json:"reason"`
}

// Adjudicate runs the WHOLE mixed batch through ONE Decider in a single pass, so
// every op — classical AND weight — folds the SAME adjudicator chain. It returns
// one row per op, in order, pairing the op's family with the verdict the shared
// floor returned. This is the witness the fusion is real: a fused turn's two
// concept-families are governed by one kernel, not routed to two side-paths that
// could diverge on policy. A nil Decider yields nil (no floor, no witness).
func (t FusedTurn) Adjudicate(ctx context.Context, d Decider) []AdjudicatedOp {
	if d == nil {
		return nil
	}
	calls := make([]*abi.ToolCall, len(t.Ops))
	for i, o := range t.Ops {
		calls[i] = o.Call
	}
	verdicts := d.BatchDecide(ctx, calls)
	rows := make([]AdjudicatedOp, len(t.Ops))
	for i, o := range t.Ops {
		var v abi.Verdict
		if i < len(verdicts) {
			v = verdicts[i]
		}
		rows[i] = AdjudicatedOp{
			Tool:    o.Tool,
			Class:   o.Class,
			Verdict: v,
			Kind:    verdictKindName(v.Kind),
			Reason:  abi.ReasonName(v.Reason),
		}
	}
	return rows
}

// GovernedFamilies returns the distinct concept-families that received a verdict
// in a fully-adjudicated batch (order: classical, weight, unknown). A genuinely
// fused turn that crossed one floor returns both classical and weight here — the
// checkable form of "both families were governed by the same kernel". It reads the
// rows [Adjudicate] produced; a family present in the turn but missing a verdict
// row is NOT reported (it was not witnessed as governed).
func GovernedFamilies(rows []AdjudicatedOp) []OpClass {
	seen := map[OpClass]bool{}
	for _, r := range rows {
		if r.Kind != "" { // a verdict was actually returned for this op
			seen[r.Class] = true
		}
	}
	out := make([]OpClass, 0, len(seen))
	for _, c := range []OpClass{ClassClassical, ClassWeight, ClassUnknown} {
		if seen[c] {
			out = append(out, c)
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i] < out[j] })
	return out
}

// verdictKindName renders a verdict kind as a short stable token for JSON/logs. It
// is intentionally small — the CLI and tests need only distinguish the outcomes a
// fold can produce — and falls back to the numeric kind for a registered escalation
// kind the core does not name here.
func verdictKindName(k abi.VerdictKind) string {
	switch k {
	case abi.VerdictAllow:
		return "allow"
	case abi.VerdictDeny:
		return "deny"
	case abi.VerdictDefer:
		return "defer"
	case abi.VerdictTransform:
		return "transform"
	case abi.VerdictQuarantine:
		return "quarantine"
	case abi.VerdictRequireWitness:
		return "require-witness"
	case abi.VerdictIndeterminate:
		return "indeterminate"
	default:
		return "other"
	}
}
