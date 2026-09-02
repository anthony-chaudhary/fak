package sessionctl

// constraint.go — the typed OUT-OF-BAND add-constraint op of the operator control
// epic (#2756, epic #2753), the floor-tightening twin of the redirect op
// (redirect.go, #2755): where redirect changes WHAT the session pursues,
// add-constraint narrows what the session MAY DO — it tightens the running
// session's adjudication floor at the next turn boundary without stopping it.
//
// The security property this op fixes is MONOTONE TIGHTENING: an out-of-band op
// may forbid a named tool, add a deny rule, or narrow the file-tree lane, and it
// may NEVER widen the floor back out (widening stays start-time / operator-
// privileged — un-constraining a session is a restart, by the epic's design). A
// widen attempt is refused with its closed reason, at the enqueue edge when the
// live floor already proves it illegal, and again at apply time for the race
// where a narrower lane landed between enqueue and apply.
//
// Like redirect.go this package owns ONLY the op payload, its validation, the
// per-session next-boundary mailbox + live constraint floor, and the deny
// predicate the loop's tool dispatch consults (ConstraintDenies). The loop-side
// consumer is internal/agent/loop_constraint.go: it drains the mailbox at the
// turn boundary (never mid-turn — an in-flight turn is not re-floored beneath
// itself) and denies a floor-forbidden tool call before dispatch with a typed
// receipt. The per-op control ENVELOPE binds at the vocabulary spine (#2754):
// the OpSpec row for this op (its CONSTRAINT_* tokens, boundary, witness shape)
// registers together with its loop-side #2766 witness row, per the reservation
// in vocab.go — registering either half alone would let the two authoritative
// tables drift.

import (
	"fmt"
	"path"
	"slices"
	"strings"
	"sync"
)

// ConstraintKind is the closed set of ways an add-constraint op may tighten the
// floor. Every kind narrows; there is deliberately no widening kind.
type ConstraintKind string

const (
	// ConstraintForbidTool forbids a named tool for the rest of the session.
	ConstraintForbidTool ConstraintKind = "forbid-tool"
	// ConstraintAddDeny adds a deny rule: a tool name, or a file-tree region
	// (slash path) writes/reads must not touch.
	ConstraintAddDeny ConstraintKind = "add-deny"
	// ConstraintNarrowLane narrows the session's file-tree lane to a region
	// within the current lane (or sets the first lane on an unbounded floor).
	ConstraintNarrowLane ConstraintKind = "narrow-lane"
)

// Constraint is the typed out-of-band add-constraint op payload. JSON tags are
// the wire shape the (spine-bound, #2754) route will carry. Exactly the field
// matching Kind is read; the others are ignored.
type Constraint struct {
	// Kind selects how the floor tightens — the closed ConstraintKind set.
	Kind ConstraintKind `json:"kind"`
	// Tool is the tool name to forbid (kind=forbid-tool).
	Tool string `json:"tool,omitempty"`
	// Rule is the deny rule to add (kind=add-deny): a tool name or a file-tree
	// region as a slash path.
	Rule string `json:"rule,omitempty"`
	// Lane is the file-tree region to narrow the lane to (kind=narrow-lane).
	Lane string `json:"lane,omitempty"`
	// Reason is the operator's stated reason, carried for the audit trail only.
	Reason string `json:"reason,omitempty"`
}

// ConstraintRefuseReason is the closed refusal vocabulary for the add-constraint
// op — the CONSTRAINT_* tokens the #2766 table reserves for this op. Two
// categories share the type: op refusals (a malformed or widening op is refused
// at the edge/apply) and floor denials (a tool call the tightened floor denies).
type ConstraintRefuseReason string

const (
	// ConstraintMalformed refuses an op whose shape cannot tighten anything —
	// unknown kind, empty payload for its kind, or a region escaping the tree.
	ConstraintMalformed ConstraintRefuseReason = "CONSTRAINT_MALFORMED"
	// ConstraintWidens refuses an op that would WIDEN the floor — today, a
	// narrow-lane naming a region not within the current lane. Tightening is
	// monotone; widening out of band is never legal.
	ConstraintWidens ConstraintRefuseReason = "CONSTRAINT_WIDENS_FLOOR"
	// ConstraintToolForbidden denies a call to a tool the floor forbids.
	ConstraintToolForbidden ConstraintRefuseReason = "CONSTRAINT_TOOL_FORBIDDEN"
	// ConstraintDenyRule denies a call matching an added deny rule.
	ConstraintDenyRule ConstraintRefuseReason = "CONSTRAINT_DENY_RULE"
	// ConstraintOutsideLane denies a call targeting a path outside the
	// narrowed file-tree lane.
	ConstraintOutsideLane ConstraintRefuseReason = "CONSTRAINT_OUTSIDE_LANE"
)

// ConstraintRefusal is a structured, closed-reason refusal — of one op, or of
// one tool call the tightened floor denies. It implements error so plumbing can
// thread it, but callers switch on Reason, never parse Detail.
type ConstraintRefusal struct {
	Reason ConstraintRefuseReason `json:"reason"`
	Detail string                 `json:"detail,omitempty"`
}

func (r *ConstraintRefusal) Error() string { return refusalString(string(r.Reason), r.Detail) }

// constraintRefuse builds a ConstraintRefusal in one line.
func constraintRefuse(reason ConstraintRefuseReason, format string, args ...any) *ConstraintRefusal {
	return &ConstraintRefusal{Reason: reason, Detail: fmt.Sprintf(format, args...)}
}

// normalizeTreePath canonicalizes a tree region / target to a clean, slash-
// separated, relative path ("" when empty), so prefix containment is exact.
func normalizeTreePath(p string) string {
	p = strings.TrimSpace(strings.ReplaceAll(p, "\\", "/"))
	if p == "" {
		return ""
	}
	p = strings.TrimSuffix(path.Clean(p), "/")
	return strings.TrimPrefix(p, "./")
}

// badTreeRegion reports whether a normalized region cannot name a tree region a
// constraint may bind: empty, the whole tree ("."), or escaping the tree ("..").
func badTreeRegion(p string) bool {
	return p == "" || p == "." || p == ".." || strings.HasPrefix(p, "../")
}

// withinTree reports whether p is region itself or inside it (path containment,
// exact at slash boundaries — "internal/auth2" is NOT within "internal/auth").
func withinTree(p, region string) bool {
	return p == region || strings.HasPrefix(p, region+"/")
}

// Validate checks the op's SHAPE — parse, don't validate later: the producer
// edge refuses a malformed op synchronously with its closed reason. Floor
// legality (the widen check) is decided against the live floor by
// EnqueueConstraint / ApplyPendingConstraints, not here.
func (c Constraint) Validate() *ConstraintRefusal {
	switch c.Kind {
	case ConstraintForbidTool:
		if strings.TrimSpace(c.Tool) == "" {
			return constraintRefuse(ConstraintMalformed, "forbid-tool needs a tool name")
		}
	case ConstraintAddDeny:
		if strings.TrimSpace(c.Rule) == "" || badTreeRegion(normalizeTreePath(c.Rule)) {
			return constraintRefuse(ConstraintMalformed, "add-deny needs a tool name or an in-tree region, got %q", c.Rule)
		}
	case ConstraintNarrowLane:
		if badTreeRegion(normalizeTreePath(c.Lane)) {
			return constraintRefuse(ConstraintMalformed, "narrow-lane needs an in-tree region, got %q", c.Lane)
		}
	default:
		return constraintRefuse(ConstraintMalformed, "unknown constraint kind %q (closed set: %s|%s|%s)",
			c.Kind, ConstraintForbidTool, ConstraintAddDeny, ConstraintNarrowLane)
	}
	return nil
}

// describe is the one-line audit description of a constraint — the Next-witness
// payload for the applied/refused record.
func (c Constraint) describe() string {
	switch c.Kind {
	case ConstraintForbidTool:
		return string(ConstraintForbidTool) + ": " + strings.TrimSpace(c.Tool)
	case ConstraintAddDeny:
		return string(ConstraintAddDeny) + ": " + normalizeTreePath(c.Rule)
	case ConstraintNarrowLane:
		return string(ConstraintNarrowLane) + ": " + normalizeTreePath(c.Lane)
	}
	return string(c.Kind)
}

// ConstraintFloor is the live, monotonically-tightening per-session floor the
// accepted constraints fold into. The zero value is the unconstrained floor.
type ConstraintFloor struct {
	// ForbiddenTools are tool names the session may no longer call.
	ForbiddenTools []string `json:"forbidden_tools,omitempty"`
	// DenyRules are added deny rules: each denies a matching tool name, or any
	// call targeting a path inside its region.
	DenyRules []string `json:"deny_rules,omitempty"`
	// Lane, when non-empty, is the narrowed file-tree lane: a path-bearing call
	// targeting anything outside it is denied. It only ever narrows.
	Lane string `json:"lane,omitempty"`
}

// Constrained reports whether the floor tightens anything.
func (f ConstraintFloor) Constrained() bool {
	return len(f.ForbiddenTools) > 0 || len(f.DenyRules) > 0 || f.Lane != ""
}

// widenRefusal is the monotonicity gate: the closed refusal when folding c into
// the floor would WIDEN it, nil when c only narrows. forbid-tool and add-deny
// only ever add; the one widening shape today is a narrow-lane naming a region
// outside the current lane.
func (f ConstraintFloor) widenRefusal(c Constraint) *ConstraintRefusal {
	if c.Kind != ConstraintNarrowLane {
		return nil
	}
	lane := normalizeTreePath(c.Lane)
	if f.Lane != "" && !withinTree(lane, f.Lane) {
		return constraintRefuse(ConstraintWidens,
			"lane %q is not within the current lane %q; an out-of-band constraint may only narrow the floor, never widen it (un-constraining is a session restart)", lane, f.Lane)
	}
	return nil
}

// tighten folds one validated constraint into the floor, refusing (unchanged
// floor) if it would widen. Duplicate tightenings are idempotent.
func (f *ConstraintFloor) tighten(c Constraint) *ConstraintRefusal {
	if ref := f.widenRefusal(c); ref != nil {
		return ref
	}
	switch c.Kind {
	case ConstraintForbidTool:
		if tool := strings.TrimSpace(c.Tool); !slices.Contains(f.ForbiddenTools, tool) {
			f.ForbiddenTools = append(f.ForbiddenTools, tool)
		}
	case ConstraintAddDeny:
		if rule := normalizeTreePath(c.Rule); !slices.Contains(f.DenyRules, rule) {
			f.DenyRules = append(f.DenyRules, rule)
		}
	case ConstraintNarrowLane:
		f.Lane = normalizeTreePath(c.Lane)
	}
	return nil
}

// Denies checks one candidate tool call against the floor: nil allows, non-nil
// denies with the closed CONSTRAINT_* reason. tool is the tool name; target is
// the call's file-tree path when the caller knows it ("" for a call with no
// path — the lane and path-shaped deny rules bind only path-bearing calls).
func (f ConstraintFloor) Denies(tool, target string) *ConstraintRefusal {
	tool = strings.TrimSpace(tool)
	tgt := normalizeTreePath(target)
	if slices.Contains(f.ForbiddenTools, tool) {
		return constraintRefuse(ConstraintToolForbidden, "tool %q was forbidden mid-session by an operator constraint", tool)
	}
	for _, rule := range f.DenyRules {
		if rule == tool {
			return constraintRefuse(ConstraintDenyRule, "tool %q matches deny rule %q", tool, rule)
		}
		if tgt != "" && withinTree(tgt, rule) {
			return constraintRefuse(ConstraintDenyRule, "target %q is inside denied region %q", tgt, rule)
		}
	}
	if f.Lane != "" && tgt != "" && !withinTree(tgt, f.Lane) {
		return constraintRefuse(ConstraintOutsideLane, "target %q is outside the narrowed lane %q", tgt, f.Lane)
	}
	return nil
}

// Directive renders the model-facing standing notice of the tightened floor the
// loop carries into a turn as a SYSTEM directive (like Objective.Directive).
// The unconstrained floor renders "" so the historical loop is unchanged.
func (f ConstraintFloor) Directive() string {
	if !f.Constrained() {
		return ""
	}
	var b strings.Builder
	b.WriteString("CONSTRAINTS (tightened out of band): the operator narrowed this session's floor.")
	if len(f.ForbiddenTools) > 0 {
		b.WriteString("\nForbidden tools: " + strings.Join(f.ForbiddenTools, ", "))
	}
	if len(f.DenyRules) > 0 {
		b.WriteString("\nDeny rules: " + strings.Join(f.DenyRules, ", "))
	}
	if f.Lane != "" {
		b.WriteString("\nFile-tree lane: " + f.Lane)
	}
	b.WriteString("\nThese bind from this turn on and only ever narrow; they cannot be widened mid-session.")
	return b.String()
}

// clone deep-copies the floor so no caller can mutate the registry through it.
func (f ConstraintFloor) clone() ConstraintFloor {
	f.ForbiddenTools = slices.Clone(f.ForbiddenTools)
	f.DenyRules = slices.Clone(f.DenyRules)
	return f
}

// constraint state is per-session, keyed by the run's trace id — the same keying
// as redirect.go's mailbox and the a2achan steer bus. pending holds constraints
// accepted out of band until the loop drains them at its next turn boundary;
// floor holds the live tightened floor the loop's tool dispatch consults.
var (
	constraintMu      sync.Mutex
	constraintPending = map[string][]Constraint{}
	constraintNext    = map[string][]NextRecord{}
	constraintFloor   = map[string]ConstraintFloor{}
)

// EnqueueConstraint accepts one add-constraint op for the session keyed by
// trace, to be applied at its next turn boundary. The op is validated HERE, at
// the edge: a malformed op is refused synchronously, and — because the live
// floor is queryable now — an op the current floor already proves WIDENING is
// refused at the edge too, so an illegal op never enters the mailbox.
func EnqueueConstraint(trace string, c Constraint) *ConstraintRefusal {
	trace = strings.TrimSpace(trace)
	if trace == "" {
		return constraintRefuse(ConstraintMalformed, "empty session trace")
	}
	if ref := c.Validate(); ref != nil {
		return ref
	}
	constraintMu.Lock()
	defer constraintMu.Unlock()
	if ref := constraintFloor[trace].widenRefusal(c); ref != nil {
		return ref
	}
	constraintPending[trace] = append(constraintPending[trace], c)
	return nil
}

// ConstraintPendingLen reports how many constraints are queued for trace
// (observability + tests).
func ConstraintPendingLen(trace string) int {
	trace = strings.TrimSpace(trace)
	if trace == "" {
		return 0
	}
	constraintMu.Lock()
	defer constraintMu.Unlock()
	return len(constraintPending[trace])
}

// ApplyPendingConstraints drains every constraint queued for trace and folds
// each into the live floor, in arrival order — the consumer half the session
// loop calls at its turn boundary. A constraint that would widen the LIVE floor
// (a race: a narrower lane landed between enqueue and apply) is refused, not
// silently applied; the mailbox is drained either way (a refused op must not
// retry forever). It returns the constraints it applied and any refusals.
// Idempotent on an empty mailbox.
func ApplyPendingConstraints(trace string) (applied []Constraint, refused []ConstraintRefusal) {
	trace = strings.TrimSpace(trace)
	if trace == "" {
		return nil, nil
	}
	constraintMu.Lock()
	defer constraintMu.Unlock()
	queued := constraintPending[trace]
	delete(constraintPending, trace)
	appendNext := func(c Constraint, result ApplyResult) {
		move := Move{
			Kind: MoveAnnotate, Render: RenderSystemDirective,
			Session: SessionAutonomous, Gate: "sessionctl-constraint",
			Source: "agent-turn-boundary", Payload: c.describe(),
		}
		// Constants above satisfy the shared closed contract; a future vocabulary
		// change that violates it is pinned by the constraint Next tests.
		if record, err := WitnessMove(move, result); err == nil {
			constraintNext[trace] = append(constraintNext[trace], record)
		}
	}
	for _, c := range queued {
		floor := constraintFloor[trace]
		if ref := floor.tighten(c); ref != nil {
			refused = append(refused, *ref)
			appendNext(c, ApplyResult{Refusal: ref.Error()})
			continue
		}
		constraintFloor[trace] = floor
		applied = append(applied, c)
		appendNext(c, ApplyResult{Applied: true})
	}
	return applied, refused
}

// CurrentConstraintFloor returns a deep copy of the live floor for trace and
// whether any constraint has tightened it. The loop reads it at the turn
// boundary to carry the standing floor directive.
func CurrentConstraintFloor(trace string) (ConstraintFloor, bool) {
	trace = strings.TrimSpace(trace)
	if trace == "" {
		return ConstraintFloor{}, false
	}
	constraintMu.Lock()
	defer constraintMu.Unlock()
	f, ok := constraintFloor[trace]
	return f.clone(), ok && f.Constrained()
}

// ConstraintDenies checks one candidate tool call against the LIVE floor for
// trace — the fold-in point the loop's tool dispatch consults before dispatching
// each call. nil allows (including an unconstrained or untraced session);
// non-nil denies with the closed CONSTRAINT_* reason.
func ConstraintDenies(trace, tool, target string) *ConstraintRefusal {
	trace = strings.TrimSpace(trace)
	if trace == "" {
		return nil
	}
	constraintMu.Lock()
	floor := constraintFloor[trace]
	constraintMu.Unlock()
	return floor.Denies(tool, target)
}

// ClearConstraints drops the live floor and any pending constraints for trace —
// the session teardown hook so a finished run leaks no per-trace state.
// Idempotent. (Dropping at teardown is not an out-of-band widen: the session is
// over; a NEW session starts from its own start-time floor.)
func ClearConstraints(trace string) {
	trace = strings.TrimSpace(trace)
	if trace == "" {
		return
	}
	constraintMu.Lock()
	defer constraintMu.Unlock()
	delete(constraintFloor, trace)
	delete(constraintPending, trace)
	delete(constraintNext, trace)
}

// ReadConstraintNextRecords returns and clears the independently re-readable
// Next witnesses emitted while the trace's constraint mailbox was applied.
// Empty/no-op drains produce no records.
func ReadConstraintNextRecords(trace string) []NextRecord {
	trace = strings.TrimSpace(trace)
	if trace == "" {
		return nil
	}
	constraintMu.Lock()
	defer constraintMu.Unlock()
	records := append([]NextRecord(nil), constraintNext[trace]...)
	delete(constraintNext, trace)
	return records
}
