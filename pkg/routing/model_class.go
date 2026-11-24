package routing

import (
	"errors"
	"fmt"
	"sort"
	"strings"
)

// ModelClassSchema identifies the model-class manifest/receipt shape.
const ModelClassSchema = "routing-model-class/1"

// ModelClass is a declared equivalence class of interchangeable model ids: members
// are peers for balancing and same-class failover. Declaration only — no signal, no
// state, no behaviour; SelectClass gives it a selector. It is the model-plane twin
// of BalanceGroup. Stepping out of a class is a declared, opt-in transition
// (ClassLadder), never an inference: a request only lands on a model an operator
// wrote down as its peer.
type ModelClass struct {
	// Name is the class's identity in the audit trail. Required, unique.
	Name string `json:"name"`
	// Models are the member model ids in declared order, which is the tie-break for
	// every strategy. Each model id belongs to exactly one class.
	Models []string `json:"models"`
	// Strategy selects how members are ordered. Empty means
	// BalanceStrategyPriority.
	Strategy BalanceStrategy `json:"strategy,omitempty"`
	// Weights is the declared per-member ratio. A member absent from this map is
	// DefaultBalanceWeight. Zero is a legal, declared DRAIN: the member keeps its
	// position and receives nothing.
	Weights map[string]int `json:"weights,omitempty"`
}

// WeightOf returns the declared weight for a member, else DefaultBalanceWeight. An
// undeclared member is a caller bug that Validate refuses, not this read.
func (c ModelClass) WeightOf(model string) int {
	if w, ok := c.Weights[model]; ok {
		return w
	}
	return DefaultBalanceWeight
}

// Validate checks the class's intrinsic invariants and fails loud on the first
// offence, naming the member or field: name, at least one member, no empty or
// duplicated member, a known strategy, weights only for declared members, no negative
// weight, and not every member drained. Cross-class invariants belong to
// ValidateModelClasses.
func (c ModelClass) Validate() error {
	if strings.TrimSpace(c.Name) == "" {
		return fmt.Errorf("routing: model class has an empty name")
	}
	if len(c.Models) == 0 {
		return fmt.Errorf("routing: model class %q has no members", c.Name)
	}
	seen := make(map[string]int, len(c.Models))
	for i, m := range c.Models {
		if m == "" {
			return fmt.Errorf("routing: model class %q member %d is empty", c.Name, i)
		}
		if first, dup := seen[m]; dup {
			return fmt.Errorf("routing: model class %q member %q is declared twice (indexes %d and %d)", c.Name, m, first, i)
		}
		seen[m] = i
	}
	strategy := c.Strategy.orDefault()
	if !strategy.IsKnown() {
		return fmt.Errorf("routing: model class %q has unknown strategy %q (known: %s)", c.Name, c.Strategy, strings.Join(BalanceStrategyNames(), ", "))
	}
	for member, w := range c.Weights {
		if _, declared := seen[member]; !declared {
			return fmt.Errorf("routing: model class %q declares weight %d for undeclared member %q", c.Name, w, member)
		}
		if w < 0 {
			return fmt.Errorf("routing: model class %q member %q has negative weight %d", c.Name, member, w)
		}
	}
	// An all-drained class can never select: a config error, refused here rather than
	// surfacing later as a declared step-down that could never have served.
	allDrained := true
	for _, m := range c.Models {
		if c.WeightOf(m) > 0 {
			allDrained = false
			break
		}
	}
	if allDrained {
		return fmt.Errorf("routing: model class %q has every member at weight 0, so it can never select", c.Name)
	}
	return nil
}

// ClassLadder is one route's declared policy: its preferred class plus the ordered
// fallback classes. Pure data; the walk never mutates it.
// The ladder is the only place a request may cross a class boundary: same-class
// balancing is proactive, stepping down is reactive and bounded. A class not named
// here is unreachable, and a ladder with no Fallbacks degrades nowhere.
type ClassLadder struct {
	// Preferred is the class the route's own model belongs to. Required.
	Preferred string `json:"preferred"`
	// Fallbacks are the ordered step-down classes. May be empty: no cross-class
	// step-down is permitted.
	Fallbacks []string `json:"fallbacks,omitempty"`
}

// MaxClassLadderDepth bounds the declared fallback chain, in hops
// (Preferred->F[0]->F[1]). Two hops cover one quality tier and one floor while
// keeping the receipt legible.
const MaxClassLadderDepth = 2

// The sentinels below are the closed vocabulary of cross-declaration failure, so a
// caller can tell "fix the reference" from "fix the cycle" with errors.Is.
var (
	// ErrModelClassUnknownRef: a ladder names a class the manifest does not declare,
	// or its own model is not a member of the class it prefers.
	ErrModelClassUnknownRef = errors.New("routing: model class reference names no class")
	// ErrModelClassCycle: following fallback refs returns to a class already visited,
	// including a ladder naming its own preferred class. A looping step-down degrades
	// forever instead of flooring.
	ErrModelClassCycle = errors.New("routing: model class fallback cycle")
	// ErrModelClassTooDeep: a ladder declares more than MaxClassLadderDepth hops.
	ErrModelClassTooDeep = errors.New("routing: model class fallback chain too deep")
	// ErrModelClassDuplicateMember: one model id in more than one class, or one class
	// name declared twice.
	ErrModelClassDuplicateMember = errors.New("routing: model id declared in more than one class")
	// ErrModelClassUnreferenced: a declared class no ladder reaches — a typo wearing a
	// declaration's clothes, so it fails loud instead of quietly never serving.
	ErrModelClassUnreferenced = errors.New("routing: model class is not referenced by any ladder")
)

// ValidateModelClasses checks a set of classes against the ladders that use them. It
// is the fail-loud entry point a manifest constructor calls, answering every question
// from the declaration alone: refs resolve, no cycle across all ladders, depth within
// MaxClassLadderDepth, a model id in at most one class, every class referenced, and
// each ladder's preferred class contains the ladder's own model. Ladders are checked
// in sorted model-id order so the first offence named is stable; every error wraps an
// ErrModelClass* sentinel.
func ValidateModelClasses(classes []ModelClass, ladders map[string]ClassLadder) error {
	byName := make(map[string]int, len(classes))
	classNames := make([]string, 0, len(classes))
	for i, c := range classes {
		if first, dup := byName[c.Name]; dup {
			return fmt.Errorf("%w: class %q is declared twice (indexes %d and %d)", ErrModelClassDuplicateMember, c.Name, first, i)
		}
		byName[c.Name] = i
		classNames = append(classNames, c.Name)
	}
	owner := make(map[string]string, totalMembers(classes))
	for _, c := range classes {
		for _, m := range c.Models {
			if prev, dup := owner[m]; dup && prev != c.Name {
				return fmt.Errorf("%w: model %q is declared in both class %q and class %q", ErrModelClassDuplicateMember, m, prev, c.Name)
			}
			owner[m] = c.Name
		}
	}

	models := make([]string, 0, len(ladders))
	for model := range ladders {
		models = append(models, model)
	}
	sort.Strings(models)

	succ := make(map[string][]string) // union of every ladder's fallback edges
	referenced := make(map[string]bool, len(byName))
	for _, model := range models {
		lad := ladders[model]
		prefIdx, ok := byName[lad.Preferred]
		if !ok {
			return fmt.Errorf("%w: ladder for model %q prefers class %q, which is not declared", ErrModelClassUnknownRef, model, lad.Preferred)
		}
		if !modelInClass(classes[prefIdx].Models, model) {
			return fmt.Errorf("%w: ladder for model %q prefers class %q, which does not contain that model", ErrModelClassUnknownRef, model, lad.Preferred)
		}
		if len(lad.Fallbacks) > MaxClassLadderDepth {
			return fmt.Errorf("%w: ladder for model %q declares %d fallback hops; the bound is %d", ErrModelClassTooDeep, model, len(lad.Fallbacks), MaxClassLadderDepth)
		}
		referenced[lad.Preferred] = true
		prev := lad.Preferred
		for i, fb := range lad.Fallbacks {
			if _, ok := byName[fb]; !ok {
				return fmt.Errorf("%w: ladder for model %q fallback %d names class %q, which is not declared", ErrModelClassUnknownRef, model, i, fb)
			}
			if fb == lad.Preferred {
				return fmt.Errorf("%w: ladder for model %q names its own preferred class %q in Fallbacks", ErrModelClassCycle, model, fb)
			}
			referenced[fb] = true
			succ[prev] = append(succ[prev], fb)
			prev = fb
		}
	}

	sort.Strings(classNames)
	for _, name := range classNames {
		if reaches(name, name, succ) {
			return fmt.Errorf("%w: fallback refs loop back to class %q", ErrModelClassCycle, name)
		}
	}
	for _, name := range classNames {
		if !referenced[name] {
			return fmt.Errorf("%w: class %q is not referenced by any ladder", ErrModelClassUnreferenced, name)
		}
	}
	return nil
}

// totalMembers sums the declared member counts, for capacity hints only.
func totalMembers(classes []ModelClass) int {
	n := 0
	for _, c := range classes {
		n += len(c.Models)
	}
	return n
}

// modelInClass reports whether model is one of the class's declared members.
func modelInClass(models []string, model string) bool {
	for _, m := range models {
		if m == model {
			return true
		}
	}
	return false
}

// reaches reports whether start is reachable from itself by following one or more
// fallback edges in succ — i.e. whether the fallback graph loops. It follows at least
// one edge before comparing, so a class with no outgoing edge cannot reach itself.
func reaches(start, cur string, succ map[string][]string) bool {
	for _, next := range succ[cur] {
		if next == start {
			return true
		}
		if reaches(start, next, succ) {
			return true
		}
	}
	return false
}

// ClassSkipReason is the closed vocabulary for why a candidate was not chosen.
// Every skip carries one code, recorded from the first failing check in the order
// Eligible, FitsContext, Capable (a declared drain overrides all), so a replay
// reproduces the receipt byte for byte.
type ClassSkipReason string

const (
	// ClassSkipNotEligible: the candidate has no eligible capacity right now. This
	// also covers a candidate ABSENT from the input map: absence is a verdict.
	ClassSkipNotEligible ClassSkipReason = "not_eligible"
	// ClassSkipDrained: the member is declared at weight 0 — a config decision,
	// reported ahead of every runtime verdict.
	ClassSkipDrained ClassSkipReason = "drained"
	// ClassSkipContextTooSmall: the request's context need exceeds the candidate's
	// declared window.
	ClassSkipContextTooSmall ClassSkipReason = "context_too_small"
	// ClassSkipCapabilityMismatch: the candidate's declared capabilities do not
	// satisfy the request (e.g. cross-family without opt-in).
	ClassSkipCapabilityMismatch ClassSkipReason = "capability_mismatch"
)

// ClassSkipReasons returns the closed skip-reason vocabulary in declaration order, as
// a fresh copy so a caller cannot reorder it.
func ClassSkipReasons() []string {
	return []string{
		string(ClassSkipNotEligible),
		string(ClassSkipDrained),
		string(ClassSkipContextTooSmall),
		string(ClassSkipCapabilityMismatch),
	}
}

// ClassCandidate is one member's input line: identity plus the caller's verdicts. The
// fabric never reads health, keys, or clocks itself.
// The booleans are the control plane's admission verdicts carried as data; the
// fabric only orders admitted candidates.
type ClassCandidate struct {
	// Model is the model id this line is about. The map key is authoritative;
	// the field exists so a serialized receipt reads without a join.
	Model string `json:"model"`
	// Eligible is the health/readiness/paused admission verdict.
	Eligible bool `json:"eligible"`
	// FitsContext is the declared-window vs request-prompt-estimate verdict.
	FitsContext bool `json:"fits_context"`
	// Capable is the declared-capability fit verdict.
	Capable bool `json:"capable"`
}

// ClassSkipped records one non-chosen candidate with its bounded reason.
type ClassSkipped struct {
	Model  string          `json:"model"`
	Class  string          `json:"class"`
	Reason ClassSkipReason `json:"reason"`
}

// ClassDecision is the receipt: who serves, from which class, and why everyone else
// did not. Transition is "" (served the requested model within its class),
// "same_class" (a different member of the requested class serves), or
// "class_step_down" (a fallback class serves). Skipped records every non-chosen
// candidate with its first failing check. Audit lines carry model ids, class names,
// weights and reason codes only, never prompt, key or credential material. Order is the
// full walk order; RequestedModel is the baseline Transition compares against.
type ClassDecision struct {
	Schema         string         `json:"schema"`
	RequestedModel string         `json:"requested_model"`
	ServedModel    string         `json:"served_model,omitempty"`
	RequestedClass string         `json:"requested_class"`
	ServedClass    string         `json:"served_class,omitempty"`
	Transition     string         `json:"transition,omitempty"`
	Chosen         bool           `json:"chosen"`
	Order          []string       `json:"order,omitempty"`
	Skipped        []ClassSkipped `json:"skipped,omitempty"`
	Audit          []string       `json:"audit,omitempty"`
}

// The closed transition vocabulary. "" is not a constant: it is the absence of a
// transition, and omitempty renders that absence honestly.
const (
	classTransitionSameClass = "same_class"
	classTransitionStepDown  = "class_step_down"
)

// SelectClass walks the ladder for one request: the preferred class first, balanced
// among its candidates via the EXISTING BalancePolicy.Select (a synthetic
// group of the selectable members); if nothing is selectable there, the fallback
// classes in declared order, the first with a selectable candidate winning. It is
// pure and replayable. Members are filtered before selecting, a declared drain
// reporting ClassSkipDrained regardless of runtime verdicts. The balance key may be
// a session identifier, so it enters only the hash, never an audit line or receipt.
func SelectClass(requested string, ladder ClassLadder, classes map[string]ModelClass, cands map[string]ClassCandidate, snap func(ModelClass) BalanceSnapshot, key string) ClassDecision {
	// The walk order: the preferred class, then the declared fallbacks.
	chain := make([]string, 0, 1+len(ladder.Fallbacks))
	chain = append(chain, ladder.Preferred)
	chain = append(chain, ladder.Fallbacks...)

	type classWalk struct {
		name       string
		candidates int
		selectable []string
		order      []string
		skipped    []ClassSkipped
	}

	walks := make([]classWalk, 0, len(chain))
	totalSkips := 0
	for _, name := range chain {
		cls, ok := classes[name]
		w := classWalk{name: name}
		if ok {
			w.candidates = len(cls.Models)
			for _, m := range cls.Models {
				cd, present := cands[m]
				switch {
				case cls.WeightOf(m) <= 0:
					// A declared drain outruns every runtime verdict.
					w.skipped = append(w.skipped, ClassSkipped{Model: m, Class: name, Reason: ClassSkipDrained})
				case !present || !cd.Eligible:
					// Absence is a verdict, not an error: no candidate line means
					// no eligible capacity.
					w.skipped = append(w.skipped, ClassSkipped{Model: m, Class: name, Reason: ClassSkipNotEligible})
				case !cd.FitsContext:
					w.skipped = append(w.skipped, ClassSkipped{Model: m, Class: name, Reason: ClassSkipContextTooSmall})
				case !cd.Capable:
					w.skipped = append(w.skipped, ClassSkipped{Model: m, Class: name, Reason: ClassSkipCapabilityMismatch})
				default:
					w.selectable = append(w.selectable, m)
				}
			}
			if len(w.selectable) > 0 {
				group := BalanceGroup{Name: cls.Name, Members: w.selectable, Strategy: cls.Strategy}
				// Resolve the class's weights over the selectable members only, so the
				// synthetic group passes BalanceGroup.Validate: a filtered-out member's
				// weight would read as undeclared and push Select onto its
				// invalid-declaration fail-open.
				group.Weights = make(map[string]int, len(w.selectable))
				for _, m := range w.selectable {
					group.Weights[m] = cls.WeightOf(m)
				}
				var s BalanceSnapshot
				if snap != nil {
					s = snap(cls)
				} else {
					// Group must equal the group name or the plan reports group_mismatch;
					// an empty snapshot with the right tag is the fail-open reading.
					s = BalanceSnapshot{Group: cls.Name}
				}
				w.order = (BalancePolicy{}).Select(group, s, key).Members
			}
			totalSkips += len(w.skipped)
		}
		walks = append(walks, w)
	}

	decision := ClassDecision{
		Schema:         ModelClassSchema,
		RequestedModel: requested,
		RequestedClass: ladder.Preferred,
	}

	audit := make([]string, 0, 2+len(walks)+totalSkips)
	audit = append(audit, "model_class: requested="+decision.RequestedModel+
		" class="+ladder.Preferred+" fallbacks="+itoa(len(ladder.Fallbacks)))
	for _, w := range walks {
		audit = append(audit, "model_class: class="+w.name+
			" candidates="+itoa(w.candidates)+" selectable="+itoa(len(w.selectable)))
		for _, sk := range w.skipped {
			audit = append(audit, "model_class: skip="+sk.Model+" class="+sk.Class+" reason="+string(sk.Reason))
		}
	}

	skipped := make([]ClassSkipped, 0, totalSkips)
	winner := -1
	for i, w := range walks {
		if len(w.order) > 0 {
			winner = i
			break
		}
	}
	if winner >= 0 {
		w := walks[winner]
		decision.Chosen = true
		decision.ServedModel = w.order[0]
		decision.ServedClass = w.name
		switch {
		case w.name != ladder.Preferred:
			decision.Transition = classTransitionStepDown
		case decision.ServedModel != decision.RequestedModel:
			decision.Transition = classTransitionSameClass
		}
		// Full walk order: the served class's strategy order (winner first), then later
		// classes' selectable members in declared class order. Earlier classes
		// contributed nothing selectable, which is why the walk reached this one.
		decision.Order = make([]string, 0, len(w.order)+1)
		decision.Order = append(decision.Order, w.order...)
		for i := winner + 1; i < len(walks); i++ {
			decision.Order = append(decision.Order, walks[i].order...)
		}
	}
	for _, w := range walks {
		skipped = append(skipped, w.skipped...)
	}
	decision.Skipped = skipped

	transition := decision.Transition
	if transition == "" {
		transition = "none"
	}
	chosen := decision.ServedModel
	if chosen == "" {
		chosen = "none"
	}
	audit = append(audit, "model_class: requested="+decision.RequestedModel+
		" class="+ladder.Preferred+" transition="+transition+" chosen="+chosen)
	decision.Audit = audit
	return decision
}
