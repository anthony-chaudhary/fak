package humanctl

import (
	"fmt"
	"sort"
	"strings"
)

// Verb is the stable identity of what a human is trying to make an agent do.
// It is intentionally independent of transport words such as send, queue, or
// inject: those say when a control arrives, not what outcome the human wants.
//
// Invariant: verb values are members of the closed catalog defined in the package index.
// Guard: uncataloged verbs fail validation when parsed or executed.
type Verb string

// Canonical verbs defined in the closed human control catalog.
const (
	FlagConcern  Verb = "flag_concern"
	Reject       Verb = "reject"
	Reinforce    Verb = "reinforce"
	Redirect     Verb = "redirect"
	Avoid        Verb = "avoid"
	Prioritize   Verb = "prioritize"
	Deprioritize Verb = "deprioritize"
	Narrow       Verb = "narrow"
	Broaden      Verb = "broaden"
	Investigate  Verb = "investigate"
	Verify       Verb = "verify"
	Retry        Verb = "retry"
	Undo         Verb = "undo"
	Continue     Verb = "continue"
	EndTurn      Verb = "end_turn"
	Pause        Verb = "pause"
	Resume       Verb = "resume"
	Stop         Verb = "stop"
)

// Family names the human outcome being controlled, rather than the UI gesture.
//
// Invariant: family membership partitions the verb index into disjoint semantic categories.
// Guard: unknown families are rejected by catalog validation.
type Family string

// Outcome families defined in the closed human control catalog.
const (
	Evaluation Family = "evaluation"
	Direction  Family = "direction"
	Allocation Family = "allocation"
	Scope      Family = "scope"
	Assurance  Family = "assurance"
	Recovery   Family = "recovery"
	Lifecycle  Family = "lifecycle"
)

// Strength is an ordinal modifier. Unspecified preserves the human's ambiguity;
// callers must not silently turn it into Medium.
//
// Invariant: non-empty strength must be one of low, medium, high, or absolute.
// Guard: unrecognised strength tokens fail validation rather than coercing defaults.
type Strength string

// Strength levels for ordinal control modulation.
const (
	StrengthUnspecified Strength = ""
	StrengthLow         Strength = "low"
	StrengthMedium      Strength = "medium"
	StrengthHigh        Strength = "high"
	StrengthAbsolute    Strength = "absolute"
)

// Definition is one row in the closed, typed index.
//
// Invariant: every definition has a non-empty verb, family, default strength, and intent.
// Guard: terminal and non-composable verbs must not permit subsequent control chaining.
type Definition struct {
	Verb            Verb
	Family          Family
	Aliases         []string
	Intent          string
	NeedsTarget     bool
	AcceptsReason   bool
	DefaultStrength Strength
	Terminal        bool
	CanCompose      bool
}

var definitions = []Definition{
	{FlagConcern, Evaluation, []string{"this seems off", "this seems wrong", "something is off"}, "record concern without inventing a diagnosis", false, true, StrengthLow, false, true},
	{Reject, Evaluation, []string{"this is wrong", "reject this"}, "mark an artifact or direction as unacceptable", false, true, StrengthHigh, false, true},
	{Reinforce, Direction, []string{"double down", "keep this direction", "more like this"}, "increase commitment to the current direction", false, true, StrengthHigh, false, true},
	{Redirect, Direction, []string{"change direction", "do this instead", "focus on"}, "replace or bend the current direction toward a target", true, true, StrengthMedium, false, true},
	{Avoid, Direction, []string{"do not", "don't", "avoid"}, "exclude an action, approach, or outcome", true, true, StrengthAbsolute, false, true},
	{Prioritize, Allocation, []string{"prioritize", "do this first"}, "allocate more attention or earlier execution to a target", true, true, StrengthHigh, false, true},
	{Deprioritize, Allocation, []string{"deprioritize", "do this later"}, "allocate less attention or later execution to a target", true, true, StrengthLow, false, true},
	{Narrow, Scope, []string{"narrow", "reduce scope", "focus only on"}, "reduce the admitted problem or solution space", true, true, StrengthMedium, false, true},
	{Broaden, Scope, []string{"broaden", "look beyond", "consider more"}, "expand the admitted problem or solution space", false, true, StrengthMedium, false, true},
	{Investigate, Assurance, []string{"rethink", "investigate", "figure out why"}, "gather evidence before choosing a corrective action", false, true, StrengthMedium, false, true},
	{Verify, Assurance, []string{"verify", "prove", "check this"}, "require an independent or checkable witness", true, true, StrengthHigh, false, true},
	{Retry, Recovery, []string{"retry", "try again"}, "repeat an attempt, normally with changed evidence or constraints", false, true, StrengthMedium, false, true},
	{Undo, Recovery, []string{"undo", "backtrack", "revert"}, "return an affected state toward a prior checkpoint", false, true, StrengthHigh, false, true},
	{Continue, Lifecycle, []string{"continue", "keep going"}, "continue execution without changing direction", false, true, StrengthMedium, false, true},
	{EndTurn, Lifecycle, []string{"end turn"}, "end the current turn without suspending the session", false, true, StrengthHigh, false, false},
	{Pause, Lifecycle, []string{"pause", "hold"}, "suspend execution while preserving resumability", false, true, StrengthHigh, false, false},
	{Resume, Lifecycle, []string{"resume"}, "resume a suspended execution", false, true, StrengthHigh, false, false},
	{Stop, Lifecycle, []string{"stop", "cancel", "abort"}, "terminate the addressed execution", false, true, StrengthAbsolute, true, false},
}

// Instruction combines a typed control with human text that remains evidence,
// not syntax. Target says what to affect; Reason says why; Text preserves any
// unstructured qualification that a closed vocabulary cannot honestly encode.
//
// Invariant: verbs requiring targets must have non-blank target strings.
// Guard: invalid verbs, illegal strengths, or forbidden reasons cause Validate to fail closed.
type Instruction struct {
	Verb     Verb     `json:"verb"`
	Strength Strength `json:"strength,omitempty"`
	Target   string   `json:"target,omitempty"`
	Reason   string   `json:"reason,omitempty"`
	Text     string   `json:"text,omitempty"`
}

// Index returns a defensive copy in stable declaration order.
//
// Invariant: the returned slice contains isolated copies of definitions and alias slices.
// Guard: mutations by callers do not corrupt the internal package catalog.
func Index() []Definition {
	out := make([]Definition, len(definitions))
	for i, d := range definitions {
		out[i] = d
		out[i].Aliases = append([]string(nil), d.Aliases...)
	}
	return out
}

// Lookup resolves a canonical verb or exact, case-insensitive alias. It does
// not pretend to classify arbitrary prose; Text on Instruction is the escape
// hatch for ambiguity and future extractors can emit confidence separately.
//
// Invariant: exact canonical verb match or alias match returns ok=true and the canonical definition.
// Guard: unmatched query strings fail closed returning false and a zero-value Definition.
func Lookup(s string) (Definition, bool) {
	n := normalize(s)
	for _, d := range definitions {
		if n == normalize(string(d.Verb)) {
			return d, true
		}
		for _, alias := range d.Aliases {
			if n == normalize(alias) {
				return d, true
			}
		}
	}
	return Definition{}, false
}

// Validate checks the typed portion without discarding unstructured context.
//
// Invariant: validated instructions reference cataloged verbs with compliant targets and strengths.
// Guard: returns a non-nil error if the verb is unknown, target is missing when required, or strength is invalid.
func (i Instruction) Validate() error {
	d, ok := Lookup(string(i.Verb))
	if !ok || d.Verb != i.Verb {
		return fmt.Errorf("humanctl: unknown verb %q", i.Verb)
	}
	if d.NeedsTarget && strings.TrimSpace(i.Target) == "" {
		return fmt.Errorf("humanctl: %s requires a target", i.Verb)
	}
	if i.Strength != StrengthUnspecified && !validStrength(i.Strength) {
		return fmt.Errorf("humanctl: invalid strength %q", i.Strength)
	}
	if !d.AcceptsReason && strings.TrimSpace(i.Reason) != "" {
		return fmt.Errorf("humanctl: %s does not accept a reason", i.Verb)
	}
	return nil
}

// EffectiveStrength applies declared defaults only when an executor explicitly
// asks for them. The original Instruction remains losslessly unspecified.
//
// Invariant: explicit strength overrides default; unspecified returns catalog default or unspecified if unknown.
// Guard: unknown verbs without explicit strength return StrengthUnspecified.
func (i Instruction) EffectiveStrength() Strength {
	if i.Strength != StrengthUnspecified {
		return i.Strength
	}
	d, ok := Lookup(string(i.Verb))
	if !ok {
		return StrengthUnspecified
	}
	return d.DefaultStrength
}

// Compose validates an ordered control program. Terminal and suspension verbs
// cannot hide trailing instructions; that would make natural-language order
// lie about what can execute.
//
// Invariant: non-composable verbs must appear strictly in the final position of the sequence.
// Guard: fails closed with an error on any invalid instruction or misplaced terminal verb.
func Compose(instructions ...Instruction) ([]Instruction, error) {
	out := append([]Instruction(nil), instructions...)
	for idx, instruction := range out {
		if err := instruction.Validate(); err != nil {
			return nil, fmt.Errorf("instruction %d: %w", idx, err)
		}
		d, _ := Lookup(string(instruction.Verb))
		if idx < len(out)-1 && !d.CanCompose {
			return nil, fmt.Errorf("instruction %d: %s must be last", idx, instruction.Verb)
		}
	}
	return out, nil
}

// Families returns stable family names for inventory and UI grouping.
//
// Invariant: returns deduplicated, lexicographically sorted family names from the catalog.
// Guard: safe read-only projection constructed on each invocation.
func Families() []Family {
	seen := map[Family]bool{}
	var out []Family
	for _, d := range definitions {
		if !seen[d.Family] {
			seen[d.Family] = true
			out = append(out, d.Family)
		}
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i] < out[j] })
	return out
}

func normalize(s string) string {
	return strings.Join(strings.Fields(strings.ToLower(strings.TrimSpace(s))), " ")
}

func validStrength(s Strength) bool {
	return s == StrengthLow || s == StrengthMedium || s == StrengthHigh || s == StrengthAbsolute
}
