package main

// hooks_scope.go — the SCOPE half of a grounded verdict (#5603).
//
// #5602 gave a verdict its denominator: how many items each gate judged. That is only half a
// claim. "3 gates judged 40 files and found nothing" still never says WHICH population those 40
// were drawn from, and fak deliberately runs the same gate names over two different populations —
// the staged set at the commit boundary, and the whole tracked tree in `fak hygiene`.
// internal/hooks/hooks.go documents two of these pairs in its own registration comments
// (UNTIERED_LEAF / TIER_DECLARED at :116-123, GOFMT / `make ci`'s gofmt-check at :124-129). Both
// halves print the word "clean", and on a shared trunk the gap between them is exactly where a
// peer's unstaged drift lives.
//
// The second narrowing is the operator's own. A gate set to `off`, escaped once, or softened to
// `warn` is deliberately NOT counted as skipped (cmd/fak/hooks.go, the `continue` above
// checkWithinBudget) — that is intent, not a degraded run, and #5299's degraded-run line has to
// keep meaning "a checker broke". The reasoning is right and this file does not disturb it. But
// intent still narrows what the verdict covers, so a report that never states it lets a run the
// operator hollowed out read exactly like a full one.
//
// Scope is therefore two independent facts, kept apart on purpose:
//
//	population — staged set vs whole tree. Not a per-run choice; a property of which command ran.
//	narrowing  — which gates the operator turned off, escaped, or left unable to refuse.
//
// Neither is a verdict. Nothing here changes what any gate decides, which gates are enabled, or
// any DefaultMode — this leaf only makes the report say what it was already quantifying over.

import (
	"fmt"
	"os"
	"strings"
)

// The two populations, named once so the staged gate and its whole-tree twin cannot drift apart
// in spelling while claiming to be different in substance.
const (
	scopePopulationStaged = "staged"
	scopePopulationTree   = "tree"
)

// scopePopulationNote spells out, in a reader's terms, what a population includes AND what it
// leaves out. The two strings are deliberately not templates of one another: the sets differ in
// both directions — staged-only cannot see an unstaged edit, whole-tree cannot see an untracked
// file — so a reader who has learned one still has to read the other.
func scopePopulationNote(population string) string {
	switch population {
	case scopePopulationTree:
		return "WHOLE-TREE (every tracked file, including edits no commit has staged; untracked files NOT judged)"
	default:
		return "STAGED-ONLY (untracked and unstaged paths NOT judged)"
	}
}

// gateNarrowing is the operator-intent half of a run's scope. It is reported separately from
// #5299's skipped ledger, and must stay separate: a gate the operator turned off and a gate that
// BROKE are different states, and folding them together would either cry degradation over a
// deliberate choice or hide a broken checker behind one.
type gateNarrowing struct {
	// NotRun names gates that produced no verdict at all this run — `off`, escaped once, or (in
	// hygiene) never selected by --gates.
	NotRun []string
	// Advisory names gates that ran and whose findings CANNOT refuse. They are the subtlest
	// case in the whole report: they appear in the per-gate ledger, they judged real candidates,
	// and their verdict still cannot stop the commit.
	Advisory []string
	// ByOperator names the subset THIS RUN moved off its compiled default — a FLEET_<NAME>_GUARD
	// env override or one-shot escape at the commit boundary, a --gates selector in hygiene. A
	// gate that ships advisory or default-off by design and one an operator quietened just now
	// are not the same claim, and this is the field that keeps them apart: without it, a run
	// somebody hollowed out reads as fak's shipped posture.
	ByOperator []string
}

func (n gateNarrowing) empty() bool {
	return len(n.NotRun) == 0 && len(n.Advisory) == 0 && len(n.ByOperator) == 0
}

// scopeNameCap bounds how many gate names a narrowing clause lists before "+N more". A
// `--gates INDEX_SYNC` sweep deselects three dozen hygiene gates, and a clause that named every
// one of them would bury the number that matters under the list that does not.
const scopeNameCap = 4

// scopeNames renders a capped, count-led list. The COUNT is never capped — a truncated list is a
// readability tradeoff, but a truncated count would be the same understatement this epic exists
// to remove.
func scopeNames(label string, names []string) string {
	if len(names) == 0 {
		return ""
	}
	shown := names
	extra := 0
	if len(shown) > scopeNameCap {
		extra = len(shown) - scopeNameCap
		shown = shown[:scopeNameCap]
	}
	list := strings.Join(shown, ", ")
	if extra > 0 {
		list = fmt.Sprintf("%s +%d more", list, extra)
	}
	return fmt.Sprintf("%d %s (%s)", len(names), label, list)
}

// clause renders the operator-narrowing sentence, or "" when the operator narrowed nothing. The
// empty case prints nothing at all rather than "0 narrowed": a permanent zero clause trains a
// reader to skip the sentence in the runs where it is the whole story.
//
// No sort here on purpose — both callers append while walking their gate registry, which is a
// slice, so the order is already stable run to run and matches the order the gates are reported
// in elsewhere. Re-sorting would make the clause disagree with the per-gate ledger beside it.
func (n gateNarrowing) clause() string {
	if n.empty() {
		return ""
	}
	var parts []string
	for _, p := range []string{
		scopeNames("not run", n.NotRun),
		scopeNames("advisory-only", n.Advisory),
		scopeNames("moved off the compiled default this run", n.ByOperator),
	} {
		if p != "" {
			parts = append(parts, p)
		}
	}
	return "gate set NARROWED by operator intent: " + strings.Join(parts, "; ")
}

// runScope is what one gate run quantified over: the population, and how the operator narrowed
// the gate set within it.
type runScope struct {
	Population string
	Narrowing  gateNarrowing
}

// note is the human clause appended to a summary line.
func (s runScope) note() string {
	note := "scope " + scopePopulationNote(s.Population)
	if c := s.Narrowing.clause(); c != "" {
		note += "; " + c
	}
	return note
}

// narrowingPayload is the machine-readable half. Every key is always present with an empty list
// rather than null, on the same rule as #5299's skipped_gates: a consumer must never have to
// decide whether an absent key means "none" or "this build does not report it".
func (s runScope) narrowingPayload() map[string]any {
	return map[string]any{
		"gates_not_run":          emptyIfNil(s.Narrowing.NotRun),
		"gates_advisory":         emptyIfNil(s.Narrowing.Advisory),
		"gates_operator_changed": emptyIfNil(s.Narrowing.ByOperator),
		"narrowed":               !s.Narrowing.empty(),
		"population_note":        scopePopulationNote(s.Population),
	}
}

func emptyIfNil(s []string) []string {
	if s == nil {
		return []string{}
	}
	return s
}

// gateModeIsEnvSet reports whether an operator set this gate's mode explicitly this run, as
// opposed to inheriting the compiled DefaultMode. It reads the same variable gateModeDefault
// does, so the two can never disagree about where a mode came from.
func gateModeIsEnvSet(modeEnv string) bool {
	return modeEnv != "" && strings.TrimSpace(os.Getenv(modeEnv)) != ""
}
