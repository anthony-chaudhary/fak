package modelroute

import (
	"fmt"
	"strings"
)

// DECLARING WHAT A SUB-AGENT IS DOING (epic #5416, track E).
//
// PlaceSpawn already answers "which rung should this child run on", and it refuses an
// empty work class on purpose: an unclassified spawn is a missing classification, not a
// routine one, and letting delegation reach a cheap placement without stating what the
// work IS would be exactly the shape a floor bypass takes. That refusal is correct and
// stays. The consequence is that nothing can call PlaceSpawn until something can say
// what the child is for.
//
// Nothing could. A spawn arrives at the gateway as an admitted tool call whose arguments
// name an agent TYPE — the caller's own structured choice of which kind of agent to run
// — and a prompt. The prompt is prose, and reading a work class out of prose is the
// guess this epic refuses everywhere else (see ClassOf: a class is DECLARED, or it is
// conservative). The type is not prose, but it is also not a work class: only an
// operator knows whether the agent type their fleet calls "explore" does bounded lookup
// work or ships code.
//
// So the missing piece is a DECLARATION, and this is it: an operator states, in the
// roster they already own, which of their spawn types do which class of work. That keeps
// the one rule the classifier is built on — the only input allowed to place work on a
// cheap rung is an operator saying so — while making the rung reachable for the traffic
// the epic exists to move.

// SpawnClass declares that sub-agents spawned under one agent TYPE do work of a given
// class. Type is the token a spawn's arguments carry (whatever the harness calls it:
// "explore", "code-reviewer", "general-purpose"); Class is the closed-vocabulary
// WorkClass whose risk floor then governs the child's placement.
type SpawnClass struct {
	Type  string    `json:"type"`
	Class WorkClass `json:"class"`
}

// SpawnClassFor resolves the work class an operator DECLARED for a spawn's agent type,
// reporting whether they declared one at all.
//
// It is fail-closed in the only direction that matters. An undeclared type — a fleet
// that added an agent type and did not classify it, a harness that names its types
// differently, a roster with no spawn_classes block at all — yields no class, and
// PlaceSpawn then refuses rather than placing the child somewhere cheap. The cost of an
// omission is that a spawn keeps whatever placement it has today; the cost of guessing
// would be a sub-agent on a laptop doing work nobody said was laptop work.
//
// The match is EXACT on a case- and whitespace-normalized token, never a prefix and
// never a glob, for the reason Admits gives about principals: "explore" must not answer
// for "explore-and-delete". A roster is small and hand-written, so an operator who wants
// two types classified writes two lines.
func (r Roster) SpawnClassFor(declaredType string) (WorkClass, bool) {
	key := normalizeSpawnType(declaredType)
	if key == "" {
		return "", false
	}
	for _, sc := range r.SpawnClasses {
		if normalizeSpawnType(sc.Type) != key {
			continue
		}
		// Re-parse rather than trusting the stored value: a Roster built in code
		// (not through Validate) could carry a class token this package does not
		// recognize, and an unrecognized class must read as "nothing was declared"
		// rather than flow onward as a WorkClass that PolicyFor will not know.
		if c, ok := parseWorkClass(string(sc.Class)); ok {
			return c, true
		}
		return "", false
	}
	return "", false
}

// normalizeSpawnType is the one place a spawn-type token is folded for comparison, so
// the lookup and the duplicate check in Validate can never disagree about what makes
// two declarations the same.
func normalizeSpawnType(raw string) string {
	return strings.ToLower(strings.TrimSpace(raw))
}

// validateSpawnClasses is the roster-load half of the declaration: every entry must name
// a non-empty type and a class this package actually knows.
//
// Both failures are LOUD rather than skipped. A silently-dropped entry is the worst
// outcome available here, because it looks identical to a correct declaration from the
// operator's side while behaving like an undeclared one — the roster says the fleet's
// reviewers are routine work, the placement says otherwise, and nothing points at the
// typo. A duplicate is refused for the same reason and not resolved last-wins: a roster
// that names one type twice is ambiguous about the single thing the entry exists to
// state, and picking a winner would make the ambiguity invisible.
func validateSpawnClasses(scs []SpawnClass) error {
	seenType := make(map[string]bool, len(scs))
	for i, sc := range scs {
		key := normalizeSpawnType(sc.Type)
		if key == "" {
			return fmt.Errorf("modelroute: spawn class %d has an empty type", i)
		}
		if err := safeRouteToken("spawn class type", key); err != nil {
			return err
		}
		if seenType[key] {
			return fmt.Errorf("modelroute: duplicate spawn class for type %q", key)
		}
		seenType[key] = true
		if strings.TrimSpace(string(sc.Class)) == "" {
			return fmt.Errorf("modelroute: spawn class for type %q has an empty class "+
				"(omit the entry to leave the type undeclared; an empty class is not a declaration)", key)
		}
		if _, ok := parseWorkClass(string(sc.Class)); !ok {
			return fmt.Errorf("modelroute: spawn class for type %q names unknown work class %q (want one of %s)",
				key, sc.Class, workClassVocabulary())
		}
	}
	return nil
}

// workClassVocabulary renders the closed class vocabulary for an error message, so a
// typo in a roster is answered with the actual options rather than a bare rejection.
func workClassVocabulary() string {
	all := []WorkClass{ClassRoutine, ClassNormalImpl, ClassUltraHard, ClassSecurityRelease}
	parts := make([]string, 0, len(all))
	for _, c := range all {
		parts = append(parts, string(c))
	}
	return strings.Join(parts, ", ")
}
