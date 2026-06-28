package callavoid

import "sort"

// coverage.go — the PRODUCER half of #820 acceptance (1): a witnessed productive-deny
// fan-out emitted FROM A DENY RULE'S STRUCTURAL COVERAGE, not a guess.
//
// witness.go gave the CONSUMER side — Account credits a WitnessedRedirect whose variants
// are enumerated. But a WitnessedRedirect with a free-form Variants slice is only as
// honest as whoever filled it: a caller could still hand-assert the very list the
// speculative axis forbids. The missing piece is the GROUND the variants come from. A
// real productive deny fires because a structural rule covers a known, bounded domain of
// calls — a policy rule's covered-variant set, a capability glob's enumerated members, a
// scope's closed alternative list. Those covered variants are exactly the futile calls a
// naive agent would have walked and the deny structurally rules out. Enumerating THAT set
// is a witness; naming a number is a guess.
//
// DenyRuleCoverage models that ground: the closed, declared set of variants a deny rule
// structurally covers, plus a hard MaxFanout the rule itself declares (a rule can never
// claim to prune more than it covers). WitnessFromCoverage turns one fired rule into a
// WitnessedRedirect whose variants are drawn from the coverage and intersected with what
// the deny actually pruned — so the credited fan-out is the size of a concrete set the
// rule names, bounded by the rule's own declaration, and Account folds it as realized.
//
// This stays the tier-1 economics leaf: it imports nothing internal (sort + the type
// above). The kernel/adjudicator caller owns turning a live abi.Verdict into a
// DenyRuleCoverage (the rule id, its declared covered-variant set, and which of those the
// fired deny pruned); this leaf supplies the pure, bounded, deduplicating producer the
// caller maps onto, the same way fold.go supplies TallyFromCounters.

// DenyRuleCoverage is the STRUCTURAL coverage one deny rule declares: the closed set of
// futile variants the rule is known to cover, and the hard ceiling the rule itself places
// on how many it may ever credit. It is the witness's GROUND — a productive deny may only
// credit variants that appear in Covered, so the credited fan-out can never exceed what
// the rule structurally rules out (no unbounded counterfactual).
type DenyRuleCoverage struct {
	// Rule is the deny rule / policy id that fired (e.g. "policy.read-scope",
	// "cap.fs.write"). It is carried onto the WitnessedRedirect for attribution; it never
	// affects the credited count.
	Rule string `json:"rule"`
	// Covered is the rule's declared structural coverage: every futile variant this rule is
	// known to rule out. A witness may credit ONLY variants drawn from this set — that is
	// what makes the fan-out structural, not asserted. Duplicates and blanks here are
	// harmless; WitnessFromCoverage deduplicates and drops them.
	Covered []string `json:"covered"`
	// MaxFanout is the rule's OWN declared ceiling on the fan-out it may credit (0 means
	// "no rule-specific ceiling — fall back to the package per-deny cap"). A rule that
	// covers a large domain but only ever prunes a small bounded slice declares that here,
	// so a rule can never credit more than it claims to. The effective per-deny bound is
	// min(MaxFanout, the package fan-out cap).
	MaxFanout int `json:"max_fanout,omitempty"`
}

// WitnessFromCoverage produces the WitnessedRedirect a fired productive deny credits, with
// its variants drawn FROM the rule's structural coverage rather than asserted. `pruned`
// names the variants the deny actually ruled out on this call; the credited variant set is
// pruned ∩ Covered — a deny may only credit a futile variant the rule structurally covers,
// so a caller cannot smuggle in a variant outside the rule's declared domain. The result
// is deduplicated, blank-free, sorted (deterministic), and clamped to the rule's own
// MaxFanout, so the witness is bounded by the rule's declaration before Account ever
// applies the package caps.
//
// An empty intersection (the deny pruned nothing the rule covers, or named nothing)
// returns a witness with no variants — which Account treats as an effective hard deny,
// crediting nothing. That is the fail-toward-zero default: a deny whose structural
// coverage cannot witness its pruning earns no realized credit.
func WitnessFromCoverage(cov DenyRuleCoverage, pruned []string) WitnessedRedirect {
	covered := make(map[string]struct{}, len(cov.Covered))
	for _, v := range cov.Covered {
		if v == "" {
			continue
		}
		covered[v] = struct{}{}
	}

	seen := make(map[string]struct{}, len(pruned))
	var variants []string
	for _, v := range pruned {
		if v == "" {
			continue // a blank names no futile call.
		}
		if _, ok := covered[v]; !ok {
			continue // not in the rule's declared coverage — a guess, never credited.
		}
		if _, dup := seen[v]; dup {
			continue
		}
		seen[v] = struct{}{}
		variants = append(variants, v)
	}
	sort.Strings(variants) // deterministic order; the credited COUNT is independent of order, but a stable witness is auditable.

	// Apply the rule's OWN declared ceiling here, before Account's package caps. A rule that
	// declares MaxFanout=N can never witness more than N variants even if it pruned more —
	// the rule's structural claim about its own reach is the tightest honest bound.
	if cov.MaxFanout > 0 && len(variants) > cov.MaxFanout {
		variants = variants[:cov.MaxFanout]
	}
	return WitnessedRedirect{Rule: cov.Rule, Variants: variants}
}

// WitnessesFromCoverage maps a batch of fired deny rules onto witnessed redirects, one per
// rule, in input order. It is the shape the adjudicator caller folds once per session: for
// each productive deny it recorded, it supplies the rule's declared coverage and the
// variants that deny pruned, and gets back the bounded, structural witnesses to hand to
// TallyFromCountersWitnessed. A rule whose intersection is empty still yields a (variant-
// less) witness so the count of attempted productive denies is preserved and surfaced as
// WitnessedEmpty rather than silently dropped.
func WitnessesFromCoverage(fired []FiredDeny) []WitnessedRedirect {
	out := make([]WitnessedRedirect, len(fired))
	for i, f := range fired {
		out[i] = WitnessFromCoverage(f.Coverage, f.Pruned)
	}
	return out
}

// FiredDeny pairs one fired deny rule's declared coverage with the variants that deny
// actually pruned on the call, so a caller can hand a whole session's productive denies to
// WitnessesFromCoverage in one slice. The caller builds it from the live adjudicator
// verdict (the rule id + its covered-variant set) and the call that was denied (which
// covered variants the deny structurally ruled out).
type FiredDeny struct {
	Coverage DenyRuleCoverage `json:"coverage"`
	Pruned   []string         `json:"pruned"`
}
