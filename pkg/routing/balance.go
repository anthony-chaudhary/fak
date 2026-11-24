package routing

import (
	"fmt"
	"strings"
)

// BalanceSchema identifies the balance-group manifest/receipt shape.
const BalanceSchema = "routing-balance/1"

// BalanceStrategy is the closed vocabulary of proactive selection strategies, each
// a pure function of (weights, scores, key). The default, priority, does not split.
type BalanceStrategy string

const (
	// BalanceStrategyPriority keeps declared order and ignores the scores.
	BalanceStrategyPriority BalanceStrategy = "priority"
	// BalanceStrategyWeightedHash ranks members by weighted rendezvous over
	// (weight x headroom), keyed on the request identity.
	BalanceStrategyWeightedHash BalanceStrategy = "weighted-hash"
	// BalanceStrategyLeastLoad puts the least loaded eligible member first; members
	// within LeastLoadTieBand are ties broken by the weighted-hash ranking.
	BalanceStrategyLeastLoad BalanceStrategy = "least-load"
	// BalanceStrategyLeastLatency puts the lowest measured latency first, with
	// ties broken by the weighted-hash ranking.
	BalanceStrategyLeastLatency BalanceStrategy = "least-latency"
	// BalanceStrategyWeightedShare allocates requests by declared weights only;
	// admission and health remain with the gateway walk.
	BalanceStrategyWeightedShare BalanceStrategy = "weighted-share"
)

// BalanceStrategyNames returns a fresh copy of the strategy vocabulary in
// declaration order.
func BalanceStrategyNames() []string {
	return []string{
		string(BalanceStrategyPriority),
		string(BalanceStrategyWeightedHash),
		string(BalanceStrategyLeastLoad),
		string(BalanceStrategyLeastLatency),
		string(BalanceStrategyWeightedShare),
	}
}

// IsKnown reports whether s is a member of the closed vocabulary.
func (s BalanceStrategy) IsKnown() bool {
	switch s {
	case BalanceStrategyPriority, BalanceStrategyWeightedHash, BalanceStrategyLeastLoad, BalanceStrategyLeastLatency, BalanceStrategyWeightedShare:
		return true
	default:
		return false
	}
}

// orDefault resolves only the zero value; an unknown strategy is left for Validate
// to reject rather than silently downshifting to a different policy.
func (s BalanceStrategy) orDefault() BalanceStrategy {
	if s == "" {
		return BalanceStrategyPriority
	}
	return s
}

// exploresUnmeasured reports whether an unmeasured member competes as idle; these
// strategies learn a member's numbers only by sending it traffic.
func (s BalanceStrategy) exploresUnmeasured() bool {
	return s == BalanceStrategyLeastLoad || s == BalanceStrategyLeastLatency
}

// competesUnmeasured reports whether an unmeasured member keeps its place among
// measured peers; demoting it would leave it unmeasured forever.
func (s BalanceStrategy) competesUnmeasured() bool {
	return s.exploresUnmeasured() || s == BalanceStrategyWeightedHash || s == BalanceStrategyWeightedShare
}

// BalanceGroup is the declared equivalence class of interchangeable targets: member
// identities, a strategy name and per-member weights. It is declaration only and
// never mutated. It must be a contiguous run of the chain: reordering across a gap
// could pull a request over a capability boundary, which is the failover plane's job.
type BalanceGroup struct {
	// Name is the group's identity in the audit trail. Required.
	Name string `json:"name"`
	// Members are the target identities in declared order, which is every
	// strategy's tie-break.
	Members []string `json:"members"`
	// Strategy selects how members are ordered; empty means priority.
	Strategy BalanceStrategy `json:"strategy,omitempty"`
	// Weights is the per-member ratio; absent means weight 1. Zero is a declared
	// drain: the member keeps its position and receives nothing.
	Weights map[string]int `json:"weights,omitempty"`
}

// DefaultBalanceWeight is the weight of a member absent from BalanceGroup.Weights.
const DefaultBalanceWeight = 1

// WeightOf returns the declared weight for a member, or DefaultBalanceWeight.
// Presence, not truthiness, decides, so an explicit 0 stays a drain.
func (g BalanceGroup) WeightOf(member string) int {
	if w, ok := g.Weights[member]; ok {
		return w
	}
	return DefaultBalanceWeight
}

// Validate checks the group's intrinsic invariants and fails on the first offence.
// Contiguity depends on the chain, so ValidateContiguous owns it.
func (g BalanceGroup) Validate() error {
	if strings.TrimSpace(g.Name) == "" {
		return fmt.Errorf("routing: balance group has an empty name")
	}
	if len(g.Members) == 0 {
		return fmt.Errorf("routing: balance group %q has no members", g.Name)
	}
	// A single-member group is legal; Plan reports it as a no-op.
	seen := make(map[string]int, len(g.Members))
	for i, m := range g.Members {
		if m == "" {
			return fmt.Errorf("routing: balance group %q member %d is empty", g.Name, i)
		}
		if first, dup := seen[m]; dup {
			return fmt.Errorf("routing: balance group %q member %q is declared twice (indexes %d and %d)", g.Name, m, first, i)
		}
		seen[m] = i
	}
	strategy := g.Strategy.orDefault()
	if !strategy.IsKnown() {
		return fmt.Errorf("routing: balance group %q has unknown strategy %q (known: %s)", g.Name, g.Strategy, strings.Join(BalanceStrategyNames(), ", "))
	}
	for member, w := range g.Weights {
		if _, declared := seen[member]; !declared {
			return fmt.Errorf("routing: balance group %q declares weight %d for undeclared member %q", g.Name, w, member)
		}
		if w < 0 {
			return fmt.Errorf("routing: balance group %q member %q has negative weight %d", g.Name, member, w)
		}
	}
	// An all-drained group can never select; refuse it here, not as a runtime no_eligible_member.
	allDrained := true
	for _, m := range g.Members {
		if g.WeightOf(m) > 0 {
			allDrained = false
			break
		}
	}
	if allDrained {
		return fmt.Errorf("routing: balance group %q has every member at weight 0, so it can never select", g.Name)
	}
	return nil
}

// ValidateContiguous checks that the members form one contiguous run of the
// resolved chain; a RouteTable constructor calls it.
func (g BalanceGroup) ValidateContiguous(order []string) error {
	positions := make([]int, 0, len(g.Members))
	for _, m := range g.Members {
		pos := -1
		for i, o := range order {
			if o == m {
				pos = i
				break
			}
		}
		if pos < 0 {
			return fmt.Errorf("routing: balance group %q member %q is not in the resolved chain", g.Name, m)
		}
		positions = append(positions, pos)
	}
	for i := 1; i < len(positions); i++ {
		if positions[i] != positions[i-1]+1 {
			return fmt.Errorf("routing: balance group %q is not contiguous in the resolved chain (positions %v)", g.Name, positions)
		}
	}
	return nil
}

// GroupPlanMember is one member's declared entry in the group receipt.
type GroupPlanMember struct {
	// Member is the member identity.
	Member string `json:"member"`
	// Weight is the declared weight (DefaultBalanceWeight when undeclared).
	Weight int `json:"weight"`
	// Drained is true for a declared weight of 0; a drained member is still a declared peer.
	Drained bool `json:"drained,omitempty"`
}

// GroupPlan is the declaration receipt an operator reads to see what a group says,
// before any measurement is involved.
type GroupPlan struct {
	// Schema is the receipt shape tag.
	Schema string `json:"schema"`
	// Name is the group's name.
	Name string `json:"name"`
	// Strategy is the resolved strategy.
	Strategy BalanceStrategy `json:"strategy"`
	// NoOp is true when the group balances nothing (a one-member group).
	NoOp bool `json:"noop,omitempty"`
	// NoOpReason names why the group is a no-op.
	NoOpReason string `json:"noop_reason,omitempty"`
	// Contiguous records whether the members form one run of the order; a caller
	// seeing false must refuse via ValidateContiguous.
	Contiguous bool `json:"contiguous"`
	// Members are the group's members in declaring-chain order.
	Members []GroupPlanMember `json:"members"`
	// Audit is one legible line per member plus one for the group.
	Audit []string `json:"audit,omitempty"`
}

// Plan renders the declaration receipt against the resolved chain. It reports a
// non-contiguous declaration rather than raising; callers that must not proceed
// call ValidateContiguous.
func (g BalanceGroup) Plan(order []string) GroupPlan {
	strategy := g.Strategy.orDefault()
	plan := GroupPlan{
		Schema:     BalanceSchema,
		Name:       g.Name,
		Strategy:   strategy,
		Contiguous: g.ValidateContiguous(order) == nil,
		Audit: []string{
			fmt.Sprintf("balance: group=%s strategy=%s members=%d declared_against=%d",
				g.Name, strategy, len(g.Members), len(order)),
		},
	}
	if len(g.Members) == 1 {
		plan.NoOp = true
		plan.NoOpReason = "single member: a group of one balances nothing, so the declared order is the order"
		plan.Audit = append(plan.Audit, fmt.Sprintf("balance: group=%s noop=%s", g.Name, plan.NoOpReason))
	}
	// A member absent from the order is appended rather than dropped.
	ranked := make([]GroupPlanMember, 0, len(g.Members))
	placed := make(map[string]bool, len(g.Members))
	for _, o := range order {
		for _, m := range g.Members {
			if m == o && !placed[m] {
				placed[m] = true
				ranked = append(ranked, g.planMember(m))
			}
		}
	}
	for _, m := range g.Members {
		if !placed[m] {
			placed[m] = true
			ranked = append(ranked, g.planMember(m))
		}
	}
	plan.Members = ranked
	for _, m := range ranked {
		if m.Drained {
			plan.Audit = append(plan.Audit, fmt.Sprintf("balance: group=%s member=%s weight=0 drained=true", g.Name, m.Member))
			continue
		}
		plan.Audit = append(plan.Audit, fmt.Sprintf("balance: group=%s member=%s weight=%d", g.Name, m.Member, m.Weight))
	}
	return plan
}

func (g BalanceGroup) planMember(member string) GroupPlanMember {
	w := g.WeightOf(member)
	return GroupPlanMember{Member: member, Weight: w, Drained: w == 0}
}
