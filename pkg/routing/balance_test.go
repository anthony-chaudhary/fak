package routing

import (
	"strings"
	"testing"
)

func TestBalanceStrategyNamesIsAClosedSetInDeclarationOrder(t *testing.T) {
	got := BalanceStrategyNames()
	want := []string{"priority", "weighted-hash", "least-load", "least-latency", "weighted-share"}
	if len(got) != len(want) {
		t.Fatalf("BalanceStrategyNames() = %v, want exactly %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("BalanceStrategyNames()[%d] = %q, want %q", i, got[i], want[i])
		}
		if !BalanceStrategy(got[i]).IsKnown() {
			t.Fatalf("declared strategy %q does not report IsKnown", got[i])
		}
	}
	// The slice is a copy: a caller that mutates it must not reorder the
	// vocabulary for the next caller.
	got[0] = "mutated"
	if BalanceStrategyNames()[0] != "priority" {
		t.Fatal("BalanceStrategyNames() returned an aliased slice")
	}
	// Nothing outside the closed set is known. This is the assertion that
	// stops a later ticket from adding a strategy silently.
	for _, s := range []BalanceStrategy{"", "round-robin", "random", "cost", "Priority"} {
		if s.IsKnown() {
			t.Fatalf("strategy %q reports IsKnown but is not a declared member", s)
		}
	}
}

func TestBalanceGroupValidateAcceptsAWellFormedGroup(t *testing.T) {
	g := BalanceGroup{
		Name:     "fast-us",
		Members:  []string{"a", "b", "c"},
		Strategy: BalanceStrategyWeightedHash,
		Weights:  map[string]int{"a": 3, "b": 1, "c": 1},
	}
	if err := g.Validate(); err != nil {
		t.Fatalf("Validate() = %v, want nil", err)
	}
	plan := g.Plan([]string{"a", "b", "c"})
	if plan.Schema != BalanceSchema {
		t.Fatalf("plan schema = %q, want %q", plan.Schema, BalanceSchema)
	}
	if plan.NoOp {
		t.Fatal("a 3-member group must not be a no-op")
	}
	if len(plan.Members) != 3 {
		t.Fatalf("plan has %d members, want 3", len(plan.Members))
	}
	for i, want := range []string{"a", "b", "c"} {
		if plan.Members[i].Member != want {
			t.Fatalf("plan member %d = %q, want %q (declared order)", i, plan.Members[i].Member, want)
		}
	}
	// One audit line for the group plus one per member.
	if len(plan.Audit) != 4 {
		t.Fatalf("plan has %d audit lines, want 4 (group + 3 members): %v", len(plan.Audit), plan.Audit)
	}
}

func TestBalanceGroupValidateFailsLoudNamingTheOffence(t *testing.T) {
	cases := []struct {
		name  string
		group BalanceGroup
		want  string
	}{
		{
			name:  "empty name",
			group: BalanceGroup{Name: "  ", Members: []string{"a", "b"}},
			want:  "empty name",
		},
		{
			name:  "no members",
			group: BalanceGroup{Name: "g", Members: nil},
			want:  "has no members",
		},
		{
			name:  "empty member",
			group: BalanceGroup{Name: "g", Members: []string{"a", ""}},
			want:  "member 1 is empty",
		},
		{
			name:  "duplicate member",
			group: BalanceGroup{Name: "g", Members: []string{"a", "b", "a"}},
			want:  `member "a" is declared twice`,
		},
		{
			name:  "negative weight",
			group: BalanceGroup{Name: "g", Members: []string{"a", "b"}, Weights: map[string]int{"a": -1}},
			want:  `member "a" has negative weight -1`,
		},
		{
			name:  "weight on undeclared member",
			group: BalanceGroup{Name: "g", Members: []string{"a", "b"}, Weights: map[string]int{"c": 2}},
			want:  `undeclared member "c"`,
		},
		{
			name:  "unknown strategy",
			group: BalanceGroup{Name: "g", Members: []string{"a", "b"}, Strategy: BalanceStrategy("round-robin")},
			want:  "unknown strategy",
		},
		{
			name:  "all members drained",
			group: BalanceGroup{Name: "g", Members: []string{"a", "b"}, Weights: map[string]int{"a": 0, "b": 0}},
			want:  "can never select",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.group.Validate()
			if err == nil {
				t.Fatalf("Validate() = nil, want an error naming %q", tc.want)
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("Validate() = %q, want it to name %q", err.Error(), tc.want)
			}
		})
	}
}

func TestBalanceGroupZeroWeightIsADrainNotAnError(t *testing.T) {
	// Portkey documents weight 0 as "stop routing traffic to it without
	// removing it from Config". fak keeps that semantic, which is the whole
	// reason a weight is never defaulted at the selection site: a truthiness
	// default would silently un-drain a member the operator meant to drain.
	g := BalanceGroup{
		Name:     "drainable",
		Members:  []string{"a", "b"},
		Strategy: BalanceStrategyWeightedHash,
		Weights:  map[string]int{"a": 5, "b": 0},
	}
	if err := g.Validate(); err != nil {
		t.Fatalf("Validate() = %v, want nil: weight 0 is a declared drain", err)
	}
	plan := g.Plan([]string{"a", "b"})
	var drained int
	for _, m := range plan.Members {
		if m.Drained {
			drained++
			if m.Member != "b" {
				t.Fatalf("wrong member reported drained: %q", m.Member)
			}
		}
	}
	if drained != 1 {
		t.Fatalf("plan reports %d drained members, want 1", drained)
	}
}

func TestBalanceGroupSingleMemberIsALegalNoOp(t *testing.T) {
	// A group of one balances nothing, but it must still be a legal
	// declaration: an operator who groups a single member has expressed "this
	// one is the class". Refusing it here would make the no-op verdict dead
	// code for every caller.
	g := BalanceGroup{Name: "solo", Members: []string{"only"}}
	if err := g.Validate(); err != nil {
		t.Fatalf("Validate() = %v, want nil: a 1-member group is a no-op, not an error", err)
	}
	plan := g.Plan([]string{"only"})
	if !plan.NoOp {
		t.Fatal("a 1-member group must report the explicit no-op verdict")
	}
	if plan.NoOpReason == "" {
		t.Fatal("a no-op group must name why it is a no-op")
	}
}

func TestBalanceGroupValidateContiguous(t *testing.T) {
	g := BalanceGroup{Name: "tier", Members: []string{"b", "c"}}
	if err := g.ValidateContiguous([]string{"a", "b", "c", "d"}); err != nil {
		t.Fatalf("ValidateContiguous() = %v, want nil for a contiguous run", err)
	}
	// A discontiguous declaration is a typo far more often than an intent,
	// and re-ordering across one would let a well-scoring member pull a
	// request across a capability boundary.
	if err := g.ValidateContiguous([]string{"b", "a", "c"}); err == nil {
		t.Fatal("ValidateContiguous() = nil, want an error for a discontiguous group")
	}
	// A member absent from the chain is also a refusal, not a silent skip.
	if err := g.ValidateContiguous([]string{"a", "b"}); err == nil {
		t.Fatal("ValidateContiguous() = nil, want an error for a member outside the chain")
	}
}

func TestBalanceGroupPlanRendersInChainOrderWithoutReordering(t *testing.T) {
	g := BalanceGroup{Name: "tier", Members: []string{"c", "a"}}
	plan := g.Plan([]string{"a", "b", "c"})
	if plan.Contiguous {
		t.Fatal("plan reports contiguous for members at chain positions 2 and 0")
	}
	// Even discontiguous, the receipt reads the way the chain does and says so.
	if plan.Members[0].Member != "a" || plan.Members[1].Member != "c" {
		t.Fatalf("plan order = %v, want [a c] in chain order", plan.Members)
	}
}

func TestBalanceGroupWeightOfDefaultsToOne(t *testing.T) {
	g := BalanceGroup{Name: "g", Members: []string{"a", "b"}, Weights: map[string]int{"a": 4}}
	if got := g.WeightOf("a"); got != 4 {
		t.Fatalf("WeightOf(a) = %d, want 4", got)
	}
	if got := g.WeightOf("b"); got != DefaultBalanceWeight {
		t.Fatalf("WeightOf(b) = %d, want the declared default %d", got, DefaultBalanceWeight)
	}
}
