package turnbench

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/agent"
)

// This file fences RUNG-NAME TOTALITY: every rung the lever-flip sweep folds must be
// addressable by a non-empty, UNIQUE canonical name, so abi.WithoutRung can ablate it
// independently and its marginal value is measurable. Two failure modes make a rung
// invisible to the whole ablation harness (#3974):
//
//   - an EMPTY By: abi.RungName yields "" (mask.go:40, "unmaskable rather than removing
//     the wrong one"), WithoutRung always carries it through (mask.go:73), and
//     defaultLevers drops it from the full sweep (leverflip.go:206) — its causal value
//     can never be measured, so the sweep quietly stops being total for exactly the rung
//     that dodges it.
//   - a DUPLICATE canonical name: WithoutRung removes EVERY rung whose RungName equals the
//     name (mask.go:78), so two rungs sharing a By collapse together and neither is
//     independently ablatable.
//
// Today every shipped rung happens to be named and distinct; nothing fenced the next one.
// This gate does — it reds the build the moment a newly-registered rung dodges the sweep.
// Precedent: architest's registry-derived adjudicator fences (TestEveryAdjudicatorIsExecFree).

// rungNameViolations reports every rung in chain that cannot be uniquely addressed by its
// canonical abi.RungName: one message per unnamed rung and per duplicate-named rung, each
// naming the offending rung's concrete Go type and citing the mask.go naming contract it
// breaks. An empty slice means the chain is TOTALLY ablatable — every rung is a distinct,
// independently-flippable lever.
func rungNameViolations(chain []abi.Adjudicator) []string {
	var out []string
	seen := map[string]bool{}
	for i, a := range chain {
		name := abi.RungName(a)
		typ := fmt.Sprintf("%T", a)
		switch {
		case name == "":
			out = append(out, fmt.Sprintf("rung #%d (%s) reports an EMPTY canonical By: abi.RungName yields \"\", "+
				"so abi.WithoutRung can never address it (mask.go: an unnamed rung is unmaskable, always carried "+
				"through) and RunLeverFlip's defaultLevers drops it from the sweep — its marginal value can never "+
				"be measured. Give the rung a stable self-reported By.", i, typ))
		case seen[name]:
			out = append(out, fmt.Sprintf("rung #%d (%s) shares canonical name %q with an earlier rung: "+
				"abi.WithoutRung removes BOTH at once (mask.go: it drops every rung whose RungName equals the "+
				"name), so neither is independently ablatable. Give the rung a distinct self-reported By.", i, typ, name))
		default:
			seen[name] = true
		}
	}
	return out
}

// TestRegisteredRungsAreAddressableByName is the live fence: after the standard registry
// setup (agent.Configure(), exactly as RunLeverFlip does), the full registered adjudicator
// chain must be totally addressable — no rung with an empty canonical By, no two rungs
// sharing one. A failure here means a newly-registered rung is invisible to the lever-flip
// ablation harness and its causal worth can never be measured.
func TestRegisteredRungsAreAddressableByName(t *testing.T) {
	agent.Configure()
	for _, msg := range rungNameViolations(abi.Adjudicators()) {
		t.Error(msg)
	}
}

// namedRung is a fixture adjudicator that self-reports a fixed By and nothing else — the
// minimal rung the totality checker probes. By=="" models a rung that reports no name;
// two namedRungs with the same by model a name collision. Both are the exact conditions
// the live fence above forbids in the registered chain.
type namedRung struct{ by string }

func (r namedRung) Adjudicate(context.Context, *abi.ToolCall) abi.Verdict { return abi.Verdict{By: r.by} }
func (r namedRung) Caps() []abi.Capability                               { return nil }

// TestRungNameTotality_FailsOnUnnamedRung witnesses the first acceptance criterion: the
// checker fails on a rung whose probed By is empty, and its message names the offending
// concrete rung type and cites the mask.go naming contract.
func TestRungNameTotality_FailsOnUnnamedRung(t *testing.T) {
	chain := []abi.Adjudicator{namedRung{"grammar"}, namedRung{""}, namedRung{"preflight"}}
	v := rungNameViolations(chain)
	if len(v) != 1 {
		t.Fatalf("want exactly 1 violation for the unnamed rung, got %d: %v", len(v), v)
	}
	if !strings.Contains(v[0], "turnbench.namedRung") {
		t.Errorf("violation must name the offending concrete rung type, got: %s", v[0])
	}
	if !strings.Contains(v[0], "EMPTY") || !strings.Contains(v[0], "mask.go") {
		t.Errorf("violation must flag the empty By and cite the mask.go naming contract, got: %s", v[0])
	}
}

// TestRungNameTotality_FailsOnDuplicateName witnesses the second acceptance criterion: the
// checker fails on two rungs sharing one canonical name, naming the concrete rung type and
// the duplicated By.
func TestRungNameTotality_FailsOnDuplicateName(t *testing.T) {
	chain := []abi.Adjudicator{namedRung{"grammar"}, namedRung{"preflight"}, namedRung{"grammar"}}
	v := rungNameViolations(chain)
	if len(v) != 1 {
		t.Fatalf("want exactly 1 violation for the duplicate name, got %d: %v", len(v), v)
	}
	if !strings.Contains(v[0], "turnbench.namedRung") {
		t.Errorf("violation must name the offending concrete rung type, got: %s", v[0])
	}
	if !strings.Contains(v[0], "shares canonical name") || !strings.Contains(v[0], `"grammar"`) {
		t.Errorf("violation must name the duplicated canonical By, got: %s", v[0])
	}
}

// TestRungNameTotality_GreenOnDistinctNames pins the checker's negative: a chain of
// distinctly-named rungs has zero violations, so the live fence above cannot mint a false
// positive on a healthy chain.
func TestRungNameTotality_GreenOnDistinctNames(t *testing.T) {
	chain := []abi.Adjudicator{namedRung{"grammar"}, namedRung{"preflight"}, namedRung{"ifc-sink"}}
	if v := rungNameViolations(chain); len(v) != 0 {
		t.Errorf("distinct-named chain must have no violations, got: %v", v)
	}
}
