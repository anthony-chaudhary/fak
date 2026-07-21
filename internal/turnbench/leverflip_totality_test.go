package turnbench

// leverflip_totality_test.go — the RUNG-NAME TOTALITY FENCE (#3974). The lever-flip
// causal sweep can only ablate a rung it can NAME: abi.RungName probes a rung's
// self-reported By, and a rung that reports "" (or a name a sibling already claimed) is
// unaddressable — abi.WithoutRung can never mask it (mask.go:78 skips a non-empty match
// only; an unnamed rung is "always carried through", mask.go:74), so defaultLevers drops
// it from the full sweep (leverflip.go:206). Such a rung's marginal value can NEVER be
// measured: the causal sweep quietly stops being total for exactly the rung that dodges
// it. Today every registered rung happens to be named and unique; nothing fenced the next
// one. This test IS that fence — the registry-derived precedent is
// architest_test.go TestEveryAdjudicatorIsExecFree.

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/agent"
)

// rungNameTotalityViolations returns one human-readable violation per rung that is
// unaddressable by the lever-flip sweep: a rung whose canonical abi.RungName is empty
// (no self-reported By — unmaskable), or one whose canonical name collides with an
// earlier rung's (a duplicate collapses two rungs into one lever, so abi.WithoutRung
// masks BOTH and per-rung attribution is impossible). Each message names the offending
// rung's CONCRETE TYPE and cites mask.go's naming contract, so a failure points straight
// at the rung to fix. An empty slice means the chain is total — every rung is a distinct,
// nameable lever.
func rungNameTotalityViolations(chain []abi.Adjudicator) []string {
	firstIdx := map[string]int{} // canonical name -> index of the first rung that claimed it
	var violations []string
	for i, a := range chain {
		name := abi.RungName(a)
		if name == "" {
			violations = append(violations, fmt.Sprintf(
				"rung #%d of type %T self-reports an empty By, so abi.RungName yields \"\" and the rung is unaddressable by name: abi.WithoutRung always carries it through (internal/abi/mask.go:74) and defaultLevers drops it from the full sweep (leverflip.go:206), so its marginal value can never be ablated. Give it a stable non-empty By (the abi.RungName contract, mask.go:34).",
				i, a))
			continue
		}
		if j, dup := firstIdx[name]; dup {
			violations = append(violations, fmt.Sprintf(
				"rungs #%d (%T) and #%d (%T) both canonicalize to By %q, so they collapse to ONE lever: abi.WithoutRung would mask BOTH and per-rung causal attribution is impossible. Give each rung a distinct By (the canonicalRungName contract, internal/abi/mask.go:48).",
				j, chain[j], i, a, name))
			continue
		}
		firstIdx[name] = i
	}
	return violations
}

// TestRungNameTotality_RegisteredChainStaysAblatable is the live fence: after the standard
// registry setup, EVERY registered adjudicator rung must be nameable and uniquely named, so
// the lever-flip sweep stays TOTAL. A future rung registered with an empty or duplicate By
// reds the trunk here with the rung's concrete type named — it cannot silently dodge the
// causal sweep. (Acceptance: green on the current registered chain.)
func TestRungNameTotality_RegisteredChainStaysAblatable(t *testing.T) {
	agent.Configure() // idempotent registry setup — the same install RunLeverFlip does (leverflip.go:147)
	chain := abi.Adjudicators()
	if len(chain) == 0 {
		t.Fatal("registered adjudicator chain is empty — the totality fence has nothing to guard")
	}
	if v := rungNameTotalityViolations(chain); len(v) > 0 {
		for _, msg := range v {
			t.Errorf("rung-name totality violated (a registered rung is not ablatable): %s", msg)
		}
	}
}

// fixtureRung is a minimal adjudicator whose self-reported By is whatever the test sets —
// the witness for the negative cases below. RungName reads only v.By, so the Verdict's
// Kind is left zero.
type fixtureRung struct{ by string }

func (f fixtureRung) Adjudicate(_ context.Context, _ *abi.ToolCall) abi.Verdict {
	return abi.Verdict{By: f.by}
}
func (f fixtureRung) Caps() []abi.Capability { return nil }

// TestRungNameTotality_EmptyByRungIsCaught witnesses acceptance criterion 1: a rung whose
// probed By is empty is flagged, and the failure names the rung's concrete type + the
// naming contract. This is the case abi.RungName documents as unmaskable — the fence turns
// that silent gap into a hard failure.
func TestRungNameTotality_EmptyByRungIsCaught(t *testing.T) {
	chain := []abi.Adjudicator{fixtureRung{by: "grammar"}, fixtureRung{by: ""}}
	v := rungNameTotalityViolations(chain)
	if len(v) != 1 {
		t.Fatalf("want exactly 1 violation for an unnamed rung, got %d: %v", len(v), v)
	}
	if !strings.Contains(v[0], "turnbench.fixtureRung") || !strings.Contains(v[0], "empty By") {
		t.Errorf("violation must name the rung's concrete type and the empty-By contract, got: %s", v[0])
	}
}

// TestRungNameTotality_DuplicateNameIsCaught witnesses acceptance criterion 2: two rungs
// that canonicalize to the same name are flagged. The two fixtures report "dup" and
// "dup(off)", so this also proves the fence compares CANONICAL names (canonicalRungName
// strips the parenthesized suffix), exactly as the sweep does — not raw By strings.
func TestRungNameTotality_DuplicateNameIsCaught(t *testing.T) {
	chain := []abi.Adjudicator{fixtureRung{by: "dup"}, fixtureRung{by: "dup(off)"}}
	v := rungNameTotalityViolations(chain)
	if len(v) != 1 {
		t.Fatalf("want exactly 1 violation for a duplicate canonical name, got %d: %v", len(v), v)
	}
	if !strings.Contains(v[0], `"dup"`) || !strings.Contains(v[0], "turnbench.fixtureRung") {
		t.Errorf("violation must name the duplicated canonical name and the rung type, got: %s", v[0])
	}
}
