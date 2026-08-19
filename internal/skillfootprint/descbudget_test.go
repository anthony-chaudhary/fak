package skillfootprint

import (
	"errors"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/capindex"
)

// fatSkillCard builds a skill card whose resident DESCRIPTION alone is deliberately
// heavy, used to push the measured floor over the budget so the growth refusal can
// be witnessed without editing a real SKILL.md.
func fatSkillCard(name string, bytes int) capindex.CapCard {
	return skillCard(name, strings.Repeat("x", bytes))
}

// TestSkillDescriptionBudgetPassesAtHEAD is the enforcement half of #5444: the real
// `.claude/skills` tree, as shipped, must pass its own committed budget. This is the
// test that actually reds a peer's commit when a new skill (or a fattened
// description) grows the resident floor past the ceiling — the userland analog of
// mcpfootprint.TestDescriptionBudgetPassesAtHEAD.
func TestSkillDescriptionBudgetPassesAtHEAD(t *testing.T) {
	fp := Measure(repoRootForTest(t))
	if fp.SkillCount == 0 {
		t.Fatal("real .claude/skills tree priced as 0 skills — the gate would be vacuous")
	}
	t.Logf("resident description floor: %d bytes (~%d est. tokens) across %d skills (budget %d, slack %d)",
		fp.DescFloor, ApproxTokens(fp.DescFloor), fp.SkillCount, SkillDescriptionBudgetBytes, SkillDescriptionRatchetSlackBytes)
	for i := 0; i < min(6, len(fp.Entries)); i++ {
		t.Logf("  top %d: %5d B  %s", i+1, fp.Entries[i].DescBytes, fp.Entries[i].Name)
	}
	if err := CheckDescriptions(fp); err != nil {
		t.Fatalf("the committed .claude/skills tree fails its own resident-description budget: %v", err)
	}
}

// TestSkillDescriptionBudgetRefusesGrowth witnesses the refusal firing: one extra
// skill whose description alone clears the ceiling must red with
// SKILL_DESC_BUDGET_EXCEEDED, and the message must name the constant to raise —
// a refusal that does not say how to justify it is just a block.
func TestSkillDescriptionBudgetRefusesGrowth(t *testing.T) {
	root := repoRootForTest(t)
	cards := capindex.NewSkillResolver(SkillsDir(root)).Index()
	if len(cards) == 0 {
		t.Fatal("real .claude/skills tree priced as 0 skills")
	}
	grown := Fold(append(append([]capindex.CapCard(nil), cards...),
		fatSkillCard("verbose-newcomer", SkillDescriptionBudgetBytes)))

	err := CheckDescriptions(grown)
	if err == nil {
		t.Fatal("the budget admitted a skills tree grown past its ceiling — resident verbosity is ungated")
	}
	var sbe *SkillDescBudgetError
	if !errors.As(err, &sbe) {
		t.Fatalf("want *SkillDescBudgetError, got %T: %v", err, err)
	}
	if sbe.Reason != ReasonSkillDescBudgetExceeded {
		t.Fatalf("Reason = %q, want %q", sbe.Reason, ReasonSkillDescBudgetExceeded)
	}
	if sbe.Measured <= sbe.Budget {
		t.Fatalf("Measured=%d must exceed Budget=%d on a growth refusal", sbe.Measured, sbe.Budget)
	}
	if sbe.SkillCount != len(cards)+1 {
		t.Errorf("SkillCount = %d, want %d — the refusal must report what it priced", sbe.SkillCount, len(cards)+1)
	}
	if msg := sbe.Error(); !strings.Contains(msg, "SkillDescriptionBudgetBytes") {
		t.Errorf("growth refusal does not name the constant to raise: %s", msg)
	}
}

// TestSkillDescriptionBudgetDemandsBankedWin proves the ratchet only ever tightens:
// a real trim that is not banked into the constant reds as SKILL_DESC_BUDGET_STALE,
// so recovered headroom cannot be silently refilled by the next skill. This is the
// half #3234 shipped without, and the half the 30% regression walked through.
func TestSkillDescriptionBudgetDemandsBankedWin(t *testing.T) {
	trimmed := Fold([]capindex.CapCard{skillCard("lean", "Use when the whole catalog fits in one line")})
	err := CheckDescriptions(trimmed)
	if err == nil {
		t.Fatal("the budget admitted an unbanked trim — the ratchet does not tighten")
	}
	var sbe *SkillDescBudgetError
	if !errors.As(err, &sbe) {
		t.Fatalf("want *SkillDescBudgetError, got %T: %v", err, err)
	}
	if sbe.Reason != ReasonSkillDescBudgetStale {
		t.Fatalf("Reason = %q, want %q", sbe.Reason, ReasonSkillDescBudgetStale)
	}
	if !strings.Contains(sbe.Error(), "Re-pin") {
		t.Errorf("stale refusal does not tell the author to re-pin the ceiling: %s", sbe.Error())
	}
}

// TestSkillDescriptionBudgetBandBoundaries pins the exact admit/refuse edges so a
// future edit to the comparison operators cannot quietly widen the band. The band is
// [budget-slack, budget]: the ceiling itself admits, one byte over refuses, sitting
// exactly slack-below still admits, one byte past the slack refuses as stale.
func TestSkillDescriptionBudgetBandBoundaries(t *testing.T) {
	const budget, slack = 1000, 100
	floorAt := func(bytes int) Floor { return Fold([]capindex.CapCard{skillCard("s", strings.Repeat("x", bytes))}) }
	for _, tc := range []struct {
		name  string
		bytes int
		want  string // "" == admit
	}{
		{"at the ceiling admits", budget, ""},
		{"one byte over the ceiling refuses", budget + 1, ReasonSkillDescBudgetExceeded},
		{"exactly slack below admits", budget - slack, ""},
		{"one byte past the slack refuses", budget - slack - 1, ReasonSkillDescBudgetStale},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fp := floorAt(tc.bytes)
			if fp.DescFloor != tc.bytes {
				t.Fatalf("synthetic floor measured %d bytes, want %d", fp.DescFloor, tc.bytes)
			}
			err := checkDescriptionsAgainst(fp, budget, slack)
			if tc.want == "" {
				if err != nil {
					t.Fatalf("bytes=%d: want admit, got %v", tc.bytes, err)
				}
				return
			}
			var sbe *SkillDescBudgetError
			if !errors.As(err, &sbe) {
				t.Fatalf("bytes=%d: want *SkillDescBudgetError %s, got %v", tc.bytes, tc.want, err)
			}
			if sbe.Reason != tc.want {
				t.Fatalf("bytes=%d: Reason = %q, want %q", tc.bytes, sbe.Reason, tc.want)
			}
		})
	}
}

// TestSkillDescriptionBudgetFailsClosed proves the gate refuses an empty catalog
// rather than greening on "I measured nothing".
func TestSkillDescriptionBudgetFailsClosed(t *testing.T) {
	if err := CheckDescriptions(Floor{}); err == nil {
		t.Fatal("the budget admitted an empty catalog — it greens on measuring nothing")
	}
}

// TestCommittedSkillBudgetMatchesMeasuredFloor proves SkillDescriptionBudgetBytes is
// a MEASUREMENT, not a hand-typed number that drifted from the tree: the committed
// ceiling must sit inside the ratchet band of the real measured floor. Raising the
// constant without the growth that justifies it reds here just as surely as growth
// without the raise does.
func TestMeasuredDescriptionFloorSnapshot(t *testing.T) {
	f := Measure(repoRootForTest(t))
	if f.DescFloor != SkillDescriptionBudgetBytes {
		t.Fatalf("measured description floor=%d, committed snapshot=%d", f.DescFloor, SkillDescriptionBudgetBytes)
	}
}

func TestCommittedSkillBudgetMatchesMeasuredFloor(t *testing.T) {
	fp := Measure(repoRootForTest(t))
	if fp.DescFloor > SkillDescriptionBudgetBytes || fp.DescFloor < SkillDescriptionBudgetBytes-SkillDescriptionRatchetSlackBytes {
		t.Fatalf("committed SkillDescriptionBudgetBytes=%d is outside the ratchet band for the measured floor of %d bytes. %v",
			SkillDescriptionBudgetBytes, fp.DescFloor, CheckDescriptions(fp))
	}
}

// TestBothRefusalTokensAreDistinctAndNamespaced pins the closed vocabulary itself:
// the two tokens are distinct, and each carries the SKILL_ prefix that separates the
// userland ratchet from mcpfootprint's DESC_BUDGET_* MCP ratchet. Collapsing them
// would make a dos_check_reason lookup resolve the wrong gate.
func TestBothRefusalTokensAreDistinctAndNamespaced(t *testing.T) {
	if ReasonSkillDescBudgetExceeded == ReasonSkillDescBudgetStale {
		t.Fatal("the two refusal tokens collapsed into one — the ratchet loses its direction")
	}
	for _, tok := range []string{ReasonSkillDescBudgetExceeded, ReasonSkillDescBudgetStale} {
		if !strings.HasPrefix(tok, "SKILL_DESC_BUDGET_") {
			t.Errorf("token %q is not namespaced under SKILL_DESC_BUDGET_ — it would collide with the MCP ratchet", tok)
		}
	}
}
