package syspromptmmu

import (
	"bytes"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/cachemeta"
)

// pass / fail are INDEPENDENT witness predicates: their verdict does not derive from the
// edit's own content — the agent never grades its own edit (invariant 5). A real caller
// passes the `dos verify` / guard verdict here.
func pass(BaseEdit) bool { return true }
func fail(BaseEdit) bool { return false }

func firstWitnessOfTier(tier Tier) string {
	for _, s := range BaseContext() {
		if s.Tier == tier {
			return s.Witness
		}
	}
	return ""
}

// TestGateRefusesSpineAlways asserts a spine-targeted edit is refused regardless of a
// passing witness — the spine is off-limits to the agent's own edits.
func TestGateRefusesSpineAlways(t *testing.T) {
	for _, op := range []EditOp{EditAdd, EditPromote, EditDemote, EditVersion} {
		e := BaseEdit{Op: op, Tier: TierSpine, Content: []byte("x"), Target: "anything"}
		if v := GateEdit(e, pass); v.Applied || v.Reason != EditRefusedSpine {
			t.Errorf("op %s on spine: applied=%v reason=%q, want refused-spine", op, v.Applied, v.Reason)
		}
	}
}

// TestGateRefusesMetaRule asserts versioning/demoting an immutable policy-floor block is
// refused (the edit-governing meta-rules), and targeting a spine block by witness is
// refused as spine.
func TestGateRefusesMetaRule(t *testing.T) {
	policyW := firstWitnessOfTier(TierPolicy)
	spineW := firstWitnessOfTier(TierSpine)
	if policyW == "" || spineW == "" {
		t.Fatal("fixture: expected both a spine and a policy block")
	}

	vPol := GateEdit(BaseEdit{Op: EditVersion, Tier: TierPolicy, Target: policyW, Content: []byte("new")}, pass)
	if vPol.Applied || vPol.Reason != EditRefusedMetaRule {
		t.Errorf("version a policy-floor block: applied=%v reason=%q, want refused-meta-rule", vPol.Applied, vPol.Reason)
	}

	// Target a spine block by witness even with a non-spine Tier field — still refused as spine.
	vSpine := GateEdit(BaseEdit{Op: EditDemote, Tier: TierOverlay, Target: spineW}, pass)
	if vSpine.Applied || vSpine.Reason != EditRefusedSpine {
		t.Errorf("demote a spine block by witness: applied=%v reason=%q, want refused-spine", vSpine.Applied, vSpine.Reason)
	}
}

// TestGateRequiresIndependentWitness asserts a structurally-valid edit is refused without
// a passing witness (nil bus = fail-closed; a failing witness = refused) and admitted
// only when the independent witness passes.
func TestGateRequiresIndependentWitness(t *testing.T) {
	e := BaseEdit{Op: EditAdd, Tier: TierPolicy, Content: []byte("learned rule: prefer X")}

	if v := GateEdit(e, nil); v.Applied || v.Reason != EditRefusedNoWitness {
		t.Errorf("nil witness: applied=%v reason=%q, want refused-no-witness", v.Applied, v.Reason)
	}
	if v := GateEdit(e, fail); v.Applied || v.Reason != EditRefusedNoWitness {
		t.Errorf("failing witness: applied=%v reason=%q, want refused-no-witness", v.Applied, v.Reason)
	}
	if v := GateEdit(e, pass); !v.Applied || v.Reason != EditOK {
		t.Errorf("passing witness: applied=%v reason=%q, want ok", v.Applied, v.Reason)
	}
}

// TestGateDegenerate covers unknown op and empty content.
func TestGateDegenerate(t *testing.T) {
	if v := GateEdit(BaseEdit{Op: EditOp(99), Tier: TierPolicy, Content: []byte("x")}, pass); v.Reason != EditRefusedUnknownOp {
		t.Errorf("unknown op reason = %q, want refused-unknown-op", v.Reason)
	}
	if v := GateEdit(BaseEdit{Op: EditAdd, Tier: TierPolicy}, pass); v.Reason != EditRefusedNoContent {
		t.Errorf("empty content reason = %q, want refused-no-content", v.Reason)
	}
}

// TestApplyAddAppendsAndRollsBackBitForBit asserts an admitted Add appends a learned
// block and NEVER mutates the input — the prior plan is a bit-for-bit rollback.
func TestApplyAddAppendsAndRollsBackBitForBit(t *testing.T) {
	base := BaseContext()
	snapshot := BaseContext() // an independent equal value to compare against after the edit

	edited, v := ApplyEdit(base, BaseEdit{Op: EditAdd, Tier: TierPolicy, Content: []byte("learned rule: prefer X over Y"), Version: "v1"}, pass)
	if !v.Applied {
		t.Fatalf("expected the edit to apply, got %q", v.Reason)
	}
	if len(edited) != len(base)+1 {
		t.Fatalf("edited len %d, want %d", len(edited), len(base)+1)
	}
	if edited[len(edited)-1].Tier != TierPolicy {
		t.Errorf("appended block tier = %v, want policy", edited[len(edited)-1].Tier)
	}
	// Rollback: the input was not mutated; it still equals a fresh BaseContext().
	if !reflect.DeepEqual(base, snapshot) {
		t.Fatal("ApplyEdit mutated its input — rollback would not be bit-for-bit")
	}
}

// TestApplyVersionReplacesLearnedRule asserts a learned rule can be versioned and the
// prior plan is unchanged (bit-for-bit rollback to the earlier version).
func TestApplyVersionReplacesLearnedRule(t *testing.T) {
	base := BaseContext()
	withRule, v := ApplyEdit(base, BaseEdit{Op: EditAdd, Tier: TierPolicy, Content: []byte("rule v1")}, pass)
	if !v.Applied {
		t.Fatalf("add failed: %q", v.Reason)
	}
	ruleW := withRule[len(withRule)-1].Witness

	v2base := append([]Segment(nil), withRule...) // snapshot for rollback comparison
	versioned, vv := ApplyEdit(withRule, BaseEdit{Op: EditVersion, Tier: TierPolicy, Target: ruleW, Content: []byte("rule v2 — refined"), Version: "v2"}, pass)
	if !vv.Applied {
		t.Fatalf("version failed: %q", vv.Reason)
	}
	if string(versioned[len(versioned)-1].Content) != "rule v2 — refined" {
		t.Errorf("versioned content = %q", versioned[len(versioned)-1].Content)
	}
	if !reflect.DeepEqual(withRule, v2base) {
		t.Fatal("version mutated the prior plan — rollback not bit-for-bit")
	}
}

// TestApplyDemoteRemovesLearnedRule asserts a learned rule can be demoted (removed) and a
// missing target is refused-not-found.
func TestApplyDemoteRemovesLearnedRule(t *testing.T) {
	base := BaseContext()
	withRule, _ := ApplyEdit(base, BaseEdit{Op: EditAdd, Tier: TierPolicy, Content: []byte("temp rule")}, pass)
	ruleW := withRule[len(withRule)-1].Witness

	demoted, v := ApplyEdit(withRule, BaseEdit{Op: EditDemote, Tier: TierPolicy, Target: ruleW}, pass)
	if !v.Applied {
		t.Fatalf("demote failed: %q", v.Reason)
	}
	if len(demoted) != len(base) {
		t.Errorf("after demote len %d, want %d", len(demoted), len(base))
	}

	if _, v := ApplyEdit(base, BaseEdit{Op: EditDemote, Tier: TierPolicy, Target: "blob-sha256:absent"}, pass); v.Applied || v.Reason != EditRefusedNotFound {
		t.Errorf("demote absent: applied=%v reason=%q, want refused-not-found", v.Applied, v.Reason)
	}
}

// TestEditComposesWithSpliceAndAudit ties Rung 5 to Rungs 2 + 6: a learned-rule edit
// produces a base that realizes through BuildSystemValue and still audits AuditOK against
// the edited plan, AND the spine blocks are byte-identical (an edit never touches the spine).
func TestEditComposesWithSpliceAndAudit(t *testing.T) {
	base := BaseContext()
	edited, v := ApplyEdit(base, BaseEdit{Op: EditAdd, Tier: TierPolicy, Content: []byte("learned: cite sources")}, pass)
	if !v.Applied {
		t.Fatalf("edit failed: %q", v.Reason)
	}

	editedPlan := PlanOf(edited)
	body := bodyWith(t, BuildSystemValue(editedPlan, []cachemeta.PromptSegment{overlaySeg("card")}), nil)

	a := AuditRealizedPrefix(body, editedPlan)
	if !a.Present || a.Diverged || a.Status != AuditOK {
		t.Fatalf("edited base audit: present=%v diverged=%v status=%q, want ok", a.Present, a.Diverged, a.Status)
	}

	// The spine blocks must be byte-identical before and after the edit.
	for i, s := range base {
		if s.Tier == TierSpine && !bytes.Equal(s.Content, edited[i].Content) {
			t.Fatalf("spine block %d changed under an edit", i)
		}
	}
}
