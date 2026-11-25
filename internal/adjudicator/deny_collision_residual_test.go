package adjudicator

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// collisionReasons are the reasons a case-colliding Deny pair may cite: the
// never-advisory classes plus every advisory-eligible heuristic rung.
var collisionReasons = []abi.ReasonCode{
	abi.ReasonPolicyBlock, abi.ReasonSecretExfil, abi.ReasonSelfModify,
	abi.ReasonMalformed, abi.ReasonDefaultDeny,
}

// collisionKeyShapes are the key pairs that fold to one name: lower+Mixed,
// UPPER+Mixed, and two non-lowercase spellings.
var collisionKeyShapes = [][2]string{
	{"wipe", "Wipe"},
	{"WIPE", "Wipe"},
	{"Wipe", "WIPE"},
}

var collisionCallCasings = []string{"wipe", "Wipe", "WIPE"}

// collisionSoftened is the oracle for "this policy treats reason r as advisory":
// only an eligible reason the policy declares advisory is softened.
func collisionSoftened(adv map[abi.ReasonCode]bool, r abi.ReasonCode) bool {
	return adv[r] && AdvisoryEligible(r)
}

// collisionAdvisorySets is the nil set plus each single-reason set over every
// reason, including the ineligible ones a raw Policy may carry.
func collisionAdvisorySets() []map[abi.ReasonCode]bool {
	sets := []map[abi.ReasonCode]bool{nil}
	for _, r := range collisionReasons {
		sets = append(sets, map[abi.ReasonCode]bool{r: true})
	}
	return sets
}

// TestDenyCollisionNeverSoftenedWins asserts that a call in any casing is hard
// denied by a reason this policy does not soften, whichever entry the map yields
// first, through New, SetPolicy, and the uninstalled Policy.NeverAdmits path.
func TestDenyCollisionNeverSoftenedWins(t *testing.T) {
	ctx := context.Background()
	for _, r1 := range collisionReasons {
		for _, r2 := range collisionReasons {
			if r1 == r2 {
				continue
			}
			for _, shape := range collisionKeyShapes {
				deny := map[string]abi.ReasonCode{shape[0]: r1, shape[1]: r2}
				for _, adv := range collisionAdvisorySets() {
					var candidates []abi.ReasonCode
					for _, r := range []abi.ReasonCode{r1, r2} {
						if !collisionSoftened(adv, r) {
							candidates = append(candidates, r)
						}
					}
					raw := Policy{Posture: PostureDefaultOpen, Deny: deny, AdvisoryReasons: adv}
					for i := 0; i < 16; i++ {
						installed := New(raw)
						swapped := New(Policy{Posture: PostureDefaultOpen})
						swapped.SetPolicy(raw)
						for _, a := range []*Adjudicator{installed, swapped} {
							for _, tool := range collisionCallCasings {
								checkCollisionDeny(ctx, t, a, deny, adv, candidates, tool)
								if !a.NeverAdmits(tool) || !raw.NeverAdmits(tool) {
									t.Fatalf("Deny %v advisory %v: NeverAdmits(%q) = false, want true", deny, adv, tool)
								}
							}
						}
					}
				}
			}
		}
	}
}

// checkCollisionDeny asserts the verdict for tool is a Deny citing one of the
// non-softened candidates, and that a candidate the operator cannot lift wins
// over a liftable one.
func checkCollisionDeny(ctx context.Context, t *testing.T, a *Adjudicator, deny map[string]abi.ReasonCode, adv map[abi.ReasonCode]bool, candidates []abi.ReasonCode, tool string) {
	t.Helper()
	v := a.Adjudicate(ctx, inlineCall(tool, `{}`))
	if v.Kind != abi.VerdictDeny {
		t.Fatalf("Deny %v advisory %v, call %q: got %v/%v, want Deny", deny, adv, tool, v.Kind, abi.ReasonName(v.Reason))
	}
	inCandidates, nonLiftable := false, false
	for _, c := range candidates {
		if c == v.Reason {
			inCandidates = true
		}
		if !operatorApprovalLiftable(c) {
			nonLiftable = true
		}
	}
	if !inCandidates {
		t.Fatalf("Deny %v advisory %v, call %q: reason %s is not a non-softened candidate", deny, adv, tool, abi.ReasonName(v.Reason))
	}
	if nonLiftable && operatorApprovalLiftable(v.Reason) {
		t.Fatalf("Deny %v advisory %v, call %q: liftable %s won over a non-liftable candidate", deny, adv, tool, abi.ReasonName(v.Reason))
	}
}

// TestDenyCollisionApprovalCannotLiftNonLiftable grants the approval id a call
// would carry and asserts a colliding pair that includes a non-liftable reason
// still denies: the approval may lift only a liftable winner, never a
// non-liftable one.
func TestDenyCollisionApprovalCannotLiftNonLiftable(t *testing.T) {
	ctx := context.Background()
	id := OperatorApprovalID("wipe", map[string]any{})
	for _, r1 := range collisionReasons {
		for _, r2 := range collisionReasons {
			if r1 == r2 || (operatorApprovalLiftable(r1) && operatorApprovalLiftable(r2)) {
				continue
			}
			for _, shape := range collisionKeyShapes {
				deny := map[string]abi.ReasonCode{shape[0]: r1, shape[1]: r2}
				for i := 0; i < 16; i++ {
					a := New(Policy{
						Posture:           PostureDefaultOpen,
						Deny:              deny,
						OperatorApprovals: map[string]bool{id: true},
					})
					for _, tool := range collisionCallCasings {
						v := a.Adjudicate(ctx, inlineCall(tool, `{}`))
						if v.Kind != abi.VerdictDeny {
							t.Fatalf("Deny %v approved %s, call %q: got %v/%v, want Deny",
								deny, id, tool, v.Kind, abi.ReasonName(v.Reason))
						}
						if v.Reason != r1 && v.Reason != r2 {
							t.Fatalf("Deny %v approved %s, call %q: reason %s is not a colliding entry",
								deny, id, tool, abi.ReasonName(v.Reason))
						}
						if !operatorApprovalLiftable(r1) && !operatorApprovalLiftable(r2) {
							continue
						}
						if operatorApprovalLiftable(v.Reason) {
							t.Fatalf("Deny %v approved %s, call %q: liftable %s won over a non-liftable entry",
								deny, id, tool, abi.ReasonName(v.Reason))
						}
					}
				}
			}
		}
	}
}
