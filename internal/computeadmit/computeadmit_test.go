package computeadmit

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/dispatchorder"
)

// Acceptance (#3269): one Decide call refuses an out-of-range expert claim with
// the closed vocabulary — ExpertParallelPlan's "ranks outside [1,N]" fail-closed
// expressed as a regionadmit-style out-of-taxonomy refusal.
func TestDecideRefusesOutOfRangeExpertClaim(t *testing.T) {
	tax := ExpertTaxonomy(64)
	for _, ranks := range []int{0, -3, 65, 1000} {
		dec := Decide(ExpertRanksRequest("ep-planner", ranks), nil, tax)
		if dec.Admit {
			t.Fatalf("ranks=%d: admitted, want out-of-taxonomy refusal", ranks)
		}
		if dec.Reason != ReasonPolicyBlock {
			t.Fatalf("ranks=%d: reason = %q, want %q", ranks, dec.Reason, ReasonPolicyBlock)
		}
		if dec.Rung != RungOutOfTaxonomy {
			t.Fatalf("ranks=%d: rung = %q, want %q", ranks, dec.Rung, RungOutOfTaxonomy)
		}
		if !strings.Contains(dec.Detail, "expert-set") {
			t.Fatalf("ranks=%d: detail %q does not name the class", ranks, dec.Detail)
		}
	}
	for _, ranks := range []int{1, 8, 64} {
		if dec := Decide(ExpertRanksRequest("ep-planner", ranks), nil, tax); !dec.Admit {
			t.Fatalf("ranks=%d: refused (%s/%s: %s), want admit", ranks, dec.Reason, dec.Rung, dec.Detail)
		}
	}
}

// An unparseable range against a declared space fails closed (out_of_taxonomy),
// never coerces.
func TestDecideRefusesUnparseableRangeAgainstDeclaredSpace(t *testing.T) {
	tax := ExpertTaxonomy(8)
	req := Request{Actor: "ep", Claim: dispatchorder.ComputeClaim{Class: ClassExpertSet, Range: "banana"}}
	dec := Decide(req, nil, tax)
	if dec.Admit || dec.Reason != ReasonPolicyBlock || dec.Rung != RungOutOfTaxonomy {
		t.Fatalf("got admit=%v reason=%q rung=%q, want POLICY_BLOCK/out_of_taxonomy", dec.Admit, dec.Reason, dec.Rung)
	}
}

// Acceptance (#3269): one Decide call refuses an over-subscribed decode claim
// as a COLLISION_RISK with RepartitionAdvice — a decode-worker over-subscription
// IS a lane collision, in the same vocabulary.
func TestDecideRefusesOversubscribedDecodeClaim(t *testing.T) {
	tax := DecodeTaxonomy(4)
	live := []Lease{{ID: "member-a", Holder: "member-a",
		Claim: dispatchorder.ComputeClaim{Class: ClassPhase, Range: "2"}}}

	dec := Decide(DecodeWorkerRequest("member-b", 2), live, tax)
	if dec.Admit {
		t.Fatal("co-targeted decode worker admitted, want COLLISION_RISK refusal")
	}
	if dec.Reason != ReasonCollisionRisk {
		t.Fatalf("reason = %q, want %q", dec.Reason, ReasonCollisionRisk)
	}
	if dec.Rung != RungRegionCollision {
		t.Fatalf("rung = %q, want %q", dec.Rung, RungRegionCollision)
	}
	if dec.Conflict == nil || dec.Conflict.ID != "member-a" {
		t.Fatalf("conflict = %+v, want lease member-a as evidence", dec.Conflict)
	}
	if dec.Advice == nil {
		t.Fatal("no RepartitionAdvice on a compute collision")
	}
	if dec.Advice.Action != "narrow_compute_range" {
		t.Fatalf("advice action = %q, want narrow_compute_range", dec.Advice.Action)
	}
	if got := dec.Advice.OverlapRegion; len(got) != 1 || got[0] != "phase-pool:2" {
		t.Fatalf("advice overlap region = %v, want [phase-pool:2]", got)
	}
	if got := dec.Advice.CollidesWith; len(got) != 1 || got[0] != "member-a" {
		t.Fatalf("advice collides_with = %v, want [member-a]", got)
	}

	// The disjoint worker admits — serialization, not a blanket hold.
	if dec := Decide(DecodeWorkerRequest("member-b", 3), live, tax); !dec.Admit {
		t.Fatalf("disjoint decode worker refused (%s: %s), want admit", dec.Reason, dec.Detail)
	}
	// And an out-of-pool worker index is the taxonomy refusal, not a collision.
	if dec := Decide(DecodeWorkerRequest("member-b", 4), live, tax); dec.Admit || dec.Rung != RungOutOfTaxonomy {
		t.Fatalf("worker 4 of [0,4): admit=%v rung=%q, want out_of_taxonomy refusal", dec.Admit, dec.Rung)
	}
}

// The C1 contention semantics ride through unchanged: different classes never
// contend, shared/shared overlaps co-run, an empty range collides
// conservatively, and a self lease never refuses its own renewal.
func TestDecideContentionSemanticsMatchDispatchPrice(t *testing.T) {
	live := []Lease{{ID: "holder-1", Claim: dispatchorder.ComputeClaim{Class: ClassDevice, Range: "0-3"}}}

	otherClass := Request{Actor: "x", Claim: dispatchorder.ComputeClaim{Class: ClassKVTier, Range: "0-3"}}
	if dec := Decide(otherClass, live, Taxonomy{}); !dec.Admit {
		t.Fatalf("different classes contended: %s", dec.Detail)
	}

	sharedLive := []Lease{{ID: "reader-1", Claim: dispatchorder.ComputeClaim{Class: ClassDevice, Range: "0-3", Mode: "shared"}}}
	sharedReq := Request{Actor: "reader-2", Claim: dispatchorder.ComputeClaim{Class: ClassDevice, Range: "1-2", Mode: "shared"}}
	if dec := Decide(sharedReq, sharedLive, Taxonomy{}); !dec.Admit {
		t.Fatalf("shared/shared overlap refused: %s", dec.Detail)
	}

	unknown := Request{Actor: "vague", Claim: dispatchorder.ComputeClaim{Class: ClassDevice}}
	dec := Decide(unknown, live, Taxonomy{})
	if dec.Admit {
		t.Fatal("empty range admitted against a same-class lease, want conservative collision")
	}
	if dec.Advice == nil || dec.Advice.Action != "declare_compute_range" {
		t.Fatalf("advice = %+v, want declare_compute_range", dec.Advice)
	}

	self := Request{Actor: "holder-1", SelfID: "holder-1",
		Claim: dispatchorder.ComputeClaim{Class: ClassDevice, Range: "0-3"}}
	if dec := Decide(self, live, Taxonomy{}); !dec.Admit {
		t.Fatalf("self renewal refused: %s", dec.Detail)
	}
}
