package laneadmit

import (
	"reflect"
	"strings"
	"testing"
)

// TestDecidePrecedenceStrongestFirst pins the documented "strongest-first" rule
// order: exclusive_lane > same_lane > tree_overlap. Every other test in this
// package constructs a scenario where exactly ONE rung can fire, so the
// precedence ORDER itself is unpinned — a refactor that reordered the Decide
// switch would silently relabel a conflict's Kind (and its operator-facing
// Detail) while every existing test still passed. These cases deliberately build
// a single live lease that satisfies MULTIPLE rungs at once and assert the
// stronger label wins.
func TestDecidePrecedenceStrongestFirst(t *testing.T) {
	cases := []struct {
		name string
		req  Request
		live Lease
		want string
	}{
		{
			// identical named lane AND identical (therefore overlapping) tree:
			// same_lane and tree_overlap both fire; same_lane must win.
			name: "same_lane beats tree_overlap",
			req:  Request{Surface: SurfaceManual, Lane: "gateway", Tree: []string{"internal/gateway/http.go"}, Holder: "me"},
			live: Lease{ID: "resolve-gateway", Lane: "gateway", Tree: []string{"internal/gateway/http.go"}, Holder: "peer"},
			want: ConflictSameLane,
		},
		{
			// abi is exclusive; the live lease is ALSO same-lane and overlapping,
			// so all three rungs fire. The requested-exclusive rung must win.
			name: "request-exclusive beats same_lane and tree_overlap",
			req:  Request{Surface: SurfaceManual, Lane: "abi", Tree: []string{"internal/abi/reg.go"}, Holder: "me"},
			live: Lease{ID: "resolve-abi", Lane: "abi", Tree: []string{"internal/abi/reg.go"}, Holder: "peer"},
			want: ConflictExclusiveLane,
		},
		{
			// the request's own lane is non-exclusive, but its tree overlaps a
			// live EXCLUSIVE lane. The live-exclusive rung must win over the bare
			// geometric overlap that would otherwise be reported.
			name: "live-exclusive beats tree_overlap",
			req:  Request{Surface: SurfaceManual, Lane: "gateway", Tree: []string{"internal/abi/reg.go"}, Holder: "me"},
			live: Lease{ID: "resolve-abi", Lane: "abi", Tree: []string{"internal/abi/**"}, Holder: "peer"},
			want: ConflictExclusiveLane,
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			v := Decide(c.req, []Lease{c.live}, tax())
			if v.Admit {
				t.Fatalf("must refuse when a rung fires, got admit: %+v", v)
			}
			if len(v.Conflicts) != 1 || v.Conflicts[0].Kind != c.want {
				t.Fatalf("want a single %s conflict (strongest rung), got %+v", c.want, v.Conflicts)
			}
		})
	}
}

// TestDecideConflictsSortedDeterministically pins the multi-conflict ordering.
// Decide sorts its Conflicts by LeaseID and derives the refusal Detail from the
// lowest-sorted conflict; with several simultaneous conflicts that ordering is
// the only thing making the evidence (and the operator-facing message)
// reproducible. No existing test drives more than one conflict, so a regression
// that dropped or reordered the sort would go unnoticed. Feed conflicts out of
// ID order and assert they come back ascending with the Detail naming the lowest.
func TestDecideConflictsSortedDeterministically(t *testing.T) {
	// Three live holders on the same named lane — all same_lane conflicts —
	// supplied deliberately out of LeaseID order. Their trees are mutually
	// disjoint from the request's, so ONLY the same_lane rung fires on each and
	// the ordering under test is not perturbed by mixed conflict kinds.
	live := []Lease{
		{ID: "resolve-gateway-30", Lane: "gateway", Tree: []string{"internal/gateway/a.go"}, Holder: "p3"},
		{ID: "resolve-gateway-10", Lane: "gateway", Tree: []string{"internal/gateway/b.go"}, Holder: "p1"},
		{ID: "resolve-gateway-20", Lane: "gateway", Tree: []string{"internal/gateway/c.go"}, Holder: "p2"},
	}
	v := Decide(
		Request{Surface: SurfaceManual, Lane: "gateway", Tree: []string{"internal/gateway/x.go"}, Holder: "me"},
		live, tax(),
	)
	if v.Admit {
		t.Fatalf("a same-lane request must refuse against live holders, got %+v", v)
	}
	if len(v.Conflicts) != 3 {
		t.Fatalf("want all 3 live holders reported as conflicts, got %+v", v.Conflicts)
	}
	gotIDs := []string{v.Conflicts[0].LeaseID, v.Conflicts[1].LeaseID, v.Conflicts[2].LeaseID}
	wantIDs := []string{"resolve-gateway-10", "resolve-gateway-20", "resolve-gateway-30"}
	if !reflect.DeepEqual(gotIDs, wantIDs) {
		t.Fatalf("conflicts must be sorted ascending by LeaseID, got %v", gotIDs)
	}
	if !strings.Contains(v.Detail, "resolve-gateway-10") {
		t.Fatalf("refusal Detail must name the lowest-sorted conflict, got %q", v.Detail)
	}
}
