package workshape

import "testing"

func TestEvaluateBroadReservesOwnerAndChildren(t *testing.T) {
	got := Evaluate(Contract{DeclaredKind: Broad, Evidence: []string{"multi-tree"}, RootSpine: "define API", IntegrationOwner: "issue-owner", Packets: []Packet{
		{ID: "docs", Tree: []string{"docs/**"}, Witness: "doc test"},
		{ID: "tests", Tree: []string{"internal/x/*_test.go"}, Witness: "go test"},
	}})
	if got.Verdict != "ADMIT_BROAD" || got.ParentCapacity != 1 || got.ChildCapacity != 2 || got.TotalCapacity != 3 || len(got.ParallelGroups) != 1 {
		t.Fatalf("broad = %+v", got)
	}
}

func TestEvaluateBoundedRejectsCrossPacketDependencies(t *testing.T) {
	got := Evaluate(Contract{DeclaredKind: Bounded, FitsDeadline: true, Packets: []Packet{{ID: "a", Tree: []string{"a"}, Witness: "x", DependsOn: []string{"b"}}}})
	if got.Verdict != "REFUSE_FALSE_BOUNDED" {
		t.Fatalf("bounded = %+v", got)
	}
}

func TestEvaluateMalformedAndExternalFailClosed(t *testing.T) {
	if got := Evaluate(Contract{DeclaredKind: Broad}); got.Verdict != "REFUSE_MALFORMED_BROAD" {
		t.Fatalf("broad = %+v", got)
	}
	if got := Evaluate(Contract{DeclaredKind: BlockedExternal, ExternalBlocker: "vendor credential"}); got.Verdict != "REFUSE_BLOCKED_EXTERNAL" || got.TotalCapacity != 0 {
		t.Fatalf("external = %+v", got)
	}
	if got := Evaluate(Contract{}); got.Verdict != "REFUSE_UNSUPPORTED_WORK_SHAPE" {
		t.Fatalf("unknown = %+v", got)
	}
}
