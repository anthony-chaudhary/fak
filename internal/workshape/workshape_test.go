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

func TestEvaluateSerializesDisjointTreesSharingContract(t *testing.T) {
	got := Evaluate(Contract{DeclaredKind: Broad, RootSpine: "own schema", IntegrationOwner: "owner", Packets: []Packet{
		{ID: "api", Tree: []string{"internal/api/**"}, Witness: "api test", SharedContract: []string{"schema/v1"}},
		{ID: "docs", Tree: []string{"docs/**"}, Witness: "doc test", SharedContract: []string{"schema/v1"}},
	}})
	if got.Verdict != "ADMIT_BROAD" || got.SemanticVerdict != "SEMANTIC_SERIALIZED" || len(got.Serialized) != 2 || len(got.SemanticReasons) == 0 {
		t.Fatalf("shared contract = %+v", got)
	}
}

func TestEvaluateAdmitsSemanticallyIndependentPackets(t *testing.T) {
	got := Evaluate(Contract{DeclaredKind: Broad, RootSpine: "root", IntegrationOwner: "owner", Packets: []Packet{
		{ID: "docs", Tree: []string{"docs/**"}, Witness: "doc test", Outputs: []string{"guide"}, Acceptance: []string{"docs green"}},
		{ID: "tests", Tree: []string{"internal/x/*_test.go"}, Witness: "go test", Outputs: []string{"witness"}, Acceptance: []string{"tests green"}},
	}})
	if got.SemanticVerdict != "SEMANTIC_PARALLEL" || len(got.ParallelGroups) != 1 || len(got.ParallelGroups[0]) != 2 {
		t.Fatalf("independent = %+v", got)
	}
}

func TestEvaluateRejectsCycleUnknownEdgeAndDuplicateAcceptance(t *testing.T) {
	cases := []Contract{
		{DeclaredKind: Broad, RootSpine: "root", IntegrationOwner: "owner", Packets: []Packet{{ID: "a", Tree: []string{"a"}, Witness: "a", DependsOn: []string{"b"}}, {ID: "b", Tree: []string{"b"}, Witness: "b", DependsOn: []string{"a"}}}},
		{DeclaredKind: Broad, RootSpine: "root", IntegrationOwner: "owner", Packets: []Packet{{ID: "a", Tree: []string{"a"}, Witness: "a", DependsOn: []string{"missing"}}}},
		{DeclaredKind: Broad, RootSpine: "root", IntegrationOwner: "owner", Packets: []Packet{{ID: "a", Tree: []string{"a"}, Witness: "a", Acceptance: []string{"ship"}}, {ID: "b", Tree: []string{"b"}, Witness: "b", Acceptance: []string{"ship"}}}},
	}
	for i, c := range cases {
		if got := Evaluate(c); got.Verdict != "REFUSE_SEMANTIC_DEPENDENCY" {
			t.Fatalf("case %d = %+v", i, got)
		}
	}
}
