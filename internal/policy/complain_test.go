package policy

import "testing"

func TestManifestComplainLoadsIntoAdjudicatorPolicy(t *testing.T) {
	rt, err := ParseRuntime([]byte(`{"complain":["custom_tool"," custom_tool ",""]}`))
	if err != nil {
		t.Fatal(err)
	}
	if len(rt.Adjudicator.Complain) != 1 || !rt.Adjudicator.Complain["custom_tool"] {
		t.Fatalf("complain = %+v", rt.Adjudicator.Complain)
	}
}

func TestManifestComplainPreservesUnknownFieldDiscipline(t *testing.T) {
	if _, err := ParseRuntime([]byte(`{"complain":["x"],"complain_typo":["y"]}`)); err == nil {
		t.Fatal("unknown field accepted")
	}
}
