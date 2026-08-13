package main

import "testing"

func TestPlanMicroSelfcheckChildrenGrantsDisjointLeases(t *testing.T) {
	children, err := foldMicroSelfcheckChildren()
	if err != nil {
		t.Fatal(err)
	}
	if len(children) != 2 || children[0].LeaseID == children[1].LeaseID || children[0].SessionID == children[1].SessionID {
		t.Fatalf("bad children: %+v", children)
	}
	for _, child := range children {
		if child.State != "queued" {
			t.Fatalf("child not queued: %+v", child)
		}
	}
}
func TestWitnessMicroEffectRefusesMissingReadback(t *testing.T) {
	if digest, ok := digestMicroReadback(""); ok || digest != "" {
		t.Fatalf("missing effect witnessed: %q %t", digest, ok)
	}
	if digest, ok := digestMicroReadback("effect"); !ok || digest == "" {
		t.Fatalf("effect not witnessed: %q %t", digest, ok)
	}
}
