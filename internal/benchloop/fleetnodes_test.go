package benchloop

import "testing"

func TestRegisteredFleetNodesAnthonyHasComparableBaseline(t *testing.T) {
	nodes := RegisteredFleetNodes()
	if len(nodes) != 1 || nodes[0].Name != "anthony-laptop" {
		t.Fatalf("nodes = %#v", nodes)
	}
	n := nodes[0]
	if n.State != "registered" || n.Baseline == nil || n.Baseline.Trials != 3 {
		t.Fatalf("registration = %#v", n)
	}
	if len(n.Baseline.DecodeTokPerS) != 3 || n.Baseline.Engine != "fak-in-kernel via compute HAL backend cuda" {
		t.Fatalf("baseline = %#v", n.Baseline)
	}
	if len(n.Capabilities) == 0 || len(n.Exclusions) == 0 || n.Witness == "" {
		t.Fatalf("capability boundary incomplete: %#v", n)
	}
}
