package modelroute

import (
	"github.com/anthony-chaudhary/fak/internal/agentdescriptor"
	"testing"
)

func TestRouteOperationReceiptPreservesAgentAcrossModelFleetChanges(t *testing.T) {
	m := Manifest{Default: Plan{Members: []Member{{Model: "frontier", Role: "primary"}}}}
	a := agentdescriptor.New("macro:stable", "micro", "frontier", "f", 1, "single")
	b := agentdescriptor.New("macro:stable", "micro", "small", "s", 100000, "fanout")
	ra, err := m.Route(Subject{AgentOperation: &a}).OperationReceipt("op-a")
	if err != nil {
		t.Fatal(err)
	}
	rb, err := m.Route(Subject{AgentOperation: &b}).OperationReceipt("op-b")
	if err != nil {
		t.Fatal(err)
	}
	if ra.Descriptor.Agent != rb.Descriptor.Agent || ra.Descriptor.Model == rb.Descriptor.Model || ra.Descriptor.Fleet == rb.Descriptor.Fleet {
		t.Fatalf("receipts coupled: a=%+v b=%+v", ra, rb)
	}
}
