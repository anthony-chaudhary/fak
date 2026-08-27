package agentdescriptor

import (
	"encoding/json"
	"testing"
)

func TestDescriptorsRoundTripAndIdentityIsIndependent(t *testing.T) {
	a := New("macro:release-steward", "micro", "frontier", "f", 1, "single")
	b := New("macro:release-steward", "micro", "small", "s", 100000, "fanout")
	for _, d := range []Descriptor{a, b} {
		raw, err := d.Marshal()
		if err != nil {
			t.Fatal(err)
		}
		got, err := Decode(raw)
		if err != nil || got.Agent != d.Agent || got.Model != d.Model || got.Fleet != d.Fleet {
			t.Fatalf("roundtrip got=%+v err=%v", got, err)
		}
	}
	if a.Agent != b.Agent || a.Model == b.Model || a.Fleet == b.Fleet {
		t.Fatalf("coordinates coupled: a=%+v b=%+v", a, b)
	}
}
func TestOperationReceiptEmitsDescriptor(t *testing.T) {
	d := New("macro:stable", "micro", "frontier", "f", 1, "single")
	raw, err := json.Marshal(OperationReceipt{Schema: Schema, OperationID: "op-1", Descriptor: d, RouteRule: "fast"})
	if err != nil {
		t.Fatal(err)
	}
	var got OperationReceipt
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatal(err)
	}
	if got.Descriptor.Agent.Identity != "macro:stable" || got.RouteRule != "fast" {
		t.Fatalf("receipt=%+v", got)
	}
}
