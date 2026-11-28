package agentdescriptor_test

import (
	"bytes"
	"encoding/json"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/pkg/routepolicy/agentdescriptor"
)

// fak-test:runtime fast est=10ms lane=default
func TestNewAndValidate(t *testing.T) {
	withModel := agentdescriptor.New("worker-1", "ephemeral", "frontier", "claude-sonnet", 4, "fanout")
	if withModel.Schema != agentdescriptor.Schema {
		t.Fatalf("Schema = %q, want %q", withModel.Schema, agentdescriptor.Schema)
	}
	if err := withModel.Validate(); err != nil {
		t.Fatalf("Validate(withModel) unexpected error: %v", err)
	}

	withoutModelID := agentdescriptor.New("coord-1", "persistent", "small", "", 1, "single")
	if err := withoutModelID.Validate(); err != nil {
		t.Fatalf("Validate(withoutModelID) should allow empty Model.ID: %v", err)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestValidateRejectsInvalidCoordinates(t *testing.T) {
	base := func() agentdescriptor.Descriptor {
		return agentdescriptor.New("agent-a", "ephemeral", "small", "m1", 2, "paired")
	}
	bad := []struct {
		name   string
		mutate func(*agentdescriptor.Descriptor)
	}{
		{name: "empty schema", mutate: func(d *agentdescriptor.Descriptor) { d.Schema = "" }},
		{name: "wrong schema", mutate: func(d *agentdescriptor.Descriptor) { d.Schema = "wrong/1" }},
		{name: "empty agent identity", mutate: func(d *agentdescriptor.Descriptor) { d.Agent.Identity = "" }},
		{name: "whitespace agent identity", mutate: func(d *agentdescriptor.Descriptor) { d.Agent.Identity = "  \t " }},
		{name: "empty agent lifecycle", mutate: func(d *agentdescriptor.Descriptor) { d.Agent.Lifecycle = "" }},
		{name: "whitespace agent lifecycle", mutate: func(d *agentdescriptor.Descriptor) { d.Agent.Lifecycle = "\n" }},
		{name: "empty model class", mutate: func(d *agentdescriptor.Descriptor) { d.Model.Class = "" }},
		{name: "whitespace model class", mutate: func(d *agentdescriptor.Descriptor) { d.Model.Class = "   " }},
		{name: "zero fleet cardinality", mutate: func(d *agentdescriptor.Descriptor) { d.Fleet.Cardinality = 0 }},
		{name: "negative fleet cardinality", mutate: func(d *agentdescriptor.Descriptor) { d.Fleet.Cardinality = -3 }},
		{name: "empty fleet topology", mutate: func(d *agentdescriptor.Descriptor) { d.Fleet.Topology = "" }},
		{name: "whitespace fleet topology", mutate: func(d *agentdescriptor.Descriptor) { d.Fleet.Topology = " \t " }},
	}
	for _, tc := range bad {
		d := base()
		tc.mutate(&d)
		if err := d.Validate(); err == nil {
			t.Errorf("%s: Validate() = nil, want error", tc.name)
		}
		if _, err := d.Marshal(); err == nil {
			t.Errorf("%s: Marshal() = nil error on invalid descriptor", tc.name)
		}
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestMarshalAndDecodeRoundTrip(t *testing.T) {
	for _, want := range []agentdescriptor.Descriptor{
		agentdescriptor.New("w-1", "ephemeral", "guard", "guard-v2", 2, "ensemble"),
		agentdescriptor.New("w-2", "daemon", "fast", "", 1, "solo"),
	} {
		raw, err := want.Marshal()
		if err != nil {
			t.Fatalf("Marshal(%+v): %v", want, err)
		}
		if want.Model.ID == "" && bytes.Contains(raw, []byte(`"id"`)) {
			t.Errorf("Marshal emitted empty Model.ID despite omitempty: %s", raw)
		}
		got, err := agentdescriptor.Decode(raw)
		if err != nil {
			t.Fatalf("Decode(%s): %v", raw, err)
		}
		if !reflect.DeepEqual(got, want) {
			t.Errorf("round-trip mismatch:\ngot:  %+v\nwant: %+v", got, want)
		}
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestDecodeRejectsMalformedAndInvalidPayloads(t *testing.T) {
	badPayloads := []struct {
		name string
		raw  string
	}{
		{name: "invalid json", raw: `{"schema":`},
		{name: "empty json object", raw: `{}`},
		{name: "wrong schema", raw: `{"schema":"other/1","agent":{"identity":"a","lifecycle":"l"},"model":{"class":"c"},"fleet":{"cardinality":1,"topology":"t"}}`},
		{name: "zero cardinality", raw: `{"schema":"fak.agent-operation-descriptor/1","agent":{"identity":"a","lifecycle":"l"},"model":{"class":"c"},"fleet":{"cardinality":0,"topology":"t"}}`},
		{name: "blank model class", raw: `{"schema":"fak.agent-operation-descriptor/1","agent":{"identity":"a","lifecycle":"l"},"model":{"class":" "},"fleet":{"cardinality":1,"topology":"t"}}`},
	}
	for _, tc := range badPayloads {
		if _, err := agentdescriptor.Decode([]byte(tc.raw)); err == nil {
			t.Errorf("%s: Decode succeeded, want error", tc.name)
		}
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestOperationReceiptJSON(t *testing.T) {
	rec := agentdescriptor.OperationReceipt{
		Schema:      agentdescriptor.Schema,
		OperationID: "op-42",
		Descriptor:  agentdescriptor.New("a1", "ephemeral", "frontier", "m1", 3, "shard"),
		RouteRule:   "hard-reasoning",
	}
	b, err := json.Marshal(rec)
	if err != nil {
		t.Fatalf("Marshal(OperationReceipt): %v", err)
	}
	var got agentdescriptor.OperationReceipt
	if err := json.Unmarshal(b, &got); err != nil {
		t.Fatalf("Unmarshal(OperationReceipt): %v", err)
	}
	if !reflect.DeepEqual(got, rec) {
		t.Errorf("OperationReceipt round-trip = %+v, want %+v", got, rec)
	}
}
