package portability

import (
	"context"
	"encoding/json"
	"errors"
	"testing"
)

func TestReferenceAdaptersConform(t *testing.T) {
	got, e := RunReferenceConformance(context.Background())
	if e != nil {
		t.Fatal(e)
	}
	if len(got) != 9 {
		t.Fatalf("got %d adapters", len(got))
	}
	for _, r := range got {
		if len(r.Passed) < 9 {
			t.Errorf("%s only passed %v", r.Kind, r.Passed)
		}
	}
}
func TestUnknownAdapterPreservedInactive(t *testing.T) {
	a := NewAdapterRegistry().Adapter("future-kind")
	r := Record{Kind: "future-kind", Name: "x", Version: "99", Active: true, Data: json.RawMessage(`{"x":1}`), Unknown: json.RawMessage(`{"future":2}`)}
	b, e := a.Export(context.Background(), r)
	if e != nil {
		t.Fatal(e)
	}
	var got Record
	_ = json.Unmarshal(b, &got)
	if got.Active || string(got.Unknown) != string(r.Unknown) {
		t.Fatalf("not preserved inactive: %+v", got)
	}
	if _, e = a.Apply(context.Background(), NewMemoryState(), "x", Plan{}); e == nil {
		t.Fatal("unknown adapter applied")
	}
}
func TestStructuredErrorsDeterministic(t *testing.T) {
	a := ReferenceRegistry().Adapter("skill")
	e := a.Validate(context.Background(), Record{Kind: "skill", Data: json.RawMessage(`{`)})
	var x *Error
	if !errors.As(e, &x) || x.Code != "MALFORMED_RECORD" || x.Operation != "validate" {
		t.Fatalf("%#v", e)
	}
}
func TestCoverageAndSkeleton(t *testing.T) {
	r := ReferenceRegistry()
	if e := r.RequireCoverage([]string{"skill", "workflow", "policy", "profile", "tool-binding", "model-binding", "session", "checkpoint", "receipt"}); e != nil {
		t.Fatal(e)
	}
	if e := r.RequireCoverage([]string{"new-kind"}); e == nil {
		t.Fatal("missing coverage accepted")
	}
	s, e := Skeleton("sample-kind")
	if e != nil || s == "" {
		t.Fatal(e)
	}
}
