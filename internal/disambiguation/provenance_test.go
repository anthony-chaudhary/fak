package disambiguation

import (
	"encoding/json"
	"errors"
	"os"
	"reflect"
	"testing"
)

func TestProvenanceValidRoundTripAndQueryPreservation(t *testing.T) {
	entry := fixtureEntry()
	want := SourceWitness{
		Kind: SourceKindGoSource, Locator: "internal/disambiguation/provenance.go#validateSourceProvenance",
		Revision: "r1+g0123456789", CheckedAt: "2026-08-11T12:34:56Z", Probe: "fak-disambiguation/provenance-v1",
	}
	entry.Sources = []SourceWitness{want}
	payload, err := json.Marshal(entry)
	if err != nil {
		t.Fatal(err)
	}
	got, err := ParseEntry(payload)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got.Sources, []SourceWitness{want}) {
		t.Fatalf("parse provenance changed: %#v", got.Sources)
	}
	result := queryResponse(got, "")
	if !reflect.DeepEqual(result.Entry.Sources, []SourceWitness{want}) {
		t.Fatalf("query provenance changed: %#v", result.Entry.Sources)
	}
	queryJSON, err := json.Marshal(result)
	if err != nil {
		t.Fatal(err)
	}
	var wire struct {
		Entry struct {
			Sources []SourceWitness `json:"sources"`
		} `json:"entry"`
	}
	if err := json.Unmarshal(queryJSON, &wire); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(wire.Entry.Sources, []SourceWitness{want}) {
		t.Fatalf("JSON provenance changed: %#v", wire.Entry.Sources)
	}
}

func TestProvenanceRejectsDeterministically(t *testing.T) {
	tests := []struct {
		name, code, field string
		mutate            func(*SourceWitness)
	}{
		{"absolute unix", ErrProvenanceLocatorAbsolute, "sources[0].locator", func(s *SourceWitness) { s.Locator = "/etc/passwd" }},
		{"absolute windows", ErrProvenanceLocatorAbsolute, "sources[0].locator", func(s *SourceWitness) { s.Locator = `C:\\work\\fak\\README.md` }},
		{"escaping", ErrProvenanceLocatorEscape, "sources[0].locator", func(s *SourceWitness) { s.Locator = "../fak-private/runbook.md" }},
		{"unnormalized", ErrProvenanceLocatorInvalid, "sources[0].locator", func(s *SourceWitness) { s.Locator = "docs/../README.md" }},
		{"private kind", ErrProvenanceSourceKind, "sources[0].kind", func(s *SourceWitness) { s.Kind = "private-repository" }},
		{"web unverifiable", ErrProvenanceSourceKind, "sources[0].kind", func(s *SourceWitness) { s.Kind = "url" }},
		{"bad revision", ErrProvenanceRevisionInvalid, "sources[0].revision", func(s *SourceWitness) { s.Revision = "not verifiable revision" }},
		{"noncanonical time", ErrProvenanceCheckedAtInvalid, "sources[0].checked_at", func(s *SourceWitness) { s.CheckedAt = "2026-08-11T12:34:56-07:00" }},
		{"unstable probe", ErrProvenanceProbeInvalid, "sources[0].probe", func(s *SourceWitness) { s.Probe = "Local Probe" }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			entry := fixtureEntry()
			entry.Sources[0] = SourceWitness{Kind: SourceKindDocument, Locator: "README.md#how-it-works", Revision: "r1+g0123456789", CheckedAt: "2026-08-11T12:34:56Z", Probe: "fak-disambiguation/provenance-v1"}
			tt.mutate(&entry.Sources[0])
			err1 := entry.Validate()
			err2 := entry.Validate()
			if err1 == nil || err1.Error() != err2.Error() {
				t.Fatalf("non-deterministic errors: %v / %v", err1, err2)
			}
			if ValidationCode(err1) != tt.code {
				t.Fatalf("code=%q want %q (%v)", ValidationCode(err1), tt.code, err1)
			}
			var ve *ValidationError
			if !errors.As(err1, &ve) || ve.Field != tt.field {
				t.Fatalf("error=%#v want field %q", ve, tt.field)
			}
		})
	}
}

func fixtureEntry() Entry {
	payload, err := os.ReadFile("testdata/entry-v1.json")
	if err != nil {
		panic(err)
	}
	entry, err := ParseEntry(payload)
	if err != nil {
		panic(err)
	}
	return entry
}
