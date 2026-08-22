package borrowprovenance

import (
	"strings"
	"testing"
)

func TestPinAndVerifyExactSource(t *testing.T) {
	source := []byte("licensed upstream excerpt\n")
	record, err := Pin("https://example.test/upstream", "abc123", "src/header.h", "Apache-2.0", "copied declarations", source)
	if err != nil {
		t.Fatal(err)
	}
	got, err := Verify(record, source)
	if err != nil {
		t.Fatal(err)
	}
	if !got.Match || got.ExpectedSHA256 != got.ActualSHA256 {
		t.Fatalf("matching source reported drift: %+v", got)
	}
	drift, err := Verify(record, append(source, 'x'))
	if err != nil {
		t.Fatal(err)
	}
	if drift.Match || drift.ExpectedSHA256 == drift.ActualSHA256 {
		t.Fatalf("changed source was accepted: %+v", drift)
	}
}

func TestRecordRejectsUnverifiablePins(t *testing.T) {
	tests := []Record{{}, {Schema: Schema, SourceURL: "https://example.test", SourceRef: "abc", SourceSHA256: strings.Repeat("A", 64)}, {Schema: Schema, SourceURL: "https://example.test", SourceRef: "abc", SourceSHA256: "short"}}
	for _, record := range tests {
		if err := record.Validate(); err == nil {
			t.Fatalf("Validate(%+v) succeeded", record)
		}
	}
}
